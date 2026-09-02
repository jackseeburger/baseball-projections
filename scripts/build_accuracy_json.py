"""Build the Model Accuracy page's data — station H (docs/architecture.md §2).

Everything the page shows is generated here, from the scoring scripts and the
scoreboard docs. Nothing on the page is typed by hand: if a number is not in
this file's output it does not appear in the browser.

    public/data/accuracy/latest.json        always rewritten
    public/data/accuracy/YYYY-MM-DD.json    written once, never overwritten

Sections:
    components            scripts/score_2026_projections.py  (offline; data in git)
    ros_backtest          scripts/run_intraseason_backtest.py (needs the PA parquet)
    game_odds             scripts/backtest_game_odds.py      (MLB Stats API + R2 market closes)
    playoff_odds_control  docs/accuracy-2026.md §2b          (the coin-flip control run)

A section that cannot be regenerated falls back to the newest committed dated
snapshot and is marked `stale` with the reason, so the nightly job never fails
because an input was missing.

Usage:
    python scripts/build_accuracy_json.py
    python scripts/build_accuracy_json.py --skip game_odds      # fast, no network
    python scripts/build_accuracy_json.py --components-json fixture.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.projections.ros import LIVE_ENGINE  # noqa: E402  (needs ROOT on sys.path)

OUT_DIR = ROOT / "public/data/accuracy"
ACCURACY_MD = ROOT / "docs/accuracy-2026.md"
VALIDATION_MD = ROOT / "docs/playoff-odds-validation.md"
ARCHITECTURE_MD = ROOT / "docs/architecture.md"
MARKET_PARQUET = ROOT / "data/parquet/market_closes_2026.parquet"
MARKET_R2_KEY = "market/market_closes_2026.parquet"
PLAYOFF_ODDS_DIR = ROOT / "public/data/playoff_odds"
PA_PARQUET = ROOT / "data/parquet/pa_outcomes_2026.parquet"

SECTIONS = ("components", "ros_backtest", "game_odds", "playoff_odds_control")

# The rest-of-season section: four components (BABIP is 4 players at the Aug 1
# cutoff, so it measures nothing there) and the four arms that answer "is
# folding in the current season worth more than our prior?".
ROS_COMPONENTS = ("k_rate", "bb_rate", "hr_rate", "iso")
# `marcel_tuned` leads because it is what the site serves
# (src/projections/ros.py); stock `marcel` stays as the arm it had to beat.
ROS_ARMS = ("marcel_tuned", "marcel", "marcel_tuned_preseason",
            "bayes_preseason", "season_to_date")
ROS_ARM_LABELS = {
    "marcel_tuned": "Tuned Marcel + 2026 to date",
    "marcel": "Stock Marcel + 2026 to date",
    "marcel_tuned_preseason": "Tuned Marcel, 2026 withheld",
    "marcel_preseason": "Stock Marcel, 2026 withheld",
    "bayes_preseason": "Bayes preseason (ours)",
    "season_to_date": "2026 rate, regressed",
}
# Read from the module that serves the projection, so the page can never mark
# an arm as live that src/projections/ros.py is not actually running.
ROS_LIVE_ARM = LIVE_ENGINE
# The control the "is in-season data worth anything?" line is measured against.
ROS_CONTROL_ARM = "marcel_tuned_preseason"

# Display names. Text only — every number comes from a generated table.
MODEL_LABELS = {
    "bayes_preseason": "Bayes (ours)",
    "depth_charts": "Depth Charts",
    "zips": "ZiPS",
    "steamer": "Steamer",
    "marcel": "Marcel",
    "marcel_tuned": "Marcel (tuned)",
    "marcel_tuned_preseason": "Marcel (tuned), preseason",
    "league_average": "League average",
    "home_constant": "Home team always",
    "win_pct_log5": "Raw win% into log5",
    "pythag_60": "Pythagenpat, 60 (production)",
    "pythag_60_sp": "Pythagenpat, 60 + starting pitcher",
    "kalshi_close": "Kalshi close",
    "polymarket_close": "Polymarket close",
}
OURS = {"bayes_preseason", "pythag_60_sp"}
BASELINE_MODELS = {"marcel", "marcel_tuned", "league_average", "home_constant",
                   "win_pct_log5"}
COMPONENT_LABELS = {"k_rate": "K% MAE", "bb_rate": "BB% MAE", "hr_rate": "HR/PA MAE",
                    "iso": "ISO MAE", "babip": "BABIP MAE"}
MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
          "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}


# ─── helpers ──────────────────────────────────────────────────────

def label_for(model: str) -> str:
    if model in MODEL_LABELS:
        return MODEL_LABELS[model]
    m = re.fullmatch(r"pythag_(\d+)", model)
    if m:
        return f"Pythagenpat, {m.group(1)}-game ballast"
    return model.replace("_", " ")


def num(v):
    """JSON-safe number: NaN/inf become null rather than invalid JSON."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def run_script(argv: list[str], timeout: int) -> tuple[bool, str]:
    """Run a repo script; return (ok, message). Never raises."""
    try:
        p = subprocess.run([sys.executable, *argv], cwd=ROOT, timeout=timeout,
                           capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return False, f"{argv[0]} timed out after {timeout}s"
    except OSError as exc:                                   # pragma: no cover
        return False, f"{argv[0]} could not start: {exc}"
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()
        return False, f"{argv[0]} exited {p.returncode}: {tail[-1] if tail else 'no output'}"
    return True, ""


def parse_md_table(text: str, heading_prefix: str) -> list[dict]:
    """First pipe table under a heading, as a list of {column: cell} dicts."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith(heading_prefix)), None)
    if start is None:
        raise ValueError(f"no heading starting {heading_prefix!r}")
    rows, header = [], None
    for ln in lines[start + 1:]:
        stripped = ln.strip()
        if not stripped.startswith("|"):
            if header is not None:
                break
            if stripped.startswith("#"):
                raise ValueError(f"no table under {heading_prefix!r}")
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(dict(zip(header, cells)))
    if not rows:
        raise ValueError(f"no table rows under {heading_prefix!r}")
    return rows


def strip_md(cell: str) -> str:
    return re.sub(r"[*`]", "", cell).strip()


def parse_doc_date(text: str) -> str | None:
    m = re.search(r"[Aa]s of ([A-Za-z]+)\.? (\d{1,2}), (\d{4})", text)
    if not m:
        return None
    month = MONTHS.get(m.group(1).lower().rstrip("."))
    if month is None:
        return None
    return date(int(m.group(3)), month, int(m.group(2))).isoformat()


def gate_rule() -> str | None:
    """The one-sentence gate rule, read out of the architecture doc's §3 quote."""
    try:
        text = ARCHITECTURE_MD.read_text()
    except OSError:
        return None
    m = re.search(r"^> (A station's model .*?)\n\n", text, re.S | re.M)
    if not m:
        return None
    return re.sub(r"\s*\n>\s*", " ", m.group(1)).strip()


# ─── section builders (pure: dict/text in, section dict out) ──────

def section_components(payload: dict) -> dict:
    """Component scoreboard: one row per system, ranked by mean MAE rank."""
    scores = payload["scores"]
    components = payload.get("components") or sorted({r["component"] for r in scores})
    mae = {}
    n_by_component, trials_by_component = {}, {}
    for r in scores:
        mae.setdefault(r["model"], {})[r["component"]] = num(r["mae"])
        n_by_component[r["component"]] = int(r["n_players"])
        trials_by_component[r["component"]] = int(r["total_trials"])
    mean_rank = {r["model"]: num(r.get("mean_rank")) for r in payload.get("mae_rank", [])}
    if not mean_rank:                                        # rank here if absent
        order = {c: sorted(mae, key=lambda m: mae[m][c]) for c in components}
        mean_rank = {m: sum(order[c].index(m) + 1 for c in components) / len(components)
                     for m in mae}

    rows = []
    for model in sorted(mae, key=lambda m: (mean_rank.get(m) is None, mean_rank.get(m), m)):
        rows.append({
            "model": model,
            "label": label_for(model),
            "is_ours": model in OURS,
            "is_baseline": model in BASELINE_MODELS,
            "is_market": False,
            "mean_rank": num(mean_rank.get(model)),
            "metrics": {c: num(mae[model].get(c)) for c in components},
        })
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    return {
        "title": "Preseason player projections vs 2026 actuals",
        "framing": ("Lower is better. Every public system beats our five Bayesian "
                    "components, and Marcel — the unit test — beats them on four of "
                    "five. Until that flips they stay out of the rollup."),
        "source": "scripts/score_2026_projections.py (docs/accuracy-2026.md §1)",
        "as_of": payload.get("as_of") or payload["generated_at"][:10],
        "n": max(n_by_component.values()) if n_by_component else None,
        "n_label": "hitters scored",
        "n_by_component": n_by_component,
        "trials_by_component": trials_by_component,
        "min_trials": payload.get("min_trials"),
        "columns": ([{"key": "rank", "label": "#", "type": "rank"},
                     {"key": "label", "label": "System", "type": "text"}]
                    + [{"key": c, "label": COMPONENT_LABELS.get(c, c), "type": "rate"}
                       for c in components]
                    + [{"key": "mean_rank", "label": "Mean rank", "type": "rank_value"}]),
        "rows": rows,
        "notes": [f"Scored on the same hitters with at least "
                  f"{payload.get('min_trials')} plate appearances, trials-weighted.",
                  "Public systems as captured before Opening Day; ours generated "
                  "the same week. No system saw any 2026 result."],
        "stale": False,
        "stale_reason": None,
    }


def _cutoff_label(cutoff: str) -> str:
    """2026-07-01 -> "Jul 1"."""
    try:
        d = date.fromisoformat(cutoff)
    except ValueError:
        return cutoff
    return f"{d:%b} {d.day}"


def section_ros(payload: dict) -> dict:
    """Rest-of-season projections: the walk-forward that put Marcel in production.

    One row per (cutoff, arm), because MAE rises with every cutoff — a shorter
    rest-of-season is a noisier target — so only the comparison *within* a
    cutoff means anything. The best cell per component is marked inside its own
    cutoff block, and the page is told not to rank the column globally.
    """
    scores = payload["scores"]
    components = [c for c in ROS_COMPONENTS
                  if c in {r["component"] for r in scores}]
    cutoffs = payload.get("cutoffs") or sorted({r["cutoff"] for r in scores})
    arms = [a for a in ROS_ARMS if a in {r["model"] for r in scores}]

    mae, n_players = {}, {}
    for r in scores:
        mae[(r["cutoff"], r["model"], r["component"])] = num(r["mae"])
        n_players[(r["cutoff"], r["component"])] = int(r["n_players"])

    rows = []
    for cutoff in cutoffs:
        best = {}
        for component in components:
            values = [(mae[(cutoff, a, component)], a) for a in arms
                      if (cutoff, a, component) in mae
                      and mae[(cutoff, a, component)] is not None]
            if values:
                best[component] = min(values)[1]
        for arm in arms:
            metrics = {c: mae.get((cutoff, arm, c)) for c in components}
            if all(v is None for v in metrics.values()):
                continue
            rows.append({
                "model": arm,
                "label": ROS_ARM_LABELS.get(arm, label_for(arm)),
                "cutoff": _cutoff_label(cutoff),
                "cutoff_date": cutoff,
                "is_production": arm == ROS_LIVE_ARM,
                "is_ours": arm == "bayes_preseason",
                "is_baseline": arm in ("marcel", "marcel_preseason",
                                       "marcel_tuned_preseason", "season_to_date"),
                "is_market": False,
                "metrics": metrics,
                "best": [c for c in components if best.get(c) == arm],
            })

    # The framing is a count, not a claim: it is recomputed from the table
    # every night, so it flips on its own if the result ever does.
    def beats(other: str) -> tuple[int, int]:
        won = total = 0
        for cutoff in cutoffs:
            for component in components:
                live = mae.get((cutoff, ROS_LIVE_ARM, component))
                rival = mae.get((cutoff, other, component))
                if live is None or rival is None:
                    continue
                total += 1
                won += live < rival
        return won, total

    vs_bayes, n_bayes = beats("bayes_preseason")
    vs_control, n_control = beats(ROS_CONTROL_ARM)
    vs_stock, n_stock = beats("marcel")
    framing = (
        "Lower is better, and only within a cutoff — a later cutoff scores a "
        "shorter, noisier rest of season, so every arm's MAE rises down the "
        "table. The live arm is tuned Marcel — the same estimator with its "
        "ballast, recency weights and age curve fitted walk-forward on "
        "2020-2024 — with the season to date folded in. It beats our preseason "
        f"Bayesian components on {vs_bayes} of {n_bayes} component-cutoff "
        f"cells, the same model without 2026 on {vs_control} of {n_control}, "
        f"and stock Marcel on {vs_stock} of {n_stock}. Most of the gain is "
        "in-season information rather than a better prior — which is why the "
        "player pages lead with this number.")

    last_pa = payload.get("last_pa_date")
    notes = [
        "Training is the prior full seasons plus the current season through the "
        "cutoff; the realized side is every plate appearance on or after it. The "
        "leakage guard rejects a training PA dated on or after the cutoff.",
        f"Scored on hitters with at least {payload.get('min_trials')} realized "
        f"trials after the cutoff, trials-weighted, on the same players across "
        f"all arms.",
        "BABIP is left out: at a one-month horizon the 100-trial floor leaves "
        "four players, and league average ties Marcel on it anyway.",
        "Tuned Marcel's constants are frozen in src/eval/marcel_params.json, "
        "fitted on 2020-2024 only — every cutoff here is out of sample for "
        "them.",
    ]
    if last_pa:
        notes.append(f"The current season runs through {last_pa} in this run, so "
                     f"the last cutoff's 'rest of season' is about a month.")
    if n_players:
        counts = ", ".join(
            f"{_cutoff_label(c)} {n_players[(c, components[0])]}"
            for c in cutoffs if (c, components[0]) in n_players)
        notes.append(f"Hitters scored per cutoff ({COMPONENT_LABELS[components[0]]}): "
                     f"{counts}.")

    return {
        "title": "Rest-of-season projections, walk-forward",
        "framing": framing,
        "source": "scripts/run_intraseason_backtest.py (docs/backtest-baselines.md, "
                  "docs/ros-projections.md)",
        "as_of": (payload.get("generated_at") or "")[:10] or None,
        "n": n_players.get((cutoffs[-1], components[0])) if cutoffs and components else None,
        "n_label": "hitters at the last cutoff",
        "predict_year": payload.get("predict_year"),
        "cutoffs": list(cutoffs),
        "arms": arms,
        "live_arm": ROS_LIVE_ARM,
        "highlight_best": False,
        "columns": ([{"key": "cutoff", "label": "Cutoff", "type": "text"},
                     {"key": "label", "label": "Arm", "type": "text"}]
                    + [{"key": c, "label": COMPONENT_LABELS.get(c, c), "type": "rate"}
                       for c in components]),
        "rows": rows,
        "notes": notes,
        "stale": False,
        "stale_reason": None,
    }


def section_game_odds(payload: dict, market_note: str | None) -> dict:
    """Walk-forward per-game table, market closes included when we have them."""
    rows = []
    # One row per model. The backtester emits a market price once per model
    # subset it scores, so a run that carries the lineup and bullpen arms lists
    # the same Kalshi close two or three times — the same price is not two
    # contenders, and duplicating the bar makes it look beatable by tie.
    seen = set()
    for r in sorted(payload["scores"], key=lambda r: r["brier"]):
        model = r["model"]
        if model in seen:
            continue
        seen.add(model)
        rows.append({
            "model": model,
            "label": label_for(model),
            "is_market": model in payload.get("market_models", []),
            "is_ours": model not in payload.get("market_models", [])
                       and model not in BASELINE_MODELS,
            "is_baseline": model in BASELINE_MODELS,
            "is_production": model == "pythag_60",
            "metrics": {"brier": num(r["brier"]), "log_loss": num(r["log_loss"]),
                        "mean_p_home": num(r["mean_p_home"])},
        })
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    has_market = bool(payload.get("market_models"))
    framing = ("Lower is better. The market's close is the bar — every venue row "
               "is a price, not a model, and it is still ahead of us.")
    if not has_market:
        framing = ("Lower is better. The market's close is the bar, and it is "
                   "missing from this run — these are our models against each other only.")
    return {
        "title": "Per-game win probability, walk-forward",
        "framing": framing,
        "source": "scripts/backtest_game_odds.py (docs/market-benchmark-2026.md)",
        "as_of": payload.get("last_date") or payload["generated_at"][:10],
        "n": int(payload["n_games"]),
        "n_label": "games scored",
        "season": payload.get("season"),
        "first_date": payload.get("first_date"),
        "last_date": payload.get("last_date"),
        "market_models": payload.get("market_models", []),
        "market_file": payload.get("market_file"),
        "realized_home_win_rate": num(payload.get("realized_home_win_rate")),
        "columns": [{"key": "rank", "label": "#", "type": "rank"},
                    {"key": "label", "label": "Model", "type": "text"},
                    {"key": "brier", "label": "Brier", "type": "score"},
                    {"key": "log_loss", "label": "Log loss", "type": "score"},
                    {"key": "mean_p_home", "label": "Mean P(home)", "type": "prob"}],
        "rows": rows,
        "notes": [n for n in [
            "Every game is predicted from results and pitcher appearances strictly "
            "before its date — no model sees the day it is scored on.",
            f"Realized home win rate on these games: "
            f"{float(payload['realized_home_win_rate']):.3f}."
            if payload.get("realized_home_win_rate") is not None else None,
        ] if n],
        "stale": not has_market,
        "stale_reason": market_note if not has_market else None,
    }


def section_control(accuracy_md: str, n_teams: int | None,
                    validation_md: str | None = None) -> dict:
    """The coin-flip control: our playoff odds vs FanGraphs vs no model at all."""
    table = parse_md_table(accuracy_md, "## 2b")
    header = list(table[0].keys())
    metric_cols = header[1:]
    rows = []
    for r in table:
        name = strip_md(r[header[0]])
        is_control = "coin flip" in name.lower()
        rows.append({
            "model": "coin_flip" if is_control else "ours",
            "label": name,
            "is_control": is_control,
            "is_ours": not is_control,
            "is_market": False,
            "metrics": {c: num(strip_md(r[c])) for c in metric_cols},
        })
    # No rank column here on purpose: a smaller gap to FanGraphs is agreement,
    # not accuracy, so ranking the two arms would imply the wrong thing.
    as_of = parse_doc_date(accuracy_md)
    sims = re.search(r"([\d,]+) sims each", accuracy_md)
    notes = [f"Both arms simulated {sims.group(1)} seasons on the same date."
             if sims else None]
    if validation_md:
        met = re.search(r"\*\*Acceptance criterion:\*\*(.+?)\*\*(Met|Not met)\.\*\*",
                        validation_md, re.S)
        if met:
            notes.append("Acceptance criterion: "
                         + re.sub(r"\s+", " ", met.group(1)).strip().rstrip(".")
                         + f". {met.group(2)}.")
    return {
        "title": "Playoff odds: the coin-flip control",
        "framing": ("Lower is a smaller gap to FanGraphs — and that is the point: a "
                    "simulator with no team-strength model at all lands within two "
                    "points of FanGraphs in September, so agreement here is "
                    "arithmetic, not skill."),
        "source": "docs/accuracy-2026.md §2b (run recorded in docs/playoff-odds-validation.md)",
        "as_of": as_of,
        "n": n_teams,
        "n_label": "teams compared",
        "columns": ([{"key": "label", "label": "Strength model", "type": "text"}]
                    + [{"key": c, "label": f"{c} gap", "type": "gap"} for c in metric_cols]),
        "rows": rows,
        "notes": [n for n in notes if n],
        "stale": True,
        "stale_reason": ("Read from the scoreboard doc: the control arm is a one-off "
                         "experiment, not a nightly job, so these numbers are the "
                         f"{as_of or 'recorded'} run and do not move with the site."),
    }


# ─── orchestration ────────────────────────────────────────────────

def ensure_market_parquet(path: Path) -> tuple[Path | None, str | None]:
    """Fetch the market closes from R2 when we can. Returns (path or None, reason)."""
    if path.exists():
        return path, None
    missing = [v for v in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL")
               if not os.getenv(v)]
    if missing:
        return None, (f"market closes not downloaded: {', '.join(missing)} not set on this "
                      f"runner (they would fetch {MARKET_R2_KEY} from R2)")
    try:
        from src.data.r2 import bucket, get_s3_client
        path.parent.mkdir(parents=True, exist_ok=True)
        get_s3_client().download_file(bucket(), MARKET_R2_KEY, str(path))
    except Exception as exc:                                  # noqa: BLE001 — never fail the job
        return None, f"market closes not downloaded from R2: {type(exc).__name__}: {exc}"
    return path, None


def ros_input_note(pa_parquet: Path = PA_PARQUET) -> str | None:
    """Why the rest-of-season backtest cannot run here, or None if it can.

    The PA-level parquet lives in R2 and is gitignored; `load_pa_outcomes` would
    download it, but only with credentials. Checking first turns a five-minute
    doomed subprocess into an instant, honest stale reason.
    """
    if pa_parquet.exists():
        return None
    missing = [v for v in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL")
               if not os.getenv(v)]
    if missing:
        return (f"rest-of-season backtest not run: {pa_parquet.name} is absent and "
                f"{', '.join(missing)} not set on this runner (they would fetch "
                f"pa_outcomes/pa_outcomes_2026.parquet from R2)")
    return None


def newest_previous(out_dir: Path) -> tuple[dict | None, str | None]:
    """Newest committed dated snapshot, for section-level fallback."""
    dated = sorted(p for p in out_dir.glob("*.json") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.stem))
    for path in reversed(dated + [out_dir / "latest.json"]):
        try:
            return json.loads(path.read_text()), path.name
        except (OSError, ValueError):
            continue
    return None, None


def fallback(name: str, reason: str, previous: dict | None, previous_name: str | None) -> dict:
    """Reuse the last good copy of a section, or record its absence."""
    section = (previous or {}).get("sections", {}).get(name)
    if section:
        section = dict(section)
        section["stale"] = True
        section["stale_reason"] = (
            f"{reason}. Showing the {section.get('as_of') or 'previous'} run "
            f"carried over from {previous_name}.")
        return section
    return {"title": name.replace("_", " ").title(), "framing": None, "source": None,
            "as_of": None, "n": None, "n_label": None, "columns": [], "rows": [],
            "notes": [], "stale": True,
            "stale_reason": f"{reason}. No previous snapshot to fall back to."}


def build_document(*, out_dir: Path = OUT_DIR, skip: tuple[str, ...] = (),
                   components_json: Path | None = None, game_odds_json: Path | None = None,
                   ros_json: Path | None = None,
                   market_parquet: Path | None = MARKET_PARQUET, season: int = 2026,
                   timeout: int = 3600, work_dir: Path | None = None,
                   git_sha: str | None = None) -> dict:
    """Assemble the whole document. Never raises for a missing input."""
    previous, previous_name = newest_previous(out_dir)
    sections: dict[str, dict] = {}
    tmp = Path(work_dir or tempfile.mkdtemp(prefix="accuracy-"))
    tmp.mkdir(parents=True, exist_ok=True)

    # 1. components — inputs are committed parquet, so this runs anywhere.
    if "components" in skip:
        sections["components"] = fallback(
            "components", "skipped by --skip components", previous, previous_name)
    else:
        path, err = components_json, ""
        if path is None:
            path = tmp / "components.json"
            ok, err = run_script(["scripts/score_2026_projections.py",
                                  "--json-out", str(path)], timeout)
            path = path if ok else None
        try:
            sections["components"] = section_components(json.loads(Path(path).read_text()))
        except Exception as exc:                              # noqa: BLE001
            sections["components"] = fallback(
                "components", err or f"could not score components: {type(exc).__name__}: {exc}",
                previous, previous_name)

    # 1b. rest-of-season walk-forward — needs the 2026 PA parquet, which comes
    # from R2. The nightly runner without R2_* credentials cannot rebuild it and
    # carries the last committed run instead.
    if "ros_backtest" in skip:
        sections["ros_backtest"] = fallback(
            "ros_backtest", "skipped by --skip ros_backtest", previous, previous_name)
    else:
        path, err = ros_json, ""
        if path is None:
            missing = ros_input_note()
            if missing:
                path, err = None, missing
            else:
                path = tmp / "ros_backtest.json"
                ok, err = run_script(
                    ["scripts/run_intraseason_backtest.py",
                     "--components", *ROS_COMPONENTS, "--json-out", str(path)], timeout)
                path = path if ok else None
        try:
            sections["ros_backtest"] = section_ros(json.loads(Path(path).read_text()))
        except Exception as exc:                              # noqa: BLE001
            sections["ros_backtest"] = fallback(
                "ros_backtest",
                err or f"could not score rest-of-season arms: {type(exc).__name__}: {exc}",
                previous, previous_name)

    # 2. game odds — needs the MLB Stats API, and R2 for the market rows.
    if "game_odds" in skip:
        sections["game_odds"] = fallback(
            "game_odds", "skipped by --skip game_odds", previous, previous_name)
    else:
        path, err, market_note = game_odds_json, "", None
        if path is None:
            market, market_note = ensure_market_parquet(market_parquet)
            path = tmp / "game_odds.json"
            argv = ["scripts/backtest_game_odds.py", "--season", str(season),
                    "--json-out", str(path)]
            if market is not None:
                argv += ["--market", str(market)]
            ok, err = run_script(argv, timeout)
            path = path if ok else None
        try:
            sections["game_odds"] = section_game_odds(
                json.loads(Path(path).read_text()), market_note)
        except Exception as exc:                              # noqa: BLE001
            sections["game_odds"] = fallback(
                "game_odds", err or f"could not backtest game odds: {type(exc).__name__}: {exc}",
                previous, previous_name)

    # 3. playoff-odds control — recorded in the scoreboard doc.
    if "playoff_odds_control" in skip:
        sections["playoff_odds_control"] = fallback(
            "playoff_odds_control", "skipped by --skip playoff_odds_control",
            previous, previous_name)
    else:
        try:
            accuracy_md = ACCURACY_MD.read_text()
            validation_md = VALIDATION_MD.read_text() if VALIDATION_MD.exists() else None
            sections["playoff_odds_control"] = section_control(
                accuracy_md, control_n(accuracy_md, validation_md), validation_md)
        except Exception as exc:                              # noqa: BLE001
            sections["playoff_odds_control"] = fallback(
                "playoff_odds_control",
                f"could not read the control table: {type(exc).__name__}: {exc}",
                previous, previous_name)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": date.today().isoformat(),
        "git_sha": git_sha if git_sha is not None else current_sha(),
        "title": "Model Accuracy",
        "subtitle": ("Every number here is generated by the scoring scripts in this "
                     "repository. Where we lose, it says so."),
        "sections": sections,
        "meta": {
            "generated_by": "scripts/build_accuracy_json.py",
            "glossary": glossary(),
            "status": [{"section": name,
                        "fresh": not sec.get("stale", True),
                        "as_of": sec.get("as_of"),
                        "source": sec.get("source"),
                        "reason": sec.get("stale_reason")}
                       for name, sec in sections.items()],
        },
    }


def control_n(accuracy_md: str, validation_md: str | None) -> int | None:
    """How many teams the control compared — from the snapshot it was run on."""
    as_of = parse_doc_date(accuracy_md)
    snapshot = PLAYOFF_ODDS_DIR / f"{as_of}.json" if as_of else None
    if snapshot is not None and snapshot.exists():
        try:
            return len(json.loads(snapshot.read_text())["teams"])
        except (OSError, ValueError, KeyError):
            pass
    if validation_md:
        m = re.search(r"\d+ of (\d+) teams", validation_md)
        if m:
            return int(m.group(1))
    return None


def glossary() -> list[dict]:
    return [
        {"term": "Brier score",
         "text": "The mean squared error of a probability: predict 0.60 for a game the "
                 "home team wins and you are charged (1 − 0.60)² = 0.16. Lower is better; "
                 "a coin flip on every game scores 0.25."},
        {"term": "Log loss",
         "text": "The same idea with a harsher penalty for confident mistakes — it is the "
                 "negative log of the probability you gave the outcome that happened. "
                 "Lower is better; always guessing 50% scores 0.693."},
        {"term": "MAE",
         "text": "Mean absolute error: the average distance between a projected rate and "
                 "the rate the player actually posted, weighted by playing time. Lower is "
                 "better, and it is in the units of the stat itself."},
        {"term": "The gate rule", "text": gate_rule() or "See docs/architecture.md §3."},
    ]


def current_sha() -> str | None:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):             # pragma: no cover
        return None
    return p.stdout.strip() or None if p.returncode == 0 else None


def write_document(doc: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    dated = out_dir / f"{doc['as_of']}.json"
    if dated.exists():
        print(f"snapshot {dated.name} already exists; not overwriting "
              f"(updating latest.json only)")
    else:
        dated.write_text(json.dumps(doc, indent=1) + "\n")
        written.append(dated)
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(doc, indent=1) + "\n")
    written.append(latest)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--skip", nargs="*", default=[], choices=SECTIONS,
                        help="mark these sections stale instead of rebuilding them")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="seconds allowed per scoring script")
    parser.add_argument("--components-json", type=Path, default=None,
                        help="use this score_2026_projections --json-out file instead of "
                             "running the script (testing)")
    parser.add_argument("--game-odds-json", type=Path, default=None,
                        help="use this backtest_game_odds --json-out file instead of "
                             "running the script (testing)")
    parser.add_argument("--ros-json", type=Path, default=None,
                        help="use this run_intraseason_backtest --json-out file "
                             "instead of running the script (testing)")
    parser.add_argument("--market-parquet", type=Path, default=MARKET_PARQUET)
    args = parser.parse_args()

    doc = build_document(out_dir=args.out_dir, skip=tuple(args.skip), season=args.season,
                         components_json=args.components_json,
                         game_odds_json=args.game_odds_json,
                         ros_json=args.ros_json,
                         market_parquet=args.market_parquet, timeout=args.timeout)
    for row in doc["meta"]["status"]:
        state = "fresh" if row["fresh"] else "STALE"
        print(f"{row['section']:<22} {state:<6} as_of={row['as_of']}"
              + (f"  — {row['reason']}" if row["reason"] else ""))
    for path in write_document(doc, args.out_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
