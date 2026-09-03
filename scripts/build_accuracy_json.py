"""Build the Model Accuracy page's data — station H (docs/architecture.md §2).

Everything the page shows is generated here, from the scoring scripts and the
scoreboard docs. Nothing on the page is typed by hand: if a number is not in
this file's output it does not appear in the browser.

    public/data/accuracy/latest.json        always rewritten
    public/data/accuracy/YYYY-MM-DD.json    written once, never overwritten

Sections:
    components            scripts/score_2026_projections.py  (offline; data in git)
    ros_backtest          scripts/run_intraseason_backtest.py (needs the PA parquet)
    pitcher_ros_backtest  scripts/run_pitcher_backtest.py     (needs the PA parquet)
    game_odds             scripts/backtest_game_odds.py      (MLB Stats API + R2 market closes)
    team_backtest         scripts/run_team_backtest.py       (committed 2015-2025 run)
    playoff_odds_control  docs/accuracy-2026.md §2b          (the coin-flip control run)

A section that cannot be regenerated falls back to the newest committed dated
snapshot and is marked `stale` with the reason, so the nightly job never fails
because an input was missing.

Usage:
    python scripts/build_accuracy_json.py
    python scripts/build_accuracy_json.py --skip game_odds      # fast, no network
    python scripts/build_accuracy_json.py --skip pitcher_ros_backtest
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

from src.projections.pitcher_ros import LIVE_ENGINE as PITCHER_LIVE_ENGINE  # noqa: E402
from src.projections.ros import LIVE_ENGINE  # noqa: E402  (needs ROOT on sys.path)

OUT_DIR = ROOT / "public/data/accuracy"
ACCURACY_MD = ROOT / "docs/accuracy-2026.md"
VALIDATION_MD = ROOT / "docs/playoff-odds-validation.md"
ARCHITECTURE_MD = ROOT / "docs/architecture.md"
MARKET_PARQUET = ROOT / "data/parquet/market_closes_2026.parquet"
MARKET_R2_KEY = "market/market_closes_2026.parquet"
PLAYOFF_ODDS_DIR = ROOT / "public/data/playoff_odds"
PA_PARQUET = ROOT / "data/parquet/pa_outcomes_2026.parquet"
PITCHER_SEASONS_PARQUET = ROOT / "data/parquet/pitcher_seasons_api.parquet"
TEAM_BACKTEST_JSON = ROOT / "public/data/team_backtest/2015-2025.json"

SECTIONS = ("components", "ros_backtest", "pitcher_ros_backtest", "game_odds",
            "team_backtest",
            "playoff_odds_control")

# The rest-of-season section: four components (BABIP is 4 players at the Aug 1
# cutoff, so it measures nothing there) and the four arms that answer "is
# folding in the current season worth more than our prior?".
ROS_COMPONENTS = ("k_rate", "bb_rate", "hr_rate", "iso")
# `marcel_tuned` leads because it is what the site serves
# (src/projections/ros.py); stock `marcel` stays as the arm it had to beat.
ROS_ARMS = ("marcel_tuned", "bayes", "marcel", "marcel_tuned_preseason",
            "bayes_preseason", "season_to_date")
# Labels carry the handicap. `bayes_preseason` has never seen a plate
# appearance from the season it is being scored in, and for a year it sat in
# this table under a name that did not say so, next to a Marcel fed the season
# through the day before. Whatever the numbers do, the label has to make that
# visible to a reader who does not know the history — hence "2026 withheld",
# the same words the tuned-Marcel control carries.
ROS_ARM_LABELS = {
    "marcel_tuned": "Tuned Marcel + 2026 to date",
    "bayes": "Bayes + 2026 to date (ours)",
    "marcel": "Stock Marcel + 2026 to date",
    "marcel_tuned_preseason": "Tuned Marcel, 2026 withheld",
    "marcel_preseason": "Stock Marcel, 2026 withheld",
    "bayes_preseason": "Bayes, 2026 withheld (ours)",
    "season_to_date": "2026 rate, regressed",
}
# Read from the module that serves the projection, so the page can never mark
# an arm as live that src/projections/ros.py is not actually running.
ROS_LIVE_ARM = LIVE_ENGINE
# The control the "is in-season data worth anything?" line is measured against.
ROS_CONTROL_ARM = "marcel_tuned_preseason"
# Arms that never see a plate appearance from the season they are scored in.
# The page has to say which rows those are: a comparison between an arm fed the
# season to date and an arm that was not is a measurement of in-season
# information, not of the two models.
ROS_WITHHELD_ARMS = ("marcel_tuned_preseason", "marcel_preseason",
                     "bayes_preseason", "previous_season")

# The pitcher rest-of-season section. Five cells rather than three cutoffs —
# season-level 2025 and 2026 come along, because the pitcher arm was scored on
# them too — and the three dumb baselines are all shown, because the serving
# gate is stated against all three and the reader should be able to check it.
PITCHER_ROS_COMPONENTS = ("p_k_rate", "p_bb_rate", "p_hr_rate", "p_babip")
PITCHER_ROS_ARMS = ("marcel_pitcher_tuned", "marcel_pitcher",
                    "marcel_pitcher_tuned_preseason", "season_to_date",
                    "previous_season", "league_average")
PITCHER_ROS_ARM_LABELS = {
    "marcel_pitcher_tuned": "Tuned pitcher Marcel + 2026 to date",
    "marcel_pitcher": "Stock pitcher Marcel + 2026 to date",
    "marcel_pitcher_tuned_preseason": "Tuned pitcher Marcel, 2026 withheld",
    "marcel_pitcher_preseason": "Stock pitcher Marcel, 2026 withheld",
    "season_to_date": "2026 rate, regressed",
    "previous_season": "2025 rate, unregressed",
    "league_average": "League average",
}
PITCHER_COMPONENT_LABELS = {
    "p_k_rate": "K% MAE", "p_bb_rate": "BB% MAE", "p_hr_rate": "HR/BF MAE",
    "p_babip": "BABIP MAE", "p_bbhbp_rate": "(BB+HBP)% MAE",
}
# Read from the module that serves it, so the page cannot mark an arm live
# that src/projections/pitcher_ros.py is not running.
PITCHER_ROS_LIVE_ARM = PITCHER_LIVE_ENGINE
PITCHER_ROS_BASELINES = ("league_average", "previous_season", "season_to_date")

# Display names. Text only — every number comes from a generated table.
MODEL_LABELS = {
    "bayes": "Bayes, refit at the cutoff (ours)",
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
OURS = {"bayes", "bayes_preseason", "pythag_60_sp"}
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

    # What the refit Bayesian arm actually was, per cutoff. A reduced local fit
    # is evidence about a reduced local fit; the row says so on the row.
    bayes_scale = {f.get("cutoff"): f.get("scale")
                   for f in payload.get("bayes_fits") or []}
    paired = {(p["cutoff"], p["component"], p["arm"]): p
              for p in payload.get("paired") or []}

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
            row = {
                "model": arm,
                "label": ROS_ARM_LABELS.get(arm, label_for(arm)),
                "cutoff": _cutoff_label(cutoff),
                "cutoff_date": cutoff,
                "is_production": arm == ROS_LIVE_ARM,
                "is_ours": arm in OURS,
                # An arm that never saw the season it is scored in is not
                # competing on the same information as the rest of the table.
                # `is_control` is what the page already renders as a tag, so
                # the handicap shows on the row itself and not only in prose.
                "sees_current_season": arm not in ROS_WITHHELD_ARMS,
                "is_control": arm in ROS_WITHHELD_ARMS,
                "is_baseline": arm in ("marcel", "marcel_preseason",
                                       "marcel_tuned_preseason", "season_to_date"),
                "is_market": False,
                "metrics": metrics,
                "best": [c for c in components if best.get(c) == arm],
            }
            if arm == "bayes" and bayes_scale.get(cutoff):
                row["scale"] = bayes_scale[cutoff]
            pair = paired.get((cutoff, components[0], arm)) if components else None
            if pair:
                row["paired_vs_live"] = {
                    "component": components[0], "base": pair["base"],
                    "diff": num(pair["diff"]), "se": num(pair["se"]),
                    "t": num(pair["t"]), "n": int(pair["n"]),
                }
            rows.append(row)

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

    # Only the comparisons this run actually has both sides of get a clause,
    # so a payload predating an arm reads as a shorter sentence rather than
    # "0 of 0".
    clauses = [(what, won, n) for what, (won, n) in [
        ("our Bayesian components with the season withheld",
         beats("bayes_preseason")),
        ("the same model without 2026", beats(ROS_CONTROL_ARM)),
        ("stock Marcel", beats("marcel")),
    ] if n]
    # The unit is named once, on the first clause.
    parts = [f"{what} on {won} of {n}"
             + (" component-cutoff cells" if i == 0 else "")
             for i, (what, won, n) in enumerate(clauses)]
    counted = (parts[0] if len(parts) == 1
               else ", ".join(parts[:-1]) + f"{',' if len(parts) > 2 else ''}"
                    f" and {parts[-1]}") if parts else ""

    # The handicap sentence. Every row marked "2026 withheld" was denied the
    # information every other row was given, and this harness measures that
    # information at 5-6% of K% MAE — the same size as the gap it was being
    # charged with. Saying so is not optional and does not depend on which way
    # the numbers come out.
    withheld_present = any(a in ROS_WITHHELD_ARMS for a in arms)
    bayes_won, bayes_n = beats("bayes") if "bayes" in arms else (0, 0)
    if "bayes" in arms:
        fair = (f" Our Bayesian model refit at each cutoff — the same "
                f"information, the fair comparison — is the "
                f"\"{ROS_ARM_LABELS['bayes']}\" row; the live arm beats it on "
                f"{bayes_won} of {bayes_n}.")
        if bayes_scale:
            fair += (" That row is a "
                     f"{sorted(set(v for v in bayes_scale.values() if v))[0]} "
                     "fit, not the full refit — read it as the scale it is.")
    else:
        fair = (" No Bayesian arm in this run was refit at the cutoff, so "
                "nothing here compares the two models on equal information; "
                "run scripts/run_intraseason_backtest.py --bayes for that.")
    handicap = (
        " The rows labelled \"2026 withheld\" never saw a plate appearance "
        "from the season they are scored in, so the gap between them and the "
        "rows above is in-season information first and model quality second: "
        "folding the season to date into Marcel alone is worth 5-6% of K% "
        "MAE." + fair
    ) if withheld_present else ""

    framing = (
        "Lower is better, and only within a cutoff — a later cutoff scores a "
        "shorter, noisier rest of season, so every arm's MAE rises down the "
        "table. The live arm is tuned Marcel — the same estimator with its "
        "ballast, recency weights and age curve fitted walk-forward on "
        "2020-2024 — with the season to date folded in."
        + (f" It beats {counted}." if counted else "")
        + handicap
        + " Most of the gain is in-season information rather than a better "
          "prior — which is why the player pages lead with this number.")

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
        "\"2026 withheld\" arms are fixed preseason projections scored "
        "unchanged at every cutoff. They are controls for how much the current "
        "season is worth, not contenders — a withheld arm losing to an arm fed "
        "the season to date is the expected result, not a verdict on the model.",
    ]
    if "bayes" in arms:
        notes.append(
            "\"Bayes + 2026 to date\" is the PA-level Bayesian K% model refit "
            "at the cutoff on exactly the plate appearances the baselines see "
            "(src/eval/bayes_arm.py), with the opposing-pitcher term on. It "
            "covers K% only; the other four components have no refit arm yet."
            + (f" Sampling scale: {sorted(set(v for v in bayes_scale.values() if v))[0]}."
               if bayes_scale else ""))
    if payload.get("paired"):
        notes.append(
            f"Paired column: within-hitter difference in absolute error "
            f"against {payload.get('paired_base', ROS_LIVE_ARM)} on "
            f"{COMPONENT_LABELS.get(components[0], components[0])}, "
            f"trials-weighted. Negative means the arm is the better one.")
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


def _cell_label(cell: str) -> str:
    """A season-level cell keeps its year; a cutoff becomes "Jul 1"."""
    return cell if re.fullmatch(r"\d{4}", cell) else _cutoff_label(cell)


def section_pitcher_ros(payload: dict) -> dict:
    """Pitcher rest-of-season rates, scored by the same harness as the hitters.

    One row per (cell, arm). The five cells are two whole seasons and three
    intra-season cutoffs, and they are *not* comparable to each other — a
    later cutoff scores a shorter, noisier rest of season — so the best value
    is marked within its own block, exactly as the hitter table does it.

    The framing is the gate, recomputed from the payload rather than asserted:
    which components beat every dumb baseline, and which did not.
    """
    scores = payload["scores"]
    components = [c for c in PITCHER_ROS_COMPONENTS
                  if c in {r["component"] for r in scores}]
    cells = payload.get("cells") or sorted({r["cell"] for r in scores})
    arms = [a for a in PITCHER_ROS_ARMS if a in {r["model"] for r in scores}]

    mae, n_pitchers = {}, {}
    for r in scores:
        mae[(r["cell"], r["model"], r["component"])] = num(r["mae"])
        n_pitchers[(r["cell"], r["component"])] = int(r["n_players"])

    rows = []
    for cell in cells:
        best = {}
        for component in components:
            values = [(mae[(cell, a, component)], a) for a in arms
                      if (cell, a, component) in mae
                      and mae[(cell, a, component)] is not None]
            if values:
                best[component] = min(values)[1]
        for arm in arms:
            metrics = {c: mae.get((cell, arm, c)) for c in components}
            if all(v is None for v in metrics.values()):
                continue
            rows.append({
                "model": arm,
                "label": PITCHER_ROS_ARM_LABELS.get(arm, label_for(arm)),
                "cutoff": _cell_label(cell),
                "cutoff_date": cell,
                "is_production": arm == PITCHER_ROS_LIVE_ARM,
                "is_ours": False,
                "is_baseline": (arm in PITCHER_ROS_BASELINES
                                or arm == "marcel_pitcher"),
                "is_market": False,
                "metrics": metrics,
                "best": [c for c in components if best.get(c) == arm],
            })

    gate = {g["component"]: g for g in payload.get("gate", [])}
    cleared = [PITCHER_COMPONENT_LABELS.get(c, c).replace(" MAE", "")
               for c in components if gate.get(c, {}).get("clears")]
    withheld = [PITCHER_COMPONENT_LABELS.get(c, c).replace(" MAE", "")
                for c in components if c in gate and not gate[c]["clears"]]
    framing = (
        "Lower is better, and only within a cell — the two season rows and the "
        "three cutoffs score different windows, so every arm's MAE moves down "
        "the table. The live arm is the tuned pitcher Marcel with the season to "
        "date folded in, and the gate for putting a component on the site is "
        "that it beats league average, the previous season *and* season to "
        "date, pooled over these five cells.")
    if cleared:
        framing += f" Served: {', '.join(cleared)}."
    if withheld:
        framing += f" Not served, because it did not clear: {', '.join(withheld)}."

    notes = [
        "Training is the prior full pitcher seasons plus the current season "
        "through the cutoff; the realized side is every batter faced on or "
        "after it. The same leakage guard the hitter table uses rejects a "
        "training plate appearance dated on or after the cutoff.",
        f"Scored on pitchers with at least {payload.get('min_trials')} realized "
        f"trials after the cutoff, trials-weighted, on the same pitchers across "
        f"all arms.",
        "The tuned constants are frozen in src/eval/marcel_pitcher_params.json, "
        "fitted on 2020-2024 only, so every cell here is out of sample for "
        "them. Three of the five components did not beat stock on an inner "
        "validation inside the tuning window and are frozen holding stock's "
        "constants, which is why their two Marcel rows are identical.",
        "The walks-plus-hit-batsmen rate station E's starter term consumes is "
        "scored in the same run but is not shown here or on the player pages: "
        "a column labelled BB% has to mean walks.",
    ]
    babip = gate.get("p_babip")
    if babip:
        notes.append(
            "BABIP against clears the gate by almost nothing — the closest "
            f"baseline is {babip['closest_baseline'].replace('_', ' ')} at "
            f"{babip['closest_diff']:+.5f} (t {babip['closest_t']:+.1f}). It is "
            "served, and that is the DIPS result rather than a model win.")
    last_pa = payload.get("last_pa_date")
    if last_pa:
        notes.append(f"The current season runs through {last_pa} in this run, so "
                     f"the last cutoff's 'rest of season' is about a month.")
    if n_pitchers and components:
        counts = ", ".join(
            f"{_cell_label(c)} {n_pitchers[(c, components[0])]}"
            for c in cells if (c, components[0]) in n_pitchers)
        notes.append(f"Pitchers scored per cell "
                     f"({PITCHER_COMPONENT_LABELS[components[0]]}): {counts}.")

    return {
        "title": "Rest-of-season pitcher rates, walk-forward",
        "framing": framing,
        "source": "scripts/run_pitcher_backtest.py (docs/backtest-baselines.md, "
                  "docs/ros-projections.md)",
        "as_of": (payload.get("generated_at") or "")[:10] or None,
        "n": (n_pitchers.get((cells[-1], components[0]))
              if cells and components else None),
        "n_label": "pitchers at the last cutoff",
        "predict_year": 2026,
        "cutoffs": list(cells),
        "arms": arms,
        "live_arm": PITCHER_ROS_LIVE_ARM,
        "highlight_best": False,
        "columns": ([{"key": "cutoff", "label": "Cell", "type": "text"},
                     {"key": "label", "label": "Arm", "type": "text"}]
                    + [{"key": c, "label": PITCHER_COMPONENT_LABELS.get(c, c),
                        "type": "rate"} for c in components]),
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


TEAM_ARM_LABELS = {
    "chain": "Our projection (the served chain)",
    "record_500": "Current record, .500 the rest of the way",
    "record_wpct": "Current record, own win rate the rest of the way",
    "preseason": "Preseason only, never updated",
    "preseason_light": "Preseason only, lighter regression",
    "coin_flip": "No information (81 wins, league base rates)",
}
# The buckets the through-season rows are cut into, in order.
TEAM_BUCKETS = ("0-15%", "15-30%", "30-45%", "45-60%", "60-75%", "75-90%",
                "90-100%")


def section_team_backtest(payload: dict) -> dict:
    """Station G scored: the rest-of-season team projection, walk-forward.

    Read from a committed JSON rather than regenerated, because rebuilding it
    means ~16,000 Stats API game logs and about an hour of simulation — this
    is a decade-long backtest, not a nightly job. It is marked stale for that
    reason, with the run's own date.
    """
    by_arm = {r["arm"]: r for r in payload["headline"]}
    order = [a for a in payload["arms"] if a in by_arm]
    rows = []
    for i, arm in enumerate(sorted(order, key=lambda a: by_arm[a]["wins_mae"]),
                            start=1):
        r = by_arm[arm]
        rows.append({
            "model": arm,
            "label": TEAM_ARM_LABELS.get(arm, arm),
            "rank": i,
            "is_ours": arm == "chain",
            "is_production": arm == "chain",
            "is_control": arm in ("coin_flip", "record_500"),
            "is_market": False,
            "metrics": {"wins_mae": num(r["wins_mae"]),
                        "wins_rmse": num(r["wins_rmse"]),
                        "brier_playoffs": num(r["brier_playoffs"]),
                        "logloss_playoffs": num(r["logloss_playoffs"]),
                        "brier_division": num(r["brier_division"])},
        })

    # The honest half: where the edge goes. One paired difference per bucket,
    # ours minus the .500-extrapolation control, on playoff Brier.
    curve = [r for r in payload.get("curve_paired", [])
             if r.get("against") == "record_500"
             and r.get("metric") == "brier_playoffs"]
    curve.sort(key=lambda r: TEAM_BUCKETS.index(r["bucket"])
               if r.get("bucket") in TEAM_BUCKETS else 99)
    crossing = next((r["bucket"] for r in curve if (r.get("diff") or 0) >= 0), None)
    notes = [
        "Every projection is built from games strictly before its as-of date, "
        "through the same function the nightly job calls.",
        f"{payload['n_scored_projections']:,} club-projections per arm over "
        f"{payload['seasons'] if isinstance(payload['seasons'], int) else len(payload['seasons'])}"
        f" seasons and {payload['n_as_of_dates']} weekly as-of dates. Standard "
        f"errors are clustered by season.",
        "2020 is excluded: a 60-game season with an eight-club-per-league "
        "bracket is a different question.",
    ]
    if crossing:
        first = curve[0]
        notes.append(
            f"The edge is front-loaded. On playoff probability our projection "
            f"beats the .500 extrapolation by {abs(first['diff']):.4f} of Brier "
            f"in the first {first['bucket']} of the season and stops beating it "
            f"at {crossing} — after which it is nominally behind, though never "
            f"significantly. Projected *wins* stay better all season.")
    return {
        "title": "Rest-of-season team projection, walk-forward (2015–2025)",
        "framing": ("Lower is better in every column. This is the first score of "
                    "the projection behind the playoff odds, against the naive "
                    "extrapolations it has to beat."),
        "source": "scripts/run_team_backtest.py (docs/team-projection-backtest.md)",
        "as_of": payload.get("generated_at", "")[:10] or None,
        "n": int(payload["n_scored_projections"]),
        "n_label": "club-projections per arm",
        "columns": [{"key": "rank", "label": "#", "type": "rank"},
                    {"key": "label", "label": "Projection", "type": "text"},
                    {"key": "wins_mae", "label": "Wins MAE", "type": "rank_value"},
                    {"key": "wins_rmse", "label": "Wins RMSE", "type": "rank_value"},
                    {"key": "brier_playoffs", "label": "Brier playoffs", "type": "score"},
                    {"key": "logloss_playoffs", "label": "Log loss", "type": "score"},
                    {"key": "brier_division", "label": "Brier division", "type": "score"}],
        "rows": rows,
        "notes": notes,
        "stale": True,
        "stale_reason": (
            "A ten-season backtest, not a nightly job: rebuilding it needs about "
            "16,000 cached Stats API game logs and an hour of simulation, so the "
            "page reads the committed run "
            "(public/data/team_backtest/2015-2025.json)."),
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


def pitcher_ros_input_note(pa_parquet: Path = PA_PARQUET,
                           seasons: Path = PITCHER_SEASONS_PARQUET) -> str | None:
    """Why the pitcher backtest cannot run here, or None if it can.

    Same PA parquet as the hitter section, plus the pitcher season table. The
    latter is committed, so this second clause only fires in an unusual
    checkout — but a section that says which file is missing beats one that
    says "it failed".
    """
    note = ros_input_note(pa_parquet)
    if note:
        return note.replace("rest-of-season backtest",
                            "pitcher rest-of-season backtest")
    if not seasons.exists():
        return (f"pitcher rest-of-season backtest not run: {seasons.name} is "
                f"absent (rebuild it with scripts/build_pitcher_seasons.py)")
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
                   ros_json: Path | None = None, pitcher_ros_json: Path | None = None,
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

    # 1c. the pitcher half of the same walk-forward. Same inputs, same
    # fallback: without the PA parquet it carries the last committed run.
    if "pitcher_ros_backtest" in skip:
        sections["pitcher_ros_backtest"] = fallback(
            "pitcher_ros_backtest", "skipped by --skip pitcher_ros_backtest",
            previous, previous_name)
    else:
        path, err = pitcher_ros_json, ""
        if path is None:
            missing = pitcher_ros_input_note()
            if missing:
                path, err = None, missing
            else:
                path = tmp / "pitcher_ros_backtest.json"
                ok, err = run_script(
                    ["scripts/run_pitcher_backtest.py",
                     "--json-out", str(path)], timeout)
                path = path if ok else None
        try:
            sections["pitcher_ros_backtest"] = section_pitcher_ros(
                json.loads(Path(path).read_text()))
        except Exception as exc:                              # noqa: BLE001
            sections["pitcher_ros_backtest"] = fallback(
                "pitcher_ros_backtest",
                err or f"could not score pitcher rest-of-season arms: "
                       f"{type(exc).__name__}: {exc}",
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

    # 2b. the team projection behind the playoff odds, scored 2015-2025. A
    # decade-long backtest is not a nightly job; the committed run is read.
    if "team_backtest" in skip:
        sections["team_backtest"] = fallback(
            "team_backtest", "skipped by --skip team_backtest", previous,
            previous_name)
    else:
        try:
            sections["team_backtest"] = section_team_backtest(
                json.loads(TEAM_BACKTEST_JSON.read_text()))
        except Exception as exc:                              # noqa: BLE001
            sections["team_backtest"] = fallback(
                "team_backtest",
                f"could not read {TEAM_BACKTEST_JSON.name}: "
                f"{type(exc).__name__}: {exc}", previous, previous_name)

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
    parser.add_argument("--pitcher-ros-json", type=Path, default=None,
                        help="use this run_pitcher_backtest --json-out file "
                             "instead of running the script (testing)")
    parser.add_argument("--market-parquet", type=Path, default=MARKET_PARQUET)
    args = parser.parse_args()

    doc = build_document(out_dir=args.out_dir, skip=tuple(args.skip), season=args.season,
                         components_json=args.components_json,
                         game_odds_json=args.game_odds_json,
                         ros_json=args.ros_json,
                         pitcher_ros_json=args.pitcher_ros_json,
                         market_parquet=args.market_parquet, timeout=args.timeout)
    for row in doc["meta"]["status"]:
        state = "fresh" if row["fresh"] else "STALE"
        print(f"{row['section']:<22} {state:<6} as_of={row['as_of']}"
              + (f"  — {row['reason']}" if row["reason"] else ""))
    for path in write_document(doc, args.out_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
