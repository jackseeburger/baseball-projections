"""Station A + B, live — write the rest-of-season projection the site serves.

    python3 scripts/build_ros_projections.py --as-of 2026-09-02

Writes `public/data/projections/latest.json` (always rewritten) and
`public/data/projections/YYYY-MM-DD.json` (written once, never overwritten),
the same archive discipline `build_accuracy_json.py` and `run_playoff_odds.py`
use.

The model is **tuned Marcel** (`src/eval/baselines.marcel_tuned`, constants in
`src/eval/marcel_params.json`) fed the partial current season — the arm that
wins the intra-season harness (docs/backtest-baselines.md) — times station B's
projected rest-of-season plate appearances. The preseason Bayesian components
and the preseason tuned Marcel ride along as labelled comparison columns; the
Bayesian one is what the site used to show as its only number.

The arm *keys* in the document stay `marcel` / `marcel_preseason` — they name a
column on the page, not a set of constants — so the document also carries
`engine`, which says outright which Marcel filled them. Dated snapshots written
before the switch carry `engine: null` and the old `method` string.

Inputs, and what happens when one is missing:

    2026 PA outcomes      R2 (`pa_outcomes/pa_outcomes_2026.parquet`), cached
                          in data/parquet/. Needs R2_* credentials the first
                          time; the nightly runner may not have them.
    2015-2025 seasons     data/parquet/hitter_seasons_api.parquet (gitignored;
                          rebuildable from the Stats API).
    playing time          built here from the MLB Stats API as of --as-of,
                          reusing scripts/build_playing_time.py.
    preseason Bayesian    data/projections/{component}_projections_2026.parquet
                          (committed). Optional — the comparison column simply
                          goes blank without it.

If any required input is unavailable the script does **not** fail: it keeps
the last committed file, re-stamps it `stale: true` with the reason, and exits
0, so the nightly job carries yesterday's projection with a visible badge
rather than serving nothing.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from src.projections.ros import (
    ARMS,
    COMPONENT_ORDER,
    COMPONENT_PREFIX,
    LIVE_ENGINE,
    TRIPLES_PER_DOUBLE,
    WOBA_WEIGHTS,
    build_ros_projections,
)

SEASON = 2026
OUT_DIR = ROOT / "public/data/projections"
SEASONS_PARQUET = ROOT / "data/parquet/hitter_seasons_api.parquet"
PROJECTIONS_DIR = ROOT / "data/projections"
COMPARISON_PARQUET = PROJECTIONS_DIR / "comparison_2026.parquet"
BIRTHDATES_PARQUET = ROOT / "data/parquet/birthdates.parquet"

# Named in one place (src/projections/ros.py) so the document cannot claim an
# engine the module does not use.
ENGINE = LIVE_ENGINE
METHOD = ("Tuned Marcel (per-component ballast, recency weights and age curve "
          "fitted walk-forward on 2020-2024 and frozen in "
          "src/eval/marcel_params.json) trained on 2015-2025 season totals "
          "plus 2026 through the day before the as-of date, multiplied by "
          "station B's projected rest-of-season plate appearances (30-day PA "
          "share, IL zeroed, one-lineup-slot cap).")
FRAMING = ("Live projection: tuned Marcel with 2026 through {through}. "
           "Preseason Bayesian shown for comparison — see Model Accuracy.")

logger = logging.getLogger("build_ros_projections")


# ─── inputs ───────────────────────────────────────────────────────

def load_names() -> pd.Series:
    """batter → display name, from whatever name source this checkout has.

    `comparison_2026.parquet` is committed and covers the ~590 hitters the rest
    of the site already names; the Chadwick birthdate table (gitignored, but
    present wherever the pipeline has run) covers everyone else. Missing names
    are not fatal — the page falls back to the MLBAM id.
    """
    names: dict[int, str] = {}
    if BIRTHDATES_PARQUET.exists():
        try:
            bd = pd.read_parquet(BIRTHDATES_PARQUET,
                                 columns=["batter", "name_first", "name_last"])
            bd = bd.dropna(subset=["batter"])
            full = (bd["name_first"].fillna("").astype(str) + " "
                    + bd["name_last"].fillna("").astype(str)).str.strip()
            names.update(dict(zip(bd["batter"].astype(int), full)))
        except Exception as exc:                              # noqa: BLE001
            logger.warning("birthdates name lookup unavailable: %s", exc)
    if COMPARISON_PARQUET.exists():
        try:
            comp = pd.read_parquet(COMPARISON_PARQUET, columns=["batter", "name"])
            comp = comp.dropna(subset=["batter", "name"])
            names.update(dict(zip(comp["batter"].astype(int), comp["name"].astype(str))))
        except Exception as exc:                              # noqa: BLE001
            logger.warning("comparison name lookup unavailable: %s", exc)
    return pd.Series(names, dtype="object")


def load_bayes_frames(projections_dir: Path = PROJECTIONS_DIR) -> dict[str, pd.DataFrame]:
    """component → preseason Bayesian projection frame, skipping absent files."""
    frames = {}
    for component in COMPONENT_ORDER:
        path = projections_dir / f"{component}_projections_2026.parquet"
        if not path.exists():
            continue
        try:
            frames[component] = pd.read_parquet(path)
        except Exception as exc:                              # noqa: BLE001
            logger.warning("could not read %s: %s", path.name, exc)
    return frames


def build_playing_time(as_of: str, refresh: bool = False):
    """Station B's projection as of `as_of`, plus the teams frame for abbrevs.

    Reuses `scripts/build_playing_time.py` rather than duplicating the
    fetch/assemble layer, but does not write its parquet — this is a read of
    station B, not a rebuild of it.
    """
    import build_playing_time as bpt
    from src.data.mlb_stats_api import fetch_schedule
    from src.projections.playing_time import project_playing_time

    rosters, logs, team_logs, teams = bpt.load([as_of], refresh=refresh)
    schedule = fetch_schedule(as_of, bpt.SEASON_END)
    remaining = bpt.games_remaining(schedule, as_of, bpt.SEASON_END)
    projection = project_playing_time(rosters[as_of], logs, remaining, as_of,
                                      team_logs=team_logs, method="last_30")
    return projection, teams


# ─── the document ─────────────────────────────────────────────────

def rate_columns() -> list[dict]:
    return [{"key": f"{COMPONENT_PREFIX[c]}_rate_{arm}", "component": c, "arm": arm}
            for c in COMPONENT_ORDER for arm in ARMS]


def round_for_json(projections: pd.DataFrame) -> pd.DataFrame:
    """Trim float precision before serializing.

    A dated snapshot is written every night and never rewritten, so the file
    is an archive line item, not a scratch file. Full float64 repr triples its
    size to buy digits nobody displays: rates are shown to four places, counts
    to one.
    """
    frame = projections.copy()
    for column in frame.columns:
        if column.endswith("_ros") and column != "woba_ros":
            frame[column] = pd.to_numeric(frame[column], errors="coerce").round(2)
        elif column == "woba_ros" or "_rate_" in column:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").round(5)
    return frame


def to_document(projections: pd.DataFrame, as_of: str, *, git_sha: str | None = None,
                season: int = SEASON, season_end: str | None = None) -> dict:
    """The site's JSON: metadata the page can render without knowing the model."""
    through = (pd.Timestamp(as_of) - pd.Timedelta(days=1)).date().isoformat()
    frame = round_for_json(projections)
    frame = frame.where(pd.notnull(frame), None)
    players = json.loads(frame.to_json(orient="records"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of,
        "season": season,
        "season_end": season_end,
        "through": through,
        "git_sha": git_sha if git_sha is not None else current_sha(),
        "title": "Rest-of-season projection",
        "n_hitters": int(len(projections)),
        "engine": ENGINE,
        "method": METHOD,
        "framing": FRAMING.format(through=through),
        "source": "scripts/build_ros_projections.py",
        "arms": [
            {"key": "marcel", "label": "Live (tuned Marcel + 2026)", "is_live": True,
             "note": "Tuned Marcel — fitted ballast, recency and age curve, "
                     "src/eval/marcel_params.json — with the season through "
                     + through + " folded in."},
            {"key": "bayes", "label": "Preseason Bayesian", "is_live": False,
             "note": "our hierarchical components, fit through 2025 — the number "
                     "the site used to show on its own."},
            {"key": "marcel_preseason", "label": "Preseason tuned Marcel",
             "is_live": False,
             "note": "the same tuned Marcel with 2026 withheld; the difference "
                     "from the live column is what the current season is worth."},
        ],
        "components": [{"key": c, "prefix": COMPONENT_PREFIX[c]} for c in COMPONENT_ORDER],
        "woba": {"weights": WOBA_WEIGHTS, "triples_per_double": TRIPLES_PER_DOUBLE,
                 "note": "FanGraphs 2024 linear weights, same constants as the "
                         "preseason wOBA on this site."},
        "stale": False,
        "stale_reason": None,
        "players": players,
    }


def stale_document(previous: dict, previous_name: str, reason: str, as_of: str) -> dict:
    """Yesterday's projection, re-stamped so the page can say why it is old."""
    doc = dict(previous)
    doc["generated_at"] = datetime.now(timezone.utc).isoformat()
    doc["stale"] = True
    doc["stale_reason"] = (
        f"{reason}. Showing the {previous.get('as_of') or 'previous'} projection "
        f"carried over from {previous_name}.")
    doc["requested_as_of"] = as_of
    return doc


def empty_document(reason: str, as_of: str) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of, "season": SEASON, "season_end": None, "through": None,
        "git_sha": current_sha(), "title": "Rest-of-season projection",
        "n_hitters": 0, "engine": ENGINE, "method": METHOD,
        "framing": FRAMING.format(through="—"),
        "source": "scripts/build_ros_projections.py", "arms": [], "components": [],
        "woba": {"weights": WOBA_WEIGHTS, "triples_per_double": TRIPLES_PER_DOUBLE},
        "stale": True,
        "stale_reason": f"{reason}. No previous projection to fall back to.",
        "players": [],
    }


def newest_previous(out_dir: Path) -> tuple[dict | None, str | None]:
    """Newest committed dated snapshot, for the stale fallback."""
    dated = sorted(p for p in out_dir.glob("*.json")
                   if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.stem))
    for path in reversed(dated + [out_dir / "latest.json"]):
        try:
            return json.loads(path.read_text()), path.name
        except (OSError, ValueError):
            continue
    return None, None


def current_sha() -> str | None:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):             # pragma: no cover
        return None
    return (p.stdout.strip() or None) if p.returncode == 0 else None


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


def build(as_of: str, *, out_dir: Path = OUT_DIR, seasons_path: Path = SEASONS_PARQUET,
          pa_dir: Path | None = None, projections_dir: Path = PROJECTIONS_DIR,
          refresh: bool = False) -> dict:
    """Assemble the document. Never raises for a missing input."""
    from src.data.pa_outcomes import load_pa_outcomes

    previous, previous_name = newest_previous(out_dir)
    try:
        import build_playing_time as bpt

        seasons = pd.read_parquet(seasons_path)
        pa = load_pa_outcomes(SEASON, data_dir=pa_dir or (ROOT / "data/parquet"))
        playing_time, teams = build_playing_time(as_of, refresh=refresh)
        projections = build_ros_projections(
            as_of, seasons, pa, playing_time,
            bayes_frames=load_bayes_frames(projections_dir),
            names=load_names(), teams=teams, season=SEASON)
    except Exception as exc:                                  # noqa: BLE001
        reason = f"could not rebuild the projection: {type(exc).__name__}: {exc}"
        logger.warning(reason)
        if previous is not None:
            return stale_document(previous, previous_name, reason, as_of)
        return empty_document(reason, as_of)

    return to_document(projections, as_of, season_end=bpt.SEASON_END)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", default=date.today().isoformat(),
                        help="projection date, YYYY-MM-DD (default: today). The "
                             "partial season runs through the day before.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seasons", type=Path, default=SEASONS_PARQUET)
    parser.add_argument("--pa-dir", type=Path, default=ROOT / "data/parquet")
    parser.add_argument("--projections-dir", type=Path, default=PROJECTIONS_DIR)
    parser.add_argument("--refresh", action="store_true",
                        help="re-pull cached Stats API responses")
    parser.add_argument("--top", type=int, default=10,
                        help="how many hitters to print (0 for none)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    doc = build(args.as_of, out_dir=args.out_dir, seasons_path=args.seasons,
                pa_dir=args.pa_dir, projections_dir=args.projections_dir,
                refresh=args.refresh)

    state = "STALE" if doc["stale"] else "fresh"
    print(f"\nrest-of-season projection {state}: {doc['n_hitters']} hitters, "
          f"as of {doc['as_of']}"
          + (f"  — {doc['stale_reason']}" if doc["stale_reason"] else ""))
    if args.top and doc["players"]:
        top = pd.DataFrame(doc["players"]).head(args.top)
        cols = ["name", "team_abbrev", "pa_ros", "k_ros", "bb_ros", "hr_ros",
                "woba_ros", "k_rate_marcel", "k_rate_bayes"]
        print(f"\nTop {len(top)} by projected rest-of-season wOBA:")
        print(top[[c for c in cols if c in top.columns]].round(3).to_string(index=False))
    for path in write_document(doc, args.out_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
