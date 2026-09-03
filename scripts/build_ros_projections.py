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

The plate appearances have the same problem and get the same treatment: the
document carries `playing_time_method`, the station B method that filled
`pa_ros`, taken from `playing_time.PRODUCTION_METHOD` rather than named here.
Snapshots written before that stamp existed carry no key at all, and were
built with the trailing-30-day share and a hard injured-list zero.

**Pitchers ride along in the same file, additively.** `pitchers` is the
pitcher block (`src/projections/pitcher_ros.py`): the tuned pitcher Marcel's
K%, BB%, HR/BF and BABIP-against times a projected count of batters faced,
stamped `pitcher_engine` and `batters_faced_method`. Nothing above it moved —
`players`, `n_hitters`, `engine`, `arms` and `components` are exactly what
they were, because the page's hitter views were written against them.

Both halves of the pitcher block have been through a gate, and the document
names each one. The **rates** cleared theirs against league average, the
previous season and season to date (`scripts/run_pitcher_backtest.py`). The
**batters faced** cleared theirs on Sept 3, 2026 against a season-to-date rate
extrapolation, a trailing-30-day one, last season prorated and no model at all
— 26 walk-forward as-of dates over 2024-2026
(`scripts/run_pitcher_workload_backtest.py`, docs/pitcher-workload.md) — so
`batters_faced_method` reads `recent_usage` rather than the `structural` it
carried while it was unscored, and `pitcher_method` spells out both the
arithmetic and the margin. It also fails on its own: a missing pitcher season
table leaves
`pitchers` empty and the hitter projection fresh, rather than taking the
site's established product stale with it.

Inputs, and what happens when one is missing:

    2026 PA outcomes      R2 (`pa_outcomes/pa_outcomes_2026.parquet`), cached
                          in data/parquet/. Needs R2_* credentials the first
                          time; the nightly runner may not have them.
    2015-2025 seasons     data/parquet/hitter_seasons_api.parquet and
                          data/parquet/pitcher_seasons_api.parquet (both
                          committed; rebuildable from the Stats API with
                          scripts/build_pitcher_seasons.py).
    playing time          built here from the MLB Stats API as of --as-of,
                          reusing scripts/build_playing_time.py. Needs the
                          transactions feed too, for the expected returns.
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

from src.projections import pitcher_ros
from src.projections.playing_time import PRODUCTION_METHOD
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
PITCHER_SEASONS_PARQUET = ROOT / "data/parquet/pitcher_seasons_api.parquet"
PROJECTIONS_DIR = ROOT / "data/projections"
COMPARISON_PARQUET = PROJECTIONS_DIR / "comparison_2026.parquet"
BIRTHDATES_PARQUET = ROOT / "data/parquet/birthdates.parquet"

# Named in one place each — src/projections/ros.py for the rate engine,
# src/projections/playing_time.py for the playing-time method — so the
# document cannot claim a model the modules do not run.
ENGINE = LIVE_ENGINE
PITCHER_ENGINE = pitcher_ros.LIVE_ENGINE
BATTERS_FACED_METHOD = pitcher_ros.BF_METHOD
PLAYING_TIME_METHOD = PRODUCTION_METHOD
METHOD = ("Tuned Marcel (per-component ballast, recency weights and age curve "
          "fitted walk-forward on 2020-2024 and frozen in "
          "src/eval/marcel_params.json) trained on 2015-2025 season totals "
          "plus 2026 through the day before the as-of date, multiplied by "
          "station B's projected rest-of-season plate appearances "
          "(horizon-blended 30-day and season PA share, one-lineup-slot cap, "
          "and the injured and optioned projected at their pre-injury share "
          "times their expected return fraction).")
PITCHER_METHOD = (
    "Tuned pitcher Marcel (per-component ballast, recency weights and a "
    "constrained age curve fitted walk-forward on 2020-2024 and frozen in "
    "src/eval/marcel_pitcher_params.json) trained on 2015-2025 pitcher season "
    "totals plus 2026 through the day before the as-of date. The rate columns "
    "cleared the serving gate against league average, the previous season and "
    "season to date; the projected batters faced cleared their own on Sept 3, "
    "2026. " + pitcher_ros.BF_METHOD_NOTE)
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

    The method is `playing_time.PRODUCTION_METHOD`, never a literal: station B
    is scored and re-gated on its own schedule (docs/playing-time.md) and the
    site has to serve whatever came through that gate, not whatever was
    current the day this file was written. Its expected-return fractions come
    from the same helper the station's own build uses.
    """
    import build_playing_time as bpt
    from src.data.mlb_stats_api import fetch_schedule
    from src.projections.playing_time import PRODUCTION_METHOD, project_playing_time

    rosters, logs, team_logs, teams = bpt.load([as_of], refresh=refresh)
    schedule = fetch_schedule(as_of, bpt.SEASON_END)
    remaining = bpt.games_remaining(schedule, as_of, bpt.SEASON_END)
    fractions = bpt.production_fractions(rosters[as_of], as_of, bpt.SEASON_END,
                                         refresh=refresh)
    projection = project_playing_time(rosters[as_of], logs, remaining, as_of,
                                      team_logs=team_logs,
                                      method=PRODUCTION_METHOD,
                                      active_fractions=fractions)
    return projection, teams


def pitcher_inputs(as_of: str, refresh: bool = False) -> dict:
    """Everything the pitcher block needs that is not in the PA parquet.

    The 40-man snapshot for `as_of` (who is on a staff, and hurt or optioned),
    the club games played and remaining either side of the cutoff, and station
    B's expected-return fractions applied to pitchers. All of it comes from the
    same cached Stats API calls station B already makes, so this adds one
    roster request per club and nothing else.

    The return-time distribution station B fits is estimated from *all*
    injured-list and option spells, not hitters' only, so reading it for
    pitchers is the same estimate rather than a borrowed one. It has now been
    scored on pitchers as well: removing it costs 4.4 batters faced a pitcher
    over 44 walk-forward cutoffs (t -37.8 on the 2024-2026 holdout), which is
    the largest single term in the workload projection
    (docs/pitcher-workload.md).
    """
    import build_playing_time as bpt
    from src.data.mlb_stats_api import (
        fetch_rosters, fetch_schedule, fetch_team_hitting_game_logs, fetch_teams,
    )

    teams = fetch_teams(SEASON, refresh=refresh)
    team_ids = teams["team_id"].tolist()
    roster = fetch_rosters(team_ids, as_of, refresh=refresh)
    pitchers = roster[~roster["is_hitter"]].copy()

    schedule = fetch_schedule(as_of, bpt.SEASON_END)
    remaining = bpt.games_remaining(schedule, as_of, bpt.SEASON_END)

    team_logs = fetch_team_hitting_game_logs(team_ids, SEASON, refresh=refresh)
    dates = pd.to_datetime(team_logs["date"])
    cutoff = pd.Timestamp(as_of).normalize()
    played = team_logs[dates < cutoff]
    recent = team_logs[(dates < cutoff)
                       & (dates >= cutoff - pd.Timedelta(days=pitcher_ros.RECENT_DAYS))]

    try:
        fractions = bpt.production_fractions(pitchers, as_of, bpt.SEASON_END,
                                             refresh=refresh)
    except Exception as exc:                                  # noqa: BLE001
        logger.warning("expected-return fractions unavailable for pitchers: %s", exc)
        fractions = None
    active = pd.Series(dtype="float64")
    if fractions is not None and len(fractions):
        active = (fractions.set_index(fractions["batter"].astype("int64"))
                  ["active_fraction"].astype(float))
    # An unavailable pitcher the distribution cannot date is projected at zero,
    # the same fallback station B uses. An active one is simply absent from the
    # mapping, which `projected_batters_faced` reads as a full horizon.
    unavailable = set(pitchers.loc[pitchers["status_code"].ne("A"),
                                   "batter"].astype("int64"))
    undated = sorted(unavailable - set(active.index))
    active = pd.concat([active, pd.Series(0.0, index=undated, dtype="float64")])

    return {
        "teams": teams,
        "team_of": pitchers.set_index(pitchers["batter"].astype("int64"))["team_id"],
        "games_remaining": remaining.set_index("team_id")["games_remaining"],
        "team_games_played": played.groupby("team_id")["game_pk"].nunique(),
        "team_games_recent": recent.groupby("team_id")["game_pk"].nunique(),
        "active_fraction": active,
        "n_on_staff": int(len(pitchers)),
    }


def build_pitchers(as_of: str, pa: pd.DataFrame, names: pd.Series,
                   seasons_path: Path = PITCHER_SEASONS_PARQUET,
                   refresh: bool = False) -> pd.DataFrame:
    """The pitcher block, or an empty frame if its own inputs are missing.

    Deliberately non-fatal on its own: the hitter projection is the site's
    established product and a missing pitcher season table must not take it
    down with it. An empty frame renders as "not built" on the page.
    """
    from src.eval import pitchers as pitcher_eval

    if not Path(seasons_path).exists():
        logger.warning("%s not found — no pitcher block", seasons_path)
        return pd.DataFrame(columns=list(pitcher_ros.OUTPUT_COLUMNS))
    seasons = pitcher_eval.normalize_pitcher_seasons(pd.read_parquet(seasons_path))
    inputs = pitcher_inputs(as_of, refresh=refresh)
    return pitcher_ros.build_pitcher_projections(
        as_of, seasons, pa,
        team_of=inputs["team_of"],
        team_games_played=inputs["team_games_played"],
        team_games_recent=inputs["team_games_recent"],
        games_remaining=inputs["games_remaining"],
        active_fraction=inputs["active_fraction"],
        names=names, teams=inputs["teams"], season=SEASON)


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
        if column in ("woba_ros", "fip_ros") or "_rate_" in column:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").round(5)
        elif column.endswith("_ros"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").round(2)
    return frame


def records(frame: pd.DataFrame) -> list[dict]:
    """One JSON record per row, rounded and with NaN as null."""
    frame = round_for_json(frame)
    frame = frame.where(pd.notnull(frame), None)
    return json.loads(frame.to_json(orient="records"))


def to_document(projections: pd.DataFrame, as_of: str, *, git_sha: str | None = None,
                season: int = SEASON, season_end: str | None = None,
                pitchers: pd.DataFrame | None = None) -> dict:
    """The site's JSON: metadata the page can render without knowing the model.

    The hitter half of the contract is exactly what it was — `players`,
    `n_hitters`, `engine`, `arms`, `components` all unchanged — because the
    page's hitter views were written against it and there is no reason to move
    them. The pitcher block is additive: `pitchers`, `n_pitchers`, and a
    parallel set of keys prefixed `pitcher_`, so a reader that has never heard
    of pitchers still finds everything it looks for.
    """
    through = (pd.Timestamp(as_of) - pd.Timedelta(days=1)).date().isoformat()
    players = records(projections)
    pitchers = (pitchers if pitchers is not None
                else pd.DataFrame(columns=list(pitcher_ros.OUTPUT_COLUMNS)))
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
        "playing_time_method": PLAYING_TIME_METHOD,
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
        # ─── the pitcher block (additive; nothing above it moved) ───
        "n_pitchers": int(len(pitchers)),
        "pitcher_engine": PITCHER_ENGINE,
        "batters_faced_method": BATTERS_FACED_METHOD,
        "pitcher_method": PITCHER_METHOD,
        "pitcher_arms": [
            {"key": "marcel", "label": "Live (tuned Marcel + 2026)", "is_live": True,
             "note": "Tuned pitcher Marcel — fitted ballast, recency and age "
                     "curve, src/eval/marcel_pitcher_params.json — with the "
                     "season through " + through + " folded in."},
            {"key": "marcel_preseason", "label": "Preseason tuned Marcel",
             "is_live": False,
             "note": "the same tuned Marcel with 2026 withheld; the difference "
                     "from the live column is what the current season is worth."},
        ],
        "pitcher_components": [
            {"key": c, "prefix": pitcher_ros.COMPONENT_PREFIX[c]}
            for c in pitcher_ros.SERVED_COMPONENTS
        ],
        "pitchers": records(pitchers),
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
        "n_hitters": 0, "engine": ENGINE,
        "playing_time_method": PLAYING_TIME_METHOD, "method": METHOD,
        "framing": FRAMING.format(through="—"),
        "source": "scripts/build_ros_projections.py", "arms": [], "components": [],
        "woba": {"weights": WOBA_WEIGHTS, "triples_per_double": TRIPLES_PER_DOUBLE},
        "stale": True,
        "stale_reason": f"{reason}. No previous projection to fall back to.",
        "players": [],
        "n_pitchers": 0, "pitcher_engine": PITCHER_ENGINE,
        "batters_faced_method": BATTERS_FACED_METHOD,
        "pitcher_method": PITCHER_METHOD, "pitcher_arms": [],
        "pitcher_components": [], "pitchers": [],
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
          pitcher_seasons_path: Path = PITCHER_SEASONS_PARQUET,
          refresh: bool = False) -> dict:
    """Assemble the document. Never raises for a missing input."""
    from src.data.pa_outcomes import load_pa_outcomes

    previous, previous_name = newest_previous(out_dir)
    try:
        import build_playing_time as bpt

        seasons = pd.read_parquet(seasons_path)
        pa = load_pa_outcomes(SEASON, data_dir=pa_dir or (ROOT / "data/parquet"))
        names = load_names()
        playing_time, teams = build_playing_time(as_of, refresh=refresh)
        projections = build_ros_projections(
            as_of, seasons, pa, playing_time,
            bayes_frames=load_bayes_frames(projections_dir),
            names=names, teams=teams, season=SEASON)
    except Exception as exc:                                  # noqa: BLE001
        reason = f"could not rebuild the projection: {type(exc).__name__}: {exc}"
        logger.warning(reason)
        if previous is not None:
            return stale_document(previous, previous_name, reason, as_of)
        return empty_document(reason, as_of)

    # The pitcher block is younger than the hitter one and fails on its own:
    # the site's established product must not go stale because a pitcher input
    # is missing. An empty frame renders as "not built" on the page.
    try:
        pitchers = build_pitchers(as_of, pa, names,
                                  seasons_path=pitcher_seasons_path,
                                  refresh=refresh)
    except Exception as exc:                                  # noqa: BLE001
        logger.warning("could not build the pitcher block: %s: %s",
                       type(exc).__name__, exc)
        pitchers = pd.DataFrame(columns=list(pitcher_ros.OUTPUT_COLUMNS))

    return to_document(projections, as_of, season_end=bpt.SEASON_END,
                       pitchers=pitchers)


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
    parser.add_argument("--pitcher-seasons", type=Path,
                        default=PITCHER_SEASONS_PARQUET)
    parser.add_argument("--refresh", action="store_true",
                        help="re-pull cached Stats API responses")
    parser.add_argument("--top", type=int, default=10,
                        help="how many hitters to print (0 for none)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    doc = build(args.as_of, out_dir=args.out_dir, seasons_path=args.seasons,
                pa_dir=args.pa_dir, projections_dir=args.projections_dir,
                pitcher_seasons_path=args.pitcher_seasons, refresh=args.refresh)

    state = "STALE" if doc["stale"] else "fresh"
    print(f"\nrest-of-season projection {state}: {doc['n_hitters']} hitters, "
          f"{doc.get('n_pitchers', 0)} pitchers, as of {doc['as_of']}"
          + (f"  — {doc['stale_reason']}" if doc["stale_reason"] else ""))
    if args.top and doc["players"]:
        top = pd.DataFrame(doc["players"]).head(args.top)
        cols = ["name", "team_abbrev", "pa_ros", "k_ros", "bb_ros", "hr_ros",
                "woba_ros", "k_rate_marcel", "k_rate_bayes"]
        print(f"\nTop {len(top)} by projected rest-of-season wOBA:")
        print(top[[c for c in cols if c in top.columns]].round(3).to_string(index=False))
    if args.top and doc.get("pitchers"):
        top = pd.DataFrame(doc["pitchers"]).head(args.top)
        cols = ["name", "team_abbrev", "role", "bf_ros", "k_ros", "bb_ros",
                "hr_ros", "fip_ros", "k_rate_marcel", "bb_rate_marcel"]
        print(f"\nTop {len(top)} pitchers by projected rest-of-season FIP:")
        print(top[[c for c in cols if c in top.columns]].round(3).to_string(index=False))
    for path in write_document(doc, args.out_dir):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
