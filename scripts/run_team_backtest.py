"""Walk-forward team backtest: score the rest-of-season projection (station G).

The site publishes playoff, division, pennant and World Series odds every
night and, until this script, no rest-of-season *team* projection had ever
been scored. `scripts/run_playoff_odds.py` contains no scoring code;
`docs/playoff-odds-validation.md` tests whether the board discriminates (the
coin-flip control) and whether it agrees with FanGraphs, neither of which is
accuracy. The player side has had a walk-forward harness since the
intra-season work. This is its team analogue.

Three stages, each resumable — the fetch is API-bound and the projection is
CPU-bound, and a container restart in the middle of either should cost only
the season it was on:

    fetch      one season's schedule, final standings, probables feed, and
               every pitcher's and hitter's game log, into
               `data/cache/statsapi/` and `data/parquet/team_backtest/`.
    project    weekly as-of dates through the season; at each one, cut the
               season, project the rest of it with every arm, checkpoint the
               season's rows to parquet.
    score      join the projections to what happened and print (or write) the
               headline table, the paired differences, the calibration
               deciles and the through-season curve.

Arms at every as-of date:

    chain              the production projection — station C's blended run
                       environment as team strength, the full per-game chain
                       on every remaining game with both probables posted,
                       and each club's regressed-FIP rotation in the bracket.
                       Built by `run_playoff_odds.chain_terms`, the same
                       function the nightly job calls.
    record_500         the club's current record, .500 the rest of the way.
                       In the Monte Carlo this is every club at .500, which is
                       the coin-flip control docs/playoff-odds-validation.md
                       has scored against since the first run.
    record_wpct        the club's current record, extrapolated at its own
                       season-to-date win rate.
    preseason          a projection made before the season and never updated:
                       last season's run rates, regressed. Same numbers at
                       every cutoff, by construction.
    coin_flip          no information at all: 81 wins for everybody and the
                       league base rate for every probability.

Usage:
    python scripts/run_team_backtest.py --stage fetch --seasons 2015-2025
    python scripts/run_team_backtest.py --stage project --seasons 2015-2025 --sims 5000
    python scripts/run_team_backtest.py --stage score --markdown
    python scripts/run_team_backtest.py --stage all --seasons 2024-2025
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.mlb_stats_api import (
    build_seasons_table, fetch_hitter_game_logs, fetch_pitcher_game_logs,
    fetch_probables, fetch_schedule, fetch_season_hitting,
    fetch_season_pitching, fetch_standings,
)
from src.eval import team_backtest as tb
from src.eval import team_season as ts
from src.sim.bracket import DEFAULT_ROTATION_SIZE
from src.sim.teams import fetch_teams

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/parquet/team_backtest"
# 2020 is excluded, deliberately and by name: a 60-game season with an
# eight-club-per-league bracket seeded by division place is not the season
# this projection projects, and folding it in would quietly average two
# different questions. See docs/team-projection-backtest.md.
EXCLUDED_SEASONS = (2020,)
DEFAULT_SEASONS = tuple(y for y in range(2015, 2026) if y not in EXCLUDED_SEASONS)
# Prior completed seasons the chain's Marcel weights read, on both sides of
# the ball. Two, matching `scripts/run_playoff_odds.PRIOR_SEASONS`.
PRIOR_SEASONS = 2
PRIOR_HITTER_CACHE = OUT_DIR / "hitter_seasons_backtest.parquet"
ARMS = ("chain", "record_500", "record_wpct", "preseason", "coin_flip")


# ─── fetch ───

def season_paths(season: int, out_dir: Path, tag: str = "") -> dict[str, Path]:
    """Where one season's fetched inputs and its projections live.

    `tag` names a projection *variant* — the sensitivity runs write
    `projections_2024_w0.parquet` beside `projections_2024.parquet` and share
    every fetched input, so a second pass over the same seasons costs no
    requests.
    """
    return {
        "schedule": out_dir / f"schedule_{season}.parquet",
        "standings": out_dir / f"standings_{season}.parquet",
        "teams": out_dir / f"teams_{season}.parquet",
        "pitching": out_dir / f"pitching_logs_{season}.parquet",
        "hitting": out_dir / f"hitting_logs_{season}.parquet",
        "projections": out_dir / f"projections_{season}{tag}.parquet",
    }


def fetch_season(season: int, out_dir: Path, *, workers: int,
                 force: bool = False) -> None:
    """Everything one season needs, cached to parquet beside the JSON cache.

    The Stats API game-log endpoint is one request per player-season and the
    chain needs the whole population on both sides of the ball, so this is
    ~1,500 requests a season. They are independent GETs and land in distinct
    cache files, so `--workers` is a straight speedup; the frame assembled is
    identical either way (`mlb_stats_api._fetch_many`).
    """
    paths = season_paths(season, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if force or not paths["teams"].exists():
        fetch_teams(season).to_parquet(paths["teams"], index=False)
    if force or not paths["schedule"].exists():
        # Through November: the postseason is the outcome side of the harness.
        sched = fetch_schedule(f"{season}-02-15", f"{season}-11-30")
        sched.to_parquet(paths["schedule"], index=False)
    if force or not paths["standings"].exists():
        fetch_standings(season).to_parquet(paths["standings"], index=False)

    # Probables: cached per month by the fetcher itself. Pulled for the whole
    # season here and truncated to the live job's window at each cutoff.
    fetch_probables(f"{season}-03-01", f"{season}-11-15")

    if force or not paths["pitching"].exists():
        pitchers = fetch_season_pitching(season)
        logs = fetch_pitcher_game_logs(pitchers["pitcher"], season,
                                       workers=workers)
        logs = logs[logs["game_type"] == "R"].reset_index(drop=True)
        logs.to_parquet(paths["pitching"], index=False)
        print(f"  {season}: {len(logs)} pitching appearances")
    if force or not paths["hitting"].exists():
        hitters = fetch_season_hitting(season)
        batters = hitters.loc[hitters["pa"] > 0, "batter"]
        logs = fetch_hitter_game_logs(batters, season, workers=workers)
        logs.to_parquet(paths["hitting"], index=False)
        print(f"  {season}: {len(logs)} hitting lines")

    # The prior seasons the chain Marcel-weights alongside the current one.
    for y in range(season - PRIOR_SEASONS, season):
        fetch_season_pitching(y)
        p = season_paths(y, out_dir)["standings"]
        if force or not p.exists():
            fetch_standings(y).to_parquet(p, index=False)
    build_seasons_table(season - PRIOR_SEASONS, season - 1,
                        cache_path=PRIOR_HITTER_CACHE)


# ─── project ───

def load_season(season: int, out_dir: Path) -> dict:
    paths = season_paths(season, out_dir)
    missing = [k for k in ("teams", "schedule", "standings", "pitching",
                           "hitting") if not paths[k].exists()]
    if missing:
        raise FileNotFoundError(
            f"{season}: run --stage fetch first (missing {', '.join(missing)})")
    prior_standings_path = season_paths(season - 1, out_dir)["standings"]
    return {
        "teams": pd.read_parquet(paths["teams"]),
        "schedule": pd.read_parquet(paths["schedule"]),
        "standings": pd.read_parquet(paths["standings"]),
        "pitching": pd.read_parquet(paths["pitching"]),
        "hitting": pd.read_parquet(paths["hitting"]),
        "prior_standings": (pd.read_parquet(prior_standings_path)
                            if prior_standings_path.exists() else None),
        "probables": fetch_probables(f"{season}-03-01", f"{season}-11-15"),
        "prior_pitching": pd.concat(
            [fetch_season_pitching(y)
             for y in range(season - PRIOR_SEASONS, season)],
            ignore_index=True),
        "prior_hitting": build_seasons_table(season - PRIOR_SEASONS, season - 1,
                                             cache_path=PRIOR_HITTER_CACHE),
    }


def coin_flip_frame(split: ts.TeamSplit) -> pd.DataFrame:
    """No information: half the schedule won, and the league's own base rates.

    The floor every other arm has to clear. Its probabilities are the
    climatological ones — a season seats `2·(3 + wild cards)` clubs in
    October, six win divisions, two win pennants and one wins the World
    Series — which is the best a forecaster with no information about baseball
    can do, and a strictly better Brier than a literal 0.5 everywhere.
    """
    fmt = split.fmt
    n = len(split.state.team_ids)
    st = split.standings.set_index("team_id")
    scheduled = split.games_played + split.games_remaining
    per_club = 2.0 * scheduled / n
    frame = pd.DataFrame({
        "season": split.season, "as_of": split.as_of, "arm": "coin_flip",
        "team_id": split.teams["team_id"].astype(int).to_numpy(),
        "abbrev": split.teams["abbrev"].to_numpy(),
        "league_id": split.teams["league_id"].astype(int).to_numpy(),
        "division_id": split.teams["division_id"].astype(int).to_numpy(),
        "strength": 0.5,
        "proj_final_wins": per_club / 2.0,
    })
    frame["wins_to_date"] = frame["team_id"].map(st["wins"]).astype(int)
    frame["losses_to_date"] = frame["team_id"].map(st["losses"]).astype(int)
    frame["proj_rest_wins"] = frame["proj_final_wins"] - frame["wins_to_date"]
    frame["p_playoffs"] = 2.0 * fmt.n_seeds / n
    frame["p_division"] = 6.0 / n
    frame["p_pennant"] = 2.0 / n
    frame["p_ws"] = 1.0 / n
    frame["games_played"] = split.games_played
    frame["games_remaining"] = split.games_remaining
    frame["club_games_remaining"] = frame["team_id"].map(
        split.club_games_remaining()).astype(int)
    return frame


def rebase_preseason(frame: pd.DataFrame, split: ts.TeamSplit) -> pd.DataFrame:
    """A fixed preseason projection, re-dated onto a later cutoff.

    The projection itself does not move — that is what "held fixed all season"
    means, and it is the arm that shows what in-season information is worth.
    Only the club's banked record is re-read, so that `proj_rest_wins` is the
    remaining wins this preseason number *implies* at this date. It can go
    negative for a club that has already beaten its whole preseason
    projection, and that is the arm's honest failure mode, not a bug.
    """
    st = split.standings.set_index("team_id")
    out = frame.copy()
    out["as_of"] = split.as_of
    out["wins_to_date"] = out["team_id"].map(st["wins"]).astype(int)
    out["losses_to_date"] = out["team_id"].map(st["losses"]).astype(int)
    out["proj_rest_wins"] = out["proj_final_wins"] - out["wins_to_date"]
    out["games_played"] = split.games_played
    out["games_remaining"] = split.games_remaining
    out["club_games_remaining"] = out["team_id"].map(
        split.club_games_remaining()).astype(int)
    return out


def project_season(season: int, data: dict, *, n_sims: int, step_days: int,
                   window_days: int, rotation_size: int,
                   verbose: bool = True) -> pd.DataFrame:
    """Every arm at every weekly cutoff of one season."""
    teams, sched = data["teams"], data["schedule"]
    cutoffs = ts.weekly_cutoffs(sched, teams, step_days=step_days)
    if not cutoffs:
        raise ValueError(f"{season}: no cutoffs from the schedule")

    # The preseason arm, computed once on opening day and re-dated after.
    opening = str(pd.to_datetime(
        ts.regular_season_games(sched, teams)["date"].astype(str)).min().date())
    pre_split = ts.split_season_at(sched, teams, opening, season)
    if data["prior_standings"] is None:
        raise FileNotFoundError(
            f"{season}: the preseason arm needs {season - 1} final standings")
    pre_strength = ts.strength_preseason(data["prior_standings"],
                                         pre_split.state.team_ids)
    pre_frame = ts.project(pre_split, pre_strength, "preseason",
                           n_sims=n_sims, seed=season)

    frames = []
    for i, as_of in enumerate(cutoffs):
        seed = season * 1000 + i
        split = ts.split_season_at(sched, teams, as_of, season)
        inputs = ts.chain_inputs_before(season, data["pitching"],
                                        data["hitting"], data["prior_pitching"],
                                        data["prior_hitting"], as_of)
        probables = ts.probables_to_window(data["probables"], as_of, window_days)
        ts.assert_team_split_clean(split, inputs=inputs, probables=probables,
                                   window_days=window_days)

        chain, diag = ts.project_chain(
            split, inputs, probables, sched, n_sims=n_sims, seed=seed,
            window_days=window_days, rotation_size=rotation_size)
        frames.append(chain)
        frames.append(ts.project(split, ts.strength_even(split), "record_500",
                                 n_sims=n_sims, seed=seed))
        frames.append(ts.project(split, ts.strength_own_rate(split),
                                 "record_wpct", n_sims=n_sims, seed=seed))
        frames.append(rebase_preseason(pre_frame, split))
        frames.append(coin_flip_frame(split))
        if verbose:
            print(f"  {season} {as_of}: {split.games_played} played, "
                  f"{split.games_remaining} left, "
                  f"{diag['n_games_with_starters']} repriced, "
                  f"{diag['n_teams_with_rotation']} rotations", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["n_sims"] = n_sims
    return out


# ─── score ───

def load_projections(seasons, out_dir: Path, tag: str = "") -> pd.DataFrame:
    frames = []
    for season in seasons:
        p = season_paths(season, out_dir, tag)["projections"]
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(f"no projections under {out_dir}")
    return pd.concat(frames, ignore_index=True)


def load_outcomes(seasons, out_dir: Path) -> pd.DataFrame:
    frames = []
    for season in seasons:
        paths = season_paths(season, out_dir)
        if not (paths["schedule"].exists() and paths["standings"].exists()):
            continue
        sched = pd.read_parquet(paths["schedule"])
        teams = pd.read_parquet(paths["teams"])
        standings = pd.read_parquet(paths["standings"])
        post = sched[sched["game_type"].isin(ts.POSTSEASON_TYPES)]
        if not len(post) or not post["home_score"].notna().any():
            # An unfinished season has no outcomes; 2026 is scored by nothing.
            continue
        frames.append(ts.season_outcomes(sched, teams, standings)
                      .assign(season=season))
    if not frames:
        raise FileNotFoundError("no completed seasons to score against")
    return pd.concat(frames, ignore_index=True)


def headline(scored: pd.DataFrame) -> pd.DataFrame:
    return tb.score(scored).set_index("arm").reindex(
        [a for a in ARMS if a in set(scored["arm"])]).reset_index()


def paired_table(scored: pd.DataFrame, arm: str = "chain") -> pd.DataFrame:
    others = [a for a in ARMS if a != arm and a in set(scored["arm"])]
    return pd.concat([tb.paired(scored, arm, other) for other in others],
                     ignore_index=True)


def curve(scored: pd.DataFrame, arm: str = "chain") -> pd.DataFrame:
    """Accuracy through the season: every arm in each fifth-of-a-season bucket."""
    return tb.score(scored, by=("bucket", "arm"))


def curve_paired(scored: pd.DataFrame, arm: str = "chain") -> pd.DataFrame:
    others = [a for a in ARMS if a != arm and a in set(scored["arm"])]
    return pd.concat([tb.paired(scored, arm, other, extra_by=("bucket",))
                      for other in others], ignore_index=True)


def _fmt(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v:.{nd}f}"


def markdown(scored: pd.DataFrame) -> str:
    """The doc tables, generated rather than typed."""
    head = headline(scored)
    lines = ["### Headline: every arm, all scored projections", "",
             "| Arm | n | Final wins MAE | RMSE | Rest-of-season win% MAE | "
             "Brier playoffs | Log loss | Brier division | Brier pennant | "
             "Brier WS |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in head.itertuples(index=False):
        lines.append(
            f"| {r.arm} | {r.n} | {r.wins_mae:.2f} | {r.wins_rmse:.2f} | "
            f"{r.rest_wpct_mae:.4f} | {r.brier_playoffs:.4f} | "
            f"{r.logloss_playoffs:.4f} | {r.brier_division:.4f} | "
            f"{r.brier_pennant:.4f} | {r.brier_ws:.4f} |")

    pair = paired_table(scored)
    lines += ["", "### Paired against the chain, clustered by season", "",
              "| Baseline | Metric | chain | baseline | Δ (chain − baseline) | "
              "se | t |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in pair.itertuples(index=False):
        lines.append(f"| {r.against} | {r.metric} | {_fmt(r.mean_a)} | "
                     f"{_fmt(r.mean_b)} | {_fmt(r.diff)} | {_fmt(r.se)} | "
                     f"{_fmt(r.t, 2)} |")

    cal = tb.calibration(scored, "chain")
    lines += ["", "### Calibration of P(playoffs), chain, pooled over seasons",
              "", "| Decile | n | Predicted range | Mean predicted | "
              "Realized | Gap |", "|---:|---:|---|---:|---:|---:|"]
    for r in cal.itertuples(index=False):
        lines.append(f"| {r.decile} | {r.n} | {r.pred_lo:.3f}–{r.pred_hi:.3f} | "
                     f"{r.mean_pred:.3f} | {r.realized:.3f} | {r.gap:+.3f} |")

    cv = curve(scored)
    lines += ["", "### Through the season", "",
              "| Season played | Arm | n | Final wins MAE | "
              "Rest-of-season win% MAE | Brier playoffs |",
              "|---|---|---:|---:|---:|---:|"]
    for r in cv.itertuples(index=False):
        lines.append(f"| {r.bucket} | {r.arm} | {r.n} | {r.wins_mae:.2f} | "
                     f"{r.rest_wpct_mae:.4f} | {r.brier_playoffs:.4f} |")

    cp = curve_paired(scored)
    lines += ["", "### Through the season, paired against the chain "
                  "(negative favours the chain)", "",
              "| Season played | Baseline | Metric | Δ | se | t | n |",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in cp.itertuples(index=False):
        if r.metric not in ("wins_abs_err", "brier_playoffs"):
            continue
        lines.append(f"| {r.bucket} | {r.against} | {r.metric} | "
                     f"{_fmt(r.diff)} | {_fmt(r.se)} | {_fmt(r.t, 2)} | "
                     f"{r.n} |")
    return "\n".join(lines)


def ascii_curve(scored: pd.DataFrame, metric: str = "brier_playoffs",
                width: int = 46) -> str:
    """The through-season curve as a plot that survives a plain-text file."""
    cv = curve(scored)
    arms = [a for a in ARMS if a in set(cv["arm"])]
    wide = cv.pivot(index="bucket", columns="arm", values=metric)
    wide = wide.reindex(columns=arms)
    lo, hi = float(np.nanmin(wide.to_numpy())), float(np.nanmax(wide.to_numpy()))
    span = max(hi - lo, 1e-9)
    marks = {"chain": "C", "record_500": "F", "record_wpct": "W",
             "preseason": "P", "coin_flip": "X"}
    lines = [f"{metric}: {lo:.4f} (left) to {hi:.4f} (right)",
             "  " + " ".join(f"{marks[a]}={a}" for a in arms)]
    for bucket, row in wide.iterrows():
        cells = [" "] * width
        for a in arms:
            v = row.get(a)
            if v is None or not np.isfinite(v):
                continue
            j = min(width - 1, max(0, int(round((v - lo) / span * (width - 1)))))
            cells[j] = marks[a] if cells[j] == " " else "*"
        lines.append(f"{str(bucket):>9} |" + "".join(cells) + "|")
    return "\n".join(lines)


def payload(scored: pd.DataFrame, seasons, n_sims) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seasons": [int(s) for s in sorted(scored["season"].unique())],
        "requested_seasons": [int(s) for s in seasons],
        "excluded_seasons": list(EXCLUDED_SEASONS),
        "n_sims": int(n_sims),
        "n_as_of_dates": int(scored.groupby("season")["as_of"].nunique().sum()),
        "n_scored_projections": int(len(scored[scored["arm"] == "chain"])),
        "arms": list(ARMS),
        "headline": json.loads(headline(scored).to_json(orient="records")),
        "paired": json.loads(paired_table(scored).to_json(orient="records")),
        "calibration": json.loads(
            tb.calibration(scored, "chain").to_json(orient="records")),
        "reliability": [tb.reliability(scored, a) for a in ARMS
                        if a in set(scored["arm"])],
        "curve": json.loads(curve(scored).to_json(orient="records")),
        "curve_fine": json.loads(
            tb.score(scored, by=("fine_bucket", "arm")).to_json(orient="records")),
        "curve_paired": json.loads(curve_paired(scored).to_json(orient="records")),
    }


# ─── cli ───

def parse_seasons(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return [s for s in out if s not in EXCLUDED_SEASONS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("fetch", "project", "score", "all"),
                        default="all")
    parser.add_argument("--seasons", default="2015-2025",
                        help="e.g. 2015-2025 or 2019,2021,2024 (2020 always "
                             "excluded; see the module docstring)")
    parser.add_argument("--sims", type=int, default=5000)
    parser.add_argument("--step-days", type=int, default=7,
                        help="spacing of the as-of dates")
    parser.add_argument("--window-days", type=int, default=ts.STARTER_WINDOW_DAYS,
                        help="how far ahead posted probables are allowed to "
                             "reach, matching the nightly job's 7. 0 lets the "
                             "starter term reach only the games on the as-of "
                             "date itself, which is strictly less than a live "
                             "run sees and is the sensitivity check on a "
                             "window that, in a season already played, is fed "
                             "by who actually started")
    parser.add_argument("--rotation-size", type=int, default=DEFAULT_ROTATION_SIZE)
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel game-log fetches")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--tag", default="",
                        help="suffix for the projection checkpoints, so a "
                             "sensitivity run (e.g. --window-days 0 --tag _w0) "
                             "sits beside the main one and shares every fetch")
    parser.add_argument("--force", action="store_true",
                        help="re-fetch and re-project seasons already on disk")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    seasons = parse_seasons(args.seasons)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("fetch", "all"):
        for season in seasons:
            print(f"fetching {season} …", flush=True)
            fetch_season(season, args.out_dir, workers=args.workers,
                         force=args.force)

    if args.stage in ("project", "all"):
        for season in seasons:
            path = season_paths(season, args.out_dir, args.tag)["projections"]
            if path.exists() and not args.force:
                print(f"{season}: projections on disk, skipping", flush=True)
                continue
            print(f"projecting {season} …", flush=True)
            frame = project_season(
                season, load_season(season, args.out_dir), n_sims=args.sims,
                step_days=args.step_days, window_days=args.window_days,
                rotation_size=args.rotation_size)
            frame.to_parquet(path, index=False)
            print(f"  wrote {path} ({len(frame)} rows)", flush=True)

    if args.stage in ("score", "all"):
        projections = load_projections(seasons, args.out_dir, args.tag)
        outcomes = load_outcomes(seasons, args.out_dir)
        scored = tb.attach_outcomes(projections, outcomes)
        n_dates = int(scored.groupby("season")["as_of"].nunique().sum())
        print(f"\n{len(scored)} scored club-projections over "
              f"{scored['season'].nunique()} seasons and {n_dates} as-of dates "
              f"({len(scored[scored['arm'] == 'chain'])} per arm)\n")
        if args.markdown:
            print(markdown(scored))
        else:
            print(headline(scored).round(4).to_string(index=False))
            print("\npaired against the chain (negative favours the chain):")
            print(paired_table(scored).round(5).to_string(index=False))
            print("\ncalibration of P(playoffs), chain:")
            print(tb.calibration(scored, "chain").round(4).to_string(index=False))
            print("\nthrough the season:")
            print(curve(scored).round(4).to_string(index=False))
            print("\n" + ascii_curve(scored))
        if args.csv_out:
            args.csv_out.parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(args.csv_out, index=False)
            print(f"\nwrote {args.csv_out}")
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(payload(scored, seasons, args.sims), indent=1) + "\n")
            print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
