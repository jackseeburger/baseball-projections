"""Choose the reliever-availability constants walk-forward, on one season.

`src/sim/reliever_usage.py` has three free knobs: the pitches thrown yesterday
that rule a man out (`HARD_1D_PITCHES`), the pitches over two days that do the
same and also divide the taper below it (`HARD_2D_PITCHES`), and what the
availability-weighted pen is measured against (`BASELINE`: the league's
relievers on the same weights, or the club's own whole pen). This script sweeps
all three on a season the term is *not* scored on, so the constants that ship
were never chosen by looking at the scored set — the gate rule in
docs/architecture.md §3.

It scores exactly the pair the gate compares:

    pythag_C_sp        the current best model (station C + announced starter)
    pythag_C_sp_bpa    the same with the 3.5-inning availability delta on top

for every cell of the grid, on one walk-forward pass: the expensive half (the
per-date Marcel/FIP rates, the run environment, the posted lineups) is built
once per date and every cell reuses it, which is the only reason a 2-D sweep
over a season is affordable.

Usage:
    python scripts/sweep_reliever_usage.py --season 2025
    python scripts/sweep_reliever_usage.py --season 2025 \
        --hard-1d 25,30,35,40,45 --hard-2d 35,45,55,65
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import scripts.backtest_game_odds as bt
from src.data.mlb_stats_api import fetch_schedule
from src.sim import lineups as lu_model
from src.sim import reliever_usage as ru_model
from src.sim import starters as sp_model
from src.sim.strength import HFA_PRIOR
from src.sim.teams import fetch_teams

BASE_MODEL = bt.C_SP_MODEL
NEW_MODEL = bt.C_SP_BPA_MODEL


def cell_day_context(bp_day: dict, bp_ctx: dict, date: str,
                     hard_1d: float, hard_2d: float, taper: float) -> dict:
    """`bullpen_day_context`'s output with one cell's availability weights.

    Only the weights change from cell to cell; the pen frames and the FIP rate
    table behind them are the same object every time, which is what makes the
    sweep one pass rather than one pass per cell.
    """
    frames, ra9, lg_ra9 = bp_day["frames"], bp_day["ra9"], bp_day["lg_ra9"]
    weights = ru_model.availability(bp_ctx["usage"], date, hard_1d=hard_1d,
                                    hard_2d=hard_2d, taper=taper)

    def available(team_id: int, starter_id: int) -> float:
        grp = frames.get(int(team_id))
        if grp is None:
            return lg_ra9
        return ru_model.available_pen_ra9(grp, ra9, lg_ra9, weights,
                                          exclude=(int(starter_id),))

    return {**bp_day, "weights": weights, "available": available,
            "lg_bpa_ra9": ru_model.league_available_pen_ra9(
                bp_day["pens"], ra9, lg_ra9, weights)}


def sweep(completed: pd.DataFrame, team_ids, min_games: int, sp_ctx: dict,
          lu_ctx: dict, bp_ctx: dict, c_ctx: dict, cells: list,
          baselines: list) -> pd.DataFrame:
    """One row per (game, cell) with the base and challenger probabilities.

    The date loop is `backtest_game_odds.walk_forward`'s, reduced to the two
    models the sweep compares.
    """
    completed = completed.sort_values("date").reset_index(drop=True)
    rows = []
    lu_history: dict[int, list[float]] = {}
    for date, day in completed.groupby("date", sort=True):
        before = completed[completed["date"] < date]
        tot = bt.team_totals(before, team_ids) if len(before) else None
        if tot is None or tot["g"].min() < min_games:
            continue
        hfa = (HFA_PRIOR * 2000 + before["home_win"].sum()) / (2000 + len(before))
        sp_day = bt.starter_day_context(tot, date, sp_ctx)
        lu_day = bt.lineup_day_context(tot, date, day, lu_ctx, lu_history)
        bp_day = bt.bullpen_day_context(tot, date, bp_ctx, sp_day)
        c_day = bt.run_env_day_context(tot, date, c_ctx, sp_day, lu_day, bp_day)
        by_cell = {cell: cell_day_context(bp_day, bp_ctx, date, *cell)
                   for cell in cells}
        for g in day.itertuples(index=False):
            base = bt.run_env_game_probs(g, c_day, sp_day, hfa)[BASE_MODEL]
            for (h1, h2, tap), cell_day in by_cell.items():
                for baseline in baselines:
                    probs = bt.run_env_game_probs(
                        g, c_day, sp_day, hfa, cell_day,
                        {**bp_ctx, "bpa_baseline": baseline})
                    rows.append({
                        "date": date, "game_pk": int(g.game_pk),
                        "home_win": bool(g.home_win), "hard_1d": h1,
                        "hard_2d": h2, "taper": tap, "baseline": baseline,
                        BASE_MODEL: base, NEW_MODEL: probs[NEW_MODEL],
                        "shift": probs["c_bpa_shift"]})
        bt.update_lineup_history(day, lu_ctx, lu_history)
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame) -> dict:
    """Paired Brier difference (challenger − base) on one cell's rows."""
    y = df["home_win"].astype(float).to_numpy()
    d = (df[NEW_MODEL].to_numpy() - y) ** 2 - (df[BASE_MODEL].to_numpy() - y) ** 2
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return {"n": len(d),
            "brier": float(np.mean((df[NEW_MODEL].to_numpy() - y) ** 2)),
            "paired": float(d.mean()), "se": float(se),
            "t": float(d.mean() / se) if se else np.nan,
            "shift": float(df["shift"].mean() / 2.0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--min-games", type=int, default=20)
    parser.add_argument("--hard-1d", default="30,40,50")
    parser.add_argument("--hard-2d", default="45,65,85")
    parser.add_argument("--taper", default="50,75,100,150")
    parser.add_argument("--baselines", default="league,team")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the per-game sweep frame here (parquet)")
    args = parser.parse_args()

    h1s = [float(x) for x in args.hard_1d.split(",")]
    h2s = [float(x) for x in args.hard_2d.split(",")]
    taps = [float(x) for x in args.taper.split(",")]
    baselines = args.baselines.split(",")
    cells = [(h1, h2, t) for h1 in h1s for h2 in h2s for t in taps if h2 >= h1]
    print(f"{len(cells)} cells x {len(baselines)} baselines on {args.season}")

    teams = fetch_teams(args.season)
    sched = fetch_schedule(f"{args.season}-03-01", f"{args.season}-10-15")
    scored = sched[sched["status"] == "Final"].dropna(subset=["home_score", "away_score"])
    scored = scored[scored["home_score"] != scored["away_score"]].copy()
    scored["home_win"] = scored["home_score"] > scored["away_score"]
    scored = scored[scored["game_type"] == "R"]

    sp_ctx = bt.build_sp_context(args.season, scored, sp_model.BALLAST_BF,
                                 sp_model.STARTER_IP)
    lu_ctx = bt.build_lu_context(
        args.season, scored, lu_model.BALLAST, bt.LU_WEIGHT, bt.LU_BASELINE,
        lu_model.BASELINE_BALLAST_GAMES,
        pa_per_game=sp_ctx["league"]["bf_per_ip"] * 9.0)
    bp_ctx = bt.build_bp_context(
        args.season, sp_model.BALLAST_BF, bt.BP_BASELINE,
        bt.bp_model.ROSTER_WINDOW_DAYS, bt.bp_model.REST_DAYS,
        bt.bp_model.REST_MIN_DAYS, bt.bp_model.RELIEF_IP,
        sp_ctx["league"], sp_ctx["prior_counts"])
    c_ctx = bt.build_c_context(
        args.season, lu_ctx, bp_ctx, bt.C_WEIGHT,
        bt.C_SHARE_WINDOW if bt.C_SHARE_WINDOW > 0 else None,
        bt.C_ROTATION_DAYS if bt.C_ROTATION_DAYS > 0 else None,
        bt.rn_model.ROTATION_TOP_N, lu_ctx["pa_per_game"])

    df = sweep(scored, teams["team_id"].to_numpy(), args.min_games,
               sp_ctx, lu_ctx, bp_ctx, c_ctx, cells, baselines)
    if df.empty:
        print("no games scored")
        return

    first = df[(df["hard_1d"] == cells[0][0]) & (df["hard_2d"] == cells[0][1])
               & (df["taper"] == cells[0][2]) & (df["baseline"] == baselines[0])]
    y = first["home_win"].astype(float).to_numpy()
    print(f"\n{len(first)} games, {BASE_MODEL} Brier "
          f"{np.mean((first[BASE_MODEL].to_numpy() - y) ** 2):.5f}\n")

    keys = ["hard_1d", "hard_2d", "taper", "baseline"]
    out = [{**dict(zip(keys, k)), **paired(grp)} for k, grp in df.groupby(keys)]
    table = pd.DataFrame(out).sort_values("paired").reset_index(drop=True)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    best = table.iloc[0]
    print(f"\nbest on {args.season}: hard_1d={best['hard_1d']:.0f}, "
          f"hard_2d={best['hard_2d']:.0f}, taper={best['taper']:.0f}, "
          f"baseline={best['baseline']} "
          f"({best['paired']:+.5f}, se {best['se']:.5f}, n={int(best['n'])})")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"wrote {len(df)} rows → {args.out}")


if __name__ == "__main__":
    main()
