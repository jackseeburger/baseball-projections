"""Day-to-day backtest of per-game win probabilities (walk-forward).

For every completed game of the season, rebuild each team's strength from
ONLY the games played before that date, predict P(home wins), and score
against the actual result. This is the honest test of the simulator's
per-game engine — the playoff-odds table mostly reflects banked standings,
but per-game probabilities are where a strength model actually earns edge.

Metrics: Brier score and log loss (lower is better). Baselines:
    home_constant   — P(home) = HFA prior, ignores teams entirely
    win_pct_log5    — raw season win% into log5 + HFA (no regression)
    pythag_60       — the production model (regressed Pythagenpat, 60 games)
    pythag_{k}      — ballast sweep to see what the data prefers

Usage:
    python scripts/backtest_game_odds.py --season 2026 --min-games 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.mlb_stats_api import fetch_schedule
from src.sim.season import from_schedule
from src.sim.strength import HFA_PRIOR, home_win_prob, pythagenpat
from src.sim.teams import fetch_teams


def team_totals(games: pd.DataFrame, team_ids) -> pd.DataFrame:
    """Runs scored/allowed, wins/losses per team from a completed-games frame."""
    home = games.groupby("home_id").agg(rs=("home_score", "sum"), ra=("away_score", "sum"),
                                        w=("home_win", "sum"), g=("home_win", "size"))
    away = games.groupby("away_id").agg(rs=("away_score", "sum"), ra=("home_score", "sum"),
                                        w=("home_win", lambda x: (~x).sum()), g=("home_win", "size"))
    tot = home.add(away, fill_value=0).reindex(team_ids).fillna(0)
    return tot


def strengths(tot: pd.DataFrame, regress_games: float) -> pd.Series:
    lg_rs = tot["rs"].sum() / max(tot["g"].sum(), 1)
    lg_ra = tot["ra"].sum() / max(tot["g"].sum(), 1)
    out = {}
    for t, r in tot.iterrows():
        rs_pg = (r["rs"] + regress_games * lg_rs) / (r["g"] + regress_games)
        ra_pg = (r["ra"] + regress_games * lg_ra) / (r["g"] + regress_games)
        out[t] = pythagenpat(rs_pg, ra_pg, 1.0)
    return pd.Series(out)


def walk_forward(completed: pd.DataFrame, team_ids, min_games: int,
                 ballasts: list[float]) -> pd.DataFrame:
    """One row per scored game with each model's P(home)."""
    completed = completed.sort_values("date").reset_index(drop=True)
    rows = []
    for date, day in completed.groupby("date", sort=True):
        before = completed[completed["date"] < date]
        tot = team_totals(before, team_ids) if len(before) else None
        if tot is None or tot["g"].min() < min_games:
            continue
        hfa_obs = (HFA_PRIOR * 2000 + before["home_win"].sum()) / (2000 + len(before))
        wp = (tot["w"] / tot["g"]).clip(0.2, 0.8)
        s_by_k = {k: strengths(tot, k) for k in ballasts}
        for g in day.itertuples(index=False):
            row = {"date": date, "home_id": g.home_id, "away_id": g.away_id,
                   "home_win": bool(g.home_win),
                   "home_constant": hfa_obs,
                   "win_pct_log5": float(home_win_prob(wp[g.home_id], wp[g.away_id], hfa_obs))}
            for k, s in s_by_k.items():
                row[f"pythag_{int(k)}"] = float(home_win_prob(s[g.home_id], s[g.away_id], hfa_obs))
            rows.append(row)
    return pd.DataFrame(rows)


def score(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    y = df["home_win"].astype(float).to_numpy()
    out = []
    for m in models:
        p = np.clip(df[m].to_numpy(), 1e-6, 1 - 1e-6)
        out.append({"model": m,
                    "brier": float(np.mean((p - y) ** 2)),
                    "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                    "mean_p_home": float(p.mean())})
    return pd.DataFrame(out).sort_values("brier").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--min-games", type=int, default=20,
                        help="skip dates until every team has this many games")
    parser.add_argument("--ballasts", default="0,30,60,100,160")
    args = parser.parse_args()
    ballasts = [float(b) for b in args.ballasts.split(",")]

    teams = fetch_teams(args.season)
    sched = fetch_schedule(f"{args.season}-03-01", f"{args.season}-10-15")
    state = from_schedule(sched, teams)
    # from_schedule drops scores; re-attach for run totals
    scored = sched[sched["status"] == "Final"].dropna(subset=["home_score", "away_score"])
    scored = scored[scored["home_score"] != scored["away_score"]].copy()
    scored["home_win"] = scored["home_score"] > scored["away_score"]
    scored = scored[scored["game_type"] == "R"]

    preds = walk_forward(scored, teams["team_id"].to_numpy(), args.min_games, ballasts)
    models = ["home_constant", "win_pct_log5"] + [f"pythag_{int(k)}" for k in ballasts]
    print(f"{len(preds)} games scored (from the date every team had {args.min_games}+ games)\n")
    print(score(preds, models).round(4).to_string(index=False))
    # Calibration of the production model
    prod = preds[["home_win", "pythag_60"]].copy()
    prod["bucket"] = pd.cut(prod["pythag_60"], [0, .4, .45, .5, .55, .6, .65, 1.0])
    cal = prod.groupby("bucket", observed=True).agg(n=("home_win", "size"),
                                                   predicted=("pythag_60", "mean"),
                                                   realized=("home_win", "mean"))
    print("\nCalibration (pythag_60):")
    print(cal.round(3).to_string())


if __name__ == "__main__":
    main()
