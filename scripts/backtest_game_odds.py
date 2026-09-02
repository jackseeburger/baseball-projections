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
    pythag_60_sp    — pythag_60 with each side's runs allowed re-weighted
                      toward its announced starting pitcher (src/sim/starters)

Usage:
    python scripts/backtest_game_odds.py --season 2026 --min-games 20
    python scripts/backtest_game_odds.py --season 2026 --min-games 20 \
        --market data/parquet/market_closes_2026.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.mlb_stats_api import fetch_probables, fetch_schedule
from src.sim import starters as sp_model
from src.sim.season import from_schedule
from src.sim.strength import HFA_PRIOR, home_win_prob, pythagenpat
from src.sim.teams import fetch_teams

SP_MODEL = "pythag_60_sp"
SP_BALLAST_GAMES = 60.0   # the production team-strength ballast this model extends


def team_totals(games: pd.DataFrame, team_ids) -> pd.DataFrame:
    """Runs scored/allowed, wins/losses per team from a completed-games frame."""
    home = games.groupby("home_id").agg(rs=("home_score", "sum"), ra=("away_score", "sum"),
                                        w=("home_win", "sum"), g=("home_win", "size"))
    away = games.groupby("away_id").agg(rs=("away_score", "sum"), ra=("home_score", "sum"),
                                        w=("home_win", lambda x: (~x).sum()), g=("home_win", "size"))
    tot = home.add(away, fill_value=0).reindex(team_ids).fillna(0)
    return tot


def team_rates(tot: pd.DataFrame, regress_games: float) -> pd.DataFrame:
    """Runs scored/allowed per game, regressed toward league average."""
    lg_rs = tot["rs"].sum() / max(tot["g"].sum(), 1)
    lg_ra = tot["ra"].sum() / max(tot["g"].sum(), 1)
    return pd.DataFrame({
        "rs_pg": (tot["rs"] + regress_games * lg_rs) / (tot["g"] + regress_games),
        "ra_pg": (tot["ra"] + regress_games * lg_ra) / (tot["g"] + regress_games),
    })


def strengths(tot: pd.DataFrame, regress_games: float) -> pd.Series:
    rates = team_rates(tot, regress_games)
    return pd.Series({t: pythagenpat(r["rs_pg"], r["ra_pg"], 1.0)
                      for t, r in rates.iterrows()})


def walk_forward(completed: pd.DataFrame, team_ids, min_games: int,
                 ballasts: list[float], sp_ctx: dict | None = None) -> pd.DataFrame:
    """One row per scored game with each model's P(home).

    Every quantity used on a date is rebuilt from games and pitcher
    appearances strictly before that date.
    """
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
        sp_day = starter_day_context(tot, date, sp_ctx) if sp_ctx else None
        for g in day.itertuples(index=False):
            row = {"date": date, "game_pk": int(g.game_pk),
                   "home_id": g.home_id, "away_id": g.away_id,
                   "home_win": bool(g.home_win),
                   "home_constant": hfa_obs,
                   "win_pct_log5": float(home_win_prob(wp[g.home_id], wp[g.away_id], hfa_obs))}
            for k, s in s_by_k.items():
                row[f"pythag_{int(k)}"] = float(home_win_prob(s[g.home_id], s[g.away_id], hfa_obs))
            if sp_day is not None:
                p_sp, flags = starter_game_prob(g, sp_day, hfa_obs)
                row[SP_MODEL] = p_sp if p_sp is not None else row[f"pythag_{int(SP_BALLAST_GAMES)}"]
                row.update(flags)
            rows.append(row)
    return pd.DataFrame(rows)


# ─── station E starting-pitcher term ───

def starter_day_context(tot: pd.DataFrame, date: str, sp_ctx: dict) -> dict:
    """Everything the starter model needs for one slate, built from the past only.

    `starters.rate_table` does the pitcher half (rates from appearances
    strictly before `date`); the team half is the same regressed run rates
    `pythag_60` uses. The live nightly job calls the same two functions for a
    single date — see `scripts/run_playoff_odds.py`.
    """
    # The current run environment enters through lg_ra9, which anchors the FIP
    # constant to season-to-date league runs per game.
    lg_ra9 = float(tot["ra"].sum() / max(tot["g"].sum(), 1))
    return {
        "sp_ra9": sp_model.rate_table(sp_ctx, date, lg_ra9,
                                      ballast=sp_ctx["ballast"]),
        "lg_ra9": lg_ra9,
        "team": team_rates(tot, SP_BALLAST_GAMES),
        "probables": sp_ctx["probables"],
        "starter_ip": sp_ctx["starter_ip"],
    }


def starter_game_prob(g, day: dict, hfa: float):
    """P(home) with each side's runs allowed blended toward its starter.

    Returns (probability or None when a starter is unknown, diagnostic flags).
    """
    sp_ids = day["probables"].get(int(g.game_pk))
    flags = {"sp_fallback": sp_ids is None, "sp_no_history": 0}
    if sp_ids is None:
        return None, flags
    p_home, no_history = sp_model.game_home_prob(
        day["team"], g.home_id, g.away_id, sp_ids, day["sp_ra9"],
        day["lg_ra9"], hfa, starter_ip=day["starter_ip"])
    flags["sp_no_history"] = no_history
    return p_home, flags


def build_sp_context(season: int, scored: pd.DataFrame, ballast: float,
                     starter_ip: float, prior_seasons: int = 2) -> dict:
    """Fetch probables + pitcher counts once for the whole backtest.

    `prior_seasons` completed seasons plus the current one are Marcel-weighted,
    matching the 5/4/3 weights in `src.sim.starters`.
    """
    probables = fetch_probables(f"{season}-03-01", f"{season}-11-15")
    probables = probables.dropna(subset=["home_sp_id", "away_sp_id"])
    probables = probables[probables["game_pk"].isin(scored["game_pk"])]
    pmap = {int(r.game_pk): (int(r.home_sp_id), int(r.away_sp_id))
            for r in probables.itertuples(index=False)}

    inputs = sp_model.rate_inputs(
        season, {p for ids in pmap.values() for p in ids},
        prior_seasons=prior_seasons)
    return {**inputs, "probables": pmap,
            "ballast": ballast, "starter_ip": starter_ip}


def join_market(preds: pd.DataFrame, closes: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add one column per venue (P(home) at the close) and keep only games
    every venue priced, so all models are scored on the same population."""
    wide = closes.pivot_table(index="game_pk", columns="venue", values="p_home_close")
    wide.columns = [f"{v}_close" for v in wide.columns]
    joined = preds.merge(wide, left_on="game_pk", right_index=True, how="inner")
    market_models = list(wide.columns)
    joined = joined.dropna(subset=market_models)
    return joined, market_models


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
    parser.add_argument("--no-starters", action="store_true",
                        help="skip the pythag_60_sp starting-pitcher model")
    parser.add_argument("--sp-ballast", type=float, default=None,
                        help="override the per-component regression ballasts in "
                             "src/sim/starters with one batters-faced number "
                             "(the defaults are stabilization points from the "
                             "literature; nothing here is fit to this season)")
    parser.add_argument("--starter-ip", type=float, default=sp_model.STARTER_IP,
                        help="innings the starter is assumed to cover")
    parser.add_argument("--market", type=Path, default=None,
                        help="market_closes parquet from scripts/backfill_market_closes.py; "
                             "scores each venue's pre-game close as a model on the common games")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="also write the printed score table to PATH as JSON "
                             "(the market-joined table when --market is given)")
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

    sp_ctx = None
    if not args.no_starters:
        sp_ctx = build_sp_context(
            args.season, scored,
            sp_model.BALLAST_BF if args.sp_ballast is None else args.sp_ballast,
            args.starter_ip)

    preds = walk_forward(scored, teams["team_id"].to_numpy(), args.min_games,
                         ballasts, sp_ctx)
    models = ["home_constant", "win_pct_log5"] + [f"pythag_{int(k)}" for k in ballasts]
    if sp_ctx is not None:
        models.append(SP_MODEL)
    print(f"{len(preds)} games scored (from the date every team had {args.min_games}+ games)\n")
    print(score(preds, models).round(4).to_string(index=False))
    if sp_ctx is not None:
        print(f"\n{SP_MODEL}: {int(preds['sp_fallback'].sum())} of {len(preds)} games fell back "
              f"to pythag_{int(SP_BALLAST_GAMES)} for a missing starter; "
              f"{int(preds['sp_no_history'].sum())} starter slots had no history "
              f"(scored at league average).")

    if args.market is not None:
        preds, market_models = join_market(preds, pd.read_parquet(args.market))
        print(f"\n{len(preds)} games also priced by every venue in {args.market.name} — "
              f"the market is the bar (docs/architecture.md §0):\n")
        print(score(preds, models + market_models).round(4).to_string(index=False))
        if sp_ctx is not None:
            print(f"\n  of these, {int(preds['sp_fallback'].sum())} fell back to "
                  f"pythag_{int(SP_BALLAST_GAMES)} for a missing starter.")
    # Calibration of the production model, and of the challenger
    for model in ["pythag_60"] + ([SP_MODEL] if sp_ctx is not None else []):
        buckets = pd.cut(preds[model], [0, .4, .45, .5, .55, .6, .65, 1.0])
        cal = preds.groupby(buckets, observed=True).agg(n=("home_win", "size"),
                                                        predicted=(model, "mean"),
                                                        realized=("home_win", "mean"))
        print(f"\nCalibration ({model}):")
        print(cal.round(3).to_string())

    if args.json_out is not None:
        import json
        from datetime import datetime, timezone
        # `preds` is the market-joined common-game set when --market was given,
        # otherwise the full walk-forward set; either way the table below is the
        # one printed last, scored on exactly the games it names.
        venues = list(market_models) if args.market is not None else []
        table = score(preds, models + venues)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "season": args.season,
            "min_games": args.min_games,
            "n_games": int(len(preds)),
            "first_date": str(preds["date"].min()),
            "last_date": str(preds["date"].max()),
            "market_file": args.market.name if args.market is not None else None,
            "market_models": venues,
            "realized_home_win_rate": float(preds["home_win"].mean()),
            "scores": json.loads(table.to_json(orient="records")),
            "sp_fallback_games": int(preds["sp_fallback"].sum()) if sp_ctx is not None else None,
            "sp_no_history_slots": int(preds["sp_no_history"].sum()) if sp_ctx is not None else None,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
