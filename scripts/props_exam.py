"""Score archived player props and run the money exam on them.

The moneyline exam (`scripts/money_exam.py`, docs/money-exam-2026.md) ended on
"every station-E model loses money at every threshold on both venues" and named
the way out — a less efficient contract. This is that test: the same
fill-aware P&L, the same fee, the same controls, on Kalshi's player props
priced by the Marcel-with-partial component rates.

Inputs:
    data/parquet/prop_closes_2026.parquet    scripts/backfill_prop_closes.py
    posted lineups + batter / pitcher game logs (cached under data/cache)

Usage:
    python scripts/props_exam.py --markdown
    python scripts/props_exam.py --stats hits hr --thresholds 0.02 0.04
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.mlb_stats_api import fetch_lineups
from src.market import pnl, props

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("props_exam")

MODELS = ["model", "league", "market", "random_edge"]
LABELS = {"model": "marcel_partial", "league": "league_rate (control)",
          "market": "market (control)", "random_edge": "random_edge (control)"}


def build(closes: pd.DataFrame, season: int, stats: tuple,
          pitcher_bf: str = "fixed") -> pd.DataFrame:
    """Price every archived close we can, walk-forward."""
    wanted = closes[closes["prop_stat"].isin(stats) & closes["player_id"].notna()]
    batter_ids = sorted({int(p) for p in
                         wanted.loc[wanted["prop_stat"] != "k", "player_id"]})
    pitcher_ids = sorted({int(p) for p in
                          wanted.loc[wanted["prop_stat"] == "k", "player_id"]})
    logger.info("%d rows to price: %d batters, %d pitchers",
                len(wanted), len(batter_ids), len(pitcher_ids))
    slots = props.lineup_slots(fetch_lineups(sorted(wanted["game_pk"].unique())))
    batter_ctx = props.batter_inputs(season, batter_ids)
    pitcher_ctx = props.pitcher_inputs(season, pitcher_ids)
    return props.price(closes, batter_ctx, pitcher_ctx, slots, stats=stats,
                       pitcher_bf=pitcher_bf)


def venue_for(fee_rate: float, frictionless: bool) -> pnl.Venue:
    """Kalshi as quoted, or a variant for the sensitivity table.

    `fee_rate=0` is a maker fill (resting inside the spread rather than
    crossing it); `frictionless` collapses the book onto the close, which
    prices the model with no spread and no fee at all — the ceiling.
    """
    return pnl.Venue("kalshi", fee_rate, 0.0 if frictionless else None,
                     round_cents=not frictionless)


def money_table(df: pd.DataFrame, thresholds, stakings, draws: int,
                seed: int, venue: pnl.Venue = pnl.KALSHI) -> pd.DataFrame:
    """`pnl.run_exam`'s grid, on one venue, clustered by game."""
    frame = pnl.add_controls(df, seed=seed)
    rows = []
    for model in MODELS:
        if model not in frame.columns:
            continue
        for threshold in thresholds:
            for staking in stakings:
                rows.append(pnl.evaluate(frame, model, venue, threshold,
                                         staking, draws=draws, seed=seed,
                                         group_col="game_pk"))
    return pd.DataFrame(rows)


def per_stat_money(df: pd.DataFrame, threshold: float, draws: int,
                   seed: int, venue: pnl.Venue = pnl.KALSHI) -> pd.DataFrame:
    """The headline threshold, flat stakes, one row per prop stat."""
    rows = []
    for stat, grp in df.groupby("prop_stat"):
        frame = pnl.add_controls(grp.reset_index(drop=True), seed=seed)
        for model in MODELS:
            r = pnl.evaluate(frame, model, venue, threshold, "flat",
                             draws=draws, seed=seed, group_col="game_pk")
            rows.append({"prop_stat": stat, **r})
    return pd.DataFrame(rows)


def pct(x) -> str:
    return "—" if pd.isna(x) else f"{x:+.1%}"


def markdown(brier: pd.DataFrame, grid: pd.DataFrame, per_stat: pd.DataFrame,
             threshold: float) -> str:
    out = ["### Brier per stat (lower is better)", "",
           "| Stat | n | games | players | over rate | ours | market | league-rate |",
           "|---|---|---|---|---|---|---|---|"]
    for r in brier.itertuples(index=False):
        out.append(f"| {r.prop_stat} | {r.n} | {r.games} | {r.players} | "
                   f"{r.over_rate:.3f} | {r.p_model:.5f} | {r.p_market:.5f} | "
                   f"{r.p_league:.5f} |")
    out += ["", f"### Money, Kalshi, flat 1u, edge ≥ {threshold:.0%}", "",
            "| Model | n bets | hit | staked | return | ROI | ROI 95% CI | "
            "mean edge | CLV | max DD | fees |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    flat = grid[(grid["staking"] == "flat") & (grid["threshold"] == threshold)]
    for r in flat.itertuples(index=False):
        hit = "—" if pd.isna(r.hit_rate) else f"{r.hit_rate:.3f}"
        edge = "—" if pd.isna(r.mean_edge) else f"{100 * r.mean_edge:.2f} pt"
        clv = "—" if pd.isna(r.clv) else f"{100 * r.clv:+.2f} pt"
        out.append(f"| {LABELS.get(r.model, r.model)} | {r.n_bets} | {hit} | "
                   f"{r.total_staked:.1f}u | {r.total_return:+.2f}u | {pct(r.roi)} | "
                   f"({pct(r.roi_lo)}, {pct(r.roi_hi)}) | {edge} | {clv} | "
                   f"{r.max_drawdown:.2f}u | {r.total_fees:.2f}u |")
    out += ["", "### ROI by threshold (flat 1u)", "",
            "| Model | " + " | ".join(f"n ≥{int(100 * t)}pt" for t in sorted(grid['threshold'].unique()))
            + " | " + " | ".join(f"ROI ≥{int(100 * t)}pt" for t in sorted(grid['threshold'].unique())) + " |",
            "|---|" + "---|" * (2 * grid["threshold"].nunique())]
    for model in MODELS:
        sub = grid[(grid["model"] == model) & (grid["staking"] == "flat")].sort_values("threshold")
        if sub.empty:
            continue
        out.append(f"| {LABELS.get(model, model)} | "
                   + " | ".join(str(int(n)) for n in sub["n_bets"]) + " | "
                   + " | ".join(pct(v) for v in sub["roi"]) + " |")
    out += ["", f"### Money per stat, flat 1u, edge ≥ {threshold:.0%}", "",
            "| Stat | Model | n bets | hit | ROI | ROI 95% CI |", "|---|---|---|---|---|---|"]
    for r in per_stat.itertuples(index=False):
        hit = "—" if pd.isna(r.hit_rate) else f"{r.hit_rate:.3f}"
        out.append(f"| {r.prop_stat} | {LABELS.get(r.model, r.model)} | {r.n_bets} | "
                   f"{hit} | {pct(r.roi)} | ({pct(r.roi_lo)}, {pct(r.roi_hi)}) |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--closes", type=Path,
                    default=Path("data/parquet/prop_closes_2026.parquet"))
    ap.add_argument("--stats", nargs="+", default=list(props.PRICEABLE))
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.02, 0.04, 0.06])
    ap.add_argument("--headline", type=float, default=0.02)
    ap.add_argument("--stakings", nargs="+", default=["flat", "kelly"])
    ap.add_argument("--draws", type=int, default=pnl.BOOTSTRAP_DRAWS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pitcher-bf", choices=["fixed", "own"], default="fixed",
                    help="batters faced by a starter: the league average (23) or "
                         "his own per-start rate to date")
    ap.add_argument("--fee-rate", type=float, default=pnl.KALSHI_TAKER_RATE,
                    help="Kalshi taker fee rate; 0 prices a maker fill")
    ap.add_argument("--frictionless", action="store_true",
                    help="fill at the close with no spread — the ceiling")
    ap.add_argument("--min-volume", type=float, default=0.0,
                    help="keep only contracts with at least this much volume traded "
                         "BEFORE first pitch; `volume_total` is a market's whole life "
                         "and an in-play market trades heaviest once the outcome is "
                         "live, so filtering on it leaks the result")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--priced-out", type=Path, default=None,
                    help="write the priced frame to parquet for reuse")
    args = ap.parse_args()

    closes = pd.read_parquet(args.closes)
    if args.min_volume:
        closes = closes[closes["volume_pre"] >= args.min_volume]
    venue = venue_for(args.fee_rate, args.frictionless)
    priced = build(closes, args.season, tuple(args.stats),
                   pitcher_bf=args.pitcher_bf)
    if args.priced_out:
        priced.to_parquet(args.priced_out, index=False)

    brier = props.brier_table(priced)
    frame = props.to_pnl_frame(priced)
    grid = money_table(frame, args.thresholds, args.stakings, args.draws,
                       args.seed, venue)
    per_stat = per_stat_money(frame, args.headline, args.draws, args.seed, venue)

    print("\n== coverage ==")
    print(closes.groupby("prop_stat").agg(archived=("market_id", "nunique")).to_string())
    print(f"\npriced {len(priced)} of {len(closes)} archived closes; "
          f"{frame['game_pk'].nunique()} games, {frame['player_id'].nunique()} players, "
          f"{frame['game_date'].min()} .. {frame['game_date'].max()}")
    print("\n== Brier ==")
    print(brier.to_string(index=False))
    print("\n== money ==")
    print(grid.to_string(index=False))
    if args.markdown:
        print("\n" + markdown(brier, grid, per_stat, args.headline))


if __name__ == "__main__":
    main()
