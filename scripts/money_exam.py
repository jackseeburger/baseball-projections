"""The money exam: what our per-game probabilities would have made, after costs.

Joins the walk-forward per-game predictions to the exchanges' reconstructed
pre-pitch closes and simulates taking the other side of the book whenever the
model disagrees with the price by more than a threshold. Brier says whether a
probability is good; this says what it was worth after the spread and the fee
(docs/architecture.md §0: *money is the exam*).

Inputs:
    --preds   scripts/backtest_game_odds.py --out  (one row per game, one
              probability column per model)
    --closes  scripts/backfill_market_closes.py    (one row per venue+game:
              p_home_close, bid, ask, home_won)

Usage:
    python scripts/money_exam.py                      # both venues, defaults
    python scripts/money_exam.py --markdown           # the docs tables
    python scripts/money_exam.py --detail-threshold 0.02 --half-spread 0.02

The fill assumption — one unit at the closing quote, top of book, no impact —
is optimistic and is stated wherever the numbers are (src/market/pnl.py).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.market import pnl

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = [
    "pythag_60_sp_lu_bp", "pythag_60_sp_lu", "pythag_60_sp", "pythag_60",
    "home_constant", "random_edge", "market",
]
LABELS = {"market": "market (control)", "random_edge": "random_edge (control)",
          "home_constant": "home_constant (control)"}
STAKING_LABEL = {"flat": "flat 1u", "kelly": "quarter-Kelly (cap 5%)"}


def pct(x, digits: int = 1) -> str:
    return "—" if pd.isna(x) else f"{100 * x:+.{digits}f}%"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |",
                      "|" + "---|" * len(header)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def headline_table(res: pd.DataFrame, venue: str, staking: str,
                   models: list[str], thresholds) -> str:
    """Rows are models, columns are edge thresholds: n bets and ROI at each."""
    g = res[(res["venue"] == venue) & (res["staking"] == staking)]
    header = ["Model"] + [f"n ≥{100 * t:.0f}pt" for t in thresholds] \
                       + [f"ROI ≥{100 * t:.0f}pt" for t in thresholds]
    rows = []
    for m in models:
        sub = g[g["model"] == m].set_index("threshold")
        if sub.empty:
            continue
        cells = [f"{int(sub.loc[t, 'n_bets'])}" if t in sub.index else "—"
                 for t in thresholds]
        cells += [pct(sub.loc[t, "roi"]) if t in sub.index else "—"
                  for t in thresholds]
        rows.append([LABELS.get(m, m)] + cells)
    return md_table(header, rows)


def detail_table(res: pd.DataFrame, venue: str, staking: str, models: list[str],
                 threshold: float) -> str:
    """Everything for one threshold: hit rate, ROI with a CI, edge, CLV, drawdown."""
    g = res[(res["venue"] == venue) & (res["staking"] == staking)
            & (res["threshold"] == threshold)].set_index("model")
    header = ["Model", "n bets", "hit", "staked", "return", "ROI",
              "ROI 95% CI", "mean edge", "CLV", "max DD", "fees"]
    rows = []
    for m in models:
        if m not in g.index:
            continue
        r = g.loc[m]
        rows.append([
            LABELS.get(m, m), f"{int(r['n_bets'])}",
            "—" if pd.isna(r["hit_rate"]) else f"{r['hit_rate']:.3f}",
            f"{r['total_staked']:.1f}u", f"{r['total_return']:+.2f}u",
            pct(r["roi"]),
            f"({pct(r['roi_lo'])}, {pct(r['roi_hi'])})" if r["n_bets"] else "—",
            "—" if pd.isna(r["mean_edge"]) else f"{100 * r['mean_edge']:.2f} pt",
            "—" if pd.isna(r["clv"]) else f"{100 * r['clv']:+.2f} pt",
            f"{r['max_drawdown']:.2f}u", f"{r['total_fees']:.2f}u",
        ])
    return md_table(header, rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path,
                    default=ROOT / "data/parquet/game_preds_2026.parquet",
                    help="backtest_game_odds.py --out parquet")
    ap.add_argument("--closes", type=Path,
                    default=ROOT / "data/parquet/market_closes_2026.parquet")
    ap.add_argument("--venues", nargs="+", default=["kalshi", "polymarket"])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--thresholds", default="0,0.02,0.04,0.06",
                    help="edge thresholds in probability units")
    ap.add_argument("--detail-threshold", type=float, default=0.02)
    ap.add_argument("--half-spread", type=float, default=pnl.DEFAULT_HALF_SPREAD,
                    help="Polymarket: half-spread assumed around the mid close")
    ap.add_argument("--kalshi-fee-rate", type=float, default=pnl.KALSHI_TAKER_RATE)
    ap.add_argument("--polymarket-fee-rate", type=float, default=pnl.POLYMARKET_TAKER_RATE,
                    help="0 by default; sports taker fees were reported in 2026")
    ap.add_argument("--kelly-fraction", type=float, default=pnl.DEFAULT_KELLY_FRACTION)
    ap.add_argument("--kelly-cap", type=float, default=pnl.DEFAULT_KELLY_CAP)
    ap.add_argument("--bootstrap", type=int, default=pnl.BOOTSTRAP_DRAWS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--markdown", action="store_true",
                    help="print the docs tables instead of the wide frame")
    ap.add_argument("--csv-out", type=Path, default=None)
    args = ap.parse_args()

    preds = pd.read_parquet(args.preds)
    closes = pd.read_parquet(args.closes)
    thresholds = [float(t) for t in args.thresholds.split(",")]
    venues = []
    for name in args.venues:
        v = pnl.VENUES[name]
        rate = args.kalshi_fee_rate if name == "kalshi" else args.polymarket_fee_rate
        v = replace(v, taker_rate=rate)
        if v.half_spread is not None:
            v = replace(v, half_spread=args.half_spread)
        venues.append(v)

    res = pnl.run_exam(preds, closes, args.models, venues, thresholds,
                       fraction=args.kelly_fraction, cap=args.kelly_cap,
                       draws=args.bootstrap, seed=args.seed)
    models = [m for m in args.models if m in set(res["model"])]

    if args.markdown:
        for v in venues:
            n_games = int(res[res["venue"] == v.name]["n_games"].iloc[0])
            book = ("quoted bid/ask" if v.half_spread is None
                    else f"mid ± {100 * v.half_spread:.1f}¢")
            print(f"\n### {v.name} — {n_games} games, {book}, "
                  f"taker fee rate {v.taker_rate:g}\n")
            for staking in ("flat", "kelly"):
                print(f"**{STAKING_LABEL[staking]}**\n")
                print(headline_table(res, v.name, staking, models, thresholds))
                print(f"\nAt edge ≥ {100 * args.detail_threshold:.0f} pts:\n")
                print(detail_table(res, v.name, staking, models,
                                   args.detail_threshold))
                print()
    else:
        cols = ["venue", "model", "threshold", "staking", "n_games", "n_bets",
                "hit_rate", "total_staked", "total_return", "roi", "roi_lo",
                "roi_hi", "mean_edge", "clv", "max_drawdown", "total_fees"]
        for v in venues:
            g = res[res["venue"] == v.name]
            print(f"\n=== {v.name}: {int(g['n_games'].iloc[0])} games priced, "
                  f"{'quoted book' if v.half_spread is None else f'mid ± {v.half_spread}'}"
                  f", fee rate {v.taker_rate:g} ===")
            print(g[cols].drop(columns="venue").round(4).to_string(index=False))
        print("\nFill assumption: one unit at the closing quote, top of book, no "
              "impact — optimistic on purpose (src/market/pnl.py).")

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.csv_out, index=False)
        print(f"\nwrote {args.csv_out}")


if __name__ == "__main__":
    main()
