"""The money exam: what our per-game probabilities would have made, after costs.

Joins the walk-forward per-game predictions to the exchanges' reconstructed
pre-pitch closes and simulates taking the other side of the book whenever the
model disagrees with the price by more than a threshold. Brier says whether a
probability is good; this says what it was worth after the spread and the fee
(docs/architecture.md §0: *money is the exam*).

`--maker` runs the other exam: instead of crossing the book, rest a limit
order at the model's price minus a margin from T−24h to first pitch and see
whether the market came to it. That one needs the hourly candle archive, and
it charges Kalshi's maker fee rather than the taker's.

Inputs:
    --preds   scripts/backtest_game_odds.py --out  (one row per game, one
              probability column per model)
    --closes  scripts/backfill_market_closes.py    (one row per venue+game:
              p_home_close, bid, ask, home_won)
    --candles scripts/backfill_kalshi_candles.py   (maker mode: one row per
              market-hour of the 24 before first pitch)

Usage:
    python scripts/money_exam.py                      # both venues, defaults
    python scripts/money_exam.py --markdown           # the docs tables
    python scripts/money_exam.py --detail-threshold 0.02 --half-spread 0.02
    python scripts/money_exam.py --maker --markdown   # the maker tables
    python scripts/money_exam.py --maker --maker-anchor post   # no hindsight at all

The taker fill assumption — one unit at the closing quote, top of book, no
impact — is optimistic and is stated wherever the numbers are; so is the maker
one, which assumes queue priority at our price (src/market/pnl.py).
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
DEFAULT_MAKER_MODELS = [
    "pythag_60_sp_lu_bp", "pythag_60_sp_lu", "pythag_60_sp", "pythag_60",
    "home_constant",
]
LABELS = {"market": "market (control)", "random_edge": "random_edge (control)",
          "home_constant": "home_constant (control)"}
STAKING_LABEL = {"flat": "flat 1u", "kelly": "quarter-Kelly (cap 5%)"}
MAKER_STAKING_LABEL = {"flat": "flat, one contract per game",
                       "kelly": "quarter-Kelly (cap 5%)"}
HALF_LABEL = {"first": "first half — where the margin is chosen",
              "second": "second half — where it is scored",
              "all": "the whole window"}


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


# ───────────────────────────── maker mode ─────────────────────────────

def maker_label(model: str) -> str:
    """`pythag_60__shuffled` → `pythag_60 (shuffled control)`."""
    if model.endswith("__shuffled"):
        return f"{model[:-len('__shuffled')]} (shuffled control)"
    return LABELS.get(model, model)


def maker_row_order(models: list[str], present: set[str]) -> list[str]:
    """Each model followed by its own control, so the two read side by side."""
    out = []
    for m in models:
        for name in (m, f"{m}__shuffled"):
            if name in present:
                out.append(name)
    return out


def cents(x, digits: int = 2) -> str:
    return "—" if pd.isna(x) else f"{100 * x:+.{digits}f}¢"


def fill_table(res: pd.DataFrame, half: str, staking: str, models: list[str],
               margins) -> str:
    """Rows are models, columns are margins: what fraction of orders filled."""
    g = res[(res["half"] == half) & (res["staking"] == staking)]
    header = ["Model"] + [f"m={m:.2f}" for m in margins]
    rows = []
    for m in maker_row_order(models, set(g["model"])):
        sub = g[g["model"] == m].set_index("margin")
        rows.append([maker_label(m)] + [
            f"{sub.loc[x, 'fill_rate']:.3f}" if x in sub.index else "—"
            for x in margins])
    return md_table(header, rows)


def maker_table(res: pd.DataFrame, half: str, staking: str, models: list[str],
                margins) -> str:
    """One row per (model, margin): what a posted contract was worth."""
    g = res[(res["half"] == half) & (res["staking"] == staking)]
    header = ["Model", "m", "posted", "fill", "crossed", "¢/posted", "¢/filled",
              "¢/game", "ROI", "ROI 95% CI"]
    rows = []
    for model in maker_row_order(models, set(g["model"])):
        sub = g[g["model"] == model].set_index("margin")
        for x in margins:
            if x not in sub.index:
                continue
            r = sub.loc[x]
            rows.append([
                maker_label(model), f"{x:.2f}", f"{int(r['n_posted'])}",
                "—" if pd.isna(r["fill_rate"]) else f"{r['fill_rate']:.3f}",
                "—" if pd.isna(r["marketable_rate"]) else f"{r['marketable_rate']:.3f}",
                cents(r["pnl_per_posted"]), cents(r["pnl_per_filled"]),
                cents(r["pnl_per_game"]),
                pct(r["roi"]) if r["n_filled"] else "—",
                f"({pct(r['roi_lo'])}, {pct(r['roi_hi'])})" if r["n_filled"] else "—",
            ])
    return md_table(header, rows)


def chosen_table(res: pd.DataFrame, staking: str, models: list[str]) -> str:
    """The margin picked on the first half, scored on the second."""
    header = ["Model", "chosen m", "¢/posted (train)", "posted", "fill",
              "¢/posted (test)", "¢/filled (test)", "ROI (test)", "ROI 95% CI",
              "hit"]
    rows = []
    present = set(res["model"])
    for model in maker_row_order(models, present):
        m = pnl.choose_margin(res, model, staking, half="first")
        if pd.isna(m):
            continue
        tr = res[(res["model"] == model) & (res["staking"] == staking)
                 & (res["half"] == "first") & (res["margin"] == m)]
        te = res[(res["model"] == model) & (res["staking"] == staking)
                 & (res["half"] == "second") & (res["margin"] == m)]
        if tr.empty or te.empty:
            continue
        t, r = tr.iloc[0], te.iloc[0]
        rows.append([
            maker_label(model), f"{m:.2f}", cents(t["pnl_per_posted"]),
            f"{int(r['n_posted'])}",
            "—" if pd.isna(r["fill_rate"]) else f"{r['fill_rate']:.3f}",
            cents(r["pnl_per_posted"]), cents(r["pnl_per_filled"]),
            pct(r["roi"]) if r["n_filled"] else "—",
            f"({pct(r['roi_lo'])}, {pct(r['roi_hi'])})" if r["n_filled"] else "—",
            "—" if pd.isna(r["hit_rate"]) else f"{r['hit_rate']:.3f}",
        ])
    return md_table(header, rows)


def run_maker(args) -> None:
    """The maker exam: quote, wait, get filled or cancel at first pitch."""
    preds = pd.read_parquet(args.preds)
    closes = pd.read_parquet(args.closes)
    candles = pd.read_parquet(args.candles)
    margins = [float(m) for m in args.margins.split(",")]
    venue = replace(pnl.VENUES["kalshi"], maker_rate=args.kalshi_maker_fee_rate)

    res = pnl.run_maker_exam(
        preds, closes, candles, args.models, margins, venue,
        stakings=("flat", "kelly"), fraction=args.kelly_fraction, cap=args.kelly_cap,
        hours=args.maker_hours, anchor=args.maker_anchor,
        maker_round_cents=args.maker_round_cents, draws=args.bootstrap,
        seed=args.seed)

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.csv_out, index=False)

    n_games = int(res[res["half"] == "all"]["n_games"].iloc[0])
    n_markets = candles["market_id"].nunique()
    windows = {h: (res[res["half"] == h]["first_date"].iloc[0],
                   res[res["half"] == h]["last_date"].iloc[0])
               for h in ("first", "second") if (res["half"] == h).any()}

    if not args.markdown:
        cols = ["half", "model", "margin", "staking", "n_games", "n_posted",
                "n_filled", "fill_rate", "marketable_rate", "pnl_per_posted",
                "pnl_per_filled", "pnl_per_game", "roi", "roi_lo", "roi_hi",
                "hit_rate", "total_return", "total_fees"]
        print(res[cols].round(5).to_string(index=False))
        if args.csv_out is not None:
            print(f"\nwrote {args.csv_out}")
        return

    print(f"\n### Kalshi maker — {n_games} games, {n_markets} markets' candles, "
          f"maker fee rate {venue.maker_rate:g}, order live T−{args.maker_hours}h "
          f"→ first pitch\n")
    for h, (a, b) in windows.items():
        print(f"* **{h} half** ({HALF_LABEL[h]}): {a} → {b}, "
              f"{int(res[res['half'] == h]['n_games'].iloc[0])} games")
    print()
    for staking in ("flat", "kelly"):
        print(f"**{MAKER_STAKING_LABEL[staking]}**\n")
        print("Fill rate by margin:\n")
        print(fill_table(res, "second", staking, args.models, margins))
        print("\nSecond half — the scored one:\n")
        print(maker_table(res, "second", staking, args.models, margins))
        print("\nFirst half — where the margin is chosen:\n")
        print(maker_table(res, "first", staking, args.models, margins))
        print("\nMargin chosen on the first half, scored on the second:\n")
        print(chosen_table(res, staking, args.models))
        print()
    if args.csv_out is not None:
        print(f"\nwrote {args.csv_out}")


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
    ap.add_argument("--maker", action="store_true",
                    help="run the maker exam (rest a limit order) instead of "
                         "the taker exam (cross the book)")
    ap.add_argument("--candles", type=Path,
                    default=ROOT / "data/market/kalshi_candles_2026.parquet",
                    help="maker: scripts/backfill_kalshi_candles.py parquet")
    ap.add_argument("--margins", default="0,0.01,0.02,0.03,0.05",
                    help="maker: how far inside fair value to quote")
    ap.add_argument("--kalshi-maker-fee-rate", type=float,
                    default=pnl.KALSHI_MAKER_RATE,
                    help="maker: 0.0175·P·(1−P) per contract, a quarter of the "
                         "taker rate (second-hand; see src/market/pnl.py)")
    ap.add_argument("--maker-round-cents", action="store_true",
                    help="maker: round the maker fee up to the whole cent, the "
                         "less charitable reading of the rounding rule")
    ap.add_argument("--maker-hours", type=int, default=pnl.MAKER_HOURS,
                    help="maker: how long before first pitch the order rests")
    ap.add_argument("--maker-anchor", choices=("close", "post"), default="close",
                    help="maker: what the model is compared against to pick a "
                         "side — the pre-pitch close (comparable to the taker "
                         "exam) or the price on the screen when the order goes "
                         "in (no hindsight at all)")
    args = ap.parse_args()

    if args.maker:
        if args.models == DEFAULT_MODELS:
            # `market` never disagrees with itself and `random_edge` has its own
            # per-model replacement here, so the maker grid uses its own default.
            args.models = DEFAULT_MAKER_MODELS
        run_maker(args)
        return

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
