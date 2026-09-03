"""Score archived player props and run the money exam on them.

The moneyline exam (`scripts/money_exam.py`, docs/money-exam-2026.md) ended on
"every station-E model loses money at every threshold on both venues" and named
the way out — a less efficient contract. This is that test: the same
fill-aware P&L, the same fee, the same controls, on Kalshi's player props
priced by the Marcel-with-partial component rates.

Three arms, all on the same contracts:

    current     station A's rates for the player, and nothing about who he faces
    matchup     the same rates with the opposing pitching folded in by log5
                (`src/market/matchup.py`) — the probable starter over his own
                expected innings, the opposing pen over the rest, and for a
                pitcher's strikeout prop the opposing posted card
    market      the venue's own close, the bar

and two ways to trade them: crossing the closing quote (the taker exam) and
resting a limit order through the hours before first pitch (`--maker`), which
needs the hourly prop candle archive.

Every free constant — the matchup weight, the maker margin — is chosen on the
**first half of the window by date** and scored on the **second**, and both
halves are printed.

Inputs:
    data/market/prop_closes_2026.parquet          scripts/backfill_prop_closes.py
    data/market/kalshi_prop_candles_2026.parquet  the same script (--maker only)
    posted lineups + batter / pitcher game logs (cached under data/cache)

Usage:
    python scripts/props_exam.py --markdown
    python scripts/props_exam.py --matchup on --markdown
    python scripts/props_exam.py --maker --matchup on --markdown
    python scripts/props_exam.py --stats hits hr --thresholds 0.02 0.04
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.mlb_stats_api import fetch_lineups, fetch_probables, fetch_schedule
from src.market import matchup as mu
from src.market import pnl, props

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("props_exam")

MODELS = ["model", "league", "market", "random_edge"]
LABELS = {"model": "marcel_partial", "matchup": "marcel_partial + matchup",
          "league": "league_rate (control)",
          "market": "market (control)", "random_edge": "random_edge (control)",
          "model__shuffled": "marcel_partial (shuffled control)",
          "matchup__shuffled": "matchup (shuffled control)"}
# How far the matchup term is allowed to pull, 1.0 being log5 exactly. Chosen
# on the first half of the window; never on the half it is scored on.
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
# How long before first pitch a club's own recent cards are pooled over when a
# start's opposing card is not in the lineup archive.
CARD_LOOKBACK_DAYS = 21


# ───────────────────────────── pricing the arms ─────────────────────────────

def contexts(closes: pd.DataFrame, season: int, stats: tuple,
             with_matchup: bool, weight: float) -> dict:
    """Everything `props.price` needs, fetched once.

    The matchup context adds four lookups on top: the pre-game **probable**
    starters (never the man who actually started), which club is home and which
    away, the posted card for each side, and every club's recent cards for the
    starts whose opposing card is missing.
    """
    wanted = closes[closes["prop_stat"].isin(stats) & closes["player_id"].notna()]
    batter_ids = sorted({int(p) for p in
                         wanted.loc[wanted["prop_stat"] != "k", "player_id"]})
    pitcher_ids = sorted({int(p) for p in
                          wanted.loc[wanted["prop_stat"] == "k", "player_id"]})
    logger.info("%d rows to price: %d batters, %d pitchers",
                len(wanted), len(batter_ids), len(pitcher_ids))
    lo, hi = str(wanted["game_date"].min()), str(wanted["game_date"].max())

    game_pks = sorted(wanted["game_pk"].unique())
    schedule = fetch_schedule(str(pd.Timestamp(lo).date() -
                                  pd.Timedelta(days=CARD_LOOKBACK_DAYS)), hi)
    schedule = schedule[schedule["game_type"] == "R"]
    if with_matchup:
        # The club's recent cards are a fallback the earliest dates in the
        # window need, so the lineup pull reaches back before the archive.
        game_pks = sorted(set(game_pks) | set(int(p) for p in schedule["game_pk"]))
    lineups = fetch_lineups(game_pks)

    ctx = {"batter_ctx": props.batter_inputs(season, batter_ids),
           "pitcher_ctx": props.pitcher_inputs(season, pitcher_ids),
           "slots": props.lineup_slots(lineups),
           "matchup_ctx": None}
    if not with_matchup:
        return ctx

    probables = fetch_probables(str(pd.Timestamp(lo).date()), hi)
    dates = {int(r.game_pk): str(r.date) for r in schedule.itertuples(index=False)}
    teams = {}
    for r in schedule.itertuples(index=False):
        teams[(int(r.game_pk), "home")] = int(r.home_id)
        teams[(int(r.game_pk), "away")] = int(r.away_id)
    probable_map = {}
    for r in probables.itertuples(index=False):
        if pd.notna(r.home_sp_id):
            probable_map[(int(r.game_pk), "home")] = int(r.home_sp_id)
        if pd.notna(r.away_sp_id):
            probable_map[(int(r.game_pk), "away")] = int(r.away_sp_id)
    cards = props.lineup_cards(lineups)
    club_cards = {}
    for (pk, side), ids in cards.items():
        team, date = teams.get((pk, side)), dates.get(pk)
        if team is not None and date is not None:
            club_cards[(team, date)] = ids
    ctx["matchup_ctx"] = {
        "ctx": mu.inputs(season), "probables": probable_map, "teams": teams,
        "cards": cards, "club_cards": club_cards,
        "sides": props.lineup_sides(lineups), "weight": float(weight),
    }
    return ctx


def price_with(closes: pd.DataFrame, ctx: dict, stats: tuple,
               pitcher_bf: str, weight: float | None = None) -> pd.DataFrame:
    """One priced frame, optionally at a different matchup weight."""
    if weight is not None and ctx["matchup_ctx"] is not None:
        ctx["matchup_ctx"]["weight"] = float(weight)
    return props.price(closes, ctx["batter_ctx"], ctx["pitcher_ctx"],
                       ctx["slots"], stats=stats, pitcher_bf=pitcher_bf,
                       matchup_ctx=ctx["matchup_ctx"])


def choose_weight(closes: pd.DataFrame, ctx: dict, stats: tuple,
                  pitcher_bf: str, grid=WEIGHT_GRID) -> tuple:
    """The matchup weight that scores best on the **first half** of the window.

    A free parameter chosen on the data it is scored on is not a result, so the
    grid is walked on the first half by date and the winner is then priced and
    scored on the second. Returns `(weight, table)` and the table is printed,
    because a boundary solution should be visible as one.
    """
    first_dates, _ = halves_of(sorted(closes["game_date"].astype(str).unique()))
    train = closes[closes["game_date"].astype(str).isin(first_dates)]
    rows = []
    for w in grid:
        priced = price_with(train, ctx, stats, pitcher_bf, weight=w)
        settled = priced[priced["over_hit"].notna()]
        y = settled["over_hit"].astype(float)
        rows.append({"weight": w, "n": len(settled),
                     "brier": props.brier(settled["p_matchup"], y)})
    table = pd.DataFrame(rows)
    best = float(table.loc[table["brier"].idxmin(), "weight"])
    return best, table


def halves_of(dates: list) -> tuple:
    """The first and second half of a window, split on the median date."""
    cut = dates[len(dates) // 2]
    return [d for d in dates if d < cut], [d for d in dates if d >= cut]


# ───────────────────────────── scoring ─────────────────────────────

def brier_halves(priced: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """`props.brier_table` per stat on each half of the window, labelled."""
    dates = sorted(priced["game_date"].astype(str).unique())
    first, second = halves_of(dates)
    out = []
    for label, keep in (("first", first), ("second", second), ("all", dates)):
        sub = priced[priced["game_date"].astype(str).isin(keep)]
        if sub.empty:
            continue
        t = props.brier_table(sub, models=models)
        t.insert(0, "half", label)
        t["first_date"], t["last_date"] = min(keep), max(keep)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def paired_halves(priced: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """`a − b` per contract, per stat, on each half. Negative means `a` wins."""
    dates = sorted(priced["game_date"].astype(str).unique())
    first, second = halves_of(dates)
    rows = []
    for label, keep in (("first", first), ("second", second), ("all", dates)):
        sub = priced[priced["game_date"].astype(str).isin(keep)]
        for stat in list(sub["prop_stat"].unique()) + ["all"]:
            grp = sub if stat == "all" else sub[sub["prop_stat"] == stat]
            rows.append({"half": label, "prop_stat": stat,
                         **props.paired_brier(grp, a, b)})
    return pd.DataFrame(rows)


def venue_for(fee_rate: float, frictionless: bool) -> pnl.Venue:
    """Kalshi as quoted, or a variant for the sensitivity table.

    `fee_rate=0` is a maker fill (resting inside the spread rather than
    crossing it); `frictionless` collapses the book onto the close, which
    prices the model with no spread and no fee at all — the ceiling.
    """
    return pnl.Venue("kalshi", fee_rate, 0.0 if frictionless else None,
                     round_cents=not frictionless,
                     maker_rate=pnl.KALSHI_MAKER_RATE)


def money_table(df: pd.DataFrame, thresholds, stakings, draws: int,
                seed: int, venue: pnl.Venue = pnl.KALSHI,
                models=None) -> pd.DataFrame:
    """`pnl.run_exam`'s grid, on one venue, clustered by game."""
    frame = pnl.add_controls(df, seed=seed)
    rows = []
    for model in (models or MODELS):
        if model not in frame.columns:
            continue
        for threshold in thresholds:
            for staking in stakings:
                rows.append(pnl.evaluate(frame, model, venue, threshold,
                                         staking, draws=draws, seed=seed,
                                         group_col="game_pk"))
    return pd.DataFrame(rows)


def per_stat_money(df: pd.DataFrame, threshold: float, draws: int,
                   seed: int, venue: pnl.Venue = pnl.KALSHI,
                   models=None) -> pd.DataFrame:
    """The headline threshold, flat stakes, one row per prop stat."""
    rows = []
    for stat, grp in df.groupby("prop_stat"):
        frame = pnl.add_controls(grp.reset_index(drop=True), seed=seed)
        for model in (models or MODELS):
            if model not in frame.columns:
                continue
            r = pnl.evaluate(frame, model, venue, threshold, "flat",
                             draws=draws, seed=seed, group_col="game_pk")
            rows.append({"prop_stat": stat, **r})
    return pd.DataFrame(rows)


# ───────────────────────────── the maker exam ─────────────────────────────

def maker_table(priced: pd.DataFrame, candles: pd.DataFrame, models: list[str],
                margins, venue: pnl.Venue, hours: int, maker_round_cents: bool,
                anchor: str, draws: int, seed: int,
                stakings=("flat",)) -> pd.DataFrame:
    """Resting orders on props: every model at every margin, on both halves.

    The same rule the moneyline maker exam runs (`src/market/pnl.py`): post at
    `P − m` on the side we favour, fill in the first archived hour that traded
    at or through the limit, cancel unfilled at first pitch, pay the maker fee.
    The bootstrap is clustered by game, which the moneyline exam did not need
    and a prop book does — a hitter's 1+, 2+ and 3+ hits are one afternoon.
    """
    frame = props.to_maker_frame(priced)
    return pnl.maker_grid(frame, candles, models, margins=margins, venue=venue,
                          stakings=stakings, hours=hours, anchor=anchor,
                          maker_round_cents=maker_round_cents, draws=draws,
                          seed=seed, split=True, group_col="game_pk")


# ───────────────────────────── markdown ─────────────────────────────

def pct(x) -> str:
    return "—" if pd.isna(x) else f"{x:+.1%}"


def cents(x) -> str:
    return "—" if pd.isna(x) else f"{100 * x:+.2f}¢"


def brier_markdown(brier: pd.DataFrame, models: list[str]) -> str:
    head = {"p_model": "current", "p_matchup": "matchup", "p_market": "market",
            "p_league": "league-rate"}
    out = ["| half | stat | n | games | over rate | "
           + " | ".join(head.get(m, m) for m in models) + " |",
           "|---|---|---|---|---|" + "---|" * len(models)]
    for r in brier.itertuples(index=False):
        vals = " | ".join(f"{getattr(r, m):.5f}" for m in models)
        out.append(f"| {r.half} | {r.prop_stat} | {r.n} | {r.games} | "
                   f"{r.over_rate:.3f} | {vals} |")
    return "\n".join(out)


def paired_markdown(paired: pd.DataFrame, a: str, b: str) -> str:
    out = [f"| half | stat | n | Brier({a}) − Brier({b}) | se | t |",
           "|---|---|---|---|---|---|"]
    for r in paired.itertuples(index=False):
        out.append(f"| {r.half} | {r.prop_stat} | {r.n} | {r.diff:+.5f} | "
                   f"{r.se:.5f} | {r.t:+.2f} |")
    return "\n".join(out)


def maker_markdown(res: pd.DataFrame, half: str) -> str:
    sub = res[(res["half"] == half) & (res["staking"] == "flat")]
    out = ["| Model | m | posted | fill | crossed | ¢/posted | ¢/filled | "
           "ROI | ROI 95% CI |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sub.itertuples(index=False):
        out.append(f"| {LABELS.get(r.model, r.model)} | {r.margin:.2f} | "
                   f"{r.n_posted} | {r.fill_rate:.3f} | {r.marketable_rate:.3f} | "
                   f"{cents(r.pnl_per_posted)} | {cents(r.pnl_per_filled)} | "
                   f"{pct(r.roi)} | ({pct(r.roi_lo)}, {pct(r.roi_hi)}) |")
    return "\n".join(out)


def markdown(brier: pd.DataFrame, grid: pd.DataFrame, per_stat: pd.DataFrame,
             threshold: float, models: list[str]) -> str:
    out = ["### Brier per stat (lower is better)", "",
           brier_markdown(brier, models),
           "", f"### Money, Kalshi, flat 1u, edge ≥ {threshold:.0%}", "",
           "| Model | n bets | hit | staked | return | ROI | ROI 95% CI | "
           "mean edge | CLV | max DD | fees |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    flat = grid[(grid["staking"] == "flat") & (grid["threshold"] == threshold)]
    for r in flat.itertuples(index=False):
        hit = "—" if pd.isna(r.hit_rate) else f"{r.hit_rate:.3f}"
        edge = "—" if pd.isna(r.mean_edge) else f"{100 * r.mean_edge:.2f} pt"
        clv = "—" if pd.isna(r.clv) else f"{100 * r.clv:+.2f} pt"
        out.append(f"| {LABELS.get(r.model, r.model)} | {r.n_bets} | {hit} | "
                   f"{r.total_staked:.1f}u | {r.total_return:+.2f}u | {pct(r.roi)} | "
                   f"({pct(r.roi_lo)}, {pct(r.roi_hi)}) | {edge} | {clv} | "
                   f"{r.max_drawdown:.2f}u | {r.total_fees:.2f}u |")
    thresholds = sorted(grid["threshold"].unique())
    out += ["", "### ROI by threshold (flat 1u)", "",
            "| Model | " + " | ".join(f"n ≥{int(100 * t)}pt" for t in thresholds)
            + " | " + " | ".join(f"ROI ≥{int(100 * t)}pt" for t in thresholds) + " |",
            "|---|" + "---|" * (2 * len(thresholds))]
    for model in grid["model"].unique():
        sub = grid[(grid["model"] == model) & (grid["staking"] == "flat")].sort_values("threshold")
        if sub.empty:
            continue
        out.append(f"| {LABELS.get(model, model)} | "
                   + " | ".join(str(int(n)) for n in sub["n_bets"]) + " | "
                   + " | ".join(pct(v) for v in sub["roi"]) + " |")
    out += ["", f"### Money per stat, flat 1u, edge ≥ {threshold:.0%}", "",
            "| Stat | Model | n bets | hit | ROI | ROI 95% CI |",
            "|---|---|---|---|---|---|"]
    for r in per_stat.itertuples(index=False):
        hit = "—" if pd.isna(r.hit_rate) else f"{r.hit_rate:.3f}"
        out.append(f"| {r.prop_stat} | {LABELS.get(r.model, r.model)} | {r.n_bets} | "
                   f"{hit} | {pct(r.roi)} | ({pct(r.roi_lo)}, {pct(r.roi_hi)}) |")
    return "\n".join(out)


def default_closes() -> Path:
    """The committed archive, with the gitignored one as a fallback.

    The closes were gitignored once and did not survive a container restart,
    which cost a full 30k-request re-fetch. They live in `data/market/` now;
    an older working copy under `data/parquet/` is still read if that is all
    there is.
    """
    root = Path(__file__).resolve().parent.parent
    for p in (root / "data/market/prop_closes_2026.parquet",
              root / "data/parquet/prop_closes_2026.parquet"):
        if p.exists():
            return p
    return root / "data/market/prop_closes_2026.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--closes", type=Path, default=None)
    ap.add_argument("--candles", type=Path, default=None,
                    help="hourly prop candles; default "
                         "data/market/kalshi_prop_candles_<season>.parquet")
    ap.add_argument("--stats", nargs="+", default=list(props.PRICEABLE))
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.02, 0.04, 0.06])
    ap.add_argument("--headline", type=float, default=0.02)
    ap.add_argument("--stakings", nargs="+", default=["flat", "kelly"])
    ap.add_argument("--draws", type=int, default=pnl.BOOTSTRAP_DRAWS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pitcher-bf", choices=["fixed", "own"], default="fixed",
                    help="batters faced by a starter: the league average (23) or "
                         "his own per-start rate to date")
    ap.add_argument("--matchup", choices=["off", "on", "auto"], default="auto",
                    help="fold the opposing pitching into the price by log5; "
                         "'auto' follows props.MATCHUP_DEFAULT, which the gate sets")
    ap.add_argument("--matchup-weight", type=float, default=None,
                    help="skip the out-of-sample search and use this weight")
    ap.add_argument("--matchup-weights", nargs="+", type=float, default=list(WEIGHT_GRID),
                    help="the grid searched on the first half of the window")
    ap.add_argument("--fee-rate", type=float, default=pnl.KALSHI_TAKER_RATE,
                    help="Kalshi taker fee rate; 0 prices a maker fill")
    ap.add_argument("--frictionless", action="store_true",
                    help="fill at the close with no spread — the ceiling")
    ap.add_argument("--min-volume", type=float, default=0.0,
                    help="keep only contracts with at least this much volume traded "
                         "BEFORE first pitch; `volume_total` is a market's whole life "
                         "and an in-play market trades heaviest once the outcome is "
                         "live, so filtering on it leaks the result")
    ap.add_argument("--maker", action="store_true",
                    help="rest a limit order instead of crossing the close")
    ap.add_argument("--maker-margins", nargs="+", type=float,
                    default=list(pnl.MAKER_MARGINS))
    ap.add_argument("--maker-hours", type=int, default=pnl.MAKER_HOURS)
    ap.add_argument("--maker-anchor", choices=["close", "post"], default="close")
    ap.add_argument("--maker-round-cents", action="store_true")
    ap.add_argument("--kalshi-maker-fee-rate", type=float, default=pnl.KALSHI_MAKER_RATE)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--priced-out", type=Path, default=None,
                    help="write the priced frame to parquet for reuse")
    ap.add_argument("--priced-in", type=Path, default=None,
                    help="reuse a priced frame instead of rebuilding it")
    args = ap.parse_args()

    closes_path = args.closes or default_closes()
    closes = pd.read_parquet(closes_path)
    if args.min_volume:
        closes = closes[closes["volume_pre"] >= args.min_volume]
    stats = tuple(args.stats)
    with_matchup = args.matchup == "on" or (args.matchup == "auto"
                                            and props.MATCHUP_DEFAULT)

    weight_table = None
    if args.priced_in:
        priced = pd.read_parquet(args.priced_in)
        with_matchup = "p_matchup" in priced.columns
        weight = args.matchup_weight
    else:
        weight = args.matchup_weight if args.matchup_weight is not None \
            else props.MATCHUP_WEIGHT
        ctx = contexts(closes, args.season, stats, with_matchup, weight)
        if with_matchup and args.matchup_weight is None:
            weight, weight_table = choose_weight(closes, ctx, stats,
                                                 args.pitcher_bf,
                                                 tuple(args.matchup_weights))
            logger.info("matchup weight chosen on the first half: %.2f", weight)
        priced = price_with(closes, ctx, stats, args.pitcher_bf, weight=weight)
    if args.priced_out:
        priced.to_parquet(args.priced_out, index=False)

    models = ["p_model"] + (["p_matchup"] if with_matchup else []) \
        + ["p_market", "p_league"]
    brier = brier_halves(priced, models)
    frame = props.to_pnl_frame(priced)
    money_models = ["model"] + (["matchup"] if with_matchup else []) \
        + ["league", "market", "random_edge"]
    venue = venue_for(args.fee_rate, args.frictionless)
    grid = money_table(frame, args.thresholds, args.stakings, args.draws,
                       args.seed, venue, models=money_models)
    per_stat = per_stat_money(frame, args.headline, args.draws, args.seed,
                              venue, models=money_models)

    print("\n== coverage ==")
    print(closes.groupby("prop_stat").agg(archived=("market_id", "nunique")).to_string())
    print(f"\npriced {len(priced)} of {len(closes)} archived closes; "
          f"{frame['game_pk'].nunique()} games, {frame['player_id'].nunique()} players, "
          f"{frame['game_date'].min()} .. {frame['game_date'].max()}")
    if weight_table is not None:
        print("\n== matchup weight, chosen on the first half ==")
        print(weight_table.to_string(index=False))
        print(f"chosen: {weight}")
    print("\n== Brier ==")
    print(brier.to_string(index=False))
    if with_matchup:
        paired = paired_halves(priced, "p_matchup", "p_model")
        print("\n== paired: matchup − current (negative = matchup wins) ==")
        print(paired.to_string(index=False))
        print("\n== paired: matchup − market ==")
        print(paired_halves(priced, "p_matchup", "p_market").to_string(index=False))
    print("\n== money (taker) ==")
    print(grid.to_string(index=False))

    maker = None
    if args.maker:
        candles_path = args.candles or (Path(__file__).resolve().parent.parent
                                        / f"data/market/kalshi_prop_candles_{args.season}.parquet")
        candles = pd.read_parquet(candles_path)
        logger.info("%d hourly candles over %d prop markets",
                    len(candles), candles["market_id"].nunique())
        maker_venue = pnl.Venue("kalshi", args.fee_rate, None, round_cents=True,
                                maker_rate=args.kalshi_maker_fee_rate)
        maker_models = ["model"] + (["matchup"] if with_matchup else [])
        maker = maker_table(priced, candles, maker_models, args.maker_margins,
                            maker_venue, args.maker_hours,
                            args.maker_round_cents, args.maker_anchor,
                            args.draws, args.seed)
        print("\n== money (maker) ==")
        print(maker.to_string(index=False))

    if args.markdown:
        print("\n" + markdown(brier, grid, per_stat, args.headline, models))
        if with_matchup:
            print("\n### Paired, matchup − current (negative = matchup wins)\n")
            print(paired_markdown(paired_halves(priced, "p_matchup", "p_model"),
                                  "matchup", "current"))
        if maker is not None:
            for half in ("first", "second"):
                print(f"\n### Maker, {half} half (flat, one contract per order)\n")
                print(maker_markdown(maker, half))


if __name__ == "__main__":
    main()
