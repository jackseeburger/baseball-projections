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
    # the posterior columns widened by the hierarchical model (BAS-92)
    python scripts/props_exam.py --matchup on --posterior \\
        --bayes-width data/eval/bas92
"""
from __future__ import annotations

import argparse
import json
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
# The posterior-probability-of-edge threshold for decide_posterior. Chosen on
# the first half by fee-waived flat-stake ROI, scored on the second — same
# walk-forward discipline as WEIGHT_GRID. All values are >= 0.5, which is what
# keeps decide_posterior from ever trading inside the spread (pnl.py).
TAU_GRID = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
# The column BAS-70's other half (src/market/props.py) is expected to add:
# the posterior standard deviation of P(over) that decide_posterior needs.
SD_COL = "p_over_sd"
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
               pitcher_bf: str, weight: float | None = None,
               bayes_width=None, draw_streams: str = "shared") -> pd.DataFrame:
    """One priced frame, optionally at a different matchup weight.

    `bayes_width` is BAS-92's `props.BayesWidth` (docs/posterior-width.md) or
    `None`, which is the served arm. It changes `p_over_bb`/`p_over_sd` and
    nothing else, so the matchup weight search above never passes one: the
    weight is chosen on `p_matchup`, which the arm cannot move.
    """
    if weight is not None and ctx["matchup_ctx"] is not None:
        ctx["matchup_ctx"]["weight"] = float(weight)
    return props.price(closes, ctx["batter_ctx"], ctx["pitcher_ctx"],
                       ctx["slots"], stats=stats, pitcher_bf=pitcher_bf,
                       matchup_ctx=ctx["matchup_ctx"],
                       bayes_width=bayes_width, draw_streams=draw_streams)


def choose_weight(closes: pd.DataFrame, ctx: dict, stats: tuple,
                  pitcher_bf: str, cut: str, grid=WEIGHT_GRID) -> tuple:
    """The matchup weight that scores best on the **first half** of the window.

    A free parameter chosen on the data it is scored on is not a result, so the
    grid is walked on the first half by date and the winner is then priced and
    scored on the second. Returns `(weight, table)` and the table is printed,
    because a boundary solution should be visible as one.
    """
    train = closes[closes["game_date"].astype(str) < cut]
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


# ─────────────────────── selecting on P(edge > 0) ───────────────────────

def choose_tau(frame: pd.DataFrame, model: str, sd_col: str, cut: str,
               venue: pnl.Venue, grid=TAU_GRID, date_col: str = "date") -> tuple:
    """The tau that scores best on the **first half**, fee-waived flat-stake ROI.

    Same discipline as `choose_weight`: a free parameter chosen on the data it
    is scored on is not a result. `venue` should already be the fee-waived one
    (`venue_for(0.0, frictionless=False)`) — fee-waived isolates the selection
    rule's own quality from the fee, which is a separate, known loser (BAS-70
    prediction 3). Ties, and a tau with zero bets, are broken toward the
    smallest tau that has *any* bets, so a boundary solution (the grid's
    least-selective end winning) is visible in the printed table rather than
    silently landing on an empty cell.
    """
    train = frame[frame[date_col].astype(str) < cut]
    rows = []
    for tau in grid:
        row = pnl.evaluate(train, model, venue, staking="flat",
                           group_col="game_pk", rule="posterior", tau=tau,
                           sd_col=sd_col)
        rows.append({"tau": tau, "n_bets": row["n_bets"], "roi": row["roi"]})
    table = pd.DataFrame(rows)
    tradeable = table[table["n_bets"] > 0]
    if tradeable.empty:
        return float("nan"), table
    best_roi = tradeable["roi"].max()
    # Among ties on ROI (the coarse grid makes exact ties plausible at small
    # n), the least-selective tau — most bets, closest to the threshold
    # rule's own bet count — is the more conservative pick.
    best = tradeable[tradeable["roi"] == best_roi].sort_values("n_bets",
                                                                ascending=False)
    return float(best.iloc[0]["tau"]), table


def matched_threshold(frame: pd.DataFrame, model: str, venue: pnl.Venue,
                      n_target: int, lo: float = 0.0, hi: float = 0.30,
                      iters: int = 40) -> float:
    """The `decide` threshold whose bet count on `frame` is closest to
    `n_target`, found by bisection.

    This is **not a chosen parameter** — it is a comparison device, found
    after the fact on the same half the posterior rule is scored on, purely
    so prediction 2 ("beats the threshold rule at a matched number of bets")
    can actually be checked rather than confounded with "the posterior rule
    just bets less." `decide`'s bet count is non-increasing in `threshold`
    (a stricter margin never adds a bet), so bisection on that count is well
    posed; ties on the count are resolved by keeping the candidate visited
    first, which biases toward the tighter threshold — the harder standard
    for the threshold rule to still be a fair comparator at.
    """
    def n_bets_at(t: float) -> int:
        return len(pnl.bet_frame(frame, model, venue, threshold=t))

    a, b = lo, hi
    best_t, best_diff = lo, abs(n_bets_at(lo) - n_target)
    for _ in range(iters):
        mid = (a + b) / 2.0
        n = n_bets_at(mid)
        diff = abs(n - n_target)
        if diff < best_diff:
            best_t, best_diff = mid, diff
        if n > n_target:
            a = mid          # need a stricter margin to shed bets
        elif n < n_target:
            b = mid           # need a looser margin to gain bets
        else:
            best_t, best_diff = mid, 0
            break
    return best_t


def posterior_comparison(frame: pd.DataFrame, model: str, sd_col: str, tau: float,
                         threshold: float, venue_fee_waived: pnl.Venue,
                         venue_as_quoted: pnl.Venue, draws: int, seed: int,
                         group_col: str = "game_pk") -> pd.DataFrame:
    """Threshold-at-2pt vs. posterior-at-tau vs. threshold-at-matched-count.

    All three rows are evaluated on the same frame (the second, held-out half
    of the window, whole or one prop stat), so the bet counts and ROIs are
    directly comparable. `matched` is threshold-only — it exists to answer
    "is the posterior rule just betting less?", not as a strategy of its own.
    """
    posterior_n = pnl.evaluate(frame, model, venue_as_quoted, staking="flat",
                               group_col=group_col, rule="posterior", tau=tau,
                               sd_col=sd_col)["n_bets"]
    matched = matched_threshold(frame, model, venue_as_quoted, posterior_n)

    specs = [
        ("threshold @ 2pt", dict(rule="threshold", threshold=threshold)),
        (f"posterior @ tau={tau:.2f}",
         dict(rule="posterior", tau=tau, sd_col=sd_col)),
        (f"threshold @ matched (t={matched:.4f})",
         dict(rule="threshold", threshold=matched)),
    ]
    rows = []
    for label, kwargs in specs:
        fw = pnl.evaluate(frame, model, venue_fee_waived, staking="flat",
                          group_col=group_col, draws=draws, seed=seed, **kwargs)
        aq = pnl.evaluate(frame, model, venue_as_quoted, staking="flat",
                          group_col=group_col, draws=draws, seed=seed, **kwargs)
        rows.append({
            "rule": label, "n_bets": fw["n_bets"],
            "roi_fee_waived": fw["roi"], "roi_fee_waived_lo": fw["roi_lo"],
            "roi_fee_waived_hi": fw["roi_hi"],
            "roi_as_quoted": aq["roi"], "roi_as_quoted_lo": aq["roi_lo"],
            "roi_as_quoted_hi": aq["roi_hi"],
        })
    return pd.DataFrame(rows)


def posterior_comparison_by_stat(frame: pd.DataFrame, model: str, sd_col: str,
                                 tau: float, threshold: float,
                                 venue_fee_waived: pnl.Venue,
                                 venue_as_quoted: pnl.Venue, draws: int,
                                 seed: int, stat_col: str = "prop_stat") -> pd.DataFrame:
    """`posterior_comparison`, pooled and broken out per prop stat.

    The matched threshold is re-found **within each stat's own rows** — a
    single pooled threshold would not give a matched *count* on a stat whose
    posterior rule bet count differs from the pool's.
    """
    out = []
    pooled = posterior_comparison(frame, model, sd_col, tau, threshold,
                                  venue_fee_waived, venue_as_quoted, draws, seed)
    pooled.insert(0, "stat", "all")
    out.append(pooled)
    for stat, grp in frame.groupby(stat_col):
        t = posterior_comparison(grp.reset_index(drop=True), model, sd_col, tau,
                                 threshold, venue_fee_waived, venue_as_quoted,
                                 draws, seed)
        t.insert(0, "stat", stat)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def posterior_comparison_markdown(table: pd.DataFrame) -> str:
    out = ["| stat | rule | n bets | ROI fee-waived | 95% CI | ROI as-quoted | 95% CI |",
           "|---|---|---|---|---|---|---|"]
    for r in table.itertuples(index=False):
        out.append(f"| {r.stat} | {r.rule} | {r.n_bets} | {pct(r.roi_fee_waived)} | "
                   f"({pct(r.roi_fee_waived_lo)}, {pct(r.roi_fee_waived_hi)}) | "
                   f"{pct(r.roi_as_quoted)} | "
                   f"({pct(r.roi_as_quoted_lo)}, {pct(r.roi_as_quoted_hi)}) |")
    return "\n".join(out)


# ───────────────────────────── scoring ─────────────────────────────

def brier_halves(priced: pd.DataFrame, models: list[str], cut: str) -> pd.DataFrame:
    """`props.brier_table` per stat on each half of the window, labelled."""
    dates = sorted(priced["game_date"].astype(str).unique())
    first = [d for d in dates if d < cut]
    second = [d for d in dates if d >= cut]
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


def paired_halves(priced: pd.DataFrame, a: str, b: str, cut: str) -> pd.DataFrame:
    """`a − b` per contract, per stat, on each half. Negative means `a` wins."""
    dates = sorted(priced["game_date"].astype(str).unique())
    first = [d for d in dates if d < cut]
    second = [d for d in dates if d >= cut]
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
                anchor: str, draws: int, seed: int, cut: str | None = None,
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
                          seed=seed, split=True, group_col="game_pk", cut=cut)


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
    ap.add_argument("--start", help="earliest game date to score (YYYY-MM-DD)")
    ap.add_argument("--end", help="latest game date to score (YYYY-MM-DD)")
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
    ap.add_argument("--posterior", action="store_true",
                    help="also run the P(edge>0) selection rule and the "
                         "matched-bet-count comparison table (BAS-70); needs "
                         f"a {SD_COL!r} column on the priced frame")
    ap.add_argument("--bayes-width", type=Path, default=None,
                    help="directory of bayes_width_<cutoff>.parquet files "
                         "(scripts/run_bayes_width.py): price the posterior "
                         "columns with the hierarchical model's width on K, "
                         "BB and HR instead of Marcel's Beta (BAS-92, "
                         "docs/posterior-width.md). Off by default; the mean "
                         "price is unchanged either way")
    ap.add_argument("--width-audit", type=Path, default=None,
                    help="write the bayes-width fallback / floor counters here")
    ap.add_argument("--draw-streams", choices=list(props.DRAW_STREAMS),
                    default="shared",
                    help="'shared' is the one advancing generator the "
                         "committed archive was priced under; 'isolated' "
                         "seeds each (date, player) separately so a change to "
                         "one player's Beta cannot move another player's "
                         "Monte Carlo draws (props.DRAW_STREAMS)")
    ap.add_argument("--tau", type=float, default=None,
                    help="skip the walk-forward search and use this tau")
    ap.add_argument("--tau-grid", nargs="+", type=float, default=list(TAU_GRID))
    args = ap.parse_args()

    closes_path = args.closes or default_closes()
    closes = pd.read_parquet(closes_path)
    if args.start:
        closes = closes[closes["game_date"].astype(str) >= args.start]
    if args.end:
        closes = closes[closes["game_date"].astype(str) <= args.end]
    if args.min_volume:
        closes = closes[closes["volume_pre"] >= args.min_volume]
    stats = tuple(args.stats)
    with_matchup = args.matchup == "on" or (args.matchup == "auto"
                                            and props.MATCHUP_DEFAULT)

    priceable = closes[closes["prop_stat"].isin(stats) & closes["player_id"].notna()]
    all_dates = sorted(priceable["game_date"].astype(str).unique())
    cut = all_dates[len(all_dates) // 2] if all_dates else ""

    weight_table = None
    bayes_width = (props.BayesWidth.load(args.bayes_width)
                   if args.bayes_width else None)
    if args.priced_in:
        priced = pd.read_parquet(args.priced_in)
        with_matchup = "p_matchup" in priced.columns
        weight = args.matchup_weight
        if bayes_width is not None:
            ap.error("--bayes-width re-prices the posterior columns, so it "
                     "cannot be combined with --priced-in")
    else:
        weight = args.matchup_weight if args.matchup_weight is not None \
            else props.MATCHUP_WEIGHT
        ctx = contexts(closes, args.season, stats, with_matchup, weight)
        if with_matchup and args.matchup_weight is None:
            # The weight is chosen on `p_matchup`, which the width arm does
            # not touch, so the search runs on the served Beta either way and
            # the chosen weight is the same number both arms are scored at.
            weight, weight_table = choose_weight(closes, ctx, stats,
                                                 args.pitcher_bf, cut,
                                                 tuple(args.matchup_weights))
            logger.info("matchup weight chosen on the first half: %.2f", weight)
        priced = price_with(closes, ctx, stats, args.pitcher_bf, weight=weight,
                            bayes_width=bayes_width,
                            draw_streams=args.draw_streams)
    if args.priced_out:
        priced.to_parquet(args.priced_out, index=False)
    if bayes_width is not None:
        print("\n== bayes width coverage ==")
        print(json.dumps({k: v for k, v in bayes_width.audit.items()
                          if k != "cutoff_used"}, indent=1))
        if args.width_audit:
            args.width_audit.parent.mkdir(parents=True, exist_ok=True)
            args.width_audit.write_text(json.dumps(bayes_width.audit, indent=1))

    models = ["p_model"] + (["p_matchup"] if with_matchup else []) \
        + ["p_market", "p_league"]
    brier = brier_halves(priced, models, cut)
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
    print(f"\nhalves split at {cut}")
    print(f"priced {len(priced)} of {len(closes)} archived closes; "
          f"{frame['game_pk'].nunique()} games, {frame['player_id'].nunique()} players, "
          f"{frame['game_date'].min()} .. {frame['game_date'].max()}")
    if weight_table is not None:
        print("\n== matchup weight, chosen on the first half ==")
        print(weight_table.to_string(index=False))
        print(f"chosen: {weight}")
    print("\n== Brier ==")
    print(brier.to_string(index=False))
    if with_matchup:
        paired = paired_halves(priced, "p_matchup", "p_model", cut)
        print("\n== paired: matchup − current (negative = matchup wins) ==")
        print(paired.to_string(index=False))
        print("\n== paired: matchup − market ==")
        print(paired_halves(priced, "p_matchup", "p_market", cut).to_string(index=False))
    print("\n== money (taker) ==")
    print(grid.to_string(index=False))

    posterior_table = None
    if args.posterior:
        primary = "matchup" if with_matchup else "model"
        if SD_COL not in frame.columns:
            logger.warning("--posterior requested but %r is not on the priced "
                           "frame; skipping the P(edge>0) selection rule "
                           "(this is the other half of BAS-70, built in "
                           "src/market/props.py)", SD_COL)
        else:
            fee_waived = venue_for(0.0, args.frictionless)
            as_quoted = venue
            train = frame[frame["date"].astype(str) < cut]
            second = frame[frame["date"].astype(str) >= cut]
            if args.tau is not None:
                tau, tau_table = args.tau, None
            else:
                tau, tau_table = choose_tau(train, primary, SD_COL, cut,
                                            fee_waived, tuple(args.tau_grid))
                logger.info("tau chosen on the first half: %.2f", tau)
            print("\n== BAS-70: selecting on P(edge>0) ==")
            if tau_table is not None:
                print("\n-- tau grid, first half, fee-waived flat-stake ROI --")
                print(tau_table.to_string(index=False))
            print(f"chosen tau: {tau}")
            posterior_table = posterior_comparison_by_stat(
                second, primary, SD_COL, tau, args.headline, fee_waived,
                as_quoted, args.draws, args.seed)
            print("\n-- second half: threshold @ 2pt vs. posterior @ tau vs. "
                  "threshold @ matched bet count (matched is a comparison "
                  "device, not a chosen parameter) --")
            print(posterior_table.to_string(index=False))
            if args.markdown:
                print("\n### BAS-70 selection comparison, second half\n")
                print(posterior_comparison_markdown(posterior_table))

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
                            args.draws, args.seed, cut=cut)
        print("\n== money (maker) ==")
        print(maker.to_string(index=False))
        print("\n== maker margin chosen on the first half, by ¢ per posted ==")
        for model in maker["model"].unique():
            m = pnl.choose_margin(maker, model)
            scored = maker[(maker["model"] == model) & (maker["half"] == "second")
                           & (maker["margin"] == m) & (maker["staking"] == "flat")]
            roi = float(scored["roi"].iloc[0]) if len(scored) else float("nan")
            print(f"  {LABELS.get(model, model):32s} m={m:.2f} "
                  f"→ second-half ROI {roi:+.1%}")

    if args.markdown:
        print("\n" + markdown(brier, grid, per_stat, args.headline, models))
        if with_matchup:
            print("\n### Paired, matchup − current (negative = matchup wins)\n")
            print(paired_markdown(paired_halves(priced, "p_matchup", "p_model", cut),
                                  "matchup", "current"))
        if maker is not None:
            for half in ("first", "second"):
                print(f"\n### Maker, {half} half (flat, one contract per order)\n")
                print(maker_markdown(maker, half))


if __name__ == "__main__":
    main()
