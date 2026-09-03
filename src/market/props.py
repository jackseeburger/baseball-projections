"""Our probability for a player prop, walk-forward, and its money exam.

Station A meets station M. `docs/money-exam-2026.md` ended on the finding that
every per-game model loses after the fee and the spread, and named the way out:
*a less efficient contract*. A player prop is the cleanest such contract we can
reach — it is priced off exactly the component rates station A models, and
`docs/backtest-baselines.md` says Marcel-with-the-partial-season is the arm
that wins K% and HR intra-season. So the question this module answers is
whether that rate model, priced into a contract and charged the same fee,
survives where the moneyline did not.

The chain, all pure functions over frames so it unit-tests without a network:

    batter game logs ─► lineups.marcel_rates()   K, BB+HBP, HR, ISO, BABIP as
                        (strictly before the      of the game date only
                         game date)
                     ─► lineups.event_rates()    per-PA probability of each
                                                 plate-appearance outcome
                     ─► pa_outcome_probs()       hit / HR per PA and expected
                                                 total bases per PA
    posted lineup    ─► slot_pa()                plate appearances the slot gets
                     ─► binom_at_least()         P(count ≥ the line)

    pitcher logs     ─► starters.marcel_rates()  K per batter faced
                     ─► binom_at_least(n = BF)   P(strikeouts ≥ the line)

**Which stats we can price, and which we cannot.** The five hitter components
support hits, home runs and total bases, and the pitcher components support
strikeouts. They do not support:

* **RBI** — an RBI is a function of who is on base in front of the hitter, not
  of his own rates; pricing it needs a lineup-sequencing simulator.
* **SB** — nothing in the rate table is a stolen-base rate; the game logs we
  cache do not carry attempts.
* **Outs recorded** — a start's length is a manager decision (leverage, pitch
  count, score) far more than a rate, and FIP says nothing about it.

Those three are archived (Part 1 and 2) and left unpriced rather than priced
badly; a Brier we could not beat with a model we do not believe would be
worse than an honest gap.

**Leakage.** Every rate is rebuilt from games strictly *before* the game's own
date, and the posted lineup is the card the club filed before first pitch —
the same guard `src/sim/lineups.py` documents. The close being scored is the
last quote before that first pitch, so the model never sees anything the price
did not.
"""
from __future__ import annotations

import logging
from math import comb, exp, factorial

import numpy as np
import pandas as pd

from src.sim import lineups as lu_model
from src.sim import starters as sp_model

logger = logging.getLogger(__name__)

# Plate appearances a batting-order slot gets in a game. Leadoff sees about
# 4.6 and the ninth hitter about 3.7 — one turn of the order apart — which is
# the same structural 1.24:1 spread `lineups.slot_pa_shares` derives from
# 38 team plate appearances. Held as an explicit table because a prop is about
# one hitter's count, not a lineup's share.
SLOT_PA = {slot: round(4.6 - 0.1125 * (slot - 1), 4) for slot in range(1, 10)}
DEFAULT_PA = 4.1                # a hitter in the game whose slot we never saw
# Batters an average start faces: 5.5 innings (starters.STARTER_IP) at the
# league's ~4.2 batters per inning.
STARTER_BF = 23.0

# Is the opposing pitcher in the hitter's price? The gate rule
# (architecture.md §3) decides: the matchup arm (`src/market/matchup.py`) is
# the default only if it beats the current price out of sample, on the half of
# the archive its one free constant was not chosen on.
#
# **It does, and since Sept 3 2026 it is on.** On the second half of the
# archive (2026-08-17 to 09-02, 30,423 settled contracts) the matchup price
# beats the current one by 0.00095 of Brier paired per contract, t = -4.4 with
# the standard error clustered by game, and it wins every stat separately —
# strikeouts by 0.0057, hits by 0.0006, total bases by 0.0005, home runs by
# 0.0001. The weight was chosen on the first half and is the boundary of its
# own grid, which is to say the first half asked for the identity undiluted.
# See docs/props-exam-2026.md.
MATCHUP_DEFAULT = True
MATCHUP_WEIGHT = 1.0

PRICEABLE = ("hits", "hr", "tb", "k")
UNPRICED = {"rbi": "needs lineup sequencing, not the batter's own rates",
            "sb": "no stolen-base rate in the component table",
            "outs": "start length is a manager decision, not a rate"}


# ───────────────────────────── distributions ─────────────────────────────

def binom_at_least(k: int, n: float, p: float) -> float:
    """P(X ≥ k) for X ~ Binomial(n, p), with a fractional `n` interpolated.

    Expected plate appearances are fractional (a leadoff hitter gets 4.6), and
    a hitter gets a whole number of them, so `n = 4.6` is scored as the mixture
    0.4·Binomial(4) + 0.6·Binomial(5) rather than rounded — rounding moves
    P(1+ hit) by about two points, which is more than the edges being traded.
    """
    p = float(min(max(p, 0.0), 1.0))
    if k <= 0:
        return 1.0
    lo = int(np.floor(n))
    w = float(n) - lo
    out = 0.0
    for trials, weight in ((lo, 1.0 - w), (lo + 1, w)):
        if weight <= 0 or trials < 0:
            continue
        if k > trials:
            continue
        below = sum(comb(trials, i) * p ** i * (1 - p) ** (trials - i)
                    for i in range(k))
        out += weight * (1.0 - below)
    return float(min(max(out, 0.0), 1.0))


def poisson_at_least(k: int, mean: float) -> float:
    """P(X ≥ k) for X ~ Poisson(mean).

    Total bases are not a count of successes in a fixed number of trials — one
    plate appearance can produce four of them — so the binomial does not apply
    and a Poisson on expected bases is the standard stand-in. It understates
    the tail slightly (bases arrive in lumps of 1, 2, 3 and 4, so the real
    distribution is more dispersed than Poisson), which the exam sees as a
    small systematic under-price of the high lines.
    """
    m = max(float(mean), 0.0)
    if k <= 0:
        return 1.0
    below = sum(exp(-m) * m ** i / factorial(i) for i in range(k))
    return float(min(max(1.0 - below, 0.0), 1.0))


def threshold(line: float) -> int:
    """A line of 2.5 is the contract "3+", so the count that pays is 3."""
    return int(np.floor(float(line)) + 1)


# ───────────────────────────── the rate → prop step ─────────────────────────

def pa_outcome_probs(rates, lg: dict) -> dict:
    """Per-plate-appearance hit and home-run probability, and expected bases.

    `rates` is one row of `lineups.marcel_rates` (or the league's own rates).
    Doubles and triples arrive together from ISO, split at the league's mix, so
    a double-or-triple is worth `2 + triple_share` bases.
    """
    ev = lu_model.event_rates({f"rate_{c}": float(rates[f"rate_{c}"])
                               for c in lu_model.COMPONENTS}, lg)
    ts = float(lg["triple_share"])
    return {
        "hit": ev["b1"] + ev["d23"] + ev["hr"],
        "hr": ev["hr"],
        "tb": ev["b1"] + (2.0 + ts) * ev["d23"] + 4.0 * ev["hr"],
    }


def batter_prop_prob(stat: str, line: float, per_pa: dict, pa: float) -> float:
    """P(the hitter clears `line`) given his per-PA rates and expected PA."""
    k = threshold(line)
    if stat == "hits":
        return binom_at_least(k, pa, per_pa["hit"])
    if stat == "hr":
        return binom_at_least(k, pa, per_pa["hr"])
    if stat == "tb":
        return poisson_at_least(k, pa * per_pa["tb"])
    raise ValueError(f"{stat} is not a batter prop this module prices")


def pitcher_prop_prob(stat: str, line: float, rate_k: float,
                      bf: float = STARTER_BF) -> float:
    """P(the starter clears `line`) from his K per batter faced."""
    if stat != "k":
        raise ValueError(f"{stat} is not a pitcher prop this module prices")
    return binom_at_least(threshold(line), bf, rate_k)


# ───────────────────────── as-of-date assembly ─────────────────────────

def batter_inputs(season: int, batter_ids, prior_seasons: int = 2,
                  refresh: bool = False) -> dict:
    """Everything the batter rate table needs, fetched once.

    Mirrors `starters.rate_inputs`: completed prior seasons plus this season's
    dated game logs, so a date cut can be applied per slate. Responses are
    cached under data/cache/statsapi/, so a rerun is offline and identical.
    """
    from src.data.mlb_stats_api import build_seasons_table, fetch_batter_game_logs

    prior = build_seasons_table(season - prior_seasons, season - 1)
    prior_counts = lu_model.normalize_counts(prior)
    logs = fetch_batter_game_logs(batter_ids, season, refresh=refresh)
    logs = logs[logs["game_type"] == "R"]
    game_logs = lu_model.normalize_counts(logs)
    game_logs["date"] = logs["date"].to_numpy()
    return {"season": season, "prior_counts": prior_counts,
            "game_logs": game_logs, "league": lu_model.league_rates(prior_counts)}


def pitcher_inputs(season: int, pitcher_ids, prior_seasons: int = 2,
                   refresh: bool = False) -> dict:
    """`starters.rate_inputs` under this module's name, for symmetry."""
    return sp_model.rate_inputs(season, pitcher_ids, prior_seasons=prior_seasons,
                                refresh=refresh)


def batter_rates(inputs: dict, as_of: str, ballast=lu_model.BALLAST) -> pd.DataFrame:
    """Marcel-with-the-partial-season component rates strictly before `as_of`."""
    current = lu_model.games_before(inputs["game_logs"], as_of)
    counts = pd.concat([inputs["prior_counts"], current], ignore_index=True)
    return lu_model.marcel_rates(counts, inputs["season"], inputs["league"],
                                 ballast=ballast)


def pitcher_rates(inputs: dict, as_of: str,
                  ballast=sp_model.BALLAST_BF) -> pd.DataFrame:
    """The same, per batter faced, for pitchers."""
    current = sp_model.appearances_before(inputs["game_logs"], as_of)
    counts = pd.concat([inputs["prior_counts"], current], ignore_index=True)
    return sp_model.marcel_rates(counts, inputs["season"], inputs["league"],
                                 ballast=ballast)


def starter_bf(inputs: dict, as_of: str, default: float = STARTER_BF,
               ballast_starts: float = 5.0) -> dict:
    """{pitcher_id: batters he is expected to face}, from his starts to date.

    `STARTER_BF` is the league's average start and is what the exam uses by
    default, but the spread around it is large — an ace works into the seventh
    and a bulk reliever is pulled after four — and the strikeout line moves
    with it. This is the pitcher analogue of the hitter's lineup slot: his own
    batters-faced per start before this date, shrunk toward the league average
    with five starts of ballast so a man with one start is not projected off
    it. Appearances on the date itself are excluded, same as the rates.
    """
    logs = inputs["game_logs"]
    if logs is None or len(logs) == 0 or "date" not in logs.columns:
        return {}
    past = logs[logs["date"].astype(str) < str(as_of)]
    if "outs" in past.columns:
        # `starters.normalize_counts` drops gamesStarted, so a start is read off
        # the line: three innings or more is not a relief appearance.
        past = past[past["outs"] >= 9]
    if past.empty:
        return {}
    agg = past.groupby("pitcher")["bf"].agg(["sum", "count"])
    shrunk = (agg["sum"] + ballast_starts * default) / (agg["count"] + ballast_starts)
    return {int(k): float(v) for k, v in shrunk.items()}


def lineup_slots(lineups: pd.DataFrame) -> dict:
    """{(game_pk, batter): slot} from `mlb_stats_api.fetch_lineups`."""
    if lineups is None or len(lineups) == 0:
        return {}
    return {(int(r.game_pk), int(r.batter)): int(r.slot)
            for r in lineups.itertuples(index=False)}


def lineup_sides(lineups: pd.DataFrame) -> dict:
    """{(game_pk, batter): "home"|"away"} — which club a hitter started for.

    The matchup term needs it to find the *other* club: the opposing probable
    starter, and the pen behind him.
    """
    if lineups is None or len(lineups) == 0:
        return {}
    return {(int(r.game_pk), int(r.batter)): str(r.side)
            for r in lineups.itertuples(index=False)}


def lineup_cards(lineups: pd.DataFrame) -> dict:
    """{(game_pk, side): [batter ids in batting order]} — the posted card.

    The mirror of `lineup_slots` for the pitcher's side of the matchup: a
    starter's strikeout price wants the nine hitters he is about to face.
    """
    if lineups is None or len(lineups) == 0:
        return {}
    out: dict = {}
    for r in lineups.sort_values(["game_pk", "side", "slot"]).itertuples(index=False):
        out.setdefault((int(r.game_pk), str(r.side)), []).append(int(r.batter))
    return out


def slot_pa(slot: int | None) -> float:
    return SLOT_PA.get(int(slot), DEFAULT_PA) if slot is not None else DEFAULT_PA


# ───────────────────────────── pricing a frame ─────────────────────────────

def price(closes: pd.DataFrame, batter_ctx: dict, pitcher_ctx: dict,
          slots: dict, stats=PRICEABLE, pitcher_bf: str = "fixed",
          matchup_ctx: dict | None = None) -> pd.DataFrame:
    """Our probability for every archived prop close we can price.

    One pass per game date so each date's rate tables are built once. Returns
    the input rows that were priced, with three probability columns:

        `p_model`   Marcel-with-partial rates for this player as of the date
        `p_league`  the league's own rates in the same contract — the control
                    that says how much of any skill is the *player* rather than
                    the shape of the distribution
        `p_market`  the venue's close, copied over for the scoring join

    and, when `matchup_ctx` is given, a fourth:

        `p_matchup` the same contract with the opposing pitching folded in by
                    log5 (`src/market/matchup.py`) — the probable starter over
                    his own expected innings, the opposing pen over the rest,
                    and for a pitcher's strikeout prop the opposing posted card

    `pitcher_bf` is "fixed" (every start faces `STARTER_BF`) or "own" (the
    pitcher's own batters faced per start to date, `starter_bf`).
    """
    wanted = closes[closes["prop_stat"].isin(list(stats))
                    & closes["player_id"].notna()].copy()
    if wanted.empty:
        cols = {"p_model": [], "p_league": [], "p_market": [], "exp_pa": []}
        if matchup_ctx is not None:
            cols["p_matchup"] = []
        return wanted.assign(**cols)
    lg = batter_ctx["league"]
    lg_per_pa = pa_outcome_probs(
        {f"rate_{c}": lg[f"rate_{c}"] for c in lu_model.COMPONENTS}, lg)
    lg_k = pitcher_ctx["league"]["rate_k"]

    out = []
    for as_of, day in wanted.groupby("game_date", sort=True):
        b_rates = batter_rates(batter_ctx, as_of)
        p_rates = pitcher_rates(pitcher_ctx, as_of)
        bf_lookup = starter_bf(pitcher_ctx, as_of) if pitcher_bf == "own" else {}
        mday = _matchup_day(matchup_ctx, as_of)
        per_pa_cache: dict[int, dict] = {}
        matchup_cache: dict = {}
        for row in day.itertuples(index=False):
            pid = int(row.player_id)
            extra = {}
            if row.prop_stat == "k":
                if pid not in p_rates.index:
                    continue
                rate = float(p_rates.loc[pid, "rate_k"])
                pa = bf_lookup.get(pid, STARTER_BF)
                p_model = pitcher_prop_prob("k", row.prop_line, rate, pa)
                p_league = pitcher_prop_prob("k", row.prop_line, lg_k, pa)
                if mday is not None:
                    adj = _pitcher_matchup_rate(mday, row, rate, b_rates, lg,
                                                matchup_cache)
                    extra["p_matchup"] = pitcher_prop_prob("k", row.prop_line,
                                                           adj, pa)
            else:
                slot = slots.get((int(row.game_pk), pid))
                if slot is None:            # not in the posted lineup
                    continue
                pa = slot_pa(slot)
                if pid not in per_pa_cache:
                    per_pa_cache[pid] = pa_outcome_probs(b_rates.loc[pid], lg) \
                        if pid in b_rates.index else lg_per_pa
                p_model = batter_prop_prob(row.prop_stat, row.prop_line,
                                           per_pa_cache[pid], pa)
                p_league = batter_prop_prob(row.prop_stat, row.prop_line,
                                            lg_per_pa, pa)
                if mday is not None:
                    per_pa = _batter_matchup_per_pa(mday, row, pid, b_rates, lg,
                                                    matchup_cache)
                    extra["p_matchup"] = batter_prop_prob(
                        row.prop_stat, row.prop_line, per_pa, pa)
            if mday is not None and "p_matchup" not in extra:
                extra["p_matchup"] = p_model
            out.append({**row._asdict(), "exp_pa": pa,
                        "p_model": p_model, "p_league": p_league,
                        "p_market": float(row.p_over_close), **extra})
    priced = pd.DataFrame(out)
    logger.info("priced %d/%d prop closes", len(priced), len(wanted))
    return priced


# ───────────────────── the matchup arm, one slate at a time ─────────────────

def _matchup_day(matchup_ctx: dict | None, as_of: str) -> dict | None:
    """The per-date lookups the matchup price needs, or None when it is off.

    Everything in here is cut strictly before `as_of` by `matchup.day_tables`,
    and the starter is the club's *probable*, never the man who actually threw
    the first pitch: a prop price is a pre-game price.
    """
    if matchup_ctx is None:
        return None
    from src.market import matchup as mu
    # Memoised on the context object, because the weight grid prices the same
    # slate several times and the tables do not depend on the weight.
    cache = matchup_ctx.setdefault("_day_cache", {})
    if as_of not in cache:
        cache[as_of] = mu.day_tables(matchup_ctx["ctx"], as_of)
    day = dict(matchup_ctx)
    day["tables"] = cache[as_of]
    day["as_of"] = as_of
    return day


def _batter_matchup_per_pa(mday: dict, row, pid: int, b_rates, lg: dict,
                           cache: dict) -> dict:
    """Per-PA outcome probabilities for one hitter against tonight's pitching."""
    from src.market import matchup as mu
    game_pk = int(row.game_pk)
    side = mday["sides"].get((game_pk, pid))
    opp = {"home": "away", "away": "home"}.get(side)
    key = ("bat", game_pk, opp)
    if key not in cache:
        sp = mday["probables"].get((game_pk, opp)) if opp else None
        team = mday["teams"].get((game_pk, opp)) if opp else None
        cache[key] = mu.hitter_factors(mday["tables"], mday["ctx"]["league"],
                                       sp, team)
    factors = cache[key]
    rates = b_rates.loc[pid] if pid in b_rates.index else \
        {f"rate_{c}": lg[f"rate_{c}"] for c in lu_model.COMPONENTS}
    return pa_outcome_probs(mu.matchup_rates(rates, factors, mday["weight"]), lg)


def _pitcher_matchup_rate(mday: dict, row, rate_k: float, b_rates, lg: dict,
                          cache: dict) -> float:
    """A starter's K per batter faced against the card he is about to face.

    The opposing club's posted card where one exists, its recent cards where
    it does not, and the league — a factor of exactly 1.0 — where neither
    does, which leaves the price where the current model put it.
    """
    from src.market import matchup as mu
    game_pk = int(row.game_pk)
    team = getattr(row, "team_id", None)
    team = int(team) if team is not None and pd.notna(team) else None
    # The club he pitches for is on the contract's ticker; the card he faces is
    # the other one. With no club on the row, or a game whose two clubs are not
    # both known, there is no way to tell the sides apart and the factor falls
    # through to 1.0 below — the current price, unchanged.
    sides = {s: mday["teams"].get((game_pk, s)) for s in ("home", "away")}
    opp = None
    if team is not None and all(v is not None for v in sides.values()):
        opp = next((s for s, v in sides.items() if int(v) != team), None)
    key = ("pit", game_pk, opp)
    if key not in cache:
        card = mday["cards"].get((game_pk, opp)) if opp else None
        f = mu.card_k_factor(card, b_rates, lg["rate_k"], slot_pa)
        if f is None:
            opp_team = mday["teams"].get((game_pk, opp)) if opp else None
            recent = mu.recent_card_ids(mday["club_cards"], opp_team,
                                        mday["as_of"]) if opp_team else []
            f = mu.card_k_factor(recent, b_rates, lg["rate_k"]) or 1.0
        cache[key] = f
    return float(mu.apply_factor(rate_k, cache[key], mday["weight"]))


# ───────────────────────────── scoring ─────────────────────────────

def brier(p, outcome) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(outcome, dtype=float)
    return float(np.mean((p - y) ** 2))


def brier_table(priced: pd.DataFrame,
                models=("p_model", "p_market", "p_league")) -> pd.DataFrame:
    """Brier per stat for each arm on the settled markets they all cover."""
    df = priced[priced["over_hit"].notna()].copy()
    df["y"] = df["over_hit"].astype(float)
    rows = []
    for stat, grp in df.groupby("prop_stat"):
        row = {"prop_stat": stat, "n": len(grp),
               "games": grp["game_pk"].nunique(),
               "players": grp["player_id"].nunique(),
               "over_rate": float(grp["y"].mean())}
        for m in models:
            row[m] = round(brier(grp[m], grp["y"]), 5)
        rows.append(row)
    total = {"prop_stat": "all", "n": len(df), "games": df["game_pk"].nunique(),
             "players": df["player_id"].nunique(),
             "over_rate": float(df["y"].mean())}
    for m in models:
        total[m] = round(brier(df[m], df["y"]), 5)
    return pd.DataFrame(rows + [total])


def paired_brier(priced: pd.DataFrame, a: str, b: str,
                 group_col: str = "game_pk") -> dict:
    """Mean per-contract Brier difference `a − b`, with a clustered SE.

    Two arms priced on the same contracts are paired by construction — the
    same afternoon, the same hitter, the same line — so the difference of the
    two Brier scores is a per-contract quantity and its mean is the honest
    comparison. The standard error clusters on the game for the same reason
    the money bootstrap does: a hitter's 1+, 2+ and 3+ hits are one
    afternoon's at bats, and treating them as three independent observations
    would report an error bar the data does not support.

    Negative `diff` means `a` is the better price.
    """
    df = priced[priced["over_hit"].notna()]
    if len(df) == 0:
        return {"n": 0, "diff": float("nan"), "se": float("nan"),
                "t": float("nan")}
    y = df["over_hit"].astype(float).to_numpy()
    d = (df[a].to_numpy(dtype=float) - y) ** 2 - (df[b].to_numpy(dtype=float) - y) ** 2
    n = len(d)
    mean = float(d.mean())
    resid = d - mean
    groups = df[group_col].to_numpy() if group_col in df else np.arange(n)
    sums = pd.Series(resid).groupby(pd.Series(groups).to_numpy()).sum().to_numpy()
    se = float(np.sqrt((sums ** 2).sum())) / n if n else float("nan")
    return {"n": n, "diff": mean, "se": se,
            "t": mean / se if se > 0 else float("nan")}


def to_pnl_frame(priced: pd.DataFrame) -> pd.DataFrame:
    """Rename the prop columns onto the shapes `src/market/pnl.py` expects.

    The P&L module speaks in `p_home_close` / `home_win` because it was
    written for moneylines; a prop is the same binary contract with "the over
    hit" in place of "the home team won", so the exam is the same code rather
    than a second copy of it.
    """
    df = priced[priced["over_hit"].notna()].copy()
    df["date"] = df["game_date"]
    df["p_home_close"] = df["p_over_close"].astype(float)
    df["home_win"] = df["over_hit"].astype(bool)
    df["model"] = df["p_model"]
    df["league"] = df["p_league"]
    if "p_matchup" in df.columns:
        df["matchup"] = df["p_matchup"]
    return df.sort_values(["date", "game_pk", "prop_stat", "prop_line"]).reset_index(drop=True)


def to_maker_frame(priced: pd.DataFrame) -> pd.DataFrame:
    """`to_pnl_frame` plus the two columns a resting order needs.

    `market_id` says which market's hourly candles are this contract's price
    path, and `first_pitch_ts` is when the order is cancelled. Both come off
    the archive row rather than being inferred, so a contract with no game
    start on file simply cannot be quoted and is dropped — an order that
    cannot be cancelled at a known first pitch is not a pre-game order.
    """
    df = to_pnl_frame(priced)
    if "market_id" not in df.columns or "game_start" not in df.columns:
        raise ValueError("priced frame carries no market_id / game_start")
    df = df[df["game_start"].notna()].copy()
    df["first_pitch_ts"] = [int(pd.Timestamp(str(s)).timestamp())
                            for s in df["game_start"]]
    return df.reset_index(drop=True)
