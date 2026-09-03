"""The other half of the plate appearance: a log5 matchup term for props.

`docs/props-exam-2026.md` ends on a named omission:

> **The opposing pitcher is not in the hitter's price, and the opposing lineup
> is not in the pitcher's.** A prop is a matchup and this model prices only one
> side of it. That is the largest single omission, and it is most of what the
> market knows that we do not.

This module is that term. It is not a new estimator and deliberately so —
`docs/methods.md` §4 says log5 is *structure we already know*, not a hypothesis
to be learned, and §7 asks whether the current model has been given everything
it could use before reaching for a more powerful one. It has not been given the
pitcher. So the arithmetic here is the identity, applied per component:

    rate_matchup = rate_hitter · rate_pitcher / rate_league

with two refinements the exam's own caveats ask for.

**The pitcher is the *probable* starter, never the actual one.** A prop price
is a pre-game price, and who eventually took the mound is a fact about how the
night went. `fetch_probables` is the pre-game announcement field, the same one
station E reads, and it is the only starter this module will look at.

**A starter does not face the whole game.** He covers `expected_starter_ip` of
nine innings — the same estimate station E uses as its workload split — and the
bullpen covers the rest, so the matchup factor a hitter sees is

    share · f(starter) + (1 − share) · f(bullpen)

where `f` is a pitcher's component rate over the league's, and the bullpen's
rate is the opposing club's own relief staff to date. A hitter who draws an ace
for five innings and a league-average pen for the other four is not priced as
though he faced the ace all night.

**Pitcher strikeout props get the mirror.** The opposing club's *posted* card,
plate-appearance weighted by slot, where a card exists; the club's recent cards
where it does not; the league where neither does.

**One estimator, both ends.** A pitcher's rates *allowed* are computed by
`src/sim/lineups.marcel_rates` — the hitter estimator — pointed at the other id
on the plate appearance, in exactly the hitter columns (K and BB+HBP and HR per
PA, extra bases per AB, hits per ball in play). That costs no new fetch: the
doubles and triples allowed were already in the cached responses. It also means
a change to the rate machinery moves both sides at once, which is the same
argument `src/sim/starters.py` makes for calling station A's provider.

**The strength of the term is a free parameter and is chosen out of sample.**
`WEIGHT` scales the adjustment, `1.0` being the identity as written and `0.0`
being the current price:

    rate = rate_hitter · (1 + w · (f − 1))

It is chosen on the first half of the archive by date and scored on the second
(`scripts/props_exam.py --matchup-weights`), which is the same discipline the
maker exam's margin goes through.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.sim import lineups as lu_model
from src.sim import starters as sp_model

logger = logging.getLogger(__name__)

# The five components the hitter side works in. A pitcher has a rate for each
# of them *allowed*, in the same units, which is what makes the log5 arithmetic
# a one-liner per component.
COMPONENTS = lu_model.COMPONENTS

# Regression ballast for a pitcher's rates allowed, in the denominator's own
# units (PA for the first three, AB for ISO, balls in play for BABIP). The
# first three and BABIP are the published split-half stabilization points
# `src/eval/pitchers.STABILIZATION` carries, doubled by the same projection
# multiplier station E chose walk-forward on 2025 (reliability is not
# projection: talent moves between samples).
#
# ISO allowed has no published stabilization point of its own. It is given the
# home-run point, which is the heaviest on the board, because a pitcher's
# extra-base suppression is the same DIPS story as his home-run rate only
# noisier — most of ISO allowed is the park and the defence behind him. Nothing
# here is fitted to the games this term is scored on.
BALLAST_ALLOWED = {
    "k": 140.0,        # 70 BF × 2
    "bbhbp": 340.0,    # 170 BF × 2
    "hr": 2600.0,      # 1300 BF × 2
    "iso": 2600.0,     # the home-run point, for want of a published one
    "babip": 4000.0,   # 2000 BIP × 2
}
# Ballast, in relief batters faced, on a club's bullpen rates. A pen throws
# 3.5 innings a night and turns over constantly, so it is a staff rather than a
# pitcher; the same component ballasts apply to the pooled staff line.
BALLAST_PEN = BALLAST_ALLOWED

# How hard the matchup term pulls. 1.0 is log5 exactly; 0.0 is the current
# price. The default is the identity, and `props_exam.py` chooses it on the
# first half of the window before scoring the second.
WEIGHT = 1.0
# A component rate is a probability and stays one. The clip never binds on real
# rates; it exists so a pathological factor cannot hand `event_rates` a rate
# outside its domain.
RATE_FLOOR, RATE_CEIL = 1e-6, 0.999
GAME_IP = sp_model.GAME_IP

COUNT_COLS = ["pa", "ab", "bip", "k", "bbhbp", "hr", "xb", "hip",
              "doubles", "triples", "sf"]


# ───────────────────────────── the identity ─────────────────────────────

def log5(hitter_rate, pitcher_rate, league_rate):
    """`hitter × pitcher ÷ league` — the matchup rate for one component.

    Bill James' log5 in its rate form: a league-average pitcher (`pitcher =
    league`) returns the hitter unchanged, a league-average hitter returns the
    pitcher, and a pitcher who allows half the league's rate halves the
    hitter's. It is an identity we assert rather than a model we fit
    (docs/methods.md §4).
    """
    lg = np.asarray(league_rate, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(hitter_rate, dtype=float) * np.asarray(pitcher_rate, dtype=float) / lg
    return np.where(np.isfinite(out) & (lg > 0), out, np.asarray(hitter_rate, dtype=float))


def apply_factor(hitter_rate, factor, weight: float = WEIGHT):
    """The hitter's rate moved `weight` of the way to its log5 value.

    `factor` is the opposing pitching's rate over the league's, so
    `weight = 1` is `log5` exactly and `weight = 0` leaves the current price
    alone. The weight is the module's one free parameter and is chosen on data
    the score never sees.
    """
    f = 1.0 + float(weight) * (np.asarray(factor, dtype=float) - 1.0)
    return np.clip(np.asarray(hitter_rate, dtype=float) * f, RATE_FLOOR, RATE_CEIL)


def blend_factor(sp_factor, pen_factor, starter_share: float):
    """One factor for the whole game: the starter's share of it, then the pen's.

    `starter_share` is `expected_starter_ip / 9`, the same workload split
    station E prices relief innings with. A hitter's plate appearances are
    assumed to be distributed like innings, which is what makes this the share
    of *his* PA the starter covers.
    """
    s = float(np.clip(starter_share, 0.0, 1.0))
    return s * float(sp_factor) + (1.0 - s) * float(pen_factor)


def factors_from_rates(pitcher_rates, league: dict,
                       components=COMPONENTS) -> dict:
    """{component: pitcher rate / league rate}, 1.0 where the rate is missing."""
    out = {}
    for c in components:
        lg = float(league.get(f"rate_{c}", 0.0) or 0.0)
        if pitcher_rates is None or lg <= 0:
            out[c] = 1.0
            continue
        try:
            v = float(pitcher_rates[f"rate_{c}"])
        except (KeyError, TypeError, ValueError):
            out[c] = 1.0
            continue
        out[c] = v / lg if np.isfinite(v) else 1.0
    return out


def matchup_rates(hitter_rates, factors: dict, weight: float = WEIGHT) -> dict:
    """A hitter's five component rates with the opposing pitching folded in."""
    return {f"rate_{c}": float(apply_factor(float(hitter_rates[f"rate_{c}"]),
                                            factors.get(c, 1.0), weight))
            for c in COMPONENTS}


# ─────────────────── a pitcher's rates, in the hitter's columns ───────────────

def allowed_counts(logs: pd.DataFrame) -> pd.DataFrame:
    """Raw pitching lines → the hitter-side count schema, keyed by pitcher.

    A batter faced *is* a plate appearance seen from the mound, so the counts
    a pitcher allows live in exactly the columns `lineups.normalize_counts`
    produces. Relabelling and reusing that function rather than writing a
    second one is what keeps the two ends of the plate appearance in the same
    units: whatever ISO means for a hitter, it means the same thing allowed.

    `date`, `team`, `gs` and `outs` ride along when the frame is a game log, so
    the caller can cut by date, split relief from starts, and read a start's
    length off the same rows.
    """
    if logs is None or len(logs) == 0:
        return pd.DataFrame(columns=["pitcher", "season", *COUNT_COLS])
    frame = logs.rename(columns={"bf": "pa"}).copy()
    frame["batter"] = pd.to_numeric(frame["pitcher"], errors="coerce").astype("int64")
    out = lu_model.normalize_counts(frame).rename(columns={"batter": "pitcher"})
    for extra in ("date", "team", "gs", "outs", "game_pk"):
        if extra in logs.columns:
            out[extra] = logs[extra].to_numpy()
    return out


def league_allowed(counts: pd.DataFrame) -> dict:
    """League rates from the pitching side — the denominator of every factor.

    Computed off the same counts the numerator is, so a league-average pitcher
    has a factor of exactly 1.0 and the matchup term is a no-op on him.
    """
    return lu_model.league_rates(counts)


def counts_before(counts: pd.DataFrame, as_of: str,
                  key: str = "pitcher") -> pd.DataFrame:
    """Season-to-date counts per `key` from games *strictly before* `as_of`.

    The leakage guard, and the same one `lineups.games_before` applies to the
    hitter: tonight's line is what the price is predicting, so it can never be
    an input to it.
    """
    cols = [key, "season", *COUNT_COLS]
    if counts is None or len(counts) == 0 or "date" not in counts.columns:
        return pd.DataFrame(columns=cols)
    past = counts[counts["date"].astype(str) < str(as_of)]
    if past.empty:
        return pd.DataFrame(columns=cols)
    return past.groupby([key, "season"], as_index=False)[COUNT_COLS].sum()


def allowed_rates(counts: pd.DataFrame, as_of_season: int, league: dict,
                  ballast=BALLAST_ALLOWED) -> pd.DataFrame:
    """Marcel-weighted, league-regressed rates *allowed*, one row per pitcher.

    `lineups.marcel_rates` under a different index: same 5/4/3 recency weights,
    same closed-form regression, pitcher ballasts instead of hitter ones.
    """
    if counts is None or len(counts) == 0:
        return pd.DataFrame(columns=["pa_weighted", *lu_model.RATE_COLS],
                            index=pd.Index([], name="pitcher", dtype="int64"))
    out = lu_model.marcel_rates(counts.rename(columns={"pitcher": "batter"}),
                                as_of_season, league, ballast=ballast)
    out.index.name = "pitcher"
    return out


def relief_rows(counts: pd.DataFrame) -> pd.DataFrame:
    """The appearances a pitcher made *out of the pen*.

    A start says nothing about a relief inning and vice versa
    (`bullpen.relief_appearances` makes the same split on the innings column),
    and the bullpen half of the matchup factor is about relief innings.
    """
    if counts is None or len(counts) == 0 or "gs" not in counts.columns:
        return counts if counts is not None else pd.DataFrame()
    gs = pd.to_numeric(counts["gs"], errors="coerce").fillna(0.0)
    return counts[gs < 1].copy()


def pen_rates(relief: pd.DataFrame, as_of: str, league: dict,
              ballast=BALLAST_PEN) -> dict:
    """{team_id: {rate_*}} for each club's relief staff, strictly before `as_of`.

    A pen is a staff, not a pitcher: who is available tonight turns over
    constantly and the man who throws the seventh is not knowable pre-game, so
    the honest unit is the club's own relief line to date, regressed to the
    league with the same component ballasts. A club with no relief innings on
    file is absent and the caller falls back to the league, which is a factor
    of exactly 1.0 — the current price.
    """
    if relief is None or len(relief) == 0 or "team" not in relief.columns:
        return {}
    past = relief[relief["date"].astype(str) < str(as_of)] \
        if "date" in relief.columns else relief
    past = past[past["team"].notna()]
    if past.empty:
        return {}
    agg = past.groupby(past["team"].astype("int64"))[COUNT_COLS].sum()
    bal = ballast if isinstance(ballast, dict) else {c: float(ballast) for c in COMPONENTS}
    out = {}
    for team, row in agg.iterrows():
        rates = {}
        for c in COMPONENTS:
            num = float(row[lu_model.RATE_NUM[c]])
            den = float(row[lu_model.RATE_DEN[c]])
            b = float(bal[c])
            rates[f"rate_{c}"] = (num + b * float(league[f"rate_{c}"])) / (den + b)
        out[int(team)] = rates
    return out


# ───────────────────────── the pitcher's side of it ─────────────────────────

def card_k_factor(batter_ids, batter_rates: pd.DataFrame, lg_k: float,
                  slot_pa=None) -> float | None:
    """A posted card's strikeout rate over the league's, weighted by slot PA.

    `batter_ids` is the card in batting order. A hitter with no rates on file
    is priced at the league's, which is what the rest of the exam does with
    him. Returns None for an empty card so the caller can fall back.
    """
    ids = [int(b) for b in (batter_ids or [])]
    if not ids or lg_k <= 0:
        return None
    num = den = 0.0
    for i, pid in enumerate(ids, start=1):
        w = float(slot_pa(i)) if slot_pa is not None else 1.0
        r = float(batter_rates.loc[pid, "rate_k"]) \
            if batter_rates is not None and pid in batter_rates.index else float(lg_k)
        num += w * r
        den += w
    return (num / den) / float(lg_k) if den > 0 else None


def recent_card_ids(cards: dict, team: int, as_of: str, games: int = 15) -> list:
    """Batters on a club's most recent posted cards before `as_of`.

    The fallback for a start whose opposing card is not in the archive: the
    club's own recent lineups, pooled. Strictly before the date, so a card
    posted for tonight's game is never one of them.
    """
    rows = [(d, ids) for (t, d), ids in cards.items()
            if int(t) == int(team) and str(d) < str(as_of)]
    if not rows:
        return []
    rows.sort(key=lambda r: r[0], reverse=True)
    out = []
    for _, ids in rows[:games]:
        out.extend(ids)
    return out


# ───────────────────────────── as-of-date assembly ─────────────────────────

def inputs(season: int, prior_seasons: int = 2, refresh: bool = False,
           workers: int = 8) -> dict:
    """Everything the matchup term needs, fetched once.

    The same shape as `props.batter_inputs` and `starters.rate_inputs`: prior
    completed seasons for the Marcel weighting, this season's dated game logs
    so a date cut can be applied per slate, and the league's own rates allowed.
    Every pitcher who threw a pitch in the season is included, because the
    bullpen half of the term is a club's whole relief staff and not just the
    men whose names are on a prop.
    """
    from src.data.mlb_stats_api import (fetch_pitcher_game_logs,
                                        fetch_season_pitching)

    prior = pd.concat([fetch_season_pitching(y)
                       for y in range(season - prior_seasons, season)],
                      ignore_index=True)
    current = fetch_season_pitching(season, refresh=refresh)
    ids = sorted({int(p) for p in current["pitcher"]})
    logs = fetch_pitcher_game_logs(ids, season, refresh=refresh, workers=workers)
    logs = logs[logs["game_type"] == "R"]
    prior_counts = allowed_counts(prior)
    game_logs = allowed_counts(logs)
    starts = sp_model.start_innings(logs)
    return {"season": season, "prior_counts": prior_counts,
            "game_logs": game_logs, "starts": starts,
            "league": league_allowed(prior_counts)}


def day_tables(ctx: dict, as_of: str, ballast=BALLAST_ALLOWED) -> dict:
    """The three lookups one slate needs, all cut strictly before `as_of`.

    `sp` — rates allowed per pitcher; `pen` — rates allowed per club's relief
    staff; `ip` — expected innings per start, station E's workload split.
    """
    current = counts_before(ctx["game_logs"], as_of)
    counts = pd.concat([ctx["prior_counts"][["pitcher", "season", *COUNT_COLS]],
                        current], ignore_index=True)
    return {
        "sp": allowed_rates(counts, ctx["season"], ctx["league"], ballast=ballast),
        "pen": pen_rates(relief_rows(ctx["game_logs"]), as_of, ctx["league"]),
        "ip": sp_model.expected_starter_ip(ctx["starts"], as_of),
    }


def hitter_factors(tables: dict, league: dict, starter_id, pen_team,
                   components=COMPONENTS) -> dict:
    """The blended {component: factor} one hitter faces tonight.

    The probable starter over his own expected innings, the opposing club's
    pen over the rest. An unknown starter is the league (factor 1.0), which
    leaves the price exactly where the current model put it.
    """
    sp_rates = None
    if starter_id is not None and starter_id in tables["sp"].index:
        sp_rates = tables["sp"].loc[int(starter_id)]
    sp_f = factors_from_rates(sp_rates, league, components)
    pen_f = factors_from_rates(tables["pen"].get(int(pen_team)) if pen_team is not None
                               else None, league, components)
    ip = tables["ip"].get(int(starter_id), sp_model.STARTER_IP) \
        if starter_id is not None else sp_model.STARTER_IP
    share = float(ip) / GAME_IP
    return {c: blend_factor(sp_f[c], pen_f[c], share) for c in components}
