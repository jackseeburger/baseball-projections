"""Posted-lineup batting quality for station E (per-game P(win)).

`starters.py` gave the per-game engine the pitcher on the mound; this module
gives it the nine hitters behind the plate. Both are aimed at the same 0.0046
Brier the market held over regressed run differential
(docs/market-benchmark-2026.md), and both take the same shape: a *delta* from
a baseline applied to the team's own regressed run rate, never an absolute
level that would quietly re-regress the team.

The chain, all pure functions over DataFrames so it unit-tests without a
network:

    season / game-log counts ─► marcel_rates()   Marcel-weighted, league-
                                                 regressed K, BB+HBP, HR per
                                                 PA, ISO per AB, BABIP per BIP
                             ─► event_rates()    those five rates → the per-PA
                                                 probability of each plate-
                                                 appearance outcome
                             ─► runs_per_pa()    linear weights → runs above
                                                 an average PA
                             ─► lineup_r9()      nine batters, weighted by the
                                                 plate appearances their slot
                                                 gets → runs per nine innings
                             ─► blend_lineup_team()  the team's own runs-scored
                                                 rate, moved by how far this
                                                 lineup sits from its baseline

The output is a runs-scored-per-9 number that drops straight into the existing
Pythagenpat → log5 → HFA pipeline in place of the team's RS/G, alongside the
starter's runs-allowed number. `scripts/backtest_game_odds.py` scores it as
`pythag_60_sp_lu`.

Provenance of every constant, in the same discipline `starters.py` uses:
Marcel's published 5/4/3 recency weights; published rate-stabilization points
for the five components; the same 2x projection multiplier on those points;
standard linear weights for the run values of plate-appearance outcomes; and
plate appearances per lineup slot from the structure of a batting order. The
only free knobs — how much of the lineup delta to apply, and what to measure
it against — were chosen walk-forward on **2025 only**. Nothing here is fit to
the 2026 games this model is scored on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Marcel's recency weights: current season, one back, two back. Same as
# starters.py, and the same normalisation (most recent weight = 1) so that
# ballasts stay denominated in real plate appearances.
MARCEL_WEIGHTS = (5.0, 4.0, 3.0)

# Each rate's numerator and denominator. Denominators differ on purpose: ISO
# is an at-bat rate and BABIP a balls-in-play rate, and regressing either in
# PA units would misstate how much sample a hitter actually has.
RATE_NUM = {"k": "k", "bbhbp": "bbhbp", "hr": "hr", "iso": "xb", "babip": "hip"}
RATE_DEN = {"k": "pa", "bbhbp": "pa", "hr": "pa", "iso": "ab", "babip": "bip"}
COMPONENTS = ("k", "bbhbp", "hr", "iso", "babip")
RATE_COLS = [f"rate_{c}" for c in COMPONENTS]

# Published stabilization points for hitter rate stats — the sample at which
# split-half correlation reaches 0.5, which is exactly a regression ballast.
# The spread is the whole point: a hitter's strikeout rate is close to real
# after two weeks, his BABIP is mostly noise for five seasons.
STABILIZATION = {"k": 60.0, "bbhbp": 120.0, "hr": 170.0,
                 "iso": 160.0, "babip": 820.0}
# Those points measure reliability (how much of *this* sample is signal).
# Projecting the *next* game also has to absorb real talent change between
# samples, so projection systems regress harder than reliability alone implies;
# roughly double is the conventional rule of thumb and is what starters.py
# uses. Walk-forward on 2025 agrees and is flat above it. Chosen on 2025 only.
PROJECTION_MULTIPLIER = 2.0
BALLAST = {c: v * PROJECTION_MULTIPLIER for c, v in STABILIZATION.items()}

# Linear weights: runs above an average plate appearance, per outcome. These
# are the standard run values from base-out run expectancy (Tango et al.);
# they move by a couple of hundredths with the run environment, which does not
# matter here because `runs_per_pa` only ever uses *differences* from the
# league, so any common additive shift in the weights cancels exactly.
LINEAR_WEIGHTS = {"bbhbp": 0.33, "k": -0.28, "hr": 1.40,
                  "b1": 0.47, "b2": 0.78, "b3": 1.09, "out": -0.25}

# Plate appearances a team gets in nine innings. A structural number (it is
# 3 outs x 9 innings plus baserunners), used only to turn runs-per-PA into
# runs-per-game; the caller can pass the league's measured value instead.
PA_PER_GAME = 38.0

# A club's lineup baseline is its own recent cards, and these two constants say
# which ones and how hard to shrink them toward league average. A window rather
# than the whole season because the question the baseline answers is "who does
# this club run out *now*", and rosters turn over; no ballast because the club
# mean is what the term is supposed to be measured against — shrinking it toward
# league average smuggles the club's absolute offensive level back into a term
# that is meant to carry only the day's news. Both chosen walk-forward on 2025
# only, where the surface is flat: 15 games unregressed is the best of the
# {15, 40, 200} x {0, 5, 20} grid by 0.00005 Brier, and the whole grid spans
# 0.0008.
BASELINE_BALLAST_GAMES = 0.0
BASELINE_WINDOW_GAMES = 15

# How much of the card's distance from that baseline to apply to the club's
# runs scored, and what the distance is measured against ("team" is the club's
# own recent cards, "league" is league average). Both chosen walk-forward on
# 2025 only, where the curve is flat for any weight from 0.25 to 0.75
# (docs/market-benchmark-2026.md). They live here rather than in a caller so
# the harness and the live odds job cannot pick different ones.
WEIGHT = 0.5
BASELINE = "team"

LINEUP_SLOTS = 9

EVENTS = ("bbhbp", "k", "hr", "b1", "d23", "out")


def normalize_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Hitter count frame → the columns this module works in.

    Accepts the Stats API column names (`pa, ab, h, doubles, triples, hr, k,
    bb, hbp, sf`) and returns `batter, season` plus the numerators and
    denominators the five rates need:

        pa, ab, bip, k, bbhbp, hr, xb, hip, doubles, triples, sf

    where `bip = ab - k - hr + sf` (balls in play), `xb = 2B + 2*3B + 3*HR`
    (extra bases, ISO's numerator) and `hip = h - hr` (hits in play, BABIP's
    numerator). Walks and hit batsmen are folded together because they are the
    same event for a run estimator.
    """
    def col(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(0.0, index=df.index, dtype=float)
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0).astype(float)

    ab, k, hr, sf = col("ab"), col("k"), col("hr"), col("sf")
    doubles, triples = col("doubles"), col("triples")
    out = pd.DataFrame({
        "batter": df["batter"].astype("int64"),
        "season": df["season"].astype("int64"),
        "pa": col("pa"), "ab": ab,
        "bip": (ab - k - hr + sf).clip(lower=0.0),
        "k": k, "bbhbp": col("bb") + col("hbp"), "hr": hr,
        "xb": doubles + 2.0 * triples + 3.0 * hr,
        "hip": (col("h") - hr).clip(lower=0.0),
        "doubles": doubles, "triples": triples, "sf": sf,
    })
    return out[["batter", "season", "pa", "ab", "bip", "k", "bbhbp", "hr",
                "xb", "hip", "doubles", "triples", "sf"]]


def league_rates(counts: pd.DataFrame) -> dict:
    """Pooled league rates plus the three shares `event_rates` needs.

    Keys: rate_k, rate_bbhbp, rate_hr, rate_iso, rate_babip, and

      * `nonab_share`   — PA that are neither an at-bat nor a walk/HBP
                          (sacrifices, catcher's interference), per PA
      * `sf_share`      — sacrifice flies per PA (they are in BIP but not AB)
      * `triple_share`  — 3B / (2B + 3B), how extra-base hits in play split

    Those three are what make `event_rates` reproduce the league exactly when
    it is handed the league's own rates, which is what lets the runs estimator
    be centred with no fitted constant.
    """
    tot = {c: float(counts[c].sum()) for c in
           ("pa", "ab", "bip", "k", "bbhbp", "hr", "xb", "hip",
            "doubles", "triples", "sf")}
    if tot["pa"] <= 0 or tot["ab"] <= 0 or tot["bip"] <= 0:
        raise ValueError("league_rates: empty hitting counts frame")
    lg = {f"rate_{c}": tot[RATE_NUM[c]] / tot[RATE_DEN[c]] for c in COMPONENTS}
    lg["nonab_share"] = (tot["pa"] - tot["ab"] - tot["bbhbp"]) / tot["pa"]
    lg["sf_share"] = tot["sf"] / tot["pa"]
    d3 = tot["doubles"] + tot["triples"]
    lg["triple_share"] = tot["triples"] / d3 if d3 > 0 else 0.0
    return lg


def games_before(game_logs: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Season-to-date counts per batter from games *strictly before* `as_of`.

    This is the leakage guard: a game on the date being predicted (the game
    itself, or the first half of a doubleheader that has not been played when
    the line is priced) never contributes to the rates used to predict it.
    """
    cols = ["batter", "season", "pa", "ab", "bip", "k", "bbhbp", "hr",
            "xb", "hip", "doubles", "triples", "sf"]
    if game_logs.empty:
        return pd.DataFrame(columns=cols)
    past = game_logs[game_logs["date"].astype(str) < str(as_of)]
    if past.empty:
        return pd.DataFrame(columns=cols)
    return past.groupby(["batter", "season"], as_index=False)[cols[2:]].sum()


def _ballast_map(ballast) -> dict:
    """A scalar applies to every component; a mapping is used as given."""
    if isinstance(ballast, dict):
        return {c: float(ballast[c]) for c in COMPONENTS}
    return {c: float(ballast) for c in COMPONENTS}


def marcel_rates(counts: pd.DataFrame, as_of_season: int, lg: dict,
                 weights: tuple = MARCEL_WEIGHTS,
                 ballast=BALLAST) -> pd.DataFrame:
    """Per-batter K, BB+HBP, HR (per PA), ISO (per AB) and BABIP (per BIP).

    `counts` holds one row per batter-season (normalize_counts schema) and may
    mix the partial current season with completed prior ones. Seasons are
    weighted by recency — `weights[i]` for `as_of_season - i` — normalised so
    the most recent weight is 1, so `ballast` stays in real plate appearances
    (or at-bats, or balls in play). `ballast` is either one number for every
    component or a {component: sample} mapping (the default, `BALLAST`).

    Returns a frame indexed by batter with the five rate columns and
    `pa_weighted` (the effective sample). A batter with no history is simply
    absent — `batter_runs_lookup` gives those league average.
    """
    w = {as_of_season - i: weights[i] / weights[0] for i in range(len(weights))}
    used = counts[counts["season"].isin(w)].copy()
    num_den = sorted(set(RATE_NUM.values()) | set(RATE_DEN.values()))
    if used.empty:
        return pd.DataFrame(columns=["pa_weighted", *RATE_COLS],
                            index=pd.Index([], name="batter", dtype="int64"))
    ws = used["season"].map(w).astype(float)
    for col in num_den:
        used[col] = used[col].astype(float) * ws
    agg = used.groupby("batter")[num_den].sum()
    bal = _ballast_map(ballast)
    out = pd.DataFrame(index=agg.index)
    out["pa_weighted"] = agg["pa"]
    for c in COMPONENTS:
        num, den = agg[RATE_NUM[c]], agg[RATE_DEN[c]]
        out[f"rate_{c}"] = ((num + bal[c] * lg[f"rate_{c}"]) / (den + bal[c]))
    return out


def event_rates(rates, lg: dict):
    """Five component rates → the per-PA probability of each outcome.

    Outcomes: `bbhbp`, `k`, `hr`, `b1` (single), `d23` (double or triple, kept
    together because ISO cannot separate them) and `out` (everything else —
    outs on balls in play, sacrifices, interference). They sum to 1 by
    construction, which is what makes the linear-weights sum in `runs_per_pa`
    invariant to any common shift in the weights.

    Handed the league's own rates this reproduces the league's own event
    frequencies exactly, so the runs estimator needs no fitted constant.

    Accepts a DataFrame of rates (returns a DataFrame) or a mapping of scalars
    (returns a dict).
    """
    scalar = not isinstance(rates, pd.DataFrame)
    r = {c: float(rates[f"rate_{c}"]) if scalar else rates[f"rate_{c}"]
         for c in COMPONENTS}
    clip = (lambda s, lo=None, hi=None: float(np.clip(s, lo if lo is not None else -np.inf,
                                                      hi if hi is not None else np.inf))) \
        if scalar else (lambda s, lo=None, hi=None: s.clip(lower=lo, upper=hi))

    ab_pa = clip(1.0 - r["bbhbp"] - lg["nonab_share"], 0.0)
    bip_pa = clip(ab_pa - r["k"] - r["hr"] + lg["sf_share"], 0.0)
    hip_pa = r["babip"] * bip_pa                     # singles + doubles + triples
    xb_nonhr = clip(r["iso"] * ab_pa - 3.0 * r["hr"], 0.0)   # 2B + 2*3B
    d23 = clip(xb_nonhr / (1.0 + lg["triple_share"]), 0.0)
    d23 = clip(d23, None, hip_pa)
    ev = {"bbhbp": r["bbhbp"], "k": r["k"], "hr": r["hr"],
          "b1": clip(hip_pa - d23, 0.0), "d23": d23}
    ev["out"] = 1.0 - ev["bbhbp"] - ev["k"] - ev["hr"] - hip_pa
    return ev if scalar else pd.DataFrame(ev, index=rates.index)


def _event_weights(lg: dict) -> dict:
    """Linear weights per outcome, with 2B and 3B blended at the league mix."""
    ts = float(lg["triple_share"])
    w = {e: LINEAR_WEIGHTS[e] for e in ("bbhbp", "k", "hr", "out")}
    w["b1"] = LINEAR_WEIGHTS["b1"]
    w["d23"] = (1.0 - ts) * LINEAR_WEIGHTS["b2"] + ts * LINEAR_WEIGHTS["b3"]
    return w


def runs_per_pa(rates, lg: dict):
    """Runs above an average plate appearance, from the five component rates.

    Linear weights on the event probabilities, centred on the league: a batter
    whose rates equal the league's scores exactly 0, so the lineup term only
    ever *re-allocates* runs around the team baseline and can never shift the
    league's run environment.

    Scalar mapping in → float out; DataFrame in → Series out.
    """
    lg_ev = event_rates({f"rate_{c}": lg[f"rate_{c}"] for c in COMPONENTS}, lg)
    ev = event_rates(rates, lg)
    w = _event_weights(lg)
    if isinstance(ev, dict):
        return float(sum(w[e] * (ev[e] - lg_ev[e]) for e in EVENTS))
    total = sum(w[e] * (ev[e] - lg_ev[e]) for e in EVENTS)
    return pd.Series(total, index=ev.index, name="runs_per_pa").astype(float)


def batter_runs_lookup(rates: pd.DataFrame, lg: dict) -> dict:
    """{batter_id: runs above average per PA}. Missing ids default to 0.0."""
    if rates.empty:
        return {}
    return {int(k): float(v) for k, v in runs_per_pa(rates, lg).items()}


def slot_pa_shares(pa_per_game: float = PA_PER_GAME,
                   slots: int = LINEUP_SLOTS) -> np.ndarray:
    """Share of a team's plate appearances that each batting-order slot gets.

    Structural, not fitted: with `T` plate appearances in a game the batter in
    slot `i` comes up at positions i, i+9, i+18 ..., so he gets about
    `(T - i)/9 + 1` of them. Normalised to sum to 1. At the modern T ~ 38 that
    is a 1.21:1 spread between leadoff and the ninth hitter, which is what the
    league actually shows. It matters because managers put their best hitters
    at the top, so equal weighting would understate a good lineup.
    """
    i = np.arange(1, slots + 1, dtype=float)
    raw = (float(pa_per_game) - i) / slots + 1.0
    raw = np.clip(raw, 0.0, None)
    return raw / raw.sum()


def lineup_r9(batter_ids, runs_lookup: dict, lg_r9: float,
              pa_per_game: float = PA_PER_GAME,
              slot_shares: np.ndarray | None = None) -> float:
    """Runs per nine innings this posted lineup is expected to score.

    The nine batters' runs-above-average-per-PA, weighted by the plate
    appearances their slot gets, scaled up to a game and added to the league's
    runs per game. A lineup of nine exactly league-average batters returns
    `lg_r9` exactly. Batters with no history are absent from `runs_lookup` and
    contribute 0 — i.e. they are treated as league average.
    """
    ids = list(batter_ids)
    if not ids:
        return float(lg_r9)
    shares = slot_pa_shares(pa_per_game, len(ids)) if slot_shares is None \
        else np.asarray(slot_shares, dtype=float)[:len(ids)]
    shares = shares / shares.sum()
    raa = np.array([runs_lookup.get(int(b), 0.0) for b in ids], dtype=float)
    return float(lg_r9 + float(pa_per_game) * float(np.dot(shares, raa)))


def team_lineup_baseline(prior_r9, lg_r9: float,
                         ballast_games: float = BASELINE_BALLAST_GAMES) -> float:
    """What this club's *typical* posted lineup is worth, regressed to league.

    `prior_r9` is the sequence of `lineup_r9` values for the team's recent
    games (strictly before the date being predicted), **re-scored with the
    same day's rates as today's lineup**. That matters: batter rates start the
    season heavily regressed and spread out as samples grow, so a baseline
    accumulated from each game's own contemporaneous estimate is compressed
    toward zero relative to today's, and the difference quietly leaks the
    league-average form back in. Scoring the same nine-man cards on one
    common yardstick removes that drift.

    The mean is shrunk toward `lg_r9` with `ballast_games` of league-average
    ballast, so a club with no history yet is simply league average.

    This exists because a lineup's distance from *league* average is largely
    the same information the team's own runs-scored rate already carries —
    adding it whole would count a good offence twice. Measured against the
    club's own baseline, the term is the news: who is resting, who is hurt,
    who just came up.
    """
    prior = np.asarray(list(prior_r9), dtype=float)
    den = prior.size + float(ballast_games)
    if den <= 0:                      # no history and no ballast
        return float(lg_r9)
    return (float(prior.sum()) + float(ballast_games) * float(lg_r9)) / den


def blend_lineup_team(lineup_r9_value, team_rs9, baseline_r9, weight: float = 1.0):
    """Expected runs scored per 9 with this lineup in front of this offence.

        team_rs9 + weight * (lineup_r9 - baseline_r9)

    i.e. the team's own (already regressed) runs-scored rate, moved by how far
    today's nine sit from the baseline they are measured against — either the
    league (`baseline_r9 = lg_r9`) or the club's own typical lineup
    (`team_lineup_baseline`).

    Why a delta rather than the more obvious `w*lineup_r9 + (1-w)*team_rs9`:
    the lineup estimate is park- and baserunning-neutral while team RS/9 is
    not, so an absolute-level blend drags every club's offence toward the
    league mean — a Coors lineup and a Petco lineup both told they score
    league-average runs. That is the mistake `starters.py` documents having
    made and scored (0.2466, worse than no pitcher at all); the delta form
    leaves park, baserunning and sequencing in `team_rs9`, where they were
    measured, and can only ever add lineup information.

    Scalars or aligned arrays.
    """
    return np.asarray(team_rs9, dtype=float) + float(weight) * (
        np.asarray(lineup_r9_value, dtype=float) - np.asarray(baseline_r9, dtype=float))
