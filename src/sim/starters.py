"""Starting-pitcher rates for station E (per-game P(win)).

The production per-game model (`strength.py`) knows only team run
differential; the market's pre-first-pitch price knows who is on the mound.
That gap is worth 0.0046 Brier (docs/market-benchmark-2026.md). This module
is the first term aimed at it.

The chain, all pure functions over DataFrames so it unit-tests without a
network:

    season/game-log counts ─► marcel_rates()  Marcel-weighted, league-regressed
                                              K, BB+HBP, HR per batter faced —
                                              computed by station A's pitcher
                                              provider, `src/eval/pitchers.py`
                             ─► fip_ra9()     FIP coefficients → runs per 9,
                                              re-centred so a league-average
                                              pitcher gives league RA/9
                             ─► blend_starter_team()  the team's runs-allowed
                                              rate, moved by how far this
                                              starter is from league average
                                              over the ~5.5 innings he covers

The output is a runs-allowed-per-9 number that drops straight into the
existing Pythagenpat → log5 → HFA pipeline in place of the team's RA/G.
`scripts/backtest_game_odds.py` scores it as `pythag_60_sp`.

Everything above `── as-of-date assembly ──` is pure; the two helpers below it
fetch (`rate_inputs`) and are shared by both callers of the term —
`scripts/backtest_game_odds.py` walking a whole season and
`scripts/run_playoff_odds.py` pricing tonight's slate — so the live odds and
the scored backtest run the same chain by construction.

**One rate table.** The Marcel-weighted pitcher rates this module runs on are
not computed here. `marcel_rates` calls `src.eval.pitchers.pitcher_rates`, the
station A provider, with station E's own constants; the pre-provider arithmetic
survives as `_marcel_rates_legacy` and `legacy=True` only so the two can be
pinned together in the tests. That is what keeps station A's site number and
station E's odds from drifting apart: there is one estimator, and changing it
moves both, visibly.

Every constant below comes from outside the test set: Marcel's published
recency weights, the standard FIP coefficients, published rate-stabilization
points for the ballasts, 5.5 innings for an average start. The one free knob
(how much harder than reliability to regress) was chosen on the 2025 season,
walk-forward, and the 2025 curve is flat over the whole plausible range.
Nothing here is fit to the 2026 games this model is scored on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.sim.strength import HFA_PRIOR, home_win_prob, pythagenpat

# Marcel's recency weights: current season, one back, two back.
MARCEL_WEIGHTS = (5.0, 4.0, 3.0)
# Published stabilization points for pitcher rate stats: the sample at which
# split-half correlation reaches 0.5, which is exactly a regression ballast.
# The ordering is the whole point — strikeouts are nearly real on sight, walks
# take a couple of months, and home-run rate stays mostly noise for years,
# which matters because FIP puts a 13x coefficient on it.
STABILIZATION_BF = {"k": 70.0, "bbhbp": 170.0, "hr": 1300.0}
# Those points measure reliability (how much of *this* sample is signal).
# Projecting the *next* start also has to absorb real talent change between
# samples, so projection systems regress harder than reliability alone implies;
# roughly double is the conventional rule of thumb. Walk-forward on the 2025
# season agrees and is flat for anything above it (2.0/3.0/4.0/6.0 all land
# within 0.00003 Brier of each other), so nothing hinges on the exact value.
# Chosen on 2025 only — never on the games this model is scored against.
PROJECTION_MULTIPLIER = 2.0
BALLAST_BF = {c: v * PROJECTION_MULTIPLIER for c, v in STABILIZATION_BF.items()}
# Standard FIP coefficients on HR, BB+HBP and K. They are applied per *inning*
# — FIP = (13·HR + 3·(BB+HBP) − 2·K)/IP + C — and the constant is what puts the
# result on a per-nine-innings run scale.
FIP_COEF = {"hr": 13.0, "bbhbp": 3.0, "k": -2.0}
# Below this a "runs allowed per 9" is a data error, not a projection, and
# Pythagenpat would go complex on a non-positive rate. Never binds in practice.
MIN_RA9 = 0.5
# Innings the average start covers; the bullpen takes the other 3.5.
STARTER_IP = 5.5
GAME_IP = 9.0
# Ballast, in starts, on a pitcher's *own* innings per start when that is used
# as the workload split instead of the flat 5.5 (`expected_starter_ip`).
#
# Zero — no regression at all beyond the clip below — chosen walk-forward on
# 2025 over the grid {0, 5, 10, 20, 40, 80}, where the gain grows monotonically
# as the ballast falls and the sign is the same at every point (t between −2.2
# and −2.9). That is not the shape a noisy estimate makes, and the reason is
# that innings per start is mostly a fact about a pitcher's *role* rather than
# his luck: an opener is an opener, a horse is a horse, and regressing the
# opener's two innings toward five and a half only throws the role away.
IP_BALLAST_STARTS = 0.0
# Innings a start is allowed to project to. A starter cannot cover more than
# nine, and below three the bullpen delta would carry more than two thirds of
# the game on a number that is mostly noise.
MIN_STARTER_IP = 3.0
# Runs per nine a club is charged for each inning of the flat 5.5 its announced
# starter is *not* expected to cover — expected length as a **level** rather
# than only as the starter/bullpen split (`starter_length_delta`).
#
# Where it comes from: a gradient-boosted model given the chain's own inputs
# reproduced the chain term for term but put ten to twenty times the chain's
# weight on the two starters' expected innings, in both scored seasons
# (docs/market-benchmark-2026.md, "It reproduces the chain, term for term").
# In the chain that number only divides the game between two staffs, so a
# league-average starter who is expected to go four and one expected to go
# seven are the same pitcher to it.
#
# The constant is a maximum-likelihood fit of this one parameter on the eleven
# completed seasons *before* the scored one, 2015-2025, with nothing else
# refit — not the blend weight, not a ballast, not the home-field edge — which
# gives 0.89 runs per nine with a standard error of 0.30 (t = 2.95); the paired
# Brier argmin on the same games is 0.88. Rounded to 0.9 and never chosen on a
# 2026 score (`scripts/sweep_starter_ip_level.py`). Choosing it on 2025 alone,
# the way the constants a station ago were chosen, would have picked 1.37 and
# scored *better* on the held-out season, so the wider pool is the conservative
# reading and not a flattering one. Zero is the chain without the term, to the
# last bit.
IP_LEVEL_RUNS = 0.9

COMPONENTS = ("k", "bbhbp", "hr")
RATE_COLS = [f"rate_{c}" for c in COMPONENTS]
# Short name here -> the station A component that projects it. The rates below
# are computed by that provider (`src/eval/pitchers.py`), so the two stations
# share one estimator and one set of constants by construction rather than by
# two files agreeing for now.
PITCHER_COMPONENTS = {"k": "p_k_rate", "bbhbp": "p_bbhbp_rate",
                      "hr": "p_hr_rate"}


def normalize_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Pitcher count frame → the columns this module works in.

    Accepts the raw Stats API column names (`bf, k, bb, hbp, hr, outs`) and
    returns `pitcher, season, bf, k, bbhbp, hr, outs`. Walks are folded
    together with hit batsmen because FIP treats them identically.
    """
    def col(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(0.0, index=df.index, dtype=float)
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0).astype(float)

    out = pd.DataFrame({
        "pitcher": df["pitcher"].astype("int64"),
        "season": df["season"].astype("int64"),
        "bf": col("bf"), "k": col("k"), "bbhbp": col("bb") + col("hbp"),
        "hr": col("hr"), "outs": col("outs"),
    })
    return out[["pitcher", "season", "bf", "k", "bbhbp", "hr", "outs"]]


def league_rates(counts: pd.DataFrame) -> dict:
    """Pooled league rates per batter faced, plus batters faced per 9 innings.

    Keys: rate_k, rate_bbhbp, rate_hr, bf_per_ip.
    """
    bf = float(counts["bf"].sum())
    outs = float(counts["outs"].sum())
    if bf <= 0:
        raise ValueError("league_rates: no batters faced in the counts frame")
    lg = {f"rate_{c}": float(counts[c].sum()) / bf for c in COMPONENTS}
    # Three outs to an inning; fall back to the modern ~4.3 if outs are absent.
    lg["bf_per_ip"] = 3.0 * bf / outs if outs > 0 else 4.3
    return lg


def appearances_before(game_logs: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Season-to-date counts per pitcher from appearances *strictly before* `as_of`.

    This is the leakage guard: an appearance on the game's own date (the start
    being predicted, or an earlier game of a doubleheader that has not been
    played when the line is priced) never contributes to the rates used to
    predict it.
    """
    cols = ["pitcher", "season", "bf", "k", "bbhbp", "hr", "outs"]
    if game_logs.empty:
        return pd.DataFrame(columns=cols)
    past = game_logs[game_logs["date"].astype(str) < str(as_of)]
    if past.empty:
        return pd.DataFrame(columns=cols)
    return (past.groupby(["pitcher", "season"], as_index=False)[
        ["bf", "k", "bbhbp", "hr", "outs"]].sum())


def _ballast_map(ballast) -> dict:
    """A scalar applies to every component; a mapping is used as given."""
    if isinstance(ballast, dict):
        return {c: float(ballast[c]) for c in COMPONENTS}
    return {c: float(ballast) for c in COMPONENTS}


def marcel_params(weights: tuple = MARCEL_WEIGHTS, ballast=BALLAST_BF) -> dict:
    """Station E's constants as the harness's `MarcelParams`, one per component.

    The only translation needed is the ballast unit. Here (and in every
    published stabilization table) a ballast is real batters faced, which land
    on the *most recent* season's weight; `MarcelParams.ballast` is denominated
    at the *average* year weight, because that is what makes the estimator
    scale-free in the weights. The two differ by `w0 / mean(w)`, which is 1.25
    for 5/4/3.
    """
    from src.eval.baselines import MarcelParams

    bal = _ballast_map(ballast)
    scale = float(weights[0]) / float(np.mean(weights))
    return {component: MarcelParams(ballast=bal[c] * scale,
                                    weights=tuple(float(x) for x in weights),
                                    peak_age=27.0, age_slope_young=0.0,
                                    age_slope_old=0.0)
            for c, component in PITCHER_COMPONENTS.items()}


def marcel_rates(counts: pd.DataFrame, as_of_season: int, lg: dict,
                 weights: tuple = MARCEL_WEIGHTS,
                 ballast=BALLAST_BF, legacy: bool = False) -> pd.DataFrame:
    """Per-pitcher K, BB+HBP and HR rates per batter faced.

    `counts` holds one row per pitcher-season (normalize_counts schema) and may
    mix the partial current season with completed prior ones. Seasons are
    weighted by recency — `weights[i]` for `as_of_season - i` — normalised so
    the most recent weight is 1, so `ballast` is denominated in real batters
    faced. `ballast` is either one number for all three components or a
    {component: batters faced} mapping (the default, `BALLAST_BF`).

    Returns a frame indexed by pitcher with rate_k, rate_bbhbp, rate_hr and
    `bf_weighted` (the effective sample behind those rates). A pitcher with no
    history is simply absent — `starter_ra9_lookup` gives those league average.

    **This is now a call into station A's pitcher provider**
    (`src/eval/pitchers.pitcher_rates`), not a second implementation of it.
    Station E and the site therefore cannot drift: the same estimator, the same
    ballasts and the same recency weights answer both. `legacy=True` runs the
    original in-module arithmetic instead, kept so the equivalence is testable
    and so a regression here could never be silent;
    `tests/test_sim/test_starters.py` pins the two together on a fixed
    pitcher-season.
    """
    if legacy:
        return _marcel_rates_legacy(counts, as_of_season, lg, weights, ballast)
    from src.eval.pitchers import pitcher_rates

    return pitcher_rates(counts, as_of_season, lg,
                         params=marcel_params(weights, ballast),
                         components=PITCHER_COMPONENTS)


def _marcel_rates_legacy(counts: pd.DataFrame, as_of_season: int, lg: dict,
                         weights: tuple = MARCEL_WEIGHTS,
                         ballast=BALLAST_BF) -> pd.DataFrame:
    """The pre-provider arithmetic, kept as the reference the provider is
    pinned against. Nothing in production calls it."""
    w = {as_of_season - i: weights[i] / weights[0] for i in range(len(weights))}
    used = counts[counts["season"].isin(w)].copy()
    if used.empty:
        return pd.DataFrame(columns=["bf_weighted", *RATE_COLS],
                            index=pd.Index([], name="pitcher", dtype="int64"))
    ws = used["season"].map(w).astype(float)
    for col in ("bf", *COMPONENTS):
        used[col] = used[col].astype(float) * ws
    agg = used.groupby("pitcher")[["bf", *COMPONENTS]].sum()
    bal = _ballast_map(ballast)
    out = pd.DataFrame(index=agg.index)
    out["bf_weighted"] = agg["bf"]
    for c in COMPONENTS:
        out[f"rate_{c}"] = ((agg[c] + bal[c] * lg[f"rate_{c}"])
                            / (agg["bf"] + bal[c]))
    return out


def fip_constant(lg: dict, lg_ra9: float) -> float:
    """Additive constant putting FIP on the league's runs-allowed-per-9 scale.

    Standard FIP is calibrated so league FIP = league ERA; here we calibrate to
    league RA/9 instead, because the team number this blends with (station D's
    regressed runs allowed per game) includes unearned runs.
    """
    return float(lg_ra9) - _fip_core(lg, lg["bf_per_ip"])


def _fip_core(rates, bf_per_ip: float):
    """(13·HR + 3·(BB+HBP) − 2·K) / IP, from per-batter-faced rates."""
    per_bf = sum(FIP_COEF[c] * rates[f"rate_{c}"] for c in COMPONENTS)
    return per_bf * bf_per_ip


def fip_ra9(rates: pd.DataFrame, lg: dict, lg_ra9: float) -> pd.Series:
    """Per-pitcher FIP expressed as runs allowed per 9 innings.

    A pitcher whose rates equal the league's scores exactly `lg_ra9`, so the
    starter term only ever *re-allocates* runs around the team baseline — it
    cannot silently shift the whole league's run environment.
    """
    if rates.empty:
        return pd.Series(dtype=float, name="sp_ra9")
    core = _fip_core(rates, lg["bf_per_ip"])
    ra9 = (core + fip_constant(lg, lg_ra9)).clip(lower=MIN_RA9)
    return pd.Series(ra9, index=rates.index, name="sp_ra9").astype(float)


def blend_starter_team(sp_ra9, team_ra9, lg_ra9, starter_ip: float = STARTER_IP,
                       game_ip: float = GAME_IP):
    """Expected runs allowed per 9 with this starter in front of this staff.

        team_ra9 + w · (sp_ra9 − lg_ra9),   w = starter_ip / game_ip

    i.e. the team's own (already regressed) runs-allowed rate, moved by the
    starter's *deviation from a league-average starter*, weighted by the share
    of the game he is expected to cover.

    Why a delta rather than the more obvious `w·sp_ra9 + (1−w)·team_ra9`: FIP
    is park- and defense-neutral, while team RA/9 is not. Mixing them on
    absolute levels quietly drags 61% of every team's run prevention back to
    the league mean — a Coors staff and a Petco staff would both be told they
    allow league-average runs for 5.5 innings — which is a *team* regression
    wearing a pitcher's uniform, and it costs more than the pitcher signal
    gains. In delta form a league-average starter leaves the team's number
    exactly where station D put it, so the term can only ever add pitcher
    information. Park, defense and framing stay in `team_ra9`, where they were
    measured.

    Using the team's own rate for the relief innings rather than a separate
    bullpen number keeps v1 to one new moving part.

    Scalars or aligned arrays.
    """
    w = float(starter_ip) / float(game_ip)
    return np.asarray(team_ra9, dtype=float) + w * (
        np.asarray(sp_ra9, dtype=float) - np.asarray(lg_ra9, dtype=float))


def start_innings(logs: pd.DataFrame) -> pd.DataFrame:
    """`pitcher, date, ip` for every *start* in a pitching game-log frame.

    The mirror of `bullpen.relief_appearances` on the innings column: only
    outings the pitcher started, because how deep he goes as a starter says
    nothing about a relief appearance and vice versa.
    """
    cols = ["pitcher", "date", "ip"]
    if logs.empty:
        return pd.DataFrame(columns=cols)
    keep = logs[pd.to_numeric(logs.get("gs", 0), errors="coerce").fillna(0) == 1]
    if keep.empty:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame({
        "pitcher": pd.to_numeric(keep["pitcher"], errors="coerce").astype("int64"),
        "date": keep["date"].astype(str),
        "ip": pd.to_numeric(keep.get("outs", 0), errors="coerce").fillna(0.0) / 3.0,
    }).reset_index(drop=True)


def expected_starter_ip(starts: pd.DataFrame, as_of: str,
                        ballast: float = IP_BALLAST_STARTS,
                        prior: float = STARTER_IP) -> dict:
    """{pitcher_id: expected innings} from his starts *strictly before* `as_of`.

    A straight regression of his own innings per start toward `prior` (the flat
    5.5 every other model uses) with `ballast` league-average starts of ballast:

        ip = (Σ innings + ballast · prior) / (starts + ballast)

    Strictly before, for the same reason every other table in this module is:
    how long tonight's start lasts is the outcome being predicted.

    A pitcher with no starts on file is absent, and callers fall back to
    `prior`, which leaves the model exactly where the flat split put it.
    """
    if starts is None or len(starts) == 0:
        return {}
    past = starts[starts["date"].astype(str) < str(as_of)]
    if past.empty:
        return {}
    agg = past.groupby("pitcher")["ip"].agg(["sum", "size"])
    ip = ((agg["sum"] + float(ballast) * float(prior))
          / (agg["size"] + float(ballast))).clip(MIN_STARTER_IP, GAME_IP)
    return {int(k): float(v) for k, v in ip.items()}


def starter_length_delta(starter_ip, level_runs: float = IP_LEVEL_RUNS,
                         prior: float = STARTER_IP, game_ip: float = GAME_IP):
    """Runs per nine a start's expected *length* carries on its own.

        level_runs · (prior − starter_ip) / game_ip

    The chain already spends `starter_ip` on arithmetic: it is the weight that
    divides the game between `blend_starter_team` and
    `bullpen.blend_bullpen_team`. This is the other half of the same fact — how
    deep a manager expects a starter to go is information about the *club*, not
    only a share of nine innings, and a flexible model given the chain's own
    inputs kept asking for it as a level (docs/market-benchmark-2026.md).

    A delta, like every other term in the chain: a starter expected to cover
    exactly `prior` leaves the club's runs-allowed rate where the split put it,
    so the term can only redistribute and `level_runs = 0` is the chain without
    it to the last bit. Centring on the flat 5.5 rather than on the day's own
    mean start is deliberate and immaterial — the centre is common to both
    clubs and cancels in log5 to within 1.2e-5 of Brier on any of the twelve
    seasons on file, whether it is the flat 5.5, the season's own mean or that
    date's — and it keeps the term free of any new walk-forward quantity for
    anything to leak through.

    Sign: a *short* expected start is charged runs. The innings the split hands
    to the pen are not the pen's average innings; they are its worst, and the
    pitchers a manager pulls at four have something wrong with them that FIP
    (which is blind to balls in play and to sequencing) has not yet priced.

    Scalars or aligned arrays.
    """
    return (float(level_runs) / float(game_ip)) * (
        float(prior) - np.asarray(starter_ip, dtype=float))


def starter_ra9_lookup(rates: pd.DataFrame, lg: dict, lg_ra9: float) -> dict:
    """{pitcher_id: FIP runs/9}. Missing ids fall back to `lg_ra9` at lookup."""
    return {int(k): float(v) for k, v in fip_ra9(rates, lg, lg_ra9).items()}


# ─── as-of-date assembly: the same rates for the backtest and the live job ───

def rate_inputs(season: int, pitcher_ids, prior_seasons: int = 2,
                refresh: bool = False) -> dict:
    """Fetch everything the rate table needs for `season`, once.

    The only function in this module that touches the network. Returns the
    handful of frames `rate_table` slices per date: `prior_counts` (completed
    seasons, Marcel-weighted alongside the current one), `game_logs` (this
    season's appearances, dated, so a date cut can be applied), and `league`
    (pooled rates from the completed seasons — the current-season logs we hold
    are starters-only, so pooling them would bias the regression target).

    Responses are cached under data/cache/statsapi/ (gitignored), so a rerun
    of either caller is offline and identical.

    `refresh` re-pulls the *current* season's game logs, and only those. Their
    cache key is the season, not a date, so a live caller asking about today
    would otherwise be served a snapshot taken whenever the file was first
    written and silently miss every start since — harmless for the backtest
    (it reads a finished season once) and wrong for the nightly job, which
    passes refresh=True. The completed prior seasons never move, so they stay
    cached either way.
    """
    from src.data.mlb_stats_api import fetch_pitcher_game_logs, fetch_season_pitching

    prior = pd.concat(
        [fetch_season_pitching(y) for y in range(season - prior_seasons, season)],
        ignore_index=True)
    prior_counts = normalize_counts(prior)
    logs = fetch_pitcher_game_logs(pitcher_ids, season, refresh=refresh)
    logs = logs[logs["game_type"] == "R"]
    game_logs = normalize_counts(logs)
    game_logs["date"] = logs["date"].to_numpy()
    return {"season": season, "prior_counts": prior_counts,
            "game_logs": game_logs, "league": league_rates(prior_counts)}


def rate_table(inputs: dict, as_of: str, lg_ra9: float,
               ballast=BALLAST_BF, legacy: bool = False) -> dict:
    """{pitcher_id: FIP runs/9} using only appearances *strictly before* `as_of`.

    Pure — `inputs` is what `rate_inputs` returns. For the nightly job `as_of`
    is today, so a start made earlier today (or last night's late game, once
    the log posts) is still excluded, exactly as in the walk-forward backtest.
    `lg_ra9` anchors the FIP constant to the season's run environment, so a
    league-average arm returns the team's own runs-allowed rate unchanged.

    The rates themselves come from station A's pitcher provider; `legacy=True`
    routes them through the pre-provider arithmetic instead, which exists so
    the equivalence is testable rather than assumed.
    """
    current = appearances_before(inputs["game_logs"], as_of)
    counts = pd.concat([inputs["prior_counts"], current], ignore_index=True)
    rates = marcel_rates(counts, inputs["season"], inputs["league"],
                         ballast=ballast, legacy=legacy)
    return starter_ra9_lookup(rates, inputs["league"], lg_ra9)


def build_rate_table(as_of: str, pitcher_ids, season: int, lg_ra9: float,
                     ballast=BALLAST_BF, prior_seasons: int = 2,
                     refresh: bool = False, legacy: bool = False) -> dict:
    """`rate_inputs` + `rate_table` for a single date (the nightly job's case).

    The backtest wants the two halves apart — it fetches once and re-slices for
    each of ~160 dates — so this convenience wrapper exists for the caller that
    only ever asks about one day.
    """
    inputs = rate_inputs(season, pitcher_ids, prior_seasons=prior_seasons,
                         refresh=refresh)
    return rate_table(inputs, as_of, lg_ra9, ballast=ballast, legacy=legacy)


def game_home_prob(team_rates: pd.DataFrame, home_id, away_id, sp_ids,
                   sp_ra9: dict, lg_ra9: float, hfa: float = HFA_PRIOR,
                   starter_ip: float = STARTER_IP) -> tuple[float, int]:
    """P(home wins) for one game with both starters known — the `pythag_60_sp` chain.

    `team_rates` is indexed by team_id with rs_pg / ra_pg columns (station D's
    regressed run rates: `strength.regressed_run_rates`, or the backtest's
    walk-forward equivalent). `sp_ids` is (home starter id, away starter id);
    an id missing from `sp_ra9` is a pitcher with no history and is scored at
    league average, which leaves that side exactly at its team rate.

    Returns (P(home), number of the two starter slots that had no history).
    """
    no_history = 0
    talent = {}
    for side, team_id, pid in (("home", home_id, sp_ids[0]),
                               ("away", away_id, sp_ids[1])):
        no_history += int(int(pid) not in sp_ra9)
        ra9 = blend_starter_team(sp_ra9.get(int(pid), lg_ra9),
                                 team_rates.loc[team_id, "ra_pg"], lg_ra9,
                                 starter_ip=starter_ip)
        talent[side] = pythagenpat(float(team_rates.loc[team_id, "rs_pg"]),
                                   float(ra9), 1.0)
    return float(home_win_prob(talent["home"], talent["away"], hfa)), no_history
