"""Station A, live, pitcher side: the rest-of-season pitcher line the site serves.

The hitter module next door (`ros.py`) is rates times station B's projected
plate appearances. This is the same shape with the other id on the row:

    rest-of-season pitcher projection
        = marcel_pitcher_tuned(prior seasons + 2026 through as_of - 1 day)
          x  a projected count of batters faced

Both halves have now been through a gate — the rates on Sept 3, 2026 and the
workload later the same day — which is what this docstring used to exist to
deny.

**The rates are gated.** They are `src/eval/pitchers.marcel_pitcher_tuned` with
the constants frozen in `src/eval/marcel_pitcher_params.json`, and every
component served here beat league average, the previous season and season to
date out of sample on the 2026 cutoffs and season-level 2025 and 2026
(`scripts/run_pitcher_backtest.py`, the pitcher section of
docs/backtest-baselines.md). A component that had not cleared would not be in
`SERVED_COMPONENTS` and would not reach the page.

**Two of them are gated twice, since BAS-79.** BB/BF and HR/BF are served as
`stuff_additive`: the tuned pitcher Marcel above, untouched, plus a correction
built from the pitch-tracking aggregates of docs/pitching-stuff.md. Which
components get that correction is `LIVE_ENGINE` below, and it was decided by a
serving gate written down before the additive arm was scored — K/BF, the
component the stuff model was built for, misses it and stays on plain Marcel.
The features are read as of the last month boundary on or before the as-of
date (`stuff_cutoff`), never a partial month, and the fit is walk-forward on
seasons strictly before the one being served; a component whose fit cannot be
built falls back to `marcel_pitcher_tuned` for that build alone.

**The batters faced are gated too, since Sept 3, 2026.** This function used to
be stamped `structural` because nobody had scored it against anything. Now
somebody has: `src/projections/pitcher_workload.py` is station B's harness
pointed at pitchers, and `projected_batters_faced` below is one of the methods
it runs, called rather than copied so what the harness scores is what the site
publishes. Over 26 walk-forward as-of dates in 2024-2026 — a fortnight apart
from May to September, 22,807 pitcher-projections — it beats a season-to-date
rate extrapolation by 5.6 batters faced a pitcher (t -16.7), the trailing-30-day
one by 4.9 (-12.1), last season prorated by 22.1 (-19.8) and projecting nobody
at all by 47.7 (-19.9), and nothing built to beat it did
([docs/pitcher-workload.md](../../docs/pitcher-workload.md)). The arithmetic is
unchanged:

    projected BF = games the club has left
                 x his appearance rate per club game  (30-day and season
                   blended, regressed toward his role's rate)
                 x his batters faced per appearance   (regressed toward his
                   role's average)
                 x station B's expected active fraction, for a pitcher who is
                   hurt or optioned

That last term is the largest single thing in here: dropping it costs 4.4
batters faced a pitcher (t -37.8), and 9.2 for a starter. Knowing a pitcher is
on the injured list this morning is worth more than every refinement tried on
the other three terms put together — and, unlike station B's hitters, an
injured *pitcher* is better projected near zero than at his pre-injury usage
discounted by an expected return date, which is the one place the two stations
disagree.

Role is read off the workload itself rather than a depth chart: a pitcher
averaging at least `STARTER_MIN_BF` batters an outing is a starter this month,
whatever he was in April. Reading it off `gamesStarted` instead — which gets
the opener right — was tried in the harness and is not better.

Everything is a pure function over DataFrames, so it unit-tests without a
network; the fetch layer is `scripts/build_ros_projections.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval import pitchers as P
from src.eval import stuff as stuff_eval
from src.eval.backtest import COMPONENTS
from src.eval.intraseason import build_training_frame, split_at_cutoff

SEASON = 2026

# The engine, named once — and, since BAS-79, per component rather than once
# for the whole station, exactly as `ros.LIVE_ENGINE` is for hitters. The
# builder stamps it into the document and the accuracy page reads it to pick
# the arm it marks live, so the scoreboard cannot score a model the site does
# not serve.
#
# `stuff_additive` is the pitching-stuff arm of docs/pitching-stuff.md: the
# monthly stuff aggregates (predicted whiffs per swing, CSW per pitch, the
# same on fastballs and non-fastballs, velocity and fastball share) added as a
# correction on top of `marcel_pitcher_tuned`, whose coefficient is pinned at
# exactly 1. Which components get it was decided by the serving gate that doc
# pre-registered *before* any additive number was read — clustered |t| > 2.5
# on the additive arm over 2022-2026 x May/Jul/Aug, 4,881 pitcher-cells, 948
# pitchers — and not by the free `stuff` fit's gate table:
#
#   p_bb_rate   -3.24% of MAE, t -6.13  -> served
#   p_hr_rate   -2.93%, t -3.93         -> served
#   p_k_rate    -2.18%, t -2.38         -> WITHHELD: misses |t| > 2.5
#   p_babip     not scored at all       -> withheld; stuff has no mechanism
#                                          for balls in play, so there is no
#                                          arm to serve
#
# K/BF is the component the ticket was written for and it is the one held
# back. The free fit cleared (-3.20%, t -3.37); pinning the baseline's
# coefficient at 1 keeps 68% of that gain — inside the 60-90% the doc
# predicted — but the standard error does not shrink with it, so the arm that
# would actually ship lands just under the pre-registered bar. The rule was
# written down first precisely so a near miss on the headline component gets
# withheld rather than argued with; it goes back in when a later ticket earns
# it, not when the number is squinted at.
MARCEL_ENGINE = "marcel_pitcher_tuned"
STUFF_ENGINE = "stuff_additive"
LIVE_ENGINE = {
    "p_k_rate": MARCEL_ENGINE,
    "p_bb_rate": STUFF_ENGINE,
    "p_hr_rate": STUFF_ENGINE,
    "p_babip": MARCEL_ENGINE,
}

# The components that cleared the serving gate, in page order. `p_bbhbp_rate`
# clears too but is not here: it is the walks-plus-hit-batsmen rate station E's
# FIP term consumes, and a site column labelled BB% has to mean walks. It is
# scored in the harness and it feeds the odds; it is not a column.
SERVED_COMPONENTS = ("p_k_rate", "p_bb_rate", "p_hr_rate", "p_babip")
COMPONENT_PREFIX = {"p_k_rate": "k", "p_bb_rate": "bb", "p_hr_rate": "hr",
                    "p_babip": "babip"}

ARMS = ("marcel", "marcel_preseason")
LIVE_PROVIDERS = {
    "marcel": P.marcel_pitcher_tuned,
    "marcel_preseason": P.marcel_pitcher_tuned_preseason,
}

# --- the batters-faced model (scored Sept 3, 2026) --------------------------

BF_METHOD = "recent_usage"
BF_METHOD_NOTE = (
    "Projected batters faced: club games remaining x the pitcher's appearance "
    "rate per club game (trailing 30 days and season blended, regressed toward "
    "his role's rate) x his batters faced per appearance (regressed toward his "
    "role's average) x station B's expected active fraction. Scored "
    "walk-forward at 26 as-of dates over 2024-2026 against a season-to-date "
    "rate extrapolation, a trailing-30-day one, last season prorated and no "
    "model at all, on 22,807 pitcher-projections: MAE 45.6 batters faced "
    "against 51.1, 50.4, 67.6 and 93.3, paired -5.6 (t -16.7), -4.9 (-12.1), "
    "-22.1 (-19.8) and -47.7 (-19.9). See docs/pitcher-workload.md."
)

# At least this many batters an outing and he is being used as a starter.
# Three innings and change; an opener sits below it, which is right, because
# an opener's workload really is a reliever's.
STARTER_MIN_BF = 12.0
# Role averages, from 2026 league usage: a start is about 5.2 innings at 4.3
# batters an inning, a relief outing about an inning; a starter takes a turn
# every fifth or sixth club game, a busy reliever about two games in five.
ROLE_BF_PER_APPEARANCE = {"SP": 22.0, "RP": 4.3}
ROLE_APPEARANCE_RATE = {"SP": 1.0 / 5.3, "RP": 0.40}
# Ballasts, in appearances and in club games. Both deliberately light: role is
# the strong signal and these only keep a three-outing sample from projecting
# a season off itself.
BF_BALLAST_APPEARANCES = 5.0
RATE_BALLAST_GAMES = 10.0
# How much of the appearance rate comes from the trailing 30 days rather than
# the whole season. Half: a reliever's season rate is the steadier estimate,
# and the recent window is what notices a call-up or a move to the rotation.
RECENT_WEIGHT = 0.5
RECENT_DAYS = 30

OUTPUT_COLUMNS = (
    ["pitcher", "name", "team_id", "team_abbrev", "as_of", "role",
     "appearances", "bf_to_date", "bf_ros"]
    + [f"{COMPONENT_PREFIX[c]}_rate_{arm}"
       for c in SERVED_COMPONENTS for arm in ARMS]
    + ["k_ros", "bb_ros", "hr_ros", "fip_ros"]
)

# Standard FIP coefficients and the per-inning form, same constants as
# `src/sim/starters.py`. The constant is re-derived per build so that a league
# average line comes back at the league's own runs allowed per nine.
FIP_COEF = {"hr": 13.0, "bb": 3.0, "k": -2.0}


def _as_date(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


# --- the stuff engine ---------------------------------------------------

def stuff_cutoff(as_of) -> pd.Timestamp:
    """The last pitching-stuff month boundary on or before `as_of`.

    The same rule `ros.contact_cutoff` states for hitters, for the same
    reason. The nightly build runs daily; the stuff artifact is monthly
    (`data/features/pitching_stuff_monthly.parquet`) and
    `stuff.window_counts` refuses a cutoff that is not the first of a month
    rather than rounding one forward, since rounding forward is leakage. So a
    build made on, say, Sept 9 cannot ask for September's stuff buckets — it
    asks for the buckets through the *last* month boundary, Sept 1, which sums
    everything strictly before September, i.e. through Aug 31. The stuff
    features therefore lag the as-of date by up to a month while the Marcel
    rates on the same row do not: a build on Sept 30 gets the same August-end
    features as one on Sept 1. `stuff_features_through` (the day before this
    cutoff) is what the served document stamps, so the lag is visible rather
    than implicit.
    """
    as_of = _as_date(as_of)
    return pd.Timestamp(year=as_of.year, month=as_of.month, day=1)


def stuff_features_through(as_of) -> pd.Timestamp:
    """The last day stuff features actually cover, for provenance."""
    return stuff_cutoff(as_of) - pd.Timedelta(days=1)


def stuff_engine_provider(
    component: str,
    seasons_table: pd.DataFrame,
    monthly: pd.DataFrame,
    pa_dir,
    as_of,
    predict_year: int = SEASON,
):
    """A `LIVE_PROVIDERS`-shaped provider for the `stuff_additive` engine.

    Fits the correction walk-forward on cell seasons strictly before
    `predict_year` (`stuff_eval.fit_live_stuff`, `fixed_base=True` by default
    — never on the season being served, exactly as the harness that cleared
    the gate did) and builds the covariates as of the last month boundary on
    or before `as_of` (`stuff_cutoff`), never a partial month. The baseline
    underneath is the same `marcel_pitcher_tuned` the gate compared it
    against, left untouched — `stuff_additive` only adds to it.
    """
    fit = stuff_eval.fit_live_stuff(
        component, seasons_table, monthly, pa_dir, predict_year)
    config = stuff_eval.StuffProviderConfig(
        monthly=monthly,
        cutoff=stuff_cutoff(as_of),
        predict_year=predict_year,
        fit=fit,
        base_provider=P.marcel_pitcher_tuned,
    )
    return stuff_eval.stuff_provider(config)


def engine_providers(
    seasons_table: pd.DataFrame | None = None,
    monthly: pd.DataFrame | None = None,
    pa_dir=None,
    as_of=None,
    predict_year: int = SEASON,
    components=SERVED_COMPONENTS,
) -> tuple[dict[str, object], dict[str, str]]:
    """(component -> provider, component -> engine actually used) for the
    `marcel` (live) arm, per `LIVE_ENGINE`.

    A component whose engine is `stuff_additive` but is missing what the stuff
    engine needs (the monthly artifact, the PA outcomes directory, the as-of
    date, or a walk-forward fit that raises) falls back to
    `marcel_pitcher_tuned` for that component alone rather than failing the
    whole build — the same honest fallback the hitter side makes
    (`ros.engine_providers`), and what the pre-registration's fourth serving
    prediction promises. `engine_used` is what the builder stamps into the
    document, so it can differ from `LIVE_ENGINE` on a bad night without ever
    claiming a model that did not run.
    """
    have_stuff_inputs = (seasons_table is not None and monthly is not None
                         and pa_dir is not None and as_of is not None)
    providers, engine_used = {}, {}
    for component in components:
        engine = LIVE_ENGINE.get(component, MARCEL_ENGINE)
        if engine == STUFF_ENGINE and have_stuff_inputs:
            try:
                providers[component] = stuff_engine_provider(
                    component, seasons_table, monthly, pa_dir, as_of,
                    predict_year)
                engine_used[component] = STUFF_ENGINE
                continue
            except Exception:                                   # noqa: BLE001
                pass
        providers[component] = P.marcel_pitcher_tuned
        engine_used[component] = MARCEL_ENGINE
    return providers, engine_used


# --- the partial season ------------------------------------------------

def partial_season(pa_frame: pd.DataFrame, as_of, season: int = SEASON,
                   since=None) -> pd.DataFrame:
    """The current season through `as_of - 1 day`, per pitcher.

    A game played *on* `as_of` has not finished when the morning's projection
    is made, so the cutoff is exclusive — the same convention the hitter module
    and station B use. `since` optionally drops everything before a date, which
    is how the trailing-30-day window is taken.
    """
    as_of = _as_date(as_of)
    year = pa_frame
    if "game_year" in year.columns:
        year = year[year["game_year"] == season]
    before, _ = split_at_cutoff(year, as_of)
    if since is not None:
        dates = pd.to_datetime(before["game_date"])
        before = before[dates >= _as_date(since)]
    return P.aggregate_pa_pitchers(before, season).assign(partial=True)


def league_rates(partial: pd.DataFrame) -> dict[str, float]:
    """Trials-weighted league rates through the cutoff, per component key.

    Doubles as the zero-history fallback: Marcel with no trials at all *is* the
    league rate, so a September call-up gets exactly that rather than a blank
    row.
    """
    out = {c: float("nan") for c in SERVED_COMPONENTS}
    out["bf_per_ip"] = 4.3
    if partial.empty:
        return out
    for component in SERVED_COMPONENTS:
        spec = COMPONENTS[component]
        den = float(partial[spec.trials].sum())
        out[component] = (float(partial[spec.successes].sum()) / den
                          if den > 0 else float("nan"))
    # Outs are not in the PA parquet; batters faced per inning is recovered
    # from the outs a plate appearance produces on average, which is what the
    # league's own strikeout and on-base rates imply.
    out["bf_per_ip"] = 3.0 / max(_outs_per_bf(partial), 1e-6)
    return out


def _outs_per_bf(partial: pd.DataFrame) -> float:
    """Outs recorded per batter faced, from the counts the parquet carries.

    Every plate appearance that is not a hit, a walk, a hit batsman or a
    reached-on-error is an out somewhere; this uses the simple identity
    (BF - H - BB - HBP) / BF, which lands within a hair of the real figure
    because double plays and baserunning outs roughly offset reached-on-error.
    """
    bf = float(partial["bf"].sum())
    if bf <= 0:
        return 3.0 / 4.3
    on_base = float((partial["h"] + partial["bb"] + partial["hbp"]).sum())
    return max(bf - on_base, 1.0) / bf


# --- the rates ---------------------------------------------------------

def pitcher_rates(seasons_table: pd.DataFrame, partial: pd.DataFrame,
                  predict_year: int = SEASON,
                  components=SERVED_COMPONENTS,
                  *,
                  stuff_monthly: pd.DataFrame | None = None,
                  stuff_pa_dir=None,
                  as_of=None) -> pd.DataFrame:
    """The live arm and its preseason control, one row per pitcher.

    Columns: `pitcher` plus `{prefix}_rate_marcel` and
    `{prefix}_rate_marcel_preseason`. The training frame is the harness's own
    (`build_training_frame`), so the model that ships is bit-for-bit the arm
    that was scored: prior full seasons from the pitcher season table plus the
    partial current season, with ages carried forward.

    The `marcel_preseason` control is always plain `marcel_pitcher_tuned` with
    the season withheld; the live arm's provider is per component, since the
    engine is (`LIVE_ENGINE`). The control exists to isolate the value of
    in-season information *with the model held fixed*, so it does not move
    with the live engine.
    """
    train = build_training_frame(seasons_table, partial, predict_year, "pitcher")
    if train.empty:
        raise ValueError("no training data: both the pitcher season table and "
                         "the partial season are empty")
    has_prior = not P.full_seasons(train).empty

    live_providers, engine_used = engine_providers(
        seasons_table, stuff_monthly, stuff_pa_dir, as_of, predict_year,
        components)

    out = pd.DataFrame({"pitcher": pd.unique(train["pitcher"])})
    for component in components:
        spec = COMPONENTS[component]
        prefix = COMPONENT_PREFIX[component]
        for arm in LIVE_PROVIDERS:
            column = f"{prefix}_rate_{arm}"
            if arm == "marcel_preseason" and not has_prior:
                out[column] = np.nan
                continue
            provider = (live_providers[component] if arm == "marcel"
                        else LIVE_PROVIDERS[arm])
            pred = provider(train, spec, predict_year)
            out = out.merge(
                pred.rename(columns={"predicted": column})[["pitcher", column]],
                on="pitcher", how="left")
    # `DataFrame.attrs` rides along with the frame rather than a second return
    # value, so the builder can read what engine each component actually used
    # without a signature change of its own — the same channel the hitter side
    # uses.
    out.attrs["pitcher_engine_used"] = engine_used
    if as_of is not None:
        out.attrs["stuff_features_through"] = (
            stuff_features_through(as_of).date().isoformat())
    return out


# --- projected batters faced (structural) ------------------------------

def role_of(bf_per_appearance) -> np.ndarray:
    """"SP" or "RP" from the workload itself, not from a depth chart."""
    return np.where(np.asarray(bf_per_appearance, dtype=float) >= STARTER_MIN_BF,
                    "SP", "RP")


def projected_batters_faced(
    partial: pd.DataFrame,
    recent: pd.DataFrame,
    team_games_played: pd.Series | dict,
    team_games_recent: pd.Series | dict,
    games_remaining: pd.Series | dict,
    team_of: pd.Series | dict,
    active_fraction: pd.Series | dict | None = None,
) -> pd.DataFrame:
    """`pitcher, role, appearances, bf_to_date, bf_ros` — the structural half.

    Args:
        partial: season-through-the-cutoff aggregate (`partial_season`).
        recent: the same over the trailing window only.
        team_games_played / team_games_recent: club games in each window.
        games_remaining: club games left after the cutoff.
        team_of: pitcher -> team_id, from the 40-man snapshot.
        active_fraction: pitcher -> expected share of the horizon he is back
            for, from station B's return-time distribution. A pitcher who is
            unavailable and absent from this mapping is projected at zero,
            which is the same fallback station B uses.

    A pitcher with no 2026 appearances at all comes back with zero projected
    batters faced and is dropped by the caller: he has no usage to extrapolate
    and inventing one would be a depth chart, not a projection.
    """
    team_of = pd.Series(team_of)
    games_remaining = pd.Series(games_remaining, dtype="float64")
    team_games_played = pd.Series(team_games_played, dtype="float64")
    team_games_recent = pd.Series(team_games_recent, dtype="float64")

    out = partial[["pitcher", "bf", "games"]].rename(
        columns={"bf": "bf_to_date", "games": "appearances"}).copy()
    out = out[out["appearances"] > 0]
    out["team_id"] = out["pitcher"].map(team_of)
    out = out[out["team_id"].notna()]
    if out.empty:
        return pd.DataFrame(columns=["pitcher", "team_id", "role",
                                     "appearances", "bf_to_date", "bf_ros"])

    raw_bf_per_app = out["bf_to_date"] / out["appearances"]
    out["role"] = role_of(raw_bf_per_app)
    prior_bf = out["role"].map(ROLE_BF_PER_APPEARANCE).astype(float)
    prior_rate = out["role"].map(ROLE_APPEARANCE_RATE).astype(float)

    # Batters faced per outing, regressed toward the role's average.
    bf_per_app = ((out["bf_to_date"] + BF_BALLAST_APPEARANCES * prior_bf)
                  / (out["appearances"] + BF_BALLAST_APPEARANCES))

    played = out["team_id"].map(team_games_played).astype(float)
    recent_played = out["team_id"].map(team_games_recent).astype(float)
    recent_app = out["pitcher"].map(
        recent.set_index("pitcher")["games"]).fillna(0.0).astype(float)

    def regressed_rate(appearances, games):
        games = games.fillna(0.0)
        return ((appearances + RATE_BALLAST_GAMES * prior_rate)
                / (games + RATE_BALLAST_GAMES))

    season_rate = regressed_rate(out["appearances"].astype(float), played)
    recent_rate = regressed_rate(recent_app, recent_played)
    rate = (1.0 - RECENT_WEIGHT) * season_rate + RECENT_WEIGHT * recent_rate

    left = out["team_id"].map(games_remaining).astype(float).fillna(0.0)
    out["bf_ros"] = (rate * left * bf_per_app).clip(lower=0.0)

    if active_fraction is not None:
        fraction = out["pitcher"].map(pd.Series(active_fraction, dtype="float64"))
        out["bf_ros"] = out["bf_ros"] * fraction.fillna(1.0).clip(0.0, 1.0)

    out["team_id"] = out["team_id"].astype("int64")
    return out[["pitcher", "team_id", "role", "appearances", "bf_to_date",
                "bf_ros"]].reset_index(drop=True)


# --- rates x workload --------------------------------------------------

def fip_constant(league: dict, lg_ra9: float = 4.30) -> float:
    """The additive constant putting FIP on a runs-per-nine scale, from the
    league's own rates through the cutoff."""
    per_bf = (FIP_COEF["hr"] * league["p_hr_rate"]
              + FIP_COEF["bb"] * league["p_bb_rate"]
              + FIP_COEF["k"] * league["p_k_rate"])
    return float(lg_ra9) - per_bf * float(league["bf_per_ip"])


def ros_pitching_line(bf_ros, k_rate, bb_rate, hr_rate, league: dict,
                      lg_ra9: float = 4.30) -> pd.DataFrame:
    """Rates x projected batters faced -> K, BB, HR and a FIP.

    The FIP is the same arithmetic station E's starter term runs, on the same
    coefficients, expressed per nine innings — so a pitcher whose rates equal
    the league's comes back at `lg_ra9` and the number is readable next to an
    ERA. Innings come from the league's batters faced per inning; this is a
    rate model, and projecting a pitcher's own innings per batter faced would
    be another ungated structural choice for no gain.
    """
    bf = np.asarray(bf_ros, dtype=float)
    k_rate, bb_rate, hr_rate = (np.asarray(x, dtype=float)
                                for x in (k_rate, bb_rate, hr_rate))
    per_bf = (FIP_COEF["hr"] * hr_rate + FIP_COEF["bb"] * bb_rate
              + FIP_COEF["k"] * k_rate)
    fip = per_bf * float(league["bf_per_ip"]) + fip_constant(league, lg_ra9)
    return pd.DataFrame({
        "bf": bf, "k": k_rate * bf, "bb": bb_rate * bf, "hr": hr_rate * bf,
        "ip": bf / float(league["bf_per_ip"]), "fip": fip,
    })


def build_pitcher_projections(
    as_of_date,
    seasons_table: pd.DataFrame,
    pa_frame: pd.DataFrame,
    *,
    team_of,
    team_games_played,
    team_games_recent,
    games_remaining,
    active_fraction=None,
    names: pd.Series | dict | None = None,
    teams: pd.DataFrame | None = None,
    season: int = SEASON,
    lg_ra9: float = 4.30,
    stuff_monthly: pd.DataFrame | None = None,
    stuff_pa_dir=None,
) -> pd.DataFrame:
    """The live rest-of-season pitcher projection, one row per projected pitcher.

    Returns a frame with `OUTPUT_COLUMNS`, sorted by projected FIP (best
    first) among pitchers with a positive projected workload.

    Args:
        stuff_monthly: `data/features/pitching_stuff_monthly.parquet`, loaded
            (`src.data.pitching_stuff.load_monthly`). Required for any
            component whose `LIVE_ENGINE` is `stuff_additive`; a component
            missing it falls back to `marcel_pitcher_tuned` for that build
            alone (`engine_providers`).
        stuff_pa_dir: the PA-outcomes directory the stuff engine's
            walk-forward fit trains on (`data/parquet/pa_outcomes`).

    Two provenance entries ride along on `.attrs`: `pitcher_engine_used`
    (component -> the engine that actually filled its `marcel` column) and
    `stuff_features_through` (the last date those features cover — see
    `stuff_cutoff`).
    """
    as_of = _as_date(as_of_date)
    partial = partial_season(pa_frame, as_of, season)
    recent = partial_season(pa_frame, as_of, season,
                            since=as_of - pd.Timedelta(days=RECENT_DAYS))
    league = league_rates(partial)

    workload = projected_batters_faced(
        partial, recent, team_games_played, team_games_recent,
        games_remaining, team_of, active_fraction)
    workload = workload[workload["bf_ros"] > 0]

    rates = pitcher_rates(seasons_table, partial, season,
                          stuff_monthly=stuff_monthly,
                          stuff_pa_dir=stuff_pa_dir, as_of=as_of)
    out = workload.merge(rates, on="pitcher", how="left")

    # Marcel with no trials at all is the league rate; a pitcher with projected
    # work and no professional record gets that rather than an empty row.
    for component in SERVED_COMPONENTS:
        column = f"{COMPONENT_PREFIX[component]}_rate_marcel"
        if column in out.columns:
            out[column] = out[column].astype(float).fillna(league[component])

    out["as_of"] = as_of.date().isoformat()
    if names is not None:
        lookup = names if isinstance(names, pd.Series) else pd.Series(names)
        out["name"] = out["pitcher"].map(lookup)
    else:
        out["name"] = pd.NA
    if teams is not None and len(teams):
        out["team_abbrev"] = out["team_id"].map(teams.set_index("team_id")["abbrev"])
    else:
        out["team_abbrev"] = pd.NA

    line = ros_pitching_line(out["bf_ros"], out["k_rate_marcel"],
                             out["bb_rate_marcel"], out["hr_rate_marcel"],
                             league, lg_ra9)
    out["k_ros"] = line["k"].values
    out["bb_ros"] = line["bb"].values
    out["hr_ros"] = line["hr"].values
    out["fip_ros"] = line["fip"].values

    for column in OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    result = (out.loc[:, OUTPUT_COLUMNS]
              .sort_values("fip_ros", ascending=True, na_position="last")
              .reset_index(drop=True))
    result.attrs.update(rates.attrs)
    return result
