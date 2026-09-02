"""Station A, live: the rest-of-season projection the site serves.

The preseason Bayesian components are what `public/data/comparison.json`
shows today. The intra-season harness (`src/eval/intraseason.py`,
[backtest-baselines.md](../../docs/backtest-baselines.md)) measured that they
lose the rest-of-season question to plain Marcel *fed the partial current
season* — by 6% on K% at a May 1 cutoff and 11% at Aug 1, and on every
component except BABIP. The gate rule (architecture.md §3) says the model in
production is the one that wins the harness, so this module builds that
model's numbers and the site serves them.

    rest-of-season projection = Marcel(prior seasons + 2026 through
                                       as_of − 1 day)  ×  station B's
                                       projected rest-of-season PA

**Which Marcel.** `marcel_tuned` — the same estimator with its ballast,
recency weights and age curve fitted walk-forward on 2020–2024 and frozen in
`src/eval/marcel_params.json`. It cleared the gate out of sample on 2025, 2026
and the three 2026 cutoffs (15/25 component × cell cells, pooled −1.10% ± 0.36
of stock Marcel's MAE; the tuning section of
[backtest-baselines.md](../../docs/backtest-baselines.md)), so the gate rule
puts it in production. The gain is BABIP (−3.3%) and K% (−2.4%); BB% keeps
Tango's constants, and HR/PA and ISO come out even.

Nothing here re-implements Marcel. `src/eval/baselines.marcel_tuned` is called
with exactly the training frame the harness builds at a cutoff
(`intraseason.build_training_frame`), so the model that ships is bit-for-bit
the arm that was scored. `marcel_tuned_preseason` — the same Marcel with the
partial season withheld — and the preseason Bayesian file ride along as
labelled comparison columns, which is what makes the improvement legible on
the page.

**The cutoff is exclusive.** A game played *on* `as_of` has not finished when
the morning's projection is made, so `split_at_cutoff` keeps only
`game_date < as_of`: the partial season runs through `as_of − 1 day`. This is
the same convention station B uses for playing time.

**From rates to a line.** The five components are rates over different
denominators, so the counting line is rebuilt with the standard identities,
using the player's *own* regressed walk and strikeout rates (not league ones)
wherever a denominator depends on them:

    BB  = bb_rate × PA          HBP = hbp_rate × PA     SF = sf_rate × PA
    AB  = PA − BB − HBP − SF    K   = k_rate × PA       HR = hr_rate × PA
    BIP = AB − K − HR + SF      H   = babip × BIP + HR
    2B + 2·3B + 3·HR = iso × AB

`hbp_rate` and `sf_rate` are league rates through the cutoff — nobody projects
hit-by-pitch — and sacrifice bunts and interference are folded into AB rather
than carried (together ~0.4% of PA, so AB runs ~0.5% high; the effect on wOBA
is under a point). The extra-base points are split into doubles and triples at
the league's triples-per-double ratio, matching `scripts/assemble_and_compare.py`
so the site's preseason and live wOBA are computed the same way.

wOBA uses the FanGraphs 2024 linear weights (`WOBA_WEIGHTS` below); its
denominator AB + BB + HBP + SF is exactly PA under the identity above, so
`woba_ros` is a per-PA rate and multiplying it back by `pa_ros` gives wOBA
points. Fed the league-average line it returns the league wOBA (.3139 on 2026
through Sept 1), which is the sanity test in `tests/test_projections/test_ros.py`.

Everything is a pure function over DataFrames — season table, PA frame,
playing-time frame in, projection frame out — so it unit-tests without a
network or R2. The fetch/assemble layer is `scripts/build_ros_projections.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval import baselines
from src.eval.backtest import COMPONENTS
from src.eval.intraseason import aggregate_pa, build_training_frame, split_at_cutoff

SEASON = 2026

# Component key -> the prefix the site's columns use. `babip` and `iso` are
# already rates, so `babip_rate_marcel` reads oddly; it is the price of one
# uniform `{stat}_rate_{arm}` naming across all five.
COMPONENT_PREFIX = {
    "k_rate": "k", "bb_rate": "bb", "hr_rate": "hr", "babip": "babip", "iso": "iso",
}
COMPONENT_ORDER = ("k_rate", "bb_rate", "hr_rate", "babip", "iso")

# The live arm first: it is the projection, the other two are comparisons.
ARMS = ("marcel", "marcel_preseason", "bayes")
MARCEL_ARMS = {"marcel": "marcel", "marcel_preseason": "marcel_preseason"}

# The engine, named once. Everything downstream reads it from here rather than
# hard-coding a string: `scripts/build_ros_projections.py` stamps it into the
# document as `engine`, and `scripts/build_accuracy_json.py` uses it to pick
# the arm the accuracy page marks as live — so the scoreboard cannot end up
# scoring a model the site does not serve.
LIVE_ENGINE = "marcel_tuned"
LIVE_PROVIDERS = {
    "marcel": baselines.marcel_tuned,
    "marcel_preseason": baselines.marcel_tuned_preseason,
}

# FanGraphs 2024 linear weights, same constants as scripts/assemble_and_compare.py.
WOBA_WEIGHTS = {
    "bb": 0.690, "hbp": 0.722, "single": 0.883,
    "double": 1.244, "triple": 1.569, "hr": 2.015,
}
# Triples per double among non-home-run extra bases. 2B + 2·3B = iso·AB − 3·HR
# with 3B/2B held at this ratio.
TRIPLES_PER_DOUBLE = 0.12

# Fallbacks for the two rates nobody projects, used only when the partial
# season is empty (a cutoff before Opening Day). 2026 through Sept 1: .0114
# and .0071.
LEAGUE_HBP_RATE = 0.0114
LEAGUE_SF_RATE = 0.0071

OUTPUT_COLUMNS = (
    ["batter", "name", "team_id", "team_abbrev", "as_of", "pa_ros"]
    + [f"{COMPONENT_PREFIX[c]}_rate_{arm}" for c in COMPONENT_ORDER for arm in ARMS]
    + ["k_ros", "bb_ros", "hr_ros", "woba_ros"]
)


def _as_date(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


# --- the partial season ------------------------------------------------

def partial_season(pa_frame: pd.DataFrame, as_of, season: int = SEASON) -> pd.DataFrame:
    """The current season through `as_of − 1 day`, in the season schema.

    The same frame `intraseason.partial_and_realized` puts on the training
    side of a cutoff, flagged `partial=True` so the baselines know to treat it
    as an incomplete year rather than a full one.
    """
    as_of = _as_date(as_of)
    year = pa_frame
    if "game_year" in year.columns:
        year = year[year["game_year"] == season]
    before, _ = split_at_cutoff(year, as_of)
    return aggregate_pa(before, season).assign(partial=True)


def league_rates(partial: pd.DataFrame) -> dict[str, float]:
    """Trials-weighted league rates through the cutoff, including HBP and SF.

    Doubles as the zero-history fallback: Marcel with no trials at all *is*
    the league rate, so a September call-up with no professional record gets
    exactly that rather than being dropped off the page.
    """
    out = {"hbp_rate": LEAGUE_HBP_RATE, "sf_rate": LEAGUE_SF_RATE}
    for c in COMPONENT_ORDER:
        out[c] = float("nan")
    if partial.empty:
        return out
    total = {c: float(partial[c].sum()) for c in
             ["pa", "ab", "bip", "k", "bb", "hr", "hbp", "sf", "hits_in_play", "xb_points"]}

    def ratio(num: str, den: str, default=float("nan")) -> float:
        return total[num] / total[den] if total[den] > 0 else default

    out["k_rate"] = ratio("k", "pa")
    out["bb_rate"] = ratio("bb", "pa")
    out["hr_rate"] = ratio("hr", "pa")
    out["babip"] = ratio("hits_in_play", "bip")
    out["iso"] = ratio("xb_points", "ab")
    out["hbp_rate"] = ratio("hbp", "pa", LEAGUE_HBP_RATE)
    out["sf_rate"] = ratio("sf", "pa", LEAGUE_SF_RATE)
    return out


# --- the rates ---------------------------------------------------------

def marcel_rates(
    seasons_table: pd.DataFrame,
    partial: pd.DataFrame,
    predict_year: int = SEASON,
    components=COMPONENT_ORDER,
) -> pd.DataFrame:
    """The live Marcel arm and its preseason control, one row per batter.

    Columns: `batter` plus `{prefix}_rate_marcel` and
    `{prefix}_rate_marcel_preseason`. The training frame is the harness's own
    (`build_training_frame`): prior full seasons from the season table plus the
    partial current season, with ages carried forward for Marcel's age
    adjustment.

    Both arms are `marcel_tuned` — the fitted-constants Marcel that cleared
    the gate (see the module docstring). The column *keys* stay `marcel` and
    `marcel_preseason`: they name a slot on the page ("the live arm", "the
    same model with 2026 withheld"), not a set of constants, and the document
    `build_ros_projections.py` writes says which engine filled them. Both arms
    move together on purpose — the preseason column exists to isolate the
    value of in-season information *with the model held fixed*, and pairing a
    tuned live arm with a stock control would confound the two.
    """
    train = build_training_frame(seasons_table, partial, predict_year)
    if train.empty:
        raise ValueError("no training data: both the season table and the "
                         "partial season are empty")
    has_prior = not baselines.full_seasons(train).empty

    out = pd.DataFrame({"batter": pd.unique(train["batter"])})
    for component in components:
        spec = COMPONENTS[component]
        prefix = COMPONENT_PREFIX[component]
        for arm, provider in LIVE_PROVIDERS.items():
            column = f"{prefix}_rate_{arm}"
            if arm == "marcel_preseason" and not has_prior:
                # No completed season to look back on — the control arm has
                # nothing to say, and saying league average would misrepresent
                # it as a real preseason projection.
                out[column] = np.nan
                continue
            pred = provider(train, spec, predict_year)
            out = out.merge(
                pred.rename(columns={"predicted": column})[["batter", column]],
                on="batter", how="left")
    return out


def bayes_rates(
    frames: dict[str, pd.DataFrame] | None,
    projection_year: int = SEASON,
    components=COMPONENT_ORDER,
) -> pd.DataFrame:
    """The preseason Bayesian rate per component, from the projection files.

    `frames` maps a component key to that component's projection frame
    (`data/projections/{component}_projections_2026.parquet`, columns `batter`,
    `projection_year`, `projected_{component}`). A component with no file is
    simply absent — the column comes back all-null and the page shows a dash.
    """
    out = pd.DataFrame({"batter": pd.Series(dtype="int64")})
    for component in components:
        column = f"{COMPONENT_PREFIX[component]}_rate_bayes"
        frame = (frames or {}).get(component)
        source = f"projected_{component}"
        if frame is None or source not in frame.columns:
            out[column] = pd.Series(dtype="float64")
            continue
        rows = frame
        if "projection_year" in rows.columns:
            rows = rows[rows["projection_year"] == projection_year]
        rows = (rows[["batter", source]].dropna(subset=["batter"])
                .drop_duplicates(subset="batter", keep="first")
                .rename(columns={source: column}))
        out = out.merge(rows, on="batter", how="outer")
    return out


# --- rates x playing time ---------------------------------------------

def ros_counting_line(
    pa_ros,
    k_rate,
    bb_rate,
    hr_rate,
    babip,
    iso,
    hbp_rate: float = LEAGUE_HBP_RATE,
    sf_rate: float = LEAGUE_SF_RATE,
) -> pd.DataFrame:
    """Component rates × projected PA → the counting line, and wOBA.

    Returns pa, bb, hbp, sf, ab, k, hr, bip, hits_in_play, h, xb_points,
    triples, doubles, singles, woba — see the module docstring for the
    identities. Everything is clipped at zero: a very high projected ISO and a
    very low projected BABIP can otherwise imply negative singles.
    """
    pa = np.asarray(pa_ros, dtype=float)
    k_rate, bb_rate, hr_rate, babip, iso = (
        np.asarray(x, dtype=float) for x in (k_rate, bb_rate, hr_rate, babip, iso))

    bb = bb_rate * pa
    hbp = float(hbp_rate) * pa
    sf = float(sf_rate) * pa
    ab = np.maximum(pa - bb - hbp - sf, 0.0)
    k = k_rate * pa
    hr = hr_rate * pa
    bip = np.maximum(ab - k - hr + sf, 0.0)
    hits_in_play = babip * bip
    h = hits_in_play + hr

    xb_points = iso * ab
    non_hr_xb = np.maximum(xb_points - 3.0 * hr, 0.0)
    triples = TRIPLES_PER_DOUBLE * non_hr_xb / (1.0 + TRIPLES_PER_DOUBLE)
    doubles = np.maximum(non_hr_xb - 2.0 * triples, 0.0)
    singles = np.maximum(h - hr - doubles - triples, 0.0)

    denominator = ab + bb + hbp + sf          # == pa, by construction
    woba_points = (WOBA_WEIGHTS["bb"] * bb
                   + WOBA_WEIGHTS["hbp"] * hbp
                   + WOBA_WEIGHTS["single"] * singles
                   + WOBA_WEIGHTS["double"] * doubles
                   + WOBA_WEIGHTS["triple"] * triples
                   + WOBA_WEIGHTS["hr"] * hr)
    woba = np.divide(woba_points, denominator,
                     out=np.full_like(pa, np.nan), where=denominator > 0)

    return pd.DataFrame({
        "pa": pa, "bb": bb, "hbp": hbp, "sf": sf, "ab": ab, "k": k, "hr": hr,
        "bip": bip, "hits_in_play": hits_in_play, "h": h,
        "xb_points": xb_points, "triples": triples, "doubles": doubles,
        "singles": singles, "woba": woba,
    })


def build_ros_projections(
    as_of_date,
    seasons_table: pd.DataFrame,
    pa_frame_2026: pd.DataFrame,
    playing_time: pd.DataFrame,
    *,
    bayes_frames: dict[str, pd.DataFrame] | None = None,
    names: pd.Series | dict | None = None,
    teams: pd.DataFrame | None = None,
    season: int = SEASON,
) -> pd.DataFrame:
    """The live rest-of-season projection, one row per projected hitter.

    Args:
        as_of_date: the morning the projection is made. The partial season
            runs through the day before; a game on this date is future
            information.
        seasons_table: prior full seasons in the harness schema (batter,
            season, pa/ab/k/bb/hr/xb_points/bip/hits_in_play, optional age).
        pa_frame_2026: PA-level outcomes for the current season
            (`src/data/pa_outcomes.load_pa_outcomes`).
        playing_time: station B's output — batter, team_id,
            projected_pa_ros. Only hitters with `projected_pa_ros > 0` are
            projected; a hitter on the IL projects to zero PA and has no
            rest-of-season line to show.
        bayes_frames: component → preseason Bayesian projection frame, for the
            comparison column. Optional.
        names: batter → display name.
        teams: team_id → abbrev frame (columns `team_id`, `abbrev`).

    Returns a frame with `OUTPUT_COLUMNS`, sorted by projected wOBA.
    """
    as_of = _as_date(as_of_date)
    partial = partial_season(pa_frame_2026, as_of, season)
    league = league_rates(partial)

    played = playing_time[playing_time["projected_pa_ros"] > 0].copy()
    played["batter"] = played["batter"].astype("int64")
    # A hitter traded mid-season can appear on two roster snapshots; keep the
    # row with the most projected PA, which is the club he is on now.
    played = (played.sort_values("projected_pa_ros", ascending=False)
              .drop_duplicates(subset="batter", keep="first"))

    rates = marcel_rates(seasons_table, partial, season)
    bayes = bayes_rates(bayes_frames, season)

    out = played[["batter", "team_id", "projected_pa_ros"]].rename(
        columns={"projected_pa_ros": "pa_ros"})
    out = out.merge(rates, on="batter", how="left").merge(bayes, on="batter", how="left")

    # Marcel with no trials at all is the league rate; a hitter with projected
    # PA and no professional record gets that rather than an empty row.
    for component in COMPONENT_ORDER:
        column = f"{COMPONENT_PREFIX[component]}_rate_marcel"
        if column in out.columns:
            out[column] = out[column].astype(float).fillna(league[component])

    out["as_of"] = as_of.date().isoformat()
    if names is not None:
        lookup = names if isinstance(names, pd.Series) else pd.Series(names)
        out["name"] = out["batter"].map(lookup)
    else:
        out["name"] = pd.NA
    if teams is not None and len(teams):
        abbrev = teams.set_index("team_id")["abbrev"]
        out["team_abbrev"] = out["team_id"].map(abbrev)
    else:
        out["team_abbrev"] = pd.NA

    line = ros_counting_line(
        out["pa_ros"],
        out[f"{COMPONENT_PREFIX['k_rate']}_rate_marcel"],
        out[f"{COMPONENT_PREFIX['bb_rate']}_rate_marcel"],
        out[f"{COMPONENT_PREFIX['hr_rate']}_rate_marcel"],
        out[f"{COMPONENT_PREFIX['babip']}_rate_marcel"],
        out[f"{COMPONENT_PREFIX['iso']}_rate_marcel"],
        hbp_rate=league["hbp_rate"], sf_rate=league["sf_rate"],
    )
    out["k_ros"] = line["k"].values
    out["bb_ros"] = line["bb"].values
    out["hr_ros"] = line["hr"].values
    out["woba_ros"] = line["woba"].values

    for column in OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return (out.loc[:, OUTPUT_COLUMNS]
            .sort_values("woba_ros", ascending=False, na_position="last")
            .reset_index(drop=True))
