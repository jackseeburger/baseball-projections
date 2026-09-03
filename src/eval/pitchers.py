"""Station A, the pitcher side: rate providers for K%, BB%, HR/BF and BABIP.

The hitter half of station A projects five rates from a season table plus the
current season through a cutoff. This module is the same machine pointed at
the other end of the plate appearance. It adds no new estimator: the
components register themselves into `backtest.COMPONENTS` under a `p_` prefix
and `marcel_pitcher` is `baselines.marcel_tuned` carrying pitcher constants,
so every provider, split, score and paired test in the harness works on
pitchers with no special case.

    p_k_rate       K / BF
    p_bb_rate      BB / BF                    — walks only, the site's BB%
    p_bbhbp_rate   (BB + HBP) / BF            — what FIP and station E use
    p_hr_rate      HR / BF
    p_babip        (H - HR) / (AB - K - HR + SF)

**Two walk components, because the definitions do not overlap.** Station E's
FIP term has always folded hit batsmen in with walks — FIP treats them
identically and so does the published stabilization point it regresses with —
while a site column labelled BB% has to mean walks. Both are projected. They
are not interchangeable: `p_bbhbp_rate` is the one `src/sim/starters.py`
consumes, and it is the one whose stock arm reproduces what station E computed
before this module existed.

**Where stock reproduces station E, and where it cannot.** With
`PITCHER_STOCK_PARAMS`, `marcel_pitcher` returns exactly the rates
`starters.marcel_rates` returned for the same counts frame — same 5/4/3
recency, same 2x published-stabilization ballasts, same league rate — and
`tests/test_sim/test_starters.py` pins that on a fixed pitcher-season.
`starters.py` is now a caller of this module rather than a second
implementation. Two definitional gaps remain, and neither is reproducible
because station E never had them:

  * **BB%** — station E has only the walks-plus-hit-batsmen rate (above).
  * **BABIP against** — FIP is *defined* to ignore balls in play, so station E
    has no such rate and never did. This module projects one because the site
    wants it and because "does BABIP-against carry any signal" is a question
    the harness can now answer; nothing in the odds chain reads it.

**Ballast units.** `MarcelParams.ballast` is denominated at the *average* year
weight (see `src/eval/baselines.py`), while the published stabilization points
— and `starters.STABILIZATION_BF` — are real batters faced, which land on the
*most recent* season's weight. The two differ by `w0 / mean(w)`, exactly 1.25
for 5/4/3, so `PITCHER_STOCK_PARAMS` carries the stabilization points scaled by
that factor. Nothing else in the file needs to know.

**Ages** come from the Chadwick register (June 30), the same age of record the
hitter side uses. Stock Marcel has no pitcher aging curve at all — station E
never aged anybody — so the stock params' slopes are zero and every age effect
in the tuned arm is something the search found rather than something assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval import intraseason, tuning
from src.eval.backtest import COMPONENTS, ComponentSpec
from src.eval.baselines import (
    SEASON_TO_DATE_BALLAST,
    MarcelParams,
    full_seasons,
    league_average,
    marcel_tuned,
    previous_season,
    season_to_date,
)

# --- the components ----------------------------------------------------------

PITCHER_COMPONENTS: dict[str, ComponentSpec] = {
    "p_k_rate": ComponentSpec("p_k_rate", "k", "bf", binomial=True,
                              id_col="pitcher"),
    "p_bb_rate": ComponentSpec("p_bb_rate", "bb", "bf", binomial=True,
                               id_col="pitcher"),
    "p_bbhbp_rate": ComponentSpec("p_bbhbp_rate", "bbhbp", "bf", binomial=True,
                                  id_col="pitcher"),
    "p_hr_rate": ComponentSpec("p_hr_rate", "hr", "bf", binomial=True,
                               id_col="pitcher"),
    "p_babip": ComponentSpec("p_babip", "hits_in_play", "bip", binomial=True,
                             id_col="pitcher"),
}
COMPONENT_ORDER = ("p_k_rate", "p_bb_rate", "p_bbhbp_rate", "p_hr_rate",
                   "p_babip")
# The three station E reads, keyed by the short name `src/sim/starters.py` uses.
STATION_E_COMPONENTS = {"k": "p_k_rate", "bbhbp": "p_bbhbp_rate",
                        "hr": "p_hr_rate"}

# Season-table columns a pitcher season needs, whether it came from the Stats
# API table or from aggregating PA rows.
COUNT_COLUMNS = ["bf", "k", "bb", "hbp", "bbhbp", "hr", "ab", "h", "sf",
                 "bip", "hits_in_play"]

# --- the constants -----------------------------------------------------------

# Published split-half stabilization points for pitcher rates, in batters
# faced (balls in play for BABIP). These are the same numbers
# `src/sim/starters.STABILIZATION_BF` carries for the three FIP inputs; the
# two extra rows are the walks-only rate, which stabilizes with walks, and
# BABIP against, the famous non-stabilizer that is the whole content of DIPS.
STABILIZATION = {
    "p_k_rate": 70.0,       # BF
    "p_bb_rate": 170.0,     # BF
    "p_bbhbp_rate": 170.0,  # BF
    "p_hr_rate": 1300.0,    # BF
    "p_babip": 2000.0,      # BIP
}
# Reliability is not projection: talent moves between samples, so projection
# systems regress about twice as hard as the stabilization point alone implies.
# Station E chose 2.0 walk-forward on 2025 and found the curve flat above it.
PROJECTION_MULTIPLIER = 2.0
STOCK_WEIGHTS = (5.0, 4.0, 3.0)
# ballast at the average year weight = ballast in real trials * w0 / mean(w).
BALLAST_SCALE = STOCK_WEIGHTS[0] / float(np.mean(STOCK_WEIGHTS))   # 1.25

PITCHER_STOCK_PARAMS: dict[str, MarcelParams] = {
    name: MarcelParams(
        ballast=stab * PROJECTION_MULTIPLIER * BALLAST_SCALE,
        weights=STOCK_WEIGHTS,
        peak_age=27.0,
        age_slope_young=0.0,
        age_slope_old=0.0,
    )
    for name, stab in STABILIZATION.items()
}

# The "just use this year" arm regresses with the reliability point itself,
# not the projection ballast — same convention as the hitter components.
SEASON_TO_DATE_BALLAST.update(STABILIZATION)

# Which way the aging curve turns. For a pitcher a *high* strikeout rate is
# good, so K% peaks at the peak age; walks, home runs and hits on balls in
# play are all bad, so those curves trough there. This is the hitter table's
# sign flipped on K% and only on K%, which is what makes it worth writing
# down: the same rate means opposite things at the two ends of the PA.
tuning.AGE_DIRECTION.update({
    "p_k_rate": 1.0, "p_bb_rate": -1.0, "p_bbhbp_rate": -1.0,
    "p_hr_rate": -1.0, "p_babip": -1.0,
})

PITCHER_PARAMS_PATH = Path(__file__).with_name("marcel_pitcher_params.json")


def load_pitcher_params(
    path: str | Path | None = None, strict: bool = False
) -> dict[str, MarcelParams]:
    """Fitted per-component params from `src/eval/marcel_pitcher_params.json`.

    Falls back to `PITCHER_STOCK_PARAMS` for any component the file does not
    carry, so `marcel_pitcher_tuned` is always callable and an unfit component
    simply *is* stock.
    """
    path = Path(path) if path is not None else PITCHER_PARAMS_PATH
    if not path.exists():
        if strict:
            raise FileNotFoundError(
                f"{path} not found — run scripts/tune_marcel_pitchers.py")
        return dict(PITCHER_STOCK_PARAMS)
    blob = json.loads(path.read_text())
    out = dict(PITCHER_STOCK_PARAMS)
    for name, d in blob.get("components", {}).items():
        out[name] = MarcelParams.from_dict(d)
    return out


def save_pitcher_params(params: dict[str, MarcelParams],
                        path: str | Path | None = None, **extra) -> Path:
    """Write the params file `load_pitcher_params` reads. `extra` is metadata."""
    path = Path(path) if path is not None else PITCHER_PARAMS_PATH
    blob = {**extra, "components": {k: v.to_dict() for k, v in params.items()}}
    path.write_text(json.dumps(blob, indent=2) + "\n")
    return path


# --- aggregation -------------------------------------------------------------

def aggregate_pa_pitchers(pa: pd.DataFrame, season: int | None = None
                          ) -> pd.DataFrame:
    """PA-level outcomes rolled up per *pitcher*, in the season-table schema.

    The same call the hitter side makes, grouped by the other id on the row.
    `pa` becomes `bf` — a batter faced is a plate appearance seen from the
    mound — and `bbhbp` is added because FIP wants the pair.
    """
    g = intraseason.aggregate_pa(pa, season, id_col="pitcher")
    g = g.rename(columns={"pa": "bf"})
    g["bbhbp"] = g["bb"] + g["hbp"]
    return g[["pitcher", "season", *COUNT_COLUMNS, "games",
              "first_game_date", "last_game_date"]]


def partial_and_realized(pa: pd.DataFrame, cutoff_date, season=None
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Season-shaped (partial, realized) pitcher frames either side of a cutoff.

    `partial` is everything strictly before the cutoff, flagged `partial=True`;
    `realized` is everything on or after it. `intraseason.assert_split_clean`
    checks both, and it never looks at the id column, so the hitter leakage
    guard guards this path unchanged.
    """
    before, after = intraseason.split_at_cutoff(pa, cutoff_date)
    return (aggregate_pa_pitchers(before, season).assign(partial=True),
            aggregate_pa_pitchers(after, season).assign(partial=False))


# Register with the intra-season harness so `backtest("p_k_rate",
# cutoff_date=...)` splits PA rows by pitcher without the caller saying so.
intraseason.AGGREGATORS["pitcher"] = partial_and_realized

# Register the components so `backtest()` / `score()` find them by name.
COMPONENTS.update(PITCHER_COMPONENTS)


def normalize_pitcher_seasons(seasons: pd.DataFrame) -> pd.DataFrame:
    """Fill in whatever derived count columns a pitcher season table is missing.

    Accepts the Stats API table (`data/parquet/pitcher_seasons_api.parquet`)
    or anything with the raw counts, and returns a frame the components can be
    read off. Identities are the harness's: BB+HBP, BIP = AB - K - HR + SF,
    hits in play = H - HR.
    """
    out = seasons.copy()
    if "bbhbp" not in out.columns:
        out["bbhbp"] = out["bb"] + out["hbp"]
    if "bip" not in out.columns:
        out["bip"] = out["ab"] - out["k"] - out["hr"] + out["sf"]
    if "hits_in_play" not in out.columns:
        out["hits_in_play"] = out["h"] - out["hr"]
    return out


# --- the providers -----------------------------------------------------------

def marcel_pitcher(train: pd.DataFrame, spec, predict_year: int, *,
                   params: "MarcelParams | dict | None" = None,
                   **kwargs) -> pd.DataFrame:
    """Marcel for pitchers on stock constants: 5/4/3 recency, 2x the published
    stabilization point as ballast, no age term.

    The dumb arm the tuned one has to beat, and — on the three components
    station E reads — the estimator station E has been running all along.
    """
    return marcel_tuned(train, spec, predict_year,
                        params=params or PITCHER_STOCK_PARAMS, **kwargs)


def marcel_pitcher_tuned(train: pd.DataFrame, spec, predict_year: int, *,
                         params: "MarcelParams | dict | None" = None,
                         **kwargs) -> pd.DataFrame:
    """`marcel_pitcher` with the fitted constants from
    `src/eval/marcel_pitcher_params.json` (stock for any unfit component)."""
    return marcel_tuned(train, spec, predict_year,
                        params=params or load_pitcher_params(), **kwargs)


def marcel_pitcher_preseason(train: pd.DataFrame, spec, predict_year: int,
                             **kwargs) -> pd.DataFrame:
    """`marcel_pitcher` with the partial current season withheld — the control
    that isolates what in-season information is worth."""
    return marcel_pitcher(full_seasons(train), spec, predict_year, **kwargs)


def marcel_pitcher_tuned_preseason(train: pd.DataFrame, spec,
                                   predict_year: int, **kwargs) -> pd.DataFrame:
    """`marcel_pitcher_tuned` with the partial current season withheld."""
    return marcel_pitcher_tuned(full_seasons(train), spec, predict_year, **kwargs)


def pitcher_tuned_provider(params: "MarcelParams | dict | None" = None):
    """Bind params to `marcel_pitcher_tuned` to get a plain provider."""

    def provider(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
        return marcel_pitcher_tuned(train, spec, predict_year, params=params)

    return provider


PITCHER_BASELINES = {
    "marcel_pitcher": marcel_pitcher,
    "previous_season": previous_season,
    "league_average": league_average,
}
PITCHER_INTRASEASON_BASELINES = {
    **PITCHER_BASELINES,
    "season_to_date": season_to_date,
    "marcel_pitcher_preseason": marcel_pitcher_preseason,
    "marcel_pitcher_tuned": marcel_pitcher_tuned,
    "marcel_pitcher_tuned_preseason": marcel_pitcher_tuned_preseason,
}


# --- the one rate table ------------------------------------------------------

def pitcher_rates(counts: pd.DataFrame, as_of_season: int, league: dict,
                  params: dict | None = None,
                  components: dict[str, str] | None = None) -> pd.DataFrame:
    """Per-pitcher rates per batter faced, in the frame `src/sim/starters.py` wants.

    This is the single entry point station E and the site share. `counts` is
    one row per pitcher-season (any mix of completed seasons and a partial
    current one); `league` is the pooled league rate per component, keyed
    `rate_k` / `rate_bbhbp` / `rate_hr`, measured by the caller — station E
    pools only the *completed* seasons for it because the current-season logs
    it holds are starters-only.

    Returns a frame indexed by pitcher with `bf_weighted` (the effective sample
    behind the rates) and one `rate_<c>` column per component. A pitcher with
    no history inside the three-season window is simply absent, which is what
    `starters.starter_ra9_lookup` treats as league average.

    The recency window is anchored on `as_of_season`, not on the last season
    present in `counts`: on opening day the partial season is empty and the
    weights must not silently shift a slot.
    """
    components = components or STATION_E_COMPONENTS
    params = params or PITCHER_STOCK_PARAMS
    cols = ["bf_weighted", *[f"rate_{c}" for c in components]]
    empty = pd.DataFrame(columns=cols,
                         index=pd.Index([], name="pitcher", dtype="int64"))
    if counts.empty or not components:
        return empty

    window = counts[counts["season"].between(as_of_season - 2, as_of_season)]
    if window.empty:
        return empty

    out = None
    for short, component in components.items():
        spec = PITCHER_COMPONENTS[component]
        pred = marcel_tuned(counts, spec, as_of_season,
                            params=params[component],
                            league=float(league[f"rate_{short}"]),
                            anchor_season=as_of_season)
        pred = pred.rename(columns={"predicted": f"rate_{short}"}).set_index("pitcher")
        out = pred if out is None else out.join(pred, how="outer")

    # The effective sample, on the same anchored weights the rates used,
    # normalised so the current season counts 1 — that makes `bf_weighted`
    # readable as real batters faced rather than as five times them.
    w = {as_of_season - i: float(v) / STOCK_WEIGHTS[0]
         for i, v in enumerate(STOCK_WEIGHTS)}
    bf = (window["bf"].astype(float) * window["season"].map(w)).groupby(
        window["pitcher"]).sum()
    out.insert(0, "bf_weighted", bf.reindex(out.index))
    return out[cols]


__all__ = [
    "COMPONENT_ORDER", "PITCHER_COMPONENTS", "PITCHER_STOCK_PARAMS",
    "PITCHER_BASELINES", "PITCHER_INTRASEASON_BASELINES", "STABILIZATION",
    "STATION_E_COMPONENTS", "aggregate_pa_pitchers", "load_pitcher_params",
    "marcel_pitcher", "marcel_pitcher_preseason", "marcel_pitcher_tuned",
    "marcel_pitcher_tuned_preseason", "normalize_pitcher_seasons",
    "partial_and_realized", "pitcher_rates", "pitcher_tuned_provider",
    "save_pitcher_params",
]
