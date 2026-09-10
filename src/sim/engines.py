"""The engines station A serves, made available to station E's per-game chain.

Station E prices a game from two rate tables: every pitcher's K, BB+HBP and HR
per batter faced (`src/sim/starters.marcel_rates`) and every batter's five
component rates (`src/sim/lineups.marcel_rates`). Both were stock Marcel —
5/4/3 recency, twice the published stabilization point as ballast, no age term
— while the site has been serving something else on the same players for two
stations: a tuned Marcel whose ballast, recency weights and age curve were
fitted walk-forward (`src/eval/marcel_params.json`,
`src/eval/marcel_pitcher_params.json`), and on top of that an additive
correction from tracked contact quality (hitters) and pitch characteristics
(pitchers).

This module is the switch that lets the chain read those engines, as three
nested rungs:

    1. `tuned`     the pitcher table on `marcel_pitcher_tuned`'s params and the
                   hitter table on `marcel_tuned`'s, ages from the Chadwick
                   register — the estimator, nothing added
    2. `stuff`     ...and `stuff_additive` on the pitcher components the site
                   serves that correction on
    3. `contact`   ...and `contact_additive` on the five hitter components

and one transform across the ladder, `match` (BAS-91,
docs/chain-engines-matched.md): each rate column of a rung's table centred on
the stock table's weighted mean at the same `as_of` and its deviations rescaled
to the stock table's weighted sd. BAS-90 found the chain's ballasts, blend and
lineup weights were all chosen against the stock tables' spread and that the
rungs widen it; `match` puts a rung's *ordering* of players into the stock
table's level and spread, so the two mechanisms can be scored apart. It fits
nothing — the stock table is the rung-0 table on the identical count frame —
and it is a no-op at rung 0 by construction.

Each rung *contains* the one below it, and rung 0 — `ChainEngines()`, the
default — is the chain exactly as it is served today, to the last bit:
`starters.marcel_params(weights, ballast)` is the same MarcelParams translation
that module already documents, the hitter mirror of it reproduces
`lineups.marcel_rates`'s own arithmetic, and no correction is applied.
`tests/test_sim/test_engines.py` pins that.

**Nothing here fits anything.** A `ChainEngines` carries fitted artifacts that
a caller built — params, a `StuffFit`/`ContactFit` per component and the
monthly feature frame — so the chain stays pure and offline and the walk-forward
guarantee lives where the fit is made (`build_engines` below, the only function
in this file that reads anything). The corrections are read at the last month
boundary on or before the date being priced, never rounded forward, which is
the same rule and the same guard (`contact.assert_month_boundary`) the served
nightly uses; see `ros.contact_cutoff` and `pitcher_ros.stuff_cutoff`.

**The three places this is not literally the served column**, stated here
because they are what a reader of the evidence has to know:

  * Station E's walk rate is `(BB+HBP)/BF` — FIP treats a hit batsman as a
    walk — and the site's served BB engine is walks only (`p_bb_rate`). The
    stuff correction applied here is fitted on `p_bbhbp_rate`, the rate the
    chain actually consumes, rather than on the site's column: a correction
    fitted on one rate and added to another would carry the wrong intercept.
    `p_bbhbp_rate`'s own covariate share was withheld under the serving rule
    (docs/serving-rules.md); this is the chain reading the served *shape* on
    the rate it has, not a claim that the (BB+HBP) arm cleared a gate.
  * The hitter mirror is the same one rung down: the chain's `bbhbp` rides on
    `bb_rate`'s tuned params, the only walk-rate Marcel the search fitted.
  * The tuned params' `league_mode` never fires, because station E measures the
    league rate itself off the completed prior seasons and hands it in
    (`marcel_tuned(..., league=...)` skips `projected_league_rate` entirely).
    So "tuned" here means the ballast, the recency weights and the age curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

# Station E's short component names -> the station A component that projects it.
# The pitcher map is `starters.PITCHER_COMPONENTS`, repeated here so this module
# does not import station E at definition time.
PITCHER_COMPONENTS = {"k": "p_k_rate", "bbhbp": "p_bbhbp_rate", "hr": "p_hr_rate"}
HITTER_COMPONENTS = {"k": "k_rate", "bbhbp": "bb_rate", "hr": "hr_rate",
                     "iso": "iso", "babip": "babip"}
# (numerator, denominator) in `lineups.normalize_counts`'s own column names, so
# the harness estimator can be run on the frame station E already holds.
HITTER_COLUMNS = {"k": ("k", "pa"), "bbhbp": ("bbhbp", "pa"),
                  "hr": ("hr", "pa"), "iso": ("xb", "ab"),
                  "babip": ("hip", "bip")}

# Which components each correction covers. The pitcher pair is the two the site
# serves `stuff_additive` on (`pitcher_ros.LIVE_ENGINE`: BB/BF and HR/BF —
# K/BF's stuff arm is withheld), mapped onto station E's names; the hitter five
# are all of `ros.LIVE_ENGINE`.
STUFF_COMPONENTS = ("bbhbp", "hr")
CONTACT_COMPONENTS = ("k", "bbhbp", "hr", "iso", "babip")

# The clip the served providers apply to a corrected rate
# (`contact.ContactProviderConfig.clip`, `stuff.StuffProviderConfig.clip`).
RATE_CLIP = (1e-4, 0.999)


def month_boundary(as_of) -> pd.Timestamp:
    """The last month boundary on or before `as_of` — never rounded forward.

    The same function `ros.contact_cutoff` and `pitcher_ros.stuff_cutoff` are,
    written once here because the chain needs it for both sides. Rounding a
    cutoff *forward* would put batted balls and pitches thrown after the date
    being priced into the features, which is why `contact.assert_month_boundary`
    refuses anything but the first of a month rather than nudging one.
    """
    d = pd.Timestamp(as_of)
    return pd.Timestamp(year=d.year, month=d.month, day=1)


def weighted_moments(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """The weighted mean and weighted sd of `x`, on the rows both are finite on.

    The population form (`Σw(x-m)² / Σw`, no ddof correction), because the
    number this is used for is a ratio of two of them measured the same way on
    the same population, and a correction that cancels is a correction that can
    only be got wrong. A row with no weight — a player the weight column has no
    entry for — is out of both moments rather than counted as zero.
    """
    x = np.asarray(x, dtype="float64")
    w = np.asarray(w, dtype="float64")
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan"), float("nan")
    x, w = x[ok], w[ok]
    total = float(w.sum())
    mean = float((w * x).sum() / total)
    var = float((w * (x - mean) ** 2).sum() / total)
    return mean, float(np.sqrt(var))


def zero_slopes(params: dict) -> dict:
    """The recalibration control: tuned ballasts and weights, no age curve.

    What is left when the age term is taken out of a tuned Marcel is a
    re-ballasted Marcel — which is exactly the arm a serving decision has to
    beat, because "the tuned engine is better" and "the tuned engine's
    constants are better calibrated" are different claims.
    """
    return {c: p.replace(age_slope_young=0.0, age_slope_old=0.0)
            for c, p in params.items()}


@dataclass(frozen=True)
class ChainEngines:
    """Which of station A's engines the chain's two rate tables run on.

    Three nested flags and the artifacts they need. The default is the chain as
    served: stock constants, no correction, and every call below returns what
    it returned before this module existed.

    `stuff` and `contact` are corrections *on top of* the tuned estimator, so
    both imply `tuned` and `contact` implies `stuff` — the ladder is a nesting,
    checked in `__post_init__` rather than left to a caller to remember.
    """
    tuned: bool = False
    stuff: bool = False
    contact: bool = False
    # Whether the tuned params' age curve is applied. False with `tuned=True` is
    # the recalibration control: the fitted ballasts and recency weights with
    # the age slopes zeroed.
    age_slopes: bool = True
    # Whether each rate column is centred and rescaled onto the stock table's
    # weighted mean and weighted sd at the same `as_of` (`_match` below). A
    # parameter-free transform: the stock table is the rung-0 table on the
    # identical count frame, so nothing is fitted and nothing is read forward.
    match: bool = False

    # {station A component: MarcelParams}; None means the fitted files'.
    pitcher_params: dict | None = None
    hitter_params: dict | None = None
    # {player id: age in the season being priced}, from the Chadwick register.
    ages: dict = field(default_factory=dict)

    # {station E short name: StuffFit / ContactFit}, fitted walk-forward on
    # cell seasons strictly before `predict_year` by the caller.
    stuff_fits: dict = field(default_factory=dict)
    contact_fits: dict = field(default_factory=dict)
    stuff_monthly: pd.DataFrame | None = None
    contact_monthly: pd.DataFrame | None = None
    predict_year: int | None = None

    # Standardized covariates memoised per month boundary. A season is ~six
    # boundaries and ~160 dates, and the pitcher table is rebuilt twice a date,
    # so without this the same window is summed twenty-odd times a day.
    _z: dict = field(default_factory=dict, compare=False, repr=False)
    # The rung-0 tables `match` centres onto, memoised per (side, as_of, rows).
    # Matching doubles the rate-table work of a walk-forward without this,
    # because both tables are rebuilt more than once a date.
    _stock: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        if self.contact and not self.stuff:
            raise ValueError("the contact rung sits on the stuff rung")
        if (self.stuff or self.contact) and not self.tuned:
            raise ValueError("the correction rungs sit on the tuned estimator")

    # --- rung 1: which constants ------------------------------------------

    def _params(self, side: str, stock: dict) -> dict:
        """{station A component: MarcelParams} for one side of the ball.

        `stock` is what station E's own module builds from its `weights` and
        `ballast` arguments — the only thing rung 0 uses, so a caller that
        overrides a ballast on the command line still gets the ballast it asked
        for.
        """
        if not self.tuned:
            return stock
        if side == "pitcher":
            params = self.pitcher_params
            if params is None:
                from src.eval.pitchers import load_pitcher_params
                params = load_pitcher_params()
        else:
            params = self.hitter_params
            if params is None:
                from src.eval.baselines import load_marcel_params
                params = load_marcel_params()
        params = {c: params[c] for c in stock}
        return params if self.age_slopes else zero_slopes(params)

    def _age_factor(self, ids, params) -> np.ndarray:
        """The tuned age multiplier for one component, over a rate table's index.

        `marcel_tuned` applies this itself when the training frame carries an
        `age` column. Station E's frame is game logs and carries none, so the
        curve is applied here instead, on the age the player is in the season
        being priced — which is what `marcel_tuned`'s own
        `age_last + (predict_year - last)` comes to for every player who
        appeared in the last training season, and is defined for the ones who
        did not as well.
        """
        from src.eval.baselines import tuned_age_adjustment

        if not self.tuned or not self.age_slopes or not self.ages:
            return np.ones(len(ids))
        age = np.array([self.ages.get(int(i), np.nan) for i in ids],
                       dtype="float64")
        return tuned_age_adjustment(age, params)

    # --- rungs 2 and 3: the corrections -----------------------------------

    def _features(self, side: str, as_of) -> pd.DataFrame:
        """Standardized covariates at the last month boundary before `as_of`."""
        cutoff = month_boundary(as_of)
        key = (side, str(cutoff.date()))
        if key in self._z:
            return self._z[key]
        if side == "pitcher":
            from src.eval import stuff as engine

            z = engine.features_at_cutoff(self.stuff_monthly, cutoff,
                                          int(self.predict_year))
        else:
            from src.eval import contact as engine

            z = engine.features_at_cutoff(self.contact_monthly, "hitter",
                                          cutoff, int(self.predict_year))
        z = z.set_index("player")
        self._z[key] = z
        return z

    def _correct(self, rates: pd.DataFrame, side: str, as_of,
                 fits: dict, components) -> pd.DataFrame:
        """Add the fitted additive correction to each corrected rate column.

        Exactly what `contact_provider` / `stuff_provider` do to a baseline
        projection: a player with no tracked contact or pitches before the
        cutoff gets z = 0 on every covariate and therefore the recalibrated
        baseline, rather than being dropped — the rate table's population is
        the baseline's, so no term downstream is quietly computed on a
        different set of players.
        """
        if rates.empty or as_of is None:
            return rates
        z = self._features(side, as_of)
        zi = z.reindex(rates.index.to_numpy())
        out = rates.copy()
        for short in components:
            fit = fits.get(short)
            if fit is None:
                continue
            frame = pd.DataFrame(
                {f: (zi[f].fillna(0.0).to_numpy(dtype="float64")
                     if f in zi.columns else np.zeros(len(rates)))
                 for f in fit.features})
            pred = fit.predict(out[f"rate_{short}"].to_numpy(dtype="float64"),
                               frame)
            out[f"rate_{short}"] = np.clip(pred, *RATE_CLIP)
        return out

    # --- the level and spread match ---------------------------------------

    def _stock_table(self, side: str, counts: pd.DataFrame,
                     as_of_season: int, lg: dict, stock: dict,
                     as_of) -> pd.DataFrame:
        """The rung-0 table on the identical count frame, memoised.

        `STOCK` is the module's own every-flag-off engine, so this is literally
        the table the served chain prices that date from — not a re-derivation
        of it — and it is built from the same `counts`, `as_of_season`, league
        rates and `stock` params the rung above was handed. The memo key
        carries the row count as well as the date because the chain rebuilds a
        table from a *different* count frame on the same date only if a caller
        hands it one, and that has to miss the memo rather than silently reuse
        another population's moments.
        """
        key = (side, str(as_of), int(len(counts)))
        table = self._stock.get(key)
        if table is None:
            table = (STOCK.pitcher_rates(counts, as_of_season, lg, stock)
                     if side == "pitcher"
                     else STOCK.hitter_rates(counts, as_of_season, lg, stock))
            self._stock[key] = table
        return table

    def _match(self, rates: pd.DataFrame, side: str, counts: pd.DataFrame,
               as_of_season: int, lg: dict, stock: dict, as_of) -> pd.DataFrame:
        """Centre and rescale every rate column onto the stock table's moments.

        For each rate column, with `m` and `s` the weighted mean and weighted
        sd over the table's own population (weights `bf_weighted` for pitchers,
        `pa_weighted` for hitters — the effective sample behind each row):

            matched = (rung - m_rung) * (s_stock / s_rung) + m_stock

        so the matched table has the stock table's weighted level and the stock
        table's weighted spread, and keeps the rung's ordering of players
        inside them. That is the whole transform: no parameter is chosen, and
        the moments come from data through `as_of` on the same population, so
        there is nothing here to leak.

        Each side is weighted by its own table's weight column rather than by a
        shared one, because the tuned recency weights change `pa_weighted`
        itself; the equality that holds afterwards is "the matched table's
        weighted moments are the stock table's weighted moments", each read
        with the weights that table carries.

        Two things are deliberately not done. A player missing from the stock
        table keeps his rung value (same count frame, so none is expected).
        And at rung 0 the table *is* the stock table, so this returns it
        untouched rather than putting it through an arithmetic identity that
        IEEE 754 would round: `match=True` with every other flag off is the
        served chain to the bit, which `tests/test_sim/test_engines.py` pins.

        The rate clip is applied after, as it is after a correction.
        """
        if rates.empty or not (self.tuned or self.stuff or self.contact):
            return rates
        stock_table = self._stock_table(side, counts, as_of_season, lg, stock,
                                        as_of)
        if stock_table.empty:
            return rates
        weight_col = "bf_weighted" if side == "pitcher" else "pa_weighted"
        w_rung = rates[weight_col].to_numpy(dtype="float64")
        w_stock = stock_table[weight_col].to_numpy(dtype="float64")
        aligned = stock_table.reindex(rates.index.to_numpy())
        present = aligned[weight_col].notna().to_numpy()
        out = rates.copy()
        for column in [c for c in rates.columns if c.startswith("rate_")]:
            if column not in stock_table.columns:
                continue
            x = rates[column].to_numpy(dtype="float64")
            m_rung, s_rung = weighted_moments(x, w_rung)
            m_stock, s_stock = weighted_moments(
                stock_table[column].to_numpy(dtype="float64"), w_stock)
            if not (np.isfinite(m_rung) and np.isfinite(m_stock)):
                continue
            scale = (s_stock / s_rung
                     if np.isfinite(s_rung) and s_rung > 0
                     and np.isfinite(s_stock) else 1.0)
            matched = np.clip((x - m_rung) * scale + m_stock, *RATE_CLIP)
            out[column] = np.where(present, matched, x)
        return out

    # --- the two rate tables ----------------------------------------------

    def pitcher_rates(self, counts: pd.DataFrame, as_of_season: int, lg: dict,
                      stock: dict, as_of=None) -> pd.DataFrame:
        """`starters.marcel_rates`'s body, at whichever rung is on."""
        from src.eval.pitchers import pitcher_rates as station_a

        params = self._params("pitcher", stock)
        rates = station_a(counts, as_of_season, lg, params=params,
                          components=PITCHER_COMPONENTS)
        if self.tuned and self.age_slopes and len(rates):
            ids = rates.index.to_numpy()
            for short, component in PITCHER_COMPONENTS.items():
                rates[f"rate_{short}"] = (
                    rates[f"rate_{short}"].to_numpy(dtype="float64")
                    * self._age_factor(ids, params[component]))
        if self.stuff:
            rates = self._correct(rates, "pitcher", as_of, self.stuff_fits,
                                  STUFF_COMPONENTS)
        if self.match:
            rates = self._match(rates, "pitcher", counts, as_of_season, lg,
                                stock, as_of)
        return rates

    def hitter_rates(self, counts: pd.DataFrame, as_of_season: int, lg: dict,
                     stock: dict, as_of=None) -> pd.DataFrame:
        """The hitter mirror of `src/eval/pitchers.pitcher_rates`.

        One `marcel_tuned` call per component on station E's own count frame,
        with the component's numerator and denominator named by
        `HITTER_COLUMNS` and the league rate handed in — which is what makes
        rung 0 reproduce `lineups.marcel_rates`'s arithmetic to the bit
        (`tests/test_sim/test_engines.py`). It does not carry the Beta
        pseudo-counts that function returns: the tuned path runs a count
        through an age curve, which is not a ballast-on-a-count formula and has
        no pseudo-counts to expose. Nothing in the chain reads them —
        `batter_runs_lookup` reads the five rates — and `src/market/props.py`,
        which does, calls `lineups.marcel_rates` directly.
        """
        from src.eval.backtest import ComponentSpec
        from src.eval.baselines import marcel_tuned

        cols = ["pa_weighted", *[f"rate_{c}" for c in HITTER_COLUMNS]]
        empty = pd.DataFrame(columns=cols,
                             index=pd.Index([], name="batter", dtype="int64"))
        if counts.empty:
            return empty
        window = counts[counts["season"].between(as_of_season - 2, as_of_season)]
        if window.empty:
            return empty

        params = self._params("hitter", stock)
        out = None
        for short, (num, den) in HITTER_COLUMNS.items():
            component = HITTER_COMPONENTS[short]
            spec = ComponentSpec(component, num, den, binomial=True,
                                 id_col="batter")
            pred = marcel_tuned(counts, spec, as_of_season,
                                params=params[component],
                                league=float(lg[f"rate_{short}"]),
                                anchor_season=as_of_season)
            if self.tuned and self.age_slopes:
                pred = pred.copy()
                pred["predicted"] = (
                    pred["predicted"].to_numpy(dtype="float64")
                    * self._age_factor(pred["batter"].to_numpy(),
                                       params[component]))
            pred = pred.rename(columns={"predicted": f"rate_{short}"}
                               ).set_index("batter")
            out = pred if out is None else out.join(pred, how="outer")

        # The effective sample on the same anchored weights, normalised so the
        # current season counts 1 — `lineups.marcel_rates`'s `pa_weighted`.
        weights = params[HITTER_COMPONENTS["k"]].weights
        w = {as_of_season - i: float(v) / float(weights[0])
             for i, v in enumerate(weights)}
        pa = (window["pa"].astype(float) * window["season"].map(w)).groupby(
            window["batter"]).sum()
        out.insert(0, "pa_weighted", pa.reindex(out.index))
        if self.contact:
            out = self._correct(out, "hitter", as_of, self.contact_fits,
                                CONTACT_COMPONENTS)
        out = out[cols]
        if self.match:
            out = self._match(out, "hitter", counts, as_of_season, lg, stock,
                              as_of)
        return out


# The chain as served: every flag off, no artifact, no correction. A module
# constant rather than a fresh object per call so the memo dict is shared.
STOCK = ChainEngines()


# --- building one, which is the only thing here that reads a file ------------

def age_map(season: int, birthdates: pd.DataFrame | None = None) -> dict:
    """{player id: age as of June 30 of `season`} from the Chadwick register.

    The same age of record station A's tuned Marcel was fitted on
    (`src/data/birthdates.py`), for every id the register carries — pitchers and
    hitters share one map because they share one id space.
    """
    from src.data.birthdates import load_birthdates, seasonal_age

    bd = load_birthdates() if birthdates is None else birthdates
    ids = bd["batter"].to_numpy()
    ages = seasonal_age(bd, ids, season)
    return {int(i): float(a) for i, a in zip(ids, ages) if np.isfinite(a)}


def fit_side(side: str, seasons_table, monthly, pa_dir, predict_year,
             components) -> dict:
    """Walk-forward fits for every component of one side, in one pass over PA.

    `fit_live_stuff` / `fit_live_contact` each rebuild the whole cell table for
    the one component they fit; the cells cost eight seasons of plate
    appearances to build and do not depend on the component, so they are built
    once here and every component is fitted off the same frame. The split, the
    seasons and the `min_trials` filter are the ones those two functions use —
    this only hoists the loop, and `tests/test_sim/test_engines.py` pins the
    result against `fit_live_stuff` on one component.
    """
    if side == "pitcher":
        from src.eval import stuff as engine
        build, fit_one = engine.build_pitcher_cells, engine.fit_stuff
        names = [PITCHER_COMPONENTS[c] for c in components]
    else:
        from src.eval import contact as engine
        build, fit_one = engine.build_hitter_cells, engine.fit_contact
        names = [HITTER_COMPONENTS[c] for c in components]

    train_seasons = tuple(s for s in engine.LIVE_CELL_SEASONS
                          if s < int(predict_year))
    if not train_seasons:
        raise ValueError(
            f"no {side} training seasons strictly before {predict_year}")
    cells = build(seasons_table, pa_dir, names, seasons=train_seasons)
    cells = engine.attach_live_features(cells, monthly)
    return {short: fit_one(cells, name, features=engine.FEATURES,
                           fixed_base=True)
            for short, name in zip(components, names)}


def build_engines(rung: int, predict_year: int, *, recalibration: bool = False,
                  age_slopes: bool = True, match: bool = False,
                  pitcher_seasons=None, hitter_seasons=None, pa_dir=None,
                  stuff_monthly=None, contact_monthly=None,
                  birthdates=None) -> ChainEngines:
    """A `ChainEngines` for rung 0-3, with every artifact loaded and fitted.

    `rung` is 0 (the served chain), 1 (`tuned`), 2 (`+stuff`) or 3
    (`+contact`); `recalibration` turns rung 1 into the control with the age
    slopes zeroed, and `age_slopes=False` does the same at any rung (the two
    are the same switch — `recalibration` is the name BAS-90's control was
    scored under and is kept so its arms still build). `match` centres and
    rescales each rate column onto the stock table's weighted moments. Every
    fit is walk-forward on cell seasons strictly before `predict_year`, because
    `fit_side` passes that year to the same filter the served nightly uses —
    nothing here can see the season being scored.
    """
    if rung == 0:
        # Matching a rung-0 table is the identity (`_match`), so this is the
        # served chain whatever `match` says; it is carried on the object so a
        # caller inspecting the engine sees the flag it asked for.
        return ChainEngines(match=match)
    ages = age_map(int(predict_year), birthdates)
    eng = ChainEngines(tuned=True, ages=ages,
                       age_slopes=age_slopes and not recalibration,
                       match=match,
                       predict_year=int(predict_year))
    if rung >= 2:
        fits = fit_side("pitcher", pitcher_seasons, stuff_monthly, pa_dir,
                        predict_year, STUFF_COMPONENTS)
        eng = replace(eng, stuff=True, stuff_fits=fits,
                      stuff_monthly=stuff_monthly, _z={})
    if rung >= 3:
        fits = fit_side("hitter", hitter_seasons, contact_monthly, pa_dir,
                        predict_year, CONTACT_COMPONENTS)
        eng = replace(eng, contact=True, contact_fits=fits,
                      contact_monthly=contact_monthly, _z={})
    return eng


__all__ = ["CONTACT_COMPONENTS", "ChainEngines", "HITTER_COLUMNS",
           "HITTER_COMPONENTS", "PITCHER_COMPONENTS", "RATE_CLIP", "STOCK",
           "STUFF_COMPONENTS", "age_map", "build_engines", "fit_side",
           "month_boundary", "weighted_moments", "zero_slopes"]
