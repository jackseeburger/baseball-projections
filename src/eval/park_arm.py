"""`marcel_tuned_park` — tuned Marcel with the batter's home park put back on.

docs/park-factors.md pre-registered one arm for the park factors
(`src/data/park_components.py`) and this is it: the served baseline
(`marcel_tuned`) multiplied by the factor of the park the batter's club plays
in, for the season being projected. No fitted parameter joins the arm — the
factor is a measurement, not a coefficient — so whatever it gains or loses is
the measurement's, which is what makes it readable against architecture.md
§3's effect floor without a recalibration control to subtract.

    predicted = marcel_tuned(...) x factor(home_park(batter), predict_year)

**Whose park.** The batter's club, defined as the club he has taken the most
plate appearances for in the *partial* season — the same information the
training frame has, cut at the same date, so the arm cannot see a trade or a
call-up that has not happened yet. A batter with no PA before the cutoff has
no club and gets exactly 1.0, which is also what a club missing from the
factor table gets.

**Full strength, and why that is worth saying.** A hitter plays about half his
games at home, so the effect of his park on his *rest-of-season rate* is
roughly the square root of the factor, not the factor. The pre-registration
specifies the factor, so the arm applies the factor; `strength` exists so the
half-strength version can be scored alongside as a diagnostic rather than
argued about. `strength=0` reproduces the base provider exactly, which makes
the comparison a clean nesting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.baselines import marcel_tuned

# Component → the column in the park-factor table. The three per-PA binomials
# match `src.models.pa_components.RATE_COMPONENTS[...].park_factor_col` (pinned
# by tests/test_eval/test_park_arm.py); BABIP and ISO have no entry there —
# they are not per-PA binomials and the hierarchical model does not serve
# them — but the harness scores them, so they are named here.
PARK_FACTOR_COLUMNS = {
    "k_rate": "k_park_factor",
    "bb_rate": "bb_park_factor",
    "hr_rate": "hr_park_factor",
    "babip": "babip_park_factor",
    "iso": "iso_park_factor",
}

ARM = "marcel_tuned_park"
# The half-strength diagnostic. Not pre-registered: reported separately, and
# never in place of `ARM`.
HALF_ARM = "marcel_tuned_park_half"
HALF_STRENGTH = 0.5


def home_team_map(pa: pd.DataFrame, cutoff_date=None,
                  id_col: str = "batter") -> pd.Series:
    """batter → the club he has the most PA for, strictly before `cutoff_date`.

    `pa` is the PA-outcome frame for the season being projected. The batting
    club is read off the half-inning exactly as `pa_rate.prepare_model_data`
    reads it, so "his park" means the same thing to the arm and to the
    hierarchical model.

    Ties go to the club whose abbreviation sorts first — arbitrary, but fixed,
    so a rebuild does not move a projection.
    """
    df = pa
    if cutoff_date is not None:
        df = df[pd.to_datetime(df["game_date"]) < pd.Timestamp(cutoff_date)]
    if df.empty:
        return pd.Series(dtype="object", name="team")
    bat_team = np.where(df["inning_topbot"] == "Top",
                        df["away_team"], df["home_team"])
    counts = (pd.DataFrame({id_col: df[id_col].to_numpy(), "team": bat_team})
              .groupby([id_col, "team"]).size().rename("pa").reset_index())
    counts = counts.sort_values([id_col, "pa", "team"],
                                ascending=[True, False, True])
    out = counts.drop_duplicates(id_col).set_index(id_col)["team"]
    out.name = "team"
    return out


def park_multipliers(park_factors: pd.DataFrame, season: int,
                     component: str) -> dict:
    """{team: factor} for one season and component; empty when unavailable.

    Unavailable covers every honest way this can go missing — no table, no
    rows for the season, no column for the component — and every one of them
    lands the arm on a neutral 1.0 rather than on some other component's
    factor.
    """
    col = PARK_FACTOR_COLUMNS.get(component)
    if park_factors is None or col is None or col not in park_factors.columns:
        return {}
    year_col = "game_year" if "game_year" in park_factors.columns else "year"
    rows = park_factors[park_factors[year_col].astype(int) == int(season)]
    rows = rows[rows[col].notna() & (rows[col] > 0)]
    return {str(t): float(f) for t, f in zip(rows["team"], rows[col])}


def park_factor_series(ids, team_map: pd.Series, multipliers: dict,
                       strength: float = 1.0) -> pd.Series:
    """The multiplier each player's projection is scaled by (1.0 by default)."""
    teams = pd.Series(list(ids)).map(team_map)
    f = teams.map(multipliers).astype("float64").fillna(1.0)
    if strength != 1.0:
        f = np.power(f, float(strength))
    return pd.Series(np.asarray(f, dtype="float64"))


def park_provider(multipliers: dict, team_map: pd.Series, *,
                  base=marcel_tuned, strength: float = 1.0, params=None):
    """`base` scaled by the batter's park factor — a harness provider.

    `base` defaults to `marcel_tuned`, the arm the pre-registration reads
    against, and takes the same (train, spec, predict_year) signature every
    provider takes. Coverage is the base's coverage exactly: a batter the
    factors know nothing about is projected, unscaled, rather than dropped,
    so adding this arm to a cell cannot change the common-player set the
    other arms are scored on.
    """
    def provider(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
        out = base(train, spec, predict_year, **({"params": params}
                                                 if params is not None else {}))
        out = out.copy()
        ident = spec.id_col
        f = park_factor_series(out[ident], team_map, multipliers, strength)
        out["predicted"] = out["predicted"].to_numpy() * f.to_numpy()
        return out

    return provider


def park_providers(park_factors: pd.DataFrame, pa: pd.DataFrame,
                   cutoff_date, season: int, component: str,
                   id_col: str = "batter", half: bool = True) -> dict:
    """{arm name: provider} for one (component, season, cutoff) cell.

    Returns `{}` when the factor table has nothing to say about this season
    and component — an arm that would be `marcel_tuned` renamed is worse than
    no arm, because it would look like a dead heat rather than a missing
    measurement.
    """
    multipliers = park_multipliers(park_factors, season, component)
    if not multipliers:
        return {}
    team_map = home_team_map(pa, cutoff_date, id_col)
    out = {ARM: park_provider(multipliers, team_map)}
    if half:
        out[HALF_ARM] = park_provider(multipliers, team_map,
                                      strength=HALF_STRENGTH)
    return out


def coverage(team_map: pd.Series, multipliers: dict, ids) -> dict:
    """How much of the scored population the arm actually moved.

    An arm whose factor is 1.0 for a third of its batters is a diluted arm,
    and that dilution belongs next to its Δ MAE rather than in a footnote.
    """
    ids = list(ids)
    teams = pd.Series(ids).map(team_map)
    known = teams.isin(list(multipliers))
    f = park_factor_series(ids, team_map, multipliers)
    return {
        "n": len(ids),
        "n_with_park": int(known.sum()),
        "share_with_park": float(known.mean()) if len(ids) else 0.0,
        "mean_abs_log_factor": float(np.mean(np.abs(np.log(f)))) if len(ids) else 0.0,
    }


__all__ = ["ARM", "HALF_ARM", "HALF_STRENGTH", "PARK_FACTOR_COLUMNS",
           "coverage", "home_team_map", "park_factor_series",
           "park_multipliers", "park_provider", "park_providers"]
