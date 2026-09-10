"""Per-park, per-season component factors for K, BB, HR, BABIP and ISO.

The park factors this repo already had were two different things, neither of
them usable by the rate models:

* `src/sim/park.py` (BAS-57) — a *run environment* multiplier per venue, built
  from schedule scores, home/road paired and regressed toward 1 with a
  pseudo-count ballast, renormalised so the games-weighted league mean is
  exactly 1. That is the right shape; it is just about runs, keyed on
  `venue_id`, and it needs the network.
* `src/data/park_factors.py` — team-season rates against the league average,
  with no home/road split at all (its own docstring says so) and the current
  season in its own numerator. `src/sim/run_environment.py` declined to use
  it, and it is the file the PA models' `k_park_factor` offset has been
  reading whenever `data/parquet/park_factors.parquet` happened to exist.

This module builds the missing artifact — the same construction as
`src/sim/park.py`, per PA component instead of per run, off the PA outcome
parquets:

    factor(park, season, component)
        = regressed, league-centred multiplier on the component's rate

**Home/away paired, both halves.** For park *p* in a window of seasons:

    home half     the club that hosts at p, batting at p, against the same
                  club batting anywhere else
    visitor half  every club batting at p, against those same clubs batting
                  anywhere else, each weighted by how much it batted at p

and the raw factor is the mean of the two ratios. Both halves control for who
plays there: a club with a patient lineup lifts the numerator and the
denominator of the home half alike, and the visitor half is a different set of
hitters entirely, so a park that only looks extreme because of its tenant
shows up in one half and not the other.

**Walk-forward.** The factor stamped for season Y is built from Y-3..Y-1 and
nothing else, so no PA of the season being projected can inform the offset
applied to it. (`src/sim/park.py` uses two seasons for the same reason and
says so; three is what docs/park-factors.md pre-registered here, and a park is
about 6,100 PA a season, so a 3-season window is ~18,000 trials.)

**Ballast.** `factor = (n·raw + b) / (n + b)` with `n` the trials at the park
in the window — `b = 0` is the raw split, `b = inf` is no park term at all, so
the ballast sweep is a clean nesting, exactly as it is for runs. `b` is chosen
once, per component, by leave-one-season-out persistence on 2015-2019
(`loso_persistence`) and then frozen in `FROZEN_BALLAST`; nothing later
re-tunes it.

**Centred on the league.** After regression the factors are renormalised so
the trials-weighted mean over the parks of a season is exactly 1. A park
redistributes a league's strikeouts across venues; it cannot change how many
the league has. Without this an offset applied to a projected rate would move
the projected league level, which is a different (and unearned) claim.

**What a "park" is here.** The PA parquet carries `home_team`, not a venue id,
so the key is the club that hosted — which is what the rate models index on
(`prepare_model_data` builds its park offset over `bat_team`) and what
docs/park-factors.md asked for. The cost is that a club changing parks carries
its old park's factor for three seasons: ATH (Oakland Coliseum through 2024,
Sutter Health Park from 2025) and TB (Tropicana Field through 2024, Steinbrenner
Field in 2025) are the live cases in this window, and `KNOWN_PARK_CHANGES`
records them so a reader is not left to spot it. Neutral-site games (London,
Mexico City, the Field of Dreams) are likewise charged to the nominal home
club; there are a handful a season.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ─── components ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParkComponent:
    """One rate whose park effect is measured: numerator and denominator as
    per-PA indicator/count columns, and the name of the factor column.

    The denominators are the ones `src.eval.backtest.HITTER_COMPONENTS` uses,
    rebuilt per PA rather than per season so they can be summed by park:

        k/bb/hr   PA
        babip     balls in play  = PA - BB - HBP - SH - CI - K - HR
                                 (identically `ab - k - hr + sf`)
        iso       AB             = PA - BB - HBP - SF - SH - CI,
                  numerator 2B + 2·3B + 3·HR (extra-base points)
    """
    name: str
    factor_col: str


PARK_COMPONENTS: tuple[ParkComponent, ...] = (
    ParkComponent("k_rate", "k_park_factor"),
    ParkComponent("bb_rate", "bb_park_factor"),
    ParkComponent("hr_rate", "hr_park_factor"),
    ParkComponent("babip", "babip_park_factor"),
    ParkComponent("iso", "iso_park_factor"),
)
COMPONENT_NAMES = tuple(c.name for c in PARK_COMPONENTS)
# component -> the column `src.models.pa_components.RateComponent.park_factor_col`
# expects to find (pinned against that registry by the tests, so the two
# vocabularies cannot drift).
FACTOR_COLUMNS = {c.name: c.factor_col for c in PARK_COMPONENTS}

# PA-parquet columns the build reads. Nothing else is loaded: 2M PAs x 46
# columns is 400 MB, this is 60.
PA_COLUMNS = ["game_year", "home_team", "away_team", "inning_topbot", "event",
              "is_k", "is_bb", "is_hbp", "is_hit", "is_hr", "is_double",
              "is_triple"]

# Counts a window sums. `<name>_n` is the numerator, `<name>_d` the trials.
COUNT_COLUMNS = tuple(f"{c}_{s}" for c in COMPONENT_NAMES for s in ("n", "d"))

# Seasons the ballast is chosen on, once. Frozen afterwards (see FROZEN_BALLAST).
BALLAST_SEASONS = (2015, 2016, 2017, 2018, 2019)
# Trials of ballast the sweep chooses among. In the component's own denominator
# (PA, BIP or AB); a park sees roughly 6,100 PA a season, so 18,000 is about
# the whole three-season window and `inf` is the model with no park term.
BALLAST_GRID = (0.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 6000.0,
                8000.0, 12000.0, 16000.0, 32000.0, 64000.0, float("inf"))
# Seasons pooled into the factor stamped for season Y: Y-3..Y-1.
WINDOW_SEASONS = 3

# Chosen by `loso_persistence` on BALLAST_SEASONS (see the sidecar JSON written
# by scripts/build_park_factors.py for the table this came off) and frozen.
FROZEN_BALLAST = {
    "k_rate": 2000.0,
    "bb_rate": 8000.0,
    "hr_rate": 3000.0,
    "babip": 3000.0,
    "iso": 4000.0,
}

# Clubs whose home park changed inside the window this artifact covers. The
# factor is keyed on the club, so for `seasons` after a move the club carries
# its previous park until the window rolls over. Recorded, not corrected:
# correcting it would need a venue id the PA parquet does not carry.
KNOWN_PARK_CHANGES = {
    "ATH": {"from": "Oakland Coliseum", "to": "Sutter Health Park (Sacramento)",
            "first_season": 2025},
    "TB": {"from": "Tropicana Field", "to": "George M. Steinbrenner Field",
           "first_season": 2025, "note": "Tropicana Field in 2026 again"},
}

DEFAULT_PA_DIR = Path("data/parquet/pa_outcomes")
DEFAULT_PATH = Path("data/features/park_factors.parquet")

SAC_FLY_EVENTS = {"sac_fly", "sac_fly_double_play"}
SAC_BUNT_EVENTS = {"sac_bunt", "sac_bunt_double_play"}
INTERFERENCE_EVENTS = {"catcher_interf"}


# ─── counts ────────────────────────────────────────────────────────────────

def pa_counts(pa: pd.DataFrame) -> pd.DataFrame:
    """PA rows → per (game_year, park, bat_team) numerator/denominator sums.

    `park` is the hosting club (`home_team`); `bat_team` is who was batting,
    read off `inning_topbot` exactly as `pa_rate.prepare_model_data` does.
    Every window figure below is a sum over rows of this table, which is 30
    parks x 30 clubs x a season — small enough that the whole 2015-2026 build
    is one groupby and some arithmetic.
    """
    df = pa
    event = df["event"] if "event" in df.columns else pd.Series("", index=df.index)
    sf = event.isin(SAC_FLY_EVENTS).astype("int64")
    sh = event.isin(SAC_BUNT_EVENTS).astype("int64")
    ci = event.isin(INTERFERENCE_EVENTS).astype("int64")
    k = df["is_k"].astype("int64")
    bb = df["is_bb"].astype("int64")
    hbp = df["is_hbp"].astype("int64")
    hr = df["is_hr"].astype("int64")
    hit = df["is_hit"].astype("int64")
    doubles = df["is_double"].astype("int64")
    triples = df["is_triple"].astype("int64")

    ab = 1 - bb - hbp - sf - sh - ci
    bip = ab - k - hr + sf

    out = pd.DataFrame({
        "game_year": df["game_year"].astype("int64"),
        "park": df["home_team"].astype(str),
        "bat_team": np.where(df["inning_topbot"] == "Top",
                             df["away_team"].astype(str),
                             df["home_team"].astype(str)),
        "k_rate_n": k, "k_rate_d": 1,
        "bb_rate_n": bb, "bb_rate_d": 1,
        "hr_rate_n": hr, "hr_rate_d": 1,
        "babip_n": hit - hr, "babip_d": bip,
        "iso_n": doubles + 2 * triples + 3 * hr, "iso_d": ab,
    })
    return (out.groupby(["game_year", "park", "bat_team"], as_index=False)
            [list(COUNT_COLUMNS)].sum())


def load_pa_counts(seasons, pa_dir: str | Path = DEFAULT_PA_DIR) -> pd.DataFrame:
    """`pa_counts` over the PA parquets for `seasons`, concatenated.

    A season with no parquet on disk is skipped rather than raised on — the
    caller reports which seasons it actually got (2020 is the one that has
    ever been missing here, and it is the season a reader most needs told
    about).
    """
    pa_dir = Path(pa_dir)
    frames = []
    for year in sorted(set(int(s) for s in seasons)):
        path = pa_dir / f"pa_outcomes_{year}.parquet"
        if not path.exists():
            continue
        frames.append(pa_counts(pd.read_parquet(path, columns=PA_COLUMNS)))
    if not frames:
        return pd.DataFrame(columns=["game_year", "park", "bat_team",
                                     *COUNT_COLUMNS])
    return pd.concat(frames, ignore_index=True)


# ─── the raw home/away paired split ────────────────────────────────────────

def _rate(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else float("nan")


def window_raw(counts: pd.DataFrame, seasons) -> pd.DataFrame:
    """Raw (unregressed) factor and trials per (park, component) over `seasons`.

    Returns [park, component, raw, n, home_ratio, visitor_ratio]: the two
    halves are kept alongside their mean because they are the check on each
    other — a park where they disagree wildly is measuring its tenant, not
    its dimensions.

    `n` is the component's trials *at the park* in the window, which is what
    the ballast is a pseudo-count of.
    """
    seasons = sorted(set(int(s) for s in seasons))
    c = counts[counts["game_year"].isin(seasons)]
    rows = []
    if c.empty:
        return pd.DataFrame(columns=["park", "component", "raw", "n",
                                     "home_ratio", "visitor_ratio"])
    by_park = c.groupby(["park", "bat_team"], as_index=False)[list(COUNT_COLUMNS)].sum()
    totals = c.groupby("bat_team")[list(COUNT_COLUMNS)].sum()

    for park, g in by_park.groupby("park"):
        g = g.set_index("bat_team")
        for comp in COMPONENT_NAMES:
            num, den = f"{comp}_n", f"{comp}_d"
            at_n = g[num]
            at_d = g[den]
            home_ratio = float("nan")
            if park in g.index:
                h_at_n, h_at_d = float(at_n[park]), float(at_d[park])
                h_all_n = float(totals.loc[park, num])
                h_all_d = float(totals.loc[park, den])
                home_ratio = _rate(h_at_n, h_at_d) / _rate(
                    h_all_n - h_at_n, h_all_d - h_at_d)

            vis = g.index[g.index != park]
            v_at_n = float(at_n[vis].sum())
            v_at_d = float(at_d[vis].sum())
            # Each visitor's rate elsewhere, weighted by how much it batted
            # here: a division rival that visits nine times should count for
            # nine times as much as an interleague club that visits once.
            w, r = [], []
            for v in vis:
                else_n = float(totals.loc[v, num]) - float(at_n[v])
                else_d = float(totals.loc[v, den]) - float(at_d[v])
                if else_d > 0 and at_d[v] > 0:
                    w.append(float(at_d[v]))
                    r.append(_rate(else_n, else_d))
            visitor_ratio = float("nan")
            if w and v_at_d > 0:
                v_else = float(np.average(r, weights=w))
                visitor_ratio = _rate(v_at_n, v_at_d) / v_else if v_else > 0 else float("nan")

            halves = [x for x in (home_ratio, visitor_ratio) if np.isfinite(x)]
            raw = float(np.mean(halves)) if halves else 1.0
            rows.append({"park": park, "component": comp, "raw": raw,
                         "n": float(at_d.sum()),
                         "home_ratio": home_ratio,
                         "visitor_ratio": visitor_ratio})
    return pd.DataFrame(rows)


def regress(raw: pd.DataFrame, ballast) -> pd.DataFrame:
    """Regress `window_raw`'s output toward 1 and centre it on the league.

    `ballast` is trials of ballast, one number or a {component: number} map.
    `inf` returns exactly 1.0 everywhere — the model with no park term — and
    0.0 returns the raw split, both without special-casing anywhere else.

    Centring is trials-weighted per component, so the mean factor over a
    season's parks is exactly 1 and the offset cannot move the league level.
    """
    if raw.empty:
        return raw.assign(factor=[])
    out = raw.copy()
    b = (ballast if isinstance(ballast, dict)
         else {c: float(ballast) for c in COMPONENT_NAMES})
    factors = []
    for comp, g in out.groupby("component"):
        bb = float(b.get(comp, 0.0))
        if not np.isfinite(bb):
            f = pd.Series(1.0, index=g.index)
        else:
            f = (g["n"] * g["raw"] + bb) / (g["n"] + bb)
            mean = float(np.average(f, weights=g["n"])) if g["n"].sum() > 0 else 1.0
            if mean > 0:
                f = f / mean
        factors.append(f)
    out["factor"] = pd.concat(factors).sort_index()
    return out


def window_for(season: int, available, n_seasons: int = WINDOW_SEASONS) -> tuple[int, ...]:
    """The seasons the factor stamped for `season` is built from: the
    `n_seasons` most recent seasons strictly before it that have data.

    Strictly before, so no PA of the season being projected can reach the
    offset applied to it. A season missing from `available` (2020, when its
    parquet is not on disk) is skipped and an older one takes its place,
    which keeps the window three seasons long rather than silently two.
    """
    have = sorted({int(s) for s in available if int(s) < int(season)},
                  reverse=True)
    return tuple(sorted(have[:n_seasons]))


def factors_for_season(counts: pd.DataFrame, season: int, ballast=None,
                       n_seasons: int = WINDOW_SEASONS) -> pd.DataFrame:
    """The walk-forward factor table stamped for one season."""
    ballast = FROZEN_BALLAST if ballast is None else ballast
    window = window_for(season, counts["game_year"].unique(), n_seasons)
    if not window:
        return pd.DataFrame(columns=["park", "component", "raw", "n", "factor"])
    out = regress(window_raw(counts, window), ballast)
    return out.assign(game_year=int(season),
                      window=[list(window)] * len(out))


def build_table(counts: pd.DataFrame, seasons, ballast=None,
                n_seasons: int = WINDOW_SEASONS) -> pd.DataFrame:
    """Long [team, game_year, component, factor, n, window] over `seasons`.

    Long is the logical shape (the key is park x season x component); it is
    pivoted to one column per component on the way to disk because
    `pa_rate.prepare_model_data` reads `RateComponent.park_factor_col` off a
    wide row. `wide_table` does that.
    """
    frames = []
    for season in sorted(set(int(s) for s in seasons)):
        f = factors_for_season(counts, season, ballast, n_seasons)
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["team", "game_year", "component",
                                     "factor", "n", "window"])
    out = pd.concat(frames, ignore_index=True).rename(columns={"park": "team"})
    return out[["team", "game_year", "component", "factor", "raw", "n",
                "home_ratio", "visitor_ratio", "window"]]


def wide_table(long: pd.DataFrame) -> pd.DataFrame:
    """[team, game_year, k_park_factor, bb_park_factor, ...], the shape
    `pa_rate.load_park_factors` hands to `prepare_model_data`.

    `n_pa` rides along as the K% denominator (plate appearances at the park
    in the window) so a reader can see how much sample is behind a row
    without opening the sidecar.
    """
    if long.empty:
        return pd.DataFrame(columns=["team", "game_year", *FACTOR_COLUMNS.values()])
    wide = long.pivot_table(index=["team", "game_year"], columns="component",
                            values="factor", aggfunc="first")
    wide = wide.rename(columns=FACTOR_COLUMNS).reset_index()
    pa = (long[long["component"] == "k_rate"]
          .set_index(["team", "game_year"])["n"].rename("n_pa"))
    wide = wide.merge(pa.reset_index(), on=["team", "game_year"], how="left")
    cols = ["team", "game_year", *[FACTOR_COLUMNS[c] for c in COMPONENT_NAMES], "n_pa"]
    return wide[cols].sort_values(["game_year", "team"]).reset_index(drop=True)


# ─── choosing the ballast, once ────────────────────────────────────────────

def loso_windows(seasons) -> dict:
    """held-out season → the other seasons nearest it, `WINDOW_SEASONS` long.

    Leave one season out, and build the estimate from a window the same
    length as production's — a four-season pool would choose a ballast for a
    sample size the stamped factors never see.
    """
    seasons = sorted(int(s) for s in seasons)
    out = {}
    for y in seasons:
        others = sorted((s for s in seasons if s != y),
                        key=lambda s: (abs(s - y), s))
        out[y] = tuple(sorted(others[:WINDOW_SEASONS]))
    return out


def loso_persistence(counts: pd.DataFrame, seasons=BALLAST_SEASONS,
                     grid=BALLAST_GRID) -> pd.DataFrame:
    """Leave-one-season-out persistence of the regressed factor, per ballast.

    For each held-out season the factor is built from the neighbouring
    seasons and compared, across parks, with what that park actually did in
    the held-out season (its own single-season raw split, centred the same
    way). Scored on the log scale:

        rmse   root mean squared log error against the held-out season —
               the number the ballast is chosen on. The target is noisy, so
               the level of the rmse is not interpretable; the *ordering*
               across ballasts is, because every ballast faces the same noise.
        corr   Pearson correlation of the two log factors across parks. A
               shrinkage that is nearly affine barely moves this, which is
               exactly why it is reported next to the rmse rather than being
               the criterion.

    Returns one row per (component, ballast, held-out season) plus pooled
    rows with `season = None`.
    """
    windows = loso_windows(seasons)
    targets = {y: regress(window_raw(counts, [y]), 0.0) for y in windows}
    # The raw split does not depend on the ballast, so it is computed once per
    # held-out season and every candidate ballast is applied to it. (Doing it
    # the other way round — the obvious loop — recomputes the same window
    # `len(grid) * len(COMPONENT_NAMES)` times and turns a two-second sweep
    # into two minutes.)
    raws = {y: window_raw(counts, window) for y, window in windows.items()}
    rows = []
    for b in grid:
        estimates = {y: regress(raw, b) for y, raw in raws.items()}
        for comp in COMPONENT_NAMES:
            errs, xs, ys = [], [], []
            for y in windows:
                est = estimates[y]
                e = est[est["component"] == comp].set_index("park")["factor"]
                t = targets[y]
                t = t[t["component"] == comp].set_index("park")["factor"]
                common = e.index.intersection(t.index)
                le = np.log(e.loc[common].to_numpy(dtype="float64"))
                lt = np.log(t.loc[common].to_numpy(dtype="float64"))
                ok = np.isfinite(le) & np.isfinite(lt)
                le, lt = le[ok], lt[ok]
                if len(le) < 2:
                    continue
                rows.append({
                    "component": comp, "ballast": float(b), "season": int(y),
                    "n_parks": int(len(le)),
                    "rmse": float(np.sqrt(np.mean((le - lt) ** 2))),
                    "corr": float(np.corrcoef(le, lt)[0, 1])
                    if np.std(le) > 0 else float("nan"),
                })
                errs.append((le - lt) ** 2)
                xs.append(le)
                ys.append(lt)
            if errs:
                le = np.concatenate(xs)
                lt = np.concatenate(ys)
                rows.append({
                    "component": comp, "ballast": float(b), "season": None,
                    "n_parks": int(len(le)),
                    "rmse": float(np.sqrt(np.mean(np.concatenate(errs)))),
                    "corr": float(np.corrcoef(le, lt)[0, 1])
                    if np.std(le) > 0 else float("nan"),
                })
    return pd.DataFrame(rows)


def choose_ballast(persistence: pd.DataFrame) -> dict:
    """{component: ballast} minimising pooled leave-one-season-out rmse.

    Ties (and a grid whose rmse is flat) break toward the *larger* ballast:
    more regression is the conservative choice for an offset that is applied
    multiplicatively to somebody's projection.
    """
    pooled = persistence[persistence["season"].isna()]
    out = {}
    for comp, g in pooled.groupby("component"):
        best = g["rmse"].min()
        cand = g[np.isclose(g["rmse"], best, rtol=1e-6)]
        out[comp] = float(cand["ballast"].max())
    return out


# ─── vacuity and persistence checks (docs/park-factors.md) ─────────────────

def log_factor_spread(long: pd.DataFrame) -> pd.DataFrame:
    """sd of the regressed log factor across the parks of each season.

    docs/park-factors.md's vacuity check: below 0.05 for HR the factors are
    not saying anything and the arm below is untestable rather than wrong.
    """
    g = long.copy()
    g["log_factor"] = np.log(g["factor"].astype("float64"))
    return (g.groupby(["component", "game_year"])["log_factor"]
            .agg(sd="std", n_parks="size").reset_index())


def year_over_year_correlation(long: pd.DataFrame) -> pd.DataFrame:
    """Correlation of a park's factor with its own factor the next season.

    One row per (component, season pair) plus a pooled row per component
    (`season = None`), which is what prediction 1 is scored on.

    Consecutive stamped factors share two of their three window seasons, so
    this is a persistence check on the *artifact* (does the number a model
    would consume move around?), not on the underlying park. `raw_col="raw"`
    scores the same thing on the unregressed window and is the harsher read.
    """
    rows = []
    for comp, g in long.groupby("component"):
        years = sorted(g["game_year"].unique())
        pooled_a, pooled_b = [], []
        for y0, y1 in zip(years, years[1:]):
            if y1 != y0 + 1:
                continue
            a = g[g["game_year"] == y0].set_index("team")["factor"]
            b = g[g["game_year"] == y1].set_index("team")["factor"]
            common = a.index.intersection(b.index)
            if len(common) < 3:
                continue
            x = a.loc[common].to_numpy(dtype="float64")
            y = b.loc[common].to_numpy(dtype="float64")
            rows.append({"component": comp, "season": int(y1),
                         "n_parks": int(len(common)),
                         "corr": float(np.corrcoef(x, y)[0, 1])})
            pooled_a.append(x)
            pooled_b.append(y)
        if pooled_a:
            x = np.concatenate(pooled_a)
            y = np.concatenate(pooled_b)
            rows.append({"component": comp, "season": None,
                         "n_parks": int(len(x)),
                         "corr": float(np.corrcoef(x, y)[0, 1])})
    return pd.DataFrame(rows)


def single_season_persistence(counts: pd.DataFrame, seasons=None) -> pd.DataFrame:
    """Year-over-year correlation of the *single-season* raw factors.

    The stamped factors overlap by construction; this is the same question
    asked of windows that share nothing, and it is the honest number for "how
    much of a park is a stable fact about the park".
    """
    years = sorted(int(y) for y in (seasons if seasons is not None
                                    else counts["game_year"].unique()))
    per = {y: regress(window_raw(counts, [y]), 0.0) for y in years}
    rows = []
    for comp in COMPONENT_NAMES:
        xs, ys = [], []
        for y0, y1 in zip(years, years[1:]):
            if y1 != y0 + 1:
                continue
            a = per[y0]
            a = a[a["component"] == comp].set_index("park")["factor"]
            b = per[y1]
            b = b[b["component"] == comp].set_index("park")["factor"]
            common = a.index.intersection(b.index)
            if len(common) < 3:
                continue
            xs.append(a.loc[common].to_numpy(dtype="float64"))
            ys.append(b.loc[common].to_numpy(dtype="float64"))
        if xs:
            x, y = np.concatenate(xs), np.concatenate(ys)
            rows.append({"component": comp, "n_pairs": int(len(x)),
                         "corr": float(np.corrcoef(x, y)[0, 1])})
    return pd.DataFrame(rows)


__all__ = [
    "BALLAST_GRID", "BALLAST_SEASONS", "COMPONENT_NAMES", "COUNT_COLUMNS",
    "DEFAULT_PATH", "DEFAULT_PA_DIR", "FACTOR_COLUMNS", "FROZEN_BALLAST",
    "KNOWN_PARK_CHANGES", "PARK_COMPONENTS", "PA_COLUMNS", "WINDOW_SEASONS",
    "ParkComponent", "build_table", "choose_ballast", "factors_for_season",
    "load_pa_counts", "log_factor_spread", "loso_persistence", "loso_windows",
    "pa_counts", "regress", "single_season_persistence", "wide_table",
    "window_for", "window_raw", "year_over_year_correlation",
]
