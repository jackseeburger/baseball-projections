"""Intra-season walk-forward: cut the season at a date, project the rest.

The season-level harness (`src/eval/backtest.py`) answers "given everything
through 2025, how good is 2026?". The product is a *rest-of-season*
projection, so the question that matters is "given everything through
2026-07-01, how good is the rest of 2026?" — and answering it needs dates,
which the season table does not have. This module supplies them from the
PA-level outcome parquets and reuses the harness's scoring path unchanged.

The split at a cutoff date produces two season-shaped frames:

    partial   PA with game_date <  cutoff, aggregated to the season schema
              and marked `partial=True` — appended to the prior full seasons
              to make the training frame.
    realized  PA with game_date >= cutoff, aggregated the same way — the
              rest-of-season outcome every provider is scored against.

`assert_split_clean` is the leakage guard: no realized PA before the cutoff,
no training PA on or after it.
"""
from __future__ import annotations

import pandas as pd

from src.eval.backtest import COMPONENTS, ComponentSpec, Provider, _run_split
from src.eval.baselines import INTRASEASON_BASELINES

# Events that are plate appearances but not at-bats. AB = PA − BB − HBP −
# SF − SH − reached-on-interference (the same identity the Stats API uses,
# so PA-derived seasons line up with `hitter_seasons_api.parquet`).
SAC_FLY_EVENTS = {"sac_fly", "sac_fly_double_play"}
SAC_BUNT_EVENTS = {"sac_bunt", "sac_bunt_double_play"}
INTERFERENCE_EVENTS = {"catcher_interf"}

# Season-schema columns produced by aggregate_pa (the harness reads a subset).
COUNT_COLUMNS = [
    "pa", "ab", "h", "doubles", "triples", "hr", "k", "bb", "hbp", "sf",
    "sh", "ci", "xb_points", "bip", "hits_in_play", "games",
]


def aggregate_pa(pa: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """Roll PA-level outcomes up to the season-table schema, one row per batter.

    Produces the columns `build_seasons_table` produces (pa, ab, h, doubles,
    triples, hr, k, bb, hbp, sf, xb_points, bip, hits_in_play) plus `games`
    and the `first_game_date` / `last_game_date` bounds the leakage guard
    checks. `season` defaults to the frame's `game_year`.

    Counts are forced to int64: the outcome flags are int8 on disk and
    `2*triples + 3*hr` silently overflows if they are left that way.
    """
    if pa.empty:
        # Typed, not just named: an object-dtype empty frame poisons the
        # concat in build_training_frame and turns `batter` into objects.
        empty = {c: pd.Series(dtype="int64")
                 for c in ["batter", "season", *COUNT_COLUMNS]}
        empty["first_game_date"] = pd.Series(dtype="datetime64[ns]")
        empty["last_game_date"] = pd.Series(dtype="datetime64[ns]")
        return pd.DataFrame(empty)

    df = pa.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    event = df["event"]
    flags = {
        "_k": df["is_k"], "_bb": df["is_bb"], "_hbp": df["is_hbp"],
        "_h": df["is_hit"], "_hr": df["is_hr"],
        "_2b": df["is_double"], "_3b": df["is_triple"],
        "_sf": event.isin(SAC_FLY_EVENTS),
        "_sh": event.isin(SAC_BUNT_EVENTS),
        "_ci": event.isin(INTERFERENCE_EVENTS),
    }
    for name, col in flags.items():
        df[name] = col.astype("int64")

    g = df.groupby("batter", as_index=False).agg(
        pa=("_k", "size"),
        k=("_k", "sum"),
        bb=("_bb", "sum"),
        hbp=("_hbp", "sum"),
        h=("_h", "sum"),
        hr=("_hr", "sum"),
        doubles=("_2b", "sum"),
        triples=("_3b", "sum"),
        sf=("_sf", "sum"),
        sh=("_sh", "sum"),
        ci=("_ci", "sum"),
        games=("game_pk", "nunique"),
        first_game_date=("game_date", "min"),
        last_game_date=("game_date", "max"),
    )
    for c in ["pa", "k", "bb", "hbp", "h", "hr", "doubles", "triples",
              "sf", "sh", "ci", "games"]:
        g[c] = g[c].astype("int64")

    g["ab"] = g["pa"] - g["bb"] - g["hbp"] - g["sf"] - g["sh"] - g["ci"]
    g["xb_points"] = g["doubles"] + 2 * g["triples"] + 3 * g["hr"]
    g["bip"] = g["ab"] - g["k"] - g["hr"] + g["sf"]
    g["hits_in_play"] = g["h"] - g["hr"]

    if season is None:
        season = int(df["game_year"].iloc[0]) if "game_year" in df else int(
            df["game_date"].dt.year.iloc[0])
    g.insert(1, "season", season)
    return g[["batter", "season", *COUNT_COLUMNS,
              "first_game_date", "last_game_date"]]


def split_at_cutoff(
    pa: pd.DataFrame, cutoff_date: str | pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(PA strictly before the cutoff, PA on or after it)."""
    cutoff = pd.Timestamp(cutoff_date)
    dates = pd.to_datetime(pa["game_date"])
    return pa[dates < cutoff].copy(), pa[dates >= cutoff].copy()


def partial_and_realized(
    pa: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    season: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Season-shaped (partial, realized) frames either side of the cutoff.

    `partial` carries `partial=True`; `realized` carries `partial=False`.
    """
    before, after = split_at_cutoff(pa, cutoff_date)
    partial = aggregate_pa(before, season).assign(partial=True)
    realized = aggregate_pa(after, season).assign(partial=False)
    return partial, realized


def assert_split_clean(
    train: pd.DataFrame,
    realized: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    predict_year: int,
) -> None:
    """Leakage guard for a dated split.

    Raises ValueError if any training row saw a game on or after the cutoff,
    if any realized row contains a game before it, if training carries a
    season after the predict year, or if realized is not the predict year.
    Rows without date bounds (prior full seasons out of the season table)
    are checked on `season` alone.
    """
    cutoff = pd.Timestamp(cutoff_date)

    if not train.empty and int(train["season"].max()) > predict_year:
        raise ValueError(
            f"leakage: training frame contains season "
            f"{int(train['season'].max())} > predict year {predict_year}"
        )
    if "last_game_date" in train.columns:
        late = train["last_game_date"].dropna()
        bad = late[late >= cutoff]
        if len(bad):
            raise ValueError(
                f"leakage: {len(bad)} training row(s) contain PA on or after "
                f"the cutoff {cutoff.date()} (latest {bad.max().date()})"
            )
    if realized.empty:
        raise ValueError("leakage guard: realized frame is empty")
    if set(realized["season"].unique()) != {predict_year}:
        raise ValueError(
            f"realized frame must be entirely season {predict_year}, got "
            f"{sorted(realized['season'].unique())}"
        )
    if "first_game_date" in realized.columns:
        early = realized["first_game_date"].dropna()
        bad = early[early < cutoff]
        if len(bad):
            raise ValueError(
                f"leakage: {len(bad)} realized row(s) contain PA before the "
                f"cutoff {cutoff.date()} (earliest {bad.min().date()})"
            )


def build_training_frame(
    seasons: pd.DataFrame,
    partial: pd.DataFrame,
    predict_year: int,
) -> pd.DataFrame:
    """Prior full seasons + the partial current season, in one frame.

    Prior seasons come from the season table (season < predict_year) and are
    marked `partial=False`; any season-table rows for the predict year are
    dropped — the partial aggregate replaces them, and keeping both would
    hand providers the full current season.
    """
    prior = seasons[seasons["season"] < predict_year].copy()
    prior["partial"] = False
    if "age" in prior.columns and "age" not in partial.columns:
        # Marcel's age adjustment reads `age` off the most recent training
        # slice, which is now the partial season. Take the predict year's age
        # from the season table when it has one; otherwise age the player
        # forward from his last prior season. Missing ages stay NaN — the
        # adjustment already falls back to 1.0 for those.
        current = seasons[seasons["season"] == predict_year]
        partial = partial.copy()
        if "age" in seasons.columns and not current.empty:
            partial["age"] = partial["batter"].map(
                current.set_index("batter")["age"])
        else:
            last_prior = (prior.sort_values("season").groupby("batter")
                          .agg(age=("age", "last"), season=("season", "last")))
            partial["age"] = partial["batter"].map(
                last_prior["age"] + (predict_year - last_prior["season"])
            )
    return pd.concat([prior, partial], ignore_index=True)


def backtest_intraseason(
    component: str,
    cutoff_date: str | pd.Timestamp,
    predict_year: int | None = None,
    *,
    seasons: pd.DataFrame,
    pa_frame: pd.DataFrame | None = None,
    partial: pd.DataFrame | None = None,
    realized: pd.DataFrame | None = None,
    providers: dict[str, Provider] | None = None,
    min_trials: int = 100,
    common_players: bool = True,
) -> pd.DataFrame:
    """One dated train/predict split: everything before `cutoff_date`, score
    the rest of `predict_year`.

    Args:
        component: key in COMPONENTS.
        cutoff_date: ISO date. Training sees PA strictly before it.
        predict_year: season being cut (default: the cutoff's year).
        seasons: season-level frame for the prior seasons (2019-2025 etc.).
        pa_frame: PA-level outcomes for the predict year. Optional if
            `partial` and `realized` are supplied pre-aggregated.
        providers: model name → provider. Defaults to the baselines plus
            `season_to_date`.
        min_trials: minimum realized (rest-of-season) trials to be scored.
        common_players: score every model on the same batters.

    Returns:
        The same long frame `backtest()` returns — feed it to score() /
        calibration().
    """
    spec: ComponentSpec = COMPONENTS[component]
    cutoff = pd.Timestamp(cutoff_date)
    predict_year = predict_year or cutoff.year
    providers = providers or dict(INTRASEASON_BASELINES)

    if partial is None or realized is None:
        if pa_frame is None:
            raise ValueError("pass pa_frame, or both partial and realized")
        pa_year = pa_frame
        if "game_year" in pa_year.columns:
            pa_year = pa_year[pa_year["game_year"] == predict_year]
        if pa_year.empty:
            raise ValueError(f"no PA rows for {predict_year} in pa_frame")
        partial, realized = partial_and_realized(pa_year, cutoff, predict_year)

    train = build_training_frame(seasons, partial, predict_year)
    assert_split_clean(train, realized, cutoff, predict_year)
    if train.empty:
        raise ValueError(f"no training data before {cutoff.date()}")

    return _run_split(
        component, spec, train, realized, predict_year,
        providers=providers, min_trials=min_trials,
        common_players=common_players,
    )
