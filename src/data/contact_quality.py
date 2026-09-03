"""Batted-ball contact quality from the Statcast archive, in monthly buckets.

Statcast has sat in R2 since 2015 and has only ever been used for descriptive
aggregates (methods.md §2 calls it "our largest untouched asset"). This module
turns the pitch-level archive into the one artifact the projection harness can
actually consume: **sufficient statistics for exit-velocity / launch-angle
contact quality, per player, per calendar month, for both sides of the ball.**

Why monthly, and not per batted ball:

* The harness cuts a season at a *date* and may only see data strictly before
  it. Monthly buckets are additive, so any cutoff that falls on the first of a
  month is reconstructed exactly by summing the buckets before it — no
  filtering of a 13-million-row table at score time, and no way for a later
  month to leak into an earlier feature.
* Every cutoff the intra-season harness uses (May 1, Jul 1, Aug 1) is a month
  boundary. `assert_month_boundary` refuses anything else rather than silently
  rounding, because rounding a cutoff *forward* is leakage.
* The result is ~250k rows for 2015-2026 — small enough to commit, so the
  1.4 GB download that produced it never has to happen twice.

What a bucket carries (all additive counts, so any window is a sum):

    bbe            batted-ball events with a tracked EV and LA
    sum_ev/sum_ev2 exit velocity mean and variance
    sum_la         launch angle mean
    n_barrel       Statcast's own barrel flag (`launch_speed_angle == 6`)
    n_hardhit      EV >= 95
    n_sweetspot    launch angle in [8, 32]
    n_gb/n_fb/n_ld/n_pu   batted-ball type mix
    evbin_*        an EV histogram in 2.5 mph bins, for quantiles (EV90)

**Barrels come from the archive, not from a re-derivation.** Statcast's
`launch_speed_angle` classifies every tracked batted ball 1-6 (weak, topped,
under, flare/burner, solid, barrel); 6 is the barrel. Hand-rolled barrel
formulas circulating publicly disagree with each other at the edges, and the
column is right there.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns read out of the pitch-level parquet. Reading these instead of all
# 119 keeps a season under 100 MB in memory.
STATCAST_COLUMNS = [
    "game_type", "type", "game_date", "game_year", "batter", "pitcher",
    "events", "bb_type", "launch_speed", "launch_angle", "launch_speed_angle",
]

# The archive was exported with missing values filled as 0 / "0" rather than
# nulls, so "no tracked exit velocity" and "a 0 mph batted ball" look the
# same on disk. Nothing is hit at under 5 mph with a tracked EV, so this is
# the missingness test, and it has to be applied before any mean.
EV_MISSING_BELOW = 5.0

# Bunts are not batted-ball events in any public contact-quality metric.
BUNT_EVENTS = {"sac_bunt", "sac_bunt_double_play"}

HARD_HIT_EV = 95.0
SWEET_SPOT_LA = (8.0, 32.0)
BARREL_CODE = 6.0

# EV histogram: 2.5 mph bins over the range where the interesting quantiles
# live, plus an under- and an over-flow bin. 22 bins is enough to interpolate
# a 90th-percentile EV to well under a mile an hour.
EV_BIN_LO, EV_BIN_HI, EV_BIN_W = 60.0, 115.0, 2.5
EV_BIN_EDGES = np.arange(EV_BIN_LO, EV_BIN_HI + EV_BIN_W / 2, EV_BIN_W)
N_EV_BINS = len(EV_BIN_EDGES) + 1  # under + interior + over
EV_BIN_COLUMNS = [f"evbin_{i:02d}" for i in range(N_EV_BINS)]

# Every additive column in a bucket. `contact_features` sums exactly these.
COUNT_COLUMNS = [
    "bbe", "sum_ev", "sum_ev2", "sum_la", "n_barrel", "n_hardhit",
    "n_sweetspot", "n_gb", "n_fb", "n_ld", "n_pu", *EV_BIN_COLUMNS,
]

SIDES = {"batter": "hitter", "pitcher": "pitcher"}


def batted_balls(pitches: pd.DataFrame) -> pd.DataFrame:
    """Regular-season batted balls with tracked exit velocity and launch angle.

    One row per batted-ball event: `type == 'X'` (the ball was put in play,
    home runs included), a real EV, a real LA, and not a bunt.
    """
    df = pitches
    keep = (
        (df["game_type"] == "R")
        & (df["type"] == "X")
        & (df["launch_speed"] > EV_MISSING_BELOW)
        & df["launch_speed"].notna()
        & df["launch_angle"].notna()
        & ~df["events"].isin(BUNT_EVENTS)
    )
    # A launch angle of exactly 0.0 is a legitimate value, so LA missingness
    # cannot be read off the angle. It rides on the EV test: the archive fills
    # both or neither.
    out = df[keep].copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    return out


def ev_bin_index(ev: np.ndarray) -> np.ndarray:
    """Histogram bin per batted ball: 0 = under 60 mph, N-1 = 115 mph and up."""
    return np.digitize(np.asarray(ev, dtype="float64"), EV_BIN_EDGES)


def monthly_buckets(bb: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Sufficient statistics per (player, season, month) for one side of the ball.

    `id_col` is "batter" or "pitcher" — the same batted ball aggregated one way
    is contact a hitter made and the other way contact a pitcher allowed.
    """
    if bb.empty:
        cols = {id_col: pd.Series(dtype="int64"),
                "season": pd.Series(dtype="int64"),
                "month": pd.Series(dtype="int64")}
        cols.update({c: pd.Series(dtype="float64") for c in COUNT_COLUMNS})
        return pd.DataFrame(cols)

    df = pd.DataFrame({
        id_col: bb[id_col].to_numpy(),
        "season": bb["game_year"].to_numpy(),
        "month": bb["game_date"].dt.month.to_numpy(),
    })
    ev = bb["launch_speed"].to_numpy(dtype="float64")
    la = bb["launch_angle"].to_numpy(dtype="float64")
    lsa = bb["launch_speed_angle"].to_numpy(dtype="float64")
    bbt = bb["bb_type"].to_numpy()

    df["bbe"] = 1.0
    df["sum_ev"] = ev
    df["sum_ev2"] = ev ** 2
    df["sum_la"] = la
    df["n_barrel"] = (lsa == BARREL_CODE).astype("float64")
    df["n_hardhit"] = (ev >= HARD_HIT_EV).astype("float64")
    df["n_sweetspot"] = ((la >= SWEET_SPOT_LA[0]) & (la <= SWEET_SPOT_LA[1])
                         ).astype("float64")
    df["n_gb"] = (bbt == "ground_ball").astype("float64")
    df["n_fb"] = (bbt == "fly_ball").astype("float64")
    df["n_ld"] = (bbt == "line_drive").astype("float64")
    df["n_pu"] = (bbt == "popup").astype("float64")

    idx = ev_bin_index(ev)
    for i, col in enumerate(EV_BIN_COLUMNS):
        df[col] = (idx == i).astype("float64")

    g = df.groupby([id_col, "season", "month"], as_index=False)[COUNT_COLUMNS].sum()
    g[id_col] = g[id_col].astype("int64")
    g["season"] = g["season"].astype("int64")
    g["month"] = g["month"].astype("int64")
    return g


def build_year(year: int, raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Monthly buckets for one season, both sides of the ball stacked.

    The returned frame carries a `side` column ("hitter"/"pitcher") and a
    single `player` id, so the two halves live in one artifact.
    """
    path = Path(raw_dir) / f"statcast_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} — pull it from R2 first")
    pitches = pd.read_parquet(path, columns=STATCAST_COLUMNS)
    bb = batted_balls(pitches)
    logger.info("%d: %d pitches -> %d batted balls", year, len(pitches), len(bb))
    frames = []
    for id_col, side in SIDES.items():
        g = monthly_buckets(bb, id_col).rename(columns={id_col: "player"})
        g.insert(0, "side", side)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def build_monthly(years, raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Monthly buckets for several seasons."""
    return pd.concat([build_year(y, raw_dir) for y in years], ignore_index=True)


DEFAULT_PATH = Path("data/features/contact_quality_monthly.parquet")


def save_monthly(df: pd.DataFrame, path: str | Path = DEFAULT_PATH) -> Path:
    """Write the artifact, counts downcast to the smallest honest dtype."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in COUNT_COLUMNS:
        if c.startswith("sum_"):
            out[c] = out[c].astype("float32")
        else:
            out[c] = out[c].astype("int32")
    out.to_parquet(path, index=False)
    return path


def load_monthly(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Read the artifact back with float counts (the feature code sums them)."""
    df = pd.read_parquet(path)
    for c in COUNT_COLUMNS:
        df[c] = df[c].astype("float64")
    return df


__all__ = [
    "COUNT_COLUMNS", "DEFAULT_PATH", "EV_BIN_COLUMNS", "EV_BIN_EDGES",
    "batted_balls", "build_monthly", "build_year", "ev_bin_index",
    "load_monthly", "monthly_buckets", "save_monthly",
]
