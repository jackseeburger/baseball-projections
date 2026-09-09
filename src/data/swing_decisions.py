"""Swing decisions from the Statcast archive: pitches in, monthly buckets out.

Contact quality (`src/data/contact_quality.py`) measures what happens *after*
the bat meets the ball. K% and BB% have no measurement of their own. What
predicts them at the pitch grain is the decision **whether to swing** and
whether the swing **found the ball**: chase rate outside the zone, swing rate
inside it, zone contact, whiff per swing, called strikes taken. Those are the
counts this module reduces the archive to, per batter, per calendar month.

Why monthly, and why counts rather than rates — the same two reasons as
contact quality. The harness cuts a season at a *date* and may only see data
strictly before it; monthly buckets are additive, so a cutoff on the first of
a month is reconstructed exactly by summing the buckets before it, and a
cutoff that is not a month boundary is refused rather than rounded (rounding
forward is leakage). **Rates are formed at read time from the summed counts
and are never stored**: a stored rate is not additive, and averaging monthly
rates is not the rate over the window.

What a bucket carries (every column a count, so any window is a sum):

    pitches_z / pitches_o    competitive pitches in / out of the zone
    swings_z  / swings_o     swings at each
    contact_z / contact_o    swings that touched the ball
    whiff_z   / whiff_o      swings that missed
    called_z                 called strikes on pitches taken in the zone

**The zone.** Statcast's own `zone` field is authoritative where it exists:
1-9 are the nine cells of the strike zone, 11-14 the four out-of-zone
quadrants. When it is null (or the archive's zero-fill, which is not a legal
zone value) the pitch falls back to `plate_x` / `plate_z` against **that
batter's own** `sz_top` / `sz_bot` for that pitch, with a half-ball-width
margin on every edge: a pitch is a strike when its *centre* is within half a
ball of the zone, since the rule is about the ball's surface and the tracking
reports its centre. Horizontally that is half the 17-inch plate plus half a
2.9-inch ball; vertically it is the batter's own top and bottom, each moved
out by the same half ball. A pitch with neither a usable `zone` nor a usable
`plate_x`/`plate_z`/`sz_top`/`sz_bot` is dropped rather than guessed.

**The swing mapping is BAS-71's, reused rather than re-derived.**
`src.data.pitching_stuff` pinned it: `hit_into_play`, `foul`, `foul_tip`,
`swinging_strike`, `swinging_strike_blocked` are swings; the two
`swinging_strike*` descriptions are whiffs; a `foul_tip` is a swing that
*touched* the ball and so counts as contact, which is how Statcast's own
whiff rate counts it. Bunts, pitchouts and intentional balls are dropped from
both numerator and denominator exactly as that module drops them — a missed
bunt is not a swing decision, and nobody is deciding whether to swing at a
pitchout.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.pitching_stuff import (
    CALLED_STRIKE_DESCRIPTIONS,
    EXCLUDED_DESCRIPTIONS,
    EXCLUDED_PITCH_TYPES,
    SWING_DESCRIPTIONS,
    WHIFF_DESCRIPTIONS,
)

logger = logging.getLogger(__name__)

# Read these instead of all 119 columns: a season is 700k-800k pitches.
STATCAST_COLUMNS = [
    "game_type", "game_date", "game_year", "batter", "description",
    "pitch_type", "zone", "plate_x", "plate_z", "sz_top", "sz_bot",
]

# Statcast's `zone`: 1-9 tile the strike zone, 11-14 the four quadrants
# outside it. Anything else (including the archive's 0 fill) is "no zone".
ZONE_IN = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9})
ZONE_OUT = frozenset({11, 12, 13, 14})

# The geometric fallback, in feet. Home plate is 17 inches wide and a
# baseball is 2.9 inches across; a pitch is a strike when any part of the ball
# crosses the zone, and the tracking reports the ball's *centre*, so the
# centre may sit half a ball outside each edge. HALF_BALL is that margin and
# it is applied to the vertical edges too, against the batter's own zone.
HALF_PLATE_FT = (17.0 / 2.0) / 12.0
HALF_BALL_FT = (2.9 / 2.0) / 12.0
ZONE_HALF_WIDTH_FT = HALF_PLATE_FT + HALF_BALL_FT

# The archive fills missing values with 0, and a 0.0 ft zone top is not a
# measurement. Nothing below this is a real strike-zone boundary.
MIN_SZ_FT = 0.1

# Every additive column in a bucket. `window_counts` sums exactly these.
COUNT_COLUMNS = [
    "pitches_z", "pitches_o", "swings_z", "swings_o",
    "contact_z", "contact_o", "whiff_z", "whiff_o", "called_z",
]

# Contact is a swing that was not a whiff — hit_into_play, foul, foul_tip.
CONTACT_DESCRIPTIONS = frozenset(SWING_DESCRIPTIONS) - frozenset(WHIFF_DESCRIPTIONS)


def competitive_pitches(pitches: pd.DataFrame) -> pd.DataFrame:
    """Regular-season pitches a batter could have decided to swing at.

    Bunts, pitchouts, intentional and automatic balls/strikes and
    position-player pitch tags are dropped, exactly as
    `src.data.pitching_stuff.competitive_pitches` drops them. Unlike that
    function this one does *not* require pitch tracking: a swing decision is
    made about a pitch whether or not its spin rate was measured. What it does
    require is a location, since a swing decision has to be about a location —
    see `zone_flags`.
    """
    df = pitches
    keep = (
        (df["game_type"] == "R")
        & ~df["description"].isin(EXCLUDED_DESCRIPTIONS)
        & ~df["pitch_type"].fillna("").isin(EXCLUDED_PITCH_TYPES)
    )
    out = df[keep].copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    return out


def zone_flags(pitches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`(in_zone, has_location)` per pitch.

    `zone` decides where it is present and legal. Where it is not, the pitch
    falls back to its own coordinates against the batter's own zone with the
    half-ball margin of `ZONE_HALF_WIDTH_FT` / `HALF_BALL_FT`. A pitch with
    neither is not counted at all — dropping it costs a fraction of a percent
    of pitches and guessing would put balls in the zone denominator.
    """
    z = pd.to_numeric(pitches["zone"], errors="coerce").to_numpy(dtype="float64")
    zi = np.isin(z, sorted(ZONE_IN))
    zo = np.isin(z, sorted(ZONE_OUT))
    known = zi | zo

    px = pd.to_numeric(pitches["plate_x"], errors="coerce").to_numpy(dtype="float64")
    pz = pd.to_numeric(pitches["plate_z"], errors="coerce").to_numpy(dtype="float64")
    top = pd.to_numeric(pitches["sz_top"], errors="coerce").to_numpy(dtype="float64")
    bot = pd.to_numeric(pitches["sz_bot"], errors="coerce").to_numpy(dtype="float64")
    geo_ok = (np.isfinite(px) & np.isfinite(pz) & np.isfinite(top)
              & np.isfinite(bot) & (top > MIN_SZ_FT) & (bot > MIN_SZ_FT)
              & (top > bot))
    with np.errstate(invalid="ignore"):
        geo_in = (np.abs(px) <= ZONE_HALF_WIDTH_FT) & (pz <= top + HALF_BALL_FT) \
            & (pz >= bot - HALF_BALL_FT)
    geo_in = geo_in & geo_ok

    in_zone = np.where(known, zi, geo_in)
    has_location = known | geo_ok
    return in_zone, has_location


def pitch_flags(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per located competitive pitch, carrying the nine bucket counts.

    The columns are already the additive quantities; `monthly_buckets` only
    groups and sums them.
    """
    in_zone, located = zone_flags(pitches)
    df = pitches[located]
    zi = in_zone[located].astype("float64")
    zo = 1.0 - zi

    desc = df["description"].to_numpy()
    swing = np.isin(desc, sorted(SWING_DESCRIPTIONS)).astype("float64")
    whiff = np.isin(desc, sorted(WHIFF_DESCRIPTIONS)).astype("float64")
    contact = np.isin(desc, sorted(CONTACT_DESCRIPTIONS)).astype("float64")
    called = np.isin(desc, sorted(CALLED_STRIKE_DESCRIPTIONS)).astype("float64")

    return pd.DataFrame({
        "batter": df["batter"].to_numpy(),
        "season": df["game_year"].to_numpy(),
        "month": pd.to_datetime(df["game_date"]).dt.month.to_numpy(),
        "pitches_z": zi, "pitches_o": zo,
        "swings_z": swing * zi, "swings_o": swing * zo,
        "contact_z": contact * zi, "contact_o": contact * zo,
        "whiff_z": whiff * zi, "whiff_o": whiff * zo,
        "called_z": called * zi,
    })


def monthly_buckets(flags: pd.DataFrame) -> pd.DataFrame:
    """Additive sufficient statistics per (batter, season, month)."""
    if flags.empty:
        cols = {"batter": pd.Series(dtype="int64"),
                "season": pd.Series(dtype="int64"),
                "month": pd.Series(dtype="int64")}
        cols.update({c: pd.Series(dtype="float64") for c in COUNT_COLUMNS})
        return pd.DataFrame(cols)
    out = flags.groupby(["batter", "season", "month"], as_index=False)[
        COUNT_COLUMNS].sum()
    for c in ("batter", "season", "month"):
        out[c] = out[c].astype("int64")
    return out


def build_year(year: int, raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Monthly swing-decision buckets for one season, read off the archive."""
    path = Path(raw_dir) / f"statcast_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} — pull it from R2 first")
    pitches = pd.read_parquet(path, columns=STATCAST_COLUMNS)
    flags = pitch_flags(competitive_pitches(pitches))
    g = monthly_buckets(flags)
    if not flags.empty:
        tot = flags[COUNT_COLUMNS].sum()
        logger.info(
            "%d: %d pitches -> %d located competitive (%.3f in zone, "
            "%.3f chase, %.3f zone contact)", year, len(pitches), len(flags),
            tot["pitches_z"] / max(len(flags), 1),
            tot["swings_o"] / max(tot["pitches_o"], 1.0),
            tot["contact_z"] / max(tot["swings_z"], 1.0))
    return g


def build_monthly(years, raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    return pd.concat([build_year(y, raw_dir) for y in years], ignore_index=True)


DEFAULT_PATH = Path("data/features/swing_decisions_monthly.parquet")


def save_monthly(df: pd.DataFrame, path: str | Path = DEFAULT_PATH) -> Path:
    """Write the artifact. Every column is a count, so every column is int32."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in COUNT_COLUMNS:
        out[c] = out[c].astype("int32")
    out.to_parquet(path, index=False, compression="zstd")
    return path


def meta_path(path: str | Path = DEFAULT_PATH) -> Path:
    """The JSON sidecar next to the parquet: `<name>.meta.json`.

    Same shape and same reasoning as the contact-quality and pitching-stuff
    sidecars: `scripts/check_freshness.py` is stdlib-only on purpose and
    cannot parse a parquet to find out when the artifact was built.
    """
    path = Path(path)
    return path.with_suffix("").with_suffix(".meta.json")


def write_meta(path: str | Path = DEFAULT_PATH, *, built_at: str | None = None,
               seasons_built: list[int] | None = None) -> Path:
    """Stamp the sidecar with when this build ran and which seasons it touched."""
    import json

    stamp = built_at or datetime.now(timezone.utc).isoformat()
    out = meta_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "built_at": stamp,
        "seasons_built": sorted(seasons_built) if seasons_built else None,
    }, indent=1) + "\n")
    return out


def load_monthly(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Read the artifact back with float counts (the feature code sums them)."""
    df = pd.read_parquet(path)
    for c in COUNT_COLUMNS:
        df[c] = df[c].astype("float64")
    return df


__all__ = [
    "COUNT_COLUMNS", "CONTACT_DESCRIPTIONS", "DEFAULT_PATH", "HALF_BALL_FT",
    "STATCAST_COLUMNS", "ZONE_HALF_WIDTH_FT", "ZONE_IN", "ZONE_OUT",
    "build_monthly", "build_year", "competitive_pitches", "load_monthly",
    "meta_path", "monthly_buckets", "pitch_flags", "save_monthly",
    "write_meta", "zone_flags",
]
