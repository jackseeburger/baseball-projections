"""Pitch characteristics ("stuff") from the Statcast archive: pitches in, monthly buckets out.

This module owns the two ends of BAS-71's stage 1. At the front it turns a
season's pitch-level parquet into the frame a classifier can read — the pitch's
own physics and nothing about where it was thrown. At the back it reduces the
model's per-pitch predictions to additive sufficient statistics per pitcher per
calendar month, which is the only shape the projection harness can consume
without leaking (see `src/data/contact_quality.py` for why monthly).

**What "stuff" means here, precisely.** Velocity, movement, release point and
extension, spin rate and axis, arm angle, the velocity/acceleration vector,
handedness, and each pitch's velocity and movement *relative to the same
pitcher's own fastball that season*. Location — `plate_x`, `plate_z`, `zone` —
is excluded by construction: a pitch that misses bats because it was at the
bottom of the zone is command, not stuff. `LOCATION_FEATURES` exists only so
`scripts/build_pitching_stuff.py` can fit the labelled "pitching" comparison,
and that arm is never merged into the stuff score.

**The two targets.**

    whiff   swinging strike, given the batter swung
    csw     called strike or whiff, given the pitch was thrown

Both come off `description`, and the mapping is worth writing down because it
is the one place a stuff model can quietly be measuring something else:

    swings      hit_into_play, foul, foul_tip, swinging_strike,
                swinging_strike_blocked
    whiffs      swinging_strike, swinging_strike_blocked
    called      called_strike

`foul_tip` is a *swing that touched the ball* — the bat did not miss — so it is
a swing and not a whiff, which is how Statcast's own whiff rate counts it.
Bunts (`foul_bunt`, `missed_bunt`, `bunt_foul_tip`) are swings in the literal
sense and are dropped from both numerator and denominator: a missed bunt says
nothing about a pitch's ability to miss a swing. Pitchouts and intentional
balls are dropped as non-competitive pitches, as are position-player
"pitch types" (`IN`, `PO`, `AB`).

**Missing tracking fields are missing, not zero.** The archive was exported
with nulls filled as 0, so `spin_axis == 0` and `arm_angle == 0` read as real
measurements on disk. They are not: `arm_angle` is 100% zero in 2015 and
`spin_axis` likewise, because neither field existed yet. `_blank_zeros` puts
them back to NaN, which LightGBM routes natively. The consequence for the
walk-forward design is real and stated rather than hidden: the model that
scores 2017 was trained on seasons that had no spin axis at all, so it cannot
use one, and the model that scores 2026 can. That is the honest walk-forward
answer — the information available before season Y is what season Y gets —
but it means the per-season log-losses below are not measuring one fixed model.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Read these instead of all 119 columns: a season is 700k-800k pitches and the
# full table is 130 MB on disk, several times that in memory.
STATCAST_COLUMNS = [
    "game_type", "game_date", "game_year", "pitcher", "p_throws", "stand",
    "description", "pitch_type", "release_speed", "effective_speed",
    "pfx_x", "pfx_z", "release_pos_x", "release_pos_y", "release_pos_z",
    "release_extension", "release_spin_rate", "spin_axis", "arm_angle",
    "vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z",
]

SWING_DESCRIPTIONS = frozenset({
    "hit_into_play", "foul", "foul_tip", "swinging_strike",
    "swinging_strike_blocked",
})
WHIFF_DESCRIPTIONS = frozenset({"swinging_strike", "swinging_strike_blocked"})
CALLED_STRIKE_DESCRIPTIONS = frozenset({"called_strike"})
# Non-competitive pitches: nobody is trying to get a swing, and a bunt attempt
# is not a swing decision about the pitch's movement.
EXCLUDED_DESCRIPTIONS = frozenset({
    "pitchout", "intent_ball", "automatic_ball", "automatic_strike",
    "swinging_pitchout", "foul_pitchout", "pitchout_hit_into_play",
    "foul_bunt", "missed_bunt", "bunt_foul_tip",
})
# Position players and pickoff throws carry these codes; "0" and "" are the
# archive's fill for an untagged pitch.
EXCLUDED_PITCH_TYPES = frozenset({"IN", "PO", "AB", "", "0", "UN", "FA"})

# A pitch with no release speed has no tracking at all — the archive fills the
# whole physics block with zeros for it.
MIN_RELEASE_SPEED = 40.0

# Fields the archive fills with 0 when the measurement does not exist. Zero is
# a legal value for none of them (a 0 mph spin rate, a 0-degree arm angle from
# a pitcher standing on the mound), so the zeros are missingness.
ZERO_IS_MISSING = ("release_spin_rate", "spin_axis", "arm_angle",
                   "release_extension", "effective_speed")

# What counts as a fastball for the "relative to his own fastball" features:
# four-seam and sinker/two-seam, the pitches every velocity-baseline convention
# uses. FT is the pre-2020 tag for what is now SI. Cutters are deliberately
# out — a cutter is a breaking-ball-shaped fastball, and using it as the
# reference would flatten exactly the difference the feature is built to see.
FASTBALL_TYPES = frozenset({"FF", "SI", "FT"})
# A pitcher-season needs this many fastballs before its mean is used as the
# reference. Below it the reference is noisier than the difference it defines.
MIN_FASTBALLS = 25

# Pitch-type groups the monthly artifact counts. Statcast's tags are
# unstable across seasons (ST/SV split out of SL in 2023, FT folded into SI in
# 2020), so the buckets are groups rather than raw tags.
PITCH_GROUPS = {
    "FF": "ff", "SI": "si", "FT": "si", "FC": "fc",
    "SL": "sl", "ST": "sl", "SV": "sl", "CS": "cu", "CU": "cu", "KC": "cu",
    "CH": "ch", "FS": "ch", "FO": "ch", "SC": "ch", "EP": "other",
    "KN": "other",
}
GROUP_ORDER = ("ff", "si", "fc", "sl", "cu", "ch", "other")

# The stuff features, in a fixed order so a saved model's columns are readable.
STUFF_FEATURES = (
    "release_speed", "effective_speed",
    "pfx_x_arm", "pfx_z",
    "release_pos_x_arm", "release_pos_y", "release_pos_z", "release_extension",
    "release_spin_rate", "spin_axis_sin", "spin_axis_cos", "arm_angle",
    "vx0_arm", "vy0", "vz0", "ax_arm", "ay", "az",
    "is_rhp",
    "d_velo_fb", "d_pfx_x_arm_fb", "d_pfx_z_fb", "d_spin_fb",
)
# The labelled "pitching" comparison — stuff plus where the pitch ended up.
LOCATION_FEATURES = STUFF_FEATURES + ("plate_x_arm", "plate_z")
# Comparison (a): what pitch was it. Comparison (b): how hard does he throw.
PITCH_TYPE_FEATURES = ("pitch_group_code",)
FASTBALL_VELO_FEATURES = ("fb_velo",)

FEATURE_SETS = {
    "stuff": STUFF_FEATURES,
    "pitch_type": PITCH_TYPE_FEATURES,
    "fb_velo": FASTBALL_VELO_FEATURES,
    "pitching": LOCATION_FEATURES,
}


def _blank_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Put the archive's zero-fill back to NaN for fields where 0 is not real."""
    for col in ZERO_IS_MISSING:
        if col in df.columns:
            df.loc[df[col] == 0.0, col] = np.nan
    return df


def competitive_pitches(pitches: pd.DataFrame) -> pd.DataFrame:
    """Regular-season pitches with tracking, minus pitchouts, bunts and lobs.

    One row per pitch. `game_date` comes back as a timestamp and the physics
    columns come back with the archive's zero-fill turned into NaN.
    """
    df = pitches
    keep = (
        (df["game_type"] == "R")
        & ~df["description"].isin(EXCLUDED_DESCRIPTIONS)
        & ~df["pitch_type"].fillna("").isin(EXCLUDED_PITCH_TYPES)
        & (df["release_speed"] > MIN_RELEASE_SPEED)
        & df["release_speed"].notna()
    )
    out = df[keep].copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    return _blank_zeros(out)


def fastball_reference(pitches: pd.DataFrame) -> pd.DataFrame:
    """Per pitcher-season fastball mean velocity, movement and spin.

    The reference the "relative to his own fastball" features difference
    against. A pitcher-season with fewer than `MIN_FASTBALLS` four-seamers and
    sinkers gets no row, so every relative feature comes back NaN for him and
    the model treats him as a pitcher whose fastball is unknown rather than as
    a pitcher whose fastball is league average. That catches 6.0% of 2015
    pitcher-seasons and 10.6% of 2024's — the rise is position players pitching
    in blowouts, which is a real change in the data and not a change in this
    rule. Among pitcher-seasons of 200 pitches or more it is 0.3% in both, and
    those are genuine: a knuckleballer has no fastball to measure against.
    """
    fb = pitches[pitches["pitch_type"].isin(FASTBALL_TYPES)]
    if fb.empty:
        return pd.DataFrame(columns=["pitcher", "game_year", "fb_velo",
                                     "fb_pfx_x_arm", "fb_pfx_z", "fb_spin",
                                     "fb_n"])
    g = fb.groupby(["pitcher", "game_year"]).agg(
        fb_velo=("release_speed", "mean"),
        fb_pfx_x_arm=("pfx_x_arm", "mean"),
        fb_pfx_z=("pfx_z", "mean"),
        fb_spin=("release_spin_rate", "mean"),
        fb_n=("release_speed", "size"),
    ).reset_index()
    return g[g["fb_n"] >= MIN_FASTBALLS]


def pitch_features(pitches: pd.DataFrame) -> pd.DataFrame:
    """The stuff feature frame plus targets, one row per competitive pitch.

    Horizontal quantities are mirrored to the pitcher's arm side (`*_arm`):
    a right-hander's slider and a left-hander's slider are the same pitch with
    opposite `pfx_x`, and without the mirror a depth-limited tree spends its
    splits rediscovering that. Handedness is still a feature, because arm-side
    and glove-side are not symmetric for the batter.

    Spin axis is a compass bearing, so 359 and 1 are neighbours; it enters as
    its sine and cosine rather than as a number that wraps.
    """
    df = pitches
    arm = np.where(df["p_throws"].to_numpy() == "R", 1.0, -1.0)
    out = pd.DataFrame(index=df.index)
    out["pitcher"] = df["pitcher"].to_numpy()
    out["game_year"] = df["game_year"].to_numpy()
    out["game_date"] = df["game_date"].to_numpy()
    out["pitch_type"] = df["pitch_type"].to_numpy()
    out["pitch_group"] = df["pitch_type"].map(PITCH_GROUPS).fillna("other").to_numpy()
    out["pitch_group_code"] = pd.Categorical(
        out["pitch_group"], categories=list(GROUP_ORDER)).codes.astype("float32")

    for col in ("release_speed", "effective_speed", "pfx_z", "release_pos_y",
                "release_pos_z", "release_extension", "release_spin_rate",
                "arm_angle", "vy0", "vz0", "ay", "az", "plate_z"):
        out[col] = df[col].to_numpy(dtype="float64")
    for col, src in (("pfx_x_arm", "pfx_x"), ("release_pos_x_arm", "release_pos_x"),
                     ("vx0_arm", "vx0"), ("ax_arm", "ax"),
                     ("plate_x_arm", "plate_x")):
        out[col] = df[src].to_numpy(dtype="float64") * arm

    axis = np.radians(df["spin_axis"].to_numpy(dtype="float64"))
    out["spin_axis_sin"] = np.sin(axis) * arm
    out["spin_axis_cos"] = np.cos(axis)
    out["is_rhp"] = (arm > 0).astype("float32")

    ref = fastball_reference(out)
    out = out.merge(ref, on=["pitcher", "game_year"], how="left")
    out["d_velo_fb"] = out["release_speed"] - out["fb_velo"]
    out["d_pfx_x_arm_fb"] = out["pfx_x_arm"] - out["fb_pfx_x_arm"]
    out["d_pfx_z_fb"] = out["pfx_z"] - out["fb_pfx_z"]
    out["d_spin_fb"] = out["release_spin_rate"] - out["fb_spin"]

    desc = df["description"].to_numpy()
    out["is_swing"] = np.isin(desc, list(SWING_DESCRIPTIONS))
    out["is_whiff"] = np.isin(desc, list(WHIFF_DESCRIPTIONS))
    out["is_called"] = np.isin(desc, list(CALLED_STRIKE_DESCRIPTIONS))
    out["is_csw"] = out["is_whiff"] | out["is_called"]
    out["is_fastball"] = out["pitch_type"].isin(FASTBALL_TYPES)
    # float32 throughout: eleven seasons of training pitches is 7.7 million
    # rows, and float64 physics columns would be 2.5 GB of them for no gain —
    # a release speed is reported to a tenth of a mile an hour.
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    return out.reset_index(drop=True)


def build_year(year: int, raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """`pitch_features` for one season, read straight off the archive."""
    path = Path(raw_dir) / f"statcast_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} — pull it from R2 first")
    pitches = pd.read_parquet(path, columns=STATCAST_COLUMNS)
    feats = pitch_features(competitive_pitches(pitches))
    logger.info("%d: %d pitches -> %d competitive (%d swings, %.3f whiff, "
                "%.3f csw)", year, len(pitches), len(feats),
                int(feats["is_swing"].sum()),
                float(feats.loc[feats["is_swing"], "is_whiff"].mean()),
                float(feats["is_csw"].mean()))
    return feats


# --- the monthly reduction ---------------------------------------------------

# Every column in a bucket is a sum, so any window is a sum of buckets and a
# first-of-month cutoff is answered exactly by the buckets strictly before it.
COUNT_COLUMNS = [
    "pitches", "swings", "p_whiff_sum", "p_csw_sum", "sum_velo",
    "fb_pitches", "fb_swings", "fb_p_whiff_sum", "fb_p_csw_sum", "fb_sum_velo",
    "nfb_pitches", "nfb_swings", "nfb_p_whiff_sum", "nfb_p_csw_sum",
    "nfb_sum_velo",
    *[f"n_{g}" for g in GROUP_ORDER],
]


def monthly_buckets(scored: pd.DataFrame) -> pd.DataFrame:
    """Additive sufficient statistics per (pitcher, season, month).

    `scored` is `pitch_features` output carrying `p_whiff` (the model's
    predicted whiff-given-swing) and `p_csw` (predicted CSW-given-pitch).

    The predicted-whiff sum is taken over **swings only**, matching the
    quantity's own denominator: a whiff rate is per swing, and summing a
    predicted whiff probability over pitches the batter never offered at would
    make the aggregate a swing-rate measurement in disguise.
    """
    if scored.empty:
        cols = {"pitcher": pd.Series(dtype="int64"),
                "season": pd.Series(dtype="int64"),
                "month": pd.Series(dtype="int64")}
        cols.update({c: pd.Series(dtype="float64") for c in COUNT_COLUMNS})
        return pd.DataFrame(cols)

    sw = scored["is_swing"].to_numpy().astype("float64")
    fb = scored["is_fastball"].to_numpy().astype("float64")
    velo = scored["release_speed"].to_numpy(dtype="float64")
    pw = scored["p_whiff"].to_numpy(dtype="float64") * sw
    pc = scored["p_csw"].to_numpy(dtype="float64")

    df = pd.DataFrame({
        "pitcher": scored["pitcher"].to_numpy(),
        "season": scored["game_year"].to_numpy(),
        "month": pd.to_datetime(scored["game_date"]).dt.month.to_numpy(),
        "pitches": 1.0, "swings": sw, "p_whiff_sum": pw, "p_csw_sum": pc,
        "sum_velo": velo,
        "fb_pitches": fb, "fb_swings": sw * fb, "fb_p_whiff_sum": pw * fb,
        "fb_p_csw_sum": pc * fb, "fb_sum_velo": velo * fb,
        "nfb_pitches": 1.0 - fb, "nfb_swings": sw * (1 - fb),
        "nfb_p_whiff_sum": pw * (1 - fb), "nfb_p_csw_sum": pc * (1 - fb),
        "nfb_sum_velo": velo * (1 - fb),
    })
    for g in GROUP_ORDER:
        df[f"n_{g}"] = (scored["pitch_group"].to_numpy() == g).astype("float64")

    out = df.groupby(["pitcher", "season", "month"], as_index=False)[
        COUNT_COLUMNS].sum()
    for c in ("pitcher", "season", "month"):
        out[c] = out[c].astype("int64")
    return out


DEFAULT_PATH = Path("data/features/pitching_stuff_monthly.parquet")


def save_monthly(df: pd.DataFrame, path: str | Path = DEFAULT_PATH) -> Path:
    """Write the artifact. Counts are int32, sums float32.

    float32 carries seven significant digits, and the largest sum in the table
    is a month of release speeds for a workhorse starter — about 45,000 — so
    the rounding is under a thousandth of a mile an hour per pitch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in COUNT_COLUMNS:
        out[c] = out[c].astype("float32" if ("sum" in c) else "int32")
    out.to_parquet(path, index=False, compression="zstd")
    return path


def load_monthly(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Read the artifact back with float counts (the feature code sums them)."""
    df = pd.read_parquet(path)
    for c in COUNT_COLUMNS:
        df[c] = df[c].astype("float64")
    return df


__all__ = [
    "CALLED_STRIKE_DESCRIPTIONS", "COUNT_COLUMNS", "DEFAULT_PATH",
    "FASTBALL_TYPES", "FASTBALL_VELO_FEATURES", "FEATURE_SETS", "GROUP_ORDER",
    "LOCATION_FEATURES", "MIN_FASTBALLS", "PITCH_GROUPS",
    "PITCH_TYPE_FEATURES", "STATCAST_COLUMNS", "STUFF_FEATURES",
    "SWING_DESCRIPTIONS", "WHIFF_DESCRIPTIONS", "build_year",
    "competitive_pitches", "fastball_reference", "load_monthly",
    "monthly_buckets", "pitch_features", "save_monthly",
]
