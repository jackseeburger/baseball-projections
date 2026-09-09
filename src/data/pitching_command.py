"""Command and location from the Statcast archive: pitches in, monthly buckets out (BAS-76).

The location half of the pitch. `src/data/pitching_stuff.py` deliberately
excludes `plate_x`/`plate_z` — that is the definition of stuff as opposed to
pitching — and BAS-71's labelled comparison arm then showed that the excluded
half is worth roughly a third of the reachable pitch-level signal (whiff|swing
log-loss .4416 with location against .4490 without, on 2026). This module is
what reads that half.

**What "command" means here, precisely.** Not "the pitcher hit his spot" —
public data has no spot. It is *where the pitch ended up*, relative to this
batter's own strike zone, in this count, against this handedness, and what
that location is worth on top of the pitch's physics. The measurement that
comes out at the back is therefore a **residual**: for each pitch, the CSW
probability a model that sees location assigns it, minus the probability the
same-shaped model that sees only stuff assigns it. A pitcher whose residual
is positive put his pitches in better places than his stuff alone implied.
Differencing is what stops the aggregate from being the stuff score again
under a new name — the served `stuff_additive` engine already has that score,
and BAS-76's whole question is what is left over.

**The feature block.** `plate_x` mirrored to the pitcher's arm side and again
to the batter's box (inside/away is not a fixed sign), `plate_z` both in feet
and in strike-zone units of this batter's own `sz_top`/`sz_bot`, the signed
distance to the nearest edge of the rulebook zone, the count, whether the
matchup is same-handed, and the pitch group. The count is in because a 3-0
fastball down the middle and an 0-2 fastball down the middle are not the same
decision, and a called strike is much likelier on the first.

**The two targets.**

    csw            called strike or whiff, given the pitch was thrown
    called_taken   called strike, given the batter did not swing

The second is the cleaner command target of the two: it holds the batter's
swing decision fixed and asks only whether the umpire called it, which is very
nearly a pure question about location. It is also the one that a stuff model
has the least business predicting, so it is where the location arm's margin
should be widest.

**The attack regions.** Statcast's own four, computed geometrically here
rather than read off the archive's `zone` column so that the definition is in
this file and testable: `heart` (well inside the rulebook zone), `shadow`
(within a ball's width either side of an edge), `chase` (outside that but
still reachable) and `waste` (further out than anyone chases). They partition
every pitch, so their counts are additive and their shares sum to one.
`in_zone` is tracked separately and deliberately crosses heart and shadow — it
is the rulebook zone, which is the thing a walk is actually adjudicated on.

The monthly reduction is `src/data/pitching_stuff.py`'s, for the same reason:
every column of a bucket is a sum over the pitches thrown in that calendar
month, so any window is a sum of buckets and a first-of-month cutoff is
reconstructed exactly by summing the buckets strictly earlier than it.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.pitching_stuff import (
    STATCAST_COLUMNS as STUFF_STATCAST_COLUMNS,
    STUFF_FEATURES,
    competitive_pitches,
    pitch_features,
)

logger = logging.getLogger(__name__)

# The stuff module's columns plus the ones location needs: the batter's own
# strike zone, the count, and the archive's `zone` tag (kept for the tests to
# cross-check the geometry against, never used as a feature).
STATCAST_COLUMNS = [*STUFF_STATCAST_COLUMNS, "sz_top", "sz_bot", "balls",
                    "strikes", "zone"]

# Half the rulebook zone's width in feet: 8.5 inches of plate plus a ball's
# radius (2.9 inches), which is where a strike is actually called.
ZONE_HALF_WIDTH = 0.83
# A ball's width. `shadow` is one of these either side of an edge — Statcast's
# own attack-region definition.
BALL_WIDTH = 0.25
# Outside `shadow` and within this of an edge is `chase`; beyond it is `waste`.
CHASE_WIDTH = 0.5
# A batter with no tracked strike zone. Below it the archive's zero-fill or a
# corrupt row is being read as a zone a foot tall.
MIN_ZONE_HEIGHT = 0.5

REGIONS = ("heart", "shadow", "chase", "waste")

# The location block. `_arm` is mirrored to the pitcher's throwing side (as
# every horizontal quantity in the stuff features is); `_bat` is mirrored to
# the batter's box, so positive is always away from the batter.
LOCATION_ONLY_FEATURES = (
    "plate_x_arm", "plate_x_bat", "plate_z",
    "plate_z_sz", "edge_dist", "abs_plate_x",
    "balls", "strikes", "count_diff", "same_hand",
    "pitch_group_code",
)
# The pre-registered stage-1 arm: stuff *and* location. Named `pitching`
# because that is what BAS-71 called the labelled comparison it inherits from,
# and because location is command rather than stuff.
PITCHING_FEATURES = STUFF_FEATURES + LOCATION_ONLY_FEATURES
# Control (a): the served stuff model's own feature set, refitted here so the
# residual is a difference between two models fitted on the same rows.
# Control (b): location with no physics. Control (c): what pitch was it.
PITCH_TYPE_FEATURES = ("pitch_group_code",)

FEATURE_SETS = {
    "pitching": PITCHING_FEATURES,
    "stuff": STUFF_FEATURES,
    "location": LOCATION_ONLY_FEATURES,
    "pitch_type": PITCH_TYPE_FEATURES,
}

# The arm the residual is taken against, and the arm it is taken from.
RESIDUAL_ARM = "pitching"
RESIDUAL_BASE_ARM = "stuff"


def location_features(pitches: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Add the location block to `src.data.pitching_stuff.pitch_features` output.

    `pitches` is the `competitive_pitches` frame (it still carries `sz_top`,
    `sz_bot`, `balls`, `strikes`, `stand`) and `feats` is the stuff feature
    frame built from it, index-aligned by position.

    A batter with no tracked strike zone gets NaN for the zone-relative
    columns and no attack region, rather than a zone invented from the league
    mean: LightGBM routes the NaN, and a pitch whose region is unknown must
    not silently land in `heart`.
    """
    df = pitches.reset_index(drop=True)
    out = feats.reset_index(drop=True).copy()

    arm = np.where(df["p_throws"].to_numpy() == "R", 1.0, -1.0)
    bat = np.where(df["stand"].to_numpy() == "R", 1.0, -1.0)
    px = df["plate_x"].to_numpy(dtype="float64")
    pz = df["plate_z"].to_numpy(dtype="float64")
    top = df["sz_top"].to_numpy(dtype="float64")
    bot = df["sz_bot"].to_numpy(dtype="float64")

    height = top - bot
    bad = ~np.isfinite(height) | (height < MIN_ZONE_HEIGHT)
    height = np.where(bad, np.nan, height)
    top = np.where(bad, np.nan, top)
    bot = np.where(bad, np.nan, bot)

    out["plate_x_arm"] = (px * arm).astype("float32")
    # Positive is away from the batter: a right-handed batter's outside pitch
    # is at negative `plate_x` from the catcher's view.
    out["plate_x_bat"] = (-px * bat).astype("float32")
    out["plate_z"] = pz.astype("float32")
    # Height in units of this batter's own zone: 0 is the bottom of his zone,
    # 1 the top, regardless of how tall he is.
    out["plate_z_sz"] = ((pz - bot) / height).astype("float32")
    out["abs_plate_x"] = np.abs(px).astype("float32")

    # Signed distance in feet to the nearest edge of the rulebook zone:
    # negative inside, positive outside. Horizontal and vertical are combined
    # the way the regions are — a pitch is outside if it misses on either axis,
    # and how far outside is the worse of the two misses.
    dx = np.abs(px) - ZONE_HALF_WIDTH
    dz = np.maximum(bot - pz, pz - top)
    inside = (dx <= 0) & (dz <= 0)
    out["edge_dist"] = np.where(inside, np.maximum(dx, dz),
                                np.maximum(np.maximum(dx, 0.0),
                                           np.maximum(dz, 0.0))).astype("float32")

    balls = df["balls"].to_numpy(dtype="float64")
    strikes = df["strikes"].to_numpy(dtype="float64")
    out["balls"] = balls.astype("float32")
    out["strikes"] = strikes.astype("float32")
    # One number for "who is ahead": +3 is 3-0, -2 is 0-2.
    out["count_diff"] = (balls - strikes).astype("float32")
    out["same_hand"] = (arm * bat > 0).astype("float32")

    # The rulebook zone, and the four attack regions. `in_zone` crosses heart
    # and shadow on purpose; the regions partition.
    d = out["edge_dist"].to_numpy(dtype="float64")
    out["in_zone"] = np.where(np.isnan(d), np.nan, (d <= 0).astype("float64")
                              ).astype("float32")
    out["region_heart"] = (d < -BALL_WIDTH).astype("float32")
    out["region_shadow"] = ((d >= -BALL_WIDTH) & (d <= BALL_WIDTH)).astype("float32")
    out["region_chase"] = ((d > BALL_WIDTH)
                           & (d <= BALL_WIDTH + CHASE_WIDTH)).astype("float32")
    out["region_waste"] = (d > BALL_WIDTH + CHASE_WIDTH).astype("float32")
    for r in REGIONS:
        out.loc[~np.isfinite(d), f"region_{r}"] = np.nan

    # The second target: a called strike, given the batter did not swing.
    out["is_taken"] = ~out["is_swing"].to_numpy()
    return out


def kept_columns() -> list[str]:
    """Every column the stage-1 fits or the monthly reduction read.

    Twelve seasons is 8.4 million pitches held in memory at once, and the
    stuff feature frame carries a dozen columns neither end of this module
    wants — the fastball-reference means the relative features were built
    from, the raw pitch-type strings, the swing flags. Dropping them at the
    door is a third of the footprint, on a box that is shared.
    """
    cols = {"pitcher", "game_year", "game_date", "is_csw", "is_called",
            "is_swing", "is_taken", "is_fastball", "in_zone"}
    cols.update(f"region_{r}" for r in REGIONS)
    for features in FEATURE_SETS.values():
        cols.update(features)
    return sorted(cols)


def build_year(year: int, raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """`pitch_features` plus the location block for one season, trimmed to
    the columns `kept_columns` names."""
    path = Path(raw_dir) / f"statcast_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} — pull it from R2 first")
    pitches = competitive_pitches(
        pd.read_parquet(path, columns=STATCAST_COLUMNS))
    feats = location_features(pitches, pitch_features(pitches))[kept_columns()]
    logger.info("%d: %d competitive pitches (%d taken, %.3f called|taken, "
                "%.3f in zone)", year, len(feats), int(feats["is_taken"].sum()),
                float(feats.loc[feats["is_taken"], "is_called"].mean()),
                float(np.nanmean(feats["in_zone"].to_numpy(dtype="float64"))))
    return feats


# --- the monthly reduction ---------------------------------------------------

# Every column is a sum, so any window is a sum of buckets and a first-of-month
# cutoff is answered exactly by the buckets strictly before it.
COUNT_COLUMNS = [
    "pitches", "takens",
    "cmd_csw_sum", "cmd_resid_sum", "cmd_resid_fb_sum", "cmd_resid_nfb_sum",
    "cs_taken_sum", "cs_resid_sum",
    "in_zone", *[f"n_{r}" for r in REGIONS],
    "fb_pitches",
]


def monthly_buckets(scored: pd.DataFrame) -> pd.DataFrame:
    """Additive sufficient statistics per (pitcher, season, month).

    `scored` is `build_year` output carrying four model columns:

        p_csw_pitching / p_csw_stuff        CSW | pitch, with and without
                                            location
        p_cs_pitching  / p_cs_stuff         called strike | taken, likewise

    The residual sums are the *differences* — the location contribution — and
    each is summed over its own target's denominator: CSW over every pitch,
    called-strike over taken pitches only. Summing a per-take probability over
    pitches the batter offered at would turn the aggregate into a swing-rate
    measurement wearing command's name, which is the mistake
    `pitching_stuff.monthly_buckets` documents for whiffs.

    A pitch with no tracked strike zone contributes to `pitches` and to the
    residual sums (the models route its NaNs) but to no attack region, so the
    region shares are fractions of the pitches whose region is known. Their
    denominator is therefore `n_heart + n_shadow + n_chase + n_waste`, not
    `pitches`, and `src/eval/command.py` divides by exactly that.
    """
    if scored.empty:
        cols = {"pitcher": pd.Series(dtype="int64"),
                "season": pd.Series(dtype="int64"),
                "month": pd.Series(dtype="int64")}
        cols.update({c: pd.Series(dtype="float64") for c in COUNT_COLUMNS})
        return pd.DataFrame(cols)

    taken = scored["is_taken"].to_numpy().astype("float64")
    fb = scored["is_fastball"].to_numpy().astype("float64")
    csw_resid = (scored["p_csw_pitching"].to_numpy(dtype="float64")
                 - scored["p_csw_stuff"].to_numpy(dtype="float64"))
    cs_resid = (scored["p_cs_pitching"].to_numpy(dtype="float64")
                - scored["p_cs_stuff"].to_numpy(dtype="float64")) * taken

    df = pd.DataFrame({
        "pitcher": scored["pitcher"].to_numpy(),
        "season": scored["game_year"].to_numpy(),
        "month": pd.to_datetime(scored["game_date"]).dt.month.to_numpy(),
        "pitches": 1.0,
        "takens": taken,
        "cmd_csw_sum": scored["p_csw_pitching"].to_numpy(dtype="float64"),
        "cmd_resid_sum": csw_resid,
        "cmd_resid_fb_sum": csw_resid * fb,
        "cmd_resid_nfb_sum": csw_resid * (1.0 - fb),
        "cs_taken_sum": scored["p_cs_pitching"].to_numpy(dtype="float64") * taken,
        "cs_resid_sum": cs_resid,
        "in_zone": np.nan_to_num(
            scored["in_zone"].to_numpy(dtype="float64"), nan=0.0),
        "fb_pitches": fb,
    })
    for r in REGIONS:
        df[f"n_{r}"] = np.nan_to_num(
            scored[f"region_{r}"].to_numpy(dtype="float64"), nan=0.0)

    out = df.groupby(["pitcher", "season", "month"], as_index=False)[
        COUNT_COLUMNS].sum()
    for c in ("pitcher", "season", "month"):
        out[c] = out[c].astype("int64")
    return out


DEFAULT_PATH = Path("data/features/pitching_command_monthly.parquet")

# Counts are whole pitches; the four residual/probability sums are not.
_SUM_COLUMNS = frozenset({"cmd_csw_sum", "cmd_resid_sum", "cmd_resid_fb_sum",
                          "cmd_resid_nfb_sum", "cs_taken_sum", "cs_resid_sum"})


def save_monthly(df: pd.DataFrame, path: str | Path = DEFAULT_PATH) -> Path:
    """Write the artifact. Counts int32, probability sums float32.

    A residual sum is a sum of differences of probabilities and can be
    negative; float32's seven significant digits are three more than the
    largest month of pitches needs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in COUNT_COLUMNS:
        out[c] = out[c].astype("float32" if c in _SUM_COLUMNS else "int32")
    out.to_parquet(path, index=False, compression="zstd")
    return path


def meta_path(path: str | Path = DEFAULT_PATH) -> Path:
    """The JSON sidecar next to the parquet: `<name>.meta.json`.

    Same shape and same reason as `src.data.pitching_stuff.meta_path` —
    `scripts/check_freshness.py` is stdlib-only and cannot open a parquet.
    """
    path = Path(path)
    return path.with_suffix("").with_suffix(".meta.json")


def write_meta(path: str | Path = DEFAULT_PATH, *, built_at: str | None = None,
               seasons_built: list[int] | None = None) -> Path:
    """Stamp the sidecar with when this build ran and which seasons it touched."""
    import json
    from datetime import datetime, timezone

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


# --- the vacuity check -------------------------------------------------------

VACUITY_MIN_PITCHES = 1000
VACUITY_MIN_R = 0.45


def season_aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    """Per pitcher-season command residual per pitch — the vacuity quantity."""
    g = monthly.groupby(["pitcher", "season"], as_index=False)[
        ["pitches", "cmd_resid_sum", "takens", "cs_resid_sum"]].sum()
    g["cmd_resid"] = g["cmd_resid_sum"] / g["pitches"].where(g["pitches"] > 0)
    g["cs_resid"] = g["cs_resid_sum"] / g["takens"].where(g["takens"] > 0)
    return g


def year_over_year(monthly: pd.DataFrame, column: str = "cmd_resid",
                   min_pitches: int = VACUITY_MIN_PITCHES) -> dict:
    """Correlation of a pitcher's season aggregate with his next season's.

    Pre-registered as the check that runs **before** stage 2: if a pitcher's
    command measurement does not carry from one season to the next, there is
    no talent for a talent-layer covariate to pool and no downstream test
    means anything. Restricted to pitcher-seasons of `min_pitches` in *both*
    years, pooled over every consecutive pair.
    """
    g = season_aggregate(monthly)
    g = g[g["pitches"] >= min_pitches]
    nxt = g.assign(season=g["season"] - 1)
    j = g.merge(nxt, on=["pitcher", "season"], suffixes=("", "_next"))
    j = j[np.isfinite(j[column]) & np.isfinite(j[f"{column}_next"])]
    if len(j) < 2:
        return {"column": column, "n_pairs": int(len(j)), "r": float("nan"),
                "min_pitches": min_pitches, "by_pair": []}
    r = float(np.corrcoef(j[column], j[f"{column}_next"])[0, 1])
    by_pair = [
        {"season": int(s), "n": int(len(h)),
         "r": (float(np.corrcoef(h[column], h[f"{column}_next"])[0, 1])
               if len(h) > 2 else float("nan"))}
        for s, h in j.groupby("season")
    ]
    # A diagnostic, **not** the pre-registered quantity: the same correlation
    # with each season's league level removed first. The walk-forward design
    # fits a different model for every scored season, so the residual's league
    # mean moves between seasons (it runs +0.015 in 2017 to -0.007 in 2026);
    # pooling pairs whose centres sit in different places attenuates the
    # pooled correlation even when the within-pair correlations do not move.
    # It is recorded so the failure can be read, and it does not change the
    # verdict — `r` above is the number the pre-registration names.
    g2 = g.copy()
    g2["_z"] = g2.groupby("season")[column].transform(
        lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s * 0.0)
    jz = g2.merge(g2.assign(season=g2["season"] - 1), on=["pitcher", "season"],
                  suffixes=("", "_next"))
    jz = jz[np.isfinite(jz["_z"]) & np.isfinite(jz["_z_next"])]
    demeaned = (float(np.corrcoef(jz["_z"], jz["_z_next"])[0, 1])
                if len(jz) > 2 else float("nan"))
    return {"column": column, "n_pairs": int(len(j)), "r": r,
            "min_pitches": min_pitches, "by_pair": by_pair,
            "r_season_demeaned_diagnostic": demeaned}


__all__ = [
    "BALL_WIDTH", "CHASE_WIDTH", "COUNT_COLUMNS", "DEFAULT_PATH",
    "FEATURE_SETS", "LOCATION_ONLY_FEATURES", "MIN_ZONE_HEIGHT",
    "PITCHING_FEATURES", "PITCH_TYPE_FEATURES", "REGIONS", "RESIDUAL_ARM",
    "RESIDUAL_BASE_ARM", "STATCAST_COLUMNS", "VACUITY_MIN_PITCHES",
    "VACUITY_MIN_R", "ZONE_HALF_WIDTH", "build_year", "load_monthly",
    "location_features", "meta_path", "monthly_buckets", "save_monthly",
    "kept_columns", "season_aggregate", "write_meta", "year_over_year",
]
