"""BAS-90 stage 1: the served layer-2 engines inside the per-game chain, scored.

Station A serves three things on top of the estimator station E's chain has
been running: a tuned Marcel (`marcel_tuned` / `marcel_pitcher_tuned`), an
additive stuff correction on two pitcher components, and an additive contact
correction on the five hitter ones. `src/sim/engines.py` is the switch that
lets the chain read them, as three nested rungs. This script runs the
walk-forward once per (season, rung), scores every rung against the market
close, and writes the evidence the pre-registration (docs/chain-engines.md)
asks for.

    rung 0   the chain as served — `pythag_C_sp_bpa_ip_lvl`
    rung 1   + the tuned constants and the age curve on both rate tables
    rung 2   + `stuff_additive` on the pitcher components the site serves it on
    rung 3   + `contact_additive` on the five hitter components
    recal    rung 1 with the tuned age slopes zeroed — the recalibration control

Two phases, both on by default:

    --run    walk `--seasons` forward once per rung, writing one prediction
             frame per cell under `--pred-dir` (nothing is scored here, so a
             re-score costs no compute). Also sweeps the station C blend weight
             on 2025 at rungs 0 and 3.
    --score  join the market closes, score every prediction the
             pre-registration named, and write `--json-out`.

Usage:
    python scripts/run_chain_engines.py \
        --market data/parquet/market_closes_2026.parquet \
        --json-out data/eval/chain_engines_stage1.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# The chain the nightly serves, and the rung below it — the two columns every
# comparison in the pre-registration is stated against. `_lvl`, not `_lvl_lu`:
# the card is a separate term and it is posted for a minority of games, so
# mixing it in would measure the card's coverage as well as the engine.
CHAIN = "pythag_C_sp_bpa_ip_lvl"
PRODUCTION = "pythag_60"
# Rungs of the chain that read only *some* of the two rate tables, scored per
# engine rung as a free decomposition of where an engine's effect enters:
# `pythag_60_sp` is the announced starter's FIP and nothing else (pitcher table
# only, no station C), `pythag_C` is the bottom-up run environment and no
# announced starter (both tables, no starter delta), and the rest add the terms
# back one at a time. Prediction 2 asks the same question by differencing two
# engine rungs; this asks it by differencing two *chain* rungs, and the two
# readings agreeing is worth more than either.
LADDER = ("pythag_60_sp", "pythag_C", "pythag_C_sp", "pythag_C_sp_bpa",
          "pythag_C_sp_bpa_ip", CHAIN)
KALSHI, POLYMARKET = "kalshi_close", "polymarket_close"

RUNGS = {
    "served": dict(rung=0, recalibration=False),
    "tuned": dict(rung=1, recalibration=False),
    "stuff": dict(rung=2, recalibration=False),
    "contact": dict(rung=3, recalibration=False),
    "recal": dict(rung=1, recalibration=True),
    # BAS-91 (docs/chain-engines-matched.md): the same rungs with the level and
    # spread match on, and the tuned age curve as a switch of its own at rung
    # 3. `contact` is BAS-90's name for R3 (rung 3, age on, unmatched) and is
    # reused rather than re-walked, so the two tickets score the same frame.
    "r1_match": dict(rung=1, match=True),
    "r3_noage": dict(rung=3, no_age=True),
    "r3_match": dict(rung=3, match=True),
    "r3_noage_match": dict(rung=3, no_age=True, match=True),
}
RUNG_ORDER = ("served", "tuned", "stuff", "contact", "recal")
# BAS-91's seven, in the pre-registration's own order. `recal` is rung 1 with
# the age slopes zeroed and unmatched; `contact` is R3.
MATCHED_ORDER = ("served", "recal", "r1_match", "contact", "r3_noage",
                 "r3_match", "r3_noage_match")
ARM_SETS = {"bas90": RUNG_ORDER, "bas91": MATCHED_ORDER}
# What each arm is called in docs/chain-engines-matched.md, so the evidence and
# the report read as the pre-registration wrote them.
ARM_LABELS = {"served": "served", "recal": "recal", "r1_match": "R1-match",
              "contact": "R3", "r3_noage": "R3-noage",
              "r3_match": "R3-match", "r3_noage_match": "R3-noage-match",
              "tuned": "rung 1", "stuff": "rung 2"}
# The three matched arms prediction 1 is about.
MATCHED_ARMS = ("r1_match", "r3_match", "r3_noage_match")
# The grid station C's blend weight was chosen on, walk-forward on 2025 only
# (docs/market-benchmark-2026.md). Re-swept here at the top rung, because the
# obvious worry about a better bottom-up half is that it deserves more weight.
BLEND_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
BLEND_SEASON = 2025
BLEND_RUNGS = ("served", "contact")
# Calibration buckets, the ones `backtest_game_odds.py` prints.
BUCKETS = [0, .4, .45, .5, .55, .6, .65, 1.0]
# The bucket prediction 4 is about: where the chain has been overconfident.
TOP_BUCKET = 0.65


# --- phase 1: walk each rung forward -----------------------------------------

def pred_path(pred_dir: Path, season: int, name: str,
              blend: float | None = None) -> Path:
    tag = f"{name}_{season}" if blend is None else f"{name}_{season}_w{blend:g}"
    return pred_dir / f"chain_engines_{tag}.parquet"


def walk(season: int, name: str, pred_dir: Path, min_games: int,
         blend: float | None = None, force: bool = False) -> Path:
    """One walk-forward season at one rung, cached on its output file.

    Shelling out rather than importing keeps this an exact reproduction of the
    scored harness: the same `main()` the market benchmark was produced by,
    with two flags added, so no code path here can diverge from the one the
    scoreboard runs.
    """
    out = pred_path(pred_dir, season, name, blend)
    if out.exists() and not force:
        print(f"  {out.name}: cached")
        return out
    spec = RUNGS[name]
    cmd = [sys.executable, str(ROOT / "scripts" / "backtest_game_odds.py"),
           "--season", str(season), "--min-games", str(min_games),
           "--engine-rung", str(spec["rung"]), "--out", str(out)]
    if spec.get("recalibration"):
        cmd.append("--engine-recalibration")
    if spec.get("no_age"):
        cmd.append("--engine-no-age")
    if spec.get("match"):
        cmd.append("--engine-match")
    if blend is not None:
        cmd += ["--c-weight", f"{blend:g}"]
    print(f"  {out.name}: running")
    subprocess.run(cmd, check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return out


# --- scoring ----------------------------------------------------------------

def brier(p, y) -> float:
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return float(np.mean((p - np.asarray(y, dtype="float64")) ** 2))


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype="float64")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def paired(p_model, p_base, y) -> dict:
    """The paired per-game Brier difference, its standard error and its t.

    Brier is a mean of per-game squared errors, so two models scored on the
    same games are a paired sample. `backtest_game_odds.paired_t_line` computes
    exactly this; it is repeated here because this script scores subsets that
    harness never joins.
    """
    y = np.asarray(y, dtype="float64")
    d = ((np.asarray(p_model, dtype="float64") - y) ** 2
         - (np.asarray(p_base, dtype="float64") - y) ** 2)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"diff": float(d.mean()), "se": se,
            "t": float(d.mean() / se) if se else float("nan"), "n": int(n)}


def calibration(df: pd.DataFrame, column: str) -> list[dict]:
    cut = pd.cut(df[column], BUCKETS)
    g = df.groupby(cut, observed=True).agg(
        n=("home_win", "size"), predicted=(column, "mean"),
        realized=("home_win", "mean"))
    return [{"bucket": str(b), "n": int(r["n"]),
             "predicted": float(r["predicted"]),
             "realized": float(r["realized"]),
             "over": float(r["predicted"] - r["realized"])}
            for b, r in g.iterrows()]


def top_bucket(df: pd.DataFrame, column: str) -> dict:
    """Prediction 4's bucket: the confident home favourites, on their own.

    `> 0.65` rather than the last `pd.cut` bin so the number does not move when
    the bucket edges do, and because that is how the pre-registration names it.
    """
    hot = df[df[column] > TOP_BUCKET]
    if hot.empty:
        return {"n": 0, "predicted": None, "realized": None, "over": None}
    pred = float(hot[column].mean())
    real = float(hot["home_win"].astype(float).mean())
    return {"n": int(len(hot)), "predicted": pred, "realized": real,
            "over": pred - real}


def join_market(preds: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """`backtest_game_odds.join_market`, kept to games every venue priced."""
    wide = closes.pivot_table(index="game_pk", columns="venue",
                              values="p_home_close")
    wide.columns = [f"{v}_close" for v in wide.columns]
    joined = preds.merge(wide, left_on="game_pk", right_index=True, how="inner")
    return joined.dropna(subset=list(wide.columns))


def load_rungs(pred_dir: Path, season: int, order=RUNG_ORDER) -> pd.DataFrame:
    """One frame: the harness's own scored rows, plus one column per rung.

    **Aligned by position, not joined on `game_pk`**, and that is not a
    shortcut. `backtest_game_odds.walk_forward` emits one row per scored game
    in a deterministic order — the same `scored` frame, sorted the same way,
    walked date by date — so row *i* is the same game in every rung's output.
    It also emits a handful of repeated `game_pk`s (one in 2026, four in 2025:
    a suspended game that appears twice on the schedule), which is the
    population every published Brier in docs/market-benchmark-2026.md is
    computed on. Merging on `game_pk` would turn each of those into 2^5 rows
    and quietly inflate every set; aligning by position keeps exactly the
    harness's rows and nothing else.

    The keys are checked rather than assumed: if any rung's `game_pk`, date or
    clubs differ row for row, the runs are not the same walk-forward and this
    raises instead of pairing two different populations.
    """
    out, sizes = None, {}
    keys = ["game_pk", "date", "home_id", "away_id", "home_win"]
    for name in order:
        df = pd.read_parquet(pred_path(pred_dir, season, name)
                             ).reset_index(drop=True)
        sizes[name] = len(df)
        if out is None:
            out = df[[*keys, PRODUCTION]].copy()
        elif not out[keys].equals(df[keys]):
            raise ValueError(
                f"{season}: rung {name!r} scored a different set of games than "
                f"{order[0]!r} — re-run the batch so every rung walks the "
                "same season")
        for column in LADDER:
            out[f"{name}::{column}"] = df[column].to_numpy(dtype="float64")
        # The served chain is both this rung's own column and the top of its
        # ladder; carrying it twice keeps every table below reading one name.
        out[name] = out[f"{name}::{CHAIN}"]
    out.attrs["per_rung_games"] = sizes
    out.attrs["repeated_game_pks"] = int(len(out) - out["game_pk"].nunique())
    return out


def score_set(df: pd.DataFrame, models: list[str]) -> list[dict]:
    y = df["home_win"].astype(float).to_numpy()
    rows = []
    for m in models:
        if m not in df.columns:
            continue
        p = df[m].to_numpy(dtype="float64")
        row = {"model": m, "brier": brier(p, y), "log_loss": log_loss(p, y),
               "mean_p_home": float(np.clip(p, 1e-6, 1 - 1e-6).mean())}
        if m != "served":
            row["paired_vs_served"] = paired(p, df["served"].to_numpy(), y)
        rows.append(row)
    return rows


def recalibration_slope(df: pd.DataFrame, column: str) -> dict:
    """Slope of a one-parameter logistic recalibration of the model's log-odds.

    1.0 is perfectly calibrated in spread, below it is overconfident and above
    it under-confident — the statistic docs/market-benchmark-2026.md reads the
    innings-level term's calibration cost with. It is here because the obvious
    way a better *rate table* makes a worse *game* model is by widening the
    spread of a chain whose ballasts, blend weight and lineup weight were all
    chosen against the old spread, and that shows up as a slope, not as a mean.

    Newton on two parameters (intercept and slope), which converges in a
    handful of steps on this shape and needs no optimiser.
    """
    p = np.clip(df[column].to_numpy(dtype="float64"), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    y = df["home_win"].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.array([0.0, 1.0])
    for _ in range(50):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return {"intercept": float(beta[0]), "slope": float(beta[1]),
            "sd_p_home": float(p.std(ddof=1))}


def deviation_correlation(df: pd.DataFrame, column: str) -> float:
    """corr(our deviation from `pythag_60`, the market's deviation from it).

    Prediction 3's statistic. Both sides are measured against the *production*
    team-strength model rather than against each other, so it asks whether the
    chain moves a game the way the market moves it rather than whether the two
    agree in level.
    """
    ours = df[column].to_numpy(dtype="float64") - df[PRODUCTION].to_numpy(dtype="float64")
    theirs = df[KALSHI].to_numpy(dtype="float64") - df[PRODUCTION].to_numpy(dtype="float64")
    return float(np.corrcoef(ours, theirs)[0, 1])


# --- the pre-registered predictions ------------------------------------------

def score_predictions(market: pd.DataFrame, all_2026: pd.DataFrame,
                      all_2025: pd.DataFrame) -> dict:
    """Predictions 1-5, the vacuity check and the recalibration control.

    Scored exactly as written in docs/chain-engines.md, each one reduced to a
    boolean beside the number it was decided on, so nothing here is a matter of
    reading the table generously.
    """
    y_m = market["home_win"].astype(float).to_numpy()
    d_market = paired(market["contact"], market["served"], y_m)
    d_2026 = paired(all_2026["contact"], all_2026["served"],
                    all_2026["home_win"].astype(float).to_numpy())
    d_2025 = paired(all_2025["contact"], all_2025["served"],
                    all_2025["home_win"].astype(float).to_numpy())

    # 2: the pitcher side against the whole ladder, on the market games. Two
    # readings of "the gain", because rung 1 is not free: against the served
    # chain (what ships) and against rung 1 (what the corrections add).
    pitcher = paired(market["stuff"], market["tuned"], y_m)["diff"]
    hitter = paired(market["contact"], market["stuff"], y_m)["diff"]
    total_served = d_market["diff"]
    total_tuned = paired(market["contact"], market["tuned"], y_m)["diff"]
    share = (pitcher / total_served if total_served < 0 else float("nan"))
    share_tuned = (pitcher / total_tuned if total_tuned < 0 else float("nan"))

    # Kalshi is "the market" here because it is the venue the benchmark's own
    # residual is quoted against (docs/market-benchmark-2026.md: 0.24364 -
    # 0.24201 = 0.00163); Polymarket rides along so a reader can see the answer
    # is not one venue's.
    corr_served = deviation_correlation(market, "served")
    corr_contact = deviation_correlation(market, "contact")
    resid_served = brier(market["served"], y_m) - brier(market[KALSHI], y_m)
    resid_contact = brier(market["contact"], y_m) - brier(market[KALSHI], y_m)
    resid_poly = {name: brier(market[name], y_m) - brier(market[POLYMARKET], y_m)
                  for name in RUNG_ORDER}
    corr_all = {name: deviation_correlation(market, name) for name in RUNG_ORDER}

    over_served = top_bucket(market, "served")
    over_contact = top_bucket(market, "contact")

    null = paired(market["tuned"], market["served"], y_m)
    delta = float(np.mean(np.abs(market["contact"].to_numpy(dtype="float64")
                                 - market["served"].to_numpy(dtype="float64"))))
    recal_gap = float(brier(market["recal"], y_m) - brier(market["contact"], y_m))

    signs = [d_market["diff"], d_2026["diff"], d_2025["diff"]]
    same_sign = all(s < 0 for s in signs) or all(s > 0 for s in signs)
    return {
        "1_rung3_vs_served": {
            "market": d_market, "all_2026": d_2026, "all_2025": d_2025,
            "threshold": -0.00020,
            "passes": bool(d_market["diff"] <= -0.00020 and same_sign),
            "same_sign_on_all_sets": bool(same_sign)},
        "2_pitcher_side_carries_half": {
            "pitcher_rung2_minus_rung1": pitcher,
            "hitter_rung3_minus_rung2": hitter,
            "total_rung3_minus_served": total_served,
            "total_rung3_minus_rung1": total_tuned,
            "share_of_served_gain": share,
            "share_of_correction_gain": share_tuned,
            "passes": bool(share >= 0.5) if np.isfinite(share) else False},
        "3_market_agreement": {
            "corr_served": corr_served, "corr_rung3": corr_contact,
            "corr_threshold": 0.79,
            "residual_served": resid_served, "residual_rung3": resid_contact,
            "residual_threshold": 0.00140,
            "corr_by_rung": corr_all,
            "residual_vs_polymarket_by_rung": resid_poly,
            "passes": bool(corr_contact >= 0.79 and resid_contact < 0.00140)},
        "4_top_bucket_not_more_overconfident": {
            "served": over_served, "rung3": over_contact,
            "passes": bool(over_contact["over"] is not None
                           and over_served["over"] is not None
                           and over_contact["over"] <= over_served["over"])},
        "5_null_rung1_alone": {
            **null, "threshold": 0.00010,
            "passes": bool(abs(null["diff"]) < 0.00010)},
        "vacuity_mean_abs_delta_p_home": {
            "value": delta, "threshold": 0.003,
            "passes": bool(delta > 0.003)},
        "recalibration_control": {
            "brier_recal": brier(market["recal"], y_m),
            "brier_rung3": brier(market["contact"], y_m),
            "gap": recal_gap, "threshold": 0.00005,
            "matches_rung3": bool(abs(recal_gap) < 0.00005)},
    }


def verdict(predictions: dict) -> dict:
    """The pre-registration's own failure conditions, applied to its own numbers.

    docs/chain-engines.md: "Prediction 1 fails, or the sign disagrees across
    the three sets, or the recalibration control matches rung 3 within .00005
    ... nothing ships." Each is a separate boolean here, and `ships` is their
    conjunction, so the verdict is read off the file rather than off a table.
    Note that the third condition firing is the *bad* case: a control that
    matches rung 3 means the gain was the ballasts.
    """
    fails = {
        "prediction_1_failed": not predictions["1_rung3_vs_served"]["passes"],
        "sign_disagrees_across_sets":
            not predictions["1_rung3_vs_served"]["same_sign_on_all_sets"],
        "recalibration_control_matches_rung3":
            predictions["recalibration_control"]["matches_rung3"],
    }
    return {**fails,
            "vacuous": not predictions["vacuity_mean_abs_delta_p_home"]["passes"],
            "ships": not any(fails.values())}


def market_agreement(market: pd.DataFrame, order) -> dict:
    """Per arm: how far the chain's deviation from `pythag_60` tracks the
    market's, and how much Brier it still gives up to each venue's close.

    BAS-90's prediction 3 read these two numbers off rung 3 alone; BAS-91 reads
    them off every arm, because the claim under test is that a *matched* table
    prices the game the way the served one does, and a correlation that comes
    back to served's while the Brier does not would say the spread was the
    whole story.
    """
    y = market["home_win"].astype(float).to_numpy()
    return {name: {
        "corr_with_market_deviation": deviation_correlation(market, name),
        "residual_vs_kalshi": brier(market[name], y) - brier(market[KALSHI], y),
        "residual_vs_polymarket": (brier(market[name], y)
                                   - brier(market[POLYMARKET], y))}
        for name in order}


def score_predictions_matched(market: pd.DataFrame, all_2026: pd.DataFrame,
                              all_2025: pd.DataFrame,
                              pooled: pd.DataFrame) -> dict:
    """BAS-91's four predictions and its vacuity check, as written.

    docs/chain-engines-matched.md, in order: (1) every matched arm's logistic
    slope within .03 of served's on each of the three sets; (2) R3-match −
    R3-noage-match >= +.00030 on the market set with the same sign on all 2025
    and all 2026; (3) R3-noage-match − served <= −.00020 on the pooled
    2025 + 2026 set with the same sign on the market set; (4) R1-match − served
    >= +.00040 on the market set. Each is a boolean beside the number it was
    decided on, and the numbers the pre-registration quotes as context (the
    served slope, the pooled se) are carried with them.
    """
    sets = {"market": market, "all_2026": all_2026, "all_2025": all_2025}
    y_m = market["home_win"].astype(float).to_numpy()

    slopes = {label: {name: recalibration_slope(df, name)["slope"]
                      for name in MATCHED_ORDER}
              for label, df in sets.items()}
    gaps = {label: {arm: slopes[label][arm] - slopes[label]["served"]
                    for arm in MATCHED_ARMS} for label in sets}
    p1 = all(abs(g) <= 0.03 for label in gaps for g in gaps[label].values())

    age_cost = {label: paired(df["r3_match"], df["r3_noage_match"],
                              df["home_win"].astype(float).to_numpy())
                for label, df in sets.items()}
    p2 = bool(age_cost["market"]["diff"] >= 0.00030
              and age_cost["all_2026"]["diff"] > 0
              and age_cost["all_2025"]["diff"] > 0)

    ship = paired(pooled["r3_noage_match"], pooled["served"],
                  pooled["home_win"].astype(float).to_numpy())
    ship_market = paired(market["r3_noage_match"], market["served"], y_m)
    p3 = bool(ship["diff"] <= -0.00020 and ship_market["diff"] < 0)

    rung1 = paired(market["r1_match"], market["served"], y_m)
    p4 = bool(rung1["diff"] >= 0.00040)

    delta = float(np.mean(np.abs(
        market["r3_noage_match"].to_numpy(dtype="float64")
        - market["served"].to_numpy(dtype="float64"))))
    return {
        "1_matching_restores_calibration": {
            "slopes": slopes, "gap_vs_served": gaps, "threshold": 0.03,
            "worst_gap": max(abs(g) for label in gaps
                             for g in gaps[label].values()),
            "passes": bool(p1)},
        "2_age_curve_costs_at_matched_tables": {
            **{label: age_cost[label] for label in sets},
            "threshold": 0.00030,
            "same_sign_on_all_sets": bool(
                age_cost["all_2026"]["diff"] > 0
                and age_cost["all_2025"]["diff"] > 0),
            "passes": p2},
        "3_ship_test_r3_noage_match_vs_served": {
            "pooled": ship, "market": ship_market,
            "all_2026": paired(all_2026["r3_noage_match"], all_2026["served"],
                               all_2026["home_win"].astype(float).to_numpy()),
            "all_2025": paired(all_2025["r3_noage_match"], all_2025["served"],
                               all_2025["home_win"].astype(float).to_numpy()),
            "threshold": -0.00020, "ship_t_threshold": -2.0,
            "passes": p3,
            "ships": bool(p3 and ship["t"] <= -2.0)},
        "4_null_matching_does_not_rescue_rung1": {
            **rung1, "threshold": 0.00040, "passes": p4,
            # The failure condition names its own reading: a rung-1 arm that
            # lands on top of served says the spread was the mechanism at rung
            # 1 as well, and M1 is dropped.
            "within_00010_of_served": bool(abs(rung1["diff"]) < 0.00010)},
        "vacuity_mean_abs_delta_p_home": {
            "value": delta, "threshold": 0.003,
            "passes": bool(delta > 0.003)},
    }


def verdict_matched(predictions: dict) -> dict:
    """BAS-91's failure conditions and ship rule, applied to its own numbers.

    docs/chain-engines-matched.md: "If 1 fails, matching is not the fix and the
    spread story is wrong... If 3 fails on sign, nothing ships. If 4 fails
    (R1-match within +.00010 of served), the doc records that spread was the
    mechanism at rung 1 too and M1 is dropped." And: "Only if prediction 3
    holds **and** t <= −2.0 on the pooled set" does anything ship.
    """
    p1 = predictions["1_matching_restores_calibration"]
    p3 = predictions["3_ship_test_r3_noage_match_vs_served"]
    p4 = predictions["4_null_matching_does_not_rescue_rung1"]
    return {
        "matching_is_not_the_fix": not p1["passes"],
        "ship_test_failed": not p3["passes"],
        "ship_test_wrong_sign": bool(p3["pooled"]["diff"] > 0
                                     or p3["market"]["diff"] > 0),
        "m1_dropped_spread_was_the_mechanism_at_rung1":
            p4["within_00010_of_served"],
        "vacuous": not predictions["vacuity_mean_abs_delta_p_home"]["passes"],
        "ships": bool(p3["ships"]),
    }


def blend_sweep(pred_dir: Path) -> dict:
    """The station C blend weight, re-swept on 2025 at rungs 0 and 3."""
    out: dict = {"season": BLEND_SEASON, "grid": list(BLEND_GRID), "rows": {}}
    for name in BLEND_RUNGS:
        row = {}
        for w in BLEND_GRID:
            path = pred_path(pred_dir, BLEND_SEASON, name, w)
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            row[f"{w:g}"] = {
                "brier": brier(df[CHAIN], df["home_win"].astype(float)),
                "n": int(len(df))}
        if row:
            best = min(row, key=lambda k: row[k]["brier"])
            out["rows"][name] = {"by_weight": row, "argmin": best}
    return out


def engine_provenance(season: int, pa_dir: Path) -> dict:
    """What the two correction fits and the tuned params actually were.

    Re-fits rungs 2 and 3 for `season` — the same call the walk-forward made,
    which is deterministic — and records the coefficients, the cell seasons and
    the row counts, so the evidence says what was in the model rather than
    naming a params file and a date.
    """
    from src.eval import pitchers as pitcher_eval
    from src.eval.baselines import load_marcel_params
    from src.sim import engines as eng_model

    stuff_monthly = pd.read_parquet(ROOT / "data/features/pitching_stuff_monthly.parquet")
    contact_monthly = pd.read_parquet(ROOT / "data/features/contact_quality_monthly.parquet")
    p_seasons = pitcher_eval.normalize_pitcher_seasons(
        pd.read_parquet(ROOT / "data/parquet/pitcher_seasons_api.parquet"))
    h_seasons = pd.read_parquet(ROOT / "data/parquet/hitter_seasons_api.parquet")

    fits = {}
    for side, table, monthly, components in (
            ("stuff", p_seasons, stuff_monthly, eng_model.STUFF_COMPONENTS),
            ("contact", h_seasons, contact_monthly, eng_model.CONTACT_COMPONENTS)):
        raw = eng_model.fit_side(
            "pitcher" if side == "stuff" else "hitter", table, monthly, pa_dir,
            season, components)
        fits[side] = {c: {"coef": {k: float(v) for k, v in f.coef.items()},
                          "n_cells": f.n_cells, "n_rows": f.n_rows,
                          "cell_seasons": list(f.seasons)}
                      for c, f in raw.items()}
    pitcher_params = pitcher_eval.load_pitcher_params()
    hitter_params = load_marcel_params()
    return {
        "season": season,
        "fits": fits,
        "tuned_params": {
            "pitcher": {c: pitcher_params[c].to_dict()
                        for c in eng_model.PITCHER_COMPONENTS.values()},
            "hitter": {c: hitter_params[c].to_dict()
                       for c in eng_model.HITTER_COMPONENTS.values()},
        },
        # The one invariant the corrections break and the chain's docstring
        # claims: a delta term cannot move the league's run environment, but an
        # additive recalibration of every pitcher's rates moves every club's
        # bottom-up runs allowed in the same direction, and station C's blend
        # carries that level rather than re-centring it.
        "note": ("stuff_additive and contact_additive carry an intercept, so "
                 "they shift the whole population's rates; the chain's "
                 "bottom-up half is a level, not a delta, and passes that "
                 "shift into the blended run environment"),
    }


def rate_table_spread(season: int, as_of: str, pa_dir: Path,
                      order=RUNG_ORDER) -> dict:
    """What each rung does to the two rate tables themselves, on one date.

    A Brier difference says the chain got worse; it does not say what moved.
    This rebuilds the pitcher table (FIP runs allowed per nine, the number
    `blend_starter_team` and the rotation both read) and the hitter table (runs
    above average per plate appearance, what the lineup and station C's runs
    scored both read) at each rung, from appearances strictly before `as_of`,
    and records the population's mean and spread. The chain's own constants —
    the ballasts, the blend weight, the lineup weight — were all chosen against
    the spread the stock tables have, so a rung that widens it is changing
    something those constants were fitted to.

    Uses the harness's own context builders, so the tables are the ones the
    walk-forward actually priced that date with.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_bgo", ROOT / "scripts" / "backtest_game_odds.py")
    bgo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bgo
    spec.loader.exec_module(bgo)

    from src.eval import pitchers as pitcher_eval
    from src.sim import engines as eng_model
    from src.sim import lineups as lu_model
    from src.sim import starters as sp_model

    teams = bgo.fetch_teams(season)
    sched = bgo.fetch_schedule(f"{season}-03-01", f"{season}-10-15")
    scored = sched[sched["status"] == "Final"].dropna(
        subset=["home_score", "away_score"])
    scored = scored[(scored["home_score"] != scored["away_score"])
                    & (scored["game_type"] == "R")].copy()
    scored["home_win"] = scored["home_score"] > scored["away_score"]

    p_seasons = pitcher_eval.normalize_pitcher_seasons(
        pd.read_parquet(ROOT / "data/parquet/pitcher_seasons_api.parquet"))
    h_seasons = pd.read_parquet(ROOT / "data/parquet/hitter_seasons_api.parquet")
    stuff_monthly = pd.read_parquet(ROOT / "data/features/pitching_stuff_monthly.parquet")
    contact_monthly = pd.read_parquet(ROOT / "data/features/contact_quality_monthly.parquet")

    # The fetches do not depend on the engine — the rung only enters at rate
    # time — so both contexts are built once and the engine is swapped in.
    sp_ctx = bgo.build_sp_context(season, scored, sp_model.BALLAST_BF,
                                  sp_model.STARTER_IP)
    lu_ctx = bgo.build_lu_context(
        season, scored, lu_model.BALLAST, lu_model.WEIGHT,
        lu_model.BASELINE, lu_model.BASELINE_BALLAST_GAMES,
        pa_per_game=sp_ctx["league"]["bf_per_ip"] * 9.0)

    out: dict = {"season": season, "as_of": as_of}
    for name in order:
        engine = eng_model.build_engines(
            RUNGS[name]["rung"], season,
            recalibration=RUNGS[name].get("recalibration", False),
            age_slopes=not RUNGS[name].get("no_age", False),
            match=RUNGS[name].get("match", False),
            pitcher_seasons=p_seasons, hitter_seasons=h_seasons, pa_dir=pa_dir,
            stuff_monthly=stuff_monthly, contact_monthly=contact_monthly)
        p_counts = pd.concat(
            [sp_ctx["prior_counts"],
             sp_model.appearances_before(sp_ctx["game_logs"], as_of)],
            ignore_index=True)
        p_rates = sp_model.marcel_rates(p_counts, season, sp_ctx["league"],
                                        ballast=sp_ctx["ballast"],
                                        engine=engine, as_of=as_of)
        ra9 = pd.Series(sp_model.starter_ra9_lookup(p_rates, sp_ctx["league"],
                                                    4.5))
        counts = pd.concat([lu_ctx["prior_counts"],
                            lu_model.games_before(lu_ctx["game_logs"], as_of)],
                           ignore_index=True)
        h_rates = lu_model.marcel_rates(counts, season, lu_ctx["league"],
                                        ballast=lu_ctx["ballast"],
                                        engine=engine, as_of=as_of)
        raa = pd.Series(lu_model.batter_runs_lookup(h_rates, lu_ctx["league"]))
        out[name] = {
            "pitcher_fip_ra9": {"n": int(len(ra9)), "mean": float(ra9.mean()),
                                "sd": float(ra9.std(ddof=1)),
                                "p05": float(ra9.quantile(0.05)),
                                "p95": float(ra9.quantile(0.95))},
            "hitter_runs_per_pa": {"n": int(len(raa)), "mean": float(raa.mean()),
                                   "sd": float(raa.std(ddof=1)),
                                   "p05": float(raa.quantile(0.05)),
                                   "p95": float(raa.quantile(0.95))},
            # The columns the match itself acts on, weighted by the effective
            # sample each table carries. `match` is an equality on exactly
            # these numbers, so they are what says the transform did what it
            # claims on the real population rather than on a synthetic frame —
            # and the two derived numbers above say what a linear rescale of a
            # rate becomes after FIP and the runs-above-average map, which is
            # the risk docs/chain-engines-matched.md records up front.
            "weighted_rate_moments": {
                "pitcher": _rate_moments(p_rates, "bf_weighted"),
                "hitter": _rate_moments(h_rates, "pa_weighted"),
            },
        }
    return out


def _rate_moments(table: pd.DataFrame, weight_col: str) -> dict:
    """{rate column: weighted mean and sd} — the moments `match` equalises."""
    from src.sim.engines import weighted_moments

    if weight_col not in table.columns:
        return {}
    w = table[weight_col].to_numpy(dtype="float64")
    out = {}
    for column in [c for c in table.columns if c.startswith("rate_")]:
        mean, sd = weighted_moments(table[column].to_numpy(dtype="float64"), w)
        out[column] = {"mean": mean, "sd": sd}
    return out


def league_shift(frames: dict, market: pd.DataFrame, order=RUNG_ORDER) -> dict:
    """How far each rung moves the *mean* P(home), which is the level check.

    A term that only redistributes strength leaves the mean where it was; one
    that moves the league's run environment moves it, because Pythagenpat is
    not linear in the environment. Reported so the reading of a Brier
    difference can tell "the model reordered the games" from "the model changed
    the scale everything is priced on".
    """
    out = {}
    for label, df in [("market", market), *[(f"all_{s}", f)
                                            for s, f in frames.items()]]:
        base = df["served"].to_numpy(dtype="float64")
        out[label] = {name: float(np.mean(df[name].to_numpy(dtype="float64")
                                          - base))
                      for name in order if name != "served"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2026,2025")
    ap.add_argument("--min-games", type=int, default=20)
    ap.add_argument("--pred-dir", type=Path,
                    default=Path("data/eval/chain_engines_preds"))
    ap.add_argument("--market", type=Path,
                    default=Path("data/parquet/market_closes_2026.parquet"))
    ap.add_argument("--json-out", type=Path,
                    default=Path("data/eval/chain_engines_stage1.json"))
    ap.add_argument("--ticket", choices=sorted(ARM_SETS), default="bas90",
                    help="which pre-registration's arms to walk and score: "
                         "bas90 (docs/chain-engines.md, the three nested "
                         "rungs and the recalibration control) or bas91 "
                         "(docs/chain-engines-matched.md, the seven matched "
                         "and age-switched arms)")
    ap.add_argument("--market-ids", type=Path, default=None,
                    help="a BAS-90-shaped evidence JSON whose market.game_ids "
                         "the market set is frozen to, so a later ticket "
                         "scores the same games as the one it follows rather "
                         "than whatever the archive holds today")
    ap.add_argument("--pa-dir", type=Path,
                    default=Path("data/parquet/pa_outcomes"),
                    help="plate-appearance outcomes, for the provenance re-fit")
    ap.add_argument("--run", action="store_true", help="only walk the seasons")
    ap.add_argument("--score", action="store_true", help="only score them")
    ap.add_argument("--force", action="store_true",
                    help="re-walk even when a prediction frame exists")
    ap.add_argument("--no-blend-sweep", action="store_true")
    ap.add_argument("--no-rate-spread", action="store_true")
    ap.add_argument("--rate-spread-date", default="2026-08-15",
                    help="the date the two rate tables are compared on, rung "
                         "by rung — a mid-season date well inside the market "
                         "window, so both tables are on a full sample")
    args = ap.parse_args()
    do_run = args.run or not args.score
    do_score = args.score or not args.run
    seasons = [int(s) for s in args.seasons.split(",")]
    args.pred_dir.mkdir(parents=True, exist_ok=True)

    arms = ARM_SETS[args.ticket]

    if do_run:
        for season in seasons:
            print(f"season {season}:")
            for name in arms:
                walk(season, name, args.pred_dir, args.min_games,
                     force=args.force)
        if not args.no_blend_sweep and BLEND_SEASON in seasons:
            print(f"blend sweep, {BLEND_SEASON}:")
            for name in BLEND_RUNGS:
                for w in BLEND_GRID:
                    walk(BLEND_SEASON, name, args.pred_dir, args.min_games,
                         blend=w, force=args.force)
    if not do_score:
        return

    closes = pd.read_parquet(args.market)
    frames = {season: load_rungs(args.pred_dir, season, arms)
              for season in seasons}
    market = join_market(frames[2026], closes).sort_values("game_pk")
    frozen = None
    if args.market_ids is not None:
        # The set the ticket being followed scored, by id. Restricting rather
        # than re-deriving is the point: `data/parquet/` is gitignored, the
        # archive can drift, and a follow-up that scored a different 756 games
        # would not be comparable to the numbers it is testing against.
        frozen = [int(g) for g in
                  json.loads(args.market_ids.read_text())["market"]["game_ids"]]
        held = market["game_pk"].astype("int64").tolist()
        missing = sorted(set(frozen) - set(held))
        market = market[market["game_pk"].isin(frozen)]
        if missing:
            print(f"warning: {len(missing)} frozen market ids are not in "
                  f"today's archive join; scoring {len(market)}")
    # The pre-registration names a 737-game Kalshi re-pull, 07-07 -> 09-02.
    # That re-pull is not in the committed archive (`data/parquet/` is
    # gitignored and the file on disk is the earlier one: Kalshi from 06-26,
    # Polymarket from 07-04, 756 games priced by both), and pulling the
    # exchanges again today would produce a third set rather than that one. So
    # the primary set here is the 756 the archive supports, and this is the
    # same archive cut to the pre-registered window — the two bracket the set
    # the prediction was written against, and every number is reported on both.
    window = market[(market["date"].astype(str) >= "2026-07-07")
                    & (market["date"].astype(str) <= "2026-09-02")]
    models = [*arms, PRODUCTION, KALSHI, POLYMARKET]
    if args.ticket == "bas91":
        # The pooled 2025 + 2026 set prediction 3 is stated on. The two seasons
        # are separate walk-forwards on disjoint games, so stacking their rows
        # is one paired sample of ~3,990 games and nothing is double counted.
        pooled = pd.concat([frames[season] for season in sorted(seasons)],
                           ignore_index=True)
        predictions = score_predictions_matched(market, frames[2026],
                                                frames[2025], pooled)
        verdict_block = verdict_matched(predictions)
    else:
        pooled = None
        predictions = score_predictions(market, frames[2026], frames[2025])
        verdict_block = verdict(predictions)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticket": args.ticket.upper().replace("BAS", "BAS-"),
        "chain_column": CHAIN,
        "arms": {name: {"label": ARM_LABELS[name], **RUNGS[name]}
                 for name in arms},
        "rungs": {name: RUNGS[name] for name in arms},
        "market_file": args.market.name,
        "market": {
            "n_games": int(len(market)),
            "first_date": str(market["date"].min()),
            "last_date": str(market["date"].max()),
            "venues": sorted(closes["venue"].unique().tolist()),
            # The exact set every number under `sets.market` was computed on.
            "game_ids": [int(g) for g in market["game_pk"]],
            "frozen_ids_from": (None if frozen is None
                                else str(args.market_ids)),
            "n_frozen_ids": (None if frozen is None else len(frozen)),
            "scores": score_set(market, models),
            "agreement": market_agreement(market, arms),
        },
        "market_window_0707": {
            "n_games": int(len(window)),
            "first_date": str(window["date"].min()),
            "last_date": str(window["date"].max()),
            "game_ids": [int(g) for g in window["game_pk"]],
            "scores": score_set(window, models),
            "predictions": (score_predictions_matched(window, frames[2026],
                                                      frames[2025], pooled)
                            if args.ticket == "bas91"
                            else score_predictions(window, frames[2026],
                                                   frames[2025])),
        },
        "sets": {
            f"all_{season}": {
                "n_games": int(len(frames[season])),
                "per_rung_games": frames[season].attrs["per_rung_games"],
                "repeated_game_pks": frames[season].attrs["repeated_game_pks"],
                "first_date": str(frames[season]["date"].min()),
                "last_date": str(frames[season]["date"].max()),
                "scores": score_set(frames[season], [*arms, PRODUCTION])}
            for season in seasons
        },
        "pooled_2025_2026": ({} if pooled is None else {
            "n_games": int(len(pooled)),
            "seasons": sorted(seasons),
            "scores": score_set(pooled, [*arms, PRODUCTION])}),
        "predictions": predictions,
        "verdict": verdict_block,
        "engines": {str(season): engine_provenance(season, args.pa_dir)
                    for season in seasons},
        "mean_p_home_shift": league_shift(frames, market, arms),
        "rate_table_spread": (
            {} if args.no_rate_spread
            else rate_table_spread(2026, args.rate_spread_date,
                                   args.pa_dir, arms)),
        # Where in the chain each engine rung's effect enters: one Brier per
        # (engine rung, chain rung), all on the same games.
        "ladder": {
            label: {chain_rung: {name: brier(
                df[f"{name}::{chain_rung}"],
                df["home_win"].astype(float)) for name in arms}
                for chain_rung in LADDER}
            for label, df in [("market", market),
                              *[(f"all_{s}", f) for s, f in frames.items()]]
        },
        "spread": {
            label: {name: recalibration_slope(df, name) for name in arms}
            for label, df in [("market", market),
                              ("market_window_0707", window),
                              *[(f"all_{s}", f) for s, f in frames.items()]]
        },
        "calibration": {
            label: {name: calibration(df, name) for name in arms}
            for label, df in [("market", market),
                              ("market_window_0707", window),
                              *[(f"all_{s}", f) for s, f in frames.items()]]
        },
        "top_bucket": {
            label: {name: top_bucket(df, name) for name in arms}
            for label, df in [("market", market),
                              ("market_window_0707", window),
                              *[(f"all_{s}", f) for s, f in frames.items()]]
        },
        "blend_sweep": ({} if args.no_blend_sweep
                        else blend_sweep(args.pred_dir)),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.json_out} ({len(market)} market games)")

    table = pd.DataFrame([
        {"rung": r["model"], "brier": round(r["brier"], 5),
         "log_loss": round(r["log_loss"], 5),
         "d_vs_served": round(r.get("paired_vs_served", {}).get("diff", 0.0), 5),
         "t": round(r.get("paired_vs_served", {}).get("t", float("nan")), 2)}
        for r in payload["market"]["scores"]])
    print(table.to_string(index=False))
    for key, value in payload["predictions"].items():
        if "passes" in value:
            print(f"{key}: {'PASS' if value['passes'] else 'fail'}")
        else:
            print(f"{key}: matches rung 3 = {value['matches_rung3']}")
    v = payload["verdict"]
    flags = ", ".join(f"{k}={v[k]}" for k in v if k != "ships")
    print(f"\nverdict: {'SHIPS' if v['ships'] else 'nothing ships'} ({flags})")


if __name__ == "__main__":
    main()
