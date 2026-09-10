"""BAS-92: score the hierarchical posterior's width against Marcel's Beta.

Reads two priced frames of the same archive — one priced with Marcel's Beta
behind the posterior columns (the served arm, BAS-70) and one with the
hierarchical model's width on K, BB and HR (`props.BayesWidth`, BAS-92) — and
writes `docs/posterior-width.md`'s five pre-registered predictions, its
vacuity check and its ship rule to `data/eval/bas92/posterior_width.json`.

Nothing here chooses a free parameter on the half it scores: the half split is
the archive's median game date, tau is chosen on the first half by fee-waived
flat ROI (`props_exam.choose_tau`), and the matched threshold is found on the
second half purely so "beats the threshold rule at a matched bet count" can be
checked at all (`props_exam.matched_threshold`, and its docstring for why that
is a comparison device and not a parameter).

Usage:
    python scripts/analyze_bas92.py \
        --beta data/eval/bas92/priced_beta.parquet \
        --bayes data/eval/bas92/priced_bayes_width.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.props_exam import (
    SD_COL, TAU_GRID, choose_tau, halves_of, matched_threshold,
    posterior_comparison, venue_for,
)
from src.market import pnl, props

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bas92")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/eval/bas92/posterior_width.json"

TICKET = "BAS-92"
PRIMARY_MODEL = "matchup"          # the served arm; --matchup on
PRIMARY_STAT = "hr"                # docs/posterior-width.md's primary set
POOLED_STATS = ("hr", "hits", "tb")
CONTROL_STAT = "k"                 # the negative control: nothing may move
HEADLINE_THRESHOLD = 0.02          # the served 2-point rule

# docs/posterior-width.md's pre-registered thresholds.
P1_SPEARMAN_MAX = 0.9
P1_CHANGED_SHARE_MIN = 0.25
P1_CHANGE_SIZE = 0.30
P2_GAIN_PRIMARY = 0.010            # points of fee-waived ROI, HR
P2_GAIN_POOLED = 0.005
P3_TERCILE_GAP = 0.020
P4_BRIER_TOL = 0.0001


# ───────────────────────────── plumbing ─────────────────────────────

def align(beta: pd.DataFrame, bayes: pd.DataFrame) -> tuple:
    """The two priced frames on exactly the same contracts, in the same order.

    A contract that one arm priced and the other did not would make every
    paired number below a comparison of two different sets; there should be no
    such row (the arm changes a Beta's width, not which rows are priceable),
    and this is where that is checked rather than assumed.
    """
    key = ["game_pk", "player_id", "prop_stat", "prop_line"]
    a = beta.sort_values(key).reset_index(drop=True)
    b = bayes.sort_values(key).reset_index(drop=True)
    if len(a) != len(b) or not (a[key].to_numpy() == b[key].to_numpy()).all():
        common = (a[key].merge(b[key], on=key, how="inner"))
        logger.warning("priced frames differ: %d vs %d rows, %d in common",
                       len(a), len(b), len(common))
        a = a.merge(common, on=key, how="inner").sort_values(key).reset_index(drop=True)
        b = b.merge(common, on=key, how="inner").sort_values(key).reset_index(drop=True)
    return a, b


def pnl_frames(beta: pd.DataFrame, bayes: pd.DataFrame) -> tuple:
    """`props.to_pnl_frame` on both, on the settled contracts both carry."""
    return props.to_pnl_frame(beta), props.to_pnl_frame(bayes)


def roi_row(frame: pd.DataFrame, label: str, model: str, fee_waived, as_quoted,
            draws: int, seed: int, **kwargs) -> dict:
    fw = pnl.evaluate(frame, model, fee_waived, staking="flat",
                      group_col="game_pk", draws=draws, seed=seed, **kwargs)
    aq = pnl.evaluate(frame, model, as_quoted, staking="flat",
                      group_col="game_pk", draws=draws, seed=seed, **kwargs)
    return {"rule": label, "n_bets": fw["n_bets"], "hit_rate": fw["hit_rate"],
            "roi_fee_waived": fw["roi"], "roi_fee_waived_lo": fw["roi_lo"],
            "roi_fee_waived_hi": fw["roi_hi"],
            "roi_as_quoted": aq["roi"], "roi_as_quoted_lo": aq["roi_lo"],
            "roi_as_quoted_hi": aq["roi_hi"]}


def p_edge_positive(frame: pd.DataFrame, model: str, sd_col: str,
                    venue: pnl.Venue) -> np.ndarray:
    """`P(edge > 0)` for the side the threshold rule would take on each row.

    The same Normal approximation `pnl.decide_posterior` uses, read out as a
    number instead of a decision: `P(p_true > ask)` on the side we would buy
    YES, `P(p_true < bid)` on the side we would buy NO, and the larger of the
    two elsewhere so a row that the threshold rule took has a probability
    attached whichever way it leans.
    """
    bid, ask = pnl.quotes(frame, venue)
    p = frame[model].to_numpy(dtype=float)
    sd = frame[sd_col].to_numpy(dtype=float)
    degenerate = ~(sd > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        above = norm.sf((ask - p) / sd)
        below = norm.cdf((bid - p) / sd)
    above = np.where(degenerate, (p > ask).astype(float), above)
    below = np.where(degenerate, (p < bid).astype(float), below)
    return np.maximum(above, below)


# ───────────────────── prediction 1: vacuity ─────────────────────

def prediction_1(beta: pd.DataFrame, bayes: pd.DataFrame) -> dict:
    """Is the Bayes width different information from the Beta's?

    Run first, and named "vacuity" in the pre-registration: below either
    threshold the single-component posterior's width is as uniform as the
    Beta's and nothing after this can resolve anything.
    """
    out = {"threshold_spearman_max": P1_SPEARMAN_MAX,
           "threshold_changed_share_min": P1_CHANGED_SHARE_MIN,
           "change_size": P1_CHANGE_SIZE, "by_stat": {}}
    for stat in sorted(set(beta["prop_stat"])):
        m = beta["prop_stat"] == stat
        a = beta.loc[m, SD_COL].to_numpy(dtype=float)
        b = bayes.loc[m, SD_COL].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        rho = float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() > 2 else float("nan")
        # A contract whose Beta sd is exactly zero has no player-specific
        # rates behind it at all (`price`'s league fallback), so there is no
        # relative change to speak of; those rows are counted as unchanged
        # rather than as an infinite change.
        pos = a > 0
        rel = np.zeros_like(a)
        ratio = np.zeros_like(a)
        np.divide(np.abs(b - a), a, out=rel, where=pos)
        np.divide(b, a, out=ratio, where=pos)
        changed = float(np.mean(np.where(pos, rel, 0.0) > P1_CHANGE_SIZE))
        out["by_stat"][stat] = {
            "n": int(m.sum()), "spearman": rho, "changed_share": changed,
            "median_sd_beta": float(np.median(a)),
            "median_sd_bayes": float(np.median(b)),
            "mean_sd_beta": float(np.mean(a)), "mean_sd_bayes": float(np.mean(b)),
            "n_zero_sd_beta": int((~pos).sum()),
            "sd_ratio_median": (float(np.median(ratio[pos])) if pos.any()
                                else float("nan")),
            "p05_sd_bayes": float(np.percentile(b, 5)),
            "p95_sd_bayes": float(np.percentile(b, 95)),
            "p05_sd_beta": float(np.percentile(a, 5)),
            "p95_sd_beta": float(np.percentile(a, 95)),
        }
    primary = out["by_stat"][PRIMARY_STAT]
    out["primary"] = {"stat": PRIMARY_STAT, **primary}
    out["passes"] = bool(primary["spearman"] < P1_SPEARMAN_MAX
                         and primary["changed_share"] >= P1_CHANGED_SHARE_MIN)
    return out


# ───────────────── prediction 2: the selection rule ─────────────────

def selection_block(frame_beta: pd.DataFrame, frame_bayes: pd.DataFrame,
                    tau_beta: float, tau_bayes: float, fee_waived, as_quoted,
                    draws: int, seed: int) -> dict:
    """`posterior_comparison` under both arms on the same second-half frame."""
    beta = posterior_comparison(frame_beta.reset_index(drop=True), PRIMARY_MODEL,
                                SD_COL, tau_beta, HEADLINE_THRESHOLD,
                                fee_waived, as_quoted, draws, seed)
    bayes = posterior_comparison(frame_bayes.reset_index(drop=True), PRIMARY_MODEL,
                                 SD_COL, tau_bayes, HEADLINE_THRESHOLD,
                                 fee_waived, as_quoted, draws, seed)
    # Row 1 is the posterior rule, row 2 the threshold rule forced to the same
    # bet count — the pre-registration's "at a matched bet count".
    gain = float(bayes.iloc[1]["roi_fee_waived"] - bayes.iloc[2]["roi_fee_waived"])
    gain_beta = float(beta.iloc[1]["roi_fee_waived"] - beta.iloc[2]["roi_fee_waived"])
    return {
        "beta": json.loads(beta.to_json(orient="records")),
        "bayes_width": json.loads(bayes.to_json(orient="records")),
        "tau_beta": tau_beta, "tau_bayes": tau_bayes,
        "gain_vs_matched_bayes": gain,
        "gain_vs_matched_beta": gain_beta,
        "posterior_interval_excludes_zero_bayes": bool(
            bayes.iloc[1]["roi_fee_waived_lo"] > 0),
    }


def prediction_2(second_beta, second_bayes, tau_beta, tau_bayes,
                 fee_waived, as_quoted, draws, seed) -> dict:
    out = {"threshold_primary": P2_GAIN_PRIMARY, "threshold_pooled": P2_GAIN_POOLED}
    sets = {
        PRIMARY_STAT: (second_beta[second_beta["prop_stat"] == PRIMARY_STAT],
                       second_bayes[second_bayes["prop_stat"] == PRIMARY_STAT]),
        "pooled_hr_hits_tb": (
            second_beta[second_beta["prop_stat"].isin(POOLED_STATS)],
            second_bayes[second_bayes["prop_stat"].isin(POOLED_STATS)]),
        "all_stats": (second_beta, second_bayes),
    }
    for name, (b, w) in sets.items():
        out[name] = selection_block(b, w, tau_beta, tau_bayes, fee_waived,
                                    as_quoted, draws, seed)
    out["passes_primary"] = bool(
        out[PRIMARY_STAT]["gain_vs_matched_bayes"] >= P2_GAIN_PRIMARY)
    out["passes_pooled"] = bool(
        out["pooled_hr_hits_tb"]["gain_vs_matched_bayes"] >= P2_GAIN_POOLED)
    out["passes"] = bool(out["passes_primary"] and out["passes_pooled"])
    return out


# ─────────────── prediction 3: the winner's curse ───────────────

def tercile_table(frame: pd.DataFrame, fee_waived, as_quoted, draws, seed) -> dict:
    """Fee-waived flat return by tercile of `P(edge > 0)`, on the contracts the
    2-point threshold rule bets.

    The bets are held fixed across terciles — every row here is one the served
    rule would take — so this asks only whether the posterior can *rank* them,
    which is exactly the winner's-curse claim.
    """
    bid, ask = pnl.quotes(frame, as_quoted)
    d = pnl.decide(frame[PRIMARY_MODEL].to_numpy(dtype=float), bid, ask,
                   HEADLINE_THRESHOLD)
    take = (d["side"] != pnl.NO_BET).to_numpy()
    sub = frame.loc[take].copy()
    if len(sub) < 30:
        return {"n": int(len(sub)), "terciles": [], "top_minus_bottom": None}
    sub["p_edge"] = p_edge_positive(sub, PRIMARY_MODEL, SD_COL, as_quoted)
    try:
        sub["tercile"] = pd.qcut(sub["p_edge"], 3, labels=["bottom", "mid", "top"],
                                 duplicates="drop")
    except ValueError:
        return {"n": int(len(sub)), "terciles": [],
                "top_minus_bottom": None,
                "note": "P(edge>0) has too few distinct values to tercile"}
    rows = []
    for label in ("bottom", "mid", "top"):
        grp = sub[sub["tercile"] == label]
        if grp.empty:
            continue
        row = roi_row(grp.reset_index(drop=True), label, PRIMARY_MODEL,
                      fee_waived, as_quoted, draws, seed,
                      rule="threshold", threshold=HEADLINE_THRESHOLD)
        row["p_edge_lo"] = float(grp["p_edge"].min())
        row["p_edge_hi"] = float(grp["p_edge"].max())
        rows.append(row)
    by = {r["rule"]: r for r in rows}
    gap = (float(by["top"]["roi_fee_waived"] - by["bottom"]["roi_fee_waived"])
           if "top" in by and "bottom" in by else None)
    return {"n": int(len(sub)), "terciles": rows, "top_minus_bottom": gap}


def prediction_3(second_beta, second_bayes, fee_waived, as_quoted,
                 draws, seed) -> dict:
    out = {"threshold": P3_TERCILE_GAP, "edge_threshold": HEADLINE_THRESHOLD}
    sets = {
        PRIMARY_STAT: (second_beta[second_beta["prop_stat"] == PRIMARY_STAT],
                       second_bayes[second_bayes["prop_stat"] == PRIMARY_STAT]),
        "pooled_hr_hits_tb": (
            second_beta[second_beta["prop_stat"].isin(POOLED_STATS)],
            second_bayes[second_bayes["prop_stat"].isin(POOLED_STATS)]),
    }
    for name, (b, w) in sets.items():
        out[name] = {
            "beta": tercile_table(b.reset_index(drop=True), fee_waived,
                                  as_quoted, draws, seed),
            "bayes_width": tercile_table(w.reset_index(drop=True), fee_waived,
                                         as_quoted, draws, seed),
        }
    gap_hr = out[PRIMARY_STAT]["bayes_width"]["top_minus_bottom"]
    gap_pooled = out["pooled_hr_hits_tb"]["bayes_width"]["top_minus_bottom"]
    out["passes_primary"] = bool(gap_hr is not None and gap_hr >= P3_TERCILE_GAP)
    out["passes_pooled"] = bool(gap_pooled is not None
                                and gap_pooled >= P3_TERCILE_GAP)
    out["passes"] = bool(out["passes_primary"] and out["passes_pooled"])
    return out


# ─────────────── prediction 4: the marginal price is a null ───────────────

def prediction_4(beta: pd.DataFrame, bayes: pd.DataFrame) -> dict:
    """Paired Brier of `p_over_bb`, bayes_width minus Beta, clustered by game.

    Negative means the wider price is the better one. The prediction is that
    it is neither: within +/-0.0001 on every stat, because the mean carries
    the price and only the width changed.
    """
    joined = beta.copy()
    joined["p_over_bb_bayes"] = bayes["p_over_bb"].to_numpy()
    out = {"tolerance": P4_BRIER_TOL, "by_stat": {}}
    worst = 0.0
    for stat in list(sorted(set(joined["prop_stat"]))) + ["all"]:
        grp = joined if stat == "all" else joined[joined["prop_stat"] == stat]
        res = props.paired_brier(grp, "p_over_bb_bayes", "p_over_bb")
        out["by_stat"][stat] = res
        if stat != "all":
            worst = max(worst, abs(res["diff"]))
    out["worst_abs_diff"] = worst
    out["passes"] = bool(worst <= P4_BRIER_TOL)
    # The other null worth stating: against the *point* price the arm sells.
    out["vs_point_price"] = {
        stat: props.paired_brier(
            joined if stat == "all" else joined[joined["prop_stat"] == stat],
            "p_over_bb_bayes", "p_matchup")
        for stat in list(sorted(set(joined["prop_stat"]))) + ["all"]}
    return out


# ─────────────── prediction 5: the negative control ───────────────

def prediction_5(beta: pd.DataFrame, bayes: pd.DataFrame,
                 second_beta, second_bayes, tau_beta, tau_bayes,
                 fee_waived, as_quoted, draws, seed) -> dict:
    """Strikeouts are priced by the pitcher Beta, which this arm never touches.

    Any movement at all is a bug, so this compares the columns exactly rather
    than to a tolerance, and then re-runs the money table to say so in the
    currency the rest of the exam is quoted in.
    """
    m = beta["prop_stat"] == CONTROL_STAT
    a = beta.loc[m]
    b = bayes.loc[m]
    identical_bb = bool(np.array_equal(a["p_over_bb"].to_numpy(),
                                       b["p_over_bb"].to_numpy()))
    identical_sd = bool(np.array_equal(a[SD_COL].to_numpy(), b[SD_COL].to_numpy()))
    kb = second_beta[second_beta["prop_stat"] == CONTROL_STAT].reset_index(drop=True)
    kw = second_bayes[second_bayes["prop_stat"] == CONTROL_STAT].reset_index(drop=True)
    out = {
        "n": int(m.sum()),
        "p_over_bb_identical": identical_bb,
        "p_over_sd_identical": identical_sd,
        "max_abs_diff_p_over_bb": float(np.abs(
            a["p_over_bb"].to_numpy() - b["p_over_bb"].to_numpy()).max()),
        "max_abs_diff_p_over_sd": float(np.abs(
            a[SD_COL].to_numpy() - b[SD_COL].to_numpy()).max()),
        "second_half_beta": json.loads(posterior_comparison(
            kb, PRIMARY_MODEL, SD_COL, tau_beta, HEADLINE_THRESHOLD,
            fee_waived, as_quoted, draws, seed).to_json(orient="records")),
        "second_half_bayes_width": json.loads(posterior_comparison(
            kw, PRIMARY_MODEL, SD_COL, tau_bayes, HEADLINE_THRESHOLD,
            fee_waived, as_quoted, draws, seed).to_json(orient="records")),
    }
    out["passes"] = bool(identical_bb and identical_sd)
    return out


# ─────────────────────────── exploratory ───────────────────────────

def hits_after_fee(second_beta, second_bayes, tau_beta, tau_bayes,
                   fee_waived, as_quoted, draws, seed) -> dict:
    """Hits under each rule, both arms, both halves' second half.

    BAS-70 flagged hits as the first positive after-fee line on the board and
    as exploratory; docs/posterior-width.md keeps it exploratory. Reported, not
    tested.
    """
    hb = second_beta[second_beta["prop_stat"] == "hits"].reset_index(drop=True)
    hw = second_bayes[second_bayes["prop_stat"] == "hits"].reset_index(drop=True)
    return {
        "beta": json.loads(posterior_comparison(
            hb, PRIMARY_MODEL, SD_COL, tau_beta, HEADLINE_THRESHOLD,
            fee_waived, as_quoted, draws, seed).to_json(orient="records")),
        "bayes_width": json.loads(posterior_comparison(
            hw, PRIMARY_MODEL, SD_COL, tau_bayes, HEADLINE_THRESHOLD,
            fee_waived, as_quoted, draws, seed).to_json(orient="records")),
    }


def width_coverage(beta: pd.DataFrame, width_dir: Path) -> dict:
    """How many priced *contracts* actually got a hierarchical width.

    `BayesWidth.audit`'s denominator is the whole daily rate table, which
    carries every batter with a prior-season line whether or not a prop was
    ever quoted on him; that number says nothing about coverage of the thing
    being scored. This one is per contract and per batter on the archive.
    """
    if not width_dir.exists():
        return {}
    bw = props.BayesWidth.load(width_dir)
    rows = beta[beta["prop_stat"] != CONTROL_STAT].copy()
    rows["cutoff"] = [bw.cutoff_for(str(d)) for d in rows["game_date"]]
    have, unseen_flag = [], []
    long = pd.concat([pd.read_parquet(f)
                      for f in sorted(width_dir.glob("bayes_width_*.parquet"))],
                     ignore_index=True)
    seen_keys = set(zip(long["cutoff"].astype(str),
                        long["batter"].astype("int64"),
                        long["component"]))
    unseen_keys = set(zip(long.loc[long["unseen"], "cutoff"].astype(str),
                          long.loc[long["unseen"], "batter"].astype("int64"),
                          long.loc[long["unseen"], "component"]))
    for cutoff, pid in zip(rows["cutoff"], rows["player_id"]):
        keys = [(str(cutoff), int(pid), c) for c in props.BAYES_WIDTH_COMPONENTS]
        have.append(all(k in seen_keys for k in keys))
        unseen_flag.append(any(k in unseen_keys for k in keys))
    rows["has_width"] = have
    rows["from_population"] = unseen_flag
    missing = rows.loc[~rows["has_width"], "player_id"]
    return {
        "n_batter_contracts": int(len(rows)),
        "n_with_bayes_width": int(rows["has_width"].sum()),
        "n_fell_back_to_the_beta": int((~rows["has_width"]).sum()),
        "n_batters": int(rows["player_id"].nunique()),
        "n_batters_without_a_bayes_row": int(missing.nunique()),
        "batters_without_a_bayes_row": sorted(int(p) for p in missing.unique()),
        "n_from_the_fitted_population": int(rows["from_population"].sum()),
        "share_from_the_fitted_population": float(rows["from_population"].mean()),
        "by_stat": {s: {"n": int(len(g)),
                        "n_with_bayes_width": int(g["has_width"].sum())}
                    for s, g in rows.groupby("prop_stat")},
    }


def bas70_reproduction(beta: pd.DataFrame, second_beta, tau_beta,
                       fee_waived, as_quoted, draws, seed) -> dict:
    """The numbers docs/posterior-props.md's Results section publishes.

    Reproduced here from the Beta arm of this run so the BAS-92 table is
    anchored to a baseline that is demonstrably the same measurement, not
    assumed to be.
    """
    settled = beta[beta["over_hit"].notna()]
    return {
        "n_settled": int(len(settled)),
        "n_games": int(settled["game_pk"].nunique()),
        "first_date": str(settled["game_date"].min()),
        "last_date": str(settled["game_date"].max()),
        "median_p_over_sd": float(settled[SD_COL].median()),
        "median_p_over_sd_by_stat": {
            s: float(g[SD_COL].median()) for s, g in settled.groupby("prop_stat")},
        "share_below_0005": float((settled[SD_COL] < 0.005).mean()),
        "paired_brier_marginal_minus_point": props.paired_brier(
            settled, "p_over_bb", "p_matchup"),
        "paired_brier_by_stat": {
            s: props.paired_brier(g, "p_over_bb", "p_matchup")
            for s, g in settled.groupby("prop_stat")},
        "tau": tau_beta,
        "second_half_all_stats": json.loads(posterior_comparison(
            second_beta.reset_index(drop=True), PRIMARY_MODEL, SD_COL, tau_beta,
            HEADLINE_THRESHOLD, fee_waived, as_quoted, draws,
            seed).to_json(orient="records")),
    }


def score_pair(beta_path: Path, bayes_path: Path, width_dir: Path,
               draws: int, seed: int) -> dict:
    """Everything docs/posterior-width.md asks for, off one pair of priced
    frames. Called once for the scored pair and once for the cross-check under
    the other Monte Carlo stream construction."""
    beta, bayes = align(pd.read_parquet(beta_path), pd.read_parquet(bayes_path))
    logger.info("%d priced rows on both arms", len(beta))
    args = argparse.Namespace(draws=draws, seed=seed, width_dir=width_dir)

    fee_waived = venue_for(0.0, False)
    as_quoted = venue_for(pnl.KALSHI_TAKER_RATE, False)

    fb, fw = pnl_frames(beta, bayes)
    dates = sorted(fb["date"].astype(str).unique())
    first, second = halves_of(dates)
    cut = second[0]
    logger.info("half split at %s: %d / %d contracts", cut,
                int(fb["date"].astype(str).isin(first).sum()),
                int(fb["date"].astype(str).isin(second).sum()))

    first_b = fb[fb["date"].astype(str) < cut]
    first_w = fw[fw["date"].astype(str) < cut]
    second_b = fb[fb["date"].astype(str) >= cut]
    second_w = fw[fw["date"].astype(str) >= cut]

    # tau, chosen on the first half, exactly as BAS-70: one grid walk over the
    # whole pooled first half, per arm.
    tau_beta, tab_beta = choose_tau(first_b, PRIMARY_MODEL, SD_COL, cut,
                                    fee_waived, TAU_GRID)
    tau_bayes, tab_bayes = choose_tau(first_w, PRIMARY_MODEL, SD_COL, cut,
                                      fee_waived, TAU_GRID)
    logger.info("tau: beta %.2f, bayes_width %.2f", tau_beta, tau_bayes)
    # And the HR-only choice, as a sensitivity: the primary set is HR, and a
    # tau chosen on a pool three quarters of which is another stat is not
    # obviously the tau that set would pick.
    hr_first_w = first_w[first_w["prop_stat"] == PRIMARY_STAT]
    tau_bayes_hr, tab_bayes_hr = choose_tau(hr_first_w, PRIMARY_MODEL, SD_COL,
                                            cut, fee_waived, TAU_GRID)
    hr_first_b = first_b[first_b["prop_stat"] == PRIMARY_STAT]
    tau_beta_hr, _ = choose_tau(hr_first_b, PRIMARY_MODEL, SD_COL, cut,
                                fee_waived, TAU_GRID)

    p1 = prediction_1(beta, bayes)
    p2 = prediction_2(second_b, second_w, tau_beta, tau_bayes,
                      fee_waived, as_quoted, args.draws, args.seed)
    p3 = prediction_3(second_b, second_w, fee_waived, as_quoted,
                      args.draws, args.seed)
    p4 = prediction_4(beta, bayes)
    p5 = prediction_5(beta, bayes, second_b, second_w, tau_beta, tau_bayes,
                      fee_waived, as_quoted, args.draws, args.seed)

    # HR at the HR-chosen tau, reported next to the primary rather than
    # instead of it.
    p2["hr_at_hr_chosen_tau"] = selection_block(
        second_b[second_b["prop_stat"] == PRIMARY_STAT],
        second_w[second_w["prop_stat"] == PRIMARY_STAT],
        tau_beta_hr, tau_bayes_hr, fee_waived, as_quoted, args.draws, args.seed)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticket": TICKET,
        "pre_registration": "docs/posterior-width.md",
        "parent": "docs/posterior-props.md (BAS-70)",
        "arms": {
            "beta": "Marcel's Beta behind p_over_bb / p_over_sd (BAS-70, served)",
            "bayes_width": ("the same mean with the bayes_walk posterior sd on "
                            "K, BB and HR (props.BayesWidth); BABIP, ISO and "
                            "pitcher strikeouts keep the Beta"),
        },
        "archive": {
            "n_priced": int(len(beta)),
            "n_settled": int(beta["over_hit"].notna().sum()),
            "n_games": int(beta["game_pk"].nunique()),
            "first_date": str(beta["game_date"].min()),
            "last_date": str(beta["game_date"].max()),
            "half_split": cut,
            "n_first_half": int(len(first_b)),
            "n_second_half": int(len(second_b)),
            "by_stat": {s: int(n) for s, n in
                        beta["prop_stat"].value_counts().items()},
        },
        "tau": {
            "beta": tau_beta, "bayes_width": tau_bayes,
            "beta_hr_only": tau_beta_hr, "bayes_width_hr_only": tau_bayes_hr,
            "grid_beta": json.loads(tab_beta.to_json(orient="records")),
            "grid_bayes_width": json.loads(tab_bayes.to_json(orient="records")),
            "grid_bayes_width_hr_only": json.loads(
                tab_bayes_hr.to_json(orient="records")),
        },
        "width_coverage": width_coverage(beta, args.width_dir),
        "bas70_reproduction": bas70_reproduction(
            beta, second_b, tau_beta, fee_waived, as_quoted, args.draws, args.seed),
        "exploratory_hits_after_fee": hits_after_fee(
            second_b, second_w, tau_beta, tau_bayes, fee_waived, as_quoted,
            args.draws, args.seed),
        "predictions": {
            "1_vacuity_the_width_is_different_information": p1,
            "2_selection_beats_the_matched_threshold": p2,
            "3_winners_curse_the_width_ranks_the_edges": p3,
            "4_null_the_marginal_price_does_not_move": p4,
            "5_negative_control_strikeouts_unchanged": p5,
        },
    }
    # docs/posterior-width.md's ship rule, verbatim: prediction 2 holding with
    # the second-half interval excluding zero on HR *and* prediction 3 holding.
    ships = bool(p2["passes_primary"]
                 and p2[PRIMARY_STAT]["posterior_interval_excludes_zero_bayes"]
                 and p3["passes"])
    payload["verdict"] = {
        "vacuous": not p1["passes"],
        "width_is_different_information": p1["passes"],
        "selection_beats_matched_threshold": p2["passes"],
        "selection_beats_matched_threshold_hr": p2["passes_primary"],
        "hr_interval_excludes_zero": bool(
            p2[PRIMARY_STAT]["posterior_interval_excludes_zero_bayes"]),
        "width_separates_real_from_spurious_edges": p3["passes"],
        "marginal_price_is_still_a_null": p4["passes"],
        "negative_control_clean": p5["passes"],
        # The pre-registration's own failure condition: 2 and 3 failing with 1
        # passing retires P(edge>0) as a selection rule for the props ledger.
        "retire_p_edge_as_a_selection_rule": bool(
            p1["passes"] and not p2["passes"] and not p3["passes"]),
        # And the other one: 1 failing sends the thread to the joint (BAS-84)
        # and measurement (BAS-85) posteriors instead.
        "wait_for_the_joint_and_measurement_posteriors": not p1["passes"],
        "ships": ships,
    }
    return payload


def headline(payload: dict) -> dict:
    """The pass/fail skeleton, for the console and for the cross-check block."""
    return {"tau": payload["tau"]["beta"], "tau_bayes": payload["tau"]["bayes_width"],
            "predictions": {
                k: {kk: vv for kk, vv in v.items()
                    if kk.startswith("passes")
                    or kk in ("worst_abs_diff", "p_over_bb_identical",
                              "p_over_sd_identical", "max_abs_diff_p_over_bb")}
                for k, v in payload["predictions"].items()},
            "spearman_hr": payload["predictions"][
                "1_vacuity_the_width_is_different_information"]["primary"]["spearman"],
            "changed_share_hr": payload["predictions"][
                "1_vacuity_the_width_is_different_information"]["primary"]["changed_share"],
            "gain_vs_matched_hr": payload["predictions"][
                "2_selection_beats_the_matched_threshold"][PRIMARY_STAT][
                    "gain_vs_matched_bayes"],
            "verdict": payload["verdict"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beta", type=Path, required=True)
    ap.add_argument("--bayes", type=Path, required=True)
    ap.add_argument("--beta-cross-check", type=Path, default=None,
                    help="the same Beta arm priced under the other "
                         "props.DRAW_STREAMS setting")
    ap.add_argument("--bayes-cross-check", type=Path, default=None)
    ap.add_argument("--cross-check-label", default="shared_draw_streams")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fits", type=Path,
                    default=ROOT / "data/eval/bas92/bayes_width_fits.json")
    ap.add_argument("--checkpoint-check", type=Path,
                    default=ROOT / "data/eval/bas92/bayes_width_checkpoint_check.json")
    ap.add_argument("--neutral-park-check", type=Path,
                    default=ROOT / "data/eval/bas92/neutral_park"
                                   "/bayes_width_checkpoint_check.json")
    ap.add_argument("--neutral-park-fits", type=Path,
                    default=ROOT / "data/eval/bas92/neutral_park"
                                   "/bayes_width_fits.json")
    ap.add_argument("--width-audit", type=Path,
                    default=ROOT / "data/eval/bas92/width_audit.json")
    ap.add_argument("--width-dir", type=Path, default=ROOT / "data/eval/bas92")
    ap.add_argument("--draws", type=int, default=pnl.BOOTSTRAP_DRAWS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    payload = score_pair(args.beta, args.bayes, args.width_dir,
                         args.draws, args.seed)
    if args.beta_cross_check and args.bayes_cross_check:
        other = score_pair(args.beta_cross_check, args.bayes_cross_check,
                           args.width_dir, args.draws, args.seed)
        payload[args.cross_check_label] = headline(other)
        payload[args.cross_check_label]["full"] = other["predictions"]
    for name, path in (("fits", args.fits),
                       ("checkpoint_check", args.checkpoint_check),
                       ("neutral_park_checkpoint_check", args.neutral_park_check),
                       ("neutral_park_fits", args.neutral_park_fits),
                       ("width_audit", args.width_audit)):
        if path and path.exists():
            payload[name] = json.loads(path.read_text())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"-> {args.out}")
    print(json.dumps(headline(payload), indent=1))


if __name__ == "__main__":
    main()
