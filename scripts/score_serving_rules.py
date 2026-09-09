"""Score three candidate covariate-share serving rules against the one in force (BAS-82).

architecture.md §3's rule since BAS-80 is a hard threshold on one number: a
layer-1 measurement is served only if its *covariate-only share* — the arm
against the same shape with the covariate removed (the recalibration control) —
clears a clustered |t| > 2.5. Two decisions already sit on the knife edge:
pitcher BB/BF's covariate-only |t| is 2.41 against the current pitcher Marcel
and 2.58 against a calibrated one half a percent of MAE away, and pitcher K/BF
misses at 2.499. A gate that flips on ±0.1 of a t is reporting sample size.

This script replays every serving decision made so far under the rule in force
and under three candidates, and writes the decision matrix to
`data/eval/serving_rules.json`.

    current            |t| > 2.5 on the covariate share against the baseline
                       the decision was made on.
    two_baseline       clears only if |t| > 2.5 on *both* the current and the
                       best-known-calibrated baseline; withholds only if under
                       2.5 on both; otherwise the current rule's verdict stands.
    effect_floor       the covariate share is at least `EFFECT_FLOOR_PCT` of the
                       baseline's own MAE *and* |t| > 2.0.
    bootstrap          a seeded cluster bootstrap over players of the covariate
                       share's t; clears at P(|t| > 2.5) >= 0.8, withholds at
                       <= 0.2, otherwise the current rule's verdict stands.

**Where the numbers come from, and the one honest caveat.** The bootstrap wants
per-cell paired differences. The committed evidence
(`data/eval/pitching_stuff_serving.json`, `pitching_stuff_recalibrated.json`,
`data/models/contact_quality_*.json`) carries only the pooled summary of each
paired comparison — `diff`, the clustered `se`, `n` cells and `n_clusters`
players — and the cells themselves cannot be rebuilt here because the PA-outcome
parquets they are scored on are not in the working tree. So every decision below
is bootstrapped on a **reconstructed cluster panel**: a seeded Gaussian panel of
`n_clusters` player contributions, affinely rescaled so that its weighted mean
and its cluster-robust standard error reproduce the committed `diff` and `se`
exactly. Every row records `bootstrap_source` saying so. That reconstruction
assumes the cluster contributions are Gaussian, which is exactly the assumption
the clustered t already makes — so the bootstrap here cannot disagree with the t
by more than Monte Carlo noise, and the P it reports is, to three digits, the
closed-form `phi(|t| - 2.5) + phi(-|t| - 2.5)` also written into each row as
`p_normal`. Run this again with a real per-cell table (`cells_from_frame`) once
the harness can be re-run and the bootstrap becomes an independent check rather
than a restatement.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# The bar architecture.md §3 writes down, and the one every candidate is
# measured against.
T_BAR = 2.5
# Candidate 2: the covariate share must be worth at least this much of the
# baseline's own MAE, and clear a looser t.
EFFECT_FLOOR_PCT = 0.75
EFFECT_FLOOR_T = 2.0
# Candidate 3.
N_BOOT = 2000
BOOT_SEED = 20260909
BOOT_CLEAR_P = 0.8
BOOT_WITHHOLD_P = 0.2

CLEAR, WITHHOLD = "clear", "withhold"


# --- the bootstrap -----------------------------------------------------------

def cells_from_frame(frame: pd.DataFrame, cluster_col: str = "player",
                     weight_col: str = "trials", diff_col: str = "d"
                     ) -> pd.DataFrame:
    """Normalize a per-cell paired-difference table to (cluster, weight, d).

    `d` is one cell's paired difference in absolute error, arm minus control,
    the same quantity `src.eval.tuning.paired_abs_error_diff` averages; the
    weight is the cell's trials and the cluster is the player, because one
    player appears at three cutoffs of five seasons and those rows are not
    fifteen independent observations.
    """
    out = pd.DataFrame({
        "cluster": frame[cluster_col].to_numpy(),
        "weight": frame[weight_col].to_numpy(dtype="float64"),
        "d": frame[diff_col].to_numpy(dtype="float64"),
    })
    if (out["weight"] <= 0).any():
        raise ValueError("cell weights must be positive")
    return out


def clustered_stats(cells: pd.DataFrame) -> dict:
    """Trials-weighted mean difference and its cluster-robust SE and t.

    The same estimator as `src.eval.tuning.paired_abs_error_diff`: sum the
    weighted residuals within a cluster, then take their spread across clusters.
    """
    w = cells["weight"].to_numpy(dtype="float64")
    d = cells["d"].to_numpy(dtype="float64")
    total = float(w.sum())
    mean = float((w * d).sum() / total)
    resid = w * (d - mean)
    groups = pd.Series(resid).groupby(cells["cluster"].to_numpy()).sum().to_numpy()
    se = float(math.sqrt(float((groups ** 2).sum())) / total)
    return {"diff": mean, "se": se, "n": int(len(cells)),
            "n_clusters": int(len(groups)),
            "t": mean / se if se > 0 else float("nan")}


def cluster_bootstrap_p(cells: pd.DataFrame, t_bar: float = T_BAR,
                        n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """P(|t| > t_bar) under a cluster bootstrap over players.

    Players are resampled with replacement (whole players, all of their cells
    together, which is what makes it a *cluster* bootstrap); the weighted mean
    and the clustered SE are recomputed on each resample and the t is left
    uncentred, because the question the gate asks is "what t would a rerun of
    this measurement produce", not "is the true effect zero".
    """
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(cells["cluster"].to_numpy())
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    w = cells["weight"].to_numpy(dtype="float64")[order]
    d = cells["d"].to_numpy(dtype="float64")[order]
    starts = np.searchsorted(codes_sorted, np.arange(codes_sorted[-1] + 1), "left")
    ends = np.searchsorted(codes_sorted, np.arange(codes_sorted[-1] + 1), "right")
    n_clusters = len(starts)
    # Per-cluster sums are all the mean needs; the SE needs the cells, so keep
    # both and index by cluster.
    sw = np.array([w[a:b].sum() for a, b in zip(starts, ends)])
    swd = np.array([(w[a:b] * d[a:b]).sum() for a, b in zip(starts, ends)])
    swd2 = np.array([(w[a:b] * d[a:b] ** 2).sum() for a, b in zip(starts, ends)])
    # sum over a cluster of w*(d - m) = swd - m*sw, so the clustered SE of a
    # resample is a closed form in these three sums.
    ts = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_clusters, n_clusters)
        total = sw[pick].sum()
        mean = swd[pick].sum() / total
        g = swd[pick] - mean * sw[pick]
        se = math.sqrt(float((g ** 2).sum())) / total
        ts[i] = mean / se if se > 0 else np.nan
    finite = ts[np.isfinite(ts)]
    return {
        "p_gt_bar": float(np.mean(np.abs(finite) > t_bar)) if len(finite) else float("nan"),
        "t_p05": float(np.percentile(finite, 5)) if len(finite) else float("nan"),
        "t_p95": float(np.percentile(finite, 95)) if len(finite) else float("nan"),
        "n_boot": int(n_boot), "seed": int(seed),
    }


def reconstruct_cells(diff: float, se: float, n: int, n_clusters: int,
                      seed: int) -> pd.DataFrame:
    """A cluster panel matching a committed summary's `diff` and `se` exactly.

    Used only where the per-cell differences are not reproducible in this tree.
    Cells are dealt out to clusters as evenly as the counts allow, given equal
    weight, drawn Gaussian and then affinely rescaled so that `clustered_stats`
    on the result returns the committed `diff` and `se`. See the module
    docstring: this makes the bootstrap a restatement of the t rather than an
    independent check, and every row it produces says so.
    """
    rng = np.random.default_rng(seed)
    cluster = np.repeat(np.arange(n_clusters), n // n_clusters)
    if len(cluster) < n:
        cluster = np.concatenate([cluster, rng.integers(0, n_clusters, n - len(cluster))])
    d = rng.standard_normal(n)
    cells = pd.DataFrame({"cluster": cluster, "weight": np.ones(n), "d": d})
    got = clustered_stats(cells)
    if not (got["se"] > 0):
        raise ValueError("degenerate reconstruction")
    cells["d"] = diff + (cells["d"] - got["diff"]) * (se / got["se"])
    return cells


def p_normal(t: float, t_bar: float = T_BAR) -> float:
    """The closed form the Gaussian reconstruction is bound to reproduce."""
    def phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return phi(abs(t) - t_bar) + phi(-abs(t) - t_bar)


# --- the rules ---------------------------------------------------------------

def verdict_current(share: dict) -> str:
    """architecture.md §3 as written: |t| > 2.5 on the covariate share."""
    return CLEAR if abs(share["t"]) > T_BAR else WITHHOLD


def verdict_two_baseline(share: dict, share_calibrated: dict | None,
                         status_quo: str) -> str:
    if share_calibrated is None:
        return status_quo
    a, b = abs(share["t"]), abs(share_calibrated["t"])
    if a > T_BAR and b > T_BAR:
        return CLEAR
    if a < T_BAR and b < T_BAR:
        return WITHHOLD
    return status_quo


def verdict_effect_floor(share: dict) -> str:
    big = abs(share["pct"]) >= EFFECT_FLOOR_PCT
    sig = abs(share["t"]) > EFFECT_FLOOR_T
    return CLEAR if (big and sig) else WITHHOLD


def verdict_bootstrap(p: float, status_quo: str) -> str:
    if p >= BOOT_CLEAR_P:
        return CLEAR
    if p <= BOOT_WITHHOLD_P:
        return WITHHOLD
    return status_quo


def score_decision(decision: dict, seed: int, n_boot: int = N_BOOT) -> dict:
    """Apply the rule in force and the three candidates to one decision."""
    share = decision["covariate_share"]
    share_cal = decision.get("covariate_share_calibrated")
    cells = decision.get("cells")
    if cells is None:
        cells = reconstruct_cells(share["diff"], share["se"], share["n"],
                                  share["n_clusters"], seed)
        source = "summary_reconstruction"
    else:
        cells = cells_from_frame(cells)
        source = "per_cell"
    boot = cluster_bootstrap_p(cells, n_boot=n_boot, seed=seed)
    current = verdict_current(share)
    row = {
        "decision": decision["decision"],
        "ticket": decision["ticket"],
        "side": decision["side"],
        "component": decision["component"],
        "served_today": decision["served_today"],
        "baseline": decision["baseline"],
        "control": decision["control"],
        # The total gain against the baseline, for context only: a component is
        # served on the *conjunction* of the gate (the arm beats the baseline)
        # and the covariate-share condition every rule here is about.
        "total_pct": decision["total_pct"],
        "total_t": decision["total_t"],
        "covariate_diff": share["diff"],
        "covariate_pct_of_baseline_mae": share["pct"],
        "covariate_t": share["t"],
        "covariate_t_calibrated": share_cal["t"] if share_cal else None,
        "n": share["n"], "n_clusters": share["n_clusters"],
        "bootstrap_p_gt_2_5": boot["p_gt_bar"],
        "bootstrap_t_p05": boot["t_p05"],
        "bootstrap_t_p95": boot["t_p95"],
        "bootstrap_source": source,
        "p_normal": p_normal(share["t"]),
        "verdict_current": current,
        "verdict_two_baseline": verdict_two_baseline(share, share_cal, current),
        "verdict_effect_floor": verdict_effect_floor(share),
        "verdict_bootstrap": verdict_bootstrap(boot["p_gt_bar"], current),
    }
    row["changes"] = sorted(
        rule for rule in ("two_baseline", "effect_floor", "bootstrap")
        if row[f"verdict_{rule}"] != current)
    return row


# --- reading the committed evidence -----------------------------------------

def _paired(rows: list[dict], component: str, arm: str, base: str) -> dict | None:
    for r in rows:
        if (r.get("component") == component and r.get("arm") == arm
                and r.get("base") == base and r.get("scope", "all") == "all"):
            return r
    return None


def _share(row: dict, baseline_mae: float) -> dict:
    """A covariate-share summary, with the effect restated as a percent of the
    *served baseline's* own MAE so the floor in candidate 2 means the same thing
    on every row. The committed `pct` is against the control's MAE instead."""
    return {"diff": row["diff"], "se": row["se"], "t": row["t"],
            "n": row["n"], "n_clusters": row["n_clusters"],
            "pct": 100.0 * row["diff"] / baseline_mae}


def load_decisions(root: Path = ROOT) -> list[dict]:
    """Every serving decision made so far, with its covariate-only share.

    Contact quality (BAS-72) is scored on `contact` vs `contact_recal`: the
    committed payloads carry no `contact_additive_recal` arm, so the covariate
    share of the *served* additive shape was never measured there and the free
    fit's share is the only evidence that exists. Pitcher stuff (BAS-79) is
    scored on the shape that ships, `stuff_additive` vs `stuff_additive_recal`,
    on both baselines from BAS-80.
    """
    decisions: list[dict] = []

    served_contact = {"hitter": True, "pitcher": False}
    for side in ("hitter", "pitcher"):
        payload = json.loads(
            (root / f"data/models/contact_quality_{side}.json").read_text())
        for component in payload["components"]:
            share_row = _paired(payload["paired"], component, "contact",
                                "contact_recal")
            total = _paired(payload["paired"], component, "contact_additive",
                            "marcel_tuned")
            if share_row is None or total is None:
                continue
            decisions.append({
                "decision": f"contact/{side}/{component}",
                "ticket": "BAS-72", "side": side, "component": component,
                "served_today": served_contact[side],
                "baseline": "marcel_tuned" if side == "hitter"
                            else "marcel_pitcher_tuned",
                "control": "contact_recal (free fit; no additive control committed)",
                "covariate_share": _share(share_row, total["base_mae"]),
                "covariate_share_calibrated": None,
                "total_t": total["t"], "total_pct": total["pct"],
            })

    serving = json.loads(
        (root / "data" / "eval" / "pitching_stuff_serving.json").read_text())
    recal = json.loads(
        (root / "data" / "eval" / "pitching_stuff_recalibrated.json").read_text())
    served_stuff = {"p_k_rate": False, "p_bb_rate": True,
                    "p_bbhbp_rate": False, "p_hr_rate": True}
    for component in serving["components"]:
        share_row = _paired(serving["paired"], component, "stuff_additive",
                            "stuff_additive_recal")
        total = _paired(serving["paired"], component, "stuff_additive",
                        "marcel_pitcher_tuned")
        cal_row = _paired(recal["paired"], component, "stuff_additive",
                          "stuff_additive_recal")
        cal_total = _paired(recal["paired"], component, "stuff_additive",
                            "marcel_pitcher_tuned")
        if share_row is None or total is None:
            continue
        decisions.append({
            "decision": f"stuff/pitcher/{component}",
            "ticket": "BAS-79", "side": "pitcher", "component": component,
            "served_today": served_stuff.get(component, False),
            "baseline": "marcel_pitcher_tuned",
            "control": "stuff_additive_recal",
            "covariate_share": _share(share_row, total["base_mae"]),
            "covariate_share_calibrated": (
                _share(cal_row, cal_total["base_mae"])
                if cal_row and cal_total else None),
            "total_t": total["t"], "total_pct": total["pct"],
        })

    return decisions


def missing_decisions(root: Path = ROOT) -> list[dict]:
    """Serving decisions the ticket asks for that have no committed evidence."""
    out = []
    if not (root / "data" / "eval" / "swing_decisions_stage2.json").exists():
        out.append({
            "decision": "swing_decisions/hitter/*",
            "ticket": "BAS-81",
            "reason": "BAS-81 has not been run: docs/swing-decisions.md is a "
                      "pre-registration with no Results section, there is no "
                      "swing-decision stage-2 payload, no feature artifact and "
                      "no src/eval/swing.py. Nothing to score.",
        })
    return out


# --- output ------------------------------------------------------------------

MATRIX_COLS = ["decision", "served_today", "total_t",
               "covariate_pct_of_baseline_mae",
               "covariate_t", "covariate_t_calibrated", "bootstrap_p_gt_2_5",
               "p_normal",
               "verdict_current", "verdict_two_baseline",
               "verdict_effect_floor", "verdict_bootstrap"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--json-out", type=Path,
                    default=ROOT / "data" / "eval" / "serving_rules.json")
    args = ap.parse_args()

    decisions = load_decisions()
    rows = [score_decision(d, seed=args.seed + i, n_boot=args.n_boot)
            for i, d in enumerate(decisions)]
    matrix = pd.DataFrame(rows)

    print("\n=== covariate-share serving rules: the decision matrix ===")
    print(matrix[MATRIX_COLS].round(4).to_string(index=False))

    print("\n--- what each candidate would change ---")
    for rule in ("two_baseline", "effect_floor", "bootstrap"):
        changed = matrix[matrix[f"verdict_{rule}"] != matrix["verdict_current"]]
        if changed.empty:
            print(f"  {rule}: nothing")
            continue
        for _, r in changed.iterrows():
            print(f"  {rule}: {r['decision']} "
                  f"{r['verdict_current']} -> {r[f'verdict_{rule}']}")

    for m in missing_decisions():
        print(f"\n--- not scoreable: {m['decision']} ({m['ticket']}) ---")
        print(f"  {m['reason']}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticket": "BAS-82",
        "rules": {
            "current": {"t_bar": T_BAR},
            "two_baseline": {"t_bar": T_BAR,
                             "note": "clears only if |t| > bar on both "
                                     "baselines, withholds only if under on "
                                     "both, else the current rule stands"},
            "effect_floor": {"min_pct_of_baseline_mae": EFFECT_FLOOR_PCT,
                             "min_abs_t": EFFECT_FLOOR_T},
            "bootstrap": {"t_bar": T_BAR, "n_boot": args.n_boot,
                          "seed": args.seed, "clear_at": BOOT_CLEAR_P,
                          "withhold_at": BOOT_WITHHOLD_P,
                          "note": "cluster bootstrap over players of the "
                                  "covariate share's t"},
        },
        "bootstrap_caveat":
            "Per-cell paired differences are not reproducible in this tree "
            "(the PA-outcome parquets the cells are scored on are absent), so "
            "rows marked bootstrap_source=summary_reconstruction bootstrap a "
            "seeded Gaussian cluster panel rescaled to the committed diff and "
            "se. That panel makes the same distributional assumption the "
            "clustered t already makes, so the bootstrap P cannot disagree "
            "with p_normal by more than Monte Carlo noise.",
        "decisions": json.loads(matrix.to_json(orient="records")),
        "not_scoreable": missing_decisions(),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
