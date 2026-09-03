"""Score the parameter-uncertainty arm against the served chain (station G).

`scripts/run_team_backtest.py --stage score` prints the station G table for
every arm. This script asks the three questions
[docs/parameter-uncertainty.md](../docs/parameter-uncertainty.md) pre-registered
and that the generic table cannot answer:

1. **Does the width help, and where?** Paired per club per date against the
   served `chain`, clustered by season, on all four probabilities and on
   projected wins — pooled and in each fifth-of-a-season bucket.
2. **Does the late-July crossover move?** The same paired difference against
   `record_500`, the .500 extrapolation the crossover is defined against,
   bucket by bucket, for `chain` and for each uncertainty arm.
3. **Is it information or is it shrinkage?** Three tuned-shrinkage controls
   built from the *served* chain's own probabilities:

       shrink_half     p' = λp + (1−λ)·0.5,  λ fitted to minimise Brier on
                       the same rows being scored — an oracle.
       shrink_base     the same toward the outcome's base rate, also fitted
                       in-sample. Stronger than shrinking toward 0.5.
       shrink_wf       the same, with λ fitted walk-forward on the seasons
                       strictly before the one being scored — the version a
                       real model would be allowed to use.

   A λ is fitted per probability column, in closed form (the Brier objective
   is a quadratic in λ). If the uncertainty arm does not beat `shrink_half`
   and `shrink_base`, it is adding calibration rather than information, and
   the doc has to say so in those words.

Usage:
    python scripts/analyse_parameter_uncertainty.py [--markdown] [--json-out P]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts.run_team_backtest import (
    DEFAULT_SEASONS, OUT_DIR, load_outcomes, load_projections,
)
from src.eval import team_backtest as tb
from src.eval.team_season import OUTCOME_OF, PROB_COLUMNS

PU_ARM = "chain_pu"
BASE_ARM = "chain"
CONTROL_ARM = "record_500"
SHRINK_ARMS = ("shrink_half", "shrink_base", "shrink_wf")


# ─── the tuned-shrinkage controls ───

def best_lambda(p: np.ndarray, y: np.ndarray, centre: float) -> float:
    """The λ that minimises mean Brier for `p' = λp + (1−λ)·centre`.

    Brier(λ) is a quadratic in λ, so this is closed form rather than a search:
    with `d = p − centre` and `e = y − centre`, the minimiser is
    `Σ d·e / Σ d²`. Unclipped on purpose — a λ above 1 would mean the
    probabilities were *under*-confident and the control is entitled to say so.
    """
    d = np.asarray(p, dtype=float) - centre
    e = np.asarray(y, dtype=float) - centre
    denom = float(np.sum(d * d))
    return float(np.sum(d * e) / denom) if denom > 0 else 1.0


def shrink_frame(scored: pd.DataFrame, arm: str, name: str,
                 mode: str) -> tuple[pd.DataFrame, dict]:
    """A synthetic arm: `arm`'s probabilities, shrunk toward a centre.

    `mode` is `"half"` (centre 0.5, λ in-sample), `"base"` (centre = the
    outcome's base rate, λ in-sample) or `"wf"` (centre 0.5, λ fitted on the
    seasons strictly before the one being scored). Everything else about the
    row — the projected wins, the club, the date — is the served chain's.
    """
    src = scored[scored["arm"] == arm].copy()
    src["arm"] = name
    lam_used: dict[str, float | dict] = {}
    for prob in PROB_COLUMNS:
        truth = OUTCOME_OF[prob]
        y = src[truth].to_numpy(dtype=float)
        p = src[prob].to_numpy(dtype=float)
        centre = 0.5 if mode in ("half", "wf") else float(y.mean())
        if mode == "wf":
            out = np.empty_like(p)
            per_season: dict[str, float] = {}
            for season in sorted(src["season"].unique()):
                mask = src["season"].to_numpy() == season
                prior = src["season"].to_numpy() < season
                # The first season has no prior seasons to fit on, so it is
                # served unshrunk. That is what walk-forward means and it is
                # a cost the control is entitled to pay.
                lam = best_lambda(p[prior], y[prior], centre) if prior.any() else 1.0
                out[mask] = lam * p[mask] + (1 - lam) * centre
                per_season[str(season)] = lam
            src[prob] = out
            lam_used[prob] = per_season
        else:
            lam = best_lambda(p, y, centre)
            src[prob] = lam * p + (1 - lam) * centre
            lam_used[prob] = {"lambda": lam, "centre": centre}
    return _rescore(src), lam_used


def _rescore(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute the per-row losses after the probabilities were changed."""
    out = frame.copy()
    for prob, truth in OUTCOME_OF.items():
        out[f"brier_{prob}"] = (out[prob] - out[truth]) ** 2
        p = out[prob].clip(lower=out["prob_floor"]).clip(
            upper=1 - out["prob_floor"])
        out[f"logloss_{prob}"] = -(out[truth] * np.log(p)
                                   + (1 - out[truth]) * np.log(1 - p))
    return out


def with_controls(scored: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frames, lams = [scored], {}
    for name, mode in (("shrink_half", "half"), ("shrink_base", "base"),
                       ("shrink_wf", "wf")):
        f, lam = shrink_frame(scored, BASE_ARM, name, mode)
        frames.append(f)
        lams[name] = lam
    return pd.concat(frames, ignore_index=True), lams


# ─── tables ───

def headline(scored: pd.DataFrame, arms) -> pd.DataFrame:
    rows = []
    for arm in arms:
        g = scored[scored["arm"] == arm]
        if g.empty:
            continue
        row = {"arm": arm, "n": int(len(g)),
               "wins_mae": float(np.abs(g["err_final_wins"]).mean()),
               "rest_wpct_mae": float(np.abs(g["err_rest_wpct"]).mean())}
        for prob in PROB_COLUMNS:
            row[f"brier_{prob[2:]}"] = float(g[f"brier_{prob}"].mean())
            row[f"logloss_{prob[2:]}"] = float(g[f"logloss_{prob}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def decomposition(scored: pd.DataFrame, arms) -> pd.DataFrame:
    """Murphy's reliability / resolution split for all four probabilities."""
    rows = []
    for arm in arms:
        if scored[scored["arm"] == arm].empty:
            continue
        for prob in PROB_COLUMNS:
            r = tb.reliability(scored, arm, prob)
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


PU_METRICS = {
    "wins_abs_err": ("err_final_wins", "abs"),
    "brier_playoffs": ("brier_p_playoffs", "raw"),
    "brier_division": ("brier_p_division", "raw"),
    "brier_pennant": ("brier_p_pennant", "raw"),
    "brier_ws": ("brier_p_ws", "raw"),
    "logloss_playoffs": ("logloss_p_playoffs", "raw"),
}


def paired_vs(scored: pd.DataFrame, arm: str, against, extra_by=()) -> pd.DataFrame:
    frames = [tb.paired(scored, arm, other, metrics=PU_METRICS,
                        extra_by=extra_by)
              for other in against if not scored[scored["arm"] == other].empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def spread_table(scored: pd.DataFrame, arms) -> pd.DataFrame:
    """How far each arm's board is from the middle, bucket by bucket.

    The mechanical half of the pre-registration: parameter uncertainty pulls
    probabilities toward the base rate, and this says by how much and when.
    """
    rows = []
    for arm in arms:
        g = scored[scored["arm"] == arm]
        if g.empty:
            continue
        for bucket, sub in g.groupby("bucket", observed=True):
            rows.append({"arm": arm, "bucket": bucket, "n": int(len(sub)),
                         "sd_p_playoffs": float(sub["p_playoffs"].std()),
                         "mean_abs_dev": float(
                             (sub["p_playoffs"] - sub["p_playoffs"].mean())
                             .abs().mean()),
                         "frac_extreme": float(
                             ((sub["p_playoffs"] < .02)
                              | (sub["p_playoffs"] > .98)).mean())})
    return pd.DataFrame(rows)


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def markdown(scored: pd.DataFrame, arms, lams) -> str:
    lines: list[str] = []
    head = headline(scored, arms)
    lines += ["### Headline: every arm", "",
              "| Arm | n | Wins MAE | Rest win% MAE | Brier playoffs | "
              "Log loss playoffs | Brier division | Brier pennant | Brier WS |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in head.itertuples(index=False):
        lines.append(
            f"| {r.arm} | {r.n} | {r.wins_mae:.3f} | {r.rest_wpct_mae:.5f} | "
            f"{r.brier_playoffs:.5f} | {r.logloss_playoffs:.5f} | "
            f"{r.brier_division:.5f} | {r.brier_pennant:.5f} | "
            f"{r.brier_ws:.5f} |")

    dec = decomposition(scored, arms)
    lines += ["", "### Murphy decomposition (brier ≈ reliability − resolution "
                  "+ uncertainty)", "",
              "| Arm | Outcome | Brier | Reliability ↓ | Resolution ↑ | "
              "Uncertainty | Residual | Skill |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in dec.itertuples(index=False):
        lines.append(f"| {r.arm} | {r.prob[2:]} | {r.brier:.5f} | "
                     f"{r.reliability:.5f} | {r.resolution:.5f} | "
                     f"{r.uncertainty:.5f} | {r.residual:+.5f} | "
                     f"{r.skill_score:.3f} |")

    others = [a for a in arms if a != PU_ARM]
    pair = paired_vs(scored, PU_ARM, others)
    lines += ["", f"### Paired: `{PU_ARM}` minus each arm "
                  "(negative favours the uncertainty arm)", "",
              "| Against | Metric | chain_pu | other | Δ | se | t | n |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in pair.itertuples(index=False):
        lines.append(f"| {r.against} | {r.metric} | {_fmt(r.mean_a, 5)} | "
                     f"{_fmt(r.mean_b, 5)} | {_fmt(r.diff, 5)} | "
                     f"{_fmt(r.se, 5)} | {_fmt(r.t, 2)} | {r.n} |")

    lines += ["", "### The crossover: each arm against `record_500`, "
                  "bucket by bucket", "",
              "Negative favours the model. The published crossover is where "
              "the `chain` row turns positive.", "",
              "| Bucket | Arm | Δ Brier playoffs | se | t | Δ wins MAE | se | t |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for arm in arms:
        if arm == CONTROL_ARM or scored[scored["arm"] == arm].empty:
            continue
        rows.append(tb.paired(scored, arm, CONTROL_ARM, metrics=PU_METRICS,
                              extra_by=("bucket",)))
    cross = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(cross):
        wide = cross.pivot_table(index=["bucket", "arm"], columns="metric",
                                 values=["diff", "se", "t"], observed=True)
        for (bucket, arm), r in wide.iterrows():
            lines.append(
                f"| {bucket} | {arm} | "
                f"{_fmt(r[('diff', 'brier_playoffs')], 5)} | "
                f"{_fmt(r[('se', 'brier_playoffs')], 5)} | "
                f"{_fmt(r[('t', 'brier_playoffs')], 2)} | "
                f"{_fmt(r[('diff', 'wins_abs_err')], 3)} | "
                f"{_fmt(r[('se', 'wins_abs_err')], 3)} | "
                f"{_fmt(r[('t', 'wins_abs_err')], 2)} |")

    lines += ["", f"### `{PU_ARM}` against `{BASE_ARM}`, bucket by bucket", "",
              "| Bucket | Metric | Δ | se | t | n |", "|---|---|---:|---:|---:|---:|"]
    bybucket = tb.paired(scored, PU_ARM, BASE_ARM, metrics=PU_METRICS,
                         extra_by=("bucket",))
    for r in bybucket.itertuples(index=False):
        lines.append(f"| {r.bucket} | {r.metric} | {_fmt(r.diff, 5)} | "
                     f"{_fmt(r.se, 5)} | {_fmt(r.t, 2)} | {r.n} |")

    sp = spread_table(scored, [BASE_ARM, PU_ARM, "chain_pu_double"])
    lines += ["", "### How far the width actually moves the board", "",
              "| Arm | Bucket | n | SD of P(playoffs) | mean abs deviation | "
              "fraction at 0/1 |", "|---|---|---:|---:|---:|---:|"]
    for r in sp.itertuples(index=False):
        lines.append(f"| {r.arm} | {r.bucket} | {r.n} | {r.sd_p_playoffs:.4f} | "
                     f"{r.mean_abs_dev:.4f} | {r.frac_extreme:.3f} |")

    lines += ["", "### Fitted shrinkage constants", "", "```",
              json.dumps(lams, indent=1, default=float), "```"]
    return "\n".join(lines)


def payload(scored: pd.DataFrame, arms, lams) -> dict:
    others = [a for a in arms if a != PU_ARM]
    cross = pd.concat(
        [tb.paired(scored, a, CONTROL_ARM, metrics=PU_METRICS,
                   extra_by=("bucket",))
         for a in arms if a != CONTROL_ARM
         and not scored[scored["arm"] == a].empty],
        ignore_index=True)
    return {
        "seasons": [int(s) for s in sorted(scored["season"].unique())],
        "n_as_of_dates": int(scored.groupby("season")["as_of"].nunique().sum()),
        "n_per_arm": int(len(scored[scored["arm"] == BASE_ARM])),
        "arms": list(arms),
        "lambdas": lams,
        "headline": json.loads(headline(scored, arms).to_json(orient="records")),
        "decomposition": json.loads(
            decomposition(scored, arms).to_json(orient="records")),
        "paired_vs_all": json.loads(
            paired_vs(scored, PU_ARM, others).to_json(orient="records")),
        "paired_by_bucket": json.loads(
            tb.paired(scored, PU_ARM, BASE_ARM, metrics=PU_METRICS,
                      extra_by=("bucket",)).to_json(orient="records")),
        "crossover": json.loads(cross.to_json(orient="records")),
        "fine_crossover": json.loads(pd.concat(
            [tb.paired(scored, a, CONTROL_ARM,
                       metrics={"brier_playoffs": ("brier_p_playoffs", "raw")},
                       extra_by=("fine_bucket",))
             for a in (BASE_ARM, PU_ARM)], ignore_index=True
        ).to_json(orient="records")),
        "spread": json.loads(spread_table(
            scored, [BASE_ARM, PU_ARM, "chain_pu_double"]
        ).to_json(orient="records")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--tag", default="")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    seasons = list(DEFAULT_SEASONS)
    projections = load_projections(seasons, args.out_dir, args.tag)
    outcomes = load_outcomes(seasons, args.out_dir)
    scored = tb.attach_outcomes(projections, outcomes)
    scored, lams = with_controls(scored)
    arms = [a for a in (BASE_ARM, PU_ARM, "chain_pu_half", "chain_pu_double",
                        "chain_pu_boot", *SHRINK_ARMS, CONTROL_ARM,
                        "record_wpct", "preseason", "coin_flip")
            if a in set(scored["arm"])]
    n_dates = int(scored.groupby("season")["as_of"].nunique().sum())
    print(f"{len(scored[scored['arm'] == BASE_ARM])} club-projections per arm "
          f"over {scored['season'].nunique()} seasons and {n_dates} as-of "
          f"dates\n")
    if args.markdown:
        print(markdown(scored, arms, lams))
    else:
        print(headline(scored, arms).round(5).to_string(index=False))
        print("\ndecomposition:")
        print(decomposition(scored, arms).round(5).to_string(index=False))
        print(f"\npaired {PU_ARM} minus each arm:")
        print(paired_vs(scored, PU_ARM,
                        [a for a in arms if a != PU_ARM]).round(5)
              .to_string(index=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload(scored, arms, lams), indent=1) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
