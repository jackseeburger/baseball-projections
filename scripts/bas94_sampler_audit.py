"""Is the borrowed `bayes_walk` comparator a sampler comparison in disguise?

BAS-94 reads `bayes_walk` off the BAS-85 grid rather than refitting it, which
saves 27 fit-hours and pins the comparator to numbers already published. But
BAS-85 ran `bayes_walk` under **pymc** (its measurement arm needed pymc, and
the twin was fit beside it), while BAS-94's prior-mean arms run under
**numpyro**. Both target the same posterior and BAS-92 validated numpyro on
this exact graph — but "should" is not a measurement, and a comparator drawn
by a different sampler is exactly the kind of thing that turns a 2% effect
into an artifact.

So one season of `bayes_walk` is refit here under numpyro, on the same cells,
and this script reports what changes: each arm's own MAE, the paired
difference between the two samplers' `bayes_walk`, and — the number that
actually matters — arm A's gap against each of them. If the two gaps agree,
the borrowed comparator carries no sampler signal and the headline tables
stand as written.

    python scripts/run_intraseason_backtest_dense.py --stage bayes \\
        --bayes-seasons 2024 --bayes-components hr_rate \\
        --variants ability_walk --out-dir data/eval/bas94/sampler_audit/2024_hr_rate
    python scripts/bas94_sampler_audit.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dense = _load("run_intraseason_backtest_dense",
              ROOT / "scripts/run_intraseason_backtest_dense.py")
bas94 = _load("analyze_bas94", ROOT / "scripts/analyze_bas94.py")

NUMPYRO_WALK = "bayes_walk_numpyro"


def load(audit_dir: Path, scored_path: Path) -> pd.DataFrame:
    """The scored cells, plus the refit walk under a name of its own.

    Renamed rather than replaced: both samplers' rows have to be in the frame
    at once for a paired comparison between them to exist at all.
    """
    shards = sorted(audit_dir.glob("*/cells_bayes.parquet"))
    if not shards:
        raise SystemExit(f"no audit shards under {audit_dir}")
    refit = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    refit = refit[refit["model"] == bas94.WALK_ARM].copy()
    refit["model"] = NUMPYRO_WALK
    scored = pd.read_parquet(scored_path)
    keys = set(map(tuple, refit[["component", "season", "cutoff"]]
                   .drop_duplicates().to_numpy().tolist()))
    keep = [(c, s, cu) in keys for c, s, cu in
            zip(scored["component"], scored["season"], scored["cutoff"])]
    return pd.concat([scored[keep], refit], ignore_index=True)


def audit(cells: pd.DataFrame) -> dict:
    out: dict = {"scope": {
        "seasons": sorted(int(s) for s in cells["season"].unique()),
        "cutoffs": sorted(cells["cutoff"].unique().tolist()),
        "components": sorted(cells["component"].unique().tolist())}}
    for component in out["scope"]["components"]:
        rows = []
        for arm, base in ((NUMPYRO_WALK, bas94.WALK_ARM),
                          (bas94.ARM_A, bas94.WALK_ARM),
                          (bas94.ARM_A, NUMPYRO_WALK),
                          (bas94.ARM_B, bas94.WALK_ARM),
                          (bas94.ARM_B, NUMPYRO_WALK)):
            r = bas94._compare(cells, arm, base, component)
            if r:
                rows.append(r)
        out[component] = rows
        pair = {r["base"]: r for r in rows if r["arm"] == bas94.ARM_A}
        if len(pair) == 2:
            a = pair[bas94.WALK_ARM]["pct_of_base_mae"]
            b = pair[NUMPYRO_WALK]["pct_of_base_mae"]
            out[f"{component}_gap_shift_pct"] = float(b - a)
        # Prediction 2 is scored at the May cutoffs, so the May gap is the one
        # that has to survive the swap — a pooled figure could hide a sampler
        # effect concentrated exactly where the prediction lives.
        out[f"{component}_regime"] = {
            bas94.WALK_ARM: bas94.regime_split(cells, bas94.ARM_A,
                                               bas94.WALK_ARM, component),
            NUMPYRO_WALK: bas94.regime_split(cells, bas94.ARM_A,
                                             NUMPYRO_WALK, component),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-dir", type=Path,
                    default=ROOT / "data/eval/bas94/sampler_audit")
    ap.add_argument("--scored", type=Path,
                    default=ROOT / "data/eval/bas94/cells_scored.parquet")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/eval/bas94/sampler_audit.json")
    args = ap.parse_args()

    cells = load(args.audit_dir, args.scored)
    payload = audit(cells)
    for component in payload["scope"]["components"]:
        print(f"\n=== {component} ===")
        print(bas94.render(payload[component]))
        shift = payload.get(f"{component}_gap_shift_pct")
        if shift is not None:
            print(f"  arm A's gap moves {shift:+.2f} points of MAE when the "
                  f"comparator is refit under numpyro")
        regime = payload.get(f"{component}_regime") or {}
        for base, split in regime.items():
            for name, r in split.items():
                print(f"  {name:<8} vs {base:<20} "
                      f"{r['pct_of_base_mae']:+7.2f}%  "
                      f"t {r['clustered_by_player_t']:+.2f}  "
                      f"{r['arm_wins_cells']}-{r['arm_loses_cells']}")
    args.out.write_text(json.dumps(payload, indent=1, default=float))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
