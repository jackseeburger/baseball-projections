"""Merge the per-(season, component) BAS-94 sweep shards into one checkpoint.

The dense sweep writes one checkpoint per process, and BAS-94 runs one process
per (season, component) for two reasons the sweep's own docstring gives: a
NumPyro process dies at around 70 XLA compilations, and two processes sharing
a checkpoint path would each write back only the cells they loaded at startup,
silently dropping the other's. So each shard gets its own `--out-dir` and this
script concatenates them into the checkpoint the scorer reads.

Idempotent and safe to run while a shard is still going: rows are keyed on
(component, model, season, cutoff, batter) and the last shard to carry a key
wins, which for disjoint shards is no choice at all. Shards are read in sorted
path order so a rerun produces the same file byte for byte.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KEY = ["component", "model", "season", "cutoff", "batter"]
FIT_KEY = ("arm", "component", "cutoff")


def merge(run_dir: Path, out_dir: Path) -> dict:
    shards = sorted(p for p in run_dir.glob("*/cells_bayes.parquet"))
    frames = [pd.read_parquet(p) for p in shards]
    summary: dict = {"shards": [str(p.parent.name) for p in shards],
                     "rows_per_shard": [int(len(f)) for f in frames]}
    if not frames:
        raise SystemExit(f"no shards under {run_dir}")
    cells = pd.concat(frames, ignore_index=True)
    before = len(cells)
    cells = cells.drop_duplicates(subset=KEY, keep="last")
    summary["rows"] = int(len(cells))
    summary["duplicate_rows_dropped"] = int(before - len(cells))
    summary["arms"] = sorted(cells["model"].unique().tolist())
    summary["cells"] = int(
        cells[["component", "season", "cutoff"]].drop_duplicates().shape[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(out_dir / "cells_bayes.parquet", index=False)

    fits: dict = {}
    for p in sorted(run_dir.glob("*/bayes_fits.json")):
        for f in json.loads(p.read_text()):
            fits[tuple(str(f.get(k)) for k in FIT_KEY)] = f
    records = [fits[k] for k in sorted(fits)]
    (out_dir / "bayes_fits.json").write_text(json.dumps(records, indent=1))
    summary["fits"] = len(records)
    summary["fit_arms"] = sorted({str(f.get("arm")) for f in records})
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=ROOT / "data/eval/bas94/run")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/eval/bas94")
    args = ap.parse_args()
    print(json.dumps(merge(args.run_dir, args.out_dir), indent=1))


if __name__ == "__main__":
    main()
