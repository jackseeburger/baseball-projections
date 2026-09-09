"""Build the monthly pitching-stuff artifact from the Statcast archive (BAS-71).

End to end: pull `statcast_<year>.parquet` from R2, turn every competitive
pitch into the physics feature frame, fit the walk-forward stuff models, score
every pitch, and reduce the scores to additive sufficient statistics per
pitcher per calendar month.

    python scripts/build_pitching_stuff.py --download
    python scripts/build_pitching_stuff.py --seasons 2015 2026
    python scripts/build_pitching_stuff.py --seasons 2022 2026 --metrics-out m.csv

The output is `data/features/pitching_stuff_monthly.parquet`, and it is
committed precisely so the 1.6 GB download and the hour of model fitting never
have to happen twice.

**The monthly grain, and what a caller may do with it.** Every column in a
bucket is a sum over the pitches thrown in that calendar month, so any window
is a sum of buckets. A cutoff that falls on the **first of a month** is
therefore reconstructed *exactly* by summing the buckets strictly earlier than
it — season < Y, or season == Y and month < the cutoff's month — with no
filtering of an eight-million-row table at score time and no way for a pitch
thrown on or after the cutoff to reach the feature. `src/eval/stuff.py`
implements exactly that sum and refuses a cutoff that is not the first of a
month rather than rounding one forward, because rounding a cutoff forward is
leakage. `tests/test_data/test_pitching_stuff.py` drives the guard with a
season whose every post-cutoff month is thousands of maximum-stuff pitches and
asserts a first-of-month cutoff sees none of them.

**Which model scored which season.** Seasons 2017 onward are scored strictly
walk-forward: the model that scores season Y was fitted on pitches from
seasons <= Y-1 and `src.models.stuff.assert_no_leak` re-checks the training
frame rather than trusting the filter. The two seed seasons, 2015 and 2016,
have no two prior seasons to train on and are scored **in sample** by the
earliest model (the one fitted on 2015-2016, which scores 2017). That is
recorded here rather than hidden: those two seasons only ever enter the
three-season recency window of a 2017 or 2018 stage-2 cell, and every holdout
season the gate is scored on (2022-2026) draws its window entirely from
strictly walk-forward scores.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.pitching_stuff import DEFAULT_PATH, build_year, monthly_buckets, save_monthly
from src.models.stuff import fit_stuff_model, walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stuff")
ROOT = Path(__file__).resolve().parent.parent

# Two prior seasons is the minimum a walk-forward fold is allowed, so 2017 is
# the first strictly-scored season.
MIN_TRAIN_SEASONS = 2


def download_archive(years: list[int], raw_dir: Path) -> None:
    """Fetch any missing `statcast_<year>.parquet` from R2."""
    from src.data.r2 import bucket, get_s3_client

    s3, b = get_s3_client(), bucket()
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        path = raw_dir / f"statcast_{year}.parquet"
        if path.exists():
            continue
        logger.info("downloading statcast_%d.parquet", year)
        s3.download_file(b, f"statcast/statcast_{year}.parquet", str(path))


def seed_scores(seasons: dict, seed_years: list[int], first_scored: int,
                **kwargs) -> dict:
    """In-sample predictions for the seasons with no two prior seasons.

    The model is the same one that scores `first_scored` — fitted on exactly
    the seed seasons — so the seed buckets are on the same scale as everything
    after them. In sample, and said so in the docstring above.
    """
    train = pd.concat([seasons[y] for y in seed_years], ignore_index=True)
    out = {}
    models = {t: fit_stuff_model(train, t, "stuff", first_scored, **kwargs)
              for t in ("whiff", "csw")}
    for y in seed_years:
        out[y] = {t: (m.predict(seasons[y]), m) for t, m in models.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs=2, type=int, default=[2015, 2026],
                    metavar=("FIRST", "LAST"),
                    help="inclusive season range to build (default 2015 2026)")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--out", type=Path, default=ROOT / DEFAULT_PATH)
    ap.add_argument("--download", action="store_true",
                    help="pull missing seasons from R2 first")
    ap.add_argument("--arms", nargs="+",
                    default=["stuff", "pitch_type", "fb_velo", "pitching"],
                    help="stage-1 comparison arms to fit and score")
    ap.add_argument("--metrics-out", type=Path,
                    default=ROOT / "data/eval/pitching_stuff_stage1.json",
                    help="per-season out-of-sample log-loss and AUC")
    ap.add_argument("--max-train-rows", type=int, default=None)
    args = ap.parse_args()

    years = list(range(args.seasons[0], args.seasons[1] + 1))
    if args.download:
        download_archive(years, args.raw_dir)

    seasons = {y: build_year(y, args.raw_dir) for y in years}
    kwargs = ({"max_rows": args.max_train_rows} if args.max_train_rows
              else {})

    score_years = [y for y in years
                   if sum(1 for x in years if x < y) >= MIN_TRAIN_SEASONS]
    metrics, scored = walk_forward(seasons, score_years, arms=tuple(args.arms),
                                   min_train_seasons=MIN_TRAIN_SEASONS,
                                   **kwargs)

    seed_years = [y for y in years if y not in score_years]
    if seed_years and score_years:
        scored.update(seed_scores(seasons, seed_years, score_years[0], **kwargs))

    frames = []
    for year in years:
        if year not in scored:
            logger.warning("%d: no stuff scores, skipped", year)
            continue
        df = seasons[year]
        df = df.assign(p_whiff=scored[year]["whiff"][0],
                       p_csw=scored[year]["csw"][0])
        g = monthly_buckets(df)
        logger.info("%d: %d buckets over %d pitchers", year, len(g),
                    g["pitcher"].nunique())
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    path = save_monthly(out, args.out)
    logger.info("wrote %s (%d rows, %.2f MB)", path, len(out),
                path.stat().st_size / 1e6)

    if not metrics.empty:
        print("\n=== stage 1: out-of-sample by season ===")
        wide = metrics.pivot_table(index=["target", "season"], columns="arm",
                                   values=["log_loss", "auc"])
        print(wide.round(5).to_string())
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps({
            "seed_seasons": seed_years, "score_seasons": score_years,
            "metrics": json.loads(metrics.to_json(orient="records")),
        }, indent=1) + "\n")
        print(f"\nwrote {args.metrics_out}")


if __name__ == "__main__":
    main()
