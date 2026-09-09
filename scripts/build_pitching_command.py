"""Build the monthly pitching-command artifact from the Statcast archive (BAS-76).

End to end: pull `statcast_<year>.parquet` from R2, turn every competitive
pitch into the physics-plus-location feature frame, fit the walk-forward
stage-1 arms, score every pitch under both the location-inclusive model and
the stuff-only model, and reduce the *difference* to additive sufficient
statistics per pitcher per calendar month.

    python scripts/build_pitching_command.py --download
    python scripts/build_pitching_command.py --seasons 2015 2026
    python scripts/build_pitching_command.py --download --update-season 2026

The output is `data/features/pitching_command_monthly.parquet`, committed for
the same reason `pitching_stuff_monthly.parquet` is: the download and the
model fitting should never have to happen twice. `--update-season <year>`
rebuilds that one season's rows in place and leaves every other season alone,
and both paths stamp a `pitching_command_monthly.meta.json` sidecar with
`built_at` and the seasons touched.

**Why the artifact is a residual.** The served pitching engine already carries
a stuff score. The question BAS-76 asks is what is *left* once that score is
accounted for, so what this writes down is, per pitch, the CSW probability the
location-inclusive model assigns minus the probability the stuff-only model
assigns — and the same difference on called-strike-given-taken. Both models
are fitted on the same rows of the same walk-forward training frame, which is
the only way their difference is a difference rather than two separate
approximations. `src/data/pitching_command.py`'s module docstring says the
rest.

**The monthly grain, and what a caller may do with it.** Every column in a
bucket is a sum over the pitches thrown in that calendar month, so any window
is a sum of buckets. A cutoff that falls on the **first of a month** is
reconstructed *exactly* by summing the buckets strictly earlier than it —
season < Y, or season == Y and month < the cutoff's month — with no way for a
pitch thrown on or after the cutoff to reach the feature.
`src/eval/command.py` implements exactly that sum and refuses a cutoff that is
not the first of a month rather than rounding one forward, because rounding a
cutoff forward is leakage. `tests/test_data/test_pitching_command.py` drives
both guards.

**Which model scored which season.** Seasons 2017 onward are scored strictly
walk-forward, and `src.models.stuff.assert_no_leak` re-checks the training
frame rather than trusting the filter. The two seed seasons, 2015 and 2016,
have no two prior seasons to train on and are scored **in sample** by the
earliest model. Every holdout season the stage-2 gate is scored on (2022-2026)
draws its window entirely from strictly walk-forward scores.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.pitching_command import (
    DEFAULT_PATH,
    VACUITY_MIN_R,
    build_year,
    monthly_buckets,
    save_monthly,
    write_meta,
    year_over_year,
)
from src.models.command import (
    ARMS,
    KEEP_ARMS,
    attach_scores,
    fit_command_model,
    training_columns,
    target_rows,
    walk_forward,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("command")
ROOT = Path(__file__).resolve().parent.parent

MIN_TRAIN_SEASONS = 2
# How far back `--update-season` reaches for training seasons; the same three
# the stage-2 recency grid spans, and the fit stays strictly walk-forward.
MAX_TRAIN_SEASONS = 3
TARGETS = ("csw", "called_taken")


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


def _score_with(models: dict, held: pd.DataFrame) -> dict:
    """Full-length predictions for every kept arm on both targets."""
    import numpy as np

    out = {}
    for target in TARGETS:
        rows = target_rows(held, target)
        out[target] = {}
        for arm in KEEP_ARMS:
            m = models[(target, arm)]
            if len(rows) == len(held):
                out[target][arm] = m.predict(held)
            else:
                p = np.full(len(held), np.nan)
                p[held.index.get_indexer(rows.index)] = m.predict(rows)
                out[target][arm] = p
    return out


def seed_scores(seasons: dict, seed_years: list[int], first_scored: int,
                **kwargs) -> dict:
    """In-sample predictions for the seasons with no two prior seasons.

    The model is the same one that scores `first_scored` — fitted on exactly
    the seed seasons — so the seed buckets are on the same scale as everything
    after them. In sample, and said so in the module docstring.
    """
    keep = training_columns(KEEP_ARMS, TARGETS)
    train = pd.concat([seasons[y][keep] for y in seed_years], ignore_index=True)
    models = {(t, a): fit_command_model(train, t, a, first_scored, **kwargs)
              for t in TARGETS for a in KEEP_ARMS}
    return {y: _score_with(models, seasons[y]) for y in seed_years}


def update_season(year: int, raw_dir: Path, out: Path, **kwargs) -> Path:
    """Rebuild one season's buckets in `out`, leaving every other season alone.

    Scored **walk-forward exactly as the committed artifact scores it**: the
    models are fitted on the seasons strictly before it that are present in
    `raw_dir`, and `src.models.stuff.assert_no_leak` (inside
    `fit_command_model`) re-checks the training frame rather than trusting the
    filter. Fewer prior seasons than the full build had is a weaker model, not
    a leaky one; `MIN_TRAIN_SEASONS` is still the floor and below it this
    refuses rather than scoring a season in sample.
    """
    train_years = sorted(
        y for y in range(year - MAX_TRAIN_SEASONS, year)
        if (raw_dir / f"statcast_{y}.parquet").exists())
    if len(train_years) < MIN_TRAIN_SEASONS:
        raise SystemExit(
            f"--update-season {year} needs at least {MIN_TRAIN_SEASONS} prior "
            f"seasons in {raw_dir} to fit walk-forward; found {train_years}. "
            f"Pass --download, or run the full build.")
    logger.info("fitting the %d models on %s", year, train_years)
    keep = training_columns(KEEP_ARMS, TARGETS)
    train = pd.concat([build_year(y, raw_dir)[keep] for y in train_years],
                      ignore_index=True)
    held = build_year(year, raw_dir)
    models = {(t, a): fit_command_model(train, t, a, year, **kwargs)
              for t in TARGETS for a in KEEP_ARMS}
    del train
    fresh = monthly_buckets(attach_scores(held, _score_with(models, held)))
    logger.info("%d: %d buckets over %d pitchers", year, len(fresh),
                fresh["pitcher"].nunique())

    if out.exists():
        existing = pd.read_parquet(out)
        combined = pd.concat([existing[existing["season"] != year], fresh],
                             ignore_index=True)
    else:
        combined = fresh
    path = save_monthly(combined, out)
    meta = write_meta(out, seasons_built=[year])
    logger.info("wrote %s (%d rows total, %d for %d, %.2f MB); %s", path,
                len(combined), len(fresh), year, path.stat().st_size / 1e6,
                meta)
    return path


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
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    help="stage-1 comparison arms to fit and score")
    ap.add_argument("--metrics-out", type=Path,
                    default=ROOT / "data/eval/pitching_command_stage1.json",
                    help="per-season out-of-sample log-loss and AUC, plus the "
                         "pre-registered vacuity check")
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument(
        "--update-season", type=int, default=None,
        help="rebuild only this season's rows in --out and leave every other "
             "season's rows untouched. Still walk-forward: the models are "
             "fitted on seasons strictly before it.")
    args = ap.parse_args()

    kwargs = ({"max_rows": args.max_train_rows} if args.max_train_rows else {})
    if args.update_season is not None:
        year = args.update_season
        if args.download:
            download_archive(
                [y for y in range(year - MAX_TRAIN_SEASONS, year + 1)],
                args.raw_dir)
        update_season(year, args.raw_dir, args.out, **kwargs)
        return

    years = list(range(args.seasons[0], args.seasons[1] + 1))
    if args.download:
        download_archive(years, args.raw_dir)

    seasons = {y: build_year(y, args.raw_dir) for y in years}

    score_years = [y for y in years
                   if sum(1 for x in years if x < y) >= MIN_TRAIN_SEASONS]
    metrics, scored = walk_forward(seasons, score_years, arms=tuple(args.arms),
                                   targets=TARGETS,
                                   min_train_seasons=MIN_TRAIN_SEASONS,
                                   **kwargs)

    seed_years = [y for y in years if y not in score_years]
    if seed_years and score_years:
        scored.update(seed_scores(seasons, seed_years, score_years[0], **kwargs))

    frames = []
    for year in years:
        if year not in scored:
            logger.warning("%d: no command scores, skipped", year)
            continue
        g = monthly_buckets(attach_scores(seasons[year], scored[year]))
        logger.info("%d: %d buckets over %d pitchers", year, len(g),
                    g["pitcher"].nunique())
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    path = save_monthly(out, args.out)
    meta = write_meta(args.out, seasons_built=years)
    logger.info("wrote %s (%d rows, %.2f MB); %s", path, len(out),
                path.stat().st_size / 1e6, meta)

    # The pre-registered vacuity check, run here rather than in the stage-2
    # script precisely because it is meant to gate stage 2: if a pitcher's
    # command does not carry from one season to the next there is no talent
    # for a talent-layer covariate to pool.
    vac = {c: year_over_year(out, c) for c in ("cmd_resid", "cs_resid")}
    print("\n=== vacuity: year-over-year correlation of the pitcher-season "
          f"command aggregate (>= 1,000 pitches both years, floor {VACUITY_MIN_R}) ===")
    for c, v in vac.items():
        print(f"  {c}: r = {v['r']:.4f} over {v['n_pairs']} pitcher-season pairs"
              f"  -> {'PASS' if v['r'] > VACUITY_MIN_R else 'FAIL'}")

    if not metrics.empty:
        print("\n=== stage 1: out-of-sample by season ===")
        wide = metrics.pivot_table(index=["target", "season"], columns="arm",
                                   values=["log_loss", "auc"])
        print(wide.round(5).to_string())

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps({
        "seed_seasons": seed_years, "score_seasons": score_years,
        "vacuity": vac, "vacuity_floor": VACUITY_MIN_R,
        "metrics": json.loads(metrics.to_json(orient="records")),
    }, indent=1) + "\n")
    print(f"\nwrote {args.metrics_out}")


if __name__ == "__main__":
    main()
