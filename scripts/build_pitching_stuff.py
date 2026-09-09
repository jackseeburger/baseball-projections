"""Build the monthly pitching-stuff artifact from the Statcast archive (BAS-71).

End to end: pull `statcast_<year>.parquet` from R2, turn every competitive
pitch into the physics feature frame, fit the walk-forward stuff models, score
every pitch, and reduce the scores to additive sufficient statistics per
pitcher per calendar month.

    python scripts/build_pitching_stuff.py --download
    python scripts/build_pitching_stuff.py --seasons 2015 2026
    python scripts/build_pitching_stuff.py --seasons 2022 2026 --metrics-out m.csv
    python scripts/build_pitching_stuff.py --download --update-season 2026

The output is `data/features/pitching_stuff_monthly.parquet`, and it is
committed precisely so the 1.6 GB download and the hour of model fitting never
have to happen twice. `--update-season <year>` rebuilds that one season's rows
in place and leaves every other season alone — the nightly refresh, so the
`stuff_additive` pitcher rates the site serves (BAS-79) read this season's
pitches instead of stopping wherever the committed artifact stopped. Both
paths stamp a `pitching_stuff_monthly.meta.json` sidecar with `built_at` and
the seasons touched, which is what `scripts/check_freshness.py` watches.

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

from src.data.pitching_stuff import (
    DEFAULT_PATH,
    build_year,
    monthly_buckets,
    save_monthly,
    write_meta,
)
from src.models.stuff import fit_stuff_model, walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stuff")
ROOT = Path(__file__).resolve().parent.parent

# Two prior seasons is the minimum a walk-forward fold is allowed, so 2017 is
# the first strictly-scored season.
MIN_TRAIN_SEASONS = 2
# How far back `--update-season` reaches for its training seasons. The full
# build trains a season on every season before it; a nightly refresh that did
# the same would download the whole archive every night for a model whose fit
# is capped at 1.5M sampled rows anyway. Three seasons is the same window the
# stage-2 recency grid spans, and the fit stays strictly walk-forward.
MAX_TRAIN_SEASONS = 3


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


def update_season(year: int, raw_dir: Path, out: Path, **kwargs) -> Path:
    """Rebuild one season's buckets in `out`, leaving every other season alone.

    What the nightly refresh runs (BAS-79), so the `stuff_additive` components
    of the served pitcher projection see this season's pitches rather than a
    committed artifact that stops in August. Re-aggregating twelve seasons and
    refitting every stage-1 arm nightly would cost the 1.6 GB download and an
    hour of fitting for rows that cannot change.

    The season is scored **walk-forward exactly as the committed artifact
    scores it**: the model is fitted on the seasons strictly before it that are
    present in `raw_dir`, and `src.models.stuff.assert_no_leak` (inside
    `fit_stuff_model`) re-checks the training frame rather than trusting the
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
    train = pd.concat([build_year(y, raw_dir) for y in train_years],
                      ignore_index=True)
    held = build_year(year, raw_dir)
    scores = {t: fit_stuff_model(train, t, "stuff", year, **kwargs).predict(held)
              for t in ("whiff", "csw")}
    del train
    fresh = monthly_buckets(held.assign(p_whiff=scores["whiff"],
                                        p_csw=scores["csw"]))
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
    ap.add_argument("--arms", nargs="+",
                    default=["stuff", "pitch_type", "fb_velo", "pitching"],
                    help="stage-1 comparison arms to fit and score")
    ap.add_argument("--metrics-out", type=Path,
                    default=ROOT / "data/eval/pitching_stuff_stage1.json",
                    help="per-season out-of-sample log-loss and AUC")
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument(
        "--update-season", type=int, default=None,
        help="rebuild only this season's rows in --out and leave every other "
             "season's rows untouched (BAS-79: the nightly refresh of the "
             "current season, so the served pitcher rates read this season's "
             "pitches, rather than re-aggregating and re-fitting twelve "
             "seasons every night). Still walk-forward: the model is fitted "
             "on seasons strictly before it.")
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
    meta = write_meta(args.out, seasons_built=years)
    logger.info("wrote %s (%d rows, %.2f MB); %s", path, len(out),
                path.stat().st_size / 1e6, meta)

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
