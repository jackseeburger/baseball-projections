"""BAS-86: build data/features/park_factors.parquet from the PA outcomes.

    python scripts/build_park_factors.py                 # rebuild the artifact
    python scripts/build_park_factors.py --choose-ballast # re-run the sweep too

The factors themselves (`src/data/park_components.py`) are per park, per
season, per component (K, BB, HR, BABIP, ISO): home/away paired, three
completed seasons of window, regressed toward 1 with a pseudo-count ballast
and renormalised so a season's parks average exactly 1. The file is written
*wide* — one `<component>_park_factor` column per component — because that is
the shape `src.models.pa_rate.prepare_model_data` reads
(`RateComponent.park_factor_col`); the logical key is still team x season x
component and `--long` writes that shape next to it for inspection.

**The ballast is chosen once and frozen.** `--choose-ballast` re-runs the
leave-one-season-out sweep on 2015-2019 and prints what it would pick; it does
*not* rewrite `FROZEN_BALLAST`, which is a constant in the module for exactly
the reason docs/park-factors.md pre-registered it that way — a number tuned
again on every build is not a frozen number. The sweep is written into the
sidecar on every build regardless (it costs five seconds), so the table behind
the frozen choice is always next to the artifact.

Nothing here touches the network: it reads `data/parquet/pa_outcomes/*` and
writes `data/features/park_factors.parquet` plus its `.meta.json` sidecar.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import park_components as pk

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data/features/park_factors.parquet"
# Seasons a factor is stamped for. 2016 is the first with any prior season on
# disk; 2026 is the season in progress, whose window (2023-2025) is complete.
DEFAULT_SEASONS = tuple(range(2016, 2027))
DATA_SEASONS = tuple(range(2015, 2027))


def _records(df: pd.DataFrame) -> list:
    """DataFrame → JSON-safe records (NaN → None, numpy scalars → Python)."""
    return json.loads(df.to_json(orient="records"))


def build(pa_dir: Path, seasons, out: Path, long_out: Path | None = None,
          ballast=None) -> dict:
    counts = pk.load_pa_counts(DATA_SEASONS, pa_dir)
    if counts.empty:
        raise SystemExit(f"no PA outcome parquets under {pa_dir}")
    have = sorted(int(y) for y in counts["game_year"].unique())

    ballast = pk.FROZEN_BALLAST if ballast is None else ballast
    long = pk.build_table(counts, seasons, ballast)
    wide = pk.wide_table(long)

    out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out, index=False)
    if long_out is not None:
        long.assign(window=long["window"].map(lambda w: ",".join(str(s) for s in w))
                    ).to_parquet(long_out, index=False)

    persistence = pk.loso_persistence(counts)
    spread = pk.log_factor_spread(long)
    yoy = pk.year_over_year_correlation(long)
    single = pk.single_season_persistence(counts)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": "data/parquet/pa_outcomes/pa_outcomes_<year>.parquet",
        "seasons_with_data": have,
        "seasons_stamped": sorted(int(s) for s in long["game_year"].unique()),
        "n_pa": int(counts["k_rate_d"].sum()),
        "components": list(pk.COMPONENT_NAMES),
        "factor_columns": dict(pk.FACTOR_COLUMNS),
        "window_seasons": pk.WINDOW_SEASONS,
        "windows": {str(int(s)): list(pk.window_for(s, have))
                    for s in sorted(int(s) for s in long["game_year"].unique())},
        "ballast": {k: float(v) for k, v in dict(ballast).items()},
        "ballast_chosen_on": list(pk.BALLAST_SEASONS),
        "ballast_grid": [float(b) for b in pk.BALLAST_GRID],
        "ballast_selection_rule": (
            "minimum pooled leave-one-season-out RMSE of log factor against "
            "the held-out season's own single-season split; ties break toward "
            "the larger ballast"
        ),
        "ballast_would_choose_now": pk.choose_ballast(persistence),
        "loso_persistence": _records(persistence),
        "log_factor_sd_by_season": _records(spread),
        "year_over_year_correlation": _records(yoy),
        "single_season_persistence": _records(single),
        "known_park_changes": pk.KNOWN_PARK_CHANGES,
        "notes": [
            "The key is the hosting club, not a venue id: the PA parquet has "
            "no venue column, and the rate models index their park offset on "
            "the batting team. A club that changes parks carries its old "
            "park's factor until the window rolls over (see "
            "known_park_changes); neutral-site games are charged to the "
            "nominal home club.",
            "2020 is in the pool as a 60-game season. It is not weighted "
            "down or up: the counts are summed, so it contributes about a "
            "third of a normal season's trials to the windows stamped for "
            "2021, 2022 and 2023.",
            "Factors are renormalised to a trials-weighted mean of 1 within "
            "each season and component, so applying one cannot move the "
            "projected league level.",
        ],
    }
    meta_path = out.with_suffix("").with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    return {"wide": wide, "long": long, "meta": meta, "meta_path": meta_path,
            "persistence": persistence, "spread": spread, "yoy": yoy,
            "single": single}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pa-dir", type=Path, default=ROOT / "data/parquet/pa_outcomes")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--long-out", type=Path, default=None,
                    help="also write the long [team, game_year, component, "
                         "factor] shape here (not needed by any consumer)")
    ap.add_argument("--seasons", nargs="+", type=int, default=list(DEFAULT_SEASONS))
    ap.add_argument("--choose-ballast", action="store_true",
                    help="print what the leave-one-season-out sweep would "
                         "pick now; the build still uses the frozen values")
    args = ap.parse_args()

    res = build(args.pa_dir, args.seasons, args.out, args.long_out)
    wide, meta = res["wide"], res["meta"]
    print(f"park factors: {len(wide)} team-season rows "
          f"({wide['game_year'].min()}-{wide['game_year'].max()}, "
          f"{wide['team'].nunique()} teams) -> {args.out}")
    print(f"sidecar -> {res['meta_path']}")
    print(f"ballast (frozen): {meta['ballast']}")

    if args.choose_ballast:
        pooled = res["persistence"][res["persistence"]["season"].isna()]
        print("\nleave-one-season-out persistence "
              f"({', '.join(str(s) for s in pk.BALLAST_SEASONS)}), pooled:")
        print(pooled.pivot_table(index="ballast", columns="component",
                                 values=["rmse", "corr"]).round(4).to_string())
        print(f"\nwould choose: {meta['ballast_would_choose_now']}")
        print(f"frozen:       {meta['ballast']}")

    print("\nsd of log factor across parks, latest stamped season:")
    latest = res["spread"][res["spread"]["game_year"] == res["spread"]["game_year"].max()]
    print(latest.to_string(index=False))
    print("\nyear-over-year correlation of the stamped factor (pooled):")
    yoy = res["yoy"]
    print(yoy[yoy["season"].isna()][["component", "corr", "n_parks"]].to_string(index=False))
    print("\n... of the single-season raw split (no window overlap):")
    print(res["single"].to_string(index=False))

    hr = res["wide"].sort_values("hr_park_factor")
    cols = ["team", "game_year", "k_park_factor", "hr_park_factor",
            "babip_park_factor"]
    last = hr[hr["game_year"] == hr["game_year"].max()]
    print(f"\nHR extremes, {int(last['game_year'].iloc[0])}:")
    print(pd.concat([last.head(3), last.tail(3)])[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
