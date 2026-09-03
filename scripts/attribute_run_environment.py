"""Where each club's top-down and bottom-up run environments disagree, and why.

The validation doc's table (docs/playoff-odds-validation.md) is the reason this
ticket exists: Atlanta allows 4.02 runs a game while the components say 4.38,
Milwaukee scores 4.84 against a component 4.61, Los Angeles bats project 5.19
against 4.78 scored. Station C blends the two halves half and half, so every
one of those gaps is currently absorbed as an unattributed lump.

This script names the parts of it that two new terms can explain:

  * **park** (`src/sim/park.py`) — the top-down half is measured in the parks
    the club has played in and the bottom-up half is park-neutral, so half the
    gap could be the ballpark. Neutralising the top-down half moves it, and
    the size of that move is the park's share.
  * **defence** (`src/sim/defence.py`) — FIP throws away every ball in play, so
    a club that fields well allows fewer runs than its components say. Adding
    its regressed BABIP-allowed residual to the bottom-up half moves that half,
    and the size of that move is defence's share.

Whatever is left after both is sequencing, baserunning, bullpen leverage, luck
and everything else neither estimator can see.

Usage:
    python scripts/attribute_run_environment.py --season 2026 --as-of 2026-09-03
    python scripts/attribute_run_environment.py --season 2026 --cached
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.mlb_stats_api import fetch_schedule
from src.sim import defence as df_model
from src.sim import game_model as gm
from src.sim import park as pk_model
from src.sim.teams import fetch_teams

REGRESS_GAMES = 60.0
NAMED = ("ATL", "MIL", "LAD", "PHI")


def completed_totals(schedule: pd.DataFrame, as_of: str,
                     team_ids) -> pd.DataFrame:
    """Runs scored / allowed and games played, from games strictly before `as_of`."""
    done = schedule[(schedule["status"] == "Final")
                    & (schedule["game_type"] == "R")
                    & (schedule["date"].astype(str) < str(as_of))]
    done = done.dropna(subset=["home_score", "away_score"])
    home = done.groupby("home_id").agg(rs=("home_score", "sum"),
                                       ra=("away_score", "sum"),
                                       g=("home_score", "size"))
    away = done.groupby("away_id").agg(rs=("away_score", "sum"),
                                       ra=("home_score", "sum"),
                                       g=("away_score", "size"))
    tot = home.add(away, fill_value=0).reindex([int(t) for t in team_ids]).fillna(0.0)
    tot.index.name = "team_id"
    return tot


def _odds_job():
    """The nightly job as a module, so this script reads exactly what it reads.

    Loaded by path rather than imported, because `scripts/` is not a package
    and the point of this script is to ask the *production* fetch for the same
    frames the odds job gets.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_playoff_odds_attr",
        Path(__file__).resolve().parent / "run_playoff_odds.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build(as_of: str, data: dict, totals: pd.DataFrame, config: gm.ChainConfig):
    lg_rs9, lg_ra9 = gm.league_run_rates(float(totals["rs"].sum()),
                                         float(totals["ra"].sum()),
                                         float(totals["g"].sum()))
    top_down = pk_model.neutral_run_rates(totals["rs"], totals["ra"],
                                          totals["g"], REGRESS_GAMES)
    return gm.build_slate(as_of, data["inputs"], top_down, lg_rs9, lg_ra9,
                          config=config, totals=totals)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--as-of", default=date.today().isoformat())
    ap.add_argument("--park-ballast", type=float, default=pk_model.BALLAST_GAMES)
    ap.add_argument("--def-ballast", type=float, default=df_model.BALLAST_BIP)
    ap.add_argument("--cached", action="store_true",
                    help="do not re-pull this season's game logs")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the table here as CSV")
    args = ap.parse_args()

    rp = _odds_job()
    teams = fetch_teams(args.season)
    sched = fetch_schedule(f"{args.season}-03-01", f"{args.season}-10-15")
    data = rp.fetch_chain_data(args.season, date.fromisoformat(args.as_of),
                               rp.STARTER_WINDOW_DAYS,
                               refresh=not args.cached, schedule=sched)
    totals = completed_totals(sched, args.as_of, teams["team_id"])

    off = build(args.as_of, data, totals,
                gm.ChainConfig(park_ballast=float("inf"),
                               def_ballast=float("inf"),
                               regress_games=REGRESS_GAMES))
    on = build(args.as_of, data, totals,
               gm.ChainConfig(park_ballast=args.park_ballast,
                              def_ballast=args.def_ballast,
                              regress_games=REGRESS_GAMES))

    raw = off.diagnostics["top_down"]
    neutral = on.diagnostics["top_down"]
    fip = off.diagnostics["bottom_up_fip_only"]
    with_def = on.diagnostics["bottom_up"]
    exposure = on.diagnostics["park_exposure"]
    deltas = on.diagnostics["def_deltas"]
    babip = on.diagnostics["def_babip"]

    abbrev = teams.set_index("team_id")["abbrev"]
    rows = []
    for t in raw.index:
        t = int(t)
        rows.append({
            "club": abbrev.get(t, t),
            "rs_top": raw.loc[t, "rs_pg"], "rs_bot": fip.loc[t, "rs_pg"],
            "rs_gap": fip.loc[t, "rs_pg"] - raw.loc[t, "rs_pg"],
            "rs_park": neutral.loc[t, "rs_pg"] - raw.loc[t, "rs_pg"],
            "rs_gap_after": fip.loc[t, "rs_pg"] - neutral.loc[t, "rs_pg"],
            "ra_top": raw.loc[t, "ra_pg"], "ra_bot": fip.loc[t, "ra_pg"],
            "ra_gap": fip.loc[t, "ra_pg"] - raw.loc[t, "ra_pg"],
            "ra_park": neutral.loc[t, "ra_pg"] - raw.loc[t, "ra_pg"],
            "ra_def": with_def.loc[t, "ra_pg"] - fip.loc[t, "ra_pg"],
            "ra_gap_after": with_def.loc[t, "ra_pg"] - neutral.loc[t, "ra_pg"],
            "park_exposure": exposure.get(t, 1.0),
            "babip_allowed": babip.get(t, float("nan")),
            "def_ra9": deltas.get(t, 0.0),
        })
    table = pd.DataFrame(rows).set_index("club").sort_values("ra_gap")

    print(f"as of {args.as_of}: {int(totals['g'].sum() / 2)} games played, "
          f"park ballast {args.park_ballast:.0f} over "
          f"{pk_model.PRIOR_SEASONS} prior seasons, defence ballast "
          f"{args.def_ballast:.0f} balls in play\n")
    print("gap = bottom-up minus top-down; park = what neutralising the "
          "top-down half moves it;\ndefence = what the BABIP residual moves "
          "the bottom-up half; after = what is left.\n")
    print(table.round(3).to_string())

    print("\nHow much of the gap the two terms explain, in runs per game:")
    for side, gap, after in (("runs scored", "rs_gap", "rs_gap_after"),
                             ("runs allowed", "ra_gap", "ra_gap_after")):
        before = table[gap].abs().mean()
        rest = table[after].abs().mean()
        print(f"  {side}: mean |gap| {before:.3f} -> {rest:.3f} "
              f"({100 * (1 - rest / before) if before else 0:.0f}% closed)")

    print("\nThe clubs the validation doc names:")
    for club in NAMED:
        if club not in table.index:
            continue
        r = table.loc[club]
        print(f"  {club}: RA {r['ra_top']:.2f} top-down vs {r['ra_bot']:.2f} "
              f"bottom-up (gap {r['ra_gap']:+.2f}); park {r['ra_park']:+.2f}, "
              f"defence {r['ra_def']:+.2f}, left {r['ra_gap_after']:+.2f}. "
              f"RS {r['rs_top']:.2f} vs {r['rs_bot']:.2f} (gap "
              f"{r['rs_gap']:+.2f}); park {r['rs_park']:+.2f}, left "
              f"{r['rs_gap_after']:+.2f}.")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
