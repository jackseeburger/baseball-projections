"""Station B-pitchers — score projected batters faced against its baselines.

    # the headline: every method, every cutoff, 2022-2026
    python3 scripts/run_pitcher_workload_backtest.py

    # only the seasons the constants were *not* chosen on
    python3 scripts/run_pitcher_workload_backtest.py --seasons 2024 2025 2026

    # innings instead of batters faced
    python3 scripts/run_pitcher_workload_backtest.py --unit outs

    # the selection sweep the blend weight was chosen from (2022-2023 only)
    python3 scripts/run_pitcher_workload_backtest.py --sweep

    # per-role calibration constants, chosen on the same two seasons
    python3 scripts/run_pitcher_workload_backtest.py --calibrate

Inputs come from `data/workload/` (`scripts/build_pitcher_workload.py`), the
model from `src/projections/pitcher_workload.py`.

**The gate.** Whatever wins has to win out of sample, on the common pitcher
set, against baselines that were given the same information. The constants
the candidate methods carry were chosen on 2022 and 2023 and frozen; 2024,
2025 and 2026 are the score. Both halves are printed, and `--seasons` is how
the holdout is read on its own.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.projections import il_returns
from src.projections import pitcher_workload as W
from build_pitcher_workload import SCORE_SEASONS, WORKLOAD_DIR, cutoffs_for

# The two seasons every constant in `pitcher_workload` was chosen on. Nothing
# about the other three enters a fit, which is what makes them the gate.
FIT_SEASONS = (2022, 2023)
HOLDOUT_SEASONS = (2024, 2025, 2026)

# The injured-list return-time distribution for a season is fitted on the
# three seasons before it, minus 2020 — the same rule station B uses, and for
# the same reason: a 60-game season censors every spell at a length no other
# year has.
IL_FIT_SEASONS = 3
IL_EXCLUDE = (2020,)

logger = logging.getLogger("pitcher_workload_backtest")


def il_fit_seasons(season: int) -> tuple[int, ...]:
    return tuple(s for s in range(season - IL_FIT_SEASONS, season)
                 if s not in IL_EXCLUDE)


def _read(kind: str, season: int) -> pd.DataFrame:
    path = WORKLOAD_DIR / f"{kind}_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run scripts/build_pitcher_workload.py --fetch "
            f"--season {season}")
    return pd.read_parquet(path)


# --- assembling a season ------------------------------------------------

def season_frames(season: int) -> dict:
    """Every frame the season's cutoffs need, plus its return-time table."""
    appearances = _read("pitcher_appearances", season)
    team_games = _read("team_games", season)
    schedule = _read("schedule", season)
    rosters = _read("pitcher_rosters", season)
    score_end = str(pd.to_datetime(team_games["date"]).max().date())

    prior = season - 1
    prior_path = WORKLOAD_DIR / f"pitcher_appearances_{prior}.parquet"
    if prior_path.exists():
        prior_totals = (pd.read_parquet(prior_path)
                        .groupby("pitcher", as_index=False)[["bf", "outs"]].sum())
    else:
        prior_totals = pd.DataFrame(columns=["pitcher", "bf", "outs"])

    # The return-time distribution: fitted on whole seasons *before* this one,
    # read at each cutoff against spells this season's transactions opened
    # strictly before it.
    fit = il_fit_seasons(season)
    frames, ends = [], {}
    for s in fit:
        path = WORKLOAD_DIR / f"transactions_{s}.parquet"
        if not path.exists():
            continue
        frames.append(pd.read_parquet(path))
        games = _read("team_games", s) if (WORKLOAD_DIR / f"team_games_{s}.parquet").exists() else None
        ends[s] = (str(pd.to_datetime(games["date"]).max().date()) if games is not None
                   else f"{s}-10-05")
    table = (il_returns.fit(pd.concat(frames, ignore_index=True), ends)
             if frames else pd.DataFrame(columns=il_returns.SURVIVAL_COLUMNS))
    events = il_returns.parse_events(_read("transactions", season))

    return {"season": season, "score_end": score_end, "appearances": appearances,
            "team_games": team_games, "schedule": schedule, "rosters": rosters,
            "prior_totals": prior_totals, "il_table": table, "events": events,
            "il_fit_seasons": fit}


def cutoff_inputs(frames: dict, cutoff: str) -> W.CutoffInputs:
    """One cutoff's inputs, including who is out and for how much longer."""
    roster = frames["rosters"]
    roster = roster[roster["cutoff"] == cutoff].reset_index(drop=True)
    score_end = frames["score_end"]
    horizon_days = (pd.Timestamp(score_end) - pd.Timestamp(cutoff)).days

    # `expected_active_fractions` speaks station B's column name; the pitchers
    # are the same players to the transaction feed.
    as_batters = roster.rename(columns={"pitcher": "batter"})
    fractions = il_returns.expected_active_fractions(
        as_batters, frames["events"], frames["il_table"], cutoff, horizon_days)
    active = (fractions.set_index(fractions["batter"].astype("int64"))
              ["active_fraction"].astype(float) if len(fractions)
              else pd.Series(dtype=float))
    open_now = il_returns.open_spells_at(frames["events"], cutoff)
    spell_start = (open_now.sort_values("start")
                   .drop_duplicates("player_id", keep="first")
                   .set_index("player_id")["start"] if len(open_now)
                   else pd.Series(dtype="datetime64[ns]"))

    return W.CutoffInputs(
        cutoff=cutoff, score_end=score_end,
        appearances=frames["appearances"], team_games=frames["team_games"],
        schedule=frames["schedule"], roster=roster,
        prior_totals=frames["prior_totals"],
        active_fraction=active, spell_start=spell_start)


def walk_forward(frames: dict, unit: str = "bf", methods=W.METHODS,
                 blend_weights=None, calibration: dict | None = None,
                 sweep_method: str = "blend"):
    """Yield `(cutoff, {method: projection}, actual, universe)` per cutoff.

    The universe is shared by every method at a cutoff — the union of everyone
    who faced a batter in the scored window and everyone any method projects
    above zero — so nobody is rewarded for declining to project someone.
    """
    season, score_end = frames["season"], frames["score_end"]
    for cutoff in cutoffs_for(season, score_end):
        if cutoff not in set(frames["rosters"]["cutoff"]):
            continue
        inputs = cutoff_inputs(frames, cutoff)
        actual = W.realized(frames["appearances"], cutoff, score_end, unit)
        projections = {}
        for m in methods:
            kwargs = {"calibration": calibration} if (
                calibration and m.startswith("blend")) else {}
            projections[m] = W.project(inputs, m, unit=unit, **kwargs)
        for w in (blend_weights or ()):
            projections[f"{sweep_method}@{w:g}"] = W.project(
                inputs, sweep_method, unit=unit, blend_weight=w)
        universe = sorted(
            set(actual["pitcher"].astype(int))
            | {int(p) for proj in projections.values()
               for p in proj.loc[proj["projected"] > 0, "pitcher"]})
        yield cutoff, projections, actual, universe


def observed_roles(frames: dict, cutoff: str, unit: str = "bf") -> pd.Series:
    """Each pitcher's role at the cutoff, from his starts before it.

    Scoring is split on this rather than on each method's own role call, so
    every method's starters are the same pitchers.
    """
    season = W.window_totals(frames["appearances"], cutoff, unit, None)
    window = W.window_totals(frames["appearances"], cutoff, unit,
                             W.ROLE_WINDOW_DAYS)
    if not len(season):
        return pd.Series(dtype=object)
    roles = W._roles_for(season.loc[:, ["pitcher"]], window, season)
    return pd.Series(roles, index=season["pitcher"].astype(int))


# --- the tables ---------------------------------------------------------

def score(seasons, unit: str = "bf", methods=W.METHODS,
          calibration: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The headline table and the paired tests, over several seasons."""
    rows, paired_rows = [], []
    for season in seasons:
        frames = season_frames(season)
        for cutoff, projections, actual, universe in walk_forward(
                frames, unit=unit, methods=methods, calibration=calibration):
            roles = observed_roles(frames, cutoff, unit)
            horizon = (pd.Timestamp(frames["score_end"]) - pd.Timestamp(cutoff)).days
            left = float(W.games_remaining(frames["schedule"], cutoff,
                                           frames["score_end"]).median())
            errors = {m: W.absolute_errors(p, actual, universe=universe)
                      for m, p in projections.items()}
            for m, proj in projections.items():
                rows.append({
                    "season": season, "cutoff": cutoff,
                    "horizon_days": horizon, "games_left": left, "method": m,
                    **W.score_projection(proj, actual, universe=universe,
                                         roles=roles),
                    "top5_capture": W.top_n_capture(proj, actual, 5, universe),
                })
            for focus in methods:
                for other in methods:
                    if focus == other:
                        continue
                    d = W.paired_difference(errors[focus], errors[other])
                    paired_rows.append({
                        "season": season, "cutoff": cutoff,
                        "horizon_days": horizon, "games_left": left,
                        "method": focus, "vs": other, **d})
            # The same paired test, split by role.
            for role in ("SP", "RP"):
                members = set(roles[roles == role].index)
                for focus in methods:
                    for other in methods:
                        if focus == other:
                            continue
                        a = errors[focus][errors[focus].index.isin(members)]
                        b = errors[other][errors[other].index.isin(members)]
                        d = W.paired_difference(a, b)
                        paired_rows.append({
                            "season": season, "cutoff": cutoff,
                            "horizon_days": horizon, "games_left": left,
                            "method": focus, "vs": other, "role": role, **d})
    table = pd.DataFrame(rows)
    paired = pd.DataFrame(paired_rows)
    if "role" not in paired.columns:
        paired["role"] = pd.NA
    paired["role"] = paired["role"].fillna("all")
    return table, paired


def pooled(paired: pd.DataFrame, role: str = "all") -> pd.DataFrame:
    """Pool the per-cutoff paired differences into one row per comparison.

    Cutoffs inside a season overlap heavily — the same pitcher is scored nine
    times — so the pitcher-level standard error would be far too small. This
    treats each **cutoff** as the unit: the mean of the per-cutoff mean
    differences, with the standard error taken across cutoffs and clustered by
    season, which is the same convention station G's backtest uses.
    """
    rows = []
    sub = paired[paired["role"] == role]
    for (method, vs), group in sub.groupby(["method", "vs"]):
        by_season = group.groupby("season")["mean"].mean()
        n_seasons = len(by_season)
        mean = float(group["mean"].mean())
        se = (float(by_season.std(ddof=1) / np.sqrt(n_seasons))
              if n_seasons > 1 else float("nan"))
        rows.append({"method": method, "vs": vs, "role": role,
                     "cutoffs": int(len(group)), "seasons": n_seasons,
                     "n_pitchers": int(group["n"].sum()),
                     "mean_mae_diff": mean, "se": se,
                     "t": mean / se if se else float("nan")})
    return pd.DataFrame(rows).sort_values(["method", "vs"])


def by_horizon(paired: pd.DataFrame, method: str, vs: str,
               role: str = "all") -> pd.DataFrame:
    """The same comparison, one row per cutoff month, so the horizon shows."""
    sub = paired[(paired["method"] == method) & (paired["vs"] == vs)
                 & (paired["role"] == role)].copy()
    sub["month"] = pd.to_datetime(sub["cutoff"]).dt.strftime("%m-%d")
    rows = []
    for month, group in sub.groupby("month"):
        n = len(group)
        mean = float(group["mean"].mean())
        se = float(group["mean"].std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        rows.append({"as_of": month, "seasons": n,
                     "games_left": float(group["games_left"].mean()),
                     "n_pitchers": int(group["n"].sum()),
                     "mean_mae_diff": mean, "se": se,
                     "t": mean / se if se else float("nan")})
    return pd.DataFrame(rows)


# --- selection (2022-2023 only) -----------------------------------------

SWEEP_WEIGHTS = tuple(i / 10 for i in range(11))


def sweep(seasons=FIT_SEASONS, unit: str = "bf",
          method: str = "blend") -> pd.DataFrame:
    """MAE against a constant blend weight at each fitting-season cutoff."""
    rows = []
    for season in seasons:
        frames = season_frames(season)
        for cutoff, projections, actual, universe in walk_forward(
                frames, unit=unit, methods=(method,),
                blend_weights=SWEEP_WEIGHTS, sweep_method=method):
            left = float(W.games_remaining(frames["schedule"], cutoff,
                                           frames["score_end"]).median())
            for name, proj in projections.items():
                if "@" not in name:
                    continue
                rows.append({
                    "season": season, "cutoff": cutoff, "games_left": left,
                    "w": float(name.split("@")[1]),
                    "mae": W.score_projection(proj, actual,
                                              universe=universe)["mae"]})
    return pd.DataFrame(rows)


def fit_logistic(curve: pd.DataFrame) -> tuple[float, float, float]:
    """Grid-search the two anchor weights against the swept per-cutoff curves.

    The objective is the mean over cutoffs of `MAE(w(h)) / min_w MAE(w)` —
    each cutoff's MAE measured against the best it could have done with a
    constant weight. Mean raw MAE would be a fit to the longest cutoff alone,
    because MAE scales with the horizon, and the horizon dependence is
    precisely what is being estimated. Station B's `--sweep` does the same.
    """
    curves, horizons = {}, {}
    for (season, cutoff), group in curve.groupby(["season", "cutoff"]):
        curves[(season, cutoff)] = group.set_index("w")["mae"].sort_index()
        horizons[(season, cutoff)] = float(group["games_left"].iloc[0])
    grid = np.arange(0.05, 0.96, 0.01)
    best = (None, None, float("inf"))
    for w_short in grid:
        for w_long in grid:
            if w_long > w_short - 0.01:
                continue
            midpoint, scale = W.logistic_from_anchors(float(w_short), float(w_long))
            excess = []
            for key, c in curves.items():
                w = float(W.horizon_weight(horizons[key], midpoint, scale))
                mae = float(np.interp(w, c.index.to_numpy(float), c.to_numpy(float)))
                excess.append(mae / float(c.min()))
            loss = float(np.mean(excess))
            if loss < best[2]:
                best = (float(w_short), float(w_long), loss)
    return best


def calibrate(seasons=FIT_SEASONS, unit: str = "bf", method: str = "blend",
              grid=np.arange(0.50, 1.51, 0.01)) -> dict:
    """The per-role multiplier that minimizes MAE on the fitting seasons.

    Rest-of-season workload has a long left tail — a pitcher who gets hurt in
    August faces nobody — and MAE is minimized at a conditional *median*, not
    a mean. A projection built as an expectation is therefore systematically
    too high for the metric it is scored on, and one constant per role is the
    smallest correction that can say so. It is chosen here and nowhere else.
    """
    pieces = []
    for season in seasons:
        frames = season_frames(season)
        for cutoff, projections, actual, universe in walk_forward(
                frames, unit=unit, methods=(method,)):
            # The role here is the *method's own* call, because that is the
            # key the constant will be applied by in production.
            pieces.append(W._aligned(projections[method], actual,
                                     universe=universe))
    allrows = pd.concat(pieces, ignore_index=True)
    out = {}
    for role in ("SP", "RP"):
        sub = allrows[allrows["role"] == role]
        p = sub["projected"].to_numpy(float)
        y = sub["realized"].to_numpy(float)
        maes = [np.abs(c * p - y).mean() for c in grid]
        out[role] = float(grid[int(np.argmin(maes))])
    return out


def calibrate_hazard(seasons=FIT_SEASONS, unit: str = "bf",
                     grid=np.arange(0.0, 0.0121, 0.0005)) -> dict:
    """The per-role attrition hazard that minimizes MAE on the fitting seasons.

    A flat per-role multiplier (`--calibrate`) says the served projection is
    about a tenth too high; the by-horizon table says the excess is four
    batters faced a pitcher in May and nothing at all in August. That is not a
    level error, it is a survival one — the model gives a healthy pitcher his
    turn every fifth day until October and some of them lose the season in
    July. One hazard per role is the smallest thing with the right shape.

    Two parameters, and the grid is coarse on purpose: at 0.0005 per club game
    the finest step moves a three-month projection by about 3%, which is
    already below the resolution the holdout can see.
    """
    projected, realized, roles, horizons = [], [], [], []
    for season in seasons:
        frames = season_frames(season)
        for cutoff, projections, actual, universe in walk_forward(
                frames, unit=unit, methods=("structural",)):
            proj = projections["structural"]
            left = W.games_remaining(frames["schedule"], cutoff,
                                     frames["score_end"])
            df = W._aligned(proj, actual, universe=universe)
            horizon = (proj.drop_duplicates("pitcher").set_index("pitcher")
                       ["team_id"].map(left))
            projected.append(df["projected"].to_numpy(float))
            realized.append(df["realized"].to_numpy(float))
            roles.append(df["role"].fillna("RP").to_numpy())
            horizons.append(df["pitcher"].map(horizon).fillna(0.0).to_numpy(float))
    p = np.concatenate(projected)
    y = np.concatenate(realized)
    r = np.concatenate(roles)
    h = np.concatenate(horizons)
    out = {}
    for role in ("SP", "RP"):
        m = r == role
        maes = [np.abs(p[m] * W.attrition_fraction(h[m], lam) - y[m]).mean()
                for lam in grid]
        out[role] = float(grid[int(np.argmin(maes))])
    return out


# --- the report ---------------------------------------------------------

def _print_headline(table: pd.DataFrame, unit: str) -> None:
    label = "batters faced" if unit == "bf" else "outs"
    print(f"\n=== Pooled over every cutoff, {label} ===")
    agg = (table.groupby("method")
           .agg(cutoffs=("mae", "size"), n=("n", "sum"), mae=("mae", "mean"),
                rmse=("rmse", "mean"), wmae=("weighted_mae", "mean"),
                bias=("bias", "mean"), sp_mae=("sp_mae", "mean"),
                rp_mae=("rp_mae", "mean"), top5=("top5_capture", "mean"))
           .sort_values("mae"))
    print(agg.round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="*", default=list(SCORE_SEASONS))
    ap.add_argument("--unit", choices=("bf", "outs"), default="bf")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calibrate-hazard", action="store_true")
    ap.add_argument("--method", default="blend",
                    help="which method --sweep and --calibrate operate on")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--csv-out", type=Path)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.sweep:
        curve = sweep(unit=args.unit, method=args.method)
        grid = curve.pivot_table(index="w", columns=["season", "cutoff"],
                                 values="mae")
        print(f"\nSelection sweep on {FIT_SEASONS} ({args.unit}): MAE against a "
              f"constant blend weight.")
        print(grid.mean(axis=1).round(3).to_string())
        best = (curve.groupby(["season", "cutoff", "games_left"])
                .apply(lambda g: g.loc[g["mae"].idxmin(), "w"],
                       include_groups=False)
                .rename("best_w").reset_index())
        print("\nBest constant weight per cutoff:")
        print(best.round(2).to_string(index=False))
        w_short, w_long, loss = fit_logistic(curve)
        h_short, h_long = W.BLEND_ANCHOR_GAMES
        print(f"\nLogistic fit on {FIT_SEASONS} only: "
              f"w({h_short:.0f} games)={w_short:.2f}, w({h_long:.0f})={w_long:.2f}; "
              f"mean excess over each cutoff's own best weight "
              f"{100 * (loss - 1):.2f}%")
        print(f"  in use: {W.BLEND_WEIGHT_SHORT} / {W.BLEND_WEIGHT_LONG}")
        return

    if args.calibrate_hazard:
        print(f"\nPer-role attrition hazard on {FIT_SEASONS} ({args.unit}): "
              f"{calibrate_hazard(unit=args.unit)}")
        return

    if args.calibrate:
        print(f"\nPer-role calibration of {args.method} on {FIT_SEASONS} "
              f"({args.unit}): {calibrate(unit=args.unit, method=args.method)}")
        return

    table, paired = score(args.seasons, unit=args.unit)
    _print_headline(table, args.unit)

    print("\n=== Per season ===")
    print(table.groupby(["season", "method"])["mae"].mean()
          .unstack("method").round(2).to_string())

    print("\n=== Paired per-pitcher MAE difference, pooled (negative = the "
          "method is better; SE clustered by season) ===")
    for role in ("all", "SP", "RP"):
        p = pooled(paired, role)
        keep = p[p["method"].isin(("structural", "blend_il", "blend_il_share",
                                   "blend"))]
        print(f"\n-- {role} --")
        print(keep.round(3).to_string(index=False))

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv_out, index=False)
        paired.to_csv(args.csv_out.with_name(
            args.csv_out.stem + "_paired.csv"), index=False)
        print(f"\nwrote {args.csv_out}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "unit": args.unit, "seasons": list(args.seasons),
            "table": table.to_dict("records"),
            "pooled": pooled(paired).to_dict("records"),
        }, indent=2, default=float))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
