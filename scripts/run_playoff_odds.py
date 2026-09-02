"""Compute playoff odds for all 30 teams and write a dated JSON snapshot.

Reads live standings + schedule from the MLB Stats API (no credentials),
runs the season Monte Carlo + bracket, and writes:

    public/data/playoff_odds/YYYY-MM-DD.json   (never overwritten — roadmap 3.1)
    public/data/playoff_odds/latest.json

Station E's starting-pitcher term (`pythag_60_sp`, Brier .2448 vs the
production model's .2462 on the 756 market-priced 2026 games — see
docs/market-benchmark-2026.md) is applied to every *remaining* game whose
probable starters the Stats API has already posted. Probables go up two to
five days out, so a seven-day window catches all of them and most of the
window is empty; games without both probables keep team strength, which is
the correct rotation-average expectation for a game whose starters nobody
has announced.

The same term prices the **postseason bracket**: each club carries an ordered
rotation — its most-used starters this season, ranked by that same regressed
FIP, ace first — and every game of every series is priced off the arm whose
turn it is, cycling and wrapping after `--rotation-size` (default 4). A short
series is 3-7 games off four announced arms, so this is where knowing the
pitcher has the most room to move an answer.

`--no-starters` reverts to team strength everywhere, regular season and
bracket alike.

Usage:
    python scripts/run_playoff_odds.py --sims 20000
    python scripts/run_playoff_odds.py --sims 2000 --dry-run   # print only
    python scripts/run_playoff_odds.py --sims 5000 --dry-run --rotation-compare
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.mlb_stats_api import fetch_probables, fetch_schedule, fetch_standings
from src.sim import starters as sp_model
from src.sim.bracket import DEFAULT_ROTATION_SIZE, Rotations
from src.sim.odds import run_playoff_odds
from src.sim.season import SeasonState, from_schedule
from src.sim.strength import (
    estimate_hfa, home_win_prob, league_ra_per_game, regressed_run_rates,
    regressed_strength,
)
from src.sim.teams import DIVISION_NAMES, fetch_teams

OUT_DIR = Path(__file__).resolve().parent.parent / "public/data/playoff_odds"

BASE_METHOD = "Regressed Pythagenpat strength, log5 + HFA, MLB tiebreakers"
SP_METHOD = (BASE_METHOD + "; pythag_60_sp starting-pitcher term on remaining "
             "games with both probables posted, and on every postseason series "
             "game from each club\'s regressed-FIP rotation")
# Probables are posted ~2-5 days ahead, so a week covers every game that has
# them and costs one extra schedule call for the empty tail.
STARTER_WINDOW_DAYS = 7
# How many pitchers beyond the rotation to rank by FIP. Two is enough to let
# the rank pick around a swingman without reaching into September call-ups.
ROTATION_POOL_EXTRA = 2


def _rotation_candidates(probables: pd.DataFrame, schedule: pd.DataFrame,
                         as_of: date, pool: int) -> dict[int, list[int]]:
    """team_id -> the `pool` pitchers who have started most for it this season.

    `fetch_probables` serves the pitcher who *actually* started for a date in
    the past (see its docstring), so the season-to-date probables feed joined
    to the schedule's team ids is a start log. Only starts strictly before
    `as_of` count — the same date cut `starters.rate_table` applies to the
    rates — and only regular-season games, so an earlier postseason round
    cannot feed itself.

    Starts are counted for the team the pitcher started *for*, so a mid-season
    trade splits his season across both clubs; a pitcher acquired in July
    therefore looks shallower on his new team than he is. That is one of the
    first-pass simplifications listed in docs/playoff-odds-validation.md.
    """
    past = probables[(probables["game_type"] == "R")
                     & (probables["date"].astype(str) < as_of.isoformat())]
    past = past.merge(schedule[["game_pk", "home_id", "away_id"]], on="game_pk")
    sides = []
    for sp_col, team_col in (("home_sp_id", "home_id"), ("away_sp_id", "away_id")):
        part = past.dropna(subset=[sp_col])[[team_col, sp_col]]
        part.columns = ["team_id", "pitcher"]
        sides.append(part)
    starts = pd.concat(sides, ignore_index=True)
    if starts.empty:
        return {}
    starts = starts.astype({"team_id": "int64", "pitcher": "int64"})
    counts = (starts.groupby(["team_id", "pitcher"]).size()
              .rename("starts").reset_index()
              .sort_values(["team_id", "starts"], ascending=[True, False]))
    return {int(t): [int(x) for x in g["pitcher"].head(pool)]
            for t, g in counts.groupby("team_id")}


def build_rotations(candidates: dict[int, list[int]], sp_ra9: dict,
                    lg_ra9: float, index_of: dict[int, int],
                    run_env: np.ndarray, rotation_size: int) -> Rotations:
    """`bracket.Rotations` from the same rate table the game overrides use.

    Each club's candidate pool (the pitchers with the most starts for it this
    season) is ranked by regressed FIP and the best `rotation_size` become the
    rotation, ace first — which is how a postseason rotation is actually set,
    and the reason the pool is built by starts rather than by FIP alone: it
    keeps a reliever with 40 good innings out of Game 1. Pitchers absent from
    the rate table have no history and sit at league average, a zero delta.

    Deliberately a first pass. It does not know about openers, a starter left
    off the roster, a club that clinches early and lines its ace up on extra
    rest, or a Game 4 covered by the bullpen. See
    docs/playoff-odds-validation.md.
    """
    by_team: dict[int, list[tuple[int, float]]] = {}
    for team_id, pool in candidates.items():
        row = index_of.get(int(team_id))
        if row is None or not pool:
            continue
        ranked = sorted(pool, key=lambda pid: sp_ra9.get(int(pid), lg_ra9))
        by_team[row] = [(int(pid), float(sp_ra9.get(int(pid), lg_ra9)) - lg_ra9)
                        for pid in ranked[:rotation_size]]
    return Rotations(by_team=by_team, lg_ra9=lg_ra9, run_env=run_env)


def starter_terms(
    season: int, state: SeasonState, standings: pd.DataFrame,
    schedule: pd.DataFrame, hfa: float, regress_games: float, as_of: date,
    window_days: int = STARTER_WINDOW_DAYS,
    rotation_size: int = DEFAULT_ROTATION_SIZE, refresh: bool = True,
) -> tuple[dict[int, float], Rotations, dict]:
    """Both starter terms off **one** rate table: game overrides and rotations.

    Returns `({game_pk: P(home)}, Rotations, diagnostics)`.

    *Regular season.* Exactly the chain `scripts/backtest_game_odds.py` scores
    as `pythag_60_sp`, through the same functions: team RS/RA regressed the
    same 60 games the production strength uses -> each side's runs allowed
    moved by how far its announced starter's regressed FIP sits from league
    average over the 5.5 innings he covers -> Pythagenpat -> log5 -> HFA. The
    backtest walks it over a whole season and this walks it over one date;
    both call `starters.rate_table` and `starters.game_home_prob`, so the live
    number and the scored number cannot drift apart.

    *Postseason.* The bracket gets each club's rotation as ordered
    `(pitcher_id, RA/9 delta)` pairs, from the same rate table — the pool is
    the season's most-used starters, the order is regressed FIP. A short
    series is 3-7 games off four announced arms, which is where the term has
    the most room to move an answer.

    One `fetch_probables` call covers both — the season to date for the start
    log and the next `window_days` for the announcements — and one
    `starters.rate_inputs` fetch covers every pitcher either term names, so
    the two terms cannot be built on different rates.

    `refresh` re-pulls probables and the season's pitcher game logs rather
    than trusting caches keyed by season (see `starters.rate_inputs`).
    """
    end = as_of + timedelta(days=window_days)
    probables = fetch_probables(f"{season}-03-01", end.isoformat(), refresh=refresh)

    # Only games the simulator is actually still drawing. This drops spring
    # and postseason game types, anything already final, and — importantly —
    # tonight's games if they have started, since those leave `remaining`.
    remaining = {int(r.game_pk): (int(r.home_id), int(r.away_id))
                 for r in state.remaining.itertuples(index=False)}
    window = probables.dropna(subset=["home_sp_id", "away_sp_id"])
    window = window[window["game_pk"].astype(int).isin(remaining)]
    pmap = {int(r.game_pk): (int(r.home_sp_id), int(r.away_sp_id))
            for r in window.itertuples(index=False)}

    # The candidate pool is a couple deeper than the rotation so the FIP rank
    # has something to choose from without reaching into September call-ups.
    candidates = _rotation_candidates(probables, schedule, as_of,
                                      pool=rotation_size + ROTATION_POOL_EXTRA)

    team = regressed_run_rates(standings, regress_games=regress_games)
    lg_ra9 = league_ra_per_game(standings)
    pitcher_ids = {p for ids in pmap.values() for p in ids}
    pitcher_ids |= {p for pool in candidates.values() for p in pool}
    sp_ra9 = sp_model.build_rate_table(as_of.isoformat(), pitcher_ids, season,
                                       lg_ra9, refresh=refresh)

    diag = {"n_games_with_starters": 0, "starter_window_days": window_days,
            "sp_no_history": 0, "rotation_size": rotation_size,
            "n_teams_with_rotation": 0}

    overrides = {}
    for game_pk, sp_ids in pmap.items():
        home_id, away_id = remaining[game_pk]
        if home_id not in team.index or away_id not in team.index:
            continue
        p_home, no_history = sp_model.game_home_prob(
            team, home_id, away_id, sp_ids, sp_ra9, lg_ra9, hfa)
        overrides[game_pk] = p_home
        diag["sp_no_history"] += no_history
    diag["n_games_with_starters"] = len(overrides)

    run_env = ((team["rs_pg"] + team["ra_pg"])
               .reindex(state.team_ids).to_numpy(dtype=float))
    rotations = build_rotations(candidates, sp_ra9, lg_ra9, state.index_of(),
                                run_env, rotation_size)
    diag["n_teams_with_rotation"] = len(rotations.by_team)
    return overrides, rotations, diag


def rotation_table(rotations: Rotations, teams: pd.DataFrame) -> pd.DataFrame:
    """One row per club: its rotation, ace first, with each starter's RA/9 delta."""
    abbrev = teams["abbrev"].to_numpy()
    rows = []
    for row, rot in sorted(rotations.by_team.items(),
                           key=lambda kv: kv[1][0][1] if kv[1] else 0.0):
        entry = {"abbrev": abbrev[row]}
        for i, (pid, delta) in enumerate(rot, start=1):
            entry[f"g{i}_sp"] = pid
            entry[f"g{i}_ra9d"] = round(delta, 3)
        rows.append(entry)
    return pd.DataFrame(rows)


def override_effect(state: SeasonState, strength: pd.Series, hfa: float,
                    overrides: dict[int, float]) -> pd.DataFrame:
    """Per-game Δ P(home), and the expected-wins shift it implies per team.

    The with/without odds tables below are two Monte Carlo runs, and in
    September the term's effect is far smaller than the sampling noise of even
    a 20k-sim bracket (see docs/playoff-odds-validation.md). This is the same
    comparison done in closed form: no sampling, so it is the term's actual
    footprint on the season. Expected wins are additive in the per-game
    probabilities, so a team's shift is just the sum of the Δ on the games it
    hosts minus the Δ on the games it visits.
    """
    rem = state.remaining
    base = pd.Series(
        home_win_prob(strength.reindex(rem["home_id"]).to_numpy(),
                      strength.reindex(rem["away_id"]).to_numpy(), hfa=hfa),
        index=rem["game_pk"].astype(int).to_numpy())
    rows = []
    for r in rem.itertuples(index=False):
        pk = int(r.game_pk)
        if pk not in overrides:
            continue
        rows.append({"game_pk": pk, "date": r.date, "home_id": int(r.home_id),
                     "away_id": int(r.away_id), "p_base": float(base[pk]),
                     "p_sp": float(overrides[pk]),
                     "delta": float(overrides[pk] - base[pk])})
    return pd.DataFrame(rows)


def expected_win_shift(effect: pd.DataFrame, teams: pd.DataFrame) -> pd.Series:
    """abbrev → change in expected wins from the overridden games (no sampling)."""
    abbrev = teams.set_index("team_id")["abbrev"]
    shift = pd.Series(0.0, index=abbrev.to_numpy())
    for r in effect.itertuples(index=False):
        shift[abbrev[r.home_id]] += r.delta
        shift[abbrev[r.away_id]] -= r.delta
    return shift.sort_values(key=lambda s: s.abs(), ascending=False)


def format_odds(odds: pd.DataFrame) -> pd.DataFrame:
    show = odds[["abbrev", "division", "wins", "losses", "strength", "mean_wins",
                 "p_playoffs", "p_division", "p_bye", "p_pennant", "p_ws"]].copy()
    for c in ("p_playoffs", "p_division", "p_bye", "p_pennant", "p_ws"):
        show[c] = (show[c] * 100).round(1)
    show["strength"] = show["strength"].round(3)
    show["mean_wins"] = show["mean_wins"].round(1)
    return show


COMPARE_PROBS = ("p_playoffs", "p_pennant", "p_ws")


def compare(base: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    """Per-team with/without table: the same seed, so every difference is the term.

    P(pennant) is carried alongside P(playoffs) and P(WS) because the bracket
    term moves the pennant round first — a rotation cannot change who makes
    October, only who survives it.
    """
    keep = ["mean_wins", *COMPARE_PROBS]
    cols = ["abbrev", "division", "wins", "losses", *keep]
    cmp = base[cols].merge(sp[["abbrev", *keep]], on="abbrev",
                           suffixes=("", "_sp"))
    for c in keep:
        cmp[f"d_{c}"] = cmp[f"{c}_sp"] - cmp[c]
    # Probabilities in percentage points, wins in wins; two decimals either way
    # because the differences live in the second one.
    for c in COMPARE_PROBS:
        for name in (c, f"{c}_sp", f"d_{c}"):
            cmp[name] = (cmp[name] * 100).round(2)
    for c in ("mean_wins", "mean_wins_sp", "d_mean_wins"):
        cmp[c] = cmp[c].round(2)
    return cmp.sort_values("p_ws_sp", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--sims", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: day-of-year, so reruns on a "
                             "day reproduce)")
    parser.add_argument("--regress-games", type=float, default=60.0)
    parser.add_argument("--no-starters", action="store_true",
                        help="ignore posted probables and postseason "
                             "rotations; price every remaining game and every "
                             "series game off team strength alone (the "
                             "pre-station-E production model)")
    parser.add_argument("--starter-window-days", type=int, default=STARTER_WINDOW_DAYS,
                        help="how far ahead to look for posted probables")
    parser.add_argument("--rotation-size", type=int, default=DEFAULT_ROTATION_SIZE,
                        help="how many starters a postseason rotation carries "
                             "before it wraps (game 1 = the club's best by "
                             "regressed FIP)")
    parser.add_argument("--rotation-compare", action="store_true",
                        help="also run the bracket with the rotations removed "
                             "and print the difference, isolating the "
                             "postseason term from the regular-season one "
                             "(one extra Monte Carlo pass)")
    parser.add_argument("--cached-pitchers", action="store_true",
                        help="reuse cached probables and pitcher game logs "
                             "instead of re-pulling them. Faster for repeated "
                             "local runs; the nightly job must NOT use it, "
                             "because those caches are keyed by season and go "
                             "stale the moment another game is played")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    today = date.today()
    seed = args.seed if args.seed is not None else today.timetuple().tm_yday

    teams = fetch_teams(args.season)
    standings = fetch_standings(args.season)
    schedule = fetch_schedule(f"{args.season}-03-01", f"{args.season}-10-15")
    state = from_schedule(schedule, teams)
    strength = regressed_strength(standings, regress_games=args.regress_games)
    hfa = estimate_hfa(state.completed)
    print(f"{len(state.completed)} games played, {len(state.remaining)} remaining, "
          f"HFA={hfa:.4f}, sims={args.sims}, seed={seed}")

    overrides: dict[int, float] = {}
    rotations: Rotations | None = None
    diag = {"n_games_with_starters": 0, "sp_no_history": 0,
            "starter_window_days": args.starter_window_days,
            "rotation_size": args.rotation_size, "n_teams_with_rotation": 0}
    if not args.no_starters:
        try:
            overrides, rotations, diag = starter_terms(
                args.season, state, standings, schedule, hfa,
                args.regress_games, today,
                window_days=args.starter_window_days,
                rotation_size=args.rotation_size,
                refresh=not args.cached_pitchers)
        except Exception as exc:   # noqa: BLE001 — the nightly job must still ship odds
            overrides, rotations = {}, None
            print(f"WARNING: starting-pitcher term unavailable ({exc!r}); "
                  f"falling back to team strength on every remaining game "
                  f"and on every postseason series")
        else:
            print(f"starters: {diag['n_games_with_starters']} of "
                  f"{len(state.remaining)} remaining games have both probables "
                  f"posted in the next {args.starter_window_days} days "
                  f"({diag['sp_no_history']} starter slots had no history, "
                  f"scored at league average)")
            print(f"rotations: {diag['n_teams_with_rotation']} clubs carry a "
                  f"{args.rotation_size}-man postseason rotation, ace first "
                  f"(RA/9 delta from league; negative is better)")
            print(rotation_table(rotations, state.teams).to_string(index=False))

    has_rotations = rotations is not None and bool(rotations.by_team)

    # Same seed everywhere, so the draw is identical on every game and every
    # series the terms do not touch and the whole difference in the tables
    # below is the term.
    base = run_playoff_odds(state, strength, hfa, n_sims=args.sims, seed=seed)
    base["division"] = base["division_id"].map(DIVISION_NAMES)
    if overrides or has_rotations:
        odds = run_playoff_odds(state, strength, hfa, n_sims=args.sims, seed=seed,
                                p_home_overrides=overrides or None,
                                rotations=rotations if has_rotations else None)
        odds["division"] = odds["division_id"].map(DIVISION_NAMES)
    else:
        odds = base

    print(f"\n─── without the starter term (team strength on all "
          f"{len(state.remaining)} remaining games and every postseason "
          f"series) ───")
    print(format_odds(base).to_string(index=False))

    if odds is not base:
        print(f"\n─── with the starter term: {len(overrides)} regular-season "
              f"games repriced, {diag['n_teams_with_rotation']} postseason "
              f"rotations ───")
        print(format_odds(odds).to_string(index=False))
        cmp = compare(base, odds)
        print("\n─── with vs without, same seed (percentage points) ───")
        print(cmp.to_string(index=False))
        for col, label in (("d_p_playoffs", "p_playoffs"),
                           ("d_p_pennant", "p_pennant"), ("d_p_ws", "p_ws"),
                           ("d_mean_wins", "mean wins")):
            print(f"max |Δ {label}| = {cmp[col].abs().max():.2f} "
                  f"({cmp.loc[cmp[col].abs().idxmax(), 'abbrev']})")

    # The bracket term on its own: the same run with the rotations removed, so
    # the regular-season overrides are held fixed and the only thing that moves
    # is how the postseason series are priced.
    if args.rotation_compare and has_rotations:
        flat = run_playoff_odds(state, strength, hfa, n_sims=args.sims, seed=seed,
                                p_home_overrides=overrides or None)
        flat["division"] = flat["division_id"].map(DIVISION_NAMES)
        rot_cmp = compare(flat, odds)
        print("\n─── postseason rotations only: bracket on team strength vs "
              "on the rotation, same seed (percentage points) ───")
        print(rot_cmp.to_string(index=False))
        for col, label in (("d_p_pennant", "p_pennant"), ("d_p_ws", "p_ws")):
            print(f"max |Δ {label}| = {rot_cmp[col].abs().max():.2f} "
                  f"({rot_cmp.loc[rot_cmp[col].abs().idxmax(), 'abbrev']})")
        movers = rot_cmp.reindex(rot_cmp["d_p_pennant"].abs()
                                 .sort_values(ascending=False).index).head(5)
        print("\ntop 5 movers on P(pennant), with their game-1/game-2 starters:")
        abbrev_row = {a: i for i, a in enumerate(state.teams["abbrev"])}
        for r in movers.itertuples(index=False):
            rot = rotations.by_team.get(abbrev_row[r.abbrev], [])
            arms = ", ".join(f"g{i}: {pid} {d:+.3f}"
                             for i, (pid, d) in enumerate(rot[:2], start=1))
            print(f"  {r.abbrev}: Δp_pennant {r.d_p_pennant:+.2f} pts, "
                  f"Δp_ws {r.d_p_ws:+.2f} pts  [{arms}]")

    if overrides:
        # The odds tables above are Monte Carlo runs; in September their
        # difference is smaller than the sim noise. This part is exact.
        effect = override_effect(state, strength, hfa, overrides)
        print(f"\n─── the regular-season term itself, in closed form (no "
              f"sampling) ───")
        print(f"{len(effect)} games repriced: mean |Δ P(home)| = "
              f"{effect['delta'].abs().mean():.4f}, max = "
              f"{effect['delta'].abs().max():.4f}")
        shift = expected_win_shift(effect, state.teams)
        print("largest expected-win shifts over those games:")
        print("  " + ", ".join(f"{t} {v:+.3f}" for t, v in shift.head(6).items()))

    if args.dry_run:
        return

    payload = {
        "season": args.season,
        "as_of": today.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_sims": args.sims,
        "seed": seed,
        "hfa": hfa,
        "games_played": int(len(state.completed)),
        "games_remaining": int(len(state.remaining)),
        "n_games_with_starters": int(diag["n_games_with_starters"]),
        "starter_window_days": int(diag["starter_window_days"]),
        "n_teams_with_rotation": int(diag["n_teams_with_rotation"]),
        "rotation_size": int(diag["rotation_size"]) if has_rotations else 0,
        "method": (SP_METHOD if odds is not base else BASE_METHOD),
        "teams": json.loads(odds.drop(columns=["division_id"]).to_json(orient="records")),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dated = args.out_dir / f"{today.isoformat()}.json"
    if dated.exists():
        print(f"snapshot {dated.name} already exists; not overwriting (updating latest.json only)")
    else:
        dated.write_text(json.dumps(payload, indent=1))
        print(f"wrote {dated}")
    (args.out_dir / "latest.json").write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out_dir / 'latest.json'}")


if __name__ == "__main__":
    main()
