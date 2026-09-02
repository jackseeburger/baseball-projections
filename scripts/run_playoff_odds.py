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
has announced. `--no-starters` reverts to team strength everywhere.

Usage:
    python scripts/run_playoff_odds.py --sims 20000
    python scripts/run_playoff_odds.py --sims 2000 --dry-run   # print only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.mlb_stats_api import fetch_probables, fetch_schedule, fetch_standings
from src.sim import starters as sp_model
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
             "games with both probables posted")
# Probables are posted ~2-5 days ahead, so a week covers every game that has
# them and costs one extra schedule call for the empty tail.
STARTER_WINDOW_DAYS = 7


def starter_overrides(
    season: int, state: SeasonState, standings: pd.DataFrame, hfa: float,
    regress_games: float, as_of: date, window_days: int = STARTER_WINDOW_DAYS,
    refresh: bool = True,
) -> tuple[dict[int, float], dict]:
    """{game_pk: P(home)} for remaining games with both starters announced.

    Runs exactly the chain `scripts/backtest_game_odds.py` scores as
    `pythag_60_sp`, through the same functions: team RS/RA regressed the same
    60 games the production strength uses → each side's runs allowed moved by
    how far its announced starter's regressed FIP sits from league average
    over the 5.5 innings he covers → Pythagenpat → log5 → HFA. The backtest
    walks it over a whole season and this walks it over one date; both call
    `starters.rate_table` and `starters.game_home_prob`, so the live number and
    the scored number cannot drift apart.

    `refresh` re-pulls today's probables and the season's pitcher game logs
    rather than trusting caches keyed by season (see `starters.rate_inputs`).

    Returns (overrides, diagnostics).
    """
    end = as_of + timedelta(days=window_days)
    probables = fetch_probables(as_of.isoformat(), end.isoformat(), refresh=refresh)
    probables = probables.dropna(subset=["home_sp_id", "away_sp_id"])

    # Only games the simulator is actually still drawing. This drops spring
    # and postseason game types, anything already final, and — importantly —
    # tonight's games if they have started, since those leave `remaining`.
    remaining = {int(r.game_pk): (int(r.home_id), int(r.away_id))
                 for r in state.remaining.itertuples(index=False)}
    probables = probables[probables["game_pk"].astype(int).isin(remaining)]
    diag = {"n_games_with_starters": 0, "starter_window_days": window_days,
            "sp_no_history": 0}
    if probables.empty:
        return {}, diag

    pmap = {int(r.game_pk): (int(r.home_sp_id), int(r.away_sp_id))
            for r in probables.itertuples(index=False)}
    team = regressed_run_rates(standings, regress_games=regress_games)
    lg_ra9 = league_ra_per_game(standings)
    sp_ra9 = sp_model.build_rate_table(
        as_of.isoformat(), {p for ids in pmap.values() for p in ids},
        season, lg_ra9, refresh=refresh)

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
    return overrides, diag


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


def compare(base: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    """Per-team with/without table: the same seed, so every difference is the term."""
    cols = ["abbrev", "division", "wins", "losses", "mean_wins", "p_playoffs", "p_ws"]
    cmp = base[cols].merge(sp[["abbrev", "mean_wins", "p_playoffs", "p_ws"]],
                           on="abbrev", suffixes=("", "_sp"))
    for c in ("mean_wins", "p_playoffs", "p_ws"):
        cmp[f"d_{c}"] = cmp[f"{c}_sp"] - cmp[c]
    # Probabilities in percentage points, wins in wins; two decimals either way
    # because the differences live in the second one.
    for c in ("p_playoffs", "p_playoffs_sp", "d_p_playoffs",
              "p_ws", "p_ws_sp", "d_p_ws"):
        cmp[c] = (cmp[c] * 100).round(2)
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
                        help="ignore posted probables; price every remaining "
                             "game off team strength alone (the pre-station-E "
                             "production model)")
    parser.add_argument("--starter-window-days", type=int, default=STARTER_WINDOW_DAYS,
                        help="how far ahead to look for posted probables")
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
    diag = {"n_games_with_starters": 0, "sp_no_history": 0,
            "starter_window_days": args.starter_window_days}
    if not args.no_starters:
        try:
            overrides, diag = starter_overrides(
                args.season, state, standings, hfa, args.regress_games, today,
                window_days=args.starter_window_days,
                refresh=not args.cached_pitchers)
        except Exception as exc:   # noqa: BLE001 — the nightly job must still ship odds
            print(f"WARNING: starting-pitcher term unavailable ({exc!r}); "
                  f"falling back to team strength on every remaining game")
        else:
            print(f"starters: {diag['n_games_with_starters']} of "
                  f"{len(state.remaining)} remaining games have both probables "
                  f"posted in the next {args.starter_window_days} days "
                  f"({diag['sp_no_history']} starter slots had no history, "
                  f"scored at league average)")

    # Same seed both ways, so the draw is identical on every game the overrides
    # do not touch and the whole difference in the table below is the term.
    base = run_playoff_odds(state, strength, hfa, n_sims=args.sims, seed=seed)
    base["division"] = base["division_id"].map(DIVISION_NAMES)
    if overrides:
        odds = run_playoff_odds(state, strength, hfa, n_sims=args.sims, seed=seed,
                                p_home_overrides=overrides)
        odds["division"] = odds["division_id"].map(DIVISION_NAMES)
    else:
        odds = base

    print(f"\n─── without the starter term (team strength on all "
          f"{len(state.remaining)} remaining games) ───")
    print(format_odds(base).to_string(index=False))

    if overrides:
        print(f"\n─── with the starter term on {len(overrides)} games "
              f"({SP_METHOD.split('; ')[1]}) ───")
        print(format_odds(odds).to_string(index=False))
        cmp = compare(base, odds)
        print("\n─── with vs without, same seed (percentage points) ───")
        print(cmp.to_string(index=False))
        print(f"\nmax |Δ p_playoffs| = {cmp['d_p_playoffs'].abs().max():.2f} pts "
              f"({cmp.loc[cmp['d_p_playoffs'].abs().idxmax(), 'abbrev']})")
        print(f"max |Δ p_ws|        = {cmp['d_p_ws'].abs().max():.2f} pts "
              f"({cmp.loc[cmp['d_p_ws'].abs().idxmax(), 'abbrev']})")
        print(f"max |Δ mean wins|   = {cmp['d_mean_wins'].abs().max():.2f} "
              f"({cmp.loc[cmp['d_mean_wins'].abs().idxmax(), 'abbrev']})")

        # The two tables above are two Monte Carlo runs; in September their
        # difference is smaller than the sim noise. This part is exact.
        effect = override_effect(state, strength, hfa, overrides)
        print(f"\n─── the term itself, in closed form (no sampling) ───")
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
        "method": SP_METHOD if overrides else BASE_METHOD,
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
