"""End-to-end playoff odds: state → sims → standings → bracket → table (2.6)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.sim.bracket import play_postseason
from src.sim.season import SeasonState, simulate_remaining, tally
from src.sim.standings import TiebreakContext, seed_league
from src.sim.teams import AL, NL

PERCENTILES = (5, 25, 50, 75, 95)


def run_playoff_odds(
    state: SeasonState, strength: pd.Series, hfa: float,
    n_sims: int = 20_000, seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    home_wins = simulate_remaining(state, strength, hfa, n_sims, rng)
    records = tally(state, home_wins)
    ctx = TiebreakContext.build(state, records, home_wins, rng)
    strength_arr = strength.reindex(state.team_ids).to_numpy()

    n = len(state.team_ids)
    counts = {k: np.zeros(n) for k in
              ("playoffs", "division", "bye", "wild_card", "pennant", "ws")}
    for s in range(n_sims):
        seeds = {lg: seed_league(s, lg, ctx) for lg in (AL, NL)}
        for lg_seeds in seeds.values():
            for t in lg_seeds.division_winners:
                counts["division"][t] += 1
            for t in lg_seeds.division_winners[:2]:
                counts["bye"][t] += 1
            for t in lg_seeds.wild_cards:
                counts["wild_card"][t] += 1
            for t in lg_seeds.seeds:
                counts["playoffs"][t] += 1
        post = play_postseason(
            {lg: sd.seeds for lg, sd in seeds.items()},
            records.wins[s], strength_arr, hfa, rng,
        )
        for t in post.pennant.values():
            counts["pennant"][t] += 1
        counts["ws"][post.champion] += 1

    out = state.teams.copy()
    out["strength"] = strength_arr
    out["wins"], out["losses"] = _current_record(state)
    out["mean_wins"] = records.wins.mean(0)
    for p in PERCENTILES:
        out[f"wins_p{p}"] = np.percentile(records.wins, p, axis=0)
    for k, v in counts.items():
        out[f"p_{k}"] = v / n_sims
    return out.sort_values("p_ws", ascending=False).reset_index(drop=True)


def _current_record(state: SeasonState) -> tuple[np.ndarray, np.ndarray]:
    idx = state.index_of()
    n = len(idx)
    w, l = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    for g in state.completed.itertuples(index=False):
        h, a = idx[int(g.home_id)], idx[int(g.away_id)]
        if g.home_win:
            w[h] += 1; l[a] += 1
        else:
            w[a] += 1; l[h] += 1
    return w, l
