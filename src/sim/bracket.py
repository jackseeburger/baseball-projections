"""Postseason bracket (roadmap 2.5).

Per league: seeds 1-2 (top two division winners) get byes. Wild Card round
is best-of-3, all games at the higher seed: 3 v 6 and 4 v 5. Division
Series best-of-5 (2-2-1): 1 v winner(4/5), 2 v winner(3/6). LCS best-of-7
(2-3-2). World Series best-of-7 (2-3-2), home field to the better regular
season record.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sim.strength import home_win_prob

SERIES_HOME_PATTERNS = {
    # wins needed → which games the higher seed hosts (1-indexed)
    2: {1, 2, 3},            # best-of-3: all at higher seed
    3: {1, 2, 5},            # best-of-5: 2-2-1
    4: {1, 2, 6, 7},         # best-of-7: 2-3-2
}


def play_series(high: int, low: int, wins_needed: int, strength: np.ndarray,
                hfa: float, rng: np.random.Generator) -> int:
    """Simulate a series; returns the winning team's row index."""
    hosts = SERIES_HOME_PATTERNS[wins_needed]
    w_high = w_low = 0
    game = 1
    while w_high < wins_needed and w_low < wins_needed:
        if game in hosts:
            p_high = home_win_prob(strength[high], strength[low], hfa)
        else:
            p_high = 1 - home_win_prob(strength[low], strength[high], hfa)
        if rng.random() < p_high:
            w_high += 1
        else:
            w_low += 1
        game += 1
    return high if w_high == wins_needed else low


@dataclass
class PostseasonResult:
    pennant: dict[int, int]     # league_id → row index of champion
    champion: int


def play_postseason(seeds_by_league: dict[int, list[int]], reg_wins: np.ndarray,
                    strength: np.ndarray, hfa: float,
                    rng: np.random.Generator) -> PostseasonResult:
    pennants = {}
    for league_id, seeds in seeds_by_league.items():
        s1, s2, s3, s4, s5, s6 = seeds
        w45 = play_series(s4, s5, 2, strength, hfa, rng)
        w36 = play_series(s3, s6, 2, strength, hfa, rng)
        d1 = play_series(s1, w45, 3, strength, hfa, rng)
        d2 = play_series(s2, w36, 3, strength, hfa, rng)
        # Higher seed hosts the LCS
        hi, lo = (d1, d2) if seeds.index(d1) < seeds.index(d2) else (d2, d1)
        pennants[league_id] = play_series(hi, lo, 4, strength, hfa, rng)
    a, b = list(pennants.values())
    if reg_wins[a] > reg_wins[b] or (reg_wins[a] == reg_wins[b] and rng.random() < 0.5):
        hi, lo = a, b
    else:
        hi, lo = b, a
    return PostseasonResult(pennants, play_series(hi, lo, 4, strength, hfa, rng))
