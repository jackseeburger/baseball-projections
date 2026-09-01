"""Team strength and per-game win probability (roadmap 2.2).

    talent win%  = Pythagenpat on run rates regressed toward league average
    P(home wins) = log5(home, away) with a home-field odds multiplier
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PYTHAGENPAT_EXP = 0.287
HFA_PRIOR = 0.540        # long-run MLB home win%, per roadmap
HFA_PRIOR_GAMES = 2000   # weight of the prior vs. current-season observed


def pythagenpat(rs: float, ra: float, games: float) -> float:
    """Expected win% from runs scored / allowed over `games`."""
    rpg = (rs + ra) / games
    x = rpg ** PYTHAGENPAT_EXP
    return float(rs ** x / (rs ** x + ra ** x))


def regressed_strength(standings: pd.DataFrame, regress_games: float = 60.0) -> pd.Series:
    """team_id → talent win%.

    Each team's runs scored / allowed per game are regressed toward the league
    average with `regress_games` of ballast, then converted with Pythagenpat.
    A 60-game ballast is a standard in-season shrinkage; tune against the
    backtest harness once historical standings are wired in.
    """
    st = standings.copy()
    st["games"] = st["wins"] + st["losses"]
    lg_rs = st["runs_scored"].sum() / st["games"].sum()
    lg_ra = st["runs_allowed"].sum() / st["games"].sum()
    out = {}
    for _, r in st.iterrows():
        g = r["games"]
        rs_pg = (r["runs_scored"] + regress_games * lg_rs) / (g + regress_games)
        ra_pg = (r["runs_allowed"] + regress_games * lg_ra) / (g + regress_games)
        out[int(r["team_id"])] = pythagenpat(rs_pg, ra_pg, 1.0)
    return pd.Series(out, name="strength")


def from_run_environment(rs_per_game: pd.Series, ra_per_game: pd.Series) -> pd.Series:
    """Hook for Phase 1.5: projected team runs scored/allowed per game →
    talent win%. Same output shape as regressed_strength()."""
    return pd.Series(
        {t: pythagenpat(rs_per_game[t], ra_per_game[t], 1.0) for t in rs_per_game.index},
        name="strength",
    )


def log5(p_a: float, p_b: float) -> float:
    """P(A beats B) on a neutral field given each team's talent win%."""
    return p_a * (1 - p_b) / (p_a * (1 - p_b) + (1 - p_a) * p_b)


def home_win_prob(p_home, p_away, hfa: float = HFA_PRIOR):
    """log5 matchup with home-field advantage applied as an odds multiplier.

    Vectorized: accepts scalars or aligned arrays.
    """
    p_home, p_away = np.asarray(p_home, dtype=float), np.asarray(p_away, dtype=float)
    p = p_home * (1 - p_away) / (p_home * (1 - p_away) + (1 - p_home) * p_away)
    odds = p / (1 - p) * (hfa / (1 - hfa))
    return odds / (1 + odds)


def estimate_hfa(completed: pd.DataFrame, prior: float = HFA_PRIOR,
                 prior_games: int = HFA_PRIOR_GAMES) -> float:
    """Season-to-date home win%, shrunk toward the long-run prior."""
    n = len(completed)
    if n == 0:
        return prior
    home_wins = int(completed["home_win"].sum())
    return (prior * prior_games + home_wins) / (prior_games + n)
