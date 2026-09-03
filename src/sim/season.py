"""Season state and Monte Carlo over the remaining schedule (roadmap 2.3).

`SeasonState` splits the regular-season schedule into completed games (with
results) and remaining games. `simulate_remaining` draws every remaining
game for every simulation at once; per-sim win totals, intradivision
records, and intraleague-last-half records (the tiebreaker inputs) are then
matrix products of that draw with one-hot team indicators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.sim.strength import home_win_prob


@dataclass
class SeasonState:
    teams: pd.DataFrame          # team_id, abbrev, name, league_id, division_id
    completed: pd.DataFrame      # date, home_id, away_id, home_win
    remaining: pd.DataFrame      # game_pk, date, home_id, away_id

    @property
    def team_ids(self) -> np.ndarray:
        return self.teams["team_id"].to_numpy()

    def index_of(self) -> dict[int, int]:
        return {int(t): i for i, t in enumerate(self.team_ids)}


def from_schedule(schedule: pd.DataFrame, teams: pd.DataFrame) -> SeasonState:
    """Split a Stats API schedule frame into completed and remaining games.

    Postponed games appear as 'Final' with no score and are dropped; their
    makeups show up as separate scheduled rows. Ties (suspended games) are
    dropped too.
    """
    reg = schedule[schedule["game_type"] == "R"].copy()
    valid_ids = set(teams["team_id"])
    reg = reg[reg["home_id"].isin(valid_ids) & reg["away_id"].isin(valid_ids)]

    is_final = reg["status"] == "Final"
    scored = reg["home_score"].notna() & reg["away_score"].notna()
    done = reg[is_final & scored & (reg["home_score"] != reg["away_score"])].copy()
    done["home_win"] = done["home_score"] > done["away_score"]
    remaining = reg[~is_final].copy()

    cols = ["date", "home_id", "away_id"]
    # game_pk rides along so a per-game probability (station E's starting
    # pitcher term) can be keyed to a specific remaining game.
    if "game_pk" in reg.columns:
        cols = ["game_pk"] + cols
    return SeasonState(
        teams=teams.sort_values("team_id").reset_index(drop=True),
        completed=done[cols + ["home_win"]].sort_values("date").reset_index(drop=True),
        remaining=remaining[cols].sort_values("date").reset_index(drop=True),
    )


def simulate_remaining(
    state: SeasonState, strength: pd.Series, hfa: float,
    n_sims: int, rng: np.random.Generator,
    p_home_overrides: dict[int, float] | None = None,
) -> np.ndarray:
    """Boolean (n_sims, n_remaining): True where the home team wins.

    Every remaining game gets log5 + HFA on team strength, which is the right
    rotation-average expectation for a game whose starters are not known yet.
    `p_home_overrides` (game_pk → P(home)) replaces that probability for the
    games it names and nothing else: the nightly job passes the
    starter-adjusted probability for the handful of upcoming games whose
    probables the Stats API has posted (`scripts/run_playoff_odds.py`).
    A game_pk that is not in the remaining schedule — it finished between the
    probables fetch and now — is ignored.

    The draw stays one vectorized comparison against a per-game probability
    vector, so the same seed gives the same outcome on every game the
    overrides do not touch.
    """
    p_home = home_win_prob(
        strength.reindex(state.remaining["home_id"]).to_numpy(),
        strength.reindex(state.remaining["away_id"]).to_numpy(),
        hfa=hfa,
    )
    if p_home_overrides:
        p_home = np.array(p_home, dtype=float, copy=True)
        if "game_pk" not in state.remaining.columns:
            raise KeyError("p_home_overrides needs a game_pk column on state.remaining")
        pos = {int(pk): i for i, pk in enumerate(state.remaining["game_pk"])}
        for game_pk, p in p_home_overrides.items():
            i = pos.get(int(game_pk))
            if i is not None:
                p_home[i] = float(p)
    return rng.random((n_sims, len(state.remaining))) < p_home


def _one_hot(ids: pd.Series, index: dict[int, int], n_teams: int, mask=None) -> np.ndarray:
    m = np.zeros((len(ids), n_teams))
    rows = np.arange(len(ids))
    cols = ids.map(index).to_numpy()
    if mask is not None:
        rows, cols = rows[mask], cols[mask]
    m[rows, cols] = 1.0
    return m


@dataclass
class SimRecords:
    """Per-sim record arrays (n_sims, n_teams) used for standings/tiebreaks."""
    wins: np.ndarray
    losses: np.ndarray
    intradiv_wins: np.ndarray
    intradiv_games: np.ndarray
    il_half_wins: np.ndarray     # intraleague wins, last half of intraleague games
    il_half_games: np.ndarray


def tally(state: SeasonState, home_wins: np.ndarray) -> SimRecords:
    """Combine completed results with simulated outcomes into per-sim records."""
    idx = state.index_of()
    n = len(state.team_ids)
    div = state.teams.set_index("team_id")["division_id"]
    lg = state.teams.set_index("team_id")["league_id"]

    def team_masks(games: pd.DataFrame):
        same_div = (div.reindex(games["home_id"]).to_numpy()
                    == div.reindex(games["away_id"]).to_numpy())
        same_lg = (lg.reindex(games["home_id"]).to_numpy()
                   == lg.reindex(games["away_id"]).to_numpy())
        return same_div, same_lg

    # --- completed ---
    c = state.completed
    c_home, c_away = _one_hot(c["home_id"], idx, n), _one_hot(c["away_id"], idx, n)
    c_hw = c["home_win"].to_numpy().astype(float)
    base_w = c_hw @ c_home + (1 - c_hw) @ c_away
    base_l = (1 - c_hw) @ c_home + c_hw @ c_away
    c_div, c_lg = team_masks(c)

    # --- remaining ---
    r = state.remaining
    r_home, r_away = _one_hot(r["home_id"], idx, n), _one_hot(r["away_id"], idx, n)
    hw = home_wins.astype(float)
    wins = base_w + hw @ r_home + (1 - hw) @ r_away
    losses = base_l + (1 - hw) @ r_home + hw @ r_away
    r_div, r_lg = team_masks(r)

    # --- intradivision ---
    intradiv_wins = (
        (c_hw * c_div) @ c_home + ((1 - c_hw) * c_div) @ c_away
        + (hw * r_div) @ r_home + ((1 - hw) * r_div) @ r_away
    )
    intradiv_games = (c_div @ c_home + c_div @ c_away
                      + r_div @ r_home + r_div @ r_away)

    # --- intraleague, last half of each team's intraleague games ---
    # Rank every intraleague game (completed then remaining, date order) per
    # team; a game counts toward the "last half" for a team if it falls in
    # the second half of that team's intraleague schedule.
    # Concatenated home/away ids as plain arrays: this loop runs once per
    # simulation *run* but 30 times inside it, and a DataFrame `.at` per game
    # made it the single most expensive thing in a whole-season sim.
    all_home = np.concatenate([c["home_id"].to_numpy(), r["home_id"].to_numpy()])
    all_away = np.concatenate([c["away_id"].to_numpy(), r["away_id"].to_numpy()])
    n_c = len(c)
    all_lg = np.concatenate([c_lg, r_lg])
    half_mask = {"c": np.zeros((len(c), n)), "r": np.zeros((len(r), n))}
    for t in state.team_ids:
        involved = ((all_home == t) | (all_away == t)) & all_lg
        order = np.flatnonzero(involved)  # already date-sorted within c then r
        last_half = order[len(order) // 2:]
        col = idx[int(t)]
        in_c = last_half[last_half < n_c]
        in_r = last_half[last_half >= n_c] - n_c
        half_mask["c"][in_c, col] = 1.0
        half_mask["r"][in_r, col] = 1.0
    hm_c, hm_r = half_mask["c"], half_mask["r"]
    # A team's win in a game counts if the game is in its last half.
    il_half_wins = (
        c_hw @ (hm_c * c_home) + (1 - c_hw) @ (hm_c * c_away)
        + hw @ (hm_r * r_home) + (1 - hw) @ (hm_r * r_away)
    )
    il_half_games = (hm_c * c_home).sum(0) + (hm_c * c_away).sum(0) \
        + (hm_r * r_home).sum(0) + (hm_r * r_away).sum(0)

    return SimRecords(
        wins=wins, losses=losses,
        intradiv_wins=intradiv_wins, intradiv_games=np.broadcast_to(intradiv_games, wins.shape),
        il_half_wins=il_half_wins, il_half_games=np.broadcast_to(il_half_games, wins.shape),
    )
