"""Per-sim league ordering with MLB tiebreakers (roadmap 2.4).

No Game 163 since 2022. Ties are broken, in order, by:
    1. head-to-head record among the tied teams
    2. intradivision record
    3. intraleague record over the last half of intraleague games
    4. coin flip (the real rule continues further; this is the practical floor)

`league_order(sim, league)` returns every team in the league best-first;
division winners are the first team from each division, seeds 1-3 are the
division winners in that order, wild cards are the next three.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.sim.season import SeasonState, SimRecords


@dataclass
class TiebreakContext:
    state: SeasonState
    records: SimRecords
    home_wins: np.ndarray                     # (n_sims, n_remaining)
    h2h_base: np.ndarray                      # (n_teams, n_teams) completed wins of i over j
    pair_games: dict[tuple[int, int], list]   # (i, j) → [(game_idx, i_is_home)]
    rng: np.random.Generator

    @classmethod
    def build(cls, state: SeasonState, records: SimRecords,
              home_wins: np.ndarray, rng: np.random.Generator) -> "TiebreakContext":
        idx = state.index_of()
        n = len(state.team_ids)
        h2h = np.zeros((n, n))
        for _, g in state.completed.iterrows():
            h, a = idx[int(g["home_id"])], idx[int(g["away_id"])]
            if g["home_win"]:
                h2h[h, a] += 1
            else:
                h2h[a, h] += 1
        pairs: dict[tuple[int, int], list] = {}
        for gi, g in enumerate(state.remaining.itertuples(index=False)):
            h, a = idx[int(g.home_id)], idx[int(g.away_id)]
            pairs.setdefault((h, a), []).append((gi, True))
            pairs.setdefault((a, h), []).append((gi, False))
        return cls(state, records, home_wins, h2h, pairs, rng)

    def h2h_wins(self, i: int, j: int, s: int) -> float:
        w = self.h2h_base[i, j]
        for gi, i_home in self.pair_games.get((i, j), []):
            hw = self.home_wins[s, gi]
            w += float(hw if i_home else not hw)
        return w


def _pct(w: float, g: float) -> float:
    return w / g if g > 0 else 0.0


def break_tie(group: list[int], s: int, ctx: TiebreakContext) -> list[int]:
    """Order a group of teams (row indices) tied on wins, best first."""
    if len(group) == 1:
        return list(group)

    def h2h(t):
        w = sum(ctx.h2h_wins(t, o, s) for o in group if o != t)
        g = sum(ctx.h2h_wins(t, o, s) + ctx.h2h_wins(o, t, s) for o in group if o != t)
        return _pct(w, g)

    def intradiv(t):
        return _pct(ctx.records.intradiv_wins[s, t], ctx.records.intradiv_games[s, t])

    def il_half(t):
        return _pct(ctx.records.il_half_wins[s, t], ctx.records.il_half_games[s, t])

    for criterion in (h2h, intradiv, il_half):
        vals = {t: criterion(t) for t in group}
        distinct = sorted(set(vals.values()), reverse=True)
        if len(distinct) > 1:
            ordered = []
            for v in distinct:
                sub = [t for t in group if vals[t] == v]
                # Sub-groups that remain tied restart the criteria chain
                # among themselves, as the MLB rule does.
                ordered.extend(break_tie(sub, s, ctx) if len(sub) > 1 else sub)
            return ordered
    perm = ctx.rng.permutation(len(group))
    return [group[i] for i in perm]


def league_order(s: int, league_rows: list[int], ctx: TiebreakContext) -> list[int]:
    """All teams in a league for sim `s`, best first, ties broken."""
    wins = ctx.records.wins[s]
    by_wins: dict[float, list[int]] = {}
    for t in league_rows:
        by_wins.setdefault(wins[t], []).append(t)
    order = []
    for w in sorted(by_wins, reverse=True):
        order.extend(break_tie(by_wins[w], s, ctx))
    return order


@dataclass
class LeagueSeeds:
    division_winners: list[int]   # seeds 1-3 (row indices)
    wild_cards: list[int]         # seeds 4 onward
    order: list[int]

    @property
    def seeds(self) -> list[int]:
        return self.division_winners + self.wild_cards


def seed_league(s: int, league_id: int, ctx: TiebreakContext,
                n_wild_cards: int = 3) -> LeagueSeeds:
    """The league's playoff field for sim `s`, seeded best-first.

    `n_wild_cards` is the size of the wild-card field per league: three since
    2022, two from 2012 to 2021 (`bracket.PlayoffFormat`). It is a parameter
    because the walk-forward team backtest scores seasons in both eras, and a
    six-club field in a five-club season would be scoring a different outcome.
    """
    teams = ctx.state.teams
    rows = [i for i, l in enumerate(teams["league_id"]) if l == league_id]
    order = league_order(s, rows, ctx)
    div_of = teams["division_id"].to_numpy()
    winners, seen = [], set()
    for t in order:
        if div_of[t] not in seen:
            seen.add(div_of[t])
            winners.append(t)
    wild = [t for t in order if t not in winners][:int(n_wild_cards)]
    return LeagueSeeds(winners, wild, order)
