"""Postseason bracket (roadmap 2.5).

Per league: seeds 1-2 (top two division winners) get byes. Wild Card round
is best-of-3, all games at the higher seed: 3 v 6 and 4 v 5. Division
Series best-of-5 (2-2-1): 1 v winner(4/5), 2 v winner(3/6). LCS best-of-7
(2-3-2). World Series best-of-7 (2-3-2), home field to the better regular
season record.

**Rotations (station E in the bracket).** A series is 3-7 games with a known
rotation, so who starts is a bigger share of the answer here than anywhere in
the regular season. `play_series` / `play_postseason` take an optional
`Rotations`: per team, an ordered list of `(pitcher_id, RA/9 delta from league
average)` from the starter model. Game 1 uses the first entry, game 2 the
second, wrapping after the rotation's own length (a 4-man rotation starts the
same pitcher in games 1 and 5). A team with no rotation is priced on team
strength for every game of the series, and because the rotation enters as a
*delta* from league average, only the side that has one moves — exactly the
property that lets `starters.blend_starter_team` mix a park- and
defense-neutral FIP with a team rate that is neither.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.sim.starters import GAME_IP, MIN_RA9, STARTER_IP
from src.sim.strength import PYTHAGENPAT_EXP, home_win_prob, pythagenpat

SERIES_HOME_PATTERNS = {
    # wins needed → which games the higher seed hosts (1-indexed)
    1: {1},                  # one game: the 2012-2021 wild card, at the higher seed
    2: {1, 2, 3},            # best-of-3: all at higher seed
    3: {1, 2, 5},            # best-of-5: 2-2-1
    4: {1, 2, 6, 7},         # best-of-7: 2-3-2
}

# Games a rotation is asked to cover before it wraps. Four is the modern
# postseason default: teams that carry five in the regular season drop to four
# in October because the off days let the top of the rotation take the extra
# start.
DEFAULT_ROTATION_SIZE = 4


@dataclass(frozen=True)
class PlayoffFormat:
    """How many clubs reach October and what the first round looks like.

    The live board only ever needs one of these, but the walk-forward team
    backtest (`src/eval/team_season.py`) scores seasons back to 2015, and the
    field was five clubs a league then, not six. Getting that wrong is not a
    modelling choice — it changes the *outcome* being scored, because two extra
    clubs "made the playoffs" in a season where they did not.

    `n_wild_cards` is per league; the three division winners are always seeded
    1-3. `wc_wins_needed` is the first round's length (1 = the single-game wild
    card of 2012-2021, 2 = the best-of-3 of 2022 on) and `byes` is how many top
    seeds skip that round.
    """
    n_wild_cards: int = 3
    wc_wins_needed: int = 2
    byes: int = 2

    @property
    def n_seeds(self) -> int:
        return 3 + self.n_wild_cards


# 2022 on: six a league, best-of-3 wild card, byes for the top two seeds.
MODERN = PlayoffFormat(n_wild_cards=3, wc_wins_needed=2, byes=2)
# 2012-2021 (2020 excepted): five a league, a one-game wild card, no byes —
# all three division winners skip it.
ONE_GAME_WILD_CARD = PlayoffFormat(n_wild_cards=2, wc_wins_needed=1, byes=0)


def format_for_season(season: int) -> PlayoffFormat:
    """The postseason field in force for a season.

    2020 raises: the 60-game season put *eight* clubs a league in a bracket
    seeded by division place rather than by record, which is a different
    seeding rule and not a parameter of this one. The team backtest excludes
    2020 for that reason and says so.
    """
    season = int(season)
    if season == 2020:
        raise ValueError(
            "2020 used an eight-club-per-league field seeded by division "
            "place over a 60-game season; it is not a parameterisation of "
            "this bracket. Exclude it explicitly.")
    if season >= 2022:
        return MODERN
    if season >= 2012:
        return ONE_GAME_WILD_CARD
    raise ValueError(f"no playoff format wired for {season}")


def strength_with_starter(p: float, ra9_delta: float, run_env: float,
                          starter_ip: float = STARTER_IP) -> float:
    """Talent win% `p`, moved by a starter sitting `ra9_delta` from league average.

    The regular-season term (`starters.blend_starter_team`) works on the team's
    runs-allowed rate and then runs Pythagenpat. The bracket is handed talent
    win% instead, so this inverts Pythagenpat at the team's run environment
    `run_env` (runs scored + allowed per game) to recover the implied RS/RA,
    moves runs allowed by `starter_ip/9 · ra9_delta` — the same weighting and
    the same delta form — and converts back.

    The inversion is exact, so `ra9_delta == 0` returns `p` unchanged to the
    last bit. That is what makes the no-rotation path in `play_series` byte
    identical to the pre-rotation model, and it is why the fallback can be
    per *team* rather than per matchup: a side without a rotation contributes
    its unmodified strength.
    """
    if ra9_delta == 0.0:
        return float(p)
    x = float(run_env) ** PYTHAGENPAT_EXP
    ratio = (p / (1.0 - p)) ** (1.0 / x)          # implied RS/RA
    ra = float(run_env) / (1.0 + ratio)
    rs = float(run_env) - ra
    ra_sp = max(ra + (float(starter_ip) / GAME_IP) * float(ra9_delta), MIN_RA9)
    return pythagenpat(rs, ra_sp, 1.0)


@dataclass(frozen=True)
class Rotations:
    """Ordered postseason rotations, keyed by the team's row index.

    `by_team[i]` is `[(pitcher_id, ra9_delta), ...]` — the delta being the
    pitcher's regressed FIP (as runs allowed per 9) minus the league's runs
    allowed per 9, i.e. negative for a pitcher better than average. Teams
    absent from `by_team` are priced on team strength.

    `run_env` is optional per-team runs scored + allowed per game (row-index
    aligned with `strength`); without it every team is assumed to play in a
    league-average environment of `2 · lg_ra9`, which affects only how much a
    given RA/9 delta bends the win probability, never the baseline.
    """
    by_team: dict[int, list[tuple[int, float]]]
    lg_ra9: float
    run_env: np.ndarray | None = None
    starter_ip: float = STARTER_IP
    # Memo for `series_game_probs`, keyed by (higher seed, lower seed, wins
    # needed). Every simulation in a run shares one `Rotations`, one strength
    # vector and one HFA, so a matchup's table is computed once for the whole
    # 20,000-sim job instead of once per series. Reusing a `Rotations` across
    # runs with a *different* strength vector would serve stale tables, so
    # build a new one per run (`run_playoff_odds.starter_terms` does).
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def starter(self, team: int, game_idx: int) -> tuple[int, float] | None:
        """`(pitcher_id, ra9_delta)` for game `game_idx` (0-based), or None."""
        rot = self.by_team.get(int(team))
        if not rot:
            return None
        return tuple(rot[game_idx % len(rot)])

    def adjusted(self, team: int, game_idx: int, p: float) -> float:
        """Team strength for game `game_idx`, moved by that game's starter."""
        sp = self.starter(team, game_idx)
        if sp is None:
            return float(p)
        env = (2.0 * self.lg_ra9 if self.run_env is None
               else float(self.run_env[int(team)]))
        return strength_with_starter(p, float(sp[1]), env,
                                     starter_ip=self.starter_ip)


def series_game_probs(high: int, low: int, wins_needed: int,
                      strength: np.ndarray, hfa: float,
                      rotations: Rotations | None = None,
                      ) -> tuple[np.ndarray, np.ndarray]:
    """P(higher seed wins) for each game of the series, hosting and visiting.

    A series is at most `2·wins_needed − 1` games and both rotations are known
    up front, so the whole matchup collapses to two short vectors indexed by
    game number: one for the games the higher seed hosts and one for the games
    it visits. `play_series` then only draws uniforms — no per-game strength
    arithmetic inside the loop, and nothing that scales with the number of
    simulations.

    With no rotation on either side every game has the same probability, so
    the vectors are length 1 and the caller indexes them modulo their length.
    """
    n_games = 2 * wins_needed - 1
    key = (int(high), int(low), wins_needed)
    if rotations is not None and key in rotations._cache:
        return rotations._cache[key]
    has_rot = rotations is not None and (
        rotations.by_team.get(int(high)) or rotations.by_team.get(int(low)))
    if not has_rot:
        flat = (np.array([home_win_prob(strength[high], strength[low], hfa)]),
                np.array([1 - home_win_prob(strength[low], strength[high], hfa)]))
        if rotations is not None:
            rotations._cache[key] = flat
        return flat
    at_home = np.empty(n_games)
    on_road = np.empty(n_games)
    for g in range(n_games):
        s_high = rotations.adjusted(high, g, strength[high])
        s_low = rotations.adjusted(low, g, strength[low])
        at_home[g] = home_win_prob(s_high, s_low, hfa)
        on_road[g] = 1 - home_win_prob(s_low, s_high, hfa)
    rotations._cache[key] = (at_home, on_road)
    return at_home, on_road


def play_series(high: int, low: int, wins_needed: int, strength: np.ndarray,
                hfa: float, rng: np.random.Generator,
                rotations: Rotations | None = None) -> int:
    """Simulate a series; returns the winning team's row index.

    One uniform is drawn per game, in game order, whether or not rotations are
    supplied — so a given seed produces the same draws as before and the
    no-rotation path reproduces the pre-rotation model exactly.
    """
    hosts = SERIES_HOME_PATTERNS[wins_needed]
    at_home, on_road = series_game_probs(high, low, wins_needed, strength, hfa,
                                         rotations)
    w_high = w_low = 0
    game = 1
    while w_high < wins_needed and w_low < wins_needed:
        table = at_home if game in hosts else on_road
        p_high = table[(game - 1) % len(table)]
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


def _league_bracket(seeds: list[int], fmt: PlayoffFormat, strength: np.ndarray,
                    hfa: float, rng: np.random.Generator,
                    rotations: Rotations | None) -> tuple[int, int]:
    """One league's wild-card and division rounds → the two LCS participants.

    Six seeds (2022 on): 3v6 and 4v5 in the wild-card round, the byes waiting
    at 1 and 2. Five seeds (2012-2021): one wild-card game between 4 and 5 at
    the higher seed, then 1 v its winner and 2 v 3.
    """
    wc = fmt.wc_wins_needed
    if fmt.n_wild_cards == 3:
        s1, s2, s3, s4, s5, s6 = seeds
        w45 = play_series(s4, s5, wc, strength, hfa, rng, rotations)
        w36 = play_series(s3, s6, wc, strength, hfa, rng, rotations)
        return (play_series(s1, w45, 3, strength, hfa, rng, rotations),
                play_series(s2, w36, 3, strength, hfa, rng, rotations))
    if fmt.n_wild_cards == 2:
        s1, s2, s3, s4, s5 = seeds
        w45 = play_series(s4, s5, wc, strength, hfa, rng, rotations)
        return (play_series(s1, w45, 3, strength, hfa, rng, rotations),
                play_series(s2, s3, 3, strength, hfa, rng, rotations))
    raise ValueError(f"no bracket wired for {fmt.n_wild_cards} wild cards")


def play_postseason(seeds_by_league: dict[int, list[int]], reg_wins: np.ndarray,
                    strength: np.ndarray, hfa: float,
                    rng: np.random.Generator,
                    rotations: Rotations | None = None,
                    fmt: PlayoffFormat = MODERN) -> PostseasonResult:
    """Both leagues' brackets plus the World Series.

    `rotations` (optional) prices each game of every series with the starter
    scheduled to pitch it; see `Rotations`. `fmt` is the postseason field
    (`PlayoffFormat`); the default is the six-club-per-league bracket in force
    since 2022, which is what the live board runs.
    """
    pennants = {}
    for league_id, seeds in seeds_by_league.items():
        d1, d2 = _league_bracket(seeds, fmt, strength, hfa, rng, rotations)
        # Higher seed hosts the LCS
        hi, lo = (d1, d2) if seeds.index(d1) < seeds.index(d2) else (d2, d1)
        pennants[league_id] = play_series(hi, lo, 4, strength, hfa, rng, rotations)
    a, b = list(pennants.values())
    if reg_wins[a] > reg_wins[b] or (reg_wins[a] == reg_wins[b] and rng.random() < 0.5):
        hi, lo = a, b
    else:
        hi, lo = b, a
    return PostseasonResult(
        pennants, play_series(hi, lo, 4, strength, hfa, rng, rotations))
