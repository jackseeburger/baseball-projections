"""The whole station E chain for one game, in one place.

Two callers price a baseball game in this repository, and until now they did
it with two different chains: `scripts/backtest_game_odds.py` walked a whole
season and scored the ladder of terms, and `scripts/run_playoff_odds.py`
priced tonight's slate with the starter term alone. The two shared the pieces
(`starters.py`, `lineups.py`, `bullpen.py`, `reliever_usage.py`,
`run_environment.py`) but assembled them separately, which is exactly the
shape a silent drift takes: a knob moves on one side, the scoreboard keeps
reporting the other side's number, and nothing fails.

This module is the assembly, once:

    RS/9 = C's blended runs scored
           + w · (posted lineup RAA/9 − the club's own recent cards)   ← lineups.py
    RA/9 = C's blended runs allowed
           + (ip/9)     · (starter FIP RA/9 − league RA/9)             ← starters.py
           + (1 − ip/9) · (available pen RA/9 − league available pen)  ← reliever_usage.py

    P(home) = log5(Pythagenpat(RS, RA) both sides) with home-field advantage

where `ip` is *this* starter's own expected innings rather than a flat 5.5,
and the blended run environment underneath both lines is station C's
bottom-up rebuild mixed half-and-half with the top-down regressed rates
(`run_environment.py`). That is the model
`scripts/backtest_game_odds.py` scores as `pythag_C_sp_bpa_ip` — Brier .24388
on the 756 market-priced 2026 games against the production model's .24619
(docs/market-benchmark-2026.md) — plus one branch it does not have: when a
club's card for the game is already posted, the lineup delta above fires. With
no card posted the delta is exactly zero, because the club's own recent cards
are what it is measured against, and the served probability is the scored one
to the last bit.

Two objects and two functions:

    ChainInputs.from_logs()  season-long frames (pitching logs, hitting logs,
                             the completed prior seasons) sliced into the
                             shapes each station module wants — fetched once
                             by either caller
    build_slate()            everything one *date* needs, every frame cut to
                             games strictly before it: the leakage guard, in
                             one place instead of two
    home_win_probability()   one game, from a slate

`build_slate` is pure — no network — so the caller owns the fetching and the
unit tests own the arithmetic. `scripts/run_playoff_odds.py` builds a slate
for today; `scripts/backtest_game_odds.py` builds one per date it scores; and
`tests/test_sim/test_chain_agreement.py` fixes a single game and asserts the
two paths return the same number.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from src.sim import bullpen as bp_model
from src.sim import lineups as lu_model
from src.sim import reliever_usage as ru_model
from src.sim import run_environment as rn_model
from src.sim import starters as sp_model
from src.sim.strength import HFA_PRIOR, home_win_prob, pythagenpat

# Pythagenpat needs a positive run rate on both sides. Never binds on real
# rates; it is here so a degenerate input cannot raise inside a Monte Carlo.
MIN_R9 = 0.5


@dataclass(frozen=True)
class ChainConfig:
    """Every knob the chain has, with its provenance in the module that owns it.

    Nothing here is fitted to the games the chain is scored on: each default is
    the constant its own module documents, chosen either from outside baseball
    (stabilization points, FIP coefficients, Marcel's weights) or walk-forward
    on the 2025 season. Held together in one frozen object so a caller can
    sweep one of them without either caller growing its own copy of the
    defaults.
    """
    # starters
    sp_ballast: dict = field(default_factory=lambda: dict(sp_model.BALLAST_BF))
    starter_ip: float = sp_model.STARTER_IP
    ip_ballast: float = sp_model.IP_BALLAST_STARTS
    # bullpen / availability
    roster_days: int = bp_model.ROSTER_WINDOW_DAYS
    hard_1d: float = ru_model.HARD_1D_PITCHES
    hard_2d: float = ru_model.HARD_2D_PITCHES
    taper: float = ru_model.TAPER_PITCHES
    # lineups
    lu_ballast: dict = field(default_factory=lambda: dict(lu_model.BALLAST))
    lineup_weight: float = lu_model.WEIGHT
    baseline_window: int = lu_model.BASELINE_WINDOW_GAMES
    baseline_ballast: float = lu_model.BASELINE_BALLAST_GAMES
    # station C
    blend_weight: float = rn_model.BLEND_WEIGHT
    share_window: int | None = rn_model.SHARE_WINDOW_DAYS
    rotation_days: int | None = rn_model.ROTATION_WINDOW_DAYS
    rotation_top_n: int = rn_model.ROTATION_TOP_N


@dataclass(frozen=True)
class ChainInputs:
    """The season-long frames the chain slices, fetched once by the caller.

    Every frame carries a `date` column and is *not* cut to any date here:
    `build_slate` applies the strictly-before filter, so one fetch serves a
    whole walk-forward season and a single live slate alike.
    """
    season: int
    # pitching
    pitcher_prior_counts: pd.DataFrame   # completed prior seasons, normalized
    pitcher_league: dict                 # starters.league_rates on those
    pitcher_counts: pd.DataFrame         # this season's appearances, dated
    relief: pd.DataFrame                 # bullpen.relief_appearances
    usage: pd.DataFrame                  # reliever_usage.appearance_pitches
    starts: pd.DataFrame                 # run_environment.start_appearances
    start_ip: pd.DataFrame               # starters.start_innings
    # hitting
    hitter_prior_counts: pd.DataFrame    # completed prior seasons, normalized
    hitter_league: dict                  # lineups.league_rates on those
    hitter_counts: pd.DataFrame          # this season's games, dated
    hitter_pa: pd.DataFrame              # batter, team_id, date, pa

    @property
    def pa_per_game(self) -> float:
        """Plate appearances a club gets in nine innings.

        Measured from the league's own batters faced per inning rather than
        assumed, which is what both callers already did separately.
        """
        return float(self.pitcher_league["bf_per_ip"]) * 9.0

    @classmethod
    def from_logs(cls, season: int, pitching_logs: pd.DataFrame,
                  hitting_logs: pd.DataFrame, pitcher_prior: pd.DataFrame,
                  hitter_prior: pd.DataFrame) -> "ChainInputs":
        """Build every frame above from the two raw game-log frames.

        `pitching_logs` is `mlb_stats_api.fetch_pitcher_game_logs` and
        `hitting_logs` is `mlb_stats_api.fetch_hitter_game_logs`, both already
        filtered to the regular season; `pitcher_prior` and `hitter_prior` are
        the completed prior seasons (`fetch_season_pitching`,
        `build_seasons_table`). Only regular-season rows should reach here —
        a spring or postseason line is neither a rate nor a workload the
        regular season should read.
        """
        p_counts = sp_model.normalize_counts(pitching_logs)
        p_counts["date"] = pitching_logs["date"].to_numpy()
        h_counts = lu_model.normalize_counts(hitting_logs)
        h_counts["date"] = hitting_logs["date"].to_numpy()
        p_prior = sp_model.normalize_counts(pitcher_prior)
        h_prior = lu_model.normalize_counts(hitter_prior)
        pa = hitting_logs.loc[:, ["batter", "team_id", "date", "pa"]].dropna(
            subset=["team_id"])
        return cls(
            season=int(season),
            pitcher_prior_counts=p_prior,
            pitcher_league=sp_model.league_rates(p_prior),
            pitcher_counts=p_counts,
            relief=bp_model.relief_appearances(pitching_logs),
            usage=ru_model.appearance_pitches(pitching_logs),
            starts=rn_model.start_appearances(pitching_logs),
            start_ip=sp_model.start_innings(pitching_logs),
            hitter_prior_counts=h_prior,
            hitter_league=lu_model.league_rates(h_prior),
            hitter_counts=h_counts,
            hitter_pa=pa,
        )


@dataclass(frozen=True)
class Slate:
    """One date's inputs, every one of them cut to games strictly before it.

    `team` is station C's blended runs scored / allowed per game, indexed by
    team_id — the prior a game inherits before anybody's starter, pen or card
    is known, and therefore also the right team-strength prior for a game
    beyond the probables horizon (`talent()`).

    `available_pen` is a callable rather than a table because the exclusion is
    tonight's announced starter, who is on the roster but is not in the pen
    behind himself.
    """
    as_of: str
    team: pd.DataFrame
    sp_ra9: dict
    lg_ra9: float
    lg_rs9: float
    expected_ip: dict
    available_pen: Callable[[int, int], float] | None
    # The league's own relievers on the same availability weights (a float), or
    # a per-club mapping when the pen is measured against the club's own whole
    # pen instead — the two readings `reliever_usage.BASELINE` chooses between.
    pen_baseline: float | dict
    runs_lookup: dict
    lineup_baseline: dict
    pa_per_game: float
    config: ChainConfig = field(default_factory=ChainConfig)
    diagnostics: dict = field(default_factory=dict)

    def talent(self) -> pd.Series:
        """team_id → talent win% with nobody named: Pythagenpat on `team`.

        This is what `strength.regressed_strength` returns for the production
        model, with station C's blend in place of the raw regressed rates.
        """
        return pd.Series(
            {int(t): pythagenpat(max(float(r["rs_pg"]), MIN_R9),
                                 max(float(r["ra_pg"]), MIN_R9), 1.0)
             for t, r in self.team.iterrows()}, name="strength")

    def run_env(self) -> pd.Series:
        """team_id → runs scored + allowed per game, the bracket's environment."""
        return (self.team["rs_pg"] + self.team["ra_pg"]).astype(float)

    def lineup_raa9(self, batter_ids) -> float:
        """A posted card's runs above league average per nine innings.

        Above *average*, so the number is scale-free and a club's baseline can
        be accumulated across dates whose league run environment differs.
        """
        return lu_model.lineup_r9(batter_ids, self.runs_lookup, 0.0,
                                  self.pa_per_game)


def side_run_rates(slate: Slate, team_id: int, starter_id=None,
                   lineup=None) -> tuple[float, float, dict]:
    """One club's expected runs scored / allowed per nine for this game.

    The three deltas stack additively on station C's blended rates: the card
    moves runs scored, the starter moves the innings he is expected to cover,
    and the pen moves the rest. Each is a *delta* from a baseline that is
    league average (starter, pen) or the club's own recent cards (lineup), so
    a league-average starter in front of a fully rested pen with the club's
    usual nine leaves the blended rates exactly where station C put them —
    which is what lets the terms be added at all.

    Returns `(rs9, ra9, flags)`.
    """
    cfg = slate.config
    flags = {"sp_no_history": 0, "lineup_slots": 0, "pen_shift": 0.0}
    rs9 = max(float(slate.team.loc[team_id, "rs_pg"]), MIN_R9)
    ra9 = max(float(slate.team.loc[team_id, "ra_pg"]), MIN_R9)

    if lineup:
        flags["lineup_slots"] = len(lineup)
        rs9 = float(lu_model.blend_lineup_team(
            slate.lineup_raa9(lineup), rs9,
            slate.lineup_baseline.get(int(team_id), 0.0),
            weight=cfg.lineup_weight))

    ip = float(cfg.starter_ip)
    if starter_id is not None:
        pid = int(starter_id)
        flags["sp_no_history"] = int(pid not in slate.sp_ra9)
        ip = float(slate.expected_ip.get(pid, cfg.starter_ip))
        ra9 = float(sp_model.blend_starter_team(
            slate.sp_ra9.get(pid, slate.lg_ra9), ra9, slate.lg_ra9,
            starter_ip=ip))
        if slate.available_pen is not None:
            pen = float(slate.available_pen(int(team_id), pid))
            base = (slate.pen_baseline.get(int(team_id), slate.lg_ra9)
                    if isinstance(slate.pen_baseline, dict)
                    else float(slate.pen_baseline))
            flags["pen_shift"] = abs(pen - base)
            ra9 = float(bp_model.blend_bullpen_team(
                pen, ra9, base, relief_ip=sp_model.GAME_IP - ip))
    return max(rs9, MIN_R9), max(ra9, MIN_R9), flags


def home_win_probability(slate: Slate, home_id: int, away_id: int,
                         starters=None, hfa: float = HFA_PRIOR,
                         lineups=None) -> tuple[float, dict]:
    """P(home wins) for one game — the number the site serves and the harness scores.

    `starters` is `(home starter id, away starter id)` or None when nobody has
    announced; `lineups` is `{"home": [nine ids], "away": [...]}` or None, and
    a side may be missing from it (one club posts before the other). Anything
    unknown falls back to the club's own blended rates, which is the correct
    rotation-average, card-average expectation for a game whose particulars
    nobody has published yet — so this one function covers every remaining
    game of a season, not only tonight's.

    Returns `(P(home), flags)`.
    """
    talent, flags = {}, {"sp_no_history": 0, "lineup_slots": 0, "pen_shift": 0.0}
    for side, team_id, i in (("home", home_id, 0), ("away", away_id, 1)):
        sp = None if starters is None else starters[i]
        card = None if lineups is None else lineups.get(side)
        rs9, ra9, f = side_run_rates(slate, team_id, sp, card)
        talent[side] = pythagenpat(rs9, ra9, 1.0)
        for k, v in f.items():
            flags[k] += v
    return float(home_win_prob(talent["home"], talent["away"], hfa)), flags


def build_slate(as_of: str, inputs: ChainInputs, top_down: pd.DataFrame,
                lg_rs9: float, lg_ra9: float, *,
                cards: dict[int, list] | None = None,
                config: ChainConfig | None = None) -> Slate:
    """Everything the games on `as_of` need, from appearances strictly before it.

    `top_down` is station D's regressed runs scored / allowed per game indexed
    by team_id (`strength.regressed_run_rates` live, or the backtest's
    walk-forward equivalent) and `lg_rs9` / `lg_ra9` are the league's own runs
    per game — the anchor every delta in the chain is centred on, so no term
    can move the league's run environment, only redistribute it.

    `cards` is `{team_id: [[nine batter ids], ...]}`, the club's recent posted
    lineups oldest-first; only the last `config.baseline_window` are used and
    they are re-scored with *this* date's rates (see
    `lineups.team_lineup_baseline` for why that matters). A club with no cards
    on file gets a baseline of league average, which with the default zero
    ballast means its first posted card is measured against itself.

    Pure: every frame is sliced here and nothing is fetched.
    """
    cfg = config or ChainConfig()
    lg_rs9, lg_ra9 = float(lg_rs9), float(lg_ra9)
    team_ids = pd.Index([int(t) for t in top_down.index], name="team_id")

    # ── the pitcher rate table: every arm, not only the announced starters ──
    counts = pd.concat(
        [inputs.pitcher_prior_counts,
         sp_model.appearances_before(inputs.pitcher_counts, as_of)],
        ignore_index=True)
    rates = sp_model.marcel_rates(counts, inputs.season, inputs.pitcher_league,
                                  ballast=cfg.sp_ballast)
    sp_ra9 = sp_model.starter_ra9_lookup(rates, inputs.pitcher_league, lg_ra9)

    # ── the pen: who is in it, how much he works, how available he is ──
    pens = bp_model.pen_window(inputs.relief, as_of, days=cfg.roster_days)
    weights = ru_model.availability(inputs.usage, as_of, hard_1d=cfg.hard_1d,
                                    hard_2d=cfg.hard_2d, taper=cfg.taper)
    frames = {int(t): g for t, g in pens.groupby("team")} if len(pens) else {}
    pen_baseline = ru_model.league_available_pen_ra9(pens, sp_ra9, lg_ra9, weights)

    def available_pen(team_id: int, starter_id: int) -> float:
        grp = frames.get(int(team_id))
        if grp is None:
            return lg_ra9
        return ru_model.available_pen_ra9(grp, sp_ra9, lg_ra9, weights,
                                          exclude=(int(starter_id),))

    # ── the hitter rate table, and station C's two halves on top of it ──
    h_counts = pd.concat(
        [inputs.hitter_prior_counts,
         lu_model.games_before(inputs.hitter_counts, as_of)], ignore_index=True)
    h_rates = lu_model.marcel_rates(h_counts, inputs.season,
                                    inputs.hitter_league, ballast=cfg.lu_ballast)
    runs_lookup = lu_model.batter_runs_lookup(h_rates, inputs.hitter_league)

    pa_per_game = inputs.pa_per_game
    shares = rn_model.team_pa_shares(inputs.hitter_pa, as_of,
                                     window_days=cfg.share_window)
    rs9 = rn_model.team_rs9(shares, runs_lookup, lg_rs9, pa_per_game)
    rotation = rn_model.rotation_window(inputs.starts, as_of,
                                        days=cfg.rotation_days,
                                        top_n=cfg.rotation_top_n)
    rot_ra9 = rn_model.rotation_ra9(rotation, sp_ra9, lg_ra9)
    pen_full = {int(t): bp_model.pen_ra9(g, sp_ra9, lg_ra9)
                for t, g in frames.items()}
    ra9 = rn_model.team_ra9(rot_ra9, pen_full, lg_ra9, team_ids=team_ids,
                            starter_ip=cfg.starter_ip)
    bottom_up = rn_model.bottom_up_rates(rs9, ra9, team_ids=team_ids)
    blended = rn_model.blend_run_env(bottom_up, top_down, cfg.blend_weight)

    # ── each club's own recent cards, re-scored on today's rates ──
    baseline = {}
    for team_id, history in (cards or {}).items():
        recent = list(history)[-cfg.baseline_window:]
        baseline[int(team_id)] = lu_model.team_lineup_baseline(
            [lu_model.lineup_r9(ids, runs_lookup, 0.0, pa_per_game)
             for ids in recent], 0.0, cfg.baseline_ballast)

    diag = {
        "n_pitchers_rated": len(sp_ra9),
        "n_batters_rated": len(runs_lookup),
        "n_pens": len(frames),
        "n_limited_arms": sum(1 for w in weights.values() if w < 1.0),
        "n_rotations": len(rot_ra9),
        "rs_missing": sorted(int(t) for t in team_ids if t not in set(rs9.index)),
        "ra_missing": sorted(int(t) for t in team_ids
                             if int(t) not in rot_ra9 or int(t) not in pen_full),
        "blend_weight": float(cfg.blend_weight),
    }
    return Slate(as_of=str(as_of), team=blended, sp_ra9=sp_ra9, lg_ra9=lg_ra9,
                 lg_rs9=lg_rs9, expected_ip=sp_model.expected_starter_ip(
                     inputs.start_ip, as_of, ballast=cfg.ip_ballast),
                 available_pen=available_pen, pen_baseline=pen_baseline,
                 runs_lookup=runs_lookup, lineup_baseline=baseline,
                 pa_per_game=pa_per_game, config=cfg, diagnostics=diag)


def league_run_rates(rs_total: float, ra_total: float, games: float) -> tuple[float, float]:
    """League runs scored / allowed per team-game — the chain's two anchors.

    Trivial, and here so both callers compute them the same way from whatever
    they hold (a standings frame live, a completed-games frame in the
    backtest). They are equal in a closed league; kept separate because the
    two halves of the chain read them for different purposes.
    """
    g = max(float(games), 1.0)
    return float(rs_total) / g, float(ra_total) / g


def strength_series(slate: Slate, team_ids=None) -> pd.Series:
    """`slate.talent()` reindexed onto a caller's team order (the Monte Carlo's)."""
    s = slate.talent()
    return s if team_ids is None else s.reindex([int(t) for t in team_ids])


__all__ = ["ChainConfig", "ChainInputs", "Slate", "build_slate",
           "home_win_probability", "side_run_rates", "strength_series",
           "league_run_rates", "MIN_R9"]
