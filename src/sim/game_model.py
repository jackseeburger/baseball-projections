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
           + λ/9        · (5.5 − ip)                                   ← starters.py

    P(home) = log5(Pythagenpat(RS, RA) both sides) with home-field advantage

where `ip` is *this* starter's own expected innings rather than a flat 5.5,
and the blended run environment underneath both lines is station C's
bottom-up rebuild mixed half-and-half with the top-down regressed rates
(`run_environment.py`). The last line is that same `ip` a second time, as a
**level** rather than as the split — a club is charged λ runs per nine for each
inning of the flat 5.5 its announced starter is not expected to cover — which
is the term issue #66 added after a model with no functional form kept asking
for it (`starters.starter_length_delta`; `ChainConfig.ip_level = 0` is the
chain without it, to the last bit). That is the model
`scripts/backtest_game_odds.py` scores as `pythag_C_sp_bpa_ip_lvl` — Brier
.24364 on the 737 market-priced 2026 games against the rung below it at .24396
and the production model's .24654 (docs/market-benchmark-2026.md) — plus one
branch it does not have: when a club's card for the game is already posted, the
lineup delta above fires. With no card posted the delta is exactly zero,
because the club's own recent cards are what it is measured against, and the
served probability is the scored one to the last bit.

Two further terms are assembled here and **not** served, because neither beat
that model out of sample (docs/market-benchmark-2026.md):

    park     both rates × the venue's run factor, with the park divided out of
             the top-down half first                                ← park.py
    defence  C's bottom-up runs allowed + the club's BABIP-allowed
             residual against the league                         ← defence.py

An infinite ballast on either one (`ChainConfig.park_ballast`,
`ChainConfig.def_ballast`) returns the chain without it to the last bit, which
is how the nightly asks for the gated model and how the sweeps stay clean
nestings.

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
from src.sim import defence as df_model
from src.sim import lineups as lu_model
from src.sim import park as pk_model
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
    # Expected innings as a *level* as well as the split. A nesting: 0.0 is the
    # chain without the term, to the last bit (`starters.starter_length_delta`).
    ip_level: float = sp_model.IP_LEVEL_RUNS
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
    # park and defence. Both are nestings: an infinite ballast on either one
    # returns the chain without that term, to the last bit.
    park_ballast: float = pk_model.BALLAST_GAMES
    def_ballast: float = df_model.BALLAST_BIP
    def_runs_per_hit: float = df_model.RUNS_PER_BIP_HIT
    # The ballast the top-down half is regressed with, needed here only because
    # park neutralisation has to happen to the *totals*, before the league
    # ballast is added (`park.neutral_run_rates`).
    regress_games: float = 60.0


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
    # park and defence. Both default empty, which is the chain without the term
    # — an older caller that hands in no schedule gets exactly what it got.
    park_games: pd.DataFrame = field(    # prior seasons' games, with the venue
        default_factory=lambda: pd.DataFrame(columns=pk_model.GAME_COLS))
    season_games: pd.DataFrame = field(  # this season's, dated: park exposure
        default_factory=lambda: pd.DataFrame(columns=pk_model.GAME_COLS))
    bip: pd.DataFrame = field(           # balls in play allowed, club and date
        default_factory=lambda: pd.DataFrame(columns=df_model.COUNT_COLS))

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
                  hitter_prior: pd.DataFrame,
                  schedule: pd.DataFrame | None = None,
                  park_games: pd.DataFrame | None = None) -> "ChainInputs":
        """Build every frame above from the two raw game-log frames.

        `pitching_logs` is `mlb_stats_api.fetch_pitcher_game_logs` and
        `hitting_logs` is `mlb_stats_api.fetch_hitter_game_logs`, both already
        filtered to the regular season; `pitcher_prior` and `hitter_prior` are
        the completed prior seasons (`fetch_season_pitching`,
        `build_seasons_table`). Only regular-season rows should reach here —
        a spring or postseason line is neither a rate nor a workload the
        regular season should read.

        `schedule` is this season's `fetch_schedule` — where each game was
        played, which is what park exposure and the road-game filter on the
        defence counts read — and `park_games` is the *prior* seasons' games
        in `park.completed_venue_games` shape. Leave either out and the chain
        runs without that term.
        """
        p_counts = sp_model.normalize_counts(pitching_logs)
        p_counts["date"] = pitching_logs["date"].to_numpy()
        h_counts = lu_model.normalize_counts(hitting_logs)
        h_counts["date"] = hitting_logs["date"].to_numpy()
        p_prior = sp_model.normalize_counts(pitcher_prior)
        h_prior = lu_model.normalize_counts(hitter_prior)
        pa = hitting_logs.loc[:, ["batter", "team_id", "date", "pa"]].dropna(
            subset=["team_id"])
        season_games = pk_model.completed_venue_games(schedule)
        home_by_game = ({int(pk): int(t) for pk, t in
                         zip(schedule["game_pk"], schedule["home_id"])}
                        if schedule is not None and len(schedule) else {})
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
            park_games=(pk_model.completed_venue_games(None)
                        if park_games is None else park_games),
            season_games=season_games,
            bip=df_model.bip_counts(pitching_logs, home_by_game),
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
    # {venue_id: run multiplier} from the *prior* seasons (`src.sim.park`).
    # Empty is the chain with no park term, and every lookup then returns 1.0.
    park_factors: dict = field(default_factory=dict)
    config: ChainConfig = field(default_factory=ChainConfig)
    diagnostics: dict = field(default_factory=dict)

    def park(self, venue_id=None) -> float:
        """The run multiplier for one venue; 1.0 for None and for an unseen park."""
        return pk_model.factor(venue_id, self.park_factors)

    def talent(self) -> pd.Series:
        """team_id → talent win% with nobody named: Pythagenpat on `team`.

        This is what `strength.regressed_strength` returns for the production
        model, with station C's blend in place of the raw regressed rates.

        Park-neutral, deliberately: this is the club's talent, and the park is
        a fact about a *game*. The Monte Carlo draws a game nobody has
        announced from these numbers, which is the same thing
        `home_win_probability` returns for that game with no venue given.
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
        # ...and the same expected innings once more, as a level. The two uses
        # are independent: the split above says who throws the innings, this
        # says what a club whose starter is expected to go four is worth. Zero
        # `ip_level` leaves the line above untouched to the last bit.
        if cfg.ip_level:
            ra9 += float(sp_model.starter_length_delta(
                ip, cfg.ip_level, prior=cfg.starter_ip))
    return max(rs9, MIN_R9), max(ra9, MIN_R9), flags


def home_win_probability(slate: Slate, home_id: int, away_id: int,
                         starters=None, hfa: float = HFA_PRIOR,
                         lineups=None, venue_id=None) -> tuple[float, dict]:
    """P(home wins) for one game — the number the site serves and the harness scores.

    `starters` is `(home starter id, away starter id)` or None when nobody has
    announced; `lineups` is `{"home": [nine ids], "away": [...]}` or None, and
    a side may be missing from it (one club posts before the other). Anything
    unknown falls back to the club's own blended rates, which is the correct
    rotation-average, card-average expectation for a game whose particulars
    nobody has published yet — so this one function covers every remaining
    game of a season, not only tonight's.

    `venue_id` is the ballpark. Both clubs' rates are multiplied by its run
    factor — symmetrically, because a park is a fact about the game and not
    about either team — which leaves each club's run *ratio* alone and moves
    only the environment Pythagenpat converts it at. `venue_id=None` (or a
    slate with no park factors) is the chain without the term, to the last bit,
    and is also what the Monte Carlo draws an unannounced game with.

    Returns `(P(home), flags)`.
    """
    park = slate.park(venue_id)
    talent = {}
    flags = {"sp_no_history": 0, "lineup_slots": 0, "pen_shift": 0.0,
             "park_factor": float(park)}
    for side, team_id, i in (("home", home_id, 0), ("away", away_id, 1)):
        sp = None if starters is None else starters[i]
        card = None if lineups is None else lineups.get(side)
        rs9, ra9, f = side_run_rates(slate, team_id, sp, card)
        rs9, ra9 = pk_model.apply_factor(rs9, ra9, park)
        talent[side] = pythagenpat(max(rs9, MIN_R9), max(ra9, MIN_R9), 1.0)
        for k, v in f.items():
            flags[k] += v
    return float(home_win_prob(talent["home"], talent["away"], hfa)), flags


def build_slate(as_of: str, inputs: ChainInputs, top_down: pd.DataFrame,
                lg_rs9: float, lg_ra9: float, *,
                cards: dict[int, list] | None = None,
                config: ChainConfig | None = None,
                totals: pd.DataFrame | None = None) -> Slate:
    """Everything the games on `as_of` need, from appearances strictly before it.

    `top_down` is station D's regressed runs scored / allowed per game indexed
    by team_id (`strength.regressed_run_rates` live, or the backtest's
    walk-forward equivalent) and `lg_rs9` / `lg_ra9` are the league's own runs
    per game — the anchor every delta in the chain is centred on, so no term
    can move the league's run environment, only redistribute it.

    `totals` is the same clubs' *unregressed* runs scored / allowed and games
    played (`rs`, `ra`, `g`, indexed by team_id), and it is what the park term
    needs: neutralising a club's park has to happen to the totals, before the
    league ballast is added, or the ballast's league-average share gets divided
    too (`park.neutral_run_rates`). Handed in, `top_down` is rebuilt from it
    park-neutral — identically to the caller's own regression when the park
    term is off, which is what keeps the sweep a clean nesting. Left out, the
    park factors still price each game's venue but the top-down half keeps
    whatever park is baked into it.

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

    # ── the park: prior seasons' factors, this season's exposure ──
    # The factors read no game of this season at all; the exposure reads only
    # games strictly before `as_of`. Both cuts are the leakage guard.
    park_factors = pk_model.run_factors(inputs.park_games,
                                        ballast=cfg.park_ballast)
    played = inputs.season_games
    if len(played):
        played = played[played["date"].astype(str) < str(as_of)]
    exposure = pk_model.team_exposure(played, park_factors, team_ids)
    if totals is not None:
        tot = totals.reindex(team_ids)
        top_down = pk_model.neutral_run_rates(
            tot["rs"], tot["ra"], tot["g"], cfg.regress_games,
            exposure=exposure if park_factors else None)

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
    # ── the defence FIP cannot see, on the runs-allowed half where the hole is
    def_deltas, def_diag = df_model.team_defence(
        inputs.bip, as_of, ballast=cfg.def_ballast,
        runs_per_hit=cfg.def_runs_per_hit)
    fip_only = rn_model.bottom_up_rates(rs9, ra9, team_ids=team_ids)
    ra9 = df_model.apply_deltas(ra9, def_deltas)
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
        "n_parks": len(park_factors),
        "park_ballast": float(cfg.park_ballast),
        "park_exposure": {int(t): float(v) for t, v in exposure.items()},
        "park_factors": dict(park_factors),
        "def_ballast": float(cfg.def_ballast),
        "def_deltas": {int(t): float(v) for t, v in def_deltas.items()},
        "def_lg_babip": float(def_diag.get("lg_babip", 0.0)),
        "def_bip_per9": float(def_diag.get("bip_per9", 0.0)),
        "def_babip": {int(t): float(v)
                      for t, v in (def_diag.get("babip") or {}).items()},
        # The two halves the blend is made of, kept apart so the gap between
        # them can be attributed term by term
        # (`scripts/attribute_run_environment.py`).
        "bottom_up": bottom_up,
        "bottom_up_fip_only": fip_only,
        "top_down": top_down,
    }
    return Slate(as_of=str(as_of), team=blended, sp_ra9=sp_ra9, lg_ra9=lg_ra9,
                 lg_rs9=lg_rs9, expected_ip=sp_model.expected_starter_ip(
                     inputs.start_ip, as_of, ballast=cfg.ip_ballast),
                 available_pen=available_pen, pen_baseline=pen_baseline,
                 runs_lookup=runs_lookup, lineup_baseline=baseline,
                 pa_per_game=pa_per_game, park_factors=park_factors,
                 config=cfg, diagnostics=diag)


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
