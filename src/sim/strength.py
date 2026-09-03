"""Team strength and per-game win probability (roadmap 2.2).

    talent win%  = Pythagenpat on run rates regressed toward league average
    P(home wins) = log5(home, away) with a home-field odds multiplier
"""
from __future__ import annotations

from dataclasses import dataclass

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


def regressed_run_rates(standings: pd.DataFrame,
                        regress_games: float = 60.0) -> pd.DataFrame:
    """team_id → runs scored / allowed per game, regressed toward the league.

    The half-step inside `regressed_strength`, exposed on its own because the
    station E starting-pitcher term needs the *rates*, not the win% they
    collapse to: it moves a team's RA/G by how far its announced starter sits
    from league average before Pythagenpat is applied
    (`src/sim/starters.blend_starter_team`). Columns: rs_pg, ra_pg.
    """
    st = standings.copy()
    games = st["wins"] + st["losses"]
    lg_rs = st["runs_scored"].sum() / games.sum()
    lg_ra = st["runs_allowed"].sum() / games.sum()
    # Build on `standings`' own row index, then relabel by team_id. Handing
    # team ids to the DataFrame constructor as `index=` would *reindex* the
    # columns by label instead of renaming the rows, silently producing NaNs.
    out = pd.DataFrame({
        "rs_pg": (st["runs_scored"] + regress_games * lg_rs) / (games + regress_games),
        "ra_pg": (st["runs_allowed"] + regress_games * lg_ra) / (games + regress_games),
    })
    out.index = pd.Index(st["team_id"].astype(int).to_numpy(), name="team_id")
    return out


def league_ra_per_game(standings: pd.DataFrame) -> float:
    """League runs allowed per game — the anchor the starter FIPs are centred on."""
    games = (standings["wins"] + standings["losses"]).sum()
    return float(standings["runs_allowed"].sum() / max(games, 1))


def regressed_strength(standings: pd.DataFrame, regress_games: float = 60.0) -> pd.Series:
    """team_id → talent win%.

    Each team's runs scored / allowed per game are regressed toward the league
    average with `regress_games` of ballast, then converted with Pythagenpat.
    A 60-game ballast is a standard in-season shrinkage; tune against the
    backtest harness once historical standings are wired in.
    """
    rates = regressed_run_rates(standings, regress_games)
    return pd.Series(
        {int(t): pythagenpat(r["rs_pg"], r["ra_pg"], 1.0) for t, r in rates.iterrows()},
        name="strength",
    )


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


# ─── parameter uncertainty (stations D and G) ────────────────────────────────
#
# Everything above returns *one number per team*. The season Monte Carlo then
# samples game outcomes around it, so every playoff probability the site
# publishes is conditional on that number being exactly right. What follows is
# the missing half: a distribution over team strength the simulator can draw a
# fresh vector from on every simulated season, so parameter uncertainty and
# game-outcome noise compose instead of only the second being counted.
#
# **Where the width comes from, and why it needs no new constant.**
# `regressed_run_rates` already shrinks a club's runs per game toward the
# league with `regress_games` (60) of ballast. That shrinkage *is* a normal
# prior: adding k pseudo-games at the league rate is the posterior mean of a
# normal-normal model whose prior variance is the game-level variance over k.
# The posterior standard deviation of the same model is then
#
#     sd(regressed rate) = s / sqrt(g + k)
#
# for a club with g games played and a game-level run standard deviation s —
# the standard error you would get from `g + k` observations. The ballast the
# model already ships therefore *names* its own uncertainty, and reading it off
# introduces no constant that was not already in production. Two sanity checks
# the derivation has to pass and does: at g = 0 the width is s/sqrt(k), which
# at MLB's run distribution is about .057 of talent win% — very close to the
# real spread of team talent, which is exactly what "we know nothing about this
# club yet" should mean; by g = 100 it is about .035, which is where a
# normal-normal posterior on a hundred games lands.
#
# **This is not the naive bootstrap, and the difference is not small.** The
# bootstrap standard error of the *shrunk estimator* — resample the club's
# games, re-shrink, look at the spread — is s*sqrt(g)/(g + k). That is the
# sampling variability of a statistic, not the posterior spread of the talent
# it estimates, and it is smaller by a factor sqrt(g/(g+k)) at every g. Worse,
# its shape is wrong: it goes to **zero** as g goes to zero, peaks at g = k,
# and so claims we are most certain about a club in the first week of April.
# The posterior version above is monotone in g, which is the only shape a
# statement about knowledge can have. Both were implemented; the bootstrap is
# available as `sampling="bootstrap"` and is scored in the doc as a control.
#
# **It is still a stand-in for a posterior, not a posterior.** A real one needs
# the Bayesian refit, which has never run end to end. This one carries no
# uncertainty from the bottom-up half of station C's blend (player rates,
# playing time, the rotation), and it treats a club's talent as fixed for the
# rest of the season — no trades, no injuries, no in-season drift. Both
# omissions push the same way, so the width here is a lower bound.
# See docs/parameter-uncertainty.md.

LOGIT_EPS = 1e-9
# Fall-back game-level standard deviation of a club's runs in one game, used
# only when fewer than two games are on the board (opening day). MLB runs per
# game per club sit near 4.5 with an SD near 3.1; the number matters for a
# single as-of date at the very start of a season and for nothing else.
PRIOR_RUN_SD = 3.1


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(p / (1.0 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def pythagenpat_array(rs, ra):
    """Vectorized `pythagenpat` on per-game run rates (`games = 1`)."""
    rs = np.clip(np.asarray(rs, dtype=float), 1e-6, None)
    ra = np.clip(np.asarray(ra, dtype=float), 1e-6, None)
    x = (rs + ra) ** PYTHAGENPAT_EXP
    a, b = rs ** x, ra ** x
    return a / (a + b)


def _pooled_game_cov(scored, allowed) -> np.ndarray:
    """League-pooled 2x2 covariance of one club-game's (runs for, runs against).

    Pooled rather than per club on purpose. This is a nuisance parameter — how
    variable one club-game's score is — and clubs differ in it far less than a
    100-game estimate of it differs from the truth. Pooling estimates it on
    ~2,500 club-games instead of ~100 and keeps a club's *width* a function of
    how much it has played, which is the thing the width is supposed to track.
    """
    s = [x for x in scored if len(x)]
    a = [x for x in allowed if len(x)]
    if not s:
        return np.diag([PRIOR_RUN_SD ** 2, PRIOR_RUN_SD ** 2])
    s, a = np.concatenate(s), np.concatenate(a)
    if len(s) < 2:
        return np.diag([PRIOR_RUN_SD ** 2, PRIOR_RUN_SD ** 2])
    return np.cov(np.vstack([s, a]), ddof=1)


@dataclass(frozen=True)
class RunRateSampling:
    """Per club: season-to-date run rates and the posterior width around them.

    `cov[i]` is the covariance of the club's *regressed* per-game (runs scored,
    runs allowed) — the pooled game-level covariance over `games[i] + ballast`.
    `sampling="bootstrap"` swaps in the sampling covariance of the shrunk
    estimator instead, which is the control the doc scores against.
    """
    team_ids: np.ndarray      # (T,)
    rs_pg: np.ndarray         # (T,) observed, unregressed
    ra_pg: np.ndarray         # (T,)
    games: np.ndarray         # (T,)
    cov: np.ndarray           # (T, 2, 2)
    game_cov: np.ndarray      # (2, 2) pooled, one club-game

    @property
    def chol(self) -> np.ndarray:
        """(T, 2, 2) lower-triangular factors, robust to a singular club."""
        out = np.zeros_like(self.cov)
        eye = 1e-12 * np.eye(2)
        for i in range(len(self.team_ids)):
            if not np.all(np.isfinite(self.cov[i])):
                continue
            try:
                out[i] = np.linalg.cholesky(self.cov[i] + eye)
            except np.linalg.LinAlgError:      # pragma: no cover - degenerate
                w, v = np.linalg.eigh(self.cov[i])
                out[i] = v @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
        return out


def run_rate_sampling(played: pd.DataFrame, team_ids,
                      ballast: float = 60.0,
                      sampling: str = "posterior") -> RunRateSampling:
    """Run rates and the width of the regressed estimate, per club.

    `played` is one row per completed game with `home_id`, `away_id`,
    `home_score`, `away_score` — the same frame the as-of standings are summed
    from, which is what keeps this walk-forward: a club's width at a cutoff
    comes from exactly the games its point estimate came from.

    `sampling` picks the width:

    * `"posterior"` (default) — `S / (g + ballast)`, the normal-normal
      posterior implied by the shrinkage the model already applies.
    * `"bootstrap"` — `S * g / (g + ballast)^2`, the sampling covariance of the
      shrunk estimator. Narrower everywhere and zero on opening day; kept
      because it is the obvious first thing to try and the doc scores it.
    """
    ids = np.asarray([int(t) for t in team_ids])
    n_t = len(ids)
    rs_pg, ra_pg, games = np.zeros(n_t), np.zeros(n_t), np.zeros(n_t)
    per_scored = [np.empty(0)] * n_t
    per_allowed = [np.empty(0)] * n_t
    if played is not None and len(played):
        h = played["home_id"].astype(int).to_numpy()
        a = played["away_id"].astype(int).to_numpy()
        hs = played["home_score"].astype(float).to_numpy()
        as_ = played["away_score"].astype(float).to_numpy()
        for i, t in enumerate(ids):
            at_home, at_away = h == t, a == t
            scored = np.concatenate([hs[at_home], as_[at_away]])
            allowed = np.concatenate([as_[at_home], hs[at_away]])
            games[i] = len(scored)
            per_scored[i], per_allowed[i] = scored, allowed
            if len(scored):
                rs_pg[i], ra_pg[i] = scored.mean(), allowed.mean()
    # Centre each club on its own mean so between-club talent does not inflate
    # the within-game variance this is meant to measure.
    game_cov = _pooled_game_cov(
        [x - x.mean() for x in per_scored if len(x) > 1],
        [x - x.mean() for x in per_allowed if len(x) > 1])

    k = float(ballast)
    if sampling == "posterior":
        denom = games + k
    elif sampling == "bootstrap":
        # Var(shrunk mean) = S*g/(g+k)^2, written as S/denom so both branches
        # share the line below. A club with no games gets no width.
        denom = np.where(games > 0,
                         (games + k) ** 2 / np.maximum(games, 1e-9), np.inf)
    else:
        raise ValueError(f"unknown sampling={sampling!r}")
    cov = game_cov[None, :, :] / denom[:, None, None]
    return RunRateSampling(ids, rs_pg, ra_pg, games, cov, game_cov)


@dataclass(frozen=True)
class StrengthDistribution:
    """A talent win% point estimate plus the width its own ballast implies.

    `point` is whatever strength vector the arm serves — station C's blend for
    the production chain, a flat .500 for the coin-flip control — and is
    returned unchanged by `draw` when `scale` is zero. The width is applied as
    a **logit-space deviation**: draw the club's regressed per-game runs scored
    and allowed from the posterior above, push them through the same
    Pythagenpat, and take the difference from the point regressed rates. Two
    properties follow and both matter:

    * the deviation is exactly zero when the draw equals the point, so
      `scale=0` reproduces the point-estimate path bit for bit, and
    * the deviation is a *shift*, so it composes with any point estimate — the
      blend, a flat .500, a preseason vector — rather than replacing it.

    `scale` multiplies the deviation. 1.0 is the width the model's own ballast
    implies and is not a fitted constant; other values exist so the width can
    be swept as a sensitivity.
    """
    point: pd.Series
    sampling: RunRateSampling
    regress_games: float = 60.0
    lg_rs: float = 4.5
    lg_ra: float = 4.5
    scale: float = 1.0

    def regressed_rates(self) -> tuple[np.ndarray, np.ndarray]:
        """The point regressed rates — `regressed_run_rates`, recomputed here."""
        g, k = self.sampling.games, float(self.regress_games)
        return ((self.sampling.rs_pg * g + k * self.lg_rs) / (g + k),
                (self.sampling.ra_pg * g + k * self.lg_ra) / (g + k))

    def point_array(self) -> np.ndarray:
        return self.point.to_numpy(dtype=float)

    def draw(self, n_sims: int, rng: np.random.Generator) -> np.ndarray:
        """(n_sims, n_teams) talent win%, one strength vector per season.

        Columns are aligned to `point.index`. With `scale == 0` this is the
        point estimate broadcast, with no draw taken from `rng` at all — the
        degenerate case is the existing model exactly, not an approximation.
        """
        base = self.point_array()
        n_sims = int(n_sims)
        if not self.scale:
            return np.broadcast_to(base, (n_sims, len(base))).copy()
        rs_r, ra_r = self.regressed_rates()
        L = self.sampling.chol                            # (T, 2, 2)
        z = rng.standard_normal((n_sims, len(base), 2))
        dev = np.einsum("tij,ntj->nti", L, z) * float(self.scale)
        drawn = pythagenpat_array(rs_r[None, :] + dev[:, :, 0],
                                  ra_r[None, :] + dev[:, :, 1])
        delta = logit(drawn) - logit(pythagenpat_array(rs_r, ra_r))[None, :]
        delta = np.where(np.isfinite(delta), delta, 0.0)
        return expit(logit(base)[None, :] + delta)

    def talent_sd(self, n: int = 4000, seed: int = 0) -> pd.Series:
        """Per club, the implied SD of talent win% — for reporting, not the sim."""
        d = self.draw(n, np.random.default_rng(seed))
        return pd.Series(d.std(0), index=self.point.index, name="talent_sd")


def strength_distribution(point: pd.Series, played: pd.DataFrame,
                          standings: pd.DataFrame,
                          regress_games: float = 60.0,
                          scale: float = 1.0,
                          sampling: str = "posterior") -> StrengthDistribution:
    """Build a `StrengthDistribution` around a served strength vector.

    `standings` supplies the league averages the regression pulls toward — the
    same two numbers `regressed_run_rates` computes — so the width is measured
    on the scale the point estimate lives on.
    """
    games = float((standings["wins"] + standings["losses"]).sum())
    lg_rs = float(standings["runs_scored"].sum() / max(games, 1.0))
    lg_ra = float(standings["runs_allowed"].sum() / max(games, 1.0))
    return StrengthDistribution(
        point=point.astype(float),
        sampling=run_rate_sampling(played, point.index,
                                   ballast=regress_games, sampling=sampling),
        regress_games=float(regress_games), lg_rs=lg_rs, lg_ra=lg_ra,
        scale=float(scale))
