"""Pins the correction in docs/posterior-props.md: for a one-shot binary
contract, the stake that maximises *posterior*-expected log growth is
exactly `kelly_stake` evaluated at the posterior mean — not something
shaded down for parameter uncertainty.

modelling-roadmap.md §1 claimed "Kelly under parameter uncertainty is
provably not Kelly at the mean; the correct stake shades down." That is
true for some problems (repeated bets against one persistent unknown `p`,
or a payoff nonlinear in `p`) but false for the thing this repo actually
prices: a single binary contract bought at cost `c`, paying $1. There,

    E[log W | p] = p · log(1 + f(1−c)/c) + (1 − p) · log(1 − f)

is *linear* in `p`, so ``E_q[E[log W | p]] = E[log W | E_q[p]]`` for any
posterior `q` — the plug-in stake at the mean is already the posterior
optimum, with no correction term. This test does not trust that algebra;
it numerically maximises the posterior-expected log growth over `f` for
several Beta posteriors and checks the maximiser lands on `kelly_stake` at
the posterior mean.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_dist, qmc

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.market import pnl

# (alpha, beta) pairs spanning posterior means 0.55-0.70 and sds ~0.01-0.15.
# A Beta(a, b) has mean a/(a+b) and sd sqrt(ab / ((a+b)^2 (a+b+1))); these
# were picked to hit that range with a mix of tight (large a+b, many
# pseudo-observations) and loose (small a+b) posteriors.
BETA_PARAMS = [
    (5_500, 4_500),     # mean .55, sd ~0.0050 (tight — lots of at-bats)
    (55, 45),            # mean .55, sd ~0.0495
    (11, 9),              # mean .55, sd ~0.1086 (loose — a handful of PAs)
    (1_200, 800),        # mean .60, sd ~0.0110
    (60, 40),             # mean .60, sd ~0.0487
    (6, 4),                # mean .60, sd ~0.1477
    (2_100, 900),        # mean .70, sd ~0.0084
    (70, 30),             # mean .70, sd ~0.0456
    (14, 6),               # mean .70, sd ~0.1000
]

COST = 0.40    # an arbitrary but representative contract price


def posterior_expected_log_growth(f, a, b, cost, draws):
    """Monte-Carlo E_q[log W(f)] for q = Beta(a, b), used only to locate the
    optimum numerically — not the closed form the test is checking against.
    """
    if f <= 0.0 or f >= 1.0:
        return 1e9  # keep the optimizer inside the feasible stake range
    p = draws
    win = np.log1p(f * (1 - cost) / cost)
    lose = np.log1p(-f)
    return -float(np.mean(p * win + (1 - p) * lose))  # negated: we minimize


@pytest.mark.parametrize("a,b", BETA_PARAMS)
def test_posterior_optimal_stake_is_kelly_at_the_mean(a, b):
    mean = a / (a + b)
    sd = (a * b / ((a + b) ** 2 * (a + b + 1))) ** 0.5
    assert 0.55 - 1e-6 <= mean <= 0.70 + 1e-6
    assert 0.004 <= sd <= 0.15

    # Several seeds, large draw count: the numerical optimum has to be
    # stable, not an artefact of one Monte-Carlo sample. Plain pseudo-random
    # draws need tens of millions of points to pin f to 1e-4 (the log-growth
    # integrand is smooth, so most of that is wasted); scrambled Sobol draws
    # transformed through the Beta's inverse CDF converge far faster for the
    # same point count, so 2**20 points per seed is enough.
    optimal_fs = []
    for seed in (0, 1, 2):
        sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
        u = sampler.random_base2(m=20).ravel()
        draws = beta_dist.ppf(u, a, b)
        result = minimize_scalar(
            posterior_expected_log_growth, args=(a, b, COST, draws),
            bounds=(1e-6, 1.0 - 1e-6), method="bounded",
            options={"xatol": 1e-10})
        optimal_fs.append(result.x)

    numerical = float(np.mean(optimal_fs))
    plugin = pnl.kelly_stake(mean, COST, fraction=1.0, cap=1.0, bankroll=1.0)
    assert numerical == pytest.approx(float(plugin), abs=1e-4)
    # And the seeds agree with each other to well inside that tolerance —
    # the numerical search, not just its mean, is stable.
    assert max(optimal_fs) - min(optimal_fs) < 2e-4
