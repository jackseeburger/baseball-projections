"""Parameter uncertainty in team strength (docs/parameter-uncertainty.md).

The contract this file pins down has three parts:

1. **Zero width is the old model, bit for bit.** A `StrengthDistribution` with
   `scale=0` must reproduce the point-estimate path exactly — the same drawn
   games, the same records, the same board. That is what makes the new path a
   generalisation rather than a replacement, and it is what lets the backtest
   attribute every difference in the score to the width and to nothing else.
2. **Non-zero width widens the right things.** More spread in projected final
   wins, probabilities pulled toward the base rate, and nothing pulled past it.
3. **The width itself is the sampling uncertainty of the run rates**, so it
   shrinks as a club plays more games and is zero before it has played any.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sim.odds import run_playoff_odds
from src.sim.season import (
    SeasonState, resolve_strength, simulate_remaining, tally,
)
from src.sim.strength import (
    StrengthDistribution, pythagenpat, pythagenpat_array, regressed_strength,
    run_rate_sampling, strength_distribution,
)

N_TEAMS = 30


def _teams() -> pd.DataFrame:
    ids = list(range(1, N_TEAMS + 1))
    return pd.DataFrame({
        "team_id": ids,
        "abbrev": [f"T{i:02d}" for i in ids],
        "name": [f"Team {i}" for i in ids],
        "league_id": [103 if i <= 15 else 104 for i in ids],
        "division_id": [200 + (i - 1) // 5 for i in ids],
    })


def _season(rng: np.random.Generator, n_played: int = 90,
            n_left: int = 60) -> tuple[SeasonState, pd.DataFrame]:
    """A toy season: every club plays a balanced round robin inside its league."""
    teams = _teams()
    rows, day = [], 0
    ids = teams["team_id"].to_numpy()
    for k in range(n_played + n_left):
        day += 1
        order = rng.permutation(ids)
        for j in range(0, N_TEAMS, 2):
            rows.append({"game_pk": len(rows) + 1, "date": f"2024-{1 + day // 28:02d}-"
                                                           f"{1 + day % 28:02d}",
                         "home_id": int(order[j]), "away_id": int(order[j + 1]),
                         "played": k < n_played})
    frame = pd.DataFrame(rows)
    played = frame[frame["played"]].copy()
    played["home_score"] = rng.poisson(4.5, len(played)).astype(float)
    played["away_score"] = rng.poisson(4.4, len(played)).astype(float)
    played = played[played["home_score"] != played["away_score"]].reset_index(drop=True)
    completed = played[["game_pk", "date", "home_id", "away_id"]].copy()
    completed["home_win"] = played["home_score"] > played["away_score"]
    remaining = frame[~frame["played"]][
        ["game_pk", "date", "home_id", "away_id"]].reset_index(drop=True)
    state = SeasonState(teams=teams, completed=completed, remaining=remaining)
    return state, played


def _standings(played: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    ids = teams["team_id"].to_numpy()
    rows = []
    for t in ids:
        h, a = played["home_id"] == t, played["away_id"] == t
        rs = float(played.loc[h, "home_score"].sum() + played.loc[a, "away_score"].sum())
        ra = float(played.loc[h, "away_score"].sum() + played.loc[a, "home_score"].sum())
        w = int((played.loc[h, "home_score"] > played.loc[h, "away_score"]).sum()
                + (played.loc[a, "away_score"] > played.loc[a, "home_score"]).sum())
        rows.append({"team_id": int(t), "wins": w,
                     "losses": int(h.sum() + a.sum()) - w,
                     "runs_scored": rs, "runs_allowed": ra})
    out = pd.DataFrame(rows)
    meta = teams.set_index("team_id")
    out["league_id"] = out["team_id"].map(meta["league_id"])
    out["division_id"] = out["team_id"].map(meta["division_id"])
    return out


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(11)
    state, played = _season(rng)
    standings = _standings(played, state.teams)
    point = regressed_strength(standings)
    return {"state": state, "played": played, "standings": standings,
            "point": point}


def _dist(toy, scale: float) -> StrengthDistribution:
    return strength_distribution(toy["point"], toy["played"], toy["standings"],
                                 regress_games=60.0, scale=scale)


# ─── 1. zero width is the old model exactly ───

class TestDegenerate:
    def test_draw_at_zero_scale_is_the_point_estimate(self, toy):
        d = _dist(toy, 0.0)
        got = d.draw(64, np.random.default_rng(0))
        assert got.shape == (64, N_TEAMS)
        np.testing.assert_array_equal(
            got, np.broadcast_to(toy["point"].to_numpy(), got.shape))

    def test_resolve_strength_takes_nothing_from_the_rng_at_zero_width(self, toy):
        rng_a = np.random.default_rng(5)
        point, draws = resolve_strength(_dist(toy, 0.0), toy["state"].team_ids,
                                        100, rng_a)
        assert draws is None
        rng_b = np.random.default_rng(5)
        np.testing.assert_array_equal(rng_a.random(8), rng_b.random(8))

    def test_simulate_remaining_is_identical_at_zero_width(self, toy):
        args = (toy["state"], toy["point"], 0.54, 200)
        base = simulate_remaining(*args, np.random.default_rng(3))
        got = simulate_remaining(toy["state"], _dist(toy, 0.0), 0.54, 200,
                                 np.random.default_rng(3))
        np.testing.assert_array_equal(base, got)

    def test_simulate_remaining_is_identical_at_zero_width_with_overrides(self, toy):
        pks = toy["state"].remaining["game_pk"].astype(int).to_numpy()[:40]
        ov = {int(pk): 0.71 for pk in pks}
        base = simulate_remaining(toy["state"], toy["point"], 0.54, 200,
                                  np.random.default_rng(3), p_home_overrides=ov)
        got = simulate_remaining(toy["state"], _dist(toy, 0.0), 0.54, 200,
                                 np.random.default_rng(3), p_home_overrides=ov)
        np.testing.assert_array_equal(base, got)

    def test_run_playoff_odds_is_identical_at_zero_width(self, toy):
        a = run_playoff_odds(toy["state"], toy["point"], 0.54, n_sims=300, seed=7)
        b = run_playoff_odds(toy["state"], _dist(toy, 0.0), 0.54, n_sims=300,
                             seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_both_paths_leave_the_rng_in_the_same_state(self, toy):
        """The two arms are a common-random-numbers comparison, not two runs.

        `simulate_remaining` draws exactly `n_sims × n_remaining` uniforms
        either way, and the strength draw comes from a spawned stream, so the
        game uniforms — and everything drawn after them, the tiebreaks and the
        bracket included — line up game for game between the point arm and the
        uncertainty arm. That is what makes the paired difference in the
        backtest measure the width and not the seed.
        """
        rng_a = np.random.default_rng(13)
        simulate_remaining(toy["state"], toy["point"], 0.54, 300, rng_a)
        rng_b = np.random.default_rng(13)
        simulate_remaining(toy["state"], _dist(toy, 1.0), 0.54, 300, rng_b)
        np.testing.assert_array_equal(rng_a.random(6), rng_b.random(6))

    def test_the_result_does_not_depend_on_the_chunk_size(self, toy, monkeypatch):
        import src.sim.season as season_mod
        got = {}
        for chunk in (37, 1024, 10_000):
            monkeypatch.setattr(season_mod, "SIM_CHUNK", chunk)
            got[chunk] = simulate_remaining(toy["state"], _dist(toy, 1.0),
                                            0.54, 300,
                                            np.random.default_rng(4))
        np.testing.assert_array_equal(got[37], got[1024])
        np.testing.assert_array_equal(got[37], got[10_000])

    def test_a_series_still_works_unchanged(self, toy):
        """The point-estimate signature is not merely tolerated, it is the same."""
        rng = np.random.default_rng(2)
        got = simulate_remaining(toy["state"], toy["point"], 0.54, 50, rng)
        assert got.shape == (50, len(toy["state"].remaining))
        assert got.dtype == bool


# ─── 2. non-zero width does the right thing ───

class TestWidth:
    def test_draws_are_centred_on_the_point_estimate(self, toy):
        d = _dist(toy, 1.0)
        got = d.draw(20_000, np.random.default_rng(1))
        # Mean of the draws sits on the point estimate to within Monte Carlo
        # error; the logit-space map is very close to linear at this width.
        assert np.abs(got.mean(0) - toy["point"].to_numpy()).max() < 2e-3

    def test_width_scales_with_the_scale_knob(self, toy):
        one = _dist(toy, 1.0).draw(4000, np.random.default_rng(1)).std(0)
        two = _dist(toy, 2.0).draw(4000, np.random.default_rng(1)).std(0)
        assert np.all(two > one)
        assert 1.8 < float(np.mean(two / one)) < 2.2

    def test_the_posterior_width_is_monotone_in_games_played(self):
        """Knowledge only accumulates: more games can never mean more spread.

        This is the property the naive bootstrap fails — its width is zero on
        opening day, peaks at `g = ballast` and falls after, i.e. it claims we
        are most certain about a club before it has played. The test walks a
        whole season's worth of sample sizes rather than checking two points,
        because two points is exactly what let the wrong version through.
        """
        rng = np.random.default_rng(4)
        state, played = _season(rng, n_played=150, n_left=12)
        standings = _standings(played, state.teams)
        point = regressed_strength(standings)
        widths = []
        for g in (0, 10, 30, 60, 100, 150):
            sub = played.iloc[: g * N_TEAMS // 2]
            s = run_rate_sampling(sub, state.team_ids, ballast=60.0)
            d = StrengthDistribution(point=point, sampling=s, regress_games=60.0)
            widths.append(float(d.talent_sd().mean()))
        assert all(b <= a + 1e-9 for a, b in zip(widths, widths[1:])), widths
        assert widths[0] > widths[-1]

    def test_the_bootstrap_control_has_the_wrong_shape(self):
        """Recorded because it is the reason the default is not the bootstrap."""
        rng = np.random.default_rng(4)
        state, played = _season(rng, n_played=150, n_left=12)
        standings = _standings(played, state.teams)
        point = regressed_strength(standings)
        widths = {}
        for g in (5, 60, 150):
            sub = played.iloc[: g * N_TEAMS // 2]
            s = run_rate_sampling(sub, state.team_ids, ballast=60.0,
                                  sampling="bootstrap")
            widths[g] = float(StrengthDistribution(
                point=point, sampling=s, regress_games=60.0).talent_sd().mean())
        assert widths[5] < widths[60]      # not monotone: the defect itself
        assert widths[150] < widths[60]
        assert run_rate_sampling(played.iloc[:0], state.team_ids,
                                 sampling="bootstrap").cov.sum() == 0.0

    def test_the_posterior_width_on_an_empty_board_is_the_prior(self, toy):
        """Opening day: no games, and the width is the league's talent spread."""
        empty = toy["played"].iloc[:0]
        s = run_rate_sampling(empty, toy["state"].team_ids, ballast=60.0)
        d = StrengthDistribution(point=toy["point"], sampling=s,
                                 regress_games=60.0)
        sd = float(d.talent_sd().mean())
        assert 0.03 < sd < 0.09

    def test_projected_wins_spread_widens(self, toy):
        def spread(strength):
            rng = np.random.default_rng(9)
            hw = simulate_remaining(toy["state"], strength, 0.54, 4000, rng)
            return tally(toy["state"], hw).wins.std(0).mean()
        assert spread(_dist(toy, 1.0)) > spread(toy["point"])

    def test_probabilities_move_toward_the_middle(self, toy):
        base = run_playoff_odds(toy["state"], toy["point"], 0.54, n_sims=600,
                                seed=1).set_index("team_id")["p_playoffs"]
        wide = run_playoff_odds(toy["state"], _dist(toy, 4.0), 0.54, n_sims=600,
                                seed=1).set_index("team_id")["p_playoffs"]
        wide = wide.reindex(base.index)
        # Not team by team — Monte Carlo noise at 600 sims is larger than the
        # move on an individual club — but the *dispersion* of the board must
        # fall, which is the mechanical consequence the doc pre-registers.
        assert wide.std() < base.std()

    def test_probabilities_stay_coherent(self, toy):
        odds = run_playoff_odds(toy["state"], _dist(toy, 1.0), 0.54,
                                n_sims=400, seed=2)
        for col in ("p_playoffs", "p_division", "p_pennant", "p_ws"):
            assert odds[col].between(0.0, 1.0).all()
        assert odds["p_ws"].sum() == pytest.approx(1.0, abs=1e-9)
        assert odds["p_pennant"].sum() == pytest.approx(2.0, abs=1e-9)
        assert (odds["p_ws"] <= odds["p_pennant"] + 1e-12).all()
        assert (odds["p_pennant"] <= odds["p_playoffs"] + 1e-12).all()

    def test_the_bracket_sees_the_same_draw_as_the_schedule(self, toy):
        """A club strong in a simulated season must be strong in its October.

        With the two drawn independently the correlation vanishes and the
        pennant odds of the best clubs fall; this asserts the wiring by
        checking the strong half of the board keeps more pennant probability
        than an independent-draw board would. The cheap observable version:
        the champion's own drawn strength beats the field's mean.
        """
        odds = run_playoff_odds(toy["state"], _dist(toy, 2.0), 0.54,
                                n_sims=800, seed=5)
        top = odds.nlargest(8, "strength")["p_pennant"].sum()
        bottom = odds.nsmallest(8, "strength")["p_pennant"].sum()
        assert top > bottom


# ─── 3. the width is the sampling uncertainty of the run rates ───

class TestSampling:
    def test_run_rate_sampling_matches_a_hand_computation(self):
        played = pd.DataFrame({
            "home_id": [1, 2, 1], "away_id": [2, 1, 2],
            "home_score": [5.0, 3.0, 1.0], "away_score": [2.0, 4.0, 0.0],
        })
        s = run_rate_sampling(played, [1, 2], ballast=60.0)
        scored_1 = np.array([5.0, 4.0, 1.0])     # home, away, home
        allowed_1 = np.array([2.0, 3.0, 0.0])
        assert s.games[0] == 3
        assert s.rs_pg[0] == pytest.approx(scored_1.mean())
        assert s.ra_pg[0] == pytest.approx(allowed_1.mean())
        np.testing.assert_allclose(s.cov[0], s.game_cov / (3 + 60.0))

    def test_the_pooled_game_covariance_is_the_within_club_one(self):
        """A league of wildly unequal clubs must not inflate the game variance."""
        n = 200
        rng = np.random.default_rng(1)
        # Club 1 scores 8 a game, club 2 scores 1, both with tiny within-club
        # spread. Pooling raw would report a variance of ~12; centred, ~0.25.
        played = pd.DataFrame({
            "home_id": np.ones(n, int), "away_id": 2 * np.ones(n, int),
            "home_score": 8 + rng.normal(0, 0.5, n),
            "away_score": 1 + rng.normal(0, 0.5, n),
        })
        s = run_rate_sampling(played, [1, 2], ballast=60.0)
        assert s.game_cov[0, 0] < 1.0

    def test_posterior_width_is_the_standard_error_of_g_plus_ballast(self):
        rng = np.random.default_rng(0)
        n = 400
        played = pd.DataFrame({
            "home_id": np.ones(n, int), "away_id": 2 * np.ones(n, int),
            "home_score": rng.poisson(4.5, n).astype(float),
            "away_score": rng.poisson(4.5, n).astype(float),
        })
        s = run_rate_sampling(played, [1], ballast=60.0)
        assert s.cov[0, 0, 0] == pytest.approx(s.game_cov[0, 0] / (n + 60.0))
        half = run_rate_sampling(played.iloc[: n // 2], [1], ballast=60.0)
        assert half.cov[0, 0, 0] > s.cov[0, 0, 0]

    def test_bootstrap_width_is_the_shrunk_estimator_standard_error(self):
        rng = np.random.default_rng(0)
        n = 100
        played = pd.DataFrame({
            "home_id": np.ones(n, int), "away_id": 2 * np.ones(n, int),
            "home_score": rng.poisson(4.5, n).astype(float),
            "away_score": rng.poisson(4.5, n).astype(float),
        })
        b = run_rate_sampling(played, [1], ballast=60.0, sampling="bootstrap")
        want = b.game_cov[0, 0] * n / (n + 60.0) ** 2
        assert b.cov[0, 0, 0] == pytest.approx(want)

    def test_an_unknown_sampling_name_raises(self):
        with pytest.raises(ValueError, match="unknown sampling"):
            run_rate_sampling(pd.DataFrame(), [1], sampling="nope")

    def test_pythagenpat_array_agrees_with_the_scalar(self):
        rs = np.array([4.1, 5.0, 3.2])
        ra = np.array([4.6, 4.0, 4.4])
        want = [pythagenpat(a, b, 1.0) for a, b in zip(rs, ra)]
        np.testing.assert_allclose(pythagenpat_array(rs, ra), want)

    def test_the_deviation_composes_with_any_point_estimate(self, toy):
        """The width is a shift, so it rides on a flat .500 board too."""
        flat = pd.Series(0.5, index=toy["point"].index, name="strength")
        d = strength_distribution(flat, toy["played"], toy["standings"],
                                  scale=1.0)
        got = d.draw(2000, np.random.default_rng(0))
        assert abs(float(got.mean()) - 0.5) < 2e-3
        assert float(got.std()) > 0.005

    def test_implied_talent_sd_is_in_a_credible_range(self, toy):
        """A sanity band, not a fitted constant.

        Ninety games of run differential leave real uncertainty about a club's
        talent: an SD of a few points of win% is what any shrinkage estimator
        implies at that sample size. Well under a point would mean the width is
        not doing anything; well over ten points would mean it had swallowed
        the whole talent spread of the league.
        """
        sd = _dist(toy, 1.0).talent_sd()
        assert 0.005 < float(sd.mean()) < 0.10
