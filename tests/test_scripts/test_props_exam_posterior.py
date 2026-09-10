"""BAS-70 selection: the walk-forward tau chooser and the matched-bet-count
comparison in scripts/props_exam.py.

These build synthetic pnl-frame-shaped data carrying `p_over_sd` — the
column BAS-70's other half (src/market/props.py, built in parallel and not
in this worktree) is expected to add — rather than running the real props
archive end to end.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "props_exam", ROOT / "scripts/props_exam.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


props_exam = _load()
from src.market import pnl  # noqa: E402


def synthetic_frame(n_per_half=300, seed=0):
    """A pnl-frame-shaped table with a first half where a tight tau wins and
    a second half where a loose tau wins — so the chooser has something to
    get wrong if it looks at the wrong half.

    Every row is a genuine edge (`p_model` on the correct side of the
    market) with a Normal-ish `p_over_sd`; win/loss is drawn so that rows
    with a *small* sd win more often in the first half (rewarding a tight
    tau there) and rows with a *large* sd win more often in the second half
    (rewarding a loose tau there) — engineered, not organic, purely to give
    the chooser a half-dependent answer to find.
    """
    rng = np.random.default_rng(seed)

    def half(date_prefix, n, favor_tight):
        mid = rng.uniform(0.3, 0.7, n)
        half_spread = 0.02
        bid, ask = mid - half_spread, mid + half_spread
        side_up = rng.choice([True, False], n)
        edge_mag = rng.uniform(0.03, 0.08, n)
        p_model = np.where(side_up, ask + edge_mag, bid - edge_mag)
        p_model = np.clip(p_model, 0.02, 0.98)
        sd = rng.uniform(0.01, 0.12, n)
        # Win probability: correlated with small sd in the "tight" half,
        # with large sd in the "loose" half — engineered so ROI genuinely
        # differs by tau between halves.
        tight_bonus = (0.12 - sd) / 0.12          # in [0, 1], big when sd small
        loose_bonus = (sd - 0.01) / 0.11           # in [0, 1], big when sd big
        p_win = 0.5 + 0.35 * (tight_bonus if favor_tight else loose_bonus)
        home_won_up = rng.uniform(0, 1, n) < p_win
        home_won = np.where(side_up, home_won_up, ~home_won_up)
        return pd.DataFrame({
            "date": [f"{date_prefix}-{i % 27 + 1:02d}" for i in range(n)],
            "game_pk": np.arange(n) + (0 if date_prefix.endswith("07") else 10_000),
            "prop_stat": rng.choice(["hits", "hr"], n),
            "model": p_model,
            "p_home_close": mid,
            "bid": bid, "ask": ask,
            "home_win": home_won,
            "p_over_sd": sd,
        })

    first = half("2026-07", n_per_half, favor_tight=True)
    second = half("2026-08", n_per_half, favor_tight=False)
    return pd.concat([first, second], ignore_index=True), "2026-08-01"


# ───────────────────────────── choose_tau ─────────────────────────────

def test_choose_tau_only_looks_at_the_first_half():
    frame, cut = synthetic_frame()
    venue = props_exam.venue_for(0.0, False)
    tau, table = props_exam.choose_tau(frame, "model", "p_over_sd", cut, venue,
                                       grid=props_exam.TAU_GRID)
    assert not np.isnan(tau)
    assert tau in props_exam.TAU_GRID

    # Sanity: the table itself is computed only on rows before `cut`.
    train = frame[frame["date"].astype(str) < cut]
    row = pnl.evaluate(train, "model", venue, staking="flat",
                       group_col="game_pk", rule="posterior", tau=tau,
                       sd_col="p_over_sd")
    matching = table[np.isclose(table["tau"], tau)]
    assert len(matching) == 1
    assert matching.iloc[0]["n_bets"] == row["n_bets"]
    assert matching.iloc[0]["roi"] == pytest.approx(row["roi"])

    # The chooser must be blind to the second half: replacing it entirely
    # (with rows that would favour a very different tau) does not change
    # the choice, because the first half alone determines it.
    poisoned_second = frame[frame["date"].astype(str) >= cut].copy()
    poisoned_second["model"] = 1.0 - poisoned_second["model"]  # nonsense
    poisoned = pd.concat([frame[frame["date"].astype(str) < cut],
                          poisoned_second], ignore_index=True)
    tau2, _ = props_exam.choose_tau(poisoned, "model", "p_over_sd", cut, venue,
                                    grid=props_exam.TAU_GRID)
    assert tau2 == tau


def test_choose_tau_picks_the_half_where_the_best_tau_actually_differs():
    # A frame explicitly engineered so the best tau on the first half is
    # different from the best tau on the second half; the chooser must
    # report the first half's answer.
    rng = np.random.default_rng(7)
    n = 400
    mid = rng.uniform(0.3, 0.7, n)
    bid, ask = mid - 0.02, mid + 0.02
    side_up = rng.choice([True, False], n)
    p_model = np.where(side_up, ask + 0.05, bid - 0.05)
    sd_first = np.full(n, 0.01)   # first half: tiny sd -> any tau clears easily
    sd_second = np.full(n, 0.20)  # second half: huge sd -> only low tau clears

    def frame_half(prefix, sd, win_rate):
        home_won_up = rng.uniform(0, 1, n) < win_rate
        home_won = np.where(side_up, home_won_up, ~home_won_up)
        return pd.DataFrame({
            "date": [f"{prefix}-{i % 27 + 1:02d}" for i in range(n)],
            "game_pk": np.arange(n) + (0 if prefix == "2026-07" else 10_000),
            "model": p_model, "p_home_close": mid, "bid": bid, "ask": ask,
            "home_win": home_won, "p_over_sd": sd,
        })

    first = frame_half("2026-07", sd_first, 0.62)
    second = frame_half("2026-08", sd_second, 0.62)
    frame = pd.concat([first, second], ignore_index=True)
    cut = "2026-08-01"
    venue = props_exam.venue_for(0.0, False)

    # On the first half every tau in the grid clears (sd tiny), so ROI is
    # flat across taus there and the chooser's tie-break keeps the least
    # selective (largest bet count) tau -> the grid's smallest tau.
    tau, table = props_exam.choose_tau(first, "model", "p_over_sd", cut, venue,
                                       grid=props_exam.TAU_GRID)
    assert tau == min(props_exam.TAU_GRID)

    # Feeding the chooser only the second half instead gives a different
    # answer (or at least a different bet-count profile at the high end of
    # the grid, since the huge sd there strands the strict taus at 0 bets).
    tau_on_second, table_second = props_exam.choose_tau(
        second, "model", "p_over_sd", "2026-09-01", venue, grid=props_exam.TAU_GRID)
    high_tau_row = table_second[np.isclose(table_second["tau"], max(props_exam.TAU_GRID))]
    assert int(high_tau_row.iloc[0]["n_bets"]) == 0

    # And on the combined frame, cut correctly at 2026-08-01, the chooser
    # reproduces the first-half-only answer, not something in between.
    tau_combined, _ = props_exam.choose_tau(frame, "model", "p_over_sd", cut,
                                            venue, grid=props_exam.TAU_GRID)
    assert tau_combined == tau


# ───────────────────────────── matched_threshold ─────────────────────────────

def test_matched_threshold_finds_a_comparable_bet_count():
    frame, cut = synthetic_frame()
    second = frame[frame["date"].astype(str) >= cut].reset_index(drop=True)
    venue = props_exam.venue_for(0.0, False)
    target = 40
    t = props_exam.matched_threshold(second, "model", venue, target)
    n = len(pnl.bet_frame(second, "model", venue, threshold=t))
    # Bisection on a step function cannot always land exactly on the target;
    # it should get close, and never wildly off for a target well inside the
    # achievable range.
    assert abs(n - target) <= max(5, int(0.1 * target))


def test_matched_threshold_zero_target_gives_zero_bets_at_some_threshold():
    frame, cut = synthetic_frame()
    second = frame[frame["date"].astype(str) >= cut].reset_index(drop=True)
    venue = props_exam.venue_for(0.0, False)
    t = props_exam.matched_threshold(second, "model", venue, 0, hi=0.30)
    n = len(pnl.bet_frame(second, "model", venue, threshold=t))
    assert n == 0


# ───────────────────────────── posterior_comparison ─────────────────────────────

def test_posterior_comparison_has_matched_bet_count_between_posterior_and_matched_rows():
    frame, cut = synthetic_frame()
    second = frame[frame["date"].astype(str) >= cut].reset_index(drop=True)
    fee_waived = props_exam.venue_for(0.0, False)
    as_quoted = pnl.KALSHI
    table = props_exam.posterior_comparison(second, "model", "p_over_sd", tau=0.7,
                                            threshold=0.02,
                                            venue_fee_waived=fee_waived,
                                            venue_as_quoted=as_quoted,
                                            draws=200, seed=0)
    assert list(table["rule"].str.startswith(("threshold @ 2pt", "posterior",
                                              "threshold @ matched")))
    posterior_row = table[table["rule"].str.startswith("posterior")].iloc[0]
    matched_row = table[table["rule"].str.startswith("threshold @ matched")].iloc[0]
    assert abs(int(posterior_row["n_bets"]) - int(matched_row["n_bets"])) <= \
        max(5, int(0.1 * max(int(posterior_row["n_bets"]), 1)))
    # fee-waived ROI is never worse than as-quoted for the same rule (the fee
    # only ever costs money on a winning-or-losing settle, never helps).
    for r in table.itertuples(index=False):
        if r.n_bets > 0:
            assert r.roi_fee_waived >= r.roi_as_quoted - 1e-9


def test_posterior_comparison_by_stat_reports_pooled_and_per_stat():
    frame, cut = synthetic_frame()
    second = frame[frame["date"].astype(str) >= cut].reset_index(drop=True)
    fee_waived = props_exam.venue_for(0.0, False)
    table = props_exam.posterior_comparison_by_stat(
        second, "model", "p_over_sd", tau=0.7, threshold=0.02,
        venue_fee_waived=fee_waived, venue_as_quoted=pnl.KALSHI,
        draws=200, seed=0)
    stats = set(table["stat"])
    assert "all" in stats
    assert stats - {"all"} == set(second["prop_stat"].unique())
    assert len(table) == 3 * len(stats)   # three rules per stat group


# ───────────────────────────── --whole-window ─────────────────────────────
#
# BAS-93 scores the July contracts with tau and the matchup weight frozen at
# the values BAS-70 chose on a different window. Nothing is chosen on July, so
# there is nothing to hold July's second half out from, and the split would
# only halve the sample. `--whole-window` says so explicitly and refuses to run
# unless every free constant really is fixed on the command line.

def test_scored_rows_defaults_to_the_second_half():
    frame, cut = synthetic_frame()
    rows, span = props_exam.scored_rows(frame, cut, whole_window=False)
    assert span == "second half"
    assert len(rows) < len(frame)
    assert (rows["date"].astype(str) >= cut).all()


def test_scored_rows_whole_window_keeps_every_row():
    frame, cut = synthetic_frame()
    rows, span = props_exam.scored_rows(frame, cut, whole_window=True)
    assert span == "whole window"
    assert len(rows) == len(frame)
    assert (rows["date"].astype(str) < cut).any()   # the first half is back


def test_scored_rows_whole_window_scores_more_bets_than_the_second_half():
    """The point of the flag: the same frozen rule, on twice the rows."""
    frame, cut = synthetic_frame()
    fee_waived = props_exam.venue_for(0.0, False)
    half, _ = props_exam.scored_rows(frame, cut, whole_window=False)
    whole, _ = props_exam.scored_rows(frame, cut, whole_window=True)
    n_half = len(pnl.bet_frame(half.reset_index(drop=True), "model",
                               fee_waived, threshold=0.02))
    n_whole = len(pnl.bet_frame(whole.reset_index(drop=True), "model",
                                fee_waived, threshold=0.02))
    assert n_whole > n_half


def _cli(*args):
    """Run the script's argument parsing only — these all exit before any
    parquet is opened, so the test needs neither the archive nor a network."""
    return subprocess.run([sys.executable, str(ROOT / "scripts/props_exam.py"), *args],
                          capture_output=True, text=True, timeout=120)


def test_whole_window_refuses_without_a_fixed_tau():
    r = _cli("--whole-window", "--matchup-weight", "1.0")
    assert r.returncode != 0
    assert "--whole-window needs --tau" in r.stderr


def test_whole_window_refuses_without_a_fixed_matchup_weight():
    r = _cli("--whole-window", "--tau", "0.65", "--matchup", "on")
    assert r.returncode != 0
    assert "--whole-window needs --matchup-weight" in r.stderr


def test_whole_window_allows_the_matchup_arm_off_without_a_weight():
    """With the matchup arm off there is no weight to fix, so only tau is
    required; the run gets past the guard and fails later on missing data."""
    r = _cli("--whole-window", "--tau", "0.65", "--matchup", "off",
             "--closes", str(ROOT / "does-not-exist.parquet"))
    assert "--whole-window needs" not in r.stderr
