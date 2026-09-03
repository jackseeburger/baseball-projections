"""Choose the expected-innings *level* constant walk-forward (issue #66).

The chain spends a starter's expected innings entirely on arithmetic: `ip` is
the weight that divides the game between `starters.blend_starter_team` and
`bullpen.blend_bullpen_team`, so a league-average starter expected to go four
and one expected to go seven are the same pitcher to it. A gradient-boosted
model given the chain's own inputs reproduced every other term and then put ten
to twenty times the chain's weight on exactly that number, in both scored
seasons (docs/market-benchmark-2026.md). `starters.starter_length_delta` is the
term that follows from it:

    ra9 += level_runs · (5.5 − ip) / 9

and `level_runs` is the one free constant this script chooses, on seasons the
term is *not* scored on — the gate rule in docs/architecture.md §3.

Where the games come from: `data/features/game_features_<season>.parquet`,
built by `scripts/build_game_features.py` off `game_model.build_slate` slates.
Those rows already carry every quantity the chain prices a game with — station
C's blended rates, the announced starter's FIP rate and expected innings, the
available pen as a delta — plus `chain_p`, the chain's own answer. So the whole
sweep is exact algebra on recorded pre-game inputs rather than a second
walk-forward pass, which is what makes *eleven* prior seasons affordable where
`sweep_reliever_usage.py` could only afford one. The script asserts it
reproduces the recorded `chain_p` before it sweeps anything; the tolerance is
1e-6, which is the float32 the parquet stores and not a modelling choice.

Three encodings are swept, because the learned model's preference is evidence
that the information is *there*, not that a level is the right way to spend it:

    level     ra9 += λ·(5.5 − ip)/9                    a main effect
    rate      ra9 −= (ip/9)·λ·(ip − 5.5)               the same, but charged
                                                       only over his own innings
    interact  ra9 += (ip/9)·λ·(ip − 5.5)·(sp − lg)     expected length as a
                                                       multiplier on the
                                                       starter's own quality,
                                                       with no main effect

Usage:
    python scripts/sweep_starter_ip_level.py --score-season 2026
    python scripts/sweep_starter_ip_level.py --score-season 2026 \
        --grid 0,0.25,0.5,0.75,1,1.25,1.5,2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.sim import starters as sp_model
from src.sim.strength import PYTHAGENPAT_EXP

FEATURE_DIR = Path("data/features")
ENCODINGS = ("level", "rate", "interact")


def load(seasons) -> pd.DataFrame:
    """Every recorded feature row for `seasons`, float64."""
    frames = []
    for s in seasons:
        path = FEATURE_DIR / f"game_features_{s}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path)
        frames.append(d.astype({c: "float64" for c in d.columns
                                if d[c].dtype == "float32"}))
    if not frames:
        raise SystemExit(f"no feature tables in {FEATURE_DIR}; run "
                         "scripts/build_game_features.py first")
    return pd.concat(frames, ignore_index=True)


def _pythagenpat(rs, ra):
    rs = np.maximum(rs, 0.5)
    ra = np.maximum(ra, 0.5)
    x = (rs + ra) ** PYTHAGENPAT_EXP
    return rs ** x / (rs ** x + ra ** x)


def _log5(p_home, p_away, hfa):
    p = p_home * (1 - p_away) / (p_home * (1 - p_away) + (1 - p_home) * p_away)
    odds = p / (1 - p) * (hfa / (1 - hfa))
    return odds / (1 + odds)


def side_ra9(d: pd.DataFrame, side: str, lam: float, mode: str) -> np.ndarray:
    """One club's runs allowed for the game: the chain, plus one encoding.

    The three lines below `base` are `starters.blend_starter_team` and
    `bullpen.blend_bullpen_team` written out on the columns the feature table
    recorded them from, which is why `lam = 0` has to reproduce `chain_p`.
    """
    ip = d[f"{side}_sp_ip"].to_numpy(float)
    sp = d[f"{side}_sp_ra9"].to_numpy(float)
    lg = d["lg_ra9"].to_numpy(float)
    pen = d[f"{side}_pen_delta"].to_numpy(float)
    prior, game_ip = sp_model.STARTER_IP, sp_model.GAME_IP
    base = (d[f"{side}_ra9"].to_numpy(float)
            + (ip / game_ip) * (sp - lg)
            + ((game_ip - ip) / game_ip) * pen)
    if mode == "level":
        return base + lam * (prior - ip) / game_ip
    if mode == "rate":
        return base - (ip / game_ip) * lam * (ip - prior)
    if mode == "interact":
        return base + (ip / game_ip) * lam * (ip - prior) * (sp - lg)
    raise ValueError(mode)


def probabilities(d: pd.DataFrame, lam: float = 0.0,
                  mode: str = "level") -> np.ndarray:
    home = _pythagenpat(d["home_rs9"].to_numpy(float), side_ra9(d, "home", lam, mode))
    away = _pythagenpat(d["away_rs9"].to_numpy(float), side_ra9(d, "away", lam, mode))
    return _log5(home, away, d["hfa_obs"].to_numpy(float))


def paired(new: np.ndarray, base: np.ndarray, y: np.ndarray):
    """Paired Brier difference (new − base), its standard error and t."""
    diff = (new - y) ** 2 - (base - y) ** 2
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return float(diff.mean()), float(se), float(diff.mean() / se), len(diff)


def max_likelihood_lambda(d: pd.DataFrame, mode: str = "level",
                          hi: float = 3.0, step: float = 0.01):
    """The one-parameter fit: λ that maximises the chain's own log likelihood.

    Nothing else is refit — not the blend weight, not the ballasts, not the
    home-field edge — so this is the size of the term the seasons ask for
    inside the model that already exists, and its curvature is a real standard
    error rather than the width of a grid.
    """
    y = d["home_win"].astype(float).to_numpy()

    def ll(lam):
        p = np.clip(probabilities(d, lam, mode), 1e-9, 1 - 1e-9)
        return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    grid = np.arange(0.0, hi + step / 2, step)
    best = float(grid[int(np.argmax([ll(x) for x in grid]))])
    h = 0.25
    curv = (ll(best + h) - 2 * ll(best) + ll(best - h)) / h ** 2
    se = float("inf") if curv >= 0 else 1.0 / np.sqrt(-curv * len(y))
    return best, se


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score-season", type=int, default=2026,
                    help="the season held out; the constant is chosen on the "
                         "completed seasons strictly before it")
    ap.add_argument("--first-season", type=int, default=2015)
    ap.add_argument("--grid", default="0,0.25,0.5,0.75,1,1.25,1.5,2",
                    help="runs per nine per missing inning of the flat 5.5")
    args = ap.parse_args()

    grid = [float(x) for x in args.grid.split(",")]
    seasons = range(args.first_season, args.score_season + 1)
    d = load(seasons)
    prior = d[d["season"] < args.score_season]

    # The table's `chain_p` is whatever its own slate priced, so a table built
    # before this term shipped carries the λ = 0 chain and one built after it
    # carries the served λ. Either reproduces exactly; which one it is says
    # what the recorded baseline means, and the sweep always measures against
    # the λ = 0 reconstruction below.
    recon = probabilities(d)
    stored = d["chain_p"].to_numpy(float)
    err = float(np.abs(recon - stored).max())
    err_served = float(np.abs(probabilities(d, sp_model.IP_LEVEL_RUNS) - stored).max())
    which = "without the level term" if err <= err_served else \
        f"with the level term at {sp_model.IP_LEVEL_RUNS}"
    print(f"{len(d)} recorded games, {seasons.start}-{args.score_season}; "
          f"the recorded chain_p is the chain {which}, reproduced to "
          f"{min(err, err_served):.2e}")
    assert min(err, err_served) < 1e-6, \
        "the sweep is not pricing the chain the table recorded"

    for mode in ENCODINGS:
        print(f"\n─── {mode} ─── paired Brier vs the chain, per season "
              f"(negative = better), x1e-5")
        head = f"{'season':>10} {'n':>6} " + "".join(f"{g:>9}" for g in grid[1:])
        print(head)
        for s, g in d.groupby("season"):
            y = g["home_win"].astype(float).to_numpy()
            base = probabilities(g)
            row = f"{int(s):>10} {len(g):>6} "
            for lam in grid[1:]:
                m, _se, _t, _n = paired(probabilities(g, lam, mode), base, y)
                row += f"{m * 1e5:>+9.2f}"
            print(row + ("   <- held out" if s == args.score_season else ""))
        y = prior["home_win"].astype(float).to_numpy()
        base = probabilities(prior)
        row = f"{'PRIOR POOL':>10} {len(prior):>6} "
        ts = []
        for lam in grid[1:]:
            m, se, t, _n = paired(probabilities(prior, lam, mode), base, y)
            row += f"{m * 1e5:>+9.2f}"
            ts.append(t)
        print(row)
        print(f"{'t':>10} {'':>6} " + "".join(f"{t:>+9.2f}" for t in ts))
        lam_ml, se_ml = max_likelihood_lambda(prior, mode)
        print(f"max-likelihood λ on {args.first_season}-{args.score_season - 1}: "
              f"{lam_ml:.2f} (se {se_ml:.2f}, t {lam_ml / se_ml:+.2f})")
        best = min(grid[1:], key=lambda x: paired(
            probabilities(prior, x, mode), base, y)[0])
        print(f"grid argmin on the same seasons: {best}")

    # ── the honest walk-forward: pick on everything before, score the season ──
    print("\n─── walk-forward: λ chosen on every completed season strictly "
          "before, scored on the season (level) ───")
    print(f"{'season':>8} {'λ*':>6} {'n':>6} {'chain':>9} {'+level':>9} "
          f"{'paired':>10} {'t':>7}")
    stacked = []
    for s in sorted(d["season"].unique())[1:]:
        past = d[d["season"] < s]
        yp = past["home_win"].astype(float).to_numpy()
        bp = probabilities(past)
        lam = min(grid[1:], key=lambda x: paired(probabilities(past, x), bp, yp)[0])
        cur = d[d["season"] == s]
        yc = cur["home_win"].astype(float).to_numpy()
        b0, b1 = probabilities(cur), probabilities(cur, lam)
        m, _se, t, n = paired(b1, b0, yc)
        stacked.append((b1 - yc) ** 2 - (b0 - yc) ** 2)
        print(f"{int(s):>8} {lam:>6.2f} {n:>6} {np.mean((b0-yc)**2):>9.5f} "
              f"{np.mean((b1-yc)**2):>9.5f} {m * 1e5:>+10.2f} {t:>+7.2f}")
    all_d = np.concatenate(stacked)
    se = all_d.std(ddof=1) / np.sqrt(len(all_d))
    print(f"{'pooled':>8} {'':>6} {len(all_d):>6} {'':>9} {'':>9} "
          f"{all_d.mean() * 1e5:>+10.2f} {all_d.mean() / se:>+7.2f}")

    print(f"\nshipped constant: starters.IP_LEVEL_RUNS = {sp_model.IP_LEVEL_RUNS}")


if __name__ == "__main__":
    main()
