"""Team defence — the balls in play FIP refuses to look at.

Station C prices a club's runs allowed off FIP: strikeouts, walks, hit
batsmen and home runs, and nothing else. That is the right estimator for a
*pitcher*, because what happens to a ball once it is in play is mostly not his
to control — but it is the wrong estimator for a *club*, because the seven men
standing behind him are the same seven every night. A club that turns balls
into outs better than the league does really allows fewer runs than its
components say, and the chain has no way to know it: Atlanta allows 4.02 runs
a game against a bottom-up 4.38 (docs/playoff-odds-validation.md), and the
w = 0.5 blend with the top-down half absorbs the difference as one
unattributed lump.

This module estimates the club-level part of it, on the one rate FIP throws
away:

    BABIP allowed = (hits − home runs) / (at-bats − strikeouts − home runs
                                          + sacrifice flies)

    ΔRA/9 = (club BABIP − league BABIP) · BIP per 9 innings · runs per hit

and the delta is added to station C's bottom-up runs allowed, where the hole
is. The top-down half already contains the club's defence — measured, in its
actual runs allowed — so nothing is added there and nothing is double-counted.

## Three deliberate choices

**Road games only.** A club's BABIP allowed at home is its defence *and* its
ballpark, and the park is priced separately (`src.sim.park`); charging it
twice would be the double-count `starters.py` records having made once
already. Road games are a near-league-average mix of parks, so a road-only
BABIP is close to park-free by construction and needs no second park table on
a rate we have no prior-season sample for. It costs half the sample, which the
ballast was going to take out anyway.

**Regressed hard.** BABIP allowed is the noisiest rate in the game — a club
takes about 2,000 balls in play on the road in a season and the true spread
between clubs is a couple of hundredths. The ballast is in balls in play,
chosen walk-forward on 2025 only, and an infinite ballast reproduces the model
without the term exactly, so the sweep is a clean nesting.

**Against the league, not against the club's own pitchers.** A "FIP-independent
expectation" built from a club's own staff would need a per-pitcher BABIP
talent estimate, and the literature is clear that pitcher BABIP barely
stabilises inside a career, let alone a season; the honest expectation for any
staff's balls in play is the league's, which is what FIP itself assumes.

## Leakage

Every count is cut to games strictly *before* the date being predicted — a
club's defence tonight cannot be informed by the balls it fields tonight, and
the same cut catches the second game of a doubleheader reading the first.
`tests/test_sim/test_defence.py` pins it on synthetic logs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Balls in play of ballast on a club's BABIP allowed, regressing it toward the
# league. Chosen walk-forward on the 2025 season only, over
# {0, 1000, 2000, 4000, 8000, inf} — see docs/market-benchmark-2026.md. `inf`
# is the model with no defence term in it, which makes the sweep a nesting.
BALLAST_BIP = 4000.0

# Runs a hit on a ball in play is worth over the out it replaces. Standard
# linear weights: a non-home-run hit is worth about +0.50 runs against an
# average plate appearance and an out about −0.26, and the difference is what a
# fielder converts. Taken from the literature, not fitted here — the ballast
# above is the only free constant this term has.
RUNS_PER_BIP_HIT = 0.75

COUNT_COLS = ["team", "date", "bip", "hits_bip", "outs"]


def bip_counts(pitching_logs: pd.DataFrame, home_by_game: dict | pd.Series,
               road_only: bool = True) -> pd.DataFrame:
    """Balls in play and hits on them, per club per date, from pitching logs.

    `pitching_logs` is `mlb_stats_api.fetch_pitcher_game_logs` (already
    filtered to the regular season) and `home_by_game` is `{game_pk: home
    team id}` off the schedule — the only thing needed to tell a road game
    from a home one. With `road_only=False` every appearance counts, which is
    the control that says how much of the term is park rather than defence.

    Pitcher rows are summed to the club that pitched them that day, so a
    reliever traded in July defends for one club before the deadline and the
    other after it, exactly as `bullpen.relief_appearances` handles the pen.
    """
    if pitching_logs is None or len(pitching_logs) == 0:
        return pd.DataFrame(columns=COUNT_COLS)
    df = pitching_logs
    if "game_type" in df.columns:
        df = df[df["game_type"].isna() | (df["game_type"] == "R")]
    if df.empty:
        return pd.DataFrame(columns=COUNT_COLS)

    def col(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.zeros(len(df), dtype=float)
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(float)

    team = pd.to_numeric(df["team"], errors="coerce")
    bip = col("ab") - col("k") - col("hr") + col("sf")
    hits_bip = col("h") - col("hr")
    out = pd.DataFrame({
        "team": team,
        "date": df["date"].astype(str).to_numpy(),
        "game_pk": pd.to_numeric(df.get("game_pk"), errors="coerce").to_numpy()
        if "game_pk" in df.columns else np.nan,
        "bip": np.clip(bip, 0.0, None),
        "hits_bip": np.clip(hits_bip, 0.0, None),
        "outs": col("outs"),
    }).dropna(subset=["team"])
    if road_only:
        home = pd.Series(dict(home_by_game or {}), dtype="float64")
        mapped = out["game_pk"].map(home) if len(home) else pd.Series(np.nan, index=out.index)
        # A game whose home club we cannot identify is dropped rather than
        # guessed: an unknown park is exactly what this filter exists to avoid.
        out = out[mapped.notna() & (mapped.to_numpy() != out["team"].to_numpy())]
    agg = out.groupby(["team", "date"], as_index=False)[["bip", "hits_bip", "outs"]].sum()
    agg["team"] = agg["team"].astype("int64")
    return agg.loc[:, COUNT_COLS]


def counts_before(counts: pd.DataFrame, as_of) -> pd.DataFrame:
    """Club totals from games strictly before `as_of` — the leakage guard.

    Returns `team, bip, hits_bip, outs`, one row per club.
    """
    cols = ["team", "bip", "hits_bip", "outs"]
    if counts is None or len(counts) == 0:
        return pd.DataFrame(columns=cols)
    past = counts[counts["date"].astype(str) < str(as_of)]
    if past.empty:
        return pd.DataFrame(columns=cols)
    return past.groupby("team", as_index=False)[["bip", "hits_bip", "outs"]].sum()


def league_babip(totals: pd.DataFrame) -> tuple[float, float]:
    """`(league BABIP allowed, balls in play per 9 innings)` from club totals.

    Both are pooled over exactly the games the club estimates are built from,
    so the residuals below sum to zero across the league by construction and
    the term can only redistribute runs, never move the league's own run
    environment.
    """
    if totals is None or len(totals) == 0:
        return 0.0, 0.0
    bip = float(totals["bip"].sum())
    hits = float(totals["hits_bip"].sum())
    outs = float(totals["outs"].sum())
    if bip <= 0 or outs <= 0:
        return 0.0, 0.0
    return hits / bip, 27.0 * bip / outs


def team_babip(totals: pd.DataFrame, lg_babip: float,
               ballast: float = BALLAST_BIP) -> dict:
    """{team_id: BABIP allowed, regressed toward the league}.

    `(hits + ballast · lg) / (bip + ballast)`. An infinite ballast returns the
    league for every club, which is the no-term setting.
    """
    if totals is None or len(totals) == 0:
        return {}
    if not np.isfinite(float(ballast)):
        return {int(t): float(lg_babip) for t in totals["team"]}
    b = float(ballast)
    est = ((totals["hits_bip"].astype(float) + b * float(lg_babip))
           / (totals["bip"].astype(float) + b))
    return {int(t): float(v) for t, v in zip(totals["team"], est)}


def ra9_deltas(babip: dict, lg_babip: float, bip_per9: float,
               runs_per_hit: float = RUNS_PER_BIP_HIT) -> dict:
    """{team_id: runs per nine to add to a FIP-built runs-allowed rate}.

        (club BABIP − league BABIP) · balls in play per 9 · runs per hit

    Negative for a club that fields better than the league, which is the
    direction Atlanta's residual points.
    """
    return {int(t): float((v - float(lg_babip)) * float(bip_per9)
                          * float(runs_per_hit))
            for t, v in (babip or {}).items()}


def apply_deltas(ra9: pd.Series, deltas: dict) -> pd.Series:
    """Station C's bottom-up runs allowed with the defence residual added.

    A club absent from `deltas` — no balls in play on file yet — is left where
    the components put it, which is the same fallback every other term uses.
    """
    if ra9 is None or len(ra9) == 0 or not deltas:
        return ra9
    add = pd.Series({int(t): float(d) for t, d in deltas.items()})
    return (ra9.astype(float) + add.reindex(ra9.index).fillna(0.0)).rename(ra9.name)


def team_defence(counts: pd.DataFrame, as_of, ballast: float = BALLAST_BIP,
                 runs_per_hit: float = RUNS_PER_BIP_HIT) -> tuple[dict, dict]:
    """The whole term for one date: `({team: ΔRA/9}, diagnostics)`.

    The one function both callers use, so the backtest and the nightly cannot
    build this estimate two ways.
    """
    totals = counts_before(counts, as_of)
    lg, bip9 = league_babip(totals)
    if lg <= 0 or bip9 <= 0:
        return {}, {"lg_babip": 0.0, "bip_per9": 0.0, "n_teams": 0}
    babip = team_babip(totals, lg, ballast=ballast)
    deltas = ra9_deltas(babip, lg, bip9, runs_per_hit=runs_per_hit)
    diag = {"lg_babip": float(lg), "bip_per9": float(bip9),
            "n_teams": len(deltas),
            "babip": babip,
            "max_delta": float(max((abs(d) for d in deltas.values()), default=0.0))}
    return deltas, diag


__all__ = ["BALLAST_BIP", "RUNS_PER_BIP_HIT", "bip_counts", "counts_before",
           "league_babip", "team_babip", "ra9_deltas", "apply_deltas",
           "team_defence"]
