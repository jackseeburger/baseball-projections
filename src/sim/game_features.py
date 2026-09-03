"""One strictly-pre-game feature row per game, cut by the chain's own guard.

Every term station E has ever added was hand-built: a log5, a pythagenpat, a
FIP delta, a fixed ballast, each worth one to four ten-thousandths of Brier.
Nothing has ever been allowed to choose its own functional form. To ask
whether a flexible model can do better with the *same information*, that
information first has to exist as a table.

This module is that table. It walks a season date by date, builds one slate per
date with `game_model.build_slate`, and reads the features off the slate —
which is the whole point: the leakage guarantee is not re-derived here, it is
inherited. `build_slate` cuts every frame it slices to games *strictly before*
the date, so a feature row for a game on 2026-07-04 cannot contain a batter's
plate appearance, a reliever's pitch count or a starter's innings from
2026-07-04 or later. There is exactly one place that cut is made, and both the
nightly and the backtest already live behind it.

What is on a row:

  * each side's **regressed** run rates (station D's top-down `pythag_60`
    rates), its **bottom-up** rates (station C's rebuild from the hitters who
    are playing and the arms who are pitching) and the **blend** the chain
    actually prices with;
  * its announced starter's FIP-based runs allowed per nine and his expected
    innings, and whether he has any history at all;
  * the availability-weighted pen behind him, as a delta from the league's own
    relievers on the same weights;
  * the posted card's runs above the club's own recent cards where a card
    exists, and a flag where it does not;
  * days of rest for the club and for the starter, how far into the season it
    is, the league's own run environment that day, the season-to-date home
    field advantage, and the game's month, weekday, venue and day/night.

There is no home indicator on a row, because every row is written from the
home club's point of view and the label is `home_win`: the indicator would be
a constant column, and the home-field edge itself is on the row as `hfa_obs`,
the season-to-date value the chain's log5 conversion uses.

The three quantities the chain itself computes from those — `chain_p`
(`pythag_C_sp_bpa_ip`, the model the nightly serves and the gate baseline),
`chain_p_lu` (the same with the posted card applied) and `pythag_60` — ride
along as **columns, not features**. A learned model handed the chain's own
answer would be a residual-learner, and the question is whether the raw
inputs are enough on their own; the blend in `scripts/train_game_learned.py`
is where the two are allowed to meet.

The walk-forward season state — each club's totals before the date, the
regressed rates those imply, the observed home-field edge — is defined here
and imported by `scripts/backtest_game_odds.py`, so the harness that scores
the chain and the table that trains its challenger cannot drift apart on what
"before this date" means.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.sim import game_model as gm
from src.sim.strength import (
    HFA_PRIOR, HFA_PRIOR_GAMES, estimate_hfa, home_win_prob, pythagenpat,
)

# The production team-strength ballast the top-down rates use (`pythag_60`).
REGRESS_GAMES = 60.0

# ─── the walk-forward season state (shared with scripts/backtest_game_odds.py) ───


def team_totals(games: pd.DataFrame, team_ids) -> pd.DataFrame:
    """Runs scored/allowed, wins/losses per team from a completed-games frame."""
    home = games.groupby("home_id").agg(rs=("home_score", "sum"), ra=("away_score", "sum"),
                                        w=("home_win", "sum"), g=("home_win", "size"))
    away = games.groupby("away_id").agg(rs=("away_score", "sum"), ra=("home_score", "sum"),
                                        w=("home_win", lambda x: (~x).sum()), g=("home_win", "size"))
    tot = home.add(away, fill_value=0).reindex(team_ids).fillna(0)
    return tot


def team_rates(tot: pd.DataFrame, regress_games: float) -> pd.DataFrame:
    """Runs scored/allowed per game, regressed toward league average."""
    lg_rs = tot["rs"].sum() / max(tot["g"].sum(), 1)
    lg_ra = tot["ra"].sum() / max(tot["g"].sum(), 1)
    return pd.DataFrame({
        "rs_pg": (tot["rs"] + regress_games * lg_rs) / (tot["g"] + regress_games),
        "ra_pg": (tot["ra"] + regress_games * lg_ra) / (tot["g"] + regress_games),
    })


def strengths(tot: pd.DataFrame, regress_games: float) -> pd.Series:
    """team_id → talent win% from those regressed rates (the `pythag_k` ladder)."""
    rates = team_rates(tot, regress_games)
    return pd.Series({t: pythagenpat(r["rs_pg"], r["ra_pg"], 1.0)
                      for t, r in rates.iterrows()})


def league_rates_from_totals(tot: pd.DataFrame) -> tuple[float, float]:
    """League runs scored / allowed per team-game — the chain's two anchors."""
    games = max(float(tot["g"].sum()), 1.0)
    return float(tot["rs"].sum()) / games, float(tot["ra"].sum()) / games


def observed_hfa(before: pd.DataFrame, prior: float = HFA_PRIOR,
                 prior_games: int = HFA_PRIOR_GAMES) -> float:
    """Season-to-date home win% shrunk toward the long-run prior."""
    return float(estimate_hfa(before, prior=prior, prior_games=prior_games))


# ─── the features themselves ───

# The model's inputs, in a fixed order. Serving and training both read this
# list, so a column can never be silently reordered between the two.
FEATURE_COLUMNS = [
    # station C's blended run environment — what the chain actually prices with
    "home_rs9", "home_ra9", "away_rs9", "away_ra9",
    # station D's top-down regressed rates (pythag_60's own numbers)
    "home_td_rs9", "home_td_ra9", "away_td_rs9", "away_td_ra9",
    # station C's bottom-up half, before the blend
    "home_bu_rs9", "home_bu_ra9", "away_bu_rs9", "away_bu_ra9",
    # the announced starters
    "home_sp_ra9", "away_sp_ra9", "home_sp_ip", "away_sp_ip",
    "home_sp_known", "away_sp_known", "home_sp_rest", "away_sp_rest",
    # the pen behind them, as a delta from the league's relievers
    "home_pen_delta", "away_pen_delta",
    # the posted card, as a delta from the club's own recent cards
    "home_card_delta", "away_card_delta", "home_has_card", "away_has_card",
    # the clubs and the date
    "home_rest", "away_rest", "home_games", "away_games",
    "lg_rs9", "lg_ra9", "hfa_obs", "month", "dow", "venue_id", "is_night",
    # differences the trees would otherwise have to discover as interactions
    "rs9_diff", "ra9_diff", "td_diff", "bu_diff", "sp_ra9_diff", "sp_ip_diff",
    "pen_diff", "card_diff", "rest_diff",
]
# Columns carried alongside the features: the label, the identifiers, and the
# hand-built chain's own answers for the same game (baselines, never inputs).
LABEL = "home_win"
META_COLUMNS = ["season", "date", "game_pk", "home_id", "away_id", LABEL,
                "chain_p", "chain_p_lu", "pythag_60"]
CATEGORICAL = ["venue_id", "month", "dow"]


@dataclass(frozen=True)
class DayContext:
    """The parts of one date that are not inside the slate.

    `top_down` is station D's regressed rates, `tot` the totals behind them,
    `hfa` the observed home-field edge and `rest`/`sp_rest` the days since each
    club's and each pitcher's last appearance. All four are built from games
    strictly before the date by `season_features`.
    """
    top_down: pd.DataFrame
    tot: pd.DataFrame
    hfa: float
    lg_rs9: float
    lg_ra9: float
    rest: dict
    sp_rest: dict


def _bottom_up(blended: float, top_down: float, weight: float) -> float:
    """Recover station C's bottom-up rate from the blend it went into.

    `blend_run_env` returns `w · bottom_up + (1 − w) · top_down`, and the slate
    keeps only the result. Inverting it here rather than rebuilding the
    bottom-up half from the raw frames is deliberate: a second construction is
    a second thing to drift, and this one is exact by algebra for any non-zero
    weight. At `w = 0` there is no bottom-up half in the blend at all and the
    feature falls back to the top-down rate.
    """
    if weight <= 0:
        return float(top_down)
    return (float(blended) - (1.0 - weight) * float(top_down)) / weight


def _side_features(slate: gm.Slate, ctx: DayContext, team_id: int,
                   starter_id, card) -> dict:
    """Every pre-game quantity the slate holds about one club in one game."""
    cfg = slate.config
    tid = int(team_id)
    blended_rs = float(slate.team.loc[tid, "rs_pg"])
    blended_ra = float(slate.team.loc[tid, "ra_pg"])
    td_rs = float(ctx.top_down.loc[tid, "rs_pg"])
    td_ra = float(ctx.top_down.loc[tid, "ra_pg"])

    out = {
        "rs9": blended_rs, "ra9": blended_ra,
        "td_rs9": td_rs, "td_ra9": td_ra,
        "bu_rs9": _bottom_up(blended_rs, td_rs, cfg.blend_weight),
        "bu_ra9": _bottom_up(blended_ra, td_ra, cfg.blend_weight),
        "games": float(ctx.tot.loc[tid, "g"]) if tid in ctx.tot.index else 0.0,
        "rest": float(ctx.rest.get(tid, np.nan)),
    }

    pid = None if starter_id is None else int(starter_id)
    out["sp_known"] = float(pid is not None and pid in slate.sp_ra9)
    out["sp_ra9"] = float(slate.sp_ra9.get(pid, slate.lg_ra9)) if pid is not None \
        else float(slate.lg_ra9)
    out["sp_ip"] = float(slate.expected_ip.get(pid, cfg.starter_ip)) if pid is not None \
        else float(cfg.starter_ip)
    out["sp_rest"] = float(ctx.sp_rest.get(pid, np.nan)) if pid is not None else np.nan

    # The pen as the chain prices it: availability-weighted, tonight's starter
    # taken out of it, measured against whatever baseline the chain measures it
    # against (the league's own relievers by default).
    if pid is not None and slate.available_pen is not None:
        pen = float(slate.available_pen(tid, pid))
        base = (slate.pen_baseline.get(tid, slate.lg_ra9)
                if isinstance(slate.pen_baseline, dict)
                else float(slate.pen_baseline))
        out["pen_delta"] = pen - base
    else:
        out["pen_delta"] = 0.0

    if card:
        raa = float(slate.lineup_raa9(card))
        out["card_delta"] = raa - float(slate.lineup_baseline.get(tid, 0.0))
        out["has_card"] = 1.0
    else:
        out["card_delta"] = 0.0
        out["has_card"] = 0.0
    return out


def game_features(slate: gm.Slate, ctx: DayContext, game, starters, cards) -> dict:
    """One feature row for one game, plus the chain's own answers for it.

    `game` is a namedtuple-ish row with `game_pk, date, home_id, away_id`
    and (optionally) `venue_id` / `day_night`; `starters` is
    `(home id, away id)` or None; `cards` is `{"home": [nine ids], ...}` or
    None.
    """
    home = _side_features(slate, ctx, int(game.home_id), None if starters is None
                          else starters[0], (cards or {}).get("home"))
    away = _side_features(slate, ctx, int(game.away_id), None if starters is None
                          else starters[1], (cards or {}).get("away"))

    date = str(game.date)
    ts = pd.Timestamp(date)
    row = {f"home_{k}": v for k, v in home.items()}
    row.update({f"away_{k}": v for k, v in away.items()})
    row.update({
        "lg_rs9": float(ctx.lg_rs9), "lg_ra9": float(ctx.lg_ra9),
        "hfa_obs": float(ctx.hfa),
        "month": float(ts.month), "dow": float(ts.dayofweek),
        "venue_id": float(getattr(game, "venue_id", np.nan) or np.nan),
        "is_night": float(str(getattr(game, "day_night", "")).lower() == "night"),
    })
    for name, key in (("rs9_diff", "rs9"), ("ra9_diff", "ra9"),
                      ("sp_ra9_diff", "sp_ra9"), ("sp_ip_diff", "sp_ip"),
                      ("pen_diff", "pen_delta"), ("card_diff", "card_delta"),
                      ("rest_diff", "rest")):
        row[name] = float(home[key]) - float(away[key])
    row["td_diff"] = (home["td_rs9"] - home["td_ra9"]) - (away["td_rs9"] - away["td_ra9"])
    row["bu_diff"] = (home["bu_rs9"] - home["bu_ra9"]) - (away["bu_rs9"] - away["bu_ra9"])

    # The hand-built chain's own answer for this game, through the one function
    # both the nightly and the backtest call. Baselines to beat, not features.
    chain_p, _ = gm.home_win_probability(slate, int(game.home_id), int(game.away_id),
                                         starters, ctx.hfa)
    chain_p_lu, _ = gm.home_win_probability(slate, int(game.home_id), int(game.away_id),
                                            starters, ctx.hfa, lineups=cards)
    row.update({
        "game_pk": int(game.game_pk), "date": date,
        "home_id": int(game.home_id), "away_id": int(game.away_id),
        "chain_p": float(chain_p), "chain_p_lu": float(chain_p_lu),
    })
    return row


# ─── the walk-forward driver ───

def rest_days(before: pd.DataFrame, date: str, team_ids) -> dict:
    """team_id → days since that club last played, from games before `date`."""
    if before.empty:
        return {}
    today = pd.Timestamp(date)
    last = pd.concat([
        before[["home_id", "date"]].rename(columns={"home_id": "team_id"}),
        before[["away_id", "date"]].rename(columns={"away_id": "team_id"}),
    ]).groupby("team_id")["date"].max()
    return {int(t): float((today - pd.Timestamp(str(d))).days)
            for t, d in last.items()}


def pitcher_rest_days(counts: pd.DataFrame, date: str) -> dict:
    """pitcher_id → days since his last appearance, from appearances before it."""
    if counts is None or len(counts) == 0:
        return {}
    past = counts[counts["date"].astype(str) < str(date)]
    if past.empty:
        return {}
    today = pd.Timestamp(str(date))
    last = past.groupby("pitcher")["date"].max()
    return {int(p): float((today - pd.Timestamp(str(d))).days)
            for p, d in last.items()}


def season_features(scored: pd.DataFrame, team_ids, inputs: gm.ChainInputs, *,
                    probables: dict, cards: dict | None = None,
                    min_games: int = 20, config: gm.ChainConfig | None = None,
                    regress_games: float = REGRESS_GAMES,
                    ballasts: tuple = (REGRESS_GAMES,),
                    progress=None) -> pd.DataFrame:
    """One row per scored game of a season, every input cut before its date.

    `scored` is the completed regular-season games (`game_pk, date, home_id,
    away_id, home_score, away_score, home_win`, plus `venue_id` / `day_night`
    when the schedule carried them); `probables` is `{game_pk: (home sp, away
    sp)}`; `cards` is `{game_pk: {"home": [nine], "away": [nine]}}`.

    The loop is the backtest's loop: skip dates until every club has
    `min_games`, rebuild the season state from games strictly before the date,
    hand it to `build_slate`, and read one row per game off the slate. The
    club's own posted-card history is banked *after* its games are scored, so
    the lineup baseline on a date only ever contains cards from games that had
    already been played.
    """
    cfg = config or gm.ChainConfig()
    cards = cards or {}
    scored = scored.sort_values("date").reset_index(drop=True)
    team_index = pd.Index([int(t) for t in team_ids], name="team_id")
    history: dict[int, list[list[int]]] = {}
    rows = []
    for date, day in scored.groupby("date", sort=True):
        before = scored[scored["date"] < date]
        if before.empty:
            continue
        tot = team_totals(before, team_index)
        if tot["g"].min() < min_games:
            continue
        top_down = team_rates(tot, regress_games)
        lg_rs9, lg_ra9 = league_rates_from_totals(tot)
        ctx = DayContext(
            top_down=top_down, tot=tot, hfa=observed_hfa(before),
            lg_rs9=lg_rs9, lg_ra9=lg_ra9,
            rest=rest_days(before, str(date), team_index),
            sp_rest=pitcher_rest_days(inputs.pitcher_counts, str(date)))
        slate = gm.build_slate(str(date), inputs, top_down, lg_rs9, lg_ra9,
                               cards=history, config=cfg)
        by_k = {k: strengths(tot, k) for k in ballasts}
        for g in day.itertuples(index=False):
            pk = int(g.game_pk)
            row = game_features(slate, ctx, g, probables.get(pk), cards.get(pk))
            row["season"] = int(inputs.season)
            row[LABEL] = bool(g.home_win)
            for k, s in by_k.items():
                row[f"pythag_{int(k)}"] = float(
                    home_win_prob(s[int(g.home_id)], s[int(g.away_id)], ctx.hfa))
            rows.append(row)
        for g in day.itertuples(index=False):
            card = cards.get(int(g.game_pk))
            if not card:
                continue
            history.setdefault(int(g.home_id), []).append(card["home"])
            history.setdefault(int(g.away_id), []).append(card["away"])
        if progress is not None:
            progress(str(date), len(rows))
    cols = META_COLUMNS + [c for c in FEATURE_COLUMNS if c not in META_COLUMNS]
    out = pd.DataFrame(rows)
    extra = [c for c in out.columns if c not in cols]
    return out.loc[:, cols + extra] if len(out) else pd.DataFrame(columns=cols)


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """`FEATURE_COLUMNS` out of a feature table, in order, as float32."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"feature table is missing {missing}")
    return df.loc[:, FEATURE_COLUMNS].astype("float32")


__all__ = ["FEATURE_COLUMNS", "META_COLUMNS", "CATEGORICAL", "LABEL",
           "REGRESS_GAMES", "DayContext", "feature_matrix", "game_features",
           "league_rates_from_totals", "observed_hfa", "pitcher_rest_days",
           "rest_days", "season_features", "strengths", "team_rates",
           "team_totals"]
