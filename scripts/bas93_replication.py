"""Score BAS-93's pre-registered predictions off an already-priced prop frame.

`docs/props-replication.md` pre-registers three predictions on the July 2026
Kalshi prop contracts the archive had never fetched, with **every free constant
frozen at its served value**: matchup weight 1.0, threshold 2 points, tau 0.65,
Kalshi taker rate 0.07, flat one unit, `DRAW_STREAMS=shared`. Nothing here
searches for anything — the constants arrive on the command line, and this
script only reads a priced frame and settles the predictions against it.

The priced frame is `scripts/props_exam.py --priced-out`; that script is what
actually prices the contracts, and it prints the same money tables this one
re-derives. This exists so the three predictions, the vacuity check and the
August comparison land in one machine-readable file the write-up can quote
instead of being scraped back out of a log.

    python scripts/props_exam.py --matchup on --posterior --matchup-weight 1.0 \
        --tau 0.65 --whole-window --start 2026-07-05 --end 2026-07-30 \
        --priced-out data/eval/bas93/priced_july.parquet
    python scripts/bas93_replication.py \
        --priced data/eval/bas93/priced_july.parquet \
        --august data/eval/props_priced_bas70.parquet \
        --out data/eval/bas93/props_replication.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.market import pnl, props

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bas93")

ROOT = Path(__file__).resolve().parent.parent

# Every constant below is frozen at its served value and none of them is a
# default this script is free to change — they are the pre-registration's.
THRESHOLD = 0.02          # the served 2-point edge gate
TAU = 0.65                # BAS-70's walk-forward tau, carried in as a constant
FEE_RATE = pnl.KALSHI_TAKER_RATE          # 0.07
PRIMARY = "matchup"       # marcel_partial + matchup, the served props model
CONTROL = "league"        # the league-rate control, prediction 2's comparator
BRIER_MODELS = ["p_model", "p_matchup", "p_market", "p_league"]

# Pre-registered thresholds, `docs/props-replication.md`.
MIN_HITS_CONTRACTS = 10_000
MIN_MEDIAN_CANDLES = 3
MAX_MIX_RATIO = 1.5
MIN_FEE_WAIVED_ROI = 0.04     # prediction 1, "half of BAS-70's observed"
MIN_CONTROL_GAP = 0.05        # prediction 2, "at least 5 points below"


def venues(fee_rate: float = FEE_RATE) -> tuple:
    """(as quoted, fee waived) — the same `venue_for` the exam uses."""
    as_quoted = pnl.Venue("kalshi", fee_rate, None, round_cents=True,
                          maker_rate=pnl.KALSHI_MAKER_RATE)
    fee_waived = pnl.Venue("kalshi", 0.0, None, round_cents=True,
                           maker_rate=pnl.KALSHI_MAKER_RATE)
    return as_quoted, fee_waived


def score(frame: pd.DataFrame, model: str, draws: int, seed: int,
          rule: str = "threshold", tau: float | None = None,
          fee_rate: float = FEE_RATE) -> dict:
    """One rule, one model, both venues, clustered by game.

    Flat one unit and `group_col="game_pk"` are the exam's own settings: a
    hitter's 1+, 2+ and 3+ hits are one afternoon, so the bootstrap resamples
    games rather than rows.
    """
    as_quoted, fee_waived = venues(fee_rate)
    kw = dict(staking="flat", group_col="game_pk", draws=draws, seed=seed)
    if rule == "posterior":
        kw.update(rule="posterior", tau=tau, sd_col="p_over_sd")
    else:
        kw.update(rule="threshold", threshold=THRESHOLD)
    aq = pnl.evaluate(frame, model, as_quoted, **kw)
    fw = pnl.evaluate(frame, model, fee_waived, **kw)
    return {
        "model": model, "rule": rule if rule == "threshold" else f"posterior@{tau}",
        "n_contracts": int(len(frame)), "n_bets": int(aq["n_bets"]),
        "hit_rate": _f(aq["hit_rate"]),
        "roi_as_quoted": _f(aq["roi"]),
        "roi_as_quoted_lo": _f(aq["roi_lo"]), "roi_as_quoted_hi": _f(aq["roi_hi"]),
        "roi_fee_waived": _f(fw["roi"]),
        "roi_fee_waived_lo": _f(fw["roi_lo"]), "roi_fee_waived_hi": _f(fw["roi_hi"]),
        "total_fees": _f(aq["total_fees"]), "clv": _f(aq["clv"]),
    }


def _f(x):
    return None if x is None or pd.isna(x) else float(x)


def by_stat(frame: pd.DataFrame, models: list[str], draws: int, seed: int,
            rule: str = "threshold", tau: float | None = None) -> pd.DataFrame:
    """`score` for every model, on every prop stat and pooled."""
    rows = []
    groups = [("all", frame)] + [(s, g.reset_index(drop=True))
                                 for s, g in frame.groupby("prop_stat")]
    for stat, grp in groups:
        withc = pnl.add_controls(grp, seed=seed)
        for model in models:
            if model not in withc.columns:
                continue
            rows.append({"prop_stat": stat,
                         **score(withc, model, draws, seed, rule, tau)})
    return pd.DataFrame(rows)


# ───────────────────────────── the vacuity check ─────────────────────────────

def vacuity(closes: pd.DataFrame, candles: pd.DataFrame,
            august: pd.DataFrame | None) -> dict:
    """The three counts `docs/props-replication.md` requires *before* scoring.

    Run and written down before any ROI is read, because a window too thin or
    a candle archive too hollowed makes the rest descriptive rather than a
    test.
    """
    hits = closes[closes["prop_stat"] == "hits"]
    ids = set(closes["market_id"])
    mine = candles[candles["market_id"].isin(ids)]
    per_contract = mine.groupby("market_id").size() if len(mine) else pd.Series(dtype=int)

    mix = closes.groupby("prop_stat").size()
    july_share = (mix / mix.sum()).to_dict()
    mix_rows, worst = {}, 0.0
    if august is not None and len(august):
        amix = august.groupby("prop_stat").size()
        aug_share = (amix / amix.sum()).to_dict()
        for stat in sorted(set(july_share) | set(aug_share)):
            j, a = july_share.get(stat, 0.0), aug_share.get(stat, 0.0)
            ratio = (j / a) if a else float("inf")
            mix_rows[stat] = {"july_share": round(j, 5), "august_share": round(a, 5),
                              "ratio": None if ratio == float("inf") else round(ratio, 4)}
            worst = max(worst, ratio, (1 / ratio) if ratio else float("inf"))

    checks = {
        "hits_contracts": {
            "value": int(len(hits)), "threshold": MIN_HITS_CONTRACTS,
            "pass": bool(len(hits) >= MIN_HITS_CONTRACTS),
            "note": "settled hits contracts archived in the window; the "
                    "listing pass already dropped everything below the "
                    "archive's min_volume of 1",
        },
        "hits_contracts_with_pre_pitch_volume": {
            "value": int((hits["volume_pre"] >= 1).sum()),
            "note": "the subset that also traded BEFORE first pitch, which is "
                    "the volume the P&L can actually be filled against",
        },
        "median_candles_per_contract": {
            "value": _f(per_contract.median()) if len(per_contract) else 0.0,
            "threshold": MIN_MEDIAN_CANDLES,
            "pass": bool(len(per_contract) and per_contract.median() >= MIN_MEDIAN_CANDLES),
        },
        "stat_mix_vs_august": {
            "shares": mix_rows, "worst_ratio": None if not mix_rows else round(worst, 4),
            "threshold": MAX_MIX_RATIO,
            "pass": bool(mix_rows) and worst <= MAX_MIX_RATIO,
        },
    }
    checks["retention"] = {
        "contracts_with_candles": int(mine["market_id"].nunique()) if len(mine) else 0,
        "contracts_archived": int(closes["market_id"].nunique()),
        "candles": int(len(mine)),
    }
    checks["pass"] = all(c.get("pass", True) for c in checks.values()
                         if isinstance(c, dict))
    return checks


# ───────────────────────────── the predictions ─────────────────────────────

def predictions(frame: pd.DataFrame, draws: int, seed: int) -> dict:
    """The three pre-registered calls, each with the numbers that settle it."""
    hits = pnl.add_controls(
        frame[frame["prop_stat"] == "hits"].reset_index(drop=True), seed=seed)
    ks = pnl.add_controls(
        frame[frame["prop_stat"] == "k"].reset_index(drop=True), seed=seed)

    p1 = score(hits, PRIMARY, draws, seed)
    ctrl = score(hits, CONTROL, draws, seed)
    p3 = score(ks, PRIMARY, draws, seed) if len(ks) else None

    lo = p1["roi_fee_waived_lo"]
    interval_excludes_zero = lo is not None and lo > 0
    p1_pass = bool(p1["roi_fee_waived"] is not None
                   and p1["roi_fee_waived"] >= MIN_FEE_WAIVED_ROI
                   and interval_excludes_zero
                   and p1["roi_as_quoted"] is not None and p1["roi_as_quoted"] > 0)

    gap = None
    if p1["roi_fee_waived"] is not None and ctrl["roi_fee_waived"] is not None:
        gap = p1["roi_fee_waived"] - ctrl["roi_fee_waived"]
    p2_pass = bool(ctrl["roi_as_quoted"] is not None and ctrl["roi_as_quoted"] < 0
                   and gap is not None and gap >= MIN_CONTROL_GAP)

    p3_pass = bool(p3 and p3["roi_as_quoted"] is not None and p3["roi_as_quoted"] < 0)

    return {
        "1_hits_lead_replicates": {
            "claim": "hits, threshold @ 2 pts: fee-waived ROI >= +4.0% with the "
                     "95% interval (clustered by game) excluding zero, and "
                     "as-quoted ROI > 0",
            "model": p1, "pass": p1_pass,
            "legs": {
                "fee_waived_roi_at_least_4pct":
                    bool(p1["roi_fee_waived"] is not None
                         and p1["roi_fee_waived"] >= MIN_FEE_WAIVED_ROI),
                "fee_waived_interval_excludes_zero": interval_excludes_zero,
                "as_quoted_roi_positive":
                    bool(p1["roi_as_quoted"] is not None and p1["roi_as_quoted"] > 0),
            },
        },
        "2_player_term_not_line_shape": {
            "claim": "the league-rate control on the same hits contracts under "
                     "the same rule is negative as quoted, and its fee-waived "
                     "ROI is at least 5 points below the model's",
            "control": ctrl, "gap_fee_waived": _f(gap), "pass": p2_pass,
            "legs": {
                "control_negative_as_quoted":
                    bool(ctrl["roi_as_quoted"] is not None and ctrl["roi_as_quoted"] < 0),
                "gap_at_least_5_points":
                    bool(gap is not None and gap >= MIN_CONTROL_GAP),
            },
        },
        "3_strikeouts_lose_as_quoted": {
            "claim": "strikeout contracts lose as quoted",
            "model": p3, "pass": p3_pass,
        },
    }


def august_halves(august_frame: pd.DataFrame, draws: int, seed: int,
                  tau: float) -> dict:
    """August's hits and pooled numbers on the whole window *and* on its halves.

    BAS-70's hits lead — "+9.5% / +12.1% fee-waived, intervals excluding zero"
    — is its **second half** (2026-08-17 → 09-02), which is what its posterior
    comparison table scores. BAS-93 scores July whole, so the honest August
    comparator is August whole; both are reported here, because the gap
    between them is itself part of the reading.
    """
    dates = sorted(august_frame["date"].astype(str).unique())
    cut = dates[len(dates) // 2] if dates else ""
    spans = {
        "whole": august_frame,
        "first_half": august_frame[august_frame["date"].astype(str) < cut],
        "second_half": august_frame[august_frame["date"].astype(str) >= cut],
    }
    out = {"cut": cut, "spans": {}}
    for span, sub in spans.items():
        block = {}
        for stat in ("hits", "all"):
            grp = sub if stat == "all" else sub[sub["prop_stat"] == stat]
            grp = pnl.add_controls(grp.reset_index(drop=True), seed=seed)
            block[stat] = {
                "threshold": score(grp, PRIMARY, draws, seed),
                "posterior": score(grp, PRIMARY, draws, seed,
                                   rule="posterior", tau=tau),
                "league_threshold": score(grp, CONTROL, draws, seed),
            }
        out["spans"][span] = block
    return out


def verdict(vac: dict, preds: dict) -> dict:
    """The pre-registration's failure conditions, applied to the numbers."""
    p1 = preds["1_hits_lead_replicates"]["pass"]
    p2 = preds["2_player_term_not_line_shape"]["pass"]
    p3 = preds["3_strikeouts_lose_as_quoted"]["pass"]
    if not vac.get("pass"):
        reading = ("The vacuity check fails, so every number below is "
                   "descriptive and settles nothing.")
    elif not p1:
        reading = ("Prediction 1 fails. The hits lead is recorded as noise on "
                   "the first data that could have confirmed it; the paper "
                   "ledger's hits gate keeps running with that written next "
                   "to it.")
    elif not p2:
        reading = ("Prediction 1 holds and 2 fails: the edge is the contract "
                   "shape, not the player term.")
    else:
        reading = ("Predictions 1 and 2 both hold: the first pre-registered "
                   "after-fee positive on the props board.")
    return {"vacuity_pass": bool(vac.get("pass")),
            "prediction_1": p1, "prediction_2": p2, "prediction_3": p3,
            "reading": reading,
            "ships": "Nothing in serving. The ledger (BAS-77, Stage 0.1) is "
                     "already running this rule on live tickets and it, not "
                     "this backtest, decides Stage 1."}


# ───────────────────────────── markdown ─────────────────────────────

def pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:+.1%}"


def money_markdown(table: pd.DataFrame, title: str) -> str:
    labels = {"matchup": "marcel_partial + matchup", "model": "marcel_partial",
              "league": "league_rate (control)", "market": "market (control)",
              "random_edge": "random_edge (control)"}
    out = [f"### {title}", "",
           "| stat | model | n bets | hit | ROI fee-waived | 95% CI | "
           "ROI as-quoted | 95% CI |",
           "|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples(index=False):
        hit = "—" if r.hit_rate is None or pd.isna(r.hit_rate) else f"{r.hit_rate:.3f}"
        out.append(
            f"| {r.prop_stat} | {labels.get(r.model, r.model)} | {r.n_bets} | {hit} | "
            f"{pct(r.roi_fee_waived)} | ({pct(r.roi_fee_waived_lo)}, "
            f"{pct(r.roi_fee_waived_hi)}) | {pct(r.roi_as_quoted)} | "
            f"({pct(r.roi_as_quoted_lo)}, {pct(r.roi_as_quoted_hi)}) |")
    return "\n".join(out)


def brier_markdown(table: pd.DataFrame, title: str) -> str:
    head = {"p_model": "current", "p_matchup": "matchup", "p_market": "market",
            "p_league": "league-rate"}
    models = [m for m in BRIER_MODELS if m in table.columns]
    out = [f"### {title}", "",
           "| stat | n | games | over rate | " +
           " | ".join(head[m] for m in models) + " |",
           "|---|---|---|---|" + "---|" * len(models)]
    for r in table.itertuples(index=False):
        vals = " | ".join(f"{getattr(r, m):.5f}" for m in models)
        out.append(f"| {r.prop_stat} | {r.n} | {r.games} | {r.over_rate:.3f} | {vals} |")
    return "\n".join(out)


def august_halves_markdown(halves: dict, july: dict | None = None) -> str:
    """The replication-vs-original table, with July's own row alongside."""
    out = ["### August (BAS-70) vs July (BAS-93), hits and pooled", "",
           "BAS-70's published hits lead is its **second half**; BAS-93 scores "
           "July whole, so the like-for-like August comparator is *August "
           "whole*.", "",
           "| window | stat | rule | n bets | ROI fee-waived | 95% CI | "
           "ROI as-quoted |", "|---|---|---|---|---|---|---|"]
    names = {"threshold": "threshold @ 2 pts", "posterior": "posterior @ 0.65",
             "league_threshold": "league-rate @ 2 pts"}
    labels = {"whole": "August whole", "first_half": "August first half",
              "second_half": "August second half (BAS-70's headline)"}
    for span in ("whole", "first_half", "second_half"):
        block = halves["spans"].get(span, {})
        for stat in ("hits", "all"):
            for key in ("threshold", "posterior", "league_threshold"):
                r = block.get(stat, {}).get(key)
                if not r:
                    continue
                out.append(f"| {labels[span]} | {stat} | {names[key]} | "
                           f"{r['n_bets']} | {pct(r['roi_fee_waived'])} | "
                           f"({pct(r['roi_fee_waived_lo'])}, "
                           f"{pct(r['roi_fee_waived_hi'])}) | "
                           f"{pct(r['roi_as_quoted'])} |")
    if july:
        for stat, r in july.items():
            out.append(f"| **July whole (BAS-93)** | {stat} | threshold @ 2 pts | "
                       f"{r['n_bets']} | {pct(r['roi_fee_waived'])} | "
                       f"({pct(r['roi_fee_waived_lo'])}, "
                       f"{pct(r['roi_fee_waived_hi'])}) | "
                       f"{pct(r['roi_as_quoted'])} |")
    return "\n".join(out)


def predictions_markdown(preds: dict, verdict_block: dict) -> str:
    out = ["### Pre-registered predictions", "",
           "| # | prediction | numbers | verdict |", "|---|---|---|---|"]
    p1 = preds["1_hits_lead_replicates"]
    m = p1["model"]
    out.append(f"| 1 | hits @ 2 pts: fee-waived ROI ≥ +4.0%, interval excludes "
               f"zero, as-quoted > 0 | {m['n_bets']} bets, fee-waived "
               f"{pct(m['roi_fee_waived'])} ({pct(m['roi_fee_waived_lo'])}, "
               f"{pct(m['roi_fee_waived_hi'])}), as-quoted "
               f"{pct(m['roi_as_quoted'])} | "
               f"**{'PASS' if p1['pass'] else 'FAIL'}** |")
    p2 = preds["2_player_term_not_line_shape"]
    c = p2["control"]
    out.append(f"| 2 | league-rate control negative as quoted and ≥ 5 pts below "
               f"the model fee-waived | control {c['n_bets']} bets, as-quoted "
               f"{pct(c['roi_as_quoted'])}, fee-waived "
               f"{pct(c['roi_fee_waived'])}; gap {pct(p2['gap_fee_waived'])} | "
               f"**{'PASS' if p2['pass'] else 'FAIL'}** |")
    p3 = preds["3_strikeouts_lose_as_quoted"]
    k = p3["model"] or {}
    out.append(f"| 3 | strikeouts lose as quoted | {k.get('n_bets', 0)} bets, "
               f"as-quoted {pct(k.get('roi_as_quoted'))} | "
               f"**{'PASS' if p3['pass'] else 'FAIL'}** |")
    out += ["", f"**Verdict.** {verdict_block['reading']} "
            f"{verdict_block['ships']}"]
    return "\n".join(out)


def vacuity_markdown(vac: dict) -> str:
    def verdict_of(check: dict) -> str:
        return "pass" if check.get("pass") else "**FAIL**"

    out = ["### Vacuity check (run before scoring)", "",
           "| check | value | threshold | verdict |", "|---|---|---|---|"]
    h = vac["hits_contracts"]
    out.append(f"| settled hits contracts | {h['value']} | ≥ {h['threshold']} | "
               f"{verdict_of(h)} |")
    m = vac["median_candles_per_contract"]
    out.append(f"| median candles per contract | {m['value']:.1f} | "
               f"≥ {m['threshold']} | {verdict_of(m)} |")
    s = vac["stat_mix_vs_august"]
    worst = "—" if s["worst_ratio"] is None else f"{s['worst_ratio']:.2f}×"
    out.append(f"| July stat mix vs August, worst ratio | {worst} | "
               f"≤ {MAX_MIX_RATIO}× | {verdict_of(s)} |")
    r = vac["retention"]
    out.append(f"| contracts with candles | {r['contracts_with_candles']} of "
               f"{r['contracts_archived']} | — | — |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priced", type=Path, required=True,
                    help="the July priced frame (props_exam.py --priced-out)")
    ap.add_argument("--august", type=Path, default=None,
                    help="BAS-70's priced frame, for the side-by-side")
    ap.add_argument("--candles", type=Path,
                    default=ROOT / "data/market/kalshi_prop_candles_2026.parquet")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/eval/bas93/props_replication.json")
    ap.add_argument("--markdown-out", type=Path, default=None)
    ap.add_argument("--draws", type=int, default=pnl.BOOTSTRAP_DRAWS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=TAU)
    args = ap.parse_args()

    priced = pd.read_parquet(args.priced)
    candles = pd.read_parquet(args.candles)
    august = pd.read_parquet(args.august) if args.august and args.august.exists() else None

    # ── vacuity first, and printed before a single ROI is computed ──
    vac = vacuity(priced, candles, august)
    print("\n== vacuity check (before scoring) ==")
    print(json.dumps(vac, indent=1))
    if not vac["pass"]:
        logger.warning("vacuity check FAILS: what follows is descriptive only")

    frame = props.to_pnl_frame(priced)
    models = ["matchup", "model", "league", "market", "random_edge"]

    preds = predictions(frame, args.draws, args.seed)
    money = by_stat(frame, models, args.draws, args.seed)
    post = by_stat(frame, ["matchup"], args.draws, args.seed,
                   rule="posterior", tau=args.tau)
    brier = props.brier_table(priced, models=[m for m in BRIER_MODELS
                                              if m in priced.columns])

    aug_money = aug_brier = aug_halves = None
    if august is not None:
        aug_frame = props.to_pnl_frame(august)
        aug_money = by_stat(aug_frame, models, args.draws, args.seed)
        aug_brier = props.brier_table(august, models=[m for m in BRIER_MODELS
                                                      if m in august.columns])
        aug_halves = august_halves(aug_frame, args.draws, args.seed, args.tau)

    vb = verdict(vac, preds)

    payload = {
        "issue": "BAS-93",
        "preregistration": "docs/props-replication.md",
        "frozen_constants": {
            "matchup_weight": props.MATCHUP_WEIGHT, "threshold": THRESHOLD,
            "tau": args.tau, "fee_rate": FEE_RATE, "staking": "flat 1u",
            "draw_streams": "shared", "bootstrap_draws": args.draws,
            "bootstrap_cluster": "game_pk", "seed": args.seed,
        },
        "window": {"start": str(priced["game_date"].min()),
                   "end": str(priced["game_date"].max()),
                   "contracts": int(len(priced)),
                   "settled": int(priced["over_hit"].notna().sum()),
                   "games": int(priced["game_pk"].nunique()),
                   "players": int(priced["player_id"].nunique())},
        "vacuity": vac,
        "predictions": preds,
        "verdict": vb,
        "money_july": money.to_dict("records"),
        "posterior_july": post.to_dict("records"),
        "brier_july": brier.to_dict("records"),
    }
    if aug_money is not None:
        payload["money_august"] = aug_money.to_dict("records")
        payload["brier_august"] = aug_brier.to_dict("records")
        payload["august_halves"] = aug_halves
        payload["august_window"] = {"start": str(august["game_date"].min()),
                                    "end": str(august["game_date"].max()),
                                    "contracts": int(len(august))}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    logger.info("wrote %s", args.out)

    md = [f"## BAS-93 — hits props after fee, replicated on the July contracts",
          "",
          f"July window {payload['window']['start']} … {payload['window']['end']}: "
          f"{payload['window']['settled']} settled contracts, "
          f"{payload['window']['games']} games, {payload['window']['players']} players. "
          f"Constants frozen: matchup weight {props.MATCHUP_WEIGHT}, threshold "
          f"{THRESHOLD:.0%}, tau {args.tau}, taker rate {FEE_RATE}, flat 1u, "
          f"DRAW_STREAMS=shared. Whole window, no half split.",
          "",
          vacuity_markdown(vac), "",
          predictions_markdown(preds, vb), "",
          money_markdown(money, "Money, July, flat 1u, edge ≥ 2 pts (whole window)"), "",
          money_markdown(post, f"Money, July, posterior rule at tau = {args.tau}"), "",
          brier_markdown(brier, "Brier per stat, July")]
    if aug_money is not None:
        july_rows = {}
        for stat in ("hits", "all"):
            row = money[(money["prop_stat"] == stat) & (money["model"] == PRIMARY)]
            if len(row):
                july_rows[stat] = row.iloc[0].to_dict()
        md += ["", august_halves_markdown(aug_halves, july_rows),
               "", money_markdown(aug_money,
                                  "Money, August (BAS-70's archive), whole "
                                  "window, same rule"),
               "", brier_markdown(aug_brier, "Brier per stat, August")]
    text = "\n".join(md)
    print("\n" + text)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(text + "\n")
        logger.info("wrote %s", args.markdown_out)


if __name__ == "__main__":
    main()
