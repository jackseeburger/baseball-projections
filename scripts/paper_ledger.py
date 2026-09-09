"""Stage 0 of docs/bankroll.md: emit, settle and score the paper ledger.

One run does three things against the committed snapshot archive, in this
order, and never in any other:

  a. **Emit.** Price every open, priceable prop in each snapshot with the
     served props model (`marcel_partial + matchup`, the arm
     docs/props-exam-2026.md scores at `--matchup on`), apply `pnl.decide` at
     the 2-point threshold, size with `pnl.stakes` at quarter Kelly capped at
     5% of the **running** paper bankroll, and write one ticket per (market,
     side).
  b. **Settle.** Close every open ticket whose market has a later result, at
     the cost and stake recorded when the ticket was written, charging the
     venue's taker fee exactly as `pnl.settle` charges it.
  c. **Maker.** Replay the same tickets as resting orders at the snapshot
     bid (yes) / `1 - ask` (no), fee waived, filled only where a later
     pre-game snapshot shows the market traded through the resting price.

No money is involved. Nothing here holds funds or places an order, and the
script needs no credentials — it reads `data/market/snapshots/` and writes
`data/market/paper_ledger.parquet` and `public/data/market/paper_ledger.json`.

**Append-only.** A ticket id, once written, is never re-priced: the run reads
the committed ledger first, re-emits only for snapshots it has never seen,
and refuses to run at all if the committed ledger holds tickets the current
snapshot archive cannot explain. That is what makes the history unfakeable —
the ledger is the evidence, and evidence you can rewrite is not evidence.

Usage:
    python scripts/paper_ledger.py                 # every unprocessed snapshot
    python scripts/paper_ledger.py --latest        # the newest snapshot only
    python scripts/paper_ledger.py --dry-run       # emit and score, write nothing
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.mlb_stats_api import fetch_lineups, fetch_probables, fetch_schedule
from src.market import matchup as mu
from src.market import paper, pnl, props, snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("paper_ledger")

ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = ROOT / "data/market/paper_ledger.parquet"
CLOSES_PATH = ROOT / "data/market/prop_closes_2026.parquet"
JSON_PATH = ROOT / "public/data/market/paper_ledger.json"

ENGINE = paper.SERVED_MODEL
FRAMING = ("A paper ledger. No money is involved and no order is placed: these "
           "are the tickets the served props model would have written against "
           "the live Kalshi and Polymarket books, priced before first pitch "
           "from the committed snapshot archive and settled from the "
           "exchange's own results. Stage 0 of docs/bankroll.md; Stage 2 — a "
           "real bankroll — has a gate it has not met.")


def current_sha() -> str | None:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):             # pragma: no cover
        return None
    return (p.stdout.strip() or None) if p.returncode == 0 else None


# ───────────────────────── the archive ─────────────────────────

def load_snapshots(paths: list[Path]) -> dict:
    """ts -> the prop rows of that snapshot, keyed by the file's own stamp."""
    out = {}
    for p in sorted(paths):
        df = snapshot.read(p)
        df = df[df["market_type"].astype(str).str.startswith("prop_")]
        if df.empty:
            continue
        out[str(df["ts"].iloc[0])] = df.reset_index(drop=True)
    return out


def load_ledger(path: Path = LEDGER_PATH) -> pd.DataFrame:
    if Path(path).exists():
        led = pd.read_parquet(path)
        for c in paper.LEDGER_COLUMNS:
            if c not in led.columns:
                led[c] = np.nan
        return led[paper.LEDGER_COLUMNS]
    return pd.DataFrame(columns=paper.LEDGER_COLUMNS)


def check_append_only(ledger: pd.DataFrame, snapshots: dict) -> None:
    """Refuse to run if the committed ledger says something the archive does not.

    Every ticket claims a (snapshot_ts, market_id) that a committed snapshot
    carried. If one does not — the archive was pruned, a file was rewritten,
    or a ledger row was hand-edited — the ledger has stopped being a record of
    the archive and there is nothing this script can honestly append to it.
    """
    if ledger.empty:
        return
    seen = {(ts, str(m)) for ts, df in snapshots.items()
            for m in df["market_id"].astype(str).unique()}
    bad = [(str(r.ticket_id), str(r.snapshot_ts), str(r.market_id))
           for r in ledger.itertuples(index=False)
           if (str(r.snapshot_ts), str(r.market_id)) not in seen]
    if bad:
        raise SystemExit(
            f"{len(bad)} committed ticket(s) the snapshot archive cannot explain, "
            f"first: {bad[0]}. The ledger is append-only against the archive; "
            f"restore the missing snapshots rather than re-running.")


# ───────────────────────── pricing one snapshot ─────────────────────────

def build_contexts(rows: pd.DataFrame, season: int) -> dict:
    """`props.price`'s context, assembled the way `props_exam.contexts` does.

    Same four matchup lookups — the pre-game **probable** starters, which club
    is home, the posted cards, and every club's recent cards for the starts
    whose opposing card is missing — plus one thing the exam does not need:
    `club_slots`, the modal lineup slot per club per date from *earlier*
    cards, which is what a ticket priced before tonight's card is posted uses
    instead of tonight's card.
    """
    batter_ids = sorted({int(p) for p in rows.loc[rows["prop_stat"] != "k", "player_id"]})
    pitcher_ids = sorted({int(p) for p in rows.loc[rows["prop_stat"] == "k", "player_id"]})
    lo, hi = str(rows["game_date"].min()), str(rows["game_date"].max())
    back = (pd.Timestamp(lo) - pd.Timedelta(days=paper.CARD_LOOKBACK_DAYS)).date().isoformat()
    schedule = fetch_schedule(back, hi)
    schedule = schedule[schedule["game_type"] == "R"]
    game_pks = sorted(set(int(p) for p in rows["game_pk"].dropna()) |
                      set(int(p) for p in schedule["game_pk"]))
    lineups = fetch_lineups(game_pks)
    probables = fetch_probables(lo, hi)

    dates = {int(r.game_pk): str(r.date) for r in schedule.itertuples(index=False)}
    teams = {}
    for r in schedule.itertuples(index=False):
        teams[(int(r.game_pk), "home")] = int(r.home_id)
        teams[(int(r.game_pk), "away")] = int(r.away_id)
    probable_map = {}
    for r in probables.itertuples(index=False):
        if pd.notna(r.home_sp_id):
            probable_map[(int(r.game_pk), "home")] = int(r.home_sp_id)
        if pd.notna(r.away_sp_id):
            probable_map[(int(r.game_pk), "away")] = int(r.away_sp_id)
    cards = props.lineup_cards(lineups)
    club_cards = {}
    for (pk, side), ids in cards.items():
        team, date = teams.get((pk, side)), dates.get(pk)
        if team is not None and date is not None:
            club_cards[(team, date)] = ids
    slots = props.lineup_slots(lineups)
    sides = props.lineup_sides(lineups)
    return {
        "batter_ctx": props.batter_inputs(season, batter_ids),
        "pitcher_ctx": props.pitcher_inputs(season, pitcher_ids),
        "slots": slots,
        "club_slots": paper.club_card_slots(slots, sides, teams, dates),
        "teams": teams,
        "matchup_ctx": {
            "ctx": mu.inputs(season), "probables": probable_map, "teams": teams,
            "cards": cards, "club_cards": club_cards, "sides": sides,
            "weight": float(props.MATCHUP_WEIGHT),
        },
    }


def slots_for(rows: pd.DataFrame, ctx: dict) -> tuple[dict, dict]:
    """(slots, context source per market) for one snapshot's rows. Guard 2.

    A game whose first pitch is more than `CARD_POSTED_HOURS` away at the
    snapshot's own timestamp does not get tonight's card: it gets the club's
    modal slot from its own earlier cards, and the ticket records
    `club_card`. Inside the window the posted card is used and the ticket
    records `posted_card`. A player in neither is not priced at all, the way
    `props.price` already drops a batter with no slot.
    """
    slots, source = {}, {}
    for r in rows.itertuples(index=False):
        pk, pid = int(r.game_pk), int(r.player_id)
        if r.prop_stat == "k":
            source[str(r.market_id)] = "probable_starter"
            continue
        if paper.card_posted(r.ts, r.game_start):
            slot = ctx["slots"].get((pk, pid))
            if slot is not None:
                slots[(pk, pid)] = slot
                source[str(r.market_id)] = "posted_card"
                continue
        for side in ("home", "away"):
            team = ctx["teams"].get((pk, side))
            cand = ctx["club_slots"].get((team, str(r.game_date)), {}) if team else {}
            if pid in cand:
                slots[(pk, pid)] = cand[pid]
                source[str(r.market_id)] = "club_card"
                break
    return slots, source


def price_snapshot(rows: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """The served price for one snapshot's open props.

    `props.price` wants a `p_over_close` column because it was written for the
    close archive; a live snapshot's counterpart is the market's own midpoint,
    and it is only carried through for the record — the decision reads `bid`
    and `ask`, never this.
    """
    rows = rows.copy()
    rows["p_over_close"] = (rows["bid"].astype(float) + rows["ask"].astype(float)) / 2.0
    slots, source = slots_for(rows, ctx)
    priced = props.price(rows, ctx["batter_ctx"], ctx["pitcher_ctx"], slots,
                         stats=props.PRICEABLE, pitcher_bf="fixed",
                         matchup_ctx=ctx["matchup_ctx"])
    if priced.empty:
        return priced
    priced["context_source"] = [source.get(str(m), "unknown")
                                for m in priced["market_id"]]
    return priced


# ───────────────────────── the run ─────────────────────────

def run(snapshots: dict, ledger: pd.DataFrame, closes: pd.DataFrame | None,
        season: int) -> pd.DataFrame:
    """Emit for every unseen snapshot, then settle, then replay the maker book."""
    seen = set(ledger["snapshot_ts"].astype(str)) if not ledger.empty else set()
    results = paper.result_index(snapshots, closes)
    new = []
    for ts in sorted(snapshots):
        if ts in seen:
            continue
        rows = paper.open_props(snapshots[ts])
        if rows.empty:
            logger.info("%s: no open priceable props", ts)
            continue
        # The bankroll a ticket is sized against is the one the ledger showed
        # at the moment it was written: 1,000 units plus everything settled
        # from a game that had already started. Sizing off the final bankroll
        # would be hindsight on the ledger's own path.
        so_far = pd.concat([ledger] + new, ignore_index=True) if (len(ledger) or new) \
            else pd.DataFrame(columns=paper.LEDGER_COLUMNS)
        bankroll = paper.START_BANKROLL
        if len(so_far):
            done = so_far[so_far["settled"].fillna(False).astype(bool) &
                          (so_far["game_start"].astype(str) < ts)]
            bankroll += float(pd.to_numeric(done["profit"], errors="coerce").fillna(0).sum())
        ctx = build_contexts(rows, season)
        priced = price_snapshot(rows, ctx)
        if priced.empty:
            continue
        tickets = paper.emit(priced, bankroll)
        # Settle what this snapshot can already close, so the next snapshot's
        # bankroll is the one the ledger really showed.
        if len(tickets):
            new.append(paper.settle_from(tickets, results))
            logger.info("%s: %d tickets on %d priced markets, bankroll %.2f",
                        ts, len(tickets), len(priced), bankroll)
    led = pd.concat([ledger] + new, ignore_index=True) if new else ledger.copy()
    led = paper.settle_from(led, results)
    led = replay_maker(led, snapshots)
    return led.sort_values(["snapshot_ts", "ticket_id"], kind="stable").reset_index(drop=True)


def replay_maker(ledger: pd.DataFrame, snapshots: dict) -> pd.DataFrame:
    """Mark every ticket filled or unfilled, and price the fills.

    Rebuilt from scratch each run rather than appended to, because a later
    snapshot can fill an order that was unfilled when the ticket was written —
    the fill is a fact about the archive, not about the ticket. The ticket's
    own price, side and stake are never touched.
    """
    led = ledger.copy()
    if led.empty:
        return led
    frames = [df[["ts", "market_id", "bid", "ask", "last"]] for df in snapshots.values()]
    if not frames:
        return led
    quotes = pd.concat(frames, ignore_index=True)
    quotes["market_id"] = quotes["market_id"].astype(str)
    by_market = {m: g for m, g in quotes.groupby("market_id", sort=False)}
    fills, fill_ts, profits = [], [], []
    for r in led.itertuples(index=False):
        ts = paper.maker_fill_ts(str(r.maker_side), float(r.maker_limit),
                                 str(r.snapshot_ts), str(r.game_start),
                                 by_market.get(str(r.market_id)))
        fills.append(ts is not None)
        fill_ts.append(ts)
        profits.append(paper.maker_profit(r._asdict()) if ts is not None
                       and bool(r.settled) else np.nan)
    led["maker_filled"] = fills
    led["maker_fill_ts"] = fill_ts
    led["maker_profit"] = profits
    return led


# ───────────────────────── the document ─────────────────────────

def by_stat(ledger: pd.DataFrame, column: str, draws: int, seed: int) -> dict:
    out = {"all": paper.roi_summary(ledger, column, draws, seed)}
    for stat, g in ledger.groupby(ledger["prop_stat"].astype(str)):
        out[stat] = paper.roi_summary(g, column, draws, seed)
    return out


def to_document(ledger: pd.DataFrame, snapshots: dict, draws: int = 2000,
                seed: int = 0) -> dict:
    """The site's JSON: three curves, the ROI tables, and the Stage 1 gate."""
    settled = ledger["settled"].fillna(False).astype(bool)
    post = ledger[ledger["posterior_side"].astype(str) == ledger["side"].astype(str)]
    filled = ledger[ledger["maker_filled"].fillna(False).astype(bool)]
    ts_all = sorted(snapshots)
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_sha(),
        "title": "Paper ledger — Stage 0",
        "source": "scripts/paper_ledger.py",
        "engine": ENGINE,
        "served_model": ENGINE,
        "gated": False,
        "framing": FRAMING,
        "rules": {
            "start_bankroll": paper.START_BANKROLL,
            "kelly_fraction": paper.KELLY_FRACTION,
            "kelly_cap": paper.KELLY_CAP,
            "threshold": paper.THRESHOLD,
            "tau": paper.TAU,
            "fee": "Kalshi taker, round_up_to_cent(0.07·C·P·(1−P)); "
                   "Polymarket taker 0; maker ledger fee waived",
        },
        "snapshots": {"n": len(ts_all),
                      "first": ts_all[0] if ts_all else None,
                      "last": ts_all[-1] if ts_all else None},
        "counts": {
            "tickets": int(len(ledger)),
            "settled": int(settled.sum()),
            "open": int((~settled).sum()),
            "void": int((ledger["result"].astype(str) == paper.VOID).sum()),
            "game_days": int(ledger["game_date"].astype(str).nunique()),
            "posterior_tickets": int(len(post)),
            "maker_filled": int(len(filled)),
            "maker_fill_rate": float(len(filled) / len(ledger)) if len(ledger) else None,
            "by_venue": {k: int(v) for k, v in
                         ledger["venue"].astype(str).value_counts().items()},
            "by_stat": {k: int(v) for k, v in
                        ledger["prop_stat"].astype(str).value_counts().items()},
            "by_context_source": {k: int(v) for k, v in
                                  ledger["context_source"].astype(str)
                                  .value_counts().items()},
        },
        "curves": {
            "taker": paper.bankroll_curve(ledger, "profit"),
            "taker_posterior": paper.bankroll_curve(post, "profit"),
            "maker": paper.bankroll_curve(filled, "maker_profit"),
        },
        "roi": {
            "taker": by_stat(ledger, "profit", draws, seed),
            "taker_posterior": by_stat(post, "profit", draws, seed),
            "maker": by_stat(filled, "maker_profit", draws, seed),
        },
        "gate": paper.gate_progress(ledger, draws=draws, seed=seed),
    }
    doc["gate"]["note"] = (
        "docs/bankroll.md Stage 1. All four conditions on the primary test "
        "(hits props, taker prices, after fees). A gate met on the maker "
        "ledger only is not met.")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot-dir", type=Path, default=snapshot.SNAPSHOT_DIR)
    ap.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    ap.add_argument("--json", dest="json_path", type=Path, default=JSON_PATH)
    ap.add_argument("--closes", type=Path, default=CLOSES_PATH)
    ap.add_argument("--latest", action="store_true",
                    help="emit for the newest snapshot only (settlement still "
                         "reads the whole archive)")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = sorted(Path(args.snapshot_dir).glob("*.jsonl.gz"))
    if not paths:
        raise SystemExit(f"no snapshots under {args.snapshot_dir}")
    snapshots = load_snapshots(paths)
    ledger = load_ledger(args.ledger)
    check_append_only(ledger, snapshots)

    emit_from = snapshots
    if args.latest:
        newest = sorted(snapshots)[-1]
        emit_from = {ts: df for ts, df in snapshots.items()
                     if ts == newest or ts in set(ledger["snapshot_ts"].astype(str))}
    closes = pd.read_parquet(args.closes) if Path(args.closes).exists() else None
    led = run(emit_from, ledger, closes, args.season)
    # Settlement and the maker replay read every snapshot, not just the ones
    # tickets came from: a result or a fill can appear in any later file.
    led = paper.settle_from(led, paper.result_index(snapshots, closes))
    led = replay_maker(led, snapshots)

    doc = to_document(led, snapshots, draws=args.draws)
    print(json.dumps({k: doc[k] for k in ("counts", "gate")}, indent=1, default=str))
    if args.dry_run:
        return
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    led.to_parquet(args.ledger, index=False)
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(doc, indent=1, default=str))
    print(f"wrote {len(led)} tickets → {args.ledger}")
    print(f"wrote → {args.json_path}")


if __name__ == "__main__":
    main()
