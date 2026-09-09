"""The Stage 0 paper ledger — docs/bankroll.md, BAS-77.

The money exam scored an archive that was assembled after the fact. This
scores the *live* one: every snapshot the market job writes is a chance to
emit the tickets the served props model would have placed at that moment,
and every later snapshot is a chance to settle the ones whose games have
finished. Two things a backtest cannot remove come out here — selection of
the archive after the fact, and hindsight on which markets settled — because
a ticket is written before the game and never rewritten afterwards.

Nothing in this module holds funds, places an order, or needs a credential.
It reads the committed snapshot archive and writes a committed ledger.

Conventions are `src/market/pnl.py`'s, one level down:

* A **ticket** is one (snapshot, market, side) decision. Its id is a hash of
  those three, so the same market seen in two snapshots is two tickets and
  the same snapshot never emits the same market twice.
* A **stake** is capital at risk in bankroll units; `pnl.settle` turns it
  into profit after the venue's taker fee.
* `p_model` is P(the over hits), which is the YES side on both venues.

The three hindsight guards, in one place because they are the whole point:

1. **A ticket may only be priced from a snapshot taken before first pitch.**
   `open_props` drops every row whose `game_start` is at or before the
   snapshot's own `ts`. A snapshot taken after the first pitch of a game
   emits nothing for that game, however open the market still looks.
2. **Rates and lineups come from before the game date.** `props.price`
   groups by `game_date` and builds each date's Marcel table from games
   strictly earlier, and the matchup arm uses the club's *probable*, never
   the man who threw the first pitch. On top of that, `card_posted` refuses
   to use a posted lineup card until the snapshot is inside
   `CARD_POSTED_HOURS` of first pitch — the lineup archive carries no
   posting timestamp, so this is the only guard available from the data
   itself, and a game outside that window falls back to the club's own
   recent cards the way the exam's fallback does.
3. **Settlement never reads the snapshot the ticket was priced from.** A
   result is only accepted from a strictly later snapshot, or from the
   committed settlement archive, and the ticket's own row is excluded by
   construction (`settle_from` compares `ts` with `>`).
"""
from __future__ import annotations

import hashlib
from datetime import timedelta

import numpy as np
import pandas as pd

from src.market import pnl, props

# docs/bankroll.md Stage 0: 1,000 units, quarter Kelly, 5% of bankroll a
# ticket, the served rule at a 2-point threshold. None of these are tunable
# from the command line on purpose — a rule that can be edited between runs
# is not a pre-registration.
START_BANKROLL = 1000.0
KELLY_FRACTION = 0.25
KELLY_CAP = 0.05
THRESHOLD = 0.02
TAU = 0.65
SERVED_MODEL = "marcel_partial + matchup"

# How close to first pitch a snapshot has to be before tonight's posted card
# is treated as information we had. Clubs post cards roughly three hours out;
# the market job's slots are 10:41, 16:37 and 23:11 UTC, so the 23:11 slot is
# inside this window for the 7pm ET slate and the 10:41 slot is not for
# anything on the card that day.
CARD_POSTED_HOURS = 3.0
# How far back a club's own recent cards are pooled when tonight's is not
# ours to see yet — the exam's own fallback horizon.
CARD_LOOKBACK_DAYS = 21

VOID = "void"

LEDGER_COLUMNS = [
    "ticket_id", "snapshot_ts", "venue", "market_id", "game_pk", "game_date",
    "game_start", "player_id", "prop_stat", "prop_line", "side", "p_model",
    "p_over_bb", "p_over_sd", "bid", "ask", "cost", "stake", "bankroll_at_emit",
    "context_source", "posterior_side", "maker_side", "maker_limit",
    "settled", "result", "won", "profit", "fee", "settle_ts", "settle_source",
    "close_reason", "maker_filled", "maker_fill_ts", "maker_profit",
]


# ───────────────────────────── tickets ─────────────────────────────

def ticket_id(snapshot_ts: str, venue: str, market_id: str, side: str) -> str:
    """A ticket's name, and the append-only key.

    A hash of exactly the three things that make a ticket what it is, so a
    re-run of the same snapshot regenerates the same id and the ledger can
    refuse to re-price it. Twelve hex characters is 48 bits; over the ~600k
    tickets a full season could produce a collision is one in ten million.
    """
    raw = f"{snapshot_ts}|{venue}|{market_id}|{side}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def open_props(snap: pd.DataFrame, stats=props.PRICEABLE) -> pd.DataFrame:
    """The rows of one snapshot a ticket could be written on. Guard 1.

    Open, priceable, with a resolved player, a quoted book, and — the guard —
    a `game_start` strictly after the snapshot's own `ts`. A market still
    listed after first pitch is a live market, and pricing it from pre-game
    rates would be pricing a game we can already see.
    """
    if snap.empty:
        return snap
    df = snap[snap["market_type"].astype(str).str.startswith("prop_")].copy()
    df = df[df["prop_stat"].isin(list(stats))]
    df = df[df["player_id"].notna() & df["game_pk"].notna()]
    df = df[df["status"].astype(str).isin(("active", "open"))]
    df = df[df["result"].isna()]
    df = df[df["bid"].notna() & df["ask"].notna()]
    df = df[df["game_start"].notna() & df["ts"].notna()]
    start = pd.to_datetime(df["game_start"], utc=True, errors="coerce")
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    keep = start.notna() & ts.notna() & (start > ts)
    return df[keep].reset_index(drop=True)


def card_posted(ts, game_start, hours: float = CARD_POSTED_HOURS) -> bool:
    """Whether tonight's lineup card is information the snapshot had. Guard 2.

    True only inside `hours` of first pitch. The lineup archive stores the
    card but not the minute it was posted, so this is a timestamp rule rather
    than a lookup; it errs the conservative way, since a card treated as
    unposted costs the price its slot-based plate appearances and falls back
    to the club's recent cards, never the other way round.
    """
    ts = pd.Timestamp(ts)
    start = pd.Timestamp(game_start)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    return bool(ts >= start - timedelta(hours=hours))


def club_card_slots(lineup_slots: dict, sides: dict, teams: dict, dates: dict,
                    lookback_days: int = CARD_LOOKBACK_DAYS) -> dict:
    """(team_id, date) -> {player_id: modal slot} from that club's earlier cards.

    The fallback for a game whose card is not ours to see yet. Strictly
    earlier dates only, pooled over `lookback_days`, and the slot kept is the
    one the club used most often — a leadoff man is a leadoff man tomorrow,
    which is all this needs to be right about, because the slot only ever
    chooses an expected plate-appearance count (4.6 down to 3.7 across the
    order, `props.SLOT_PA`).
    """
    by_team: dict = {}
    for (pk, pid), slot in lineup_slots.items():
        side = sides.get((pk, pid))
        team = teams.get((pk, side)) if side else None
        date = dates.get(pk)
        if team is None or date is None:
            continue
        by_team.setdefault(int(team), []).append((str(date), int(pid), int(slot)))
    out: dict = {}
    for team, rows in by_team.items():
        for target in sorted({d for d, _, _ in rows}):
            lo = (pd.Timestamp(target) - pd.Timedelta(days=lookback_days)).date().isoformat()
            counts: dict = {}
            for d, pid, slot in rows:
                if lo <= d < target:
                    counts.setdefault(pid, {}).setdefault(slot, 0)
                    counts[pid][slot] += 1
            if counts:
                out[(team, target)] = {pid: max(c.items(), key=lambda kv: kv[1])[0]
                                       for pid, c in counts.items()}
    return out


def emit(priced: pd.DataFrame, bankroll: float, threshold: float = THRESHOLD,
         tau: float = TAU) -> pd.DataFrame:
    """One ticket per (market, side) the served rule takes, sized on `bankroll`.

    The served rule is `pnl.decide` at the 2-point threshold on the matchup
    arm — the one docs/props-exam-2026.md scores — crossing the quoted book:
    YES at the ask, NO at ``1 - bid``. The stake is `pnl.stakes` at quarter
    Kelly capped at 5% **of the bankroll passed in**, which is the running
    paper bankroll, not a fixed notional: a ledger that sizes every ticket
    off 1,000 units is not the ledger the Stage 1 drawdown condition is about.

    `posterior_side` carries what `pnl.decide_posterior` at tau would have
    done with the same row. It is a flag, not a second stake, so recording
    the second ledger costs one extra call and nothing else.
    """
    if priced.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    p = priced["p_matchup"] if "p_matchup" in priced else priced["p_model"]
    p = p.astype(float).to_numpy()
    bid = priced["bid"].astype(float).to_numpy()
    ask = priced["ask"].astype(float).to_numpy()
    d = pnl.decide(p, bid, ask, threshold=threshold)
    post = pnl.decide_posterior(priced["p_over_bb"].astype(float).to_numpy(),
                                priced["p_over_sd"].astype(float).to_numpy(),
                                bid, ask, tau=tau)
    side = d["side"].to_numpy()
    take = side != pnl.NO_BET
    if not take.any():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    cost = d["cost"].to_numpy(dtype=float)
    stake = pnl.stakes(side, p, cost, staking="kelly", fraction=KELLY_FRACTION,
                       cap=KELLY_CAP, bankroll=float(bankroll))
    # The maker order rests where docs/bankroll.md says: the bid we would join
    # on the YES side, `1 - ask` on the NO side (joining the offer, in YES
    # terms). Both are the passive price, never the crossing one.
    maker_limit = np.where(side == "yes", bid, 1.0 - ask)

    out = pd.DataFrame({
        "snapshot_ts": priced["ts"].astype(str).to_numpy(),
        "venue": priced["venue"].astype(str).to_numpy(),
        "market_id": priced["market_id"].astype(str).to_numpy(),
        "game_pk": pd.to_numeric(priced["game_pk"], errors="coerce").to_numpy(),
        "game_date": priced["game_date"].astype(str).to_numpy(),
        "game_start": priced["game_start"].astype(str).to_numpy(),
        "player_id": pd.to_numeric(priced["player_id"], errors="coerce").to_numpy(),
        "prop_stat": priced["prop_stat"].astype(str).to_numpy(),
        "prop_line": priced["prop_line"].astype(float).to_numpy(),
        "side": side, "p_model": p,
        "p_over_bb": priced["p_over_bb"].astype(float).to_numpy(),
        "p_over_sd": priced["p_over_sd"].astype(float).to_numpy(),
        "bid": bid, "ask": ask, "cost": cost, "stake": stake,
        "bankroll_at_emit": float(bankroll),
        "context_source": priced["context_source"].astype(str).to_numpy(),
        "posterior_side": post["side"].to_numpy(),
        "maker_side": side, "maker_limit": maker_limit,
    })[take].reset_index(drop=True)
    # A capped Kelly stake of zero is not a ticket; `decide` can take a side
    # whose Kelly fraction rounds to nothing when the edge is all spread.
    out = out[out["stake"] > 0].reset_index(drop=True)
    out.insert(0, "ticket_id", [ticket_id(r.snapshot_ts, r.venue, r.market_id, r.side)
                                for r in out.itertuples(index=False)])
    for c in LEDGER_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out["settled"] = False
    out["maker_filled"] = False
    return out[LEDGER_COLUMNS]


# ───────────────────────────── settlement ─────────────────────────────

def result_index(snapshots: dict, closes: pd.DataFrame | None = None) -> pd.DataFrame:
    """market_id -> the settlement, and the earliest evidence for it.

    Two committed sources, both post-game and both keyed on `market_id`:

    * the snapshot archive's own `result` column, stamped with the `ts` of
      the snapshot that carried it — the source docs/bankroll.md assumes;
    * `data/market/prop_closes_2026.parquet`, the settlement archive the
      props exam was scored on, whose `over_hit` is the box score.

    The second exists because the first does not cover Kalshi. The snapshot
    job asks Kalshi for **open** markets only (`kalshi.fetch_all(status=
    "open")`), so a Kalshi prop leaves the archive the moment it settles and
    no later snapshot ever carries its result. Polymarket keeps settled
    markets listed and does carry them. Reported in docs, not worked around.
    """
    rows = []
    for ts, snap in snapshots.items():
        if snap is None or snap.empty or "result" not in snap:
            continue
        s = snap[snap["result"].notna() &
                 snap["market_type"].astype(str).str.startswith("prop_")]
        for r in s.itertuples(index=False):
            val = str(r.result).lower()
            won = True if val == "yes" else (False if val == "no" else None)
            rows.append({"market_id": str(r.market_id), "over_hit": won,
                         "settle_ts": str(ts), "settle_source": "snapshot"})
    if closes is not None and len(closes):
        for r in closes.itertuples(index=False):
            won = None if pd.isna(r.over_hit) else bool(r.over_hit)
            rows.append({"market_id": str(r.market_id), "over_hit": won,
                         "settle_ts": str(r.game_start),
                         "settle_source": "closes_archive"})
    if not rows:
        return pd.DataFrame(columns=["market_id", "over_hit", "settle_ts",
                                     "settle_source"])
    df = pd.DataFrame(rows).sort_values(["market_id", "settle_ts"], kind="stable")
    return df.drop_duplicates("market_id", keep="first").reset_index(drop=True)


def settle_from(ledger: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Settle every open ticket whose market has a result. Guard 3.

    The evidence must be **strictly later** than the snapshot the ticket was
    priced from — a result carried by the same snapshot would mean the market
    was already settled when we claim to have bought it, which `open_props`
    already refuses and this refuses again from the other direction.

    A market that settles to neither yes nor no — a scratch, a postponement,
    a void — is closed at zero with a reason rather than dropped, because a
    ledger that quietly loses its voids is a ledger that can be edited.
    """
    led = ledger.copy()
    if led.empty or results is None or results.empty:
        return led
    r = results.drop_duplicates("market_id").set_index("market_id")
    open_rows = [i for i in led.index
                 if not bool(led.at[i, "settled"] is True)]
    for i in open_rows:
        if bool(led.at[i, "settled"]):
            continue
        mid = str(led.at[i, "market_id"])
        if mid not in r.index:
            continue
        row = r.loc[mid]
        if str(row["settle_ts"]) <= str(led.at[i, "snapshot_ts"]):
            continue                       # not later than the price: not evidence
        led.at[i, "settled"] = True
        led.at[i, "settle_ts"] = str(row["settle_ts"])
        led.at[i, "settle_source"] = str(row["settle_source"])
        over = row["over_hit"]
        if over is None or (isinstance(over, float) and pd.isna(over)):
            led.at[i, "result"] = VOID
            led.at[i, "won"] = False
            led.at[i, "profit"] = 0.0
            led.at[i, "fee"] = 0.0
            led.at[i, "maker_profit"] = 0.0
            led.at[i, "close_reason"] = "void or scratch: stake returned, no fee"
            continue
        over = bool(over)
        venue = pnl.VENUES.get(str(led.at[i, "venue"]), pnl.KALSHI)
        profit, fee = pnl.settle([led.at[i, "side"]], [float(led.at[i, "cost"])],
                                 [float(led.at[i, "stake"])], [over], venue=venue)
        led.at[i, "result"] = "yes" if over else "no"
        led.at[i, "won"] = bool((str(led.at[i, "side"]) == "yes") == over)
        led.at[i, "profit"] = float(profit[0])
        led.at[i, "fee"] = float(fee[0])
        led.at[i, "close_reason"] = "settled from " + str(row["settle_source"])
    return led


# ───────────────────────────── the maker ledger ─────────────────────────────

def maker_fill_ts(side: str, limit: float, after_ts: str, game_start: str,
                  quotes: pd.DataFrame) -> str | None:
    """The first later pre-game snapshot that traded through a resting order.

    The exam's rule, translated from hourly candles to three snapshots a day:
    a YES bid at `limit` is filled the first time the market **prints or
    offers** at or below it (`last <= limit` or `ask <= limit`); a NO bid at
    `limit` is the same statement about ``1 - limit`` on the other side
    (`last >= 1 - limit` or `bid >= 1 - limit`), because buying NO at `limit`
    is selling YES at ``1 - limit``. Only snapshots strictly after the order
    was placed and at or before first pitch count; an order that reaches
    first pitch unfilled is cancelled and recorded as unfilled, not dropped.

    **This is pessimistic, deliberately.** The candle replay in
    docs/props-exam-2026.md could see every hour the market traded and was
    optimistic about queue position; this sees three moments a day. A fill
    that happened and reversed between two snapshots is invisible here and
    counts as no fill. So an unfilled ticket is not evidence the price never
    came — it is evidence the archive never saw it come, and the fill rate
    this produces is a floor.
    """
    if quotes is None or len(quotes) == 0 or not np.isfinite(limit):
        return None
    q = quotes[(quotes["ts"].astype(str) > str(after_ts)) &
               (quotes["ts"].astype(str) <= str(game_start))]
    if q.empty:
        return None
    last = pd.to_numeric(q["last"], errors="coerce").to_numpy(dtype=float)
    bid = pd.to_numeric(q["bid"], errors="coerce").to_numpy(dtype=float)
    ask = pd.to_numeric(q["ask"], errors="coerce").to_numpy(dtype=float)
    eps = 1e-9
    if side == "yes":
        hit = ((last <= limit + eps) & np.isfinite(last)) | \
              ((ask <= limit + eps) & np.isfinite(ask))
    else:
        thru = 1.0 - limit
        hit = ((last >= thru - eps) & np.isfinite(last)) | \
              ((bid >= thru - eps) & np.isfinite(bid))
    ts = q["ts"].astype(str).to_numpy()
    order = np.argsort(ts, kind="stable")
    for i in order:
        if bool(hit[i]):
            return str(ts[i])
    return None


def maker_profit(row) -> float:
    """Profit on one filled maker ticket, fee waived (docs/bankroll.md).

    Same arithmetic as `pnl.settle` with the resting price as the cost and no
    fee at all — not even Kalshi's 0.0175 maker rate, because Stage 0's maker
    ledger is defined as the fee-waived counterfactual the props exam quoted
    (−0.6% pooled against −5.7% as a taker) rather than a second fee schedule.
    """
    cost = float(row["maker_limit"])
    if not np.isfinite(cost) or cost <= 0:
        return 0.0
    stake = float(row["stake"])
    if str(row["result"]) == VOID or not row["settled"]:
        return 0.0
    over = str(row["result"]) == "yes"
    won = (str(row["side"]) == "yes") == over
    return stake / cost * float(won) - stake


# ───────────────────────────── the curve and the gate ─────────────────────────────

def bankroll_curve(ledger: pd.DataFrame, column: str = "profit",
                   start: float = START_BANKROLL) -> list[dict]:
    """Bankroll after each settled game-day, in settlement order.

    Ordered by settlement date, which is the order the money actually moved
    in; ordering by emission would let a ticket written on Monday and settled
    on Friday move the curve before Tuesday's did.
    """
    df = ledger[ledger["settled"].fillna(False).astype(bool)].copy()
    if df.empty:
        return [{"date": None, "profit": 0.0, "bankroll": start, "n": 0}][:0]
    day = df["settle_ts"].astype(str).str[:10]
    prof = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    g = pd.DataFrame({"date": day, "profit": prof}).groupby("date", sort=True)
    out = g["profit"].agg(["sum", "size"]).reset_index()
    out.columns = ["date", "profit", "n"]
    out["bankroll"] = start + out["profit"].cumsum()
    return out.to_dict("records")


def roi_summary(ledger: pd.DataFrame, column: str = "profit", draws: int = 2000,
                seed: int = 0) -> dict:
    """ROI after fees with the clustered bootstrap `pnl` already ships.

    Clustered on `game_pk`, for the reason docs/props-exam-2026.md gives: a
    hitter's 1+, 2+ and 3+ hits are one afternoon's at bats, and a row-wise
    bootstrap would call them three independent observations and report an
    interval far tighter than the data supports.
    """
    df = ledger[ledger["settled"].fillna(False).astype(bool)]
    df = df[df["result"].astype(str) != VOID]
    if df.empty:
        return {"n": 0, "staked": 0.0, "profit": 0.0, "roi": None,
                "roi_lo": None, "roi_hi": None, "max_drawdown": 0.0,
                "hit_rate": None}
    profit = pd.to_numeric(df[column], errors="coerce").fillna(0.0).to_numpy()
    stake = pd.to_numeric(df["stake"], errors="coerce").fillna(0.0).to_numpy()
    lo, hi = pnl.bootstrap_roi_ci(profit, stake, draws=draws, seed=seed,
                                  groups=df["game_pk"].to_numpy())
    return {"n": int(len(df)), "staked": float(stake.sum()),
            "profit": float(profit.sum()),
            "roi": float(profit.sum() / stake.sum()) if stake.sum() else None,
            "roi_lo": lo, "roi_hi": hi,
            "hit_rate": float(df["won"].astype(float).mean()),
            "max_drawdown": pnl.max_drawdown(profit)}


# docs/bankroll.md Stage 1, verbatim: 1,000 settled tickets and 21 game-days
# on the primary test, an after-fee ROI interval excluding zero, and a
# realised drawdown no worse than 1.5x what quarter Kelly implies.
GATE_TICKETS = 1000
GATE_GAME_DAYS = 21
GATE_DRAWDOWN_MULTIPLE = 1.5
PRIMARY_STAT = "hits"


def kelly_implied_drawdown(ledger: pd.DataFrame, column: str = "profit") -> float:
    """What quarter Kelly implies the deepest trough should be, in units.

    A bettor whose per-ticket profit has mean `mu > 0` and standard deviation
    `sigma` runs a random walk with drift, whose expected maximum drawdown is
    ``sigma^2 / (2·mu)`` — the classical result, and accurate enough for a
    gate that then multiplies it by 1.5. For `mu <= 0` no such bound exists:
    the walk drifts down and the deepest trough is the whole path. So a
    ledger with no measured edge gets its own realised drawdown as the
    reference, which makes the drawdown condition unmeetable rather than
    trivially met — the honest direction for a gate to fail in.
    """
    df = ledger[ledger["settled"].fillna(False).astype(bool)]
    df = df[df["result"].astype(str) != VOID]
    if df.empty:
        return 0.0
    p = pd.to_numeric(df[column], errors="coerce").fillna(0.0).to_numpy()
    mu = float(np.mean(p))
    sigma = float(np.std(p, ddof=1)) if len(p) > 1 else 0.0
    if mu <= 0 or sigma <= 0:
        return float(max(pnl.max_drawdown(p), 0.0))
    return float(sigma ** 2 / (2.0 * mu))


def gate_progress(ledger: pd.DataFrame, stat: str = PRIMARY_STAT,
                  draws: int = 2000, seed: int = 0) -> dict:
    """Stage 1 progress on the primary test, each condition scored separately.

    Every count is on the *primary* test — hits props, taker prices, settled
    — because that is what docs/bankroll.md's gate is written about. A gate
    met on the maker ledger only is not met, so nothing here reads a maker
    column.
    """
    settled = ledger["settled"].fillna(False).astype(bool)
    prim = ledger[(ledger["prop_stat"].astype(str) == stat) & settled]
    prim = prim[prim["result"].astype(str) != VOID]
    s = roi_summary(prim, draws=draws, seed=seed)
    game_days = int(prim["game_date"].astype(str).nunique()) if len(prim) else 0
    implied = kelly_implied_drawdown(prim)
    allowed = GATE_DRAWDOWN_MULTIPLE * implied
    excludes_zero = bool(s["roi_lo"] is not None and s["roi_lo"] > 0)
    dd_ok = bool(s["n"] > 0 and s["roi"] is not None and s["roi"] > 0
                 and s["max_drawdown"] <= allowed)
    conds = {
        "tickets": {"have": s["n"], "need": GATE_TICKETS,
                    "met": bool(s["n"] >= GATE_TICKETS)},
        "game_days": {"have": game_days, "need": GATE_GAME_DAYS,
                      "met": bool(game_days >= GATE_GAME_DAYS)},
        "interval_excludes_zero": {"roi": s["roi"], "lo": s["roi_lo"],
                                   "hi": s["roi_hi"], "met": excludes_zero},
        "drawdown": {"realised": s["max_drawdown"], "kelly_implied": implied,
                     "allowed": allowed, "met": dd_ok},
    }
    return {"stat": stat, "conditions": conds,
            "met": bool(all(c["met"] for c in conds.values()))}
