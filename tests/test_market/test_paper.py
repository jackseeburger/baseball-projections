"""The paper ledger's guards and arithmetic (docs/bankroll.md, BAS-77).

The two that matter most are the hindsight guards: a snapshot taken after
first pitch must emit nothing, and settlement must never read a result from
the snapshot the ticket was priced from. Everything else here — sizing
against the running bankroll, the append-only refusal, the five settlement
cases, the maker fill rule, the gate — is arithmetic that has to be pinned
because it is what the gate to real money is computed from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.market import paper, pnl
from src.market.schema import FIELDS


def snap_row(**kw) -> dict:
    row = {f: None for f in FIELDS}
    row.update({
        "ts": "2026-09-02T18:00:00+00:00", "venue": "kalshi",
        "market_id": "KXMLBHIT-A", "market_type": "prop_hits",
        "game_pk": 700001, "game_date": "2026-09-02",
        "game_start": "2026-09-02T23:00:00+00:00",
        "player_id": 12345, "prop_stat": "hits", "prop_line": 0.5,
        "bid": 0.50, "ask": 0.55, "last": 0.52, "status": "active",
        "result": None, "season": 2026,
    })
    row.update(kw)
    return row


def priced_row(**kw) -> dict:
    row = snap_row()
    row.update({"p_model": 0.70, "p_matchup": 0.70, "p_over_bb": 0.70,
                "p_over_sd": 0.02, "context_source": "posted_card"})
    row.update(kw)
    return row


def ticket(**kw) -> pd.Series:
    t = paper.emit(pd.DataFrame([priced_row()]), paper.START_BANKROLL).iloc[0].copy()
    for k, v in kw.items():
        t[k] = v
    return t


# ───────────────────────── guard 1: no ticket after first pitch ─────────────

def test_no_ticket_from_a_snapshot_taken_after_first_pitch():
    """A snapshot whose ts is past the game's start emits nothing for it.

    The market is still listed and still quoted — Kalshi runs prop books live
    — so nothing but the guard stops the model pricing a game it can already
    watch.
    """
    live = pd.DataFrame([snap_row(ts="2026-09-03T00:30:00+00:00")])
    assert live["status"].iloc[0] == "active"       # still open, still quoted
    assert paper.open_props(live).empty

    before = pd.DataFrame([snap_row(ts="2026-09-02T18:00:00+00:00")])
    assert len(paper.open_props(before)) == 1


def test_a_snapshot_exactly_at_first_pitch_is_also_refused():
    at = pd.DataFrame([snap_row(ts="2026-09-02T23:00:00+00:00")])
    assert paper.open_props(at).empty


# ───────────────────────── guard 3: settlement is never same-snapshot ───────

def test_settlement_never_reads_the_snapshot_the_ticket_was_priced_from():
    t = pd.DataFrame([ticket()])
    ts = str(t["snapshot_ts"].iloc[0])
    same = pd.DataFrame([{"market_id": "KXMLBHIT-A", "over_hit": True,
                          "settle_ts": ts, "settle_source": "snapshot"}])
    assert not bool(paper.settle_from(t, same)["settled"].iloc[0])

    later = same.assign(settle_ts="2026-09-03T01:00:00+00:00")
    out = paper.settle_from(t, later)
    assert bool(out["settled"].iloc[0])
    assert out["settle_ts"].iloc[0] == "2026-09-03T01:00:00+00:00"


def test_card_posted_window():
    """Tonight's card is not information a morning snapshot had."""
    assert not paper.card_posted("2026-09-02T10:41:00+00:00", "2026-09-02T23:00:00+00:00")
    assert paper.card_posted("2026-09-02T23:11:00+00:00", "2026-09-03T00:00:00+00:00")


# ───────────────────────── sizing against the running bankroll ──────────────

def test_stake_scales_with_the_bankroll_it_is_given():
    """Quarter Kelly on a bankroll, not on a fixed notional.

    p = 0.70 against an ask of 0.55 is a full-Kelly fraction of
    (0.70 - 0.55) / 0.45 = 0.3333; a quarter of that is 0.0833, above the 5%
    cap, so both stakes are the cap and the ratio is the bankroll ratio.
    """
    rows = pd.DataFrame([priced_row()])
    small = paper.emit(rows, 1000.0)["stake"].iloc[0]
    big = paper.emit(rows, 1500.0)["stake"].iloc[0]
    assert small == pytest.approx(1000.0 * paper.KELLY_CAP)
    assert big == pytest.approx(1.5 * small)


def test_stake_is_uncapped_kelly_when_the_edge_is_small():
    # p = 0.58 against an ask of 0.55: full Kelly (0.58-0.55)/0.45 = 0.06667,
    # quarter of it 0.016667, under the 5% cap. 0.57 would not be a bet at
    # all - `decide` compares strictly, so an edge of exactly 2 points passes.
    rows = pd.DataFrame([priced_row(p_matchup=0.58, p_model=0.58, p_over_bb=0.58)])
    t = paper.emit(rows, 1000.0)
    assert t["stake"].iloc[0] == pytest.approx(1000.0 * 0.25 * (0.03 / 0.45))


def test_no_ticket_inside_the_threshold():
    rows = pd.DataFrame([priced_row(p_matchup=0.56, p_model=0.56, p_over_bb=0.56)])
    assert paper.emit(rows, 1000.0).empty


def test_posterior_flag_rides_on_the_same_row():
    rows = pd.DataFrame([priced_row(p_over_bb=0.70, p_over_sd=0.02)])
    t = paper.emit(rows, 1000.0)
    # P(p_true > 0.55) with mean 0.70 and sd 0.02 is essentially 1, so the
    # posterior rule at tau = 0.65 takes the same side.
    assert t["posterior_side"].iloc[0] == "yes"
    wide = pd.DataFrame([priced_row(p_over_bb=0.56, p_over_sd=0.30)])
    assert paper.emit(wide, 1000.0).empty or \
        paper.emit(wide, 1000.0)["posterior_side"].iloc[0] == ""


# ───────────────────────── the append-only refusal ──────────────────────────

def test_append_only_refuses_a_ledger_the_archive_cannot_explain(tmp_path):
    from scripts import paper_ledger as pl

    snaps = {"2026-09-02T18:00:00+00:00": pd.DataFrame([snap_row()])}
    led = pd.DataFrame([ticket()])
    pl.check_append_only(led, snaps)                       # explained: fine

    orphan = led.copy()
    orphan.loc[0, "market_id"] = "KXMLBHIT-NOT-IN-ARCHIVE"
    with pytest.raises(SystemExit, match="cannot explain"):
        pl.check_append_only(orphan, snaps)


def test_ticket_id_is_stable_and_distinguishes_sides_and_snapshots():
    a = paper.ticket_id("t1", "kalshi", "M", "yes")
    assert a == paper.ticket_id("t1", "kalshi", "M", "yes")
    assert a != paper.ticket_id("t1", "kalshi", "M", "no")
    assert a != paper.ticket_id("t2", "kalshi", "M", "yes")


# ───────────────────────── settlement arithmetic ────────────────────────────

def settled_one(side, cost, stake, over):
    t = pd.DataFrame([ticket(side=side, cost=cost, stake=stake, maker_limit=0.4)])
    res = pd.DataFrame([{"market_id": "KXMLBHIT-A", "over_hit": over,
                         "settle_ts": "2026-09-09T00:00:00+00:00",
                         "settle_source": "snapshot"}])
    return paper.settle_from(t, res).iloc[0]


def test_yes_win():
    """1 unit at 50c is 2 contracts; each returns 1 - 0.50 - fee.

    The Kalshi fee at 50c is ceil_cents(0.07 * 0.25) = 0.02, so the profit is
    2 * (1 - 0.50) - 2 * 0.02 = 0.96 on a 1-unit stake.
    """
    r = settled_one("yes", 0.50, 1.0, True)
    assert bool(r["won"]) and r["result"] == "yes"
    assert r["profit"] == pytest.approx(0.96)
    assert r["fee"] == pytest.approx(0.04)


def test_yes_loss():
    r = settled_one("yes", 0.50, 1.0, False)
    assert not bool(r["won"])
    assert r["profit"] == pytest.approx(-1.04)             # stake plus the fee


def test_no_win():
    """A NO bought at 1 - bid wins when the over does not hit."""
    r = settled_one("no", 0.40, 1.0, False)
    contracts = 1 / 0.40
    fee = contracts * pnl.fee_per_contract(0.40)
    assert bool(r["won"])
    assert r["profit"] == pytest.approx(contracts - 1.0 - fee)


def test_no_loss():
    r = settled_one("no", 0.40, 1.0, True)
    contracts = 1 / 0.40
    assert r["profit"] == pytest.approx(-1.0 - contracts * pnl.fee_per_contract(0.40))


def test_void_closes_at_zero_with_a_reason():
    r = settled_one("yes", 0.50, 1.0, None)
    assert r["result"] == paper.VOID
    assert r["profit"] == 0.0 and r["fee"] == 0.0
    assert "void" in str(r["close_reason"])
    # A void is settled — it is closed — but it is not scored.
    assert bool(r["settled"])
    assert paper.roi_summary(pd.DataFrame([r]))["n"] == 0


def test_polymarket_pays_no_taker_fee():
    t = pd.DataFrame([ticket(venue="polymarket", side="yes", cost=0.50, stake=1.0)])
    res = pd.DataFrame([{"market_id": "KXMLBHIT-A", "over_hit": True,
                         "settle_ts": "2026-09-09T00:00:00+00:00",
                         "settle_source": "snapshot"}])
    r = paper.settle_from(t, res).iloc[0]
    assert r["fee"] == 0.0 and r["profit"] == pytest.approx(1.0)


# ───────────────────────── the maker fill rule ──────────────────────────────

def quotes(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=["ts", "bid", "ask", "last"])


def test_maker_yes_fills_when_a_later_snapshot_prints_through_the_bid():
    q = quotes(("2026-09-02T20:00:00+00:00", 0.50, 0.55, 0.53),
               ("2026-09-02T22:00:00+00:00", 0.44, 0.48, 0.46))
    got = paper.maker_fill_ts("yes", 0.50, "2026-09-02T18:00:00+00:00",
                              "2026-09-02T23:00:00+00:00", q)
    assert got == "2026-09-02T22:00:00+00:00"


def test_maker_yes_does_not_fill_when_the_market_only_ever_traded_above():
    q = quotes(("2026-09-02T20:00:00+00:00", 0.60, 0.65, 0.63))
    assert paper.maker_fill_ts("yes", 0.50, "2026-09-02T18:00:00+00:00",
                               "2026-09-02T23:00:00+00:00", q) is None


def test_maker_no_side_reads_the_mirrored_price():
    # Resting NO at 0.45 is selling YES at 0.55; a later bid of 0.58 is
    # through it.
    q = quotes(("2026-09-02T20:00:00+00:00", 0.58, 0.62, 0.60))
    assert paper.maker_fill_ts("no", 0.45, "2026-09-02T18:00:00+00:00",
                               "2026-09-02T23:00:00+00:00", q) is not None
    q2 = quotes(("2026-09-02T20:00:00+00:00", 0.40, 0.44, 0.42))
    assert paper.maker_fill_ts("no", 0.45, "2026-09-02T18:00:00+00:00",
                               "2026-09-02T23:00:00+00:00", q2) is None


def test_maker_order_is_cancelled_at_first_pitch():
    """A price that only came after the game started is not a fill."""
    q = quotes(("2026-09-03T01:00:00+00:00", 0.30, 0.34, 0.32))
    assert paper.maker_fill_ts("yes", 0.50, "2026-09-02T18:00:00+00:00",
                               "2026-09-02T23:00:00+00:00", q) is None


def test_maker_never_fills_from_its_own_snapshot():
    q = quotes(("2026-09-02T18:00:00+00:00", 0.20, 0.24, 0.22))
    assert paper.maker_fill_ts("yes", 0.50, "2026-09-02T18:00:00+00:00",
                               "2026-09-02T23:00:00+00:00", q) is None


def test_maker_profit_is_fee_waived_at_the_resting_price():
    r = ticket(side="yes", maker_limit=0.50, stake=1.0, settled=True, result="yes")
    assert paper.maker_profit(r.to_dict()) == pytest.approx(1.0)   # 2 contracts, no fee
    r2 = ticket(side="yes", maker_limit=0.50, stake=1.0, settled=True, result="no")
    assert paper.maker_profit(r2.to_dict()) == pytest.approx(-1.0)


# ───────────────────────── the Stage 1 gate ─────────────────────────────────

def synthetic(n: int, days: int, per_ticket: float, sd: float = 0.0,
              seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prof = per_ticket + (rng.normal(0, sd, n) if sd else np.zeros(n))
    dates = [f"2026-08-{1 + (i % days):02d}" for i in range(n)]
    return pd.DataFrame({
        "ticket_id": [f"t{i}" for i in range(n)], "snapshot_ts": dates,
        "rule_version": paper.RULE_VERSION,
        "prop_stat": "hits", "settled": True, "result": "yes",
        "won": True, "profit": prof, "stake": 1.0, "game_pk": np.arange(n),
        "game_date": dates, "settle_ts": dates,
    })


def test_gate_fails_every_condition_on_a_thin_losing_ledger():
    g = paper.gate_progress(synthetic(50, 3, -0.05, sd=0.5), draws=200)
    c = g["conditions"]
    assert not g["met"]
    assert c["tickets"] == {"have": 50, "need": 1000, "met": False}
    assert c["game_days"]["have"] == 3 and not c["game_days"]["met"]
    assert not c["interval_excludes_zero"]["met"]
    assert not c["drawdown"]["met"]


def test_gate_clears_on_a_ledger_that_meets_all_four():
    """1,200 tickets over 25 game-days at +4c a ticket with modest noise.

    +0.04 on a 1-unit stake is a 4% ROI, which is inside the +2% to +6% the
    pre-registration predicts for hits, and sd 0.15 leaves the interval well
    clear of zero at this many tickets.
    """
    g = paper.gate_progress(synthetic(1200, 25, 0.04, sd=0.15), draws=500)
    c = g["conditions"]
    assert c["tickets"]["met"] and c["game_days"]["met"]
    assert c["interval_excludes_zero"]["met"], c["interval_excludes_zero"]
    assert c["drawdown"]["met"], c["drawdown"]
    assert g["met"]


# ───────────────────── Amendment 1: the per-slate cap ───────────────────────

def slate(n: int, **kw) -> pd.DataFrame:
    """`n` identical priceable rows on distinct markets — one evening's slate."""
    rows = []
    for i in range(n):
        r = priced_row(market_id=f"KXMLBHIT-{i}", game_pk=700000 + i, **kw)
        rows.append(r)
    return pd.DataFrame(rows)


def test_slate_cap_scales_every_stake_pro_rata():
    """20 tickets at the 5% per-ticket cap is 100% of bankroll; 20% is allowed.

    Each ticket wants 50 units of a 1,000-unit bankroll, so the slate wants
    1,000 and may have 200 — a factor of 0.2 applied to every one of them,
    and the same 20 tickets, unchanged in side, price and count.
    """
    rows = slate(20)
    t = paper.emit(rows, 1000.0)
    assert len(t) == 20
    assert t["stake"].sum() == pytest.approx(1000.0 * paper.SLATE_CAP)
    assert t["stake"].nunique() == 1
    assert t["stake"].iloc[0] == pytest.approx(0.2 * 1000.0 * paper.KELLY_CAP)
    assert set(t["side"]) == {"yes"}


def test_slate_cap_leaves_a_slate_under_the_cap_untouched():
    """Three tickets at 5% is 15% of bankroll — inside the cap, so nothing moves."""
    t = paper.emit(slate(3), 1000.0)
    assert t["stake"].sum() == pytest.approx(3 * 1000.0 * paper.KELLY_CAP)
    assert t["stake"].iloc[0] == pytest.approx(1000.0 * paper.KELLY_CAP)


def test_slate_cap_preserves_relative_size():
    """The scaling is one common factor, so the ratio between two tickets holds."""
    small = priced_row(market_id="small", game_pk=1, p_matchup=0.58, p_model=0.58,
                       p_over_bb=0.58)
    big = priced_row(market_id="big", game_pk=2)
    rows = pd.DataFrame([small, big] + [priced_row(market_id=f"f{i}", game_pk=10 + i)
                                        for i in range(20)])
    uncapped = paper.emit(rows, 1000.0, slate_cap=1e9)
    capped = paper.emit(rows, 1000.0)
    ratio_u = (uncapped.set_index("market_id")["stake"]["small"] /
               uncapped.set_index("market_id")["stake"]["big"])
    ratio_c = (capped.set_index("market_id")["stake"]["small"] /
               capped.set_index("market_id")["stake"]["big"])
    assert ratio_c == pytest.approx(ratio_u)
    assert capped["stake"].sum() == pytest.approx(200.0)


def test_emitted_tickets_are_stamped_with_the_amended_rule_version():
    t = paper.emit(pd.DataFrame([priced_row()]), 1000.0)
    assert set(t["rule_version"]) == {paper.RULE_VERSION} == {"stage0.1"}
    assert "rule_version" in paper.LEDGER_COLUMNS


def test_the_gate_ignores_pre_amendment_tickets():
    """A ledger that would clear the gate under `stage0` clears nothing.

    docs/bankroll.md Amendment 1 restarts the Stage 1 window: the tickets
    written under the unamended sizing stay in the ledger and count toward
    nothing.
    """
    led = synthetic(1200, 25, 0.04, sd=0.15)
    led["rule_version"] = paper.RULE_STAGE0
    g = paper.gate_progress(led, draws=200)
    assert g["rule_version"] == paper.RULE_VERSION
    assert g["conditions"]["tickets"]["have"] == 0
    assert not g["met"]

    mixed = pd.concat([led, synthetic(300, 7, 0.04, sd=0.15, seed=2)],
                      ignore_index=True)
    assert paper.gate_progress(mixed, draws=200)["conditions"]["tickets"]["have"] == 300


def test_the_curve_restarts_at_the_amendment():
    """Two rule versions are two segments, each starting from 1,000 units."""
    pre = synthetic(4, 2, -300.0)
    pre["rule_version"] = paper.RULE_STAGE0
    post = synthetic(4, 2, 1.0, seed=3)
    post["settle_ts"] = post["game_date"] = [f"2026-09-{9 + (i % 2):02d}"
                                             for i in range(4)]
    curve = paper.bankroll_curve(pd.concat([pre, post], ignore_index=True))
    first_post = next(p for p in curve if p["rule_version"] == paper.RULE_VERSION)
    assert first_post["bankroll"] == pytest.approx(paper.START_BANKROLL + 2.0)
    assert min(p["bankroll"] for p in curve
               if p["rule_version"] == paper.RULE_STAGE0) < 0


def test_gate_counts_only_the_primary_stat():
    led = synthetic(1200, 25, 0.04, sd=0.15)
    led["prop_stat"] = "tb"
    assert paper.gate_progress(led, draws=200)["conditions"]["tickets"]["have"] == 0


def test_kelly_implied_drawdown_falls_back_when_there_is_no_edge():
    """With no positive drift there is nothing to imply, so the path is it."""
    losing = synthetic(200, 5, -0.02, sd=0.3)
    implied = paper.kelly_implied_drawdown(losing)
    assert implied == pytest.approx(pnl.max_drawdown(losing["profit"].to_numpy()))


def test_bankroll_curve_starts_at_the_paper_bankroll_and_compounds_by_day():
    led = synthetic(6, 3, 1.0)
    curve = paper.bankroll_curve(led)
    assert len(curve) == 3
    assert curve[0]["bankroll"] == pytest.approx(paper.START_BANKROLL + 2.0)
    assert curve[-1]["bankroll"] == pytest.approx(paper.START_BANKROLL + 6.0)


# ─────────── settlement sources: the snapshot first (BAS-78) ───────────

def closes_row(market_id="KXMLBHIT-A", over_hit=True):
    return pd.DataFrame([{"market_id": market_id, "over_hit": over_hit,
                          "game_start": "2026-09-02T23:00:00+00:00"}])


def settled_snapshot(result="yes", ts="2026-09-03T05:00:00+00:00"):
    return {ts: pd.DataFrame([snap_row(ts=ts, status="settled", result=result)])}


def test_the_snapshot_is_preferred_over_the_closes_archive():
    """Both sources have it; the exchange's own settlement is the one used."""
    idx = paper.result_index(settled_snapshot("yes"), closes_row(over_hit=True))
    assert len(idx) == 1
    assert idx["settle_source"].iloc[0] == "snapshot"
    assert bool(idx["over_hit"].iloc[0])


def test_the_closes_archive_is_still_the_fallback():
    idx = paper.result_index({}, closes_row(over_hit=False))
    assert idx["settle_source"].iloc[0] == "closes_archive"
    assert not bool(idx["over_hit"].iloc[0])


def test_a_disagreement_is_logged_and_the_snapshot_wins(caplog):
    """The venue says no, the box score says yes. That is worth a warning."""
    with caplog.at_level("WARNING"):
        idx = paper.result_index(settled_snapshot("no"), closes_row(over_hit=True))
    assert idx["settle_source"].iloc[0] == "snapshot"
    assert not bool(idx["over_hit"].iloc[0])
    assert "KXMLBHIT-A" in caplog.text
    assert "exchange's own settlement" in caplog.text


def test_agreement_is_not_logged(caplog):
    with caplog.at_level("WARNING"):
        paper.result_index(settled_snapshot("yes"), closes_row(over_hit=True))
    assert "KXMLBHIT-A" not in caplog.text


def test_a_kalshi_ticket_settles_from_the_snapshot_alone():
    """The point of the settled pull: no backfill file needed.

    Before BAS-78 this returned nothing, because the open pull dropped a
    Kalshi prop from the archive the moment it settled.
    """
    t = pd.DataFrame([ticket()])
    out = paper.settle_from(t, paper.result_index(settled_snapshot("yes"), None))
    assert bool(out["settled"].iloc[0])
    assert out["settle_source"].iloc[0] == "snapshot"
    assert out["result"].iloc[0] == "yes"
