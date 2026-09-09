"""The ledger driver's own arithmetic: the running bankroll and the amendment.

docs/bankroll.md Amendment 1 (2026-09-09) caps the per-slate exposure, resets
the paper bankroll to 1,000 units and restarts the Stage 1 window. Two of
those three are decisions this script makes on every run, so they are pinned
here. No network and no snapshot archive is read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts import paper_ledger
from src.market import paper


def led_row(**kw) -> dict:
    row = {c: None for c in paper.LEDGER_COLUMNS}
    row.update({
        "ticket_id": "t1", "snapshot_ts": "2026-09-02T18:00:00+00:00",
        "venue": "kalshi", "market_id": "m1", "game_pk": 1,
        "game_date": "2026-09-02", "game_start": "2026-09-02T23:00:00+00:00",
        "prop_stat": "hits", "side": "yes", "stake": 50.0, "cost": 0.55,
        "settled": True, "result": "no", "won": False, "profit": -50.0,
        "rule_version": paper.RULE_STAGE0,
    })
    row.update(kw)
    return row


LATER = "2026-09-10T18:00:00+00:00"


def test_a_fresh_ledger_sizes_from_the_pre_registered_bankroll():
    assert paper_ledger.running_bankroll(
        pd.DataFrame(columns=paper.LEDGER_COLUMNS), LATER) == 1000.0


def test_the_amended_rule_resumes_from_1000_however_deep_the_old_hole_is():
    """The pre-amendment ruin does not size the first amended slate.

    Twenty-five settled losses of 50 units each is a bankroll of −250 under
    the unamended rule. Amendment 1 restarts at 1,000, so the first ticket
    written under `stage0.1` is sized against 1,000 and can be written at all.
    """
    ruined = pd.DataFrame([led_row(ticket_id=f"t{i}") for i in range(25)])
    assert ruined["profit"].sum() == -1250.0
    assert paper_ledger.running_bankroll(ruined, LATER) == pytest.approx(1000.0)
    t = paper.emit(pd.DataFrame([{
        "ts": LATER, "venue": "kalshi", "market_id": "KXMLBHIT-Z",
        "game_pk": 900001, "game_date": "2026-09-10",
        "game_start": "2026-09-10T23:00:00+00:00", "player_id": 1,
        "prop_stat": "hits", "prop_line": 0.5, "bid": 0.50, "ask": 0.55,
        "p_model": 0.70, "p_matchup": 0.70, "p_over_bb": 0.70, "p_over_sd": 0.02,
        "context_source": "posted_card",
    }]), paper_ledger.running_bankroll(ruined, LATER))
    assert len(t) == 1
    assert t["bankroll_at_emit"].iloc[0] == 1000.0
    assert t["rule_version"].iloc[0] == paper.RULE_VERSION


def test_amended_profits_do_compound_and_pre_amendment_ones_do_not():
    mixed = pd.DataFrame([
        led_row(ticket_id="old", profit=-400.0),
        led_row(ticket_id="new", profit=25.0, rule_version=paper.RULE_VERSION),
    ])
    assert paper_ledger.running_bankroll(mixed, LATER) == pytest.approx(1025.0)


def test_a_ticket_whose_game_has_not_started_does_not_move_the_bankroll():
    """Sizing off a result the ledger could not yet have seen is hindsight."""
    rows = pd.DataFrame([led_row(profit=25.0, rule_version=paper.RULE_VERSION,
                                 game_start="2026-09-20T23:00:00+00:00")])
    assert paper_ledger.running_bankroll(rows, LATER) == 1000.0


def test_an_unstamped_committed_row_is_read_as_pre_amendment(tmp_path):
    """A ledger written before the amendment carries no version; it is `stage0`."""
    path = tmp_path / "ledger.parquet"
    old = pd.DataFrame([led_row()]).drop(columns=["rule_version"])
    old.to_parquet(path, index=False)
    led = paper_ledger.load_ledger(path)
    assert list(led["rule_version"]) == [paper.RULE_STAGE0]
    assert paper.gate_progress(led, draws=50)["conditions"]["tickets"]["have"] == 0


def test_the_document_reports_both_versions_and_names_the_gated_one():
    led = pd.DataFrame([
        led_row(ticket_id="a", game_pk=1),
        led_row(ticket_id="b", game_pk=2, profit=10.0, result="yes", won=True,
                rule_version=paper.RULE_VERSION),
    ])
    led["settle_ts"] = "2026-09-11T05:00:00+00:00"
    led["maker_filled"] = False
    led["posterior_side"] = "yes"
    doc = paper_ledger.to_document(led, {"2026-09-11T05:00:00+00:00": None},
                                   draws=50)
    assert doc["counts"]["by_rule_version"] == {paper.RULE_STAGE0: 1,
                                                paper.RULE_VERSION: 1}
    assert set(doc["roi_by_rule_version"]) == {paper.RULE_STAGE0, paper.RULE_VERSION}
    assert doc["roi_by_rule_version"][paper.RULE_VERSION]["scored_by_the_gate"]
    assert not doc["roi_by_rule_version"][paper.RULE_STAGE0]["scored_by_the_gate"]
    assert doc["gate"]["rule_version"] == paper.RULE_VERSION
    assert doc["rules"]["slate_cap"] == paper.SLATE_CAP
    assert doc["amendments"][0]["date"] == paper.AMENDMENT_DATE
