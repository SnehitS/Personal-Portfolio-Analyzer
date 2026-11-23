from datetime import date, datetime

import pytest

from src.data_pipeline.models import (
    AccountSummary, Holding, InsiderTrade, PortfolioSnapshot, Transaction,
)
from src.portfolio_analysis import ideas
from src.sentiment_pipeline.aggregate import TickerSentiment

TODAY = date(2026, 11, 23)


def _h(ticker, qty=10, cost=100.0, price=100.0, **kw):
    return Holding(ticker, qty, cost, price, **kw)


def _snap(*holdings, cash=0.0, txns=()):
    return PortfolioSnapshot("test", datetime(2026, 11, 23, 16), AccountSummary(cash),
                             list(holdings), list(txns))


def _by(found):
    return {i.ticker: i for i in found}


UP = dict(sma_50=90.0, sma_200=80.0)       # price 100 > 90 > 80
DOWN = dict(sma_50=110.0, sma_200=120.0)   # price 100 < 110 < 120
TARGET_UP = dict(analyst_target=130.0, analyst_count=10)
TARGET_DOWN = dict(analyst_target=90.0, analyst_count=10)


def test_actions_follow_the_score():
    snap = _snap(_h("ADD", **UP, **TARGET_UP), _h("HOLD"),
                 _h("TRIM", **DOWN, **TARGET_DOWN),
                 _h("SELL", rsi_14=80, **DOWN, **TARGET_DOWN), cash=10_000)
    bearish = TickerSentiment("SELL", -0.5, 5, 0, 5)
    found = _by(ideas.suggest(snap, today=TODAY, sentiment=[bearish]))
    assert {t: i.action for t, i in found.items()} == {
        "ADD": "add", "HOLD": "hold", "TRIM": "trim", "SELL": "sell"}
    assert found["SELL"].score == -3   # the most negative score possible
    assert any("bearish" in r for r in found["SELL"].reasons)


def test_strongest_actions_come_first():
    snap = _snap(_h("HOLD"), _h("ADD", **UP, **TARGET_UP), _h("TRIM", **DOWN, **TARGET_DOWN),
                 cash=10_000)
    assert [i.action for i in ideas.suggest(snap, today=TODAY)] == ["trim", "add", "hold"]


def test_overweight_is_always_trimmed_back_to_the_limit():
    snap = _snap(_h("BIG", qty=40, **UP, **TARGET_UP), _h("OTHER"), cash=5_000)  # 4000 of 10000
    idea = _by(ideas.suggest(snap, {"max_weight": 0.25}, TODAY))["BIG"]
    assert idea.action == "trim"
    assert idea.trim_to == 0.25
    assert ideas.trim_amount(idea, snap) == pytest.approx(1500)   # 40% -> 25% of $10,000


def test_no_adds_close_to_the_limit():
    snap = _snap(_h("NEAR", qty=22, **UP, **TARGET_UP), cash=7_800)   # 22%
    idea = ideas.suggest(snap, {"max_weight": 0.25}, TODAY)[0]
    assert idea.action == "hold"
    assert "already 22%" in idea.notes[0]


def test_thin_signals_are_ignored():
    snap = _snap(_h("AAA", analyst_target=200.0, analyst_count=1), cash=10_000)
    thin_news = TickerSentiment("AAA", 0.9, 1, 1, 0)
    idea = ideas.suggest(snap, today=TODAY, sentiment=[thin_news])[0]
    assert idea.signals == [] and idea.action == "hold"


def test_cost_basis_is_not_a_signal():
    loser = _h("AAA", cost=200.0)   # down 50% from cost, nothing else known
    assert ideas.suggest(_snap(loser, cash=10_000), today=TODAY)[0].action == "hold"


def test_insider_buys_count_and_sales_dont():
    buy = InsiderTrade("AAA", date(2026, 11, 2), "DOE, JANE", "CEO", "P", 1000, 50.0)
    small = InsiderTrade("AAA", date(2026, 11, 2), "ROE, RICK", "CFO", "P", 10, 50.0)
    sale = InsiderTrade("BBB", date(2026, 11, 2), "DOE, JANE", "CEO", "S", 10**6, 50.0)
    snap = _snap(_h("AAA"), _h("BBB"), cash=10_000)
    found = _by(ideas.suggest(snap, today=TODAY, insider_buys=[buy, small, sale]))
    assert found["AAA"].score == 1
    assert "an insider bought $50,000" in found["AAA"].reasons[0]
    assert found["BBB"].signals == []


def test_etfs_are_skipped():
    assert ideas.suggest(_snap(_h("SPY", sector="ETF")), today=TODAY) == []


def test_trim_notes_cover_taxes_and_earnings():
    txns = [Transaction("AAA", "buy", date(2025, 12, 1), 10, 80.0, -800.0)]
    snap = _snap(_h("AAA", cost=80.0, next_earnings=date(2026, 11, 26), **DOWN, **TARGET_DOWN),
                 cash=10_000, txns=txns)
    notes = ideas.suggest(snap, today=TODAY)[0].notes
    assert any("earnings Thu Nov 26" in n for n in notes)
    assert any("turn long-term on Dec 02" in n for n in notes)


def test_trim_of_a_loser_mentions_the_deduction_and_wash_sale():
    txns = [Transaction("AAA", "buy", date(2026, 1, 5), 10, 150.0, -1500.0)]
    snap = _snap(_h("AAA", cost=150.0, **DOWN, **TARGET_DOWN), cash=10_000, txns=txns)
    notes = ideas.suggest(snap, today=TODAY)[0].notes
    assert any("$500 loss" in n and "wash sale" in n for n in notes)


def test_tax_notes_skipped_when_history_is_incomplete():
    txns = [Transaction("AAA", "buy", date(2026, 1, 5), 4, 150.0, -600.0)]   # hold 10
    snap = _snap(_h("AAA", cost=150.0, **DOWN, **TARGET_DOWN), cash=10_000, txns=txns)
    notes = ideas.suggest(snap, today=TODAY)[0].notes
    assert notes == ["trade history doesn't cover every share, so tax notes are left out"]


def test_render_groups_by_action():
    snap = _snap(_h("BIG", qty=40), _h("ADD", **UP, **TARGET_UP), cash=5_000)
    md = ideas.render(ideas.suggest(snap, {"max_weight": 0.25}, TODAY), snap, TODAY)
    assert md.startswith("# Ideas: Mon Nov 23, 2026")
    assert md.index("## Trim") < md.index("## Add")
    assert "sell about $1,500 to get back to 25%" in md
    assert "+1 uptrend" in md
    assert "not financial advice" in md
