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


def _add(ticker, score, weight):
    return ideas.Idea(ticker, "add", score, weight)


def test_cash_goes_to_adds_in_proportion_to_score():
    snap = _snap(_h("AAA"), _h("BBB"), cash=8_000)   # equity 10,000; 500 reserve
    plan = ideas.deploy_cash([_add("AAA", 3, 0.1), _add("BBB", 2, 0.1),
                              ideas.Idea("CCC", "hold", 0, 0.1)], snap, {"max_weight": 1.0})
    assert plan.reserve == 500 and plan.investable == 7_500
    assert plan.buys == {"AAA": 4_500, "BBB": 3_000}
    assert not plan.capped


def test_capped_adds_pass_cash_to_the_others():
    snap = _snap(_h("AAA"), _h("BBB"), cash=8_000)
    # ceiling 0.25 * 0.8 * 10,000 = 2,000 per position; each already holds 1,000
    plan = ideas.deploy_cash([_add("AAA", 3, 0.1), _add("BBB", 1, 0.0)], snap,
                             {"max_weight": 0.25})
    assert plan.buys == {"BBB": 2_000, "AAA": 1_000}
    assert plan.capped and plan.leftover == pytest.approx(4_500)


def test_no_adds_means_no_buys():
    snap = _snap(_h("AAA"), cash=9_000)
    plan = ideas.deploy_cash([ideas.Idea("AAA", "hold", 0, 0.1)], snap)
    assert plan.buys == {} and not plan.capped
    assert "No add ideas today" in ideas.render([], snap, TODAY, plan)


def test_small_buys_and_reserve_only_cash_are_skipped():
    snap = _snap(_h("AAA"), _h("BBB"), cash=1_000)   # equity 3,000; reserve 150
    plan = ideas.deploy_cash([_add("AAA", 10, 0.1), _add("BBB", 0.5, 0.1)], snap,
                             {"max_weight": 1.0, "min_buy": 100})
    assert list(plan.buys) == ["AAA"]                # BBB's ~$40 share is too small
    tiny = ideas.deploy_cash([_add("AAA", 2, 0.1)], _snap(_h("AAA"), cash=40))
    assert tiny.investable == 0 and tiny.buys == {}
    assert "Nothing to deploy" in ideas.render([], _snap(_h("AAA"), cash=40), TODAY, tiny)


def test_render_and_headline_show_the_cash_plan():
    snap = _snap(_h("BIG", qty=40), _h("ADD", **UP, **TARGET_UP), cash=5_000)
    found = ideas.suggest(snap, {"max_weight": 0.25}, TODAY)
    plan = ideas.deploy_cash(found, snap, {"max_weight": 0.25})
    md = ideas.render(found, snap, TODAY, plan)
    assert "## Cash" in md and "of ADD**" in md
    line = ideas.headline(found, plan)
    assert line.startswith("Ideas: trim BIG; add ADD; buy $") and line.endswith(" ADD")
    assert ideas.headline([ideas.Idea("X", "hold", 0, 0.1)]) == ""


def test_watchlist_scores_tickers_you_dont_own():
    snap = _snap(_h("OWN", **UP, **TARGET_UP), cash=10_000)
    candidates = [_h("OWN", qty=0, **UP, **TARGET_UP), _h("LONG", qty=0, **UP, **TARGET_UP),
                  _h("SHRT", qty=0, **DOWN, **TARGET_DOWN), _h("MEH", qty=0),
                  _h("FUND", qty=0, sector="ETF", **UP, **TARGET_UP),
                  _h("NOPRICE", qty=0, price=0.0)]
    found = _by(ideas.watchlist(candidates, snap, {"start_weight": 0.03}, TODAY))
    assert set(found) == {"LONG", "SHRT", "MEH"}       # held, ETFs and unpriced are skipped
    assert found["LONG"].action == "buy" and found["LONG"].cap == 0.03
    assert found["SHRT"].action == "short"
    assert any("Robinhood doesn't let you short" in n for n in found["SHRT"].notes)
    assert found["MEH"].action == "watch"


def test_new_longs_get_a_starter_size_in_the_cash_plan():
    snap = _snap(_h("AAA"), cash=9_000)                # equity 10,000; 500 reserve
    new = ideas.Idea("NEW", "buy", 2, 0.0, cap=0.03)
    plan = ideas.deploy_cash([_add("AAA", 2, 0.1), new], snap, {"max_weight": 1.0})
    assert plan.buys["NEW"] == 300                     # 3% of 10,000
    assert plan.buys["AAA"] == 7_000                   # 80% ceiling: 8,000 less the 1,000 held
    assert plan.capped
    md = ideas.render([], snap, TODAY, plan, [new])
    assert "### New longs" in md and "start at up to 3%" in md
    assert "$300 of NEW** (new position)" in md
    assert ideas.headline([new], plan).startswith("Ideas: new NEW; buy ")


def test_render_says_how_to_fill_an_empty_watchlist():
    md = ideas.render([], _snap(cash=100), TODAY, watch=[])
    assert "## Watchlist" in md and "under `watchlist`" in md
