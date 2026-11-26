from datetime import date, datetime

import pandas as pd
import pytest

from src.data_pipeline.models import (
    AccountSummary, Holding, OptionPosition, PortfolioSnapshot, Transaction,
)
from src.portfolio_analysis import ideas
from src.portfolio_analysis import option_ideas as oi

TODAY = date(2026, 11, 23)
EXPIRY = date(2026, 12, 24)   # 31 days out


def _chain(spot=100.0, iv=0.40, spans=False):
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    def side(kind):
        rows = []
        for k in strikes:
            d = abs(oi.bs_delta(kind, spot, k, 31, iv))
            price = max(0.05, round(d * 8, 2))          # rough, but rises with delta
            rows.append({"strike": float(k), "bid": price - 0.05, "ask": price + 0.05,
                         "lastPrice": price, "impliedVolatility": iv})
        return pd.DataFrame(rows)
    return oi.Chain(EXPIRY, spans, side("call"), side("put"), spot)


def _snap(*holdings, cash=0.0, options=(), txns=()):
    return PortfolioSnapshot("test", datetime(2026, 11, 23, 16), AccountSummary(cash),
                             list(holdings), list(txns), list(options))


def test_bs_delta():
    assert oi.bs_delta("call", 100, 100, 30, 0.3) == pytest.approx(0.53, abs=0.02)
    assert oi.bs_delta("put", 100, 100, 30, 0.3) == pytest.approx(-0.47, abs=0.02)
    assert oi.bs_delta("call", 100, 150, 30, 0.3) < 0.01
    assert oi.bs_delta("call", 100, 100, 0, 0.3) is None


def test_expiry_prefers_one_before_earnings():
    expiries = [date(2026, 11, 27), date(2026, 12, 18), date(2026, 12, 24), date(2027, 1, 15)]
    assert oi.pick_expiry(expiries, TODAY) == (date(2026, 12, 24), False)
    assert oi.pick_expiry(expiries, TODAY, avoid=date(2026, 12, 20)) == (date(2026, 12, 18), False)
    # earnings before every expiry in range: take the nearest to a month, and say so
    assert oi.pick_expiry(expiries, TODAY, avoid=date(2026, 12, 1)) == (date(2026, 12, 24), True)
    assert oi.pick_expiry([date(2026, 11, 27)], TODAY) is None


def test_strike_is_out_of_the_money_near_the_target_delta():
    chain = _chain()
    call = oi.pick_strike(chain.calls, "AAA", "call", 100.0, EXPIRY, TODAY, 0.25)
    put = oi.pick_strike(chain.puts, "AAA", "put", 100.0, EXPIRY, TODAY, 0.30)
    assert call.strike > 100 and call.delta == pytest.approx(0.25, abs=0.08)
    assert put.strike < 100 and put.delta == pytest.approx(0.30, abs=0.08)
    assert call.annualized == pytest.approx(call.premium / 100 * 365 / 31)
    assert put.basis == put.strike


def _idea(ticker, action):
    return ideas.Idea(ticker, action, 0, 0.1)


def test_covered_calls_need_100_uncovered_shares_and_skip_adds():
    short_call = OptionPosition("COV", "call", 120, EXPIRY, -1, -100.0)
    snap = _snap(Holding("BIG", 250, 90.0, 100.0), Holding("FEW", 50, 90.0, 100.0),
                 Holding("COV", 100, 90.0, 100.0), Holding("ADD", 300, 90.0, 100.0),
                 cash=1_000, options=[short_call])
    found = [_idea("BIG", "trim"), _idea("FEW", "hold"), _idea("COV", "hold"),
             _idea("ADD", "add")]
    calls = oi.covered_calls(snap, found, lambda t, e: _chain(), {}, TODAY)
    assert [(o.ticker, o.contracts) for o in calls] == [("BIG", 2)]
    assert "does the trim at a higher price" in calls[0].notes[0]


def test_covered_call_notes_earnings_taxes_and_low_iv():
    txns = [Transaction("AAA", "buy", date(2026, 6, 1), 100, 80.0, -8000.0)]
    snap = _snap(Holding("AAA", 100, 80.0, 100.0, next_earnings=date(2026, 12, 10)),
                 txns=txns)
    calls = oi.covered_calls(snap, [_idea("AAA", "hold")], lambda t, e: _chain(spans=True),
                             {}, TODAY, iv_ranks={"AAA": 0.1})
    notes = " | ".join(calls[0].notes)
    assert "earnings on Dec 10 fall before expiry" in notes
    assert "taxed as short-term" in notes
    assert "IV is low" in notes


def test_thin_premiums_are_skipped():
    snap = _snap(Holding("AAA", 100, 80.0, 100.0))
    assert oi.covered_calls(snap, [_idea("AAA", "hold")], lambda t, e: _chain(iv=0.05),
                            {"min_yield": 0.05}, TODAY) == []


def test_puts_need_the_cash_for_collateral():
    snap = _snap(Holding("OWN", 10, 90.0, 100.0), cash=20_000)    # equity 21,000
    new = ideas.Idea("NEW", "buy", 2, 0.0)
    info = {"OWN": snap.holdings[0], "NEW": Holding("NEW", 0, 0.0, 100.0)}
    puts, too_big = oi.cash_secured_puts(snap, [_idea("OWN", "add"), new, _idea("X", "hold")],
                                         info, 19_000, lambda t, e: _chain(), {}, TODAY,
                                         max_weight=0.25)
    assert {o.ticker for o in puts} == {"OWN", "NEW"} and too_big == []
    assert all(o.contracts == 1 for o in puts)
    assert all("if assigned you buy 100 shares" in o.notes[0] for o in puts)
    assert all(any("over the 25% limit" in n for n in o.notes) for o in puts)

    puts, too_big = oi.cash_secured_puts(snap, [new], info, 500, lambda t, e: _chain(), {}, TODAY)
    assert puts == [] and "more than the $500 above the reserve" in too_big[0]


def test_render():
    snap = _snap(Holding("BIG", 200, 90.0, 100.0), cash=20_000)
    calls = oi.covered_calls(snap, [_idea("BIG", "hold")], lambda t, e: _chain(), {}, TODAY)
    puts, _ = oi.cash_secured_puts(snap, [_idea("BIG", "add")], {"BIG": snap.holdings[0]},
                                   19_000, lambda t, e: _chain(), {}, TODAY)
    md = oi.render(calls, puts, ["ZZZ: a put needs $50,000 of cash"])
    assert md.startswith("## Options")
    assert "### Covered calls" in md and "sell 2 Dec 24 $" in md and " calls for about $" in md
    assert "### Cash-secured puts" in md and "do one or the other" in md
    assert "_ZZZ: a put needs $50,000 of cash_" in md
    assert "No covered calls" in oi.render([], [], [])
