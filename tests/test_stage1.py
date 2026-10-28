from datetime import date, datetime

import pytest

from src.data_pipeline import storage
from src.data_pipeline.models import AccountSummary, Holding, PortfolioSnapshot
from src.data_pipeline.sources.csv_import import CsvSource, parse_money
from src.portfolio_analysis import metrics

CSV = '''"Activity Date","Process Date","Settle Date","Instrument","Description","Trans Code","Quantity","Price","Amount"
"01/02/2025","01/02/2025","01/03/2025","","ACH Deposit","ACH","","","$5,000.00"
"01/05/2025","01/05/2025","01/06/2025","AAPL","Apple
CUSIP: 037833100","Buy","10","$100.00","($1,000.00)"
"03/01/2025","03/01/2025","03/02/2025","AAPL","Apple","Buy","10","$200.00","($2,000.00)"
"06/01/2025","06/01/2025","06/02/2025","AAPL","Apple","Sell","5","$250.00","$1,250.00"
"06/10/2025","06/10/2025","06/10/2025","AAPL","Cash Div","CDIV","","","$3.00"
"07/01/2025","07/01/2025","07/01/2025","NVDA","Nvidia","Buy","4","$50.00","($200.00)"
"08/01/2025","08/01/2025","08/01/2025","NVDA","Split","SPL","4","",""
"","","","","","","","",""
"The data provided is for informational purposes only."
'''

CFG = {
    "max_position_weight": 0.25, "max_sector_weight": 0.40, "large_loss_pct": 0.20,
    "earnings_window_days": 7, "long_term_window_days": 30, "idle_cash_pct": 0.10,
    "option_expiry_window_days": 7, "high_short_interest": 0.15,
}


def _csv_snapshot(tmp_path):
    path = tmp_path / "activity.csv"
    path.write_text(CSV, encoding="utf-8")
    return CsvSource(path).fetch_snapshot()


def test_parse_money():
    assert parse_money("$1,234.56") == 1234.56
    assert parse_money("($1,234.56)") == -1234.56
    assert parse_money("") == 0.0


def test_csv_rebuilds_average_cost_positions(tmp_path):
    snap = _csv_snapshot(tmp_path)
    by_ticker = {h.ticker: h for h in snap.holdings}
    assert by_ticker["AAPL"].quantity == 15
    assert by_ticker["AAPL"].average_cost == 150  # sells don't change average cost
    assert by_ticker["NVDA"].quantity == 8       # 2:1 split
    assert by_ticker["NVDA"].average_cost == 25
    assert snap.account.cash == 5000 - 1000 - 2000 + 1250 + 3 - 200
    assert snap.account.cash_is_estimate


def test_fifo_lots_consume_oldest_first_and_adjust_splits(tmp_path):
    lots = metrics.fifo_lots(_csv_snapshot(tmp_path).transactions)
    assert [(l.opened, l.quantity) for l in lots["AAPL"]] == [
        (date(2025, 1, 5), 5), (date(2025, 3, 1), 10)]
    assert lots["NVDA"][0].quantity == 8 and lots["NVDA"][0].price == 25


def test_risk_flags():
    snap = PortfolioSnapshot(
        source="test", taken_at=datetime.now(), account=AccountSummary(cash=100),
        holdings=[
            Holding("BIG", 10, 100, 70, sector="Tech", next_earnings=date(2026, 1, 3)),
            Holding("SML", 1, 100, 100, sector="Energy"),
        ],
    )
    flags = metrics.risk_flags(snap, CFG, today=date(2026, 1, 1))
    assert any("BIG is" in f and "of the portfolio" in f for f in flags)
    assert any("BIG is down 30%" in f for f in flags)
    assert any("BIG reports earnings" in f for f in flags)
    assert any("Tech is" in f for f in flags)


def test_long_term_flag(tmp_path):
    snap = _csv_snapshot(tmp_path)
    flags = metrics.risk_flags(snap, CFG, today=date(2026, 2, 15))
    assert any("10 AAPL shares turn long-term on Mar 02" in f for f in flags)


def test_storage_roundtrip(tmp_path):
    snap = _csv_snapshot(tmp_path)
    for h in snap.holdings:
        h.price = 200.0
    db = tmp_path / "p.db"
    sid = storage.save_snapshot(snap, db)
    storage.save_snapshot(snap, db)  # transactions dedupe across runs
    rows = storage.list_snapshots(db_path=db)
    assert len(rows) == 2 and rows[0]["total_equity"] == snap.total_equity
    with storage.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM holdings WHERE snapshot_id=?", (sid,)).fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == len(snap.transactions)


# --- Stage 1.5: richer position data -------------------------------------------------

import pandas as pd

from src.portfolio_analysis import technicals as ta


def _series(values, start="2025-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_technicals_basics():
    s = _series(range(100, 400))  # steady uptrend
    assert ta.trailing_return(s, 5) == pytest.approx(399 / 394 - 1)
    assert ta.max_drawdown(s) == 0
    assert ta.rsi(s) == 100.0
    assert ta.trend_label(399, ta.sma(s, 50), ta.sma(s, 200)) == "uptrend"
    assert ta.max_drawdown(_series([100, 120, 60, 90])) == pytest.approx(-0.5)
    assert ta.beta(s, s) == pytest.approx(1.0)


def test_realized_gains_and_dividends(tmp_path):
    txns = _csv_snapshot(tmp_path).transactions
    [g] = metrics.realized_gains(txns, 2025)
    assert g.quantity == 5 and g.cost == 500 and g.proceeds == 1250 and not g.long_term
    assert metrics.dividend_income(txns, 2025) == 3.0
    assert metrics.realized_gains(txns, 2026) == []


def test_portfolio_risk_flags_correlated_pairs():
    a = _series([100 + i + (i % 3) for i in range(120)])
    closes = pd.DataFrame({"A": a, "B": a * 2, "SPY": a})
    snap = PortfolioSnapshot("t", datetime.now(), AccountSummary(0),
                             [Holding("A", 1, 1, 100), Holding("B", 1, 1, 100)])
    risk = metrics.portfolio_risk(snap, closes, "SPY")
    assert risk.beta == pytest.approx(1.0, abs=1e-6)
    assert risk.correlated_pairs[0][:2] == ("A", "B")


def test_load_latest_snapshot_roundtrip(tmp_path):
    snap = _csv_snapshot(tmp_path)
    for h in snap.holdings:
        h.price, h.next_earnings, h.rsi_14 = 200.0, date(2026, 11, 2), 55.5
    db = tmp_path / "p.db"
    storage.save_snapshot(snap, db)
    loaded = storage.load_latest_snapshot(db)
    aapl = next(h for h in loaded.holdings if h.ticker == "AAPL")
    assert aapl.next_earnings == date(2026, 11, 2) and aapl.rsi_14 == 55.5
    assert len(loaded.transactions) == len(snap.transactions)
    assert loaded.transactions[0].date == snap.transactions[0].date
