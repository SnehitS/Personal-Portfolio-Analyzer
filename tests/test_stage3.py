from datetime import date

import pytest

from src.update_engine.report import report_path, write_report


def test_report_paths(tmp_path):
    day = date(2026, 11, 7)
    assert report_path("daily", day, tmp_path) == tmp_path / "daily" / "2026-11-07.md"
    assert report_path("weekly", day, tmp_path) == tmp_path / "weekly" / "2026-W45.md"
    with pytest.raises(ValueError):
        report_path("monthly", day, tmp_path)


def test_write_report_overwrites_same_day(tmp_path):
    day = date(2026, 11, 7)
    write_report("daily", "# first", day, tmp_path)
    path = write_report("daily", "# second\n\n", day, tmp_path)
    assert path.read_text(encoding="utf-8") == "# second\n"


def _snap(taken_at, source="robinhood", price=100.0):
    from src.data_pipeline.models import AccountSummary, Holding, PortfolioSnapshot
    return PortfolioSnapshot(source, taken_at, AccountSummary(cash=1000.0),
                             [Holding("AAA", 10, 90.0, price)])


def test_load_snapshot_before(tmp_path):
    from datetime import datetime

    from src.data_pipeline import storage

    db = tmp_path / "p.db"
    storage.save_snapshot(_snap(datetime(2026, 11, 6, 16), price=100), db)
    storage.save_snapshot(_snap(datetime(2026, 11, 6, 17), source="csv", price=101), db)
    last_id = storage.save_snapshot(_snap(datetime(2026, 11, 7, 16), price=105), db)

    assert storage.load_latest_snapshot(db).holdings[0].price == 105
    assert storage.load_snapshot(last_id, db).taken_at == datetime(2026, 11, 7, 16)
    before = datetime(2026, 11, 7)
    assert storage.load_snapshot_before(before, db_path=db).holdings[0].price == 101
    assert storage.load_snapshot_before(before, "robinhood", db).holdings[0].price == 100
    assert storage.load_snapshot_before(datetime(2026, 11, 1), db_path=db) is None


def test_compare_snapshots():
    from datetime import datetime

    from src.data_pipeline.models import AccountSummary, Holding, PortfolioSnapshot, Transaction
    from src.update_engine.changes import compare

    before = PortfolioSnapshot("robinhood", datetime(2026, 11, 6, 16), AccountSummary(500.0), [
        Holding("AAA", 10, 90.0, 100.0), Holding("BBB", 5, 40.0, 50.0),
        Holding("CCC", 2, 10.0, 10.0)])
    deposit = Transaction("", "deposit", datetime(2026, 11, 7).date(), 0, 0, 200.0, "d1")
    old_deposit = Transaction("", "deposit", datetime(2026, 11, 1).date(), 0, 0, 999.0, "d0")
    after = PortfolioSnapshot("robinhood", datetime(2026, 11, 7, 16), AccountSummary(100.0), [
        Holding("AAA", 12, 91.0, 110.0), Holding("BBB", 5, 40.0, 45.0),
        Holding("DDD", 4, 25.0, 25.0)], transactions=[old_deposit, deposit])

    change = compare(before, after)
    status = {p.ticker: p.status for p in change.positions}
    assert status == {"AAA": "added", "BBB": "held", "CCC": "closed", "DDD": "new"}

    aaa = next(p for p in change.positions if p.ticker == "AAA")
    assert aaa.price_change_pct == pytest.approx(0.10)
    assert aaa.market_gain == pytest.approx(100.0)          # 10 shares held through x $10
    assert [p.ticker for p in change.movers()] == ["AAA", "BBB"]

    assert change.equity_before == pytest.approx(1770.0)
    assert change.equity_after == pytest.approx(1745.0)
    assert change.net_deposits == pytest.approx(200.0)      # only the deposit after Nov 6
    assert change.market_gain == pytest.approx(-225.0)
