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


def test_daily_report_sections():
    from datetime import datetime, timezone

    from src.data_pipeline.models import AccountSummary, Holding, NewsItem, PortfolioSnapshot
    from src.portfolio_analysis.catalysts import Catalyst
    from src.sentiment_pipeline.aggregate import ArticleRead, TickerSentiment
    from src.update_engine import daily
    from src.update_engine.changes import compare

    before = PortfolioSnapshot("robinhood", datetime(2026, 11, 6, 16), AccountSummary(500.0),
                               [Holding("AAA", 10, 90.0, 100.0)])
    after = PortfolioSnapshot("robinhood", datetime(2026, 11, 7, 16), AccountSummary(440.0),
                              [Holding("AAA", 10, 90.0, 110.0, previous_close=105.0),
                               Holding("BBB", 3, 20.0, 20.0)])
    item = NewsItem("AAA", "yahoo", datetime(2026, 11, 7, tzinfo=timezone.utc),
                    "AAA beats estimates", "https://x/1")
    best = ArticleRead(item, 0.8, "positive", "finbert")
    sent = [TickerSentiment("AAA", 0.6, 3, 2, 0, most_positive=best),
            TickerSentiment("BBB", None, 0, 0, 0)]
    events = [Catalyst(date(2026, 11, 10), "AAA", "earnings", "after close")]

    md = daily.render(after, date(2026, 11, 7), compare(before, after), sent, events,
                      ["AAA is 68% of the portfolio"])
    assert md.startswith("# Daily report: Sat Nov 07, 2026")
    assert "+$100 (+6.67%) from the market" in md
    assert "| AAA | +10.00% | +$100 | 68.8% |" in md
    assert "Opened **BBB**: +3 shares" in md
    assert "| AAA | bullish | +0.60 | 3 |" in md and "| BBB |" not in md.split("News")[1]
    assert "- **AAA** (+0.80): AAA beats estimates" in md
    assert "| Tue Nov 10 | 3d | AAA | earnings | after close |" in md
    assert "## Risk flags" in md


def test_daily_report_first_snapshot_uses_previous_close():
    from datetime import datetime

    from src.data_pipeline.models import AccountSummary, Holding, PortfolioSnapshot
    from src.update_engine import daily

    snap = PortfolioSnapshot("csv", datetime(2026, 11, 7, 16), AccountSummary(0.0),
                             [Holding("AAA", 10, 90.0, 99.0, previous_close=100.0)])
    md = daily.render(snap, date(2026, 11, 7))
    assert "| Ticker | Today | P/L | Weight |" in md
    assert "| AAA | -1.00% | -$10 | 100.0% |" in md
    assert "Since last snapshot" not in md and "## Risk flags" not in md


def test_weekly_report_labels_and_week_start():
    from datetime import datetime

    from src.data_pipeline.models import AccountSummary, Holding, PortfolioSnapshot
    from src.update_engine import weekly
    from src.update_engine.changes import compare

    assert weekly.week_start(date(2026, 11, 13)) == datetime(2026, 11, 9)   # Fri -> Mon
    assert weekly.week_start(date(2026, 11, 9)) == datetime(2026, 11, 9)    # Mon -> Mon

    # Bought 2 more shares with cash at an unchanged price: no market gain
    before = PortfolioSnapshot("robinhood", datetime(2026, 11, 6, 16), AccountSummary(200.0),
                               [Holding("AAA", 10, 90.0, 100.0)])
    after = PortfolioSnapshot("robinhood", datetime(2026, 11, 13, 16), AccountSummary(0.0),
                              [Holding("AAA", 12, 90.0, 100.0)])
    md = weekly.render(after, date(2026, 11, 13), compare(before, after))
    assert md.startswith("# Weekly report: week of Nov 09, 2026")
    assert "- **This week:** +$0 (+0.00%) from the market" in md
    assert "## Position changes this week" in md


def test_alerts():
    from datetime import datetime, timezone

    from src.data_pipeline.models import AccountSummary, Holding, NewsItem, PortfolioSnapshot
    from src.portfolio_analysis.catalysts import Catalyst
    from src.sentiment_pipeline.aggregate import ArticleRead, TickerSentiment
    from src.update_engine import alerts

    now = datetime(2026, 11, 10, 15, tzinfo=timezone.utc)
    snap = PortfolioSnapshot("robinhood", datetime(2026, 11, 10, 10), AccountSummary(0.0), [
        Holding("BIG", 10, 100.0, 94.0, previous_close=100.0,
                week52_high=150.0, week52_low=93.0),                 # -6%, near low
        Holding("TINY", 1, 50.0, 30.0, previous_close=28.0),          # +7%, -40% vs cost
        Holding("CALM", 100, 10.0, 10.0, previous_close=10.0)])
    bad = NewsItem("BIG", "yahoo", now, "BIG recalls product", "u1")
    sent = [TickerSentiment("BIG", -0.6, 4, 0, 3, most_negative=ArticleRead(bad, -0.9, "negative", "finbert")),
            TickerSentiment("CALM", -0.9, 1, 0, 1)]                     # too few articles
    events = [Catalyst(date(2026, 11, 11), "BIG", "earnings", "after close"),
              Catalyst(date(2026, 11, 20), "CALM", "earnings"),       # too far out
              Catalyst(date(2026, 11, 11), "CALM", "option expiry", "short 1x 10C")]
    filings = [NewsItem("BIG", "sec", now, "8-K: Departure of an officer", "f1", kind="filing", form="8-K"),
               NewsItem("BIG", "sec", now, "Form 4", "f2", kind="filing", form="4")]

    found = alerts.check(snap, {}, date(2026, 11, 10), sent, events, filings, now)
    msgs = [a.message for a in found]
    assert "BIG down 6.0% today (48% of the portfolio)" in msgs
    assert "BIG is at its 52-week low" in msgs
    assert "TINY is 40% below your average cost" in msgs
    assert "BIG reports earnings tomorrow (after close)" in msgs
    assert "CALM short 1x 10C expires Wed Nov 11" in msgs
    assert any(m.startswith("BIG news is bearish (-0.60 over 4 articles): BIG recalls") for m in msgs)
    assert "BIG: 8-K: Departure of an officer" in msgs
    assert not any("CALM news" in m or "Form 4" in m or "CALM reports" in m for m in msgs)
    assert found[0].level == "high" and found[-1].level == "watch"

    md = alerts.render(found, date(2026, 11, 10))
    assert md.index("## Act on") < md.index("## Watch")
    assert "Nothing needs attention" in alerts.render([], date(2026, 11, 10))
