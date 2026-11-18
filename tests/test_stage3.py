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


def _alert_snap():
    from datetime import datetime

    from src.data_pipeline.models import AccountSummary, Holding, PortfolioSnapshot

    return PortfolioSnapshot("robinhood", datetime(2026, 11, 10, 10), AccountSummary(0.0), [
        Holding("BIG", 10, 100.0, 105.0, previous_close=100.0),       # exactly +5%, 49% weight
        Holding("SMALL", 1, 100.0, 80.0, previous_close=84.0),        # -4.8%, exactly -20% vs cost
        Holding("REST", 100, 10.0, 10.0, previous_close=10.0)])


def test_alerts_thresholds_are_inclusive():
    from src.update_engine import alerts

    found = alerts.check(_alert_snap(), {}, date(2026, 11, 10))
    msgs = [a.message for a in found]
    assert "BIG up 5.0% today (49% of the portfolio)" in msgs
    assert "SMALL is 20% below your average cost" in msgs
    assert not any(m.startswith("SMALL up") or m.startswith("SMALL down") for m in msgs)


def test_alerts_level_follows_position_weight():
    from src.update_engine import alerts

    found = alerts.check(_alert_snap(), {"big_move_pct": 0.04}, date(2026, 11, 10))
    by_ticker = {(a.ticker, a.kind): a.level for a in found}
    assert by_ticker[("BIG", "move")] == "high"
    assert by_ticker[("SMALL", "move")] == "watch"
    assert by_ticker[("SMALL", "loss")] == "watch"


def test_alerts_skip_past_events_and_stale_filings():
    from datetime import datetime, timedelta, timezone

    from src.data_pipeline.models import NewsItem
    from src.portfolio_analysis.catalysts import Catalyst
    from src.update_engine import alerts

    now = datetime(2026, 11, 10, 15, tzinfo=timezone.utc)
    events = [Catalyst(date(2026, 11, 9), "BIG", "earnings"),
              Catalyst(date(2026, 11, 9), "REST", "option expiry", "long 1x 10C")]
    filings = [NewsItem("BIG", "sec", now - timedelta(hours=30), "8-K: Old news", "f1",
                        kind="filing", form="8-K")]
    found = alerts.check(_alert_snap(), {"big_move_pct": 1.0, "loss_from_cost": -1.0},
                         date(2026, 11, 10), [], events, filings, now)
    assert found == []
    assert "Nothing needs attention" in alerts.render(found, date(2026, 11, 10))


def test_notify_compose():
    from src.update_engine.alerts import Alert
    from src.update_engine import notify

    found = [Alert("high", "BIG", "move", "BIG down 6.0% today"),
             Alert("watch", "TINY", "loss", "TINY is 40% below your average cost")]
    msg = notify.compose(found, [], {}, "Equity $10,000.00, today -$300 (-2.91%)")
    assert msg.title == "Portfolio: 1 high, 1 watch alerts"
    assert msg.body.splitlines() == ["Equity $10,000.00, today -$300 (-2.91%)",
                                     "! BIG down 6.0% today", "- TINY is 40% below your average cost"]
    assert msg.urgent

    only_high = notify.compose(found, [], {"min_level": "high"})
    assert only_high.title == "Portfolio: 1 high alert" and "TINY" not in only_high.body

    assert notify.compose([], [], {}) is None
    assert notify.compose([], [], {"always": True}).title == "Portfolio: all clear"
    broke = notify.compose([], ["snapshot failed: login"], {})
    assert broke.title == "Portfolio: run had problems" and "x snapshot failed: login" in broke.body
    assert not broke.urgent


def test_notify_send_reports_each_channel(monkeypatch):
    from src.update_engine import notify

    sent = []
    monkeypatch.setitem(notify.CHANNELS, "desktop", lambda m, c: sent.append(m.title))
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    msg = notify.Message("t", "b")
    results = notify.send(msg, {"channels": ["desktop", "ntfy", "pager"]})
    assert sent == ["t"]
    assert results["desktop"] == "sent"
    assert results["ntfy"].startswith("failed: set NTFY_TOPIC")
    assert results["pager"].startswith("unknown channel")


def test_run_keeps_going_after_a_failed_step(monkeypatch, tmp_path):
    from argparse import Namespace

    from src import cli
    from src.update_engine import notify

    calls, sent = [], []

    def boom(args, cfg):
        calls.append("snapshot")
        raise EOFError("no MFA code")

    def report(args, cfg):
        calls.append(args.kind)
        return [] if args.kind == "alerts" else None

    monkeypatch.setattr(cli, "cmd_snapshot", boom)
    monkeypatch.setattr(cli, "cmd_news", lambda a, c: calls.append("news"))
    monkeypatch.setattr(cli, "cmd_report", report)
    monkeypatch.setattr(cli.storage, "load_latest_snapshot", lambda db: None)
    monkeypatch.setattr(notify, "send", lambda m, c: sent.append(m) or {"desktop": "sent"})
    monkeypatch.setattr(cli, "date", type("D", (), {"today": staticmethod(lambda: date(2026, 11, 13))}))

    args = Namespace(force=False, weekly=False, no_snapshot=False, no_news=False, no_notify=False)
    with pytest.raises(SystemExit):
        cli.cmd_run(args, {"database": str(tmp_path / "x.db")})
    assert calls == ["snapshot", "news", "daily", "weekly", "alerts"]   # Nov 13 2026 is a Friday
    assert sent and sent[0].title == "Portfolio: run had problems"
    assert "snapshot failed: no MFA code" in sent[0].body


def test_run_skips_weekends(monkeypatch):
    from argparse import Namespace

    from src import cli

    monkeypatch.setattr(cli, "date", type("D", (), {"today": staticmethod(lambda: date(2026, 11, 14))}))
    monkeypatch.setattr(cli, "cmd_snapshot", lambda a, c: pytest.fail("ran on a Saturday"))
    cli.cmd_run(Namespace(force=False, weekly=False, no_snapshot=False, no_news=False,
                          no_notify=False), {})


def test_redact_hides_api_keys():
    from src.data_pipeline.news.http import redact

    msg = "Max retries exceeded with url: /api?symbol=CAVA&token=abc123 (Caused by X)"
    assert redact(msg) == "Max retries exceeded with url: /api?symbol=CAVA&token=*** (Caused by X)"
    assert redact("q?function=CPI&apikey=XYZ") == "q?function=CPI&apikey=***"


def test_wait_for_network():
    from src import cli

    answers = iter([False, False, True])
    slept = []
    assert cli._wait_for_network(10, check=lambda: next(answers), sleep=slept.append)
    assert slept == [15, 15]
    assert not cli._wait_for_network(0, check=lambda: False, sleep=slept.append)
