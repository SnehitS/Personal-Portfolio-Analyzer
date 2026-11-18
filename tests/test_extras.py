"""Alpha Vantage extras: insider trades, macro backdrop and earnings calls (all offline)."""

from datetime import date, datetime, timezone
from types import SimpleNamespace

from src.data_pipeline import alpha_vantage as av
from src.data_pipeline.models import (
    AccountSummary, Holding, InsiderTrade, PortfolioSnapshot,
)


def _trade(code="P", shares=1000.0, price=50.0, day=date(2026, 11, 2), who="TAN, LIP BU",
           ticker="INTC"):
    return InsiderTrade(ticker, day, who, "Director, CEO", code, shares, price)


def test_parse_insiders_skips_old_and_bad_rows():
    payload = {"data": [
        {"transaction_date": "2026-11-02", "executive": "TAN, LIP BU", "executive_title": "CEO",
         "acquisition_or_disposal": "P", "shares": "150000.0", "share_price": "64.7312"},
        {"transaction_date": "2026-10-30", "executive": "X", "acquisition_or_disposal": "M",
         "shares": "33008.0", "share_price": ""},
        {"transaction_date": "2024-01-01", "executive": "OLD", "acquisition_or_disposal": "S",
         "shares": "5", "share_price": "1"},
        {"transaction_date": "bad", "shares": "1"},
    ]}
    trades = av.parse_insiders("INTC", payload, since=date(2026, 1, 1))
    assert [t.code for t in trades] == ["P", "M"]
    assert trades[0].name == "Lip Bu Tan" and round(trades[0].value) == 9_709_680
    assert trades[1].price is None and trades[1].value is None


def test_insiders_combine_fills_and_pick_notable():
    from src.portfolio_analysis import insiders

    fills = [_trade("S", 100_000, 350.0, who="SAMUELI, HENRY", ticker="AVGO"),
             _trade("S", 50_000, 360.0, who="SAMUELI, HENRY", ticker="AVGO")]
    small_buy = _trade("P", 100, 50.0, who="ROE, RICH")  # $5K: below min_buy
    big_buy = _trade("P", 150_000, 64.73)
    paperwork = _trade("F", 14_739, 66.0)              # tax withholding
    small_sale = _trade("S", 1_000, 100.0, who="DOE, JANE")
    picked = insiders.notable([*fills, small_buy, big_buy, paperwork, small_sale])
    assert [(t.ticker, t.code) for t in picked] == [("INTC", "P"), ("AVGO", "S")]
    merged = picked[1]
    assert merged.shares == 150_000 and round(merged.price, 2) == 353.33
    assert insiders.describe(picked[0]) == \
        "INTC: Lip Bu Tan (Director, CEO) bought 150,000 shares (~$9.7M) on Nov 02"
    assert insiders.notable([small_buy], {"min_buy": 1_000}) == [small_buy]


def test_quarter_labels():
    assert av.quarter_candidates(date(2026, 10, 21)) == ["2026Q3", "2026Q4"]   # NFLX in Oct
    assert av.quarter_candidates(date(2026, 12, 10)) == ["2026Q4", "2027Q1"]   # AVGO in Dec
    assert av.quarter_candidates(date(2026, 1, 20)) == ["2025Q4", "2026Q1"]
    assert av.quarter_candidates(date(2026, 11, 17)) == ["2026Q3", "2026Q4"]  # a late Q3 report
    assert av.recent_quarters(date(2026, 2, 1)) == ["2026Q1", "2025Q4"]


def test_macro_backdrop():
    from src.portfolio_analysis import macro

    tsy = [(date(2026, 10, 16), 5.09), (date(2026, 11, 6), 4.88), (date(2026, 11, 16), 5.12)]
    ff = [(date(2026, 7, 1), 3.63), (date(2026, 10, 1), 3.88)]
    cpi = [(date(2025, 9, 1), 100.0), (date(2025, 10, 1), 100.0),
           (date(2026, 9, 1), 103.7), (date(2026, 10, 1), 103.9)]
    lines = macro.backdrop({"treasury_10y": tsy, "fed_funds": ff, "cpi": cpi})
    assert lines == [
        "- **10-year Treasury:** 5.12% (+24 bp over 1 week, +3 bp over 1 month), as of Nov 16",
        "- **Fed funds:** 3.88% (+25 bp over 3 months), October average",
        "- **Inflation (CPI):** 3.9% year over year in October, up from 3.7% the month before",
    ]
    assert macro.backdrop({}) == []


def _snap():
    return PortfolioSnapshot("robinhood", datetime(2026, 11, 18, 16), AccountSummary(0.0), [
        Holding("NFLX", 10, 90.0, 60.0, sector="Communication Services"),
        Holding("INTC", 10, 100.0, 57.0, sector="Technology"),
        Holding("SPY", 1, 500.0, 600.0, sector="ETF")])


def test_refresh_stores_everything_and_skips_what_is_fresh(monkeypatch, tmp_path):
    from src.data_pipeline import storage
    from src.update_engine import extras

    db = tmp_path / "p.db"
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test")
    calls = []

    def fake_query(self, function, **params):
        calls.append((function, params.get("symbol"), params.get("quarter")))
        self.calls_left -= 1
        if function == "INSIDER_TRANSACTIONS":
            return {"data": [{"transaction_date": "2026-11-10", "executive": "TAN, LIP BU",
                              "acquisition_or_disposal": "P", "shares": "150000",
                              "share_price": "64.73"}] if params["symbol"] == "INTC" else []}
        if function == "EARNINGS_CALL_TRANSCRIPT":
            text = [{"speaker": "CFO", "content": "Revenue grew."}] \
                if params["quarter"] == "2026Q3" else []
            return {"symbol": params["symbol"], "transcript": text}
        return {"data": [{"date": "2026-11-16", "value": "5.12"}]}

    monkeypatch.setattr(av.Client, "query", fake_query)
    now = datetime(2026, 11, 18, 21, tzinfo=timezone.utc)
    reports = {"NFLX": {"date": date(2026, 11, 17), "hour": "amc"}}
    summaries = []

    def summarize(t, q, entries, cfg):
        summaries.append((t, q))
        return "- Revenue grew."

    res = extras.refresh(_snap(), {}, db, date(2026, 11, 18), now, reports, summarize)
    assert res.calls_new == [("NFLX", "2026Q3")] and summaries == [("NFLX", "2026Q3")]
    assert res.insider_trades_new == 1
    assert sorted(res.macro_updated) == ["cpi", "fed_funds", "treasury_10y"]
    assert ("INSIDER_TRANSACTIONS", "SPY", None) not in calls          # ETFs have no insiders
    assert [c[2] for c in calls if c[0] == "EARNINGS_CALL_TRANSCRIPT"] == ["2026Q3"]
    assert storage.load_earnings_calls(["NFLX"], db_path=db)[0]["summary"] == "- Revenue grew."

    calls.clear()
    again = extras.refresh(_snap(), {}, db, date(2026, 11, 18), now, reports, summarize)
    assert calls == [] and again.calls_used == 0                    # same day: nothing refetched

    new = extras.new_insider_trades(["INTC", "NFLX"], {}, db, now)
    assert [t.ticker for t in new] == ["INTC"]
    md = extras.insider_section(["INTC"], {}, db, date(2026, 11, 18), days=30)
    assert md[0] == "## Insider trades, last 30 days" and "bought 150,000" in md[2]
    assert extras.calls_section(["NFLX"], db, now, days=3)[2].startswith("**NFLX 2026Q3**")
    assert extras.macro_section(db, date(2026, 11, 18))[2].startswith("- **10-year Treasury:**")


def test_refresh_without_a_key_does_nothing(monkeypatch, tmp_path):
    from src.update_engine import extras

    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    res = extras.refresh(_snap(), {}, tmp_path / "p.db", reports={})
    assert res.stopped == "ALPHAVANTAGE_API_KEY isn't set" and res.calls_used == 0


def test_client_stops_at_quota(monkeypatch):
    import pytest

    from src.data_pipeline.news.alphavantage import RateLimited

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test")
    resp = SimpleNamespace(json=lambda: {"Information": "Our standard API rate limit is 25 "
                                                        "requests per day."})
    monkeypatch.setattr(av.http, "get", lambda *a, **kw: resp)
    client = av.Client(5)
    with pytest.raises(RateLimited):
        client.query("CPI")
    assert not client.usable and client.query("CPI") is None


def test_alerts_include_insider_buys_and_new_calls():
    from src.update_engine import alerts

    found = alerts.check(_snap(), {}, date(2026, 11, 18),
                         insider_trades=[_trade("P", 150_000, 64.73),
                                         _trade("S", 30_000, 66.0, who="PETERS, GREG",
                                                ticker="NFLX")],
                         calls=[("NFLX", "2026Q3", "- Ads are behind target.\n- Churn fell.")])
    by_kind = {(a.ticker, a.kind): a for a in found}
    assert by_kind[("INTC", "insider")].level == "high"       # a buy in a big position
    assert by_kind[("NFLX", "insider")].level == "watch"      # sales are context
    assert by_kind[("NFLX", "call")].message == "NFLX 2026Q3 earnings call: Ads are behind target."


def test_earnings_call_summary_keeps_bullets():
    from src.sentiment_pipeline import earnings_call

    seen = {}

    class FakeMessages:
        def create(self, **kw):
            seen.update(kw)
            text = "Here's the summary:\n- Guidance raised.\n- Margins up."
            return SimpleNamespace(stop_reason="end_turn", _request_id="r",
                                   content=[SimpleNamespace(type="text", text=text)])

    out = earnings_call.summarize("NFLX", "2026Q3",
                                  [{"speaker": "CFO", "title": "CFO", "content": "We raised it."}],
                                  client=SimpleNamespace(messages=FakeMessages()))
    assert out == "- Guidance raised.\n- Margins up."
    assert seen["model"] == "claude-haiku-5-5" and seen["output_config"] == {"effort": "low"}
    assert "CFO (CFO): We raised it." in seen["messages"][0]["content"]
