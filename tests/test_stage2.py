"""Stage 2: news, sentiment, options and catalysts. All offline: providers are tested on
saved payload shapes, and the LLM through a stub client."""

import json
import math
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from src.data_pipeline import earnings_calendar, storage
from src.data_pipeline.models import (
    AccountSummary, Holding, NewsItem, OptionPosition, PortfolioSnapshot, SentimentScore,
    Transaction,
)
from src.data_pipeline.news import (
    alphavantage, dedupe, finnhub, headline_key, mention_relevance, name_keyword, sec_edgar,
    yahoo,
)
from src.data_pipeline.options_data import pick_expiries
from src.portfolio_analysis import catalysts
from src.portfolio_analysis import options as op
from src.sentiment_pipeline import aggregate, finbert, llm

NOW = datetime(2026, 11, 3, 15, tzinfo=timezone.utc)


def _item(url="u", ticker="AAPL", source="yahoo", hours_ago=0, **kw):
    return NewsItem(ticker, source, NOW - timedelta(hours=hours_ago), kw.pop("headline", url),
                    url, **kw)


# --- storage ----------------------------------------------------------------------------

def test_news_storage_ignores_repeats_and_filters_in_utc(tmp_path):
    db = tmp_path / "t.db"
    eastern = timezone(timedelta(hours=-5))
    old = NewsItem("AAPL", "yahoo", datetime(2026, 11, 1, 9, tzinfo=eastern), "old", "u1")
    new = _item("u2")
    assert storage.save_news([old, new, new], db) == 2
    assert storage.save_news([new], db) == 0
    # 09:00 Eastern is 14:00 UTC, so a 13:00 UTC cutoff keeps it and 15:00 drops it
    assert len(storage.load_news(["aapl"], datetime(2026, 11, 1, 13, tzinfo=timezone.utc),
                                 db)) == 2
    assert [n.url for n in storage.load_news(
        ["AAPL"], datetime(2026, 11, 1, 15, tzinfo=timezone.utc), db)] == ["u2"]


def test_sentiment_storage_keeps_one_row_per_model(tmp_path):
    db = tmp_path / "t.db"
    storage.save_sentiment([SentimentScore("AAPL", "u", "finbert", "positive", 0.5)], db)
    storage.save_sentiment([SentimentScore("AAPL", "u", "finbert", "negative", -0.5),
                            SentimentScore("AAPL", "u", "llm", "neutral", 0.0)], db)
    rows = storage.load_sentiment(["AAPL"], db_path=db)
    assert len(rows) == 2
    assert storage.load_sentiment(model="finbert", db_path=db)[0].score == -0.5


def test_iv_history(tmp_path):
    db = tmp_path / "t.db"
    storage.save_iv("NVDA", date(2026, 11, 2), 0.40, 200, db)
    storage.save_iv("NVDA", date(2026, 11, 3), 0.45, 205, db)
    storage.save_iv("NVDA", date(2026, 11, 3), 0.47, 206, db)  # same day replaces
    assert storage.load_iv_history("NVDA", date(2026, 11, 1), db) == [0.40, 0.47]


# --- providers ----------------------------------------------------------------------------

RSS = """<?xml version="1.0" encoding="UTF-8"?><rss><channel>
<item><title>Apple — “big” day</title><link>https://x.com/a?.tsrc=rss</link>
<description>Summary</description><pubDate>Tue, 03 Nov 2026 21:27:49 +0000</pubDate></item>
<item><title>No date</title><link>https://x.com/b</link></item>
</channel></rss>""".encode("utf-8")


def test_yahoo_rss_parses_utf8_bytes_and_strips_tracking():
    items = yahoo.parse_feed("AAPL", RSS)
    assert len(items) == 1
    assert items[0].headline == "Apple — “big” day"
    assert items[0].url == "https://x.com/a"
    assert items[0].published_at == datetime(2026, 11, 3, 21, 27, 49, tzinfo=timezone.utc)


def test_finnhub_parse():
    items = finnhub.parse("AAPL", [
        {"datetime": 1762200000, "headline": " Beat ", "url": "u", "source": "Reuters"},
        {"datetime": 1762200000, "headline": "", "url": "u2"},
    ])
    assert [(n.headline, n.publisher) for n in items] == [("Beat", "Reuters")]


def test_alphavantage_parse_and_quota():
    payload = {"feed": [{
        "title": "T", "url": "u", "time_published": "20261103T120000", "source": "B",
        "ticker_sentiment": [{"ticker": "MSFT", "relevance_score": "0.9",
                              "ticker_sentiment_score": "0.5"},
                             {"ticker": "AAPL", "relevance_score": "0.3",
                              "ticker_sentiment_score": "-0.2"}],
    }]}
    (n,) = alphavantage.parse("AAPL", payload)
    assert (n.provider_sentiment, n.relevance) == (-0.2, 0.3)
    with pytest.raises(alphavantage.RateLimited):
        alphavantage.parse("AAPL", {"Information": "standard API rate limit is 25 requests per day"})


def test_sec_filings_describe_8k_items_and_stop_at_cutoff():
    payload = {"name": "Apple Inc.", "filings": {"recent": {
        "form": ["8-K", "424B2", "4", "10-Q"],
        "acceptanceDateTime": ["2026-11-02T21:30:23.000Z", "2026-11-01T10:00:00.000Z",
                               "2026-10-29T22:30:14.000Z", "2026-08-01T20:00:00.000Z"],
        "filingDate": ["2026-11-02", "2026-11-01", "2026-10-29", "2026-08-01"],
        "accessionNumber": ["0000320193-26-000024", "x", "0001140361-26-041201", "y"],
        "primaryDocument": ["aapl-20261102.htm", "d.htm", "xslF345X06/form4.xml", "q.htm"],
        "items": ["2.02,9.01", "", "", ""],
    }}}
    items = sec_edgar.parse_submissions("AAPL", 320193, payload, NOW - timedelta(days=30))
    assert [n.form for n in items] == ["8-K", "4"]  # 424B2 isn't tracked; 10-Q is too old
    assert items[0].headline == "Apple Inc. filed 8-K: earnings results"
    assert items[0].url.endswith("/320193/000032019326000024/aapl-20261102.htm")
    assert all(n.kind == "filing" for n in items)
    assert sec_edgar.describe("8-K", "5.02,8.01") == "executive or director change"
    assert sec_edgar.describe("8-K", "8.01") == "other event"


def test_earnings_calendar_parse_keeps_earliest_per_ticker():
    out = earnings_calendar.parse([
        {"symbol": "AAPL", "date": "2027-01-28", "hour": "amc", "epsEstimate": 2.9},
        {"symbol": "AAPL", "date": "2026-11-20", "hour": "bmo", "epsEstimate": 1.1},
        {"symbol": "ZZZ", "date": "2026-11-10"},
    ], {"AAPL"})
    assert out == {"AAPL": {"date": date(2026, 11, 20), "hour": "bmo", "eps_estimate": 1.1,
                            "revenue_estimate": None}}


# --- merging and relevance ------------------------------------------------------------------

def test_dedupe_prefers_richer_source_and_matches_headlines():
    a = _item("https://yahoo/1", headline="Apple Beats Estimates!", source="yahoo")
    b = _item("https://av/1", headline="Apple beats estimates", source="alphavantage",
              provider_sentiment=0.4)
    c = _item("https://yahoo/1", ticker="MSFT", headline="Apple beats estimates")
    out = dedupe([a, b, c])
    assert sorted((n.ticker, n.source) for n in out) == [("AAPL", "alphavantage"),
                                                          ("MSFT", "yahoo")]
    assert headline_key("Apple's  Q3: beat!") == "apple s q3 beat"


def test_mention_relevance():
    assert name_keyword("NVIDIA Corporation") == "nvidia"
    assert name_keyword("The Walt Disney Company") == "walt"
    rel = lambda h, s="": mention_relevance(_item(ticker="NVDA", headline=h, summary=s),
                                            "NVIDIA Corporation")
    assert rel("Nvidia raises guidance") == 1.0
    assert rel("Why NVDA fell") == 1.0
    assert rel("Chip stocks rally", "Nvidia led gains") == 0.5
    assert rel("Why Herbalife stock is up") == 0.1
    assert rel("Unvidiax launches") == 0.1


# --- sentiment ------------------------------------------------------------------------------

def test_finbert_score_is_positive_minus_negative():
    s = finbert.to_score(_item(), {"positive": 0.7, "negative": 0.2, "neutral": 0.1})
    assert (s.label, s.score, s.confidence) == ("positive", 0.5, 0.7)


class _FakeMessages:
    def __init__(self, body=None, stop_reason="end_turn"):
        self.body, self.stop_reason, self.calls = body, stop_reason, []

    def create(self, **kw):
        self.calls.append(kw)
        content = [NS(type="thinking"), NS(type="text", text=json.dumps(self.body or {}))]
        return NS(stop_reason=self.stop_reason, _request_id="req_1", content=content)


def test_llm_scorer_skips_filings_maps_ids_and_clamps():
    fake = _FakeMessages({"articles": [
        {"id": 0, "label": "negative", "score": -1.6, "confidence": 0.8, "rationale": "Miss."},
        {"id": 1, "label": "neutral", "score": 0.0, "confidence": 0.9, "rationale": "Macro."},
        {"id": 7, "label": "neutral", "score": 0.0, "confidence": 0.9, "rationale": "?"},
    ]})
    scorer = llm.LlmScorer(client=NS(beta=NS(messages=fake)))
    items = [_item("u0"), _item("u1", kind="filing"), _item("u2")]
    out = scorer.score(items, {"AAPL": "Apple Inc."})
    assert [(s.url, s.score) for s in out] == [("u0", -1.0), ("u2", 0.0)]
    call = fake.calls[0]
    assert call["model"] == llm.DEFAULT_MODEL
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "Apple Inc." in call["messages"][0]["content"]
    assert "[1]" in call["messages"][0]["content"] and "[2]" not in call["messages"][0]["content"]


def test_llm_scorer_returns_nothing_on_refusal():
    scorer = llm.LlmScorer(client=NS(beta=NS(messages=_FakeMessages(stop_reason="refusal"))))
    assert scorer.score([_item()]) == []


def test_best_reads_prefer_llm_then_finbert_then_provider():
    news = [_item("u1"), _item("u2"), _item("u3", provider_sentiment=0.3),
            _item("u4", kind="filing")]
    scores = [SentimentScore("AAPL", "u1", "finbert", "negative", -0.8),
              SentimentScore("AAPL", "u1", "claude-opus-5-5", "negative", -0.4),
              SentimentScore("AAPL", "u2", "finbert", "positive", 0.6)]
    reads = aggregate.best_reads(news, scores)
    assert [(r.item.url, r.model, r.score) for r in reads] == [
        ("u1", "claude-opus-5-5", -0.4), ("u2", "finbert", 0.6), ("u3", "yahoo", 0.3)]


def test_summary_weights_recent_relevant_articles_more():
    reads = [
        aggregate.ArticleRead(_item("new"), -0.5, "negative", "m", 1.0),
        aggregate.ArticleRead(_item("old", hours_ago=96), 0.8, "positive", "m", 1.0),
        aggregate.ArticleRead(_item("off", relevance=0.5), 0.3, "positive", "m"),
    ]
    ts = aggregate.summarize(reads, "AAPL", half_life_hours=48, now=NOW)
    # weights 1, 0.25, 0.5  ->  (-0.5 + 0.2 + 0.15) / 1.75
    assert ts.score == pytest.approx(-0.15 / 1.75)
    assert (ts.articles, ts.positive, ts.negative) == (3, 2, 1)
    assert ts.most_negative.item.url == "new" and ts.most_positive.item.url == "old"
    assert aggregate.summarize(reads, "MSFT").score is None


# --- options --------------------------------------------------------------------------------

def _chain(ivs_calls, ivs_puts, bids=(6, 3, 1)):
    strikes = [95, 100, 105]
    calls = pd.DataFrame({"strike": strikes, "bid": list(bids), "ask": [b + 0.4 for b in bids],
                          "lastPrice": list(bids), "impliedVolatility": ivs_calls})
    puts = pd.DataFrame({"strike": strikes, "bid": list(bids[::-1]),
                         "ask": [b + 0.4 for b in bids[::-1]], "lastPrice": list(bids[::-1]),
                         "impliedVolatility": ivs_puts})
    return calls, puts


def test_expiry_view_uses_atm_straddle_and_quoted_iv():
    calls, puts = _chain([0.30, 0.32, 0.34], [0.36, 0.33, 0.31])
    calls.loc[0, ["bid", "ask"]] = 0  # unquoted strike: its IV is ignored
    v = op.expiry_view(calls, puts, 101, date(2026, 12, 3), today=date(2026, 11, 3))
    assert v.atm_strike == 100 and v.days == 30
    assert v.straddle == pytest.approx(3.2 + 3.2)
    assert v.expected_move_pct == pytest.approx(6.4 / 101)
    # nearest four strikes to 101 are all three; quoted IVs: calls 0.32, 0.34; puts all three
    assert v.atm_iv == pytest.approx((0.32 + 0.34 + 0.36 + 0.33 + 0.31) / 5)
    assert v.iv_move_pct == pytest.approx(v.atm_iv * math.sqrt(30 / 365))


def test_interpolated_iv_in_total_variance():
    a = op.ExpiryView(date(2026, 11, 23), 20, 100, 0.30, None, 100)
    b = op.ExpiryView(date(2026, 12, 13), 40, 100, 0.40, None, 100)
    expected = math.sqrt((0.09 * 20 + (0.16 * 40 - 0.09 * 20) * 0.5) / 30)
    assert op.interpolated_iv([b, a], 30) == pytest.approx(expected)
    assert op.interpolated_iv([a, b], 10) == 0.30 and op.interpolated_iv([a, b], 90) == 0.40


def test_earnings_move_strips_ordinary_variance():
    pre = op.ExpiryView(date(2026, 11, 13), 10, 100, None, 4.0, 100)    # 4% over 10 days
    post = op.ExpiryView(date(2026, 11, 20), 17, 100, None, 8.0, 100)   # 8% over 17 days
    move = op.earnings_move([pre, post], date(2026, 11, 17))
    assert move == pytest.approx(math.sqrt(0.08 ** 2 - 0.04 ** 2 * 16 / 10))
    assert op.earnings_move([post], date(2026, 11, 17)) == 0.08
    assert op.earnings_move([pre], date(2026, 11, 17)) is None


def test_iv_rank_and_realized_vol():
    assert op.iv_rank(0.5, [0.3] * 10) is None
    assert op.iv_rank(0.4, [0.2] * 10 + [0.6] * 10) == pytest.approx(0.5)
    flat = pd.Series([100.0] * 40)
    assert op.realized_vol(flat) == 0.0
    assert op.realized_vol(pd.Series([100.0] * 5)) is None


def test_past_earnings_moves_span_the_report_date():
    s = pd.Series([100, 101, 100, 110, 111],
                  index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05",
                                        "2026-01-06", "2026-01-07"]))
    assert op.past_earnings_moves(s, [date(2026, 1, 5), date(2025, 1, 1)]) == [
        pytest.approx(110 / 101 - 1)]


def test_pick_expiries_covers_30_days_and_earnings():
    today = date(2026, 11, 3)
    exps = [today] + [today + timedelta(days=d) for d in (1, 3, 8, 10, 15, 17, 24, 31, 59, 150)]
    picked = pick_expiries(exps, today, nearest=2, must_cover=[date(2026, 12, 25)])
    assert picked == [today + timedelta(days=d) for d in (1, 3, 31, 59)]


# --- catalysts ------------------------------------------------------------------------------

def test_catalysts_merge_sources_within_horizon():
    today = date(2026, 11, 3)
    snap = PortfolioSnapshot(
        "t", datetime.now(), AccountSummary(0),
        holdings=[Holding("AAPL", 10, 100, 200, next_earnings=date(2026, 11, 20),
                          ex_dividend_date=date(2026, 11, 10), dividend_yield=0.4),
                  Holding("NVDA", 5, 100, 200, next_earnings=date(2027, 2, 25)),
                  Holding("MSFT", 1, 100, 100)],  # lots won't match: skipped for long-term
        transactions=[Transaction("NVDA", "buy", date(2025, 11, 20), 5, 100, -500, "r1")],
        options=[OptionPosition("NVDA", "call", 250, date(2026, 11, 20), -1, 50)],
    )
    detail = {"AAPL": {"date": date(2026, 11, 19), "hour": "amc", "eps_estimate": 2.9}}
    macro = [{"date": "2026-11-25", "event": "FOMC minutes"},
             {"date": "2026-12-20", "event": "too late"}]
    events = catalysts.build(snap, today, 30, macro, detail)
    assert [(c.day.day, c.ticker, c.kind) for c in events] == [
        (10, "AAPL", "ex-dividend"), (19, "AAPL", "earnings"), (20, "NVDA", "option expiry"),
        (21, "NVDA", "long-term"), (25, "", "macro")]
    assert events[1].detail == "after close, EPS est. 2.90"
    assert events[2].detail == "short 1x 250C"
