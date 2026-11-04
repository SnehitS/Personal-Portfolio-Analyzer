"""Per-ticker news from several providers, merged into one de-duplicated list.

Providers without a configured API key are skipped, so Yahoo (no key) always works.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

from src.data_pipeline.models import NewsItem
from src.data_pipeline.news import alphavantage, finnhub, sec_edgar, yahoo

log = logging.getLogger(__name__)

# When two providers carry the same story, keep the one earlier in this list
# (Alpha Vantage ships its own sentiment, Finnhub has fuller summaries).
PRIORITY = ["sec", "alphavantage", "finnhub", "yahoo"]


def _fetchers(cfg: dict) -> dict:
    """Provider name -> fetch(ticker, since). Keyed providers only appear with a key."""
    out = {"yahoo": lambda ticker, since: yahoo.fetch(ticker), "sec": sec_edgar.Fetcher()}
    if finnhub.api_key():
        out["finnhub"] = finnhub.fetch
    if alphavantage.api_key():
        out["alphavantage"] = alphavantage.Fetcher(cfg.get("alphavantage_max_calls", 20))
    return out


def headline_key(headline: str) -> str:
    """Normalize a headline so the same story from two providers compares equal."""
    return " ".join(re.findall(r"[a-z0-9]+", headline.lower()))


_CORP_WORDS = {"inc", "corp", "corporation", "co", "company", "ltd", "plc", "holdings",
               "group", "the", "class", "sa", "nv", "ag", "llc", "lp", "trust", "etf"}


def name_keyword(name: str | None) -> str | None:
    """The distinctive word of a company name: "NVIDIA Corporation" -> "nvidia"."""
    words = [w for w in re.findall(r"[a-z0-9&]+", (name or "").lower()) if w not in _CORP_WORDS]
    return words[0] if words and len(words[0]) >= 3 else None


def mention_relevance(item: NewsItem, name: str | None = None) -> float:
    """1.0 if the headline names the company, 0.5 if only the summary does, else 0.1.
    Feeds like Yahoo's mix in market roundups and other companies' news."""
    ticker = re.escape(item.ticker)
    pattern = rf"(?<![A-Za-z]){ticker}(?![A-Za-z])"
    word = name_keyword(name)

    def mentions(text: str) -> bool:
        return bool(re.search(pattern, text)) or bool(
            word and re.search(rf"\b{re.escape(word)}", text.lower()))

    if mentions(item.headline):
        return 1.0
    return 0.5 if mentions(item.summary) else 0.1


def dedupe(items: list[NewsItem]) -> list[NewsItem]:
    """One item per (ticker, URL) and per (ticker, headline), preferring PRIORITY order."""
    rank = {name: i for i, name in enumerate(PRIORITY)}
    seen_urls: set[tuple[str, str]] = set()
    seen_heads: set[tuple[str, str]] = set()
    out = []
    for n in sorted(items, key=lambda n: rank.get(n.source, len(rank))):
        url_key, head_key = (n.ticker, n.url), (n.ticker, headline_key(n.headline))
        if url_key in seen_urls or head_key in seen_heads:
            continue
        seen_urls.add(url_key)
        seen_heads.add(head_key)
        out.append(n)
    return sorted(out, key=lambda n: n.published_at, reverse=True)


def fetch_news(tickers: list[str], cfg: dict, now: datetime | None = None,
               names: dict[str, str] | None = None) -> list[NewsItem]:
    """News for each ticker from every enabled provider within `lookback_days`.
    `names` (ticker -> company name) sharpens the relevance guess for unscored feeds."""
    names = names or {}
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=cfg.get("lookback_days", 7))
    fetchers = _fetchers(cfg)
    items: list[NewsItem] = []
    for name in cfg.get("sources", list(fetchers)):
        fetch = fetchers.get(name)
        if fetch is None:
            continue
        for ticker in tickers:
            try:
                items += [n for n in fetch(ticker, since) if n.published_at >= since]
            except Exception as exc:  # one bad provider shouldn't sink the run
                log.warning("%s news for %s failed: %s", name, ticker, exc)

    for n in items:
        if n.relevance is None and n.kind == "article":
            n.relevance = mention_relevance(n, names.get(n.ticker))
    merged = dedupe(items)
    cap = cfg.get("max_per_ticker", 25)
    counts: dict[str, int] = {}
    kept = []
    for n in merged:
        counts[n.ticker] = counts.get(n.ticker, 0) + 1
        if counts[n.ticker] <= cap or n.kind == "filing":
            kept.append(n)
    return kept
