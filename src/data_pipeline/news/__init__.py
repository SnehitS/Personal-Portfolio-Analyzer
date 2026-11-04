"""Per-ticker news from several providers, merged into one de-duplicated list.

Providers without a configured API key are skipped, so Yahoo (no key) always works.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

from src.data_pipeline.models import NewsItem
from src.data_pipeline.news import yahoo

log = logging.getLogger(__name__)

# When two providers carry the same story, keep the one earlier in this list
# (Alpha Vantage ships its own sentiment, Finnhub has fuller summaries).
PRIORITY = ["sec", "alphavantage", "finnhub", "yahoo"]


def _fetchers() -> dict:
    return {"yahoo": yahoo.fetch}


def headline_key(headline: str) -> str:
    """Normalize a headline so the same story from two providers compares equal."""
    return " ".join(re.findall(r"[a-z0-9]+", headline.lower()))


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


def fetch_news(tickers: list[str], cfg: dict, now: datetime | None = None) -> list[NewsItem]:
    """News for each ticker from every enabled provider within `lookback_days`."""
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=cfg.get("lookback_days", 7))
    fetchers = _fetchers()
    items: list[NewsItem] = []
    for name in cfg.get("sources", list(fetchers)):
        fetch = fetchers.get(name)
        if fetch is None:
            continue
        for ticker in tickers:
            try:
                items += [n for n in fetch(ticker) if n.published_at >= since]
            except Exception as exc:  # one bad provider shouldn't sink the run
                log.warning("%s news for %s failed: %s", name, ticker, exc)

    merged = dedupe(items)
    cap = cfg.get("max_per_ticker", 25)
    counts: dict[str, int] = {}
    kept = []
    for n in merged:
        counts[n.ticker] = counts.get(n.ticker, 0) + 1
        if counts[n.ticker] <= cap or n.kind == "filing":
            kept.append(n)
    return kept
