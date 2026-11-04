"""Alpha Vantage NEWS_SENTIMENT (free key, 25 calls/day). Each article comes with
Alpha Vantage's own per-ticker sentiment and relevance scores."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from src.data_pipeline.models import NewsItem
from src.data_pipeline.news import http

log = logging.getLogger(__name__)

URL = "https://www.alphavantage.co/query"


class RateLimited(Exception):
    """Alpha Vantage answers HTTP 200 with an "Information" note once the daily quota is
    used, so the caller can stop asking for the rest of the run."""


def api_key() -> str | None:
    return os.getenv("ALPHAVANTAGE_API_KEY") or None


def _ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def parse(ticker: str, payload: dict) -> list[NewsItem]:
    if "feed" not in payload:
        note = payload.get("Information") or payload.get("Note") or payload.get("Error Message")
        if note and any(w in note.lower() for w in ("rate limit", "premium", "per day")):
            raise RateLimited(note)
        if note:
            log.warning("Alpha Vantage for %s: %s", ticker, note)
        return []
    items = []
    for art in payload["feed"]:
        mine = next((s for s in art.get("ticker_sentiment", [])
                     if s.get("ticker", "").upper() == ticker), None)
        try:
            published = _ts(art["time_published"])
        except (KeyError, ValueError):
            continue
        items.append(NewsItem(
            ticker=ticker, source="alphavantage", published_at=published,
            headline=art.get("title", "").strip(), url=art.get("url", ""),
            summary=(art.get("summary") or "").strip(), publisher=art.get("source") or "",
            provider_sentiment=float(mine["ticker_sentiment_score"]) if mine else None,
            relevance=float(mine["relevance_score"]) if mine else None,
        ))
    return [n for n in items if n.headline and n.url]


class Fetcher:
    """Spends at most `max_calls` requests per run and stops once the quota is hit."""

    def __init__(self, max_calls: int = 20):
        self.calls_left = max_calls

    def __call__(self, ticker: str, since: datetime) -> list[NewsItem]:
        key = api_key()
        if not key or self.calls_left <= 0:
            return []
        self.calls_left -= 1
        params = {"function": "NEWS_SENTIMENT", "tickers": ticker, "limit": 50,
                  "time_from": since.strftime("%Y%m%dT%H%M"), "apikey": key}
        resp = http.get(URL, params, min_interval=1.0)
        if resp is None:
            return []
        try:
            return parse(ticker, resp.json())
        except RateLimited as exc:
            log.warning("Alpha Vantage quota reached, skipping it for this run: %s", exc)
            self.calls_left = 0
            return []
