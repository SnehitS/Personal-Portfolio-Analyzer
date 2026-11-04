"""Finnhub company news (free key, 60 calls/min): https://finnhub.io/docs/api/company-news"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from src.data_pipeline.models import NewsItem
from src.data_pipeline.news import http

URL = "https://finnhub.io/api/v1/company-news"


def api_key() -> str | None:
    return os.getenv("FINNHUB_API_KEY") or None


def parse(ticker: str, rows: list[dict]) -> list[NewsItem]:
    items = []
    for r in rows:
        if not (r.get("headline") and r.get("url") and r.get("datetime")):
            continue
        items.append(NewsItem(
            ticker=ticker, source="finnhub",
            published_at=datetime.fromtimestamp(r["datetime"], tz=timezone.utc),
            headline=r["headline"].strip(), url=r["url"],
            summary=(r.get("summary") or "").strip(), publisher=r.get("source") or "",
        ))
    return items


def fetch(ticker: str, since: datetime) -> list[NewsItem]:
    key = api_key()
    if not key:
        return []
    params = {"symbol": ticker, "from": since.date().isoformat(),
              "to": datetime.now(timezone.utc).date().isoformat(), "token": key}
    resp = http.get(URL, params, min_interval=1.1)  # stays under 60/min
    if resp is None:
        return []
    rows = resp.json()
    return parse(ticker, rows) if isinstance(rows, list) else []
