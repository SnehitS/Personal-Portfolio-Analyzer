"""Kalshi prediction market prices (public market data, no account or key needed)."""

from __future__ import annotations

import logging
from datetime import datetime

from src.data_pipeline.news.http import get
from src.portfolio_analysis.odds import OddsEvent, Quote, quote_prob

log = logging.getLogger(__name__)

API = "https://api.elections.kalshi.com/trade-api/v2"


def _when(text: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")) if text else None
    except ValueError:
        return None


def parse_event(raw: dict, series: str) -> OddsEvent | None:
    """One Kalshi event and its markets as an OddsEvent; None when nothing is trading."""
    markets = [m for m in raw.get("markets") or [] if m.get("status") in (None, "active", "open")]
    if not markets:
        return None
    ladder = all(m.get("strike_type") == "greater" and m.get("floor_strike") is not None
                 for m in markets)
    quotes = []
    for m in markets:
        prob = quote_prob(m.get("yes_bid_dollars"), m.get("yes_ask_dollars"),
                          m.get("last_price_dollars"))
        if prob is None:
            continue
        strike = float(m["floor_strike"]) if ladder else None
        quotes.append(Quote(m.get("yes_sub_title") or m.get("ticker", ""), prob, strike))
    if not quotes:
        return None
    closes = min((c for c in (_when(m.get("close_time")) for m in markets) if c), default=None)
    return OddsEvent(series, raw["event_ticker"], raw.get("title") or raw["event_ticker"],
                     closes, "ladder" if ladder else "choice", quotes)


def fetch_series(series: str, events: int = 1) -> list[OddsEvent]:
    """The next `events` open events in a series, soonest first."""
    resp = get(f"{API}/events", {"series_ticker": series, "status": "open",
                                 "with_nested_markets": "true", "limit": 20}, min_interval=0.2)
    if resp is None:
        return []
    try:
        raw = resp.json().get("events") or []
    except ValueError:
        log.warning("Kalshi sent something other than JSON for %s", series)
        return []
    parsed = [e for e in (parse_event(r, series) for r in raw) if e]
    parsed.sort(key=lambda e: (e.closes is None, e.closes))
    return parsed[:events]
