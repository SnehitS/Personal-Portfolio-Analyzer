"""Option chains and earnings history from yfinance, turned into per-expiry views."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

from src.data_pipeline import market_data  # noqa: F401  (silences yfinance's pandas warnings)

from src.portfolio_analysis.options import ExpiryView, expiry_view

log = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # ETFs 404 on earnings calendars


def pick_expiries(expiries: list[date], today: date, max_days: int = 120, nearest: int = 6,
                  must_cover: list[date] | None = None) -> list[date]:
    """The nearest few expiries, plus the first one past 30 days (for 30-day IV) and the
    first one after each date in `must_cover` (e.g. earnings), skipping same-day expiries."""
    live = sorted(e for e in expiries if 0 < (e - today).days <= max_days)
    picked = set(live[:nearest])
    for target in [today + timedelta(days=30)] + list(must_cover or []):
        nxt = next((e for e in live if e >= target), None)
        if nxt:
            picked.add(nxt)
    return sorted(picked)


def spot_price(ticker: yf.Ticker) -> float | None:
    try:
        return float(ticker.fast_info["last_price"])
    except Exception:
        return None


def option_views(symbol: str, spot: float | None = None, must_cover: list[date] | None = None,
                 today: date | None = None) -> list[ExpiryView]:
    today = today or date.today()
    t = yf.Ticker(symbol)
    try:
        expiries = [date.fromisoformat(e) for e in t.options]
    except Exception as exc:
        log.warning("no option chain for %s: %s", symbol, exc)
        return []
    spot = spot or spot_price(t)
    if not spot or not expiries:
        return []
    views = []
    for exp in pick_expiries(expiries, today, must_cover=must_cover):
        try:
            chain = t.option_chain(exp.isoformat())
        except Exception as exc:
            log.warning("%s %s chain failed: %s", symbol, exp, exc)
            continue
        v = expiry_view(chain.calls, chain.puts, spot, exp, today)
        if v:
            views.append(v)
    return views


def past_earnings_dates(symbol: str, today: date | None = None, limit: int = 12) -> list[date]:
    today = today or date.today()
    try:
        df = yf.Ticker(symbol).get_earnings_dates(limit=limit)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    days = {ts.date() for ts in pd.to_datetime(df.index)}
    return sorted(d for d in days if d < today)


def next_earnings(symbol: str, today: date | None = None) -> date | None:
    today = today or date.today()
    try:
        dates = (yf.Ticker(symbol).calendar or {}).get("Earnings Date") or []
    except Exception:
        return None
    upcoming = sorted(d for d in dates if isinstance(d, (date, datetime)) and d >= today)
    return upcoming[0] if upcoming else None
