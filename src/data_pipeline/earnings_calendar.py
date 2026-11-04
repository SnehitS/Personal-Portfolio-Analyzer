"""Finnhub earnings calendar: report date, before/after market, EPS and revenue estimates."""

from __future__ import annotations

from datetime import date

from src.data_pipeline.news import finnhub, http

URL = "https://finnhub.io/api/v1/calendar/earnings"


def parse(rows: list[dict], tickers: set[str]) -> dict[str, dict]:
    """Earliest upcoming report per ticker."""
    out: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: r.get("date") or ""):
        sym = (r.get("symbol") or "").upper()
        if sym not in tickers or sym in out or not r.get("date"):
            continue
        out[sym] = {"date": date.fromisoformat(r["date"]), "hour": r.get("hour") or "",
                    "eps_estimate": r.get("epsEstimate"),
                    "revenue_estimate": r.get("revenueEstimate")}
    return out


def fetch(tickers: list[str], start: date, end: date) -> dict[str, dict]:
    key = finnhub.api_key()
    if not key or not tickers:
        return {}
    rows: list[dict] = []
    for sym in tickers:  # per symbol: the unfiltered calendar is capped at ~1500 rows
        resp = http.get(URL, {"from": start.isoformat(), "to": end.isoformat(),
                              "symbol": sym, "token": key}, min_interval=1.1)
        if resp is not None:
            rows += resp.json().get("earningsCalendar") or []
    return parse(rows, {t.upper() for t in tickers})
