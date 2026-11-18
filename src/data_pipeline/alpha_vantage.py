"""Alpha Vantage endpoints that work on the free key (25 calls/day): insider trades,
earnings call transcripts, and macro series (Fed funds, CPI, 10-year Treasury).

News sentiment lives in news/alphavantage.py; on free keys it now needs a paid plan.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from src.data_pipeline.models import InsiderTrade
from src.data_pipeline.news import http
from src.data_pipeline.news.alphavantage import URL, RateLimited, api_key

log = logging.getLogger(__name__)

# name -> (function, extra params). Values come back as strings, newest first.
MACRO_SERIES = {
    "fed_funds": ("FEDERAL_FUNDS_RATE", {"interval": "monthly"}),
    "cpi": ("CPI", {"interval": "monthly"}),
    "treasury_10y": ("TREASURY_YIELD", {"interval": "daily", "maturity": "10year"}),
}


class Client:
    """Counts calls so one run can't spend the whole day's quota, and stops for the rest
    of the run once Alpha Vantage says the quota is used up."""

    def __init__(self, max_calls: int = 20):
        self.calls_left = max_calls
        self.stopped = ""

    @property
    def usable(self) -> bool:
        return bool(api_key()) and self.calls_left > 0 and not self.stopped

    def query(self, function: str, **params) -> dict | None:
        if not self.usable:
            return None
        self.calls_left -= 1
        resp = http.get(URL, {"function": function, **params, "apikey": api_key()},
                        min_interval=1.2, timeout=30)
        if resp is None:
            return None
        try:
            payload = resp.json()
        except ValueError:
            log.warning("Alpha Vantage %s returned something other than JSON", function)
            return None
        note = payload.get("Information") or payload.get("Note") or payload.get("Error Message")
        if note and len(payload) == 1:
            if any(w in note.lower() for w in ("rate limit", "per day", "premium")):
                self.stopped = note
                log.warning("Alpha Vantage stopped for this run: %s", note)
                raise RateLimited(note)
            log.warning("Alpha Vantage %s: %s", function, note)
            return None
        return payload


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_insiders(ticker: str, payload: dict, since: date) -> list[InsiderTrade]:
    out = []
    for r in payload.get("data") or []:
        try:
            day = date.fromisoformat(r["transaction_date"])
        except (KeyError, ValueError):
            continue
        shares = _float(r.get("shares"))
        if day < since or not shares:
            continue
        out.append(InsiderTrade(
            ticker=ticker, day=day, executive=(r.get("executive") or "").strip(),
            title=(r.get("executive_title") or "").strip(),
            code=(r.get("acquisition_or_disposal") or "").strip().upper(),
            shares=shares, price=_float(r.get("share_price")),
            security=(r.get("security_type") or "").strip()))
    return out


def insiders(client: Client, ticker: str, days: int = 365) -> list[InsiderTrade] | None:
    """The last `days` of a company's insider transactions, or None if the call failed."""
    payload = client.query("INSIDER_TRANSACTIONS", symbol=ticker)
    if payload is None:
        return None
    return parse_insiders(ticker, payload, date.today() - timedelta(days=days))


def parse_series(payload: dict) -> list[tuple[date, float]]:
    out = []
    for r in payload.get("data") or []:
        value = _float(r.get("value"))
        try:
            out.append((date.fromisoformat(r["date"]), value))
        except (KeyError, ValueError):
            continue
    return [(d, v) for d, v in out if v is not None]


def macro_series(client: Client, name: str) -> list[tuple[date, float]] | None:
    function, params = MACRO_SERIES[name]
    payload = client.query(function, **params)
    return None if payload is None else parse_series(payload)


def transcript(client: Client, ticker: str, quarter: str) -> list[dict] | None:
    """[{speaker, title, content}, ...]; [] when Alpha Vantage has none for that quarter."""
    payload = client.query("EARNINGS_CALL_TRANSCRIPT", symbol=ticker, quarter=quarter)
    if payload is None:
        return None
    return [{"speaker": t.get("speaker", ""), "title": t.get("title", ""),
             "content": t.get("content", "")}
            for t in payload.get("transcript") or [] if t.get("content")]


def quarter_label(day: date) -> str:
    return f"{day.year}Q{(day.month - 1) // 3 + 1}"


def quarter_candidates(reported: date) -> list[str]:
    """Alpha Vantage labels transcripts by fiscal quarter. Companies report within about 50
    days of a quarter's end, so try the calendar quarter 50 days back first, then the next
    one (companies whose fiscal year runs a little ahead, like Broadcom's)."""
    ended = reported - timedelta(days=50)
    first = quarter_label(ended)
    year, q = ended.year, (ended.month - 1) // 3 + 1
    nxt = f"{year + (q == 4)}Q{q % 4 + 1}"
    return [first, nxt]


def recent_quarters(today: date) -> list[str]:
    """This quarter's label, then last quarter's: the newest transcripts that can exist."""
    current = quarter_label(today)
    year, q = today.year, (today.month - 1) // 3 + 1
    return [current, f"{year - (q == 1)}Q{(q - 2) % 4 + 1}"]
