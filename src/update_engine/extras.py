"""Alpha Vantage extras: refresh insider trades, macro series and earnings calls, and the
report sections that show them.

Everything is stored, and each fetch is logged so a second run on the same day spends no
calls. With the free key's 25 calls a day: one call per holding for insiders, three for
macro, and a few per earnings season for transcripts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.data_pipeline import alpha_vantage as av
from src.data_pipeline import storage
from src.data_pipeline.models import PortfolioSnapshot
from src.data_pipeline.news.alphavantage import RateLimited
from src.portfolio_analysis import insiders, macro

log = logging.getLogger(__name__)

DEFAULTS = {
    "max_calls": 20,          # per run; the free key allows 25 a day
    "refresh_hours": 20,      # don't refetch anything newer than this
    "call_lookback_days": 7,  # look for transcripts of reports this recent
}


@dataclass
class RefreshResult:
    insider_trades_new: int = 0
    macro_updated: list[str] = field(default_factory=list)
    calls_new: list[tuple[str, str]] = field(default_factory=list)  # (ticker, quarter)
    calls_used: int = 0
    stopped: str = ""


def _fresh(key: str, now: datetime, hours: float, db: Path) -> bool:
    last = storage.last_fetched(key, db)
    return last is not None and now - last < timedelta(hours=hours)


def stock_tickers(snap: PortfolioSnapshot) -> list[str]:
    return [h.ticker for h in sorted(snap.holdings, key=lambda h: -h.market_value)
            if h.sector != "ETF"]


def recent_reports(tickers: list[str], today: date, days: int) -> dict[str, dict]:
    """{ticker: {"date", "hour"}} for holdings that reported in the last `days` days."""
    from src.data_pipeline import earnings_calendar
    found = earnings_calendar.fetch(tickers, today - timedelta(days=days), today)
    return {t: r for t, r in found.items() if r["date"] <= today}


def refresh(snap: PortfolioSnapshot, cfg: dict, db: Path, today: date | None = None,
            now: datetime | None = None, reports: dict[str, dict] | None = None,
            summarize=None) -> RefreshResult:
    c = {**DEFAULTS, **(cfg.get("alphavantage") or {})}
    today = today or date.today()
    now = now or datetime.now(timezone.utc)
    client = av.Client(c["max_calls"])
    result = RefreshResult()
    if not av.api_key():
        result.stopped = "ALPHAVANTAGE_API_KEY isn't set"
        return result
    tickers = stock_tickers(snap)
    hours = c["refresh_hours"]
    try:
        # Earnings calls first: they're rare, and the most useful on the day they land
        if reports is None:
            reports = recent_reports(tickers, today, c["call_lookback_days"])
        stored = {(r["ticker"], r["reported"]) for r in storage.load_earnings_calls(tickers, db_path=db)}
        for ticker, rep in reports.items():
            reported = rep["date"]
            key = f"call:{ticker}:{reported}"
            if (ticker, reported.isoformat()) in stored or _fresh(key, now, hours, db):
                continue
            if reported == today and rep.get("hour") == "amc":
                continue  # reports after the close; the transcript comes tomorrow at the earliest
            for quarter in av.quarter_candidates(reported):
                entries = av.transcript(client, ticker, quarter)
                if entries:
                    summary = (summarize or _summarize)(ticker, quarter, entries, cfg)
                    model = (cfg.get("earnings_calls") or {}).get("model") if summary else None
                    storage.save_earnings_call(ticker, quarter, reported, json.dumps(entries),
                                               summary, model, now, db)
                    result.calls_new.append((ticker, quarter))
                    break
            storage.mark_fetched(key, now, db)

        for ticker in tickers:
            key = f"insiders:{ticker}"
            if _fresh(key, now, hours, db):
                continue
            trades = av.insiders(client, ticker)
            if trades is None:
                continue
            result.insider_trades_new += storage.save_insider_trades(trades, now, db)
            storage.mark_fetched(key, now, db)

        for name in av.MACRO_SERIES:
            key = f"macro:{name}"
            if _fresh(key, now, hours, db):
                continue
            points = av.macro_series(client, name)
            if points is None:
                continue
            storage.save_macro(name, [p for p in points if p[0] >= today - timedelta(days=800)], db)
            storage.mark_fetched(key, now, db)
            result.macro_updated.append(name)
    except RateLimited as exc:
        result.stopped = str(exc)
    result.calls_used = c["max_calls"] - client.calls_left
    return result


def _summarize(ticker: str, quarter: str, entries: list[dict], cfg: dict) -> str | None:
    from src.sentiment_pipeline import earnings_call
    model = (cfg.get("earnings_calls") or {}).get("model", earnings_call.DEFAULT_MODEL)
    return earnings_call.summarize(ticker, quarter, entries, model)


# --- report sections ---------------------------------------------------------------------

def new_insider_trades(tickers: list[str], cfg: dict, db: Path, now: datetime,
                       days: int = 14) -> list:
    """Notable trades first seen in the last day, for alerts."""
    trades = storage.load_insider_trades(tickers, (now - timedelta(days=days)).date(),
                                         now - timedelta(hours=24), db)
    return insiders.notable(trades, cfg.get("insiders"))


def insider_section(tickers: list[str], cfg: dict, db: Path, today: date, days: int) -> list[str]:
    trades = storage.load_insider_trades(tickers, today - timedelta(days=days), db_path=db)
    picked = insiders.notable(trades, cfg.get("insiders"))[:10]
    if not picked:
        return []
    return [f"## Insider trades, last {days} days", "",
            *(f"- {insiders.describe(t)}" for t in picked)]


def macro_section(db: Path, today: date) -> list[str]:
    since = today - timedelta(days=800)
    lines = macro.backdrop({name: storage.load_macro(name, since, db)
                            for name in av.MACRO_SERIES})
    return ["## Macro backdrop", "", *lines] if lines else []


def calls_section(tickers: list[str], db: Path, now: datetime, days: int) -> list[str]:
    rows = storage.load_earnings_calls(tickers, now - timedelta(days=days), db)
    out = []
    for r in rows:
        body = r["summary"] or "_Transcript saved; set ANTHROPIC_API_KEY to get a summary._"
        out += ["", f"**{r['ticker']} {r['quarter']}** (reported {r['reported']})", "", body]
    return ["## Earnings calls", *out] if out else []


def first_bullet(summary: str | None) -> str:
    for line in (summary or "").splitlines():
        if line.startswith("- "):
            return line[2:]
    return "summary in today's report"
