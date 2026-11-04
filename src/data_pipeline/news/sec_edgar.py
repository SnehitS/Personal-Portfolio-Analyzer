"""SEC EDGAR filings (no key). SEC asks for a User-Agent naming you and an email:
set SEC_USER_AGENT="Your Name you@example.com" in .env. Limit is 10 requests/second."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from src.data_pipeline.models import NewsItem
from src.data_pipeline.news import http

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"
CIK_CACHE = Path("data/news_raw/sec_company_tickers.json")
CIK_CACHE_DAYS = 7

# Forms worth knowing about as a holder, with plain descriptions
FORMS = {
    "8-K": "current report",
    "10-Q": "quarterly report",
    "10-K": "annual report",
    "4": "insider transaction",
    "SC 13D": "activist stake (5%+)",
    "SC 13G": "passive stake (5%+)",
    "SCHEDULE 13D": "activist stake (5%+)",
    "SCHEDULE 13G": "passive stake (5%+)",
    "S-3": "shelf registration (possible share sale)",
    "424B5": "share offering prospectus",
    "DEF 14A": "proxy statement",
}

# 8-K item codes -> what happened
EIGHT_K_ITEMS = {
    "1.01": "material agreement", "1.02": "agreement terminated", "1.03": "bankruptcy",
    "1.05": "cybersecurity incident", "2.01": "acquisition or sale of assets",
    "2.02": "earnings results", "2.03": "new debt", "2.05": "restructuring costs",
    "2.06": "impairment", "3.01": "delisting notice", "3.02": "unregistered share sale",
    "4.01": "auditor change", "4.02": "prior financials no longer reliable",
    "5.01": "change in control", "5.02": "executive or director change",
    "5.03": "bylaws change", "5.07": "shareholder vote results", "7.01": "Reg FD disclosure",
    "8.01": "other event",
}


def user_agent() -> str:
    ua = os.getenv("SEC_USER_AGENT")
    if not ua:
        log.warning("SEC_USER_AGENT isn't set; SEC may reject requests without a contact email")
    return ua or http.DEFAULT_UA


def cik_map() -> dict[str, int]:
    """Ticker -> CIK, cached on disk because the file is a few MB and rarely changes."""
    fresh = CIK_CACHE.exists() and time.time() - CIK_CACHE.stat().st_mtime < CIK_CACHE_DAYS * 86400
    if not fresh:
        resp = http.get(TICKERS_URL, min_interval=0.15, user_agent=user_agent())
        if resp is not None:
            CIK_CACHE.parent.mkdir(parents=True, exist_ok=True)
            CIK_CACHE.write_text(resp.text, encoding="utf-8")
    if not CIK_CACHE.exists():
        return {}
    rows = json.loads(CIK_CACHE.read_text(encoding="utf-8")).values()
    return {r["ticker"].upper(): int(r["cik_str"]) for r in rows}


def describe(form: str, items: str) -> str:
    what = FORMS.get(form, form)
    if form == "8-K" and items:
        codes = [EIGHT_K_ITEMS[c] for c in items.split(",") if c in EIGHT_K_ITEMS]
        codes = [c for c in codes if c != "other event"] or codes
        if codes:
            what = ", ".join(codes)
    return what


def parse_submissions(ticker: str, cik: int, payload: dict, since: datetime) -> list[NewsItem]:
    recent = payload.get("filings", {}).get("recent", {})
    name = payload.get("name") or ticker
    out = []
    for i, form in enumerate(recent.get("form", [])):
        if form not in FORMS:
            continue
        accepted = recent["acceptanceDateTime"][i] or recent["filingDate"][i]
        try:
            when = datetime.fromisoformat(accepted.replace("Z", "+00:00"))
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when < since:
            break  # newest first
        items = recent.get("items", [""] * (i + 1))[i]
        url = ARCHIVE_URL.format(cik=cik, accession=recent["accessionNumber"][i].replace("-", ""),
                                 doc=recent["primaryDocument"][i])
        what = describe(form, items)
        out.append(NewsItem(
            ticker=ticker, source="sec", published_at=when.astimezone(timezone.utc),
            headline=f"{name} filed {form}: {what}", url=url, summary=what,
            publisher="SEC EDGAR", kind="filing", form=form,
        ))
    return out


class Fetcher:
    def __init__(self):
        self._ciks: dict[str, int] | None = None

    def __call__(self, ticker: str, since: datetime) -> list[NewsItem]:
        if self._ciks is None:
            self._ciks = cik_map()
        cik = self._ciks.get(ticker.upper().replace(".", "-"))
        if cik is None:
            return []  # ETFs and foreign listings often have no EDGAR filer
        resp = http.get(SUBMISSIONS_URL.format(cik=cik), min_interval=0.15,
                        user_agent=user_agent())
        return parse_submissions(ticker, cik, resp.json(), since) if resp is not None else []
