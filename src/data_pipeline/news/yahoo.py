"""Yahoo Finance headline RSS feed: no key, roughly the latest 20 headlines per ticker."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import timezone
from email.utils import parsedate_to_datetime

from src.data_pipeline.models import NewsItem
from src.data_pipeline.news import http

log = logging.getLogger(__name__)

FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline"


def parse_feed(ticker: str, xml_text: str) -> list[NewsItem]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("Yahoo RSS for %s didn't parse: %s", ticker, exc)
        return []
    items = []
    for node in root.iter("item"):
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        pub = node.findtext("pubDate")
        if not (title and link and pub):
            continue
        try:
            published = parsedate_to_datetime(pub).astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        items.append(NewsItem(
            ticker=ticker, source="yahoo", published_at=published, headline=title,
            url=link.split("?.tsrc=")[0],
            summary=(node.findtext("description") or "").strip(), publisher="Yahoo Finance",
        ))
    return items


def fetch(ticker: str) -> list[NewsItem]:
    resp = http.get(FEED, {"s": ticker, "region": "US", "lang": "en-US"},
                    min_interval=0.5, user_agent="Mozilla/5.0")
    return parse_feed(ticker, resp.text) if resp is not None else []
