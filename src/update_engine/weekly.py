"""Weekly report: the daily layout, compared against the end of last week."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from src.data_pipeline.models import PortfolioSnapshot
from src.portfolio_analysis.catalysts import Catalyst
from src.sentiment_pipeline.aggregate import TickerSentiment
from src.update_engine import daily
from src.update_engine.changes import PortfolioChange


def week_start(day: date) -> datetime:
    """Midnight on the Monday of `day`'s week. The weekly report compares against the last
    snapshot before this, i.e. the end of the previous week."""
    monday = day - timedelta(days=day.weekday())
    return datetime.combine(monday, datetime.min.time())


def render(snap: PortfolioSnapshot, today: date, change: PortfolioChange | None = None,
           sentiment: list[TickerSentiment] | None = None,
           events: list[Catalyst] | None = None, flags: list[str] | None = None,
           movers: int = 10, news_days: int = 7) -> str:
    monday = week_start(today).date()
    return daily.render(
        snap, today, change, sentiment, events, flags, movers,
        title=f"# Weekly report: week of {monday:%b %d, %Y}",
        change_label="This week", since="this week", news_window=f"last {news_days} days")
