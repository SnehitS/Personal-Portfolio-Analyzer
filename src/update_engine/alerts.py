"""Event alerts: things that happened or are about to, worth a look today.

Risk flags (overweight positions, idle cash) describe standing conditions and live in the
reports; alerts are for what's new: big moves, fresh filings, earnings tomorrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from src.data_pipeline.models import NewsItem, PortfolioSnapshot
from src.portfolio_analysis.catalysts import Catalyst
from src.portfolio_analysis.metrics import weights
from src.sentiment_pipeline.aggregate import TickerSentiment

LEVELS = {"high": 0, "watch": 1}

DEFAULTS = {
    "big_move_pct": 0.05,          # holding moved this much since the previous close
    "near_extreme": 0.02,          # within 2% of the 52-week range's top or bottom
    "loss_from_cost": -0.20,       # position down this much from average cost
    "earnings_days": 2,            # earnings within this many days
    "expiry_days": 2,              # held option expires within this many days
    "bearish_score": -0.40,        # ticker sentiment at or below this...
    "min_articles": 3,             # ...across at least this many articles
    "filing_hours": 24,            # 8-Ks filed this recently
    "high_weight": 0.05,           # moves and filings in positions this big are "high"
}


@dataclass
class Alert:
    level: str        # high or watch
    ticker: str       # "" for market-wide
    kind: str         # move, extreme, loss, earnings, expiry, sentiment, filing
    message: str


def check(snap: PortfolioSnapshot, cfg: dict | None = None, today: date | None = None,
          sentiment: list[TickerSentiment] | None = None,
          events: list[Catalyst] | None = None, filings: list[NewsItem] | None = None,
          now: datetime | None = None) -> list[Alert]:
    c = {**DEFAULTS, **(cfg or {})}
    today = today or date.today()
    now = now or datetime.now(timezone.utc)
    w = weights(snap)
    out: list[Alert] = []

    def level(ticker: str) -> str:
        return "high" if w.get(ticker, 0) >= c["high_weight"] else "watch"

    for h in snap.holdings:
        move = h.day_change_pct
        if move is not None and abs(move) >= c["big_move_pct"]:
            out.append(Alert(level(h.ticker), h.ticker, "move",
                             f"{h.ticker} {'up' if move > 0 else 'down'} {abs(move):.1%} today "
                             f"({w.get(h.ticker, 0):.0%} of the portfolio)"))
        pos = h.week52_position
        if pos is not None and pos >= 1 - c["near_extreme"]:
            out.append(Alert("watch", h.ticker, "extreme", f"{h.ticker} is at its 52-week high"))
        elif pos is not None and pos <= c["near_extreme"]:
            out.append(Alert("watch", h.ticker, "extreme", f"{h.ticker} is at its 52-week low"))
        pl = h.unrealized_pl_pct
        if pl is not None and pl <= c["loss_from_cost"]:
            out.append(Alert("watch", h.ticker, "loss",
                             f"{h.ticker} is {abs(pl):.0%} below your average cost"))

    for e in events or []:
        days = (e.day - today).days
        if e.kind == "earnings" and days <= c["earnings_days"]:
            when = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days"
            detail = f" ({e.detail})" if e.detail else ""
            out.append(Alert(level(e.ticker), e.ticker, "earnings",
                             f"{e.ticker} reports earnings {when}{detail}"))
        elif e.kind == "option expiry" and days <= c["expiry_days"]:
            out.append(Alert("high", e.ticker, "expiry",
                             f"{e.ticker} {e.detail or 'option'} expires {e.day:%a %b %d}"))

    for s in sentiment or []:
        if (s.score is not None and s.score <= c["bearish_score"]
                and s.articles >= c["min_articles"]):
            worst = f": {s.most_negative.item.headline}" if s.most_negative else ""
            out.append(Alert(level(s.ticker), s.ticker, "sentiment",
                             f"{s.ticker} news is bearish ({s.score:+.2f} over "
                             f"{s.articles} articles){worst}"))

    cutoff = now - timedelta(hours=c["filing_hours"])
    for f in filings or []:
        if f.kind == "filing" and f.form.startswith("8-K") and f.published_at >= cutoff:
            out.append(Alert(level(f.ticker), f.ticker, "filing", f"{f.ticker}: {f.headline}"))

    return sorted(out, key=lambda a: (LEVELS[a.level], -w.get(a.ticker, 0), a.ticker))


def render(alerts: list[Alert], today: date) -> str:
    lines = [f"# Alerts: {today:%a %b %d, %Y}", ""]
    if not alerts:
        return "\n".join(lines + ["Nothing needs attention today."]) + "\n"
    for lvl, heading in (("high", "Act on"), ("watch", "Watch")):
        mine = [a for a in alerts if a.level == lvl]
        if mine:
            lines += [f"## {heading}", "", *(f"- {a.message}" for a in mine), ""]
    return "\n".join(lines).rstrip() + "\n"
