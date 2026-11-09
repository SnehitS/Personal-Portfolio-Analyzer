"""Daily report: how the portfolio moved, what traded, what the news says, what's coming."""

from __future__ import annotations

from datetime import date

from src.data_pipeline.models import PortfolioSnapshot
from src.portfolio_analysis.catalysts import Catalyst
from src.portfolio_analysis.metrics import portfolio_day_change, weights
from src.sentiment_pipeline.aggregate import TickerSentiment
from src.update_engine.changes import PortfolioChange

TRADE_VERBS = {"new": "Opened", "added": "Added to", "trimmed": "Trimmed", "closed": "Closed"}


def _pct(x: float | None, fmt: str = "+.2%") -> str:
    return "-" if x is None else format(x, fmt)


def _money(x: float, signed: bool = False) -> str:
    sign = ("+" if x >= 0 else "-") if signed else ("-" if x < 0 else "")
    return f"{sign}${abs(x):,.0f}"


def _table(headers: list[str], rows: list[list[str]], right: set[int] = frozenset()) -> list[str]:
    align = ["---:" if i in right else "---" for i in range(len(headers))]
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |",
            *("| " + " | ".join(r) + " |" for r in rows)]


def _summary(snap: PortfolioSnapshot, change: PortfolioChange | None) -> list[str]:
    cash_pct = snap.account.cash / snap.total_equity if snap.total_equity else 0.0
    lines = [f"- **Equity:** {_money(snap.total_equity)} "
             f"(cash {_money(snap.account.cash)}, {cash_pct:.0%})"]
    day = portfolio_day_change(snap)
    if day is not None:
        lines.append(f"- **Today:** {_pct(day)} on stock holdings")
    if change:
        line = (f"- **Since last snapshot:** {_money(change.market_gain, signed=True)} "
                f"({_pct(change.market_gain_pct)}) from the market")
        if abs(change.net_deposits) >= 1:
            line += f", {_money(change.net_deposits, signed=True)} net deposits"
        lines.append(line)
    return lines


def _movers(snap: PortfolioSnapshot, change: PortfolioChange | None, n: int) -> list[str]:
    w = weights(snap)
    if change:
        rows = [[p.ticker, _pct(p.price_change_pct), _money(p.market_gain, signed=True),
                 f"{p.weight_after:.1%}"] for p in change.movers(n)]
        headers = ["Ticker", "Price", "P/L", "Weight"]
    else:  # first snapshot: fall back to today's move against the previous close
        moved = [h for h in snap.holdings if h.day_change_pct is not None]
        moved.sort(key=lambda h: -abs(h.quantity * (h.price - h.previous_close)))
        rows = [[h.ticker, _pct(h.day_change_pct),
                 _money(h.quantity * (h.price - h.previous_close), signed=True),
                 f"{w.get(h.ticker, 0):.1%}"] for h in moved[:n]]
        headers = ["Ticker", "Today", "P/L", "Weight"]
    if not rows:
        return []
    return ["## Biggest movers", "", *_table(headers, rows, right={1, 2, 3})]


def _trades(change: PortfolioChange | None) -> list[str]:
    if not change:
        return []
    traded = change.by_status(*TRADE_VERBS)
    if not traded:
        return []
    lines = ["## Position changes", ""]
    for p in traded:
        delta = p.shares_after - p.shares_before
        lines.append(f"- {TRADE_VERBS[p.status]} **{p.ticker}**: {delta:+g} shares "
                     f"(weight {p.weight_before:.1%} -> {p.weight_after:.1%})")
    return lines


def _news(sentiment: list[TickerSentiment]) -> list[str]:
    scored = [s for s in sentiment if s.score is not None]
    if not scored:
        return []
    scored.sort(key=lambda s: -abs(s.score))
    lines = ["## News sentiment", "",
             *_table(["Ticker", "Read", "Score", "Articles"],
                     [[s.ticker, s.label, f"{s.score:+.2f}", str(s.articles)] for s in scored],
                     right={2, 3})]
    notable = [(s, s.most_positive if s.score > 0 else s.most_negative)
               for s in scored if s.label != "mixed"]
    notable = [(s, r) for s, r in notable if r]
    if notable:
        lines += ["", "Headlines driving it:", ""]
        lines += [f"- **{s.ticker}** ({r.score:+.2f}): {r.item.headline}" for s, r in notable]
    return lines


def _catalysts(events: list[Catalyst], today: date) -> list[str]:
    if not events:
        return []
    rows = [[f"{c.day:%a %b %d}", f"{(c.day - today).days}d", c.ticker or "market", c.kind,
             c.detail] for c in events]
    return ["## Coming up", "", *_table(["Date", "In", "Ticker", "Event", "Detail"], rows,
                                         right={1})]


def render(snap: PortfolioSnapshot, today: date, change: PortfolioChange | None = None,
           sentiment: list[TickerSentiment] | None = None,
           events: list[Catalyst] | None = None, flags: list[str] | None = None,
           movers: int = 5) -> str:
    """The daily report as Markdown. Sections with nothing to say are left out."""
    sections = [
        [f"# Daily report: {today:%a %b %d, %Y}", "",
         f"_Snapshot {snap.taken_at:%Y-%m-%d %H:%M} ({snap.source})_", "",
         *_summary(snap, change)],
        _movers(snap, change, movers),
        _trades(change),
        _news(sentiment or []),
        _catalysts(events or [], today),
        ["## Risk flags", "", *(f"- {f}" for f in flags)] if flags else [],
    ]
    return "\n\n".join("\n".join(s) for s in sections if s) + "\n"
