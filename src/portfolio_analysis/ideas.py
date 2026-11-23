"""Add / hold / trim / sell ideas for the stocks you already hold.

Rules, not predictions: each signal adds or subtracts points, and the total picks the
action. Every point comes with the reason behind it, so you can disagree with a rule
instead of trusting a number. Position size overrides the score: an overweight holding
always gets a trim, and nothing near the limit gets an add.

What you paid is deliberately not a signal. A stock doesn't know your cost basis; it only
matters for taxes, which show up as notes on when to sell, not whether to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from src.data_pipeline.models import Holding, InsiderTrade, PortfolioSnapshot
from src.portfolio_analysis.metrics import Lot, fifo_lots, lot_mismatches, weights
from src.sentiment_pipeline.aggregate import TickerSentiment

ACTIONS = ("add", "hold", "trim", "sell")

DEFAULTS = {
    "add_score": 2.0,            # total at or above this: add
    "trim_score": -2.0,          # at or below this: trim
    "sell_score": -3.0,          # at or below this: sell (every bearish signal agrees)
    "max_weight": 0.25,          # above this, always trim back to it
    "add_room": 0.8,             # no adds once a position is 80% of max_weight
    "analyst_upside": 0.20,      # mean target this far above the price: +1
    "analyst_downside": -0.05,   # this far below: -1
    "min_analysts": 3,           # ignore targets from fewer analysts than this
    "rsi_low": 30,
    "rsi_high": 70,
    "min_articles": 3,           # news sentiment needs this many articles to count
    "insider_buy_min": 25_000,   # open-market insider buys this big count as +1
    "earnings_days": 7,          # earnings this close: note it, the move can swamp the idea
    "long_term_days": 45,        # gains lots turning long-term this soon: note it on trims
}


@dataclass
class Signal:
    points: float
    reason: str


@dataclass
class Idea:
    ticker: str
    action: str                  # add, hold, trim, sell
    score: float
    weight: float
    signals: list[Signal] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    trim_to: float | None = None  # target weight when trimming for size

    @property
    def reasons(self) -> list[str]:
        return [f"{s.points:+g} {s.reason}" for s in self.signals]


def _signals(h: Holding, c: dict, sentiment: TickerSentiment | None,
             insider_buys: list[InsiderTrade]) -> list[Signal]:
    out: list[Signal] = []

    if h.sma_50 and h.sma_200:
        if h.price > h.sma_50 > h.sma_200:
            out.append(Signal(1, "uptrend (above its 50-day, which is above its 200-day)"))
        elif h.price < h.sma_50 < h.sma_200:
            out.append(Signal(-1, "downtrend (below its 50-day, which is below its 200-day)"))

    if h.rsi_14 is not None:
        if h.rsi_14 <= c["rsi_low"]:
            out.append(Signal(0.5, f"RSI {h.rsi_14:.0f}, oversold"))
        elif h.rsi_14 >= c["rsi_high"]:
            out.append(Signal(-0.5, f"RSI {h.rsi_14:.0f}, overbought"))

    upside = h.analyst_upside
    if upside is not None and (h.analyst_count or 0) >= c["min_analysts"]:
        who = f"{h.analyst_count} analysts' mean target"
        if upside >= c["analyst_upside"]:
            out.append(Signal(1, f"{who} is {upside:.0%} above the price"))
        elif upside <= c["analyst_downside"]:
            out.append(Signal(-1, f"{who} is {-upside:.0%} below the price"))

    if sentiment and sentiment.score is not None and sentiment.articles >= c["min_articles"]:
        if sentiment.label == "bullish":
            out.append(Signal(0.5, f"news is bullish ({sentiment.score:+.2f} over "
                                   f"{sentiment.articles} articles)"))
        elif sentiment.label == "bearish":
            out.append(Signal(-0.5, f"news is bearish ({sentiment.score:+.2f} over "
                                    f"{sentiment.articles} articles)"))

    buys = [t for t in insider_buys if t.ticker == h.ticker and (t.value or 0) >= c["insider_buy_min"]]
    if buys:
        total = sum(t.value or 0 for t in buys)
        who = "an insider" if len({t.executive for t in buys}) == 1 else "insiders"
        out.append(Signal(1, f"{who} bought ${total:,.0f} on the open market"))

    return out


def _tax_notes(h: Holding, lots: list[Lot], today: date, c: dict) -> list[str]:
    notes = []
    soon = [l for l in lots if not l.is_long_term(today)
            and l.long_term_date - today <= timedelta(days=c["long_term_days"])
            and h.price > l.price]
    if soon:
        first = min(l.long_term_date for l in soon)
        shares = sum(l.quantity for l in soon)
        notes.append(f"{shares:g} shares with a gain turn long-term on {first:%b %d}; "
                     f"selling those before then is taxed as short-term")
    loss = h.unrealized_pl
    if loss < 0:
        notes.append(f"selling realizes a ${-loss:,.0f} loss you can deduct; buying it back "
                     f"within 30 days voids that (wash sale)")
    return notes


def suggest(snap: PortfolioSnapshot, cfg: dict | None = None, today: date | None = None,
            sentiment: list[TickerSentiment] | None = None,
            insider_buys: list[InsiderTrade] | None = None) -> list[Idea]:
    """One idea per stock holding (ETFs are skipped), strongest action first.

    insider_buys: recent open-market purchases (code P) in your holdings."""
    c = {**DEFAULTS, **(cfg or {})}
    today = today or date.today()
    w = weights(snap)
    by_ticker = {s.ticker: s for s in sentiment or []}
    buys = [t for t in insider_buys or [] if t.code == "P"]
    lots = fifo_lots(snap.transactions)
    unreliable = {t for t, _, _ in lot_mismatches(snap.holdings, lots)} if snap.transactions else set()

    ideas = []
    for h in snap.holdings:
        if h.sector == "ETF":
            continue
        weight = w.get(h.ticker, 0.0)
        signals = _signals(h, c, by_ticker.get(h.ticker), buys)
        score = sum(s.points for s in signals)

        if score <= c["sell_score"]:
            action = "sell"
        elif score <= c["trim_score"]:
            action = "trim"
        elif score >= c["add_score"]:
            action = "add"
        else:
            action = "hold"

        idea = Idea(h.ticker, action, score, weight, signals)
        if weight > c["max_weight"]:
            idea.signals.append(Signal(0, f"{weight:.0%} of the portfolio, over the "
                                          f"{c['max_weight']:.0%} limit"))
            idea.trim_to = c["max_weight"]
            if action in ("add", "hold"):
                idea.action = "trim"
        elif action == "add" and weight >= c["max_weight"] * c["add_room"]:
            idea.action = "hold"
            idea.notes.append(f"signals say add, but it's already {weight:.0%} of the "
                              f"portfolio (limit {c['max_weight']:.0%})")

        if h.next_earnings and 0 <= (h.next_earnings - today).days <= c["earnings_days"]:
            idea.notes.append(f"reports earnings {h.next_earnings:%a %b %d}; the move can "
                              f"outweigh everything above")
        if idea.action in ("trim", "sell"):
            if h.ticker in unreliable:
                idea.notes.append("trade history doesn't cover every share, so tax notes "
                                  "are left out")
            else:
                idea.notes += _tax_notes(h, lots.get(h.ticker, []), today, c)
        ideas.append(idea)

    order = {"sell": 0, "trim": 1, "add": 2, "hold": 3}
    return sorted(ideas, key=lambda i: (order[i.action], -abs(i.score), -i.weight))


def trim_amount(idea: Idea, snap: PortfolioSnapshot) -> float | None:
    """Dollars to sell to get an overweight position back to its target weight."""
    if idea.trim_to is None:
        return None
    return max(0.0, (idea.weight - idea.trim_to) * snap.total_equity)


def render(ideas: list[Idea], snap: PortfolioSnapshot, today: date) -> str:
    lines = [f"# Ideas: {today:%a %b %d, %Y}", "",
             "_Rules-based suggestions from stored data, not financial advice. "
             "Each reason shows the points it added._", ""]
    if not ideas:
        return "\n".join(lines + ["No stock holdings to look at."]) + "\n"
    for action, heading in (("sell", "Sell"), ("trim", "Trim"), ("add", "Add"), ("hold", "Hold")):
        mine = [i for i in ideas if i.action == action]
        if not mine:
            continue
        lines += [f"## {heading}", ""]
        if action == "hold":
            lines += [f"- **{i.ticker}** ({i.weight:.0%}, score {i.score:+g})"
                      + (f": {'; '.join(s.reason for s in i.signals)}" if i.signals else "")
                      for i in mine]
            lines.append("")
            continue
        for i in mine:
            head = f"**{i.ticker}** ({i.weight:.0%} of the portfolio, score {i.score:+g})"
            amount = trim_amount(i, snap)
            if amount:
                head += f": sell about ${amount:,.0f} to get back to {i.trim_to:.0%}"
            lines += [f"- {head}",
                      *(f"  - {r}" for r in i.reasons),
                      *(f"  - _Note: {n}_" for n in i.notes)]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
