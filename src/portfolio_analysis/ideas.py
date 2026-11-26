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

ACTIONS = ("add", "hold", "trim", "sell")     # holdings
WATCH_ACTIONS = ("buy", "short", "watch")      # watchlist tickers you don't own

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
    "cash_reserve": 0.05,        # keep this share of equity in cash when deploying the rest
    "min_buy": 100,              # don't suggest buys smaller than this ($)
    "buy_score": 2.0,            # watchlist: points needed to open a new long...
    "short_score": -2.0,         # ...or to call it a short
    "start_weight": 0.03,        # a new long starts at this share of equity, at most
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
    cap: float | None = None      # most this position should reach from new buys

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


def _earnings_note(h: Holding, today: date, c: dict) -> list[str]:
    if h.next_earnings and 0 <= (h.next_earnings - today).days <= c["earnings_days"]:
        return [f"reports earnings {h.next_earnings:%a %b %d}; the move can outweigh "
                f"everything above"]
    return []


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

        idea.notes += _earnings_note(h, today, c)
        if idea.action in ("trim", "sell"):
            if h.ticker in unreliable:
                idea.notes.append("trade history doesn't cover every share, so tax notes "
                                  "are left out")
            else:
                idea.notes += _tax_notes(h, lots.get(h.ticker, []), today, c)
        ideas.append(idea)

    order = {"sell": 0, "trim": 1, "add": 2, "hold": 3}
    return sorted(ideas, key=lambda i: (order[i.action], -abs(i.score), -i.weight))


def watchlist(candidates: list[Holding], snap: PortfolioSnapshot, cfg: dict | None = None,
              today: date | None = None, sentiment: list[TickerSentiment] | None = None,
              insider_buys: list[InsiderTrade] | None = None) -> list[Idea]:
    """Buy, short or watch for tickers you don't own, scored like holdings.

    Tickers you already hold and ETFs are skipped. A buy is sized as a starter position
    (start_weight), never past the add ceiling."""
    c = {**DEFAULTS, **(cfg or {})}
    today = today or date.today()
    held = {h.ticker for h in snap.holdings}
    by_ticker = {s.ticker: s for s in sentiment or []}
    buys = [t for t in insider_buys or [] if t.code == "P"]
    out = []
    for h in candidates:
        if h.ticker in held or h.sector == "ETF" or not h.price:
            continue
        signals = _signals(h, c, by_ticker.get(h.ticker), buys)
        score = sum(s.points for s in signals)
        action = ("buy" if score >= c["buy_score"] else
                  "short" if score <= c["short_score"] else "watch")
        idea = Idea(h.ticker, action, score, 0.0, signals,
                    cap=min(c["start_weight"], c["max_weight"] * c["add_room"]))
        idea.notes += _earnings_note(h, today, c)
        if action == "short":
            idea.notes.append("Robinhood doesn't let you short stock; a put is the way to act "
                              "on this, and it can expire worthless")
        out.append(idea)
    order = {"buy": 0, "short": 1, "watch": 2}
    return sorted(out, key=lambda i: (order[i.action], -abs(i.score)))


def headline(ideas: list[Idea], plan: CashPlan | None = None) -> str:
    """One line for notifications: the non-hold ideas and any cash buys."""
    names = {"sell": "sell", "trim": "trim", "add": "add", "buy": "new", "short": "short"}
    parts = [f"{names[a]} {', '.join(i.ticker for i in ideas if i.action == a)}"
             for a in names if any(i.action == a for i in ideas)]
    if plan and plan.buys:
        parts.append("buy " + ", ".join(f"${v:,.0f} {t}" for t, v in plan.buys.items()))
    return "Ideas: " + "; ".join(parts) if parts else ""


def trim_amount(idea: Idea, snap: PortfolioSnapshot) -> float | None:
    """Dollars to sell to get an overweight position back to its target weight."""
    if idea.trim_to is None:
        return None
    return max(0.0, (idea.weight - idea.trim_to) * snap.total_equity)


@dataclass
class CashPlan:
    investable: float                 # cash above the reserve
    reserve: float                    # cash kept back
    buys: dict[str, float] = field(default_factory=dict)   # ticker -> dollars, biggest first
    capped: bool = False              # some cash had nowhere to go: every add hit its limit

    @property
    def leftover(self) -> float:
        return self.investable - sum(self.buys.values())


def deploy_cash(ideas: list[Idea], snap: PortfolioSnapshot, cfg: dict | None = None) -> CashPlan:
    """Split cash above the reserve across the add ideas, in proportion to their scores.

    Watchlist buys take part too. No buy takes a position past the add ceiling
    (add_room x max_weight), or a new one past its starter size, and whatever a capped
    position can't take goes to the others. Cash nothing can use is left over: no add
    ideas means no buys, not a forced pick."""
    c = {**DEFAULTS, **(cfg or {})}
    equity = snap.total_equity
    reserve = min(snap.account.cash, c["cash_reserve"] * equity)
    plan = CashPlan(max(0.0, snap.account.cash - reserve), reserve)
    ceiling = c["max_weight"] * c["add_room"] * equity
    adds = [i for i in ideas if i.action in ("add", "buy") and i.score > 0]
    scores = {i.ticker: i.score for i in adds}
    room = {i.ticker: max(0.0, (ceiling if i.cap is None else i.cap * equity) - i.weight * equity)
            for i in adds}
    alloc = {t: 0.0 for t in room}
    left = plan.investable
    while left > 0.01:
        open_ = {t: scores[t] for t in room if room[t] - alloc[t] > 0.01}
        if not open_:
            plan.capped = bool(adds)
            break
        total = sum(open_.values())
        spent = 0.0
        for t, score in open_.items():
            amount = min(left * score / total, room[t] - alloc[t])
            alloc[t] += amount
            spent += amount
        left -= spent
    for t, amount in sorted(alloc.items(), key=lambda kv: -kv[1]):
        if amount >= c["min_buy"]:
            plan.buys[t] = float(int(amount))   # whole dollars, rounded down
    return plan


def _cash_lines(plan: CashPlan, new: set[str] = frozenset()) -> list[str]:
    lines = ["## Cash", ""]
    if plan.investable < 1:
        return lines + [f"Nothing to deploy beyond the ${plan.reserve:,.0f} reserve.", ""]
    lines.append(f"${plan.investable:,.0f} above the ${plan.reserve:,.0f} reserve.")
    if plan.buys:
        lines += ["", *(f"- Buy about **${amount:,.0f} of {t}**"
                        + (" (new position)" if t in new else "")
                        for t, amount in plan.buys.items())]
        if plan.capped:
            lines += ["", f"${plan.leftover:,.0f} stays in cash: the buy ideas are at their "
                          f"size limit."]
    else:
        lines[-1] += (" The buy ideas are at their size limit, so it stays in cash."
                      if plan.capped else " No add ideas today, so it stays in cash.")
    return lines + [""]


def _idea_lines(i: Idea, head: str) -> list[str]:
    return [f"- {head}", *(f"  - {r}" for r in i.reasons), *(f"  - _Note: {n}_" for n in i.notes)]


def _holding_lines(ideas: list[Idea], snap: PortfolioSnapshot) -> list[str]:
    if not ideas:
        return ["No stock holdings to look at.", ""]
    lines = []
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
            lines += _idea_lines(i, head)
        lines.append("")
    return lines


def _watch_lines(watch: list[Idea]) -> list[str]:
    lines = ["## Watchlist", ""]
    if not watch:
        return lines + ["Nothing to score: add tickers you don't own under `watchlist` "
                        "in settings.", ""]
    for action, heading in (("buy", "New longs"), ("short", "Shorts")):
        mine = [i for i in watch if i.action == action]
        if not mine:
            continue
        lines += [f"### {heading}", ""]
        for i in mine:
            head = f"**{i.ticker}** (score {i.score:+g})"
            if action == "buy":
                head += f": start at up to {i.cap:.0%} of the portfolio"
            lines += _idea_lines(i, head)
        lines.append("")
    rest = [i for i in watch if i.action == "watch"]
    if rest:
        lines += ["Still watching: " + ", ".join(f"{i.ticker} ({i.score:+g})" for i in rest), ""]
    return lines


def render(ideas: list[Idea], snap: PortfolioSnapshot, today: date,
           plan: CashPlan | None = None, watch: list[Idea] | None = None,
           sections: list[str] | None = None) -> str:
    """The ideas report. `watch` adds the watchlist and `sections` are extra markdown
    blocks (options ideas, market odds) that go after the cash plan."""
    lines = [f"# Ideas: {today:%a %b %d, %Y}", "",
             "_Rules-based suggestions from stored data, not financial advice. "
             "Each reason shows the points it added._", ""]
    lines += _holding_lines(ideas, snap)
    if watch is not None:
        lines += _watch_lines(watch)
    if plan is not None:
        lines += _cash_lines(plan, {i.ticker for i in watch or []})
    for block in sections or []:
        lines += [block.rstrip(), ""]
    return "\n".join(lines).rstrip() + "\n"
