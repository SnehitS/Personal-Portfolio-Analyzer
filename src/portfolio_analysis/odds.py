"""What prediction markets expect for the macro events that move stocks (pure functions).

Two kinds of event:
- choice: outcomes where one wins (the Fed cuts, holds or hikes; a recession starts or not).
  Each outcome's price is the market's probability for it.
- ladder: "above X" contracts at several strikes (CPI, unemployment). Their prices trace
  P(value > X), so the strike where it crosses 50% is the market's median, and the 10%
  and 90% crossings give a range.

Prices are the bid/ask midpoint, so a quiet market's odds can be a few points off.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class Quote:
    outcome: str                 # "Hike 25bps", "Above 3.4%"
    prob: float                  # 0 to 1
    strike: float | None = None  # ladders only


@dataclass
class OddsEvent:
    series: str
    event: str                   # KXFEDDECISION-26DEC
    title: str
    closes: datetime | None
    kind: str                    # choice or ladder
    quotes: list[Quote]

    @property
    def unit(self) -> str:
        return "%" if any("%" in q.outcome for q in self.quotes) else ""


def quote_prob(bid, ask, last, max_spread: float = 0.20) -> float | None:
    """Mid of the yes bid and ask; the last trade when the spread is too wide to mean much."""
    b, a, l = (float(x) if x not in (None, "") else None for x in (bid, ask, last))
    if a is not None and a > 0 and b is not None and a - b <= max_spread:
        return (a + b) / 2
    if l is not None and l > 0:
        return l
    return None


def crossing(quotes: list[Quote], level: float) -> tuple[float | None, str]:
    """The strike where P(above) falls through `level`, interpolated between strikes.

    Returns (value, "") or (edge strike, "above"/"below") when the whole ladder is on one
    side of the level, or (None, "") with no strikes."""
    pts = sorted((q.strike, q.prob) for q in quotes if q.strike is not None)
    if not pts:
        return None, ""
    if pts[0][1] < level:
        return pts[0][0], "below"
    if pts[-1][1] >= level:
        return pts[-1][0], "above"
    for (x0, p0), (x1, p1) in zip(pts, pts[1:]):
        if p0 >= level > p1:
            return x0 + (x1 - x0) * (p0 - level) / (p0 - p1), ""
    return None, ""


def _fmt(x: float, unit: str) -> str:
    return f"{x:.1f}%" if unit == "%" else f"{x:,.0f}"   # ladders step 0.1 at best


def _point(quotes: list[Quote], level: float, unit: str) -> str:
    x, side = crossing(quotes, level)
    if x is None:
        return "?"
    return {"below": "under ", "above": "over "}.get(side, "") + _fmt(x, unit)


def median(event: OddsEvent) -> float | None:
    x, side = crossing(event.quotes, 0.5)
    return x if not side else None


def summary(event: OddsEvent, min_prob: float = 0.05) -> str:
    if event.kind == "ladder":
        u = event.unit
        return (f"median {_point(event.quotes, 0.5, u)}, 80% range "
                f"{_point(event.quotes, 0.9, u)} to {_point(event.quotes, 0.1, u)}")
    if len(event.quotes) == 1:
        return f"{event.quotes[0].prob:.0%} yes"
    top = sorted(event.quotes, key=lambda q: -q.prob)
    shown = [q for q in top if q.prob >= min_prob] or top[:1]
    return ", ".join(f"{q.outcome} {q.prob:.0%}" for q in shown)


def change(event: OddsEvent, before: list[Quote] | None, min_move: float = 0.05) -> str:
    """How the odds moved since `before` (an earlier day's quotes), when it's worth saying."""
    if not before:
        return ""
    if event.kind == "ladder":
        then = median(OddsEvent(event.series, event.event, "", None, "ladder", before))
        now = median(event)
        if then is None or now is None or abs(now - then) < 1e-9:
            return ""
        return f"median was {_fmt(then, event.unit)}"
    old = {q.outcome: q.prob for q in before}
    moves = [(q, q.prob - old[q.outcome]) for q in event.quotes if q.outcome in old]
    big = [(q, d) for q, d in moves if abs(d) >= min_move]
    if not big:
        return ""
    q, d = max(big, key=lambda qd: abs(qd[1]))
    label = "yes" if len(event.quotes) == 1 else q.outcome
    return f"{label} {'up' if d > 0 else 'down'} from {old[q.outcome]:.0%}"


def render(events: list[OddsEvent], before: dict[str, tuple[date, list[Quote]]] | None = None,
           today: date | None = None) -> str:
    """Market odds as a report section. `before` maps event -> (day, quotes) from about a
    week ago, for the change column."""
    lines = ["## Market odds", "",
             "_Kalshi prices as probabilities: what traders are betting, not a forecast "
             "of ours._", ""]
    if not events:
        return "\n".join(lines + ["No open markets for the series in settings."]) + "\n"
    for e in events:
        when = f" ({e.closes:%b %d})" if e.closes else ""
        line = f"- **{e.title}**{when}: {summary(e)}"
        then = (before or {}).get(e.event)
        moved = change(e, then[1] if then else None)
        if moved and then:
            ago = (today - then[0]).days if today else None
            line += f"; {moved}" + (f" {ago} days ago" if ago else "")
        lines.append(line)
    return "\n".join(lines) + "\n"
