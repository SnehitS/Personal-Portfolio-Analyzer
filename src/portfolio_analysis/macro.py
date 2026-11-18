"""Macro backdrop from stored series: where rates and inflation are and which way they move."""

from __future__ import annotations

from datetime import date, timedelta


def _at_or_before(points: list[tuple[date, float]], day: date) -> float | None:
    """Value on the latest date <= day. Points are oldest first."""
    found = None
    for d, v in points:
        if d > day:
            break
        found = v
    return found


def _bp(change: float) -> str:
    return f"{change * 100:+.0f} bp"


def treasury_line(points: list[tuple[date, float]]) -> str | None:
    if not points:
        return None
    last_day, last = points[-1]
    parts = []
    for label, days in (("1 week", 7), ("1 month", 30)):
        before = _at_or_before(points, last_day - timedelta(days=days))
        if before is not None:
            parts.append(f"{_bp(last - before)} over {label}")
    tail = f" ({', '.join(parts)})" if parts else ""
    return f"**10-year Treasury:** {last:.2f}%{tail}, as of {last_day:%b %d}"


def fed_funds_line(points: list[tuple[date, float]]) -> str | None:
    if not points:
        return None
    last_day, last = points[-1]
    before = _at_or_before(points, last_day - timedelta(days=92))
    tail = f" ({_bp(last - before)} over 3 months)" if before is not None else ""
    return f"**Fed funds:** {last:.2f}%{tail}, {last_day:%B} average"


def inflation(points: list[tuple[date, float]], month: date) -> float | None:
    """Year-over-year CPI change for the month starting `month`."""
    now = _at_or_before(points, month)
    year_ago = _at_or_before(points, month.replace(year=month.year - 1))
    return now / year_ago - 1 if now and year_ago else None


def cpi_line(points: list[tuple[date, float]]) -> str | None:
    if not points:
        return None
    last_day = points[-1][0]
    now = inflation(points, last_day)
    if now is None:
        return None
    prev_month = (last_day.replace(day=1) - timedelta(days=1)).replace(day=1)
    prev = inflation(points, prev_month)
    trend = ""
    if prev is not None:
        word = "up from" if now > prev + 0.0005 else "down from" if now < prev - 0.0005 else "same as"
        trend = f", {word} {prev:.1%} the month before"
    return f"**Inflation (CPI):** {now:.1%} year over year in {last_day:%B}{trend}"


def backdrop(series: dict[str, list[tuple[date, float]]]) -> list[str]:
    """Markdown bullet lines; empty when nothing is stored yet."""
    lines = [treasury_line(series.get("treasury_10y", [])),
             fed_funds_line(series.get("fed_funds", [])),
             cpi_line(series.get("cpi", []))]
    return [f"- {line}" for line in lines if line]
