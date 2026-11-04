"""Upcoming dated events that can move a holding or change what you should do with it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from src.data_pipeline.models import PortfolioSnapshot
from src.portfolio_analysis.metrics import fifo_lots, lot_mismatches, weights

KIND_ORDER = {"earnings": 0, "macro": 1, "option expiry": 2, "ex-dividend": 3, "long-term": 4}


@dataclass
class Catalyst:
    day: date
    ticker: str           # "" for market-wide events
    kind: str             # earnings, ex-dividend, option expiry, long-term, macro
    detail: str = ""
    weight: float = 0.0   # share of the portfolio exposed, for sorting what matters


def build(snap: PortfolioSnapshot, today: date, horizon_days: int = 30,
          macro: list[dict] | None = None,
          earnings_detail: dict[str, dict] | None = None) -> list[Catalyst]:
    """`earnings_detail` maps ticker -> {"date", "hour", "eps_estimate"} from Finnhub and
    overrides yfinance's date when present. `macro` is [{"date", "event"}] from settings."""
    end = today + timedelta(days=horizon_days)
    w = weights(snap)
    earnings_detail = earnings_detail or {}
    out: list[Catalyst] = []

    def add(day: date | None, ticker: str, kind: str, detail: str = "") -> None:
        if day and today <= day <= end:
            out.append(Catalyst(day, ticker, kind, detail, w.get(ticker, 0.0)))

    for h in snap.holdings:
        info = earnings_detail.get(h.ticker, {})
        day = info.get("date") or h.next_earnings
        parts = []
        if info.get("hour") in ("bmo", "amc"):
            parts.append("before open" if info["hour"] == "bmo" else "after close")
        if info.get("eps_estimate") is not None:
            parts.append(f"EPS est. {info['eps_estimate']:.2f}")
        add(day, h.ticker, "earnings", ", ".join(parts))
        if h.ex_dividend_date:
            add(h.ex_dividend_date, h.ticker, "ex-dividend",
                f"yield {h.dividend_yield:.2f}%" if h.dividend_yield else "")

    for o in snap.options:
        side = "short" if o.contracts < 0 else "long"
        add(o.expiration, o.underlying, "option expiry",
            f"{side} {abs(o.contracts):g}x {o.strike:g}{o.option_type[:1].upper()}")

    lots = fifo_lots(snap.transactions)
    unreliable = {t for t, _, _ in lot_mismatches(snap.holdings, lots)}
    for ticker, ticker_lots in lots.items():
        if ticker in unreliable:
            continue
        for lot in ticker_lots:
            if not lot.is_long_term(today):
                add(lot.long_term_date, ticker, "long-term",
                    f"{lot.quantity:g} shares bought {lot.opened:%Y-%m-%d} at ${lot.price:,.2f}")

    for m in macro or []:
        day = m.get("date")
        day = date.fromisoformat(day) if isinstance(day, str) else day
        add(day, "", "macro", m.get("event", ""))

    return sorted(out, key=lambda c: (c.day, KIND_ORDER.get(c.kind, 9), -c.weight))
