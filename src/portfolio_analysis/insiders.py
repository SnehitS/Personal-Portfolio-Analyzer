"""Which insider trades are worth a look.

Open-market purchases are the useful signal: insiders sell for many reasons (taxes, a house,
a pre-scheduled 10b5-1 plan) but buy for one. Option exercises, tax withholding and awards
are paperwork and are ignored.
"""

from __future__ import annotations

from dataclasses import replace

from src.data_pipeline.models import InsiderTrade

DEFAULTS = {
    "min_buy": 25_000,      # purchases at least this big (dollars)
    "min_sale": 1_000_000,  # sales at least this big
}


def combine(trades: list[InsiderTrade]) -> list[InsiderTrade]:
    """One Form 4 often lists a sale as many fills at different prices. Merge each person's
    trades of one kind on one day into a single trade at the average price."""
    groups: dict[tuple, list[InsiderTrade]] = {}
    for t in trades:
        groups.setdefault((t.ticker, t.day, t.executive, t.code), []).append(t)
    out = []
    for group in groups.values():
        shares = sum(t.shares for t in group)
        value = sum(t.value or 0 for t in group)
        priced = sum(t.shares for t in group if t.price)
        out.append(replace(group[0], shares=shares, price=value / priced if priced else None,
                           fetched_at=max((t.fetched_at for t in group if t.fetched_at),
                                          default=None)))
    return out


def notable(trades: list[InsiderTrade], cfg: dict | None = None) -> list[InsiderTrade]:
    c = {**DEFAULTS, **(cfg or {})}
    out = []
    for t in combine(trades):
        value = t.value or 0
        if (t.code == "P" and value >= c["min_buy"]) or (t.code == "S" and value >= c["min_sale"]):
            out.append(t)
    return sorted(out, key=lambda t: (t.code != "P", -(t.value or 0)))


def money(x: float) -> str:
    if x >= 1e6:
        return f"${x / 1e6:,.1f}M"
    return f"${x / 1e3:,.0f}K" if x >= 1e3 else f"${x:,.0f}"


def describe(t: InsiderTrade) -> str:
    """"INTC: Lip Bu Tan (Director, CEO) bought 150,000 shares (~$9.7M) on Nov 02"."""
    who = t.name + (f" ({t.title})" if t.title else "")
    value = f" (~{money(t.value)})" if t.value else ""
    return f"{t.ticker}: {who} {t.verb or t.code} {t.shares:,.0f} shares{value} on {t.day:%b %d}"
