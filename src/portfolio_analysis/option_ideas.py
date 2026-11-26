"""Covered calls and cash-secured puts that fit the stock ideas (pure functions, no network).

- Covered calls go on holdings you'd hold or trim anyway, with 100+ shares not already
  covered. Never on adds: you want more of those, so capping the upside works against you.
- Cash-secured puts go on adds and new longs, when the cash above the reserve covers the
  collateral. Assignment is buying the stock you wanted, for less than today's price.

Strikes are picked by delta (Black-Scholes on each contract's own IV), which is roughly the
chance the option ends in the money. Expiries are about a month out, ending before earnings
when the chain allows it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import pandas as pd

from src.data_pipeline.models import Holding, PortfolioSnapshot
from src.portfolio_analysis.metrics import fifo_lots
from src.portfolio_analysis.options import mid

DEFAULTS = {
    "call_delta": 0.25,          # covered calls: about a 1 in 4 chance of being called away
    "put_delta": 0.30,           # puts: about a 3 in 10 chance of buying the stock
    "target_days": 30,
    "min_days": 21,
    "max_days": 50,
    "min_yield": 0.05,           # skip premiums under 5% a year on the shares or collateral
    "rate": 0.04,                # risk-free rate for delta
    "call_on": ["hold", "trim"], # holding actions that get covered calls
    "low_iv_rank": 0.30,         # note when IV is in the bottom 30% of its stored range
}


@dataclass
class Chain:
    expiration: date
    spans_earnings: bool         # no expiry in range ended before the next earnings
    calls: pd.DataFrame
    puts: pd.DataFrame
    spot: float


@dataclass
class OptionPick:
    ticker: str
    kind: str                    # call or put
    expiration: date
    days: int
    strike: float
    premium: float               # mid, per share
    delta: float                 # absolute value
    iv: float
    spot: float

    @property
    def basis(self) -> float:
        """What the premium is a return on: the shares for a call, the collateral for a put."""
        return self.spot if self.kind == "call" else self.strike

    @property
    def yield_pct(self) -> float:
        return self.premium / self.basis

    @property
    def annualized(self) -> float:
        return self.yield_pct * 365 / self.days if self.days > 0 else 0.0

    @property
    def distance(self) -> float:
        """How far the strike is from the price: above it for calls, below for puts."""
        return self.strike / self.spot - 1 if self.kind == "call" else 1 - self.strike / self.spot


@dataclass
class OptionIdea:
    strategy: str                # covered call or cash-secured put
    pick: OptionPick
    contracts: int
    notes: list[str] = field(default_factory=list)

    @property
    def ticker(self) -> str:
        return self.pick.ticker

    @property
    def premium_total(self) -> float:
        return self.pick.premium * 100 * self.contracts

    @property
    def collateral(self) -> float:
        return self.pick.strike * 100 * self.contracts if self.pick.kind == "put" else 0.0


def bs_delta(kind: str, spot: float, strike: float, days: int, iv: float,
             rate: float = 0.04) -> float | None:
    """Black-Scholes delta: 0 to 1 for calls, -1 to 0 for puts."""
    if spot <= 0 or strike <= 0 or days <= 0 or iv <= 0:
        return None
    t = days / 365
    d1 = (math.log(spot / strike) + (rate + iv * iv / 2) * t) / (iv * math.sqrt(t))
    n = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    return n if kind == "call" else n - 1


def pick_expiry(expiries: list[date], today: date, avoid: date | None = None,
                cfg: dict | None = None) -> tuple[date, bool] | None:
    """The expiry nearest target_days within [min_days, max_days], preferring one that
    ends before `avoid` (earnings). Returns (expiry, spans_avoid) or None."""
    c = {**DEFAULTS, **(cfg or {})}
    window = [e for e in expiries if c["min_days"] <= (e - today).days <= c["max_days"]]
    if not window:
        return None
    before = [e for e in window if avoid is None or e < avoid or avoid < today]
    pool = before or window
    best = min(pool, key=lambda e: abs((e - today).days - c["target_days"]))
    return best, not before


def pick_strike(df: pd.DataFrame, ticker: str, kind: str, spot: float, expiration: date,
                today: date, target_delta: float, rate: float = 0.04) -> OptionPick | None:
    """The quoted out-of-the-money contract whose delta is closest to target_delta."""
    days = (expiration - today).days
    best: OptionPick | None = None
    for _, row in df.iterrows():
        strike = float(row["strike"])
        if (kind == "call" and strike <= spot) or (kind == "put" and strike >= spot):
            continue
        if not (row.get("bid") or 0) > 0:
            continue
        iv = row.get("impliedVolatility")
        premium = mid(row)
        if not premium or iv is None or not 0.01 < iv < 5:
            continue
        delta = bs_delta(kind, spot, strike, days, float(iv), rate)
        if delta is None:
            continue
        pick = OptionPick(ticker, kind, expiration, days, strike, premium, abs(delta),
                          float(iv), spot)
        if best is None or abs(pick.delta - target_delta) < abs(best.delta - target_delta):
            best = pick
    return best


ChainFn = Callable[[str, date | None], "Chain | None"]


def _iv_note(ticker: str, iv_ranks: dict[str, float] | None, c: dict) -> list[str]:
    rank = (iv_ranks or {}).get(ticker)
    if rank is not None and rank < c["low_iv_rank"]:
        return [f"IV is low for this stock (rank {rank:.0%}), so the premium is thin"]
    return []


def _assignment_tax(h: Holding, snap: PortfolioSnapshot, pick: OptionPick) -> list[str]:
    lots = fifo_lots(snap.transactions).get(h.ticker, [])
    short = [l for l in lots if not l.is_long_term(pick.expiration) and pick.strike > l.price]
    if not short:
        return []
    shares = sum(l.quantity for l in short)
    return [f"{shares:g} shares would still be short-term at expiry; if they're called away "
            f"the gain is taxed as short-term"]


def covered_calls(snap: PortfolioSnapshot, holding_ideas: list, chain_for: ChainFn,
                  cfg: dict | None = None, today: date | None = None,
                  iv_ranks: dict[str, float] | None = None) -> list[OptionIdea]:
    c = {**DEFAULTS, **(cfg or {})}
    today = today or date.today()
    held = {h.ticker: h for h in snap.holdings}
    out = []
    for idea in holding_ideas:
        h = held.get(idea.ticker)
        if h is None or idea.action not in c["call_on"]:
            continue
        covered = sum(-o.contracts for o in snap.options
                      if o.underlying == h.ticker and o.option_type == "call" and o.contracts < 0)
        free = int(h.quantity // 100) - int(covered)
        if free < 1:
            continue
        chain = chain_for(h.ticker, h.next_earnings)
        if chain is None:
            continue
        pick = pick_strike(chain.calls, h.ticker, "call", chain.spot, chain.expiration, today,
                           c["call_delta"], c["rate"])
        if pick is None or pick.annualized < c["min_yield"]:
            continue
        o = OptionIdea("covered call", pick, free)
        if idea.action == "trim":
            o.notes.append("it's a trim idea, and being called away does the trim at a "
                           "higher price")
        if chain.spans_earnings and h.next_earnings:
            o.notes.append(f"earnings on {h.next_earnings:%b %d} fall before expiry; a jump "
                           f"past the strike gets the shares called away")
        o.notes += _assignment_tax(h, snap, pick) if snap.transactions else []
        o.notes += _iv_note(h.ticker, iv_ranks, c)
        out.append(o)
    return sorted(out, key=lambda o: -o.pick.annualized)


def cash_secured_puts(snap: PortfolioSnapshot, buy_ideas: list, info: dict[str, Holding],
                      cash: float, chain_for: ChainFn, cfg: dict | None = None,
                      today: date | None = None, iv_ranks: dict[str, float] | None = None,
                      max_weight: float | None = None) -> tuple[list[OptionIdea], list[str]]:
    """One put per add or new-long idea that the cash covers. Also returns notes on the
    ones whose collateral is more than the cash, so the report can say why they're missing."""
    c = {**DEFAULTS, **(cfg or {})}
    today = today or date.today()
    equity = snap.total_equity
    out, too_big = [], []
    for idea in buy_ideas:
        if idea.action not in ("add", "buy"):
            continue
        h = info.get(idea.ticker)
        chain = chain_for(idea.ticker, h.next_earnings if h else None)
        if chain is None:
            continue
        pick = pick_strike(chain.puts, idea.ticker, "put", chain.spot, chain.expiration, today,
                           c["put_delta"], c["rate"])
        if pick is None or pick.annualized < c["min_yield"]:
            continue
        o = OptionIdea("cash-secured put", pick, 1)
        if o.collateral > cash:
            too_big.append(f"{idea.ticker}: a put needs ${o.collateral:,.0f} of cash, more than "
                           f"the ${cash:,.0f} above the reserve")
            continue
        net = pick.strike - pick.premium
        o.notes.append(f"if assigned you buy 100 shares at ${pick.strike:,.2f}, "
                       f"${net:,.2f} after the premium ({1 - net / pick.spot:.0%} below today)")
        held_value = h.market_value if h else 0.0   # watchlist tickers hold 0 shares
        after = (held_value + pick.strike * 100) / equity if equity else 0
        if max_weight is not None and after > max_weight:
            o.notes.append(f"assignment would make it {after:.0%} of the portfolio, over the "
                           f"{max_weight:.0%} limit")
        if chain.spans_earnings and h and h.next_earnings:
            o.notes.append(f"earnings on {h.next_earnings:%b %d} fall before expiry")
        o.notes += _iv_note(idea.ticker, iv_ranks, c)
        out.append(o)
    return sorted(out, key=lambda o: -o.pick.annualized), too_big


def _line(o: OptionIdea) -> list[str]:
    p = o.pick
    head = (f"- **{p.ticker}**: sell {o.contracts} {p.expiration:%b %d} ${p.strike:g} {p.kind}"
            f"{'s' if o.contracts > 1 else ''} for about ${p.premium:.2f} a share "
            f"(${o.premium_total:,.0f})")
    if p.kind == "call":
        body = (f"{p.yield_pct:.1%} of the price, {p.annualized:.0%} a year. Called away above "
                f"${p.strike:g}, {p.distance:.0%} above today; about a {p.delta:.0%} chance.")
    else:
        body = (f"{p.yield_pct:.1%} on ${o.collateral:,.0f} of collateral, {p.annualized:.0%} a "
                f"year. About a {p.delta:.0%} chance you end up buying.")
    return [head, f"  - {body}", *(f"  - _Note: {n}_" for n in o.notes)]


def render(calls: list[OptionIdea], puts: list[OptionIdea], too_big: list[str]) -> str:
    lines = ["## Options", "",
             "_Mid prices from the last chain; fills are usually a little worse. Chances are "
             "each contract's delta._", ""]
    if not calls and not puts and not too_big:
        return "\n".join(lines + ["No covered calls or cash-secured puts fit today's ideas."])
    if calls:
        lines += ["### Covered calls", ""]
        for o in calls:
            lines += _line(o)
        lines.append("")
    if puts or too_big:
        lines += ["### Cash-secured puts", ""]
        for o in puts:
            lines += _line(o)
        if puts:
            lines += ["", "Each put ties up cash the plan above would spend: do one or the "
                          "other."]
        lines += [f"- _{t}_" for t in too_big]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
