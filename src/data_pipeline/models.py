"""Normalized data models shared by every portfolio source."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Holding:
    ticker: str
    quantity: float
    average_cost: float
    price: float
    name: str | None = None

    # Enrichment (filled by market_data.enrich_holdings)
    previous_close: float | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    beta: float | None = None
    dividend_yield: float | None = None
    week52_high: float | None = None
    week52_low: float | None = None
    analyst_target: float | None = None
    analyst_target_low: float | None = None
    analyst_target_high: float | None = None
    analyst_count: int | None = None
    analyst_rating: str | None = None
    next_earnings: date | None = None
    last_eps_surprise_pct: float | None = None

    # Valuation and business quality
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_sales: float | None = None
    ev_to_ebitda: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    free_cash_flow: float | None = None
    debt_to_equity: float | None = None

    # Ownership and positioning
    short_pct_float: float | None = None
    short_ratio: float | None = None  # days to cover
    inst_ownership: float | None = None
    insider_ownership: float | None = None
    ex_dividend_date: date | None = None
    payout_ratio: float | None = None

    # Price behavior (filled by market_data.apply_price_stats)
    return_1w: float | None = None
    return_1m: float | None = None
    return_3m: float | None = None
    return_ytd: float | None = None
    return_1y: float | None = None
    volatility_1y: float | None = None
    max_drawdown_1y: float | None = None
    rsi_14: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    beta_1y: float | None = None  # computed against the first benchmark

    @property
    def market_value(self) -> float:
        return self.quantity * self.price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.average_cost

    @property
    def unrealized_pl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pl_pct(self) -> float | None:
        return self.unrealized_pl / self.cost_basis if self.cost_basis else None

    @property
    def day_change_pct(self) -> float | None:
        if not self.previous_close:
            return None
        return self.price / self.previous_close - 1

    @property
    def week52_position(self) -> float | None:
        """0.0 = at 52-week low, 1.0 = at 52-week high."""
        if self.week52_high is None or self.week52_low is None:
            return None
        span = self.week52_high - self.week52_low
        return (self.price - self.week52_low) / span if span else None

    @property
    def analyst_upside(self) -> float | None:
        return self.analyst_target / self.price - 1 if self.analyst_target and self.price else None


@dataclass
class OptionPosition:
    underlying: str
    option_type: str      # "call" or "put"
    strike: float
    expiration: date
    contracts: float      # negative for short positions
    average_cost: float   # per contract (premium x 100), as Robinhood reports it
    mark_price: float | None = None  # per share; multiply by 100 for contract value
    implied_volatility: float | None = None
    delta: float | None = None
    theta: float | None = None

    @property
    def market_value(self) -> float:
        return self.contracts * (self.mark_price or 0.0) * 100

    @property
    def days_to_expiry(self) -> int:
        return (self.expiration - date.today()).days


@dataclass
class Transaction:
    """A single account event. `kind` is one of: buy, sell, dividend, split."""

    ticker: str
    kind: str
    date: date
    quantity: float
    price: float
    amount: float  # signed cash impact (negative for buys)


@dataclass
class AccountSummary:
    cash: float
    buying_power: float | None = None
    # True when cash was inferred (e.g. summed from a CSV) rather than reported
    cash_is_estimate: bool = False


@dataclass
class PortfolioSnapshot:
    source: str
    taken_at: datetime
    account: AccountSummary
    holdings: list[Holding]
    transactions: list[Transaction] = field(default_factory=list)
    options: list[OptionPosition] = field(default_factory=list)

    @property
    def holdings_value(self) -> float:
        return (sum(h.market_value for h in self.holdings)
                + sum(o.market_value for o in self.options))

    @property
    def total_equity(self) -> float:
        return self.holdings_value + self.account.cash
