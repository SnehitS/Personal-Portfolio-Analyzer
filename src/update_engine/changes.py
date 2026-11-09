"""What changed between two snapshots: price moves, trades, weights and equity."""

from __future__ import annotations

from dataclasses import dataclass

from src.data_pipeline.models import PortfolioSnapshot
from src.portfolio_analysis.metrics import weights

# Share counts closer than this are treated as unchanged (fractional-share rounding)
_SHARE_TOLERANCE = 1e-6


@dataclass
class PositionChange:
    ticker: str
    status: str             # new, closed, added, trimmed, held
    shares_before: float
    shares_after: float
    price_before: float | None
    price_after: float | None
    weight_before: float
    weight_after: float

    @property
    def price_change_pct(self) -> float | None:
        if not self.price_before or self.price_after is None:
            return None
        return self.price_after / self.price_before - 1

    @property
    def market_gain(self) -> float:
        """Dollars made or lost from price on the shares held through both snapshots."""
        if self.price_before is None or self.price_after is None:
            return 0.0
        return min(self.shares_before, self.shares_after) * (self.price_after - self.price_before)


@dataclass
class PortfolioChange:
    equity_before: float
    equity_after: float
    cash_before: float
    cash_after: float
    net_deposits: float     # deposits minus withdrawals between the two snapshots
    positions: list[PositionChange]

    @property
    def equity_change(self) -> float:
        return self.equity_after - self.equity_before

    @property
    def market_gain(self) -> float:
        """Equity change that wasn't new money coming in or going out."""
        return self.equity_change - self.net_deposits

    @property
    def market_gain_pct(self) -> float | None:
        return self.market_gain / self.equity_before if self.equity_before else None

    def by_status(self, *statuses: str) -> list[PositionChange]:
        return [p for p in self.positions if p.status in statuses]

    def movers(self, n: int = 5) -> list[PositionChange]:
        """Positions held through both snapshots, biggest dollar impact first."""
        held = [p for p in self.positions if p.shares_before and p.shares_after]
        return sorted(held, key=lambda p: -abs(p.market_gain))[:n]


def _status(before: float, after: float) -> str:
    if before <= _SHARE_TOLERANCE:
        return "new"
    if after <= _SHARE_TOLERANCE:
        return "closed"
    if after - before > _SHARE_TOLERANCE:
        return "added"
    if before - after > _SHARE_TOLERANCE:
        return "trimmed"
    return "held"


def compare(before: PortfolioSnapshot, after: PortfolioSnapshot) -> PortfolioChange:
    old = {h.ticker: h for h in before.holdings}
    new = {h.ticker: h for h in after.holdings}
    w_old, w_new = weights(before), weights(after)

    positions = []
    for ticker in sorted(old.keys() | new.keys()):
        a, b = old.get(ticker), new.get(ticker)
        shares_a, shares_b = (a.quantity if a else 0.0), (b.quantity if b else 0.0)
        positions.append(PositionChange(
            ticker=ticker,
            status=_status(shares_a, shares_b),
            shares_before=shares_a,
            shares_after=shares_b,
            price_before=a.price if a else None,
            price_after=b.price if b else None,
            weight_before=w_old.get(ticker, 0.0),
            weight_after=w_new.get(ticker, 0.0),
        ))

    start, end = before.taken_at.date(), after.taken_at.date()
    deposits = sum(t.amount for t in after.transactions
                   if t.kind in ("deposit", "withdrawal") and start < t.date <= end)
    return PortfolioChange(before.total_equity, after.total_equity, before.account.cash,
                           after.account.cash, deposits, positions)
