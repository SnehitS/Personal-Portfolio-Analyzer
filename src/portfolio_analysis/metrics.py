"""Portfolio-level analytics computed from a snapshot."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.data_pipeline.models import PortfolioSnapshot, Transaction
from src.portfolio_analysis import technicals as ta

LONG_TERM_DAYS = 365


@dataclass
class Lot:
    ticker: str
    opened: date
    quantity: float
    price: float

    @property
    def long_term_date(self) -> date:
        return self.opened + timedelta(days=LONG_TERM_DAYS + 1)

    def is_long_term(self, today: date) -> bool:
        return today >= self.long_term_date


@dataclass
class RealizedGain:
    ticker: str
    sold: date
    quantity: float
    proceeds: float
    cost: float
    long_term: bool

    @property
    def gain(self) -> float:
        return self.proceeds - self.cost


def weights(snap: PortfolioSnapshot) -> dict[str, float]:
    total = snap.total_equity
    return {h.ticker: h.market_value / total for h in snap.holdings} if total else {}


def sector_weights(snap: PortfolioSnapshot) -> dict[str, float]:
    total = snap.holdings_value
    out: dict[str, float] = defaultdict(float)
    for h in snap.holdings:
        out[h.sector or "Unknown"] += h.market_value / total if total else 0.0
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def portfolio_day_change(snap: PortfolioSnapshot) -> float | None:
    prev = sum(h.quantity * h.previous_close for h in snap.holdings if h.previous_close)
    now = sum(h.market_value for h in snap.holdings if h.previous_close)
    return now / prev - 1 if prev else None


def _run_fifo(txns: list[Transaction]) -> tuple[dict[str, list[Lot]], list[RealizedGain]]:
    """Replay transactions FIFO (Robinhood's default cost-basis method)."""
    lots: dict[str, deque[Lot]] = defaultdict(deque)
    realized: list[RealizedGain] = []
    for t in sorted(txns, key=lambda t: t.date):
        book = lots[t.ticker]
        if t.kind == "buy":
            book.append(Lot(t.ticker, t.date, t.quantity, t.price))
        elif t.kind == "sell":
            remaining = t.quantity
            while remaining > 1e-9 and book:
                lot = book[0]
                take = min(lot.quantity, remaining)
                realized.append(RealizedGain(t.ticker, t.date, take, take * t.price,
                                             take * lot.price, lot.is_long_term(t.date)))
                lot.quantity -= take
                remaining -= take
                if lot.quantity <= 1e-9:
                    book.popleft()
        elif t.kind == "split":
            held = sum(l.quantity for l in book)
            if held > 0:
                ratio = (held + t.quantity) / held
                for lot in book:
                    lot.quantity *= ratio
                    lot.price /= ratio
    return {k: list(v) for k, v in lots.items() if v}, realized


def fifo_lots(txns: list[Transaction]) -> dict[str, list[Lot]]:
    """Open tax lots per ticker."""
    return _run_fifo(txns)[0]


def realized_gains(txns: list[Transaction], year: int | None = None) -> list[RealizedGain]:
    gains = _run_fifo(txns)[1]
    return [g for g in gains if year is None or g.sold.year == year]


def dividend_income(txns: list[Transaction], year: int | None = None) -> float:
    return sum(t.amount for t in txns
               if t.kind == "dividend" and (year is None or t.date.year == year))


@dataclass
class PortfolioRisk:
    beta: float | None
    volatility: float | None
    correlated_pairs: list[tuple[str, str, float]]


def portfolio_risk(snap: PortfolioSnapshot, closes: pd.DataFrame, benchmark: str,
                   corr_threshold: float = 0.8) -> PortfolioRisk:
    """Weighted beta, annualized volatility of today's mix, and highly correlated pairs."""
    tickers = [h.ticker for h in snap.holdings if h.ticker in closes]
    total = sum(h.market_value for h in snap.holdings if h.ticker in closes)
    if not tickers or not total:
        return PortfolioRisk(None, None, [])
    w = pd.Series({h.ticker: h.market_value / total for h in snap.holdings if h.ticker in closes})

    rets = closes[tickers].pct_change(fill_method=None).dropna(how="all").fillna(0)
    port_rets = rets @ w[tickers]
    vol = float(port_rets.std() * (ta.TRADING_DAYS ** 0.5)) if len(port_rets) > 20 else None
    beta = None
    if benchmark in closes:
        port_curve = (1 + port_rets).cumprod()
        beta = ta.beta(port_curve, closes[benchmark].reindex(port_curve.index))

    corr = rets.corr()
    pairs = [
        (a, b, float(corr.loc[a, b]))
        for i, a in enumerate(tickers) for b in tickers[i + 1:]
        if corr.loc[a, b] >= corr_threshold
    ]
    return PortfolioRisk(beta, vol, sorted(pairs, key=lambda p: -p[2]))


def risk_flags(snap: PortfolioSnapshot, cfg: dict, today: date | None = None) -> list[str]:
    today = today or date.today()
    flags: list[str] = []
    w = weights(snap)

    for h in snap.holdings:
        if w.get(h.ticker, 0) > cfg["max_position_weight"]:
            flags.append(f"{h.ticker} is {w[h.ticker]:.0%} of the portfolio "
                         f"(threshold {cfg['max_position_weight']:.0%})")
        if h.unrealized_pl_pct is not None and h.unrealized_pl_pct < -cfg["large_loss_pct"]:
            flags.append(f"{h.ticker} is down {-h.unrealized_pl_pct:.0%} from cost basis")
        if h.next_earnings and 0 <= (h.next_earnings - today).days <= cfg["earnings_window_days"]:
            flags.append(f"{h.ticker} reports earnings {h.next_earnings:%a %b %d}")

        if h.rsi_14 is not None and (h.rsi_14 >= 70 or h.rsi_14 <= 30):
            state = "overbought" if h.rsi_14 >= 70 else "oversold"
            flags.append(f"{h.ticker} RSI {h.rsi_14:.0f} ({state})")
        if h.sma_200 and h.price < h.sma_200 and h.sma_50 and h.sma_50 < h.sma_200:
            flags.append(f"{h.ticker} is below its 200-day average and in a downtrend")
        if h.short_pct_float and h.short_pct_float > cfg["high_short_interest"]:
            flags.append(f"{h.ticker} short interest is {h.short_pct_float:.0%} of float")
        if h.ex_dividend_date and 0 <= (h.ex_dividend_date - today).days <= cfg["earnings_window_days"]:
            flags.append(f"{h.ticker} goes ex-dividend {h.ex_dividend_date:%a %b %d}")

    for sector, sw in sector_weights(snap).items():
        if sw > cfg["max_sector_weight"]:
            flags.append(f"{sector} is {sw:.0%} of holdings "
                         f"(threshold {cfg['max_sector_weight']:.0%})")

    window = timedelta(days=cfg["long_term_window_days"])
    for ticker, lots in fifo_lots(snap.transactions).items():
        for lot in lots:
            if not lot.is_long_term(today) and lot.long_term_date - today <= window:
                flags.append(f"{lot.quantity:g} {ticker} shares turn long-term on "
                             f"{lot.long_term_date:%b %d} (selling earlier = short-term gains tax)")

    for o in snap.options:
        if o.days_to_expiry <= cfg["option_expiry_window_days"]:
            flags.append(f"{o.underlying} {o.strike:g}{o.option_type[:1].upper()} expires "
                         f"{o.expiration:%b %d} ({o.days_to_expiry}d)")

    if snap.total_equity and snap.account.cash / snap.total_equity > cfg["idle_cash_pct"]:
        flags.append(f"{snap.account.cash / snap.total_equity:.0%} of the account is "
                     f"uninvested cash")
    return flags
