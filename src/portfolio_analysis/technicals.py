"""Price-behavior stats computed from daily closes (pure functions, no network)."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

TRADING_DAYS = 252


def trailing_return(closes: pd.Series, days: int) -> float | None:
    closes = closes.dropna()
    if len(closes) <= days:
        return None
    return float(closes.iloc[-1] / closes.iloc[-1 - days] - 1)


def ytd_return(closes: pd.Series, today: date | None = None) -> float | None:
    closes = closes.dropna()
    year = (today or date.today()).year
    prior = closes[closes.index.year < year]
    if prior.empty:
        return None
    return float(closes.iloc[-1] / prior.iloc[-1] - 1)


def annualized_volatility(closes: pd.Series) -> float | None:
    rets = closes.dropna().pct_change().dropna()
    return float(rets.std() * math.sqrt(TRADING_DAYS)) if len(rets) > 20 else None


def max_drawdown(closes: pd.Series) -> float | None:
    """Worst peak-to-trough decline, as a negative fraction."""
    closes = closes.dropna()
    if closes.empty:
        return None
    return float((closes / closes.cummax() - 1).min())


def rsi(closes: pd.Series, period: int = 14) -> float | None:
    """Wilder's RSI. Above 70 is conventionally overbought, below 30 oversold."""
    delta = closes.dropna().diff().dropna()
    if len(delta) < period:
        return None
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if loss == 0:
        return 100.0
    return float(100 - 100 / (1 + gain / loss))


def sma(closes: pd.Series, window: int) -> float | None:
    closes = closes.dropna()
    return float(closes.iloc[-window:].mean()) if len(closes) >= window else None


def beta(closes: pd.Series, benchmark: pd.Series) -> float | None:
    rets = pd.concat([closes, benchmark], axis=1).dropna().pct_change().dropna()
    if len(rets) < 60:
        return None
    var = rets.iloc[:, 1].var()
    return float(rets.iloc[:, 0].cov(rets.iloc[:, 1]) / var) if var else None


def trend_label(price: float, sma50: float | None, sma200: float | None) -> str:
    if sma50 is None or sma200 is None:
        return "-"
    if price > sma50 > sma200:
        return "uptrend"
    if price < sma50 < sma200:
        return "downtrend"
    return "mixed"
