"""Market data enrichment via yfinance (free, no API key)."""

from __future__ import annotations

import logging
import math
import warnings
from datetime import date, datetime

import pandas as pd
import yfinance as yf

from src.data_pipeline.models import Holding
from src.portfolio_analysis import technicals as ta

log = logging.getLogger(__name__)

# yfinance internals trigger pandas deprecation warnings that are not actionable here.
# Must run after `import yfinance`, which installs its own 'default' filter on import.
warnings.filterwarnings("ignore", message="The inplace keyword")

# Holding field -> yfinance `info` key
_INFO_FIELDS = {
    "sector": "sector",
    "industry": "industry",
    "market_cap": "marketCap",
    "beta": "beta",
    "dividend_yield": "dividendYield",
    "week52_high": "fiftyTwoWeekHigh",
    "week52_low": "fiftyTwoWeekLow",
    "analyst_target": "targetMeanPrice",
    "analyst_target_low": "targetLowPrice",
    "analyst_target_high": "targetHighPrice",
    "analyst_count": "numberOfAnalystOpinions",
    "analyst_rating": "recommendationKey",
    "forward_pe": "forwardPE",
    "peg_ratio": "trailingPegRatio",
    "price_to_sales": "priceToSalesTrailing12Months",
    "ev_to_ebitda": "enterpriseToEbitda",
    "profit_margin": "profitMargins",
    "revenue_growth": "revenueGrowth",
    "earnings_growth": "earningsGrowth",
    "free_cash_flow": "freeCashflow",
    "debt_to_equity": "debtToEquity",
    "short_pct_float": "shortPercentOfFloat",
    "short_ratio": "shortRatio",
    "inst_ownership": "heldPercentInstitutions",
    "insider_ownership": "heldPercentInsiders",
    "payout_ratio": "payoutRatio",
}


def _next_earnings(ticker) -> date | None:
    try:
        cal = ticker.calendar or {}
        dates = cal.get("Earnings Date") or []
    except Exception:
        return None
    upcoming = sorted(d for d in dates if isinstance(d, date) and d >= date.today())
    return upcoming[0] if upcoming else None


def _last_eps_surprise(ticker) -> float | None:
    """Most recent reported EPS surprise, as a fraction (0.05 = beat by 5%)."""
    try:
        df = ticker.get_earnings_dates(limit=8)
        reported = df.dropna(subset=["Reported EPS"])
        value = reported["Surprise(%)"].iloc[0]  # newest first
    except Exception:
        return None
    return None if pd.isna(value) else float(value) / 100


def enrich_holdings(holdings: list[Holding]) -> None:
    """Fill fundamentals, analyst data, ownership and earnings info in place.

    Sets `price` only when the source didn't provide one (e.g. CSV imports).
    """
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # ETFs 404 on fundamentals
    for h in holdings:
        t = yf.Ticker(h.ticker)
        try:
            info = t.info or {}
        except Exception as exc:
            log.warning("yfinance lookup failed for %s: %s", h.ticker, exc)
            continue

        if not h.price:
            h.price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
        h.name = h.name or info.get("shortName")
        h.previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        for field_name, key in _INFO_FIELDS.items():
            value = info.get(key)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                setattr(h, field_name, value)
        h.pe_ratio = h.pe_ratio or info.get("trailingPE")
        if not h.sector and info.get("quoteType") == "ETF":
            h.sector = "ETF"
        if info.get("exDividendDate"):
            h.ex_dividend_date = datetime.fromtimestamp(info["exDividendDate"]).date()

        is_etf = info.get("quoteType") == "ETF"
        h.next_earnings = None if is_etf else _next_earnings(t)
        h.last_eps_surprise_pct = None if is_etf else _last_eps_surprise(t)


def price_history(symbols: list[str], period: str = "1y") -> pd.DataFrame:
    """Daily adjusted closes, one column per symbol."""
    if not symbols:
        return pd.DataFrame()
    data = yf.download(sorted(set(symbols)), period=period, auto_adjust=True,
                       progress=False, group_by="column")
    closes = data["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(symbols[0])
    return closes


def apply_price_stats(holdings: list[Holding], closes: pd.DataFrame, benchmark: str) -> None:
    bench = closes[benchmark] if benchmark in closes else None
    for h in holdings:
        if h.ticker not in closes:
            continue
        s = closes[h.ticker]
        h.return_1w = ta.trailing_return(s, 5)
        h.return_1m = ta.trailing_return(s, 21)
        h.return_3m = ta.trailing_return(s, 63)
        h.return_ytd = ta.ytd_return(s)
        h.return_1y = ta.trailing_return(s, len(s.dropna()) - 1)
        h.volatility_1y = ta.annualized_volatility(s)
        h.max_drawdown_1y = ta.max_drawdown(s)
        h.rsi_14 = ta.rsi(s)
        h.sma_50 = ta.sma(s, 50)
        h.sma_200 = ta.sma(s, 200)
        h.beta_1y = ta.beta(s, bench) if bench is not None else None


def benchmark_day_change(symbols: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info
            out[sym] = info["regularMarketPrice"] / info["regularMarketPreviousClose"] - 1
        except Exception:
            out[sym] = None
    return out
