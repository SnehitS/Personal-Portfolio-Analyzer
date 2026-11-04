"""Option-implied expectations from a chain (pure functions, no network).

- ATM implied volatility per expiry, from the strikes nearest the spot price
- Expected move to an expiry: the at-the-money straddle price, which is what the market
  is paying for a move in either direction
- 30-day IV interpolated in total variance between the expiries around 30 days
- IV rank against our own stored history, and IV vs realized volatility
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd


@dataclass
class ExpiryView:
    expiration: date
    days: int
    atm_strike: float
    atm_iv: float | None       # annualized, e.g. 0.32
    straddle: float | None     # ATM call + put mid, per share
    spot: float

    @property
    def expected_move_pct(self) -> float | None:
        return self.straddle / self.spot if self.straddle and self.spot else None

    @property
    def iv_move_pct(self) -> float | None:
        """One standard deviation implied by IV alone: iv * sqrt(t)."""
        if self.atm_iv is None or self.days <= 0:
            return None
        return self.atm_iv * math.sqrt(self.days / 365)


def mid(row: pd.Series) -> float | None:
    """Bid/ask midpoint; last trade only when there's no live quote."""
    bid, ask = row.get("bid") or 0.0, row.get("ask") or 0.0
    if bid > 0 and ask >= bid:
        return float(bid + ask) / 2
    last = row.get("lastPrice")
    return float(last) if last and last > 0 else None


def _quoted(df: pd.DataFrame) -> pd.DataFrame:
    """Contracts with a live bid; Yahoo's IV on unquoted strikes is noise."""
    return df[(df["bid"] > 0) & (df["ask"] > 0)] if len(df) else df


def expiry_view(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, expiration: date,
                today: date | None = None, strikes_each_side: int = 2) -> ExpiryView | None:
    today = today or date.today()
    if not spot or calls.empty or puts.empty:
        return None
    common = sorted(set(calls["strike"]) & set(puts["strike"]))
    if not common:
        return None
    atm = min(common, key=lambda k: abs(k - spot))
    c = calls[calls["strike"] == atm].iloc[0]
    p = puts[puts["strike"] == atm].iloc[0]
    c_mid, p_mid = mid(c), mid(p)
    straddle = float(c_mid + p_mid) if c_mid is not None and p_mid is not None else None

    # IV: average the few quoted strikes nearest spot on both sides, calls and puts
    near = sorted(common, key=lambda k: abs(k - spot))[: strikes_each_side * 2]
    ivs = [float(v) for df in (_quoted(calls), _quoted(puts))
           for v in df[df["strike"].isin(near)]["impliedVolatility"]
           if v and 0.01 < v < 5]
    atm_iv = sum(ivs) / len(ivs) if ivs else None
    return ExpiryView(expiration, (expiration - today).days, atm, atm_iv, straddle, spot)


def interpolated_iv(views: list[ExpiryView], target_days: int = 30) -> float | None:
    """IV at a constant maturity, interpolating total variance (iv^2 * t) between the
    expiries either side of `target_days`; flat beyond the ends."""
    pts = sorted((v.days, v.atm_iv) for v in views if v.atm_iv and v.days > 0)
    if not pts:
        return None
    if target_days <= pts[0][0]:
        return pts[0][1]
    if target_days >= pts[-1][0]:
        return pts[-1][1]
    for (d0, iv0), (d1, iv1) in zip(pts, pts[1:]):
        if d0 <= target_days <= d1:
            w0, w1 = iv0 ** 2 * d0, iv1 ** 2 * d1
            var = w0 + (w1 - w0) * (target_days - d0) / (d1 - d0)
            return math.sqrt(var / target_days)
    return None


def first_expiry_after(views: list[ExpiryView], day: date) -> ExpiryView | None:
    """The first expiry on or after `day`, e.g. the one that captures an earnings move.
    Earnings after the close need the expiry after that date; callers pass day + 1 then."""
    later = [v for v in views if v.expiration >= day]
    return min(later, key=lambda v: v.expiration) if later else None


def earnings_move(views: list[ExpiryView], earnings: date) -> float | None:
    """Implied size of the earnings move alone, as a fraction of spot.

    The straddle to the first expiry after earnings prices the event plus ordinary days.
    Take the ordinary per-day variance from the last expiry before earnings and remove it:
    event^2 = post^2 - pre^2 * (other days in post) / pre_days. Without a pre-earnings
    expiry, fall back to the whole post-earnings straddle (an overestimate)."""
    post = first_expiry_after(views, earnings + timedelta(days=1))
    if post is None or post.expected_move_pct is None:
        return None
    pre = max((v for v in views if v.expiration < earnings and v.expected_move_pct
               and v.days > 0), key=lambda v: v.expiration, default=None)
    if pre is None:
        return post.expected_move_pct
    ordinary = pre.expected_move_pct ** 2 * max(post.days - 1, 0) / pre.days
    event_var = post.expected_move_pct ** 2 - ordinary
    return math.sqrt(event_var) if event_var > 0 else post.expected_move_pct


def iv_rank(current: float | None, history: list[float]) -> float | None:
    """Where today's IV sits in its stored range, 0 = lowest seen, 1 = highest.
    Needs some history to mean anything; returns None under 20 observations."""
    if current is None or len(history) < 20:
        return None
    lo, hi = min(history), max(history)
    return (current - lo) / (hi - lo) if hi > lo else None


def realized_vol(closes: pd.Series, days: int = 30) -> float | None:
    """Annualized close-to-close volatility over the last `days` trading days."""
    rets = closes.dropna().pct_change().dropna().iloc[-days:]
    return float(rets.std() * math.sqrt(252)) if len(rets) >= max(10, days // 2) else None


def past_earnings_moves(closes: pd.Series, dates: list[date]) -> list[float]:
    """Absolute close-to-close move across each past earnings report. Uses the close
    before the report date and the close after it, which covers both pre-market and
    after-hours reports."""
    closes = closes.dropna()
    idx = closes.index.date if hasattr(closes.index, "date") else closes.index
    out = []
    for d in dates:
        before = closes[idx < d]
        after = closes[idx > d]
        if len(before) and len(after):
            out.append(abs(float(after.iloc[0] / before.iloc[-1] - 1)))
    return out
