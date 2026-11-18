"""Robinhood source via the unofficial robin_stocks library.

READ-ONLY: only the read functions below are imported. Nothing in this module
(or this project) can place, modify, or cancel orders.
"""

from __future__ import annotations

import os
import threading
from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.data_pipeline.market_data import split_transactions, stock_splits
from src.data_pipeline.models import (
    AccountSummary, Holding, OptionPosition, PortfolioSnapshot, Transaction,
)
from src.data_pipeline.sources.base import PortfolioSource

MARKET_TZ = ZoneInfo("America/New_York")
SESSION_DAYS = 30       # how long a saved login is asked to last (Robinhood may cap it)
LOGIN_TIMEOUT_MIN = 10  # how long to wait for you to approve a login in the Robinhood app


def _f(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def market_date(timestamp: str) -> date:
    """Robinhood timestamps are UTC; trade dates (and holding periods) follow New York time."""
    utc = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return utc.astimezone(MARKET_TZ).date()


class RobinhoodSource(PortfolioSource):
    name = "robinhood"

    def __init__(self, username: str | None = None, password: str | None = None,
                 totp_secret: str | None = None, session_days: float = SESSION_DAYS,
                 login_timeout_min: float = LOGIN_TIMEOUT_MIN):
        self.username = username or os.getenv("ROBINHOOD_USERNAME")
        self.password = password or os.getenv("ROBINHOOD_PASSWORD")
        self.totp_secret = totp_secret or os.getenv("ROBINHOOD_TOTP_SECRET")
        if not self.username or not self.password:
            raise ValueError("Set ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD in .env")
        self.session_days = session_days
        self.login_timeout_min = login_timeout_min

    def _login(self) -> None:
        from robin_stocks.robinhood import login

        mfa_code = None
        if self.totp_secret:
            import pyotp
            mfa_code = pyotp.TOTP(self.totp_secret).now()
        # Without a TOTP secret, Robinhood asks you to approve the login in its app (or for
        # an SMS/email code). The session is cached in ~/.tokens so most runs skip that.
        # robin_stocks waits forever for an app approval, so run it in a thread and give up
        # after a while: an unattended run then fails cleanly instead of hanging.
        result: dict = {}

        def attempt() -> None:
            try:
                result["ok"] = login(self.username, self.password, mfa_code=mfa_code,
                                     store_session=True,
                                     expiresIn=int(self.session_days * 86400))
            except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
                result["error"] = e

        worker = threading.Thread(target=attempt, name="robinhood-login", daemon=True)
        worker.start()
        worker.join(self.login_timeout_min * 60)
        if worker.is_alive():
            raise TimeoutError(f"Robinhood login wasn't approved within "
                               f"{self.login_timeout_min:g} minutes. Approve it in the "
                               "Robinhood app, or run `snapshot` by hand.")
        if "error" in result:
            raise result["error"]

    def fetch_snapshot(self) -> PortfolioSnapshot:
        from robin_stocks.robinhood import build_holdings, load_account_profile

        self._login()

        holdings = [
            Holding(
                ticker=ticker,
                name=data.get("name"),
                quantity=_f(data.get("quantity")) or 0.0,
                average_cost=_f(data.get("average_buy_price")) or 0.0,
                price=_f(data.get("price")) or 0.0,
                pe_ratio=_f(data.get("pe_ratio")),
            )
            for ticker, data in build_holdings().items()
        ]

        profile = load_account_profile() or {}
        account = AccountSummary(
            cash=_f(profile.get("cash")) or 0.0,
            buying_power=_f(profile.get("buying_power")),
        )

        return PortfolioSnapshot(
            source=self.name,
            taken_at=datetime.now(),
            account=account,
            holdings=holdings,
            transactions=self._fetch_transactions(),
            options=self._fetch_options(),
        )

    def _fetch_options(self) -> list[OptionPosition]:
        from robin_stocks.robinhood import (
            get_open_option_positions, get_option_instrument_data_by_id,
            get_option_market_data_by_id,
        )

        positions: list[OptionPosition] = []
        for pos in get_open_option_positions() or []:
            option_id = pos["option"].rstrip("/").split("/")[-1]
            inst = get_option_instrument_data_by_id(option_id) or {}
            market = get_option_market_data_by_id(option_id) or [{}]
            market = market[0] if isinstance(market, list) else market
            contracts = _f(pos.get("quantity")) or 0.0
            positions.append(OptionPosition(
                underlying=pos.get("chain_symbol", ""),
                option_type=inst.get("type", ""),
                strike=_f(inst.get("strike_price")) or 0.0,
                expiration=datetime.strptime(inst["expiration_date"], "%Y-%m-%d").date(),
                contracts=-contracts if pos.get("type") == "short" else contracts,
                average_cost=abs(_f(pos.get("average_price")) or 0.0),
                mark_price=_f(market.get("adjusted_mark_price")),
                implied_volatility=_f(market.get("implied_volatility")),
                delta=_f(market.get("delta")),
                theta=_f(market.get("theta")),
            ))
        return positions

    def _fetch_transactions(self) -> list[Transaction]:
        from robin_stocks.robinhood import (
            get_all_stock_orders, get_bank_transfers, get_dividends, get_symbol_by_url,
        )

        symbols: dict[str, str] = {}

        def symbol(url: str) -> str:
            if url not in symbols:
                symbols[url] = get_symbol_by_url(url)
            return symbols[url]

        txns: list[Transaction] = []
        for order in get_all_stock_orders() or []:
            if order.get("state") != "filled":
                continue
            qty = _f(order.get("cumulative_quantity")) or 0.0
            price = _f(order.get("average_price")) or 0.0
            side = order.get("side")
            txns.append(Transaction(
                ticker=symbol(order["instrument"]),
                kind=side,
                date=market_date(order["last_transaction_at"]),
                quantity=qty,
                price=price,
                amount=-qty * price if side == "buy" else qty * price,
                ref=order["id"],
            ))

        for div in get_dividends() or []:
            if div.get("state") != "paid":
                continue
            txns.append(Transaction(
                ticker=symbol(div["instrument"]),
                kind="dividend",
                date=market_date(div["paid_at"]),
                quantity=_f(div.get("position")) or 0.0,
                price=_f(div.get("rate")) or 0.0,
                amount=_f(div.get("amount")) or 0.0,
                ref=div["id"],
            ))

        # Transfers started in the app, plus ones pulled from the bank side
        for transfer in (get_bank_transfers() or []) + (get_bank_transfers("received") or []):
            if transfer.get("state") != "completed":
                continue
            amount = transfer.get("amount")
            amount = _f(amount.get("amount") if isinstance(amount, dict) else amount) or 0.0
            deposit = transfer.get("direction") == "deposit"
            txns.append(Transaction(
                ticker="",
                kind="deposit" if deposit else "withdrawal",
                date=market_date(transfer["created_at"]),
                quantity=0.0,
                price=0.0,
                amount=amount if deposit else -amount,
                ref=transfer["id"],
            ))

        # Order history keeps pre-split share counts and prices
        bought = {t.ticker for t in txns if t.kind == "buy"}
        txns += split_transactions(txns, stock_splits(bought))
        txns.sort(key=lambda t: t.date)
        return txns
