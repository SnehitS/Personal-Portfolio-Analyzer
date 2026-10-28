"""Robinhood activity CSV fallback.

Export from Robinhood: Account -> Reports & statements -> Reports -> generate an
account activity CSV covering your full history. The CSV only has transactions,
so positions are rebuilt from them and prices come from market_data enrichment.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.data_pipeline.models import AccountSummary, Holding, PortfolioSnapshot, Transaction
from src.data_pipeline.sources.base import PortfolioSource

TRANS_CODES = {"Buy": "buy", "Sell": "sell", "CDIV": "dividend", "SPL": "split"}


def parse_money(raw: str) -> float:
    """'$1,234.56' -> 1234.56, '($1,234.56)' -> -1234.56, '' -> 0.0"""
    raw = (raw or "").strip()
    if not raw:
        return 0.0
    negative = raw.startswith("(") and raw.endswith(")")
    value = float(raw.strip("()").replace("$", "").replace(",", ""))
    return -value if negative else value


def parse_quantity(raw: str) -> float:
    # Some rows suffix quantities, e.g. "0.5S" for share amounts
    raw = (raw or "").strip().rstrip("S")
    return float(raw.replace(",", "")) if raw else 0.0


def read_activity_csv(path: Path) -> tuple[list[Transaction], float]:
    """Return (stock transactions, net cash from every row)."""
    txns: list[Transaction] = []
    net_cash = 0.0
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            code = (row.get("Trans Code") or "").strip()
            if not code:
                continue  # blank rows and the trailing disclaimer
            amount = parse_money(row.get("Amount", ""))
            net_cash += amount
            kind = TRANS_CODES.get(code)
            if kind is None:
                continue
            txns.append(Transaction(
                ticker=row["Instrument"].strip(),
                kind=kind,
                date=datetime.strptime(row["Activity Date"].strip(), "%m/%d/%Y").date(),
                quantity=parse_quantity(row.get("Quantity", "")),
                price=parse_money(row.get("Price", "")),
                amount=amount,
            ))
    txns.sort(key=lambda t: t.date)
    return txns, net_cash


def rebuild_positions(txns: list[Transaction]) -> list[Holding]:
    """Average-cost positions (the method Robinhood displays) from transactions."""
    qty: dict[str, float] = defaultdict(float)
    cost: dict[str, float] = defaultdict(float)
    for t in txns:
        if t.kind == "buy":
            qty[t.ticker] += t.quantity
            cost[t.ticker] += t.quantity * t.price
        elif t.kind == "sell" and qty[t.ticker] > 0:
            avg = cost[t.ticker] / qty[t.ticker]
            qty[t.ticker] -= t.quantity
            cost[t.ticker] -= t.quantity * avg
        elif t.kind == "split":
            qty[t.ticker] += t.quantity  # split rows list the added shares; cost unchanged

    return [
        Holding(ticker=ticker, quantity=q, average_cost=cost[ticker] / q, price=0.0)
        for ticker, q in sorted(qty.items())
        if q > 1e-6
    ]


class CsvSource(PortfolioSource):
    name = "csv"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fetch_snapshot(self) -> PortfolioSnapshot:
        txns, net_cash = read_activity_csv(self.path)
        return PortfolioSnapshot(
            source=self.name,
            taken_at=datetime.now(),
            account=AccountSummary(cash=net_cash, cash_is_estimate=True),
            holdings=rebuild_positions(txns),
            transactions=txns,
        )
