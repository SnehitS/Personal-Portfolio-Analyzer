"""SQLite snapshot store: one row per run, one row per holding per run."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, fields
from datetime import date, datetime
from pathlib import Path

from src.data_pipeline.models import (
    AccountSummary, Holding, OptionPosition, PortfolioSnapshot, Transaction,
)

DEFAULT_DB = Path("data/processed/portfolio.db")

_HOLDING_COLS = [f.name for f in fields(Holding)]
_OPTION_COLS = [f.name for f in fields(OptionPosition)]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at TEXT NOT NULL,
    source TEXT NOT NULL,
    cash REAL NOT NULL,
    cash_is_estimate INTEGER NOT NULL,
    buying_power REAL,
    holdings_value REAL NOT NULL,
    total_equity REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS holdings (
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
    {", ".join(_HOLDING_COLS)},
    market_value REAL,
    unrealized_pl REAL
);
CREATE TABLE IF NOT EXISTS option_positions (
    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id),
    {", ".join(_OPTION_COLS)},
    market_value REAL
);
CREATE TABLE IF NOT EXISTS transactions (
    ticker TEXT, kind TEXT, date TEXT, quantity REAL, price REAL, amount REAL,
    UNIQUE (ticker, kind, date, quantity, price, amount)
);
"""


def connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _add_missing_columns(conn, "holdings", _HOLDING_COLS)
    _add_missing_columns(conn, "option_positions", _OPTION_COLS)
    return conn


def _add_missing_columns(conn: sqlite3.Connection, table: str, columns: list[str]) -> None:
    """Keep old databases usable when new fields are added to the models."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for col in columns:
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col}")


def save_snapshot(snap: PortfolioSnapshot, db_path: Path = DEFAULT_DB) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO snapshots (taken_at, source, cash, cash_is_estimate, buying_power,"
            " holdings_value, total_equity) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (snap.taken_at.isoformat(timespec="seconds"), snap.source, snap.account.cash,
             int(snap.account.cash_is_estimate), snap.account.buying_power,
             snap.holdings_value, snap.total_equity),
        )
        snapshot_id = cur.lastrowid

        placeholders = ", ".join("?" * (len(_HOLDING_COLS) + 3))
        conn.executemany(
            f"INSERT INTO holdings (snapshot_id, {', '.join(_HOLDING_COLS)}, market_value,"
            f" unrealized_pl) VALUES ({placeholders})",
            [
                (snapshot_id, *_sql_values(asdict(h)), h.market_value, h.unrealized_pl)
                for h in snap.holdings
            ],
        )
        conn.executemany(
            f"INSERT INTO option_positions (snapshot_id, {', '.join(_OPTION_COLS)}, market_value)"
            f" VALUES ({', '.join('?' * (len(_OPTION_COLS) + 2))})",
            [(snapshot_id, *_sql_values(asdict(o)), o.market_value) for o in snap.options],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO transactions VALUES (?, ?, ?, ?, ?, ?)",
            [(t.ticker, t.kind, t.date.isoformat(), t.quantity, t.price, t.amount)
             for t in snap.transactions],
        )
    return snapshot_id


def list_snapshots(limit: int = 30, db_path: Path = DEFAULT_DB) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM snapshots ORDER BY taken_at DESC LIMIT ?", (limit,)
        ).fetchall()


def load_latest_snapshot(db_path: Path = DEFAULT_DB) -> PortfolioSnapshot | None:
    with connect(db_path) as conn:
        snap = conn.execute("SELECT * FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
        if snap is None:
            return None
        holdings = [_from_row(Holding, r) for r in conn.execute(
            "SELECT * FROM holdings WHERE snapshot_id = ?", (snap["id"],))]
        options = [_from_row(OptionPosition, r) for r in conn.execute(
            "SELECT * FROM option_positions WHERE snapshot_id = ?", (snap["id"],))]
        txns = [Transaction(r["ticker"], r["kind"], date.fromisoformat(r["date"]),
                            r["quantity"], r["price"], r["amount"])
                for r in conn.execute("SELECT * FROM transactions ORDER BY date")]
    return PortfolioSnapshot(
        source=snap["source"],
        taken_at=datetime.fromisoformat(snap["taken_at"]),
        account=AccountSummary(snap["cash"], snap["buying_power"], bool(snap["cash_is_estimate"])),
        holdings=holdings,
        transactions=txns,
        options=options,
    )


def _from_row(cls, row: sqlite3.Row):
    kwargs = {}
    for f in fields(cls):
        value = row[f.name] if f.name in row.keys() else None
        if value is not None and "date" in str(f.type) and isinstance(value, str):
            value = date.fromisoformat(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _sql_values(row: dict) -> list:
    return [v.isoformat() if hasattr(v, "isoformat") else v for v in row.values()]
