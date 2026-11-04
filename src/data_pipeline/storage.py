"""SQLite snapshot store: one row per run, one row per holding per run."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, fields
from datetime import date, datetime, timezone
from pathlib import Path

from src.data_pipeline.models import (
    AccountSummary, Holding, NewsItem, OptionPosition, PortfolioSnapshot, SentimentScore,
    Transaction,
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
    source TEXT NOT NULL, ref TEXT NOT NULL,
    ticker TEXT, kind TEXT, date TEXT, quantity REAL, price REAL, amount REAL,
    PRIMARY KEY (source, ref)
);
CREATE TABLE IF NOT EXISTS news (
    ticker TEXT NOT NULL, url TEXT NOT NULL,
    source TEXT, published_at TEXT, headline TEXT, summary TEXT, publisher TEXT,
    kind TEXT, form TEXT, provider_sentiment REAL, relevance REAL,
    fetched_at TEXT,
    PRIMARY KEY (ticker, url)
);
CREATE INDEX IF NOT EXISTS news_by_date ON news (ticker, published_at);
CREATE TABLE IF NOT EXISTS sentiment (
    ticker TEXT NOT NULL, url TEXT NOT NULL, model TEXT NOT NULL,
    label TEXT, score REAL, confidence REAL, rationale TEXT, scored_at TEXT,
    PRIMARY KEY (ticker, url, model)
);
"""


def connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _drop_legacy_transactions(conn)
    conn.executescript(SCHEMA)
    _add_missing_columns(conn, "holdings", _HOLDING_COLS)
    _add_missing_columns(conn, "option_positions", _OPTION_COLS)
    return conn


def _drop_legacy_transactions(conn: sqlite3.Connection) -> None:
    """The first transactions table had no source or ref, so rows from Robinhood and a CSV
    could both be kept for one trade. Every snapshot refetches full history, so drop it."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(transactions)")}
    if cols and "ref" not in cols:
        conn.execute("DROP TABLE transactions")


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
            "INSERT OR REPLACE INTO transactions (source, ref, ticker, kind, date, quantity,"
            " price, amount) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(snap.source, t.ref, t.ticker, t.kind, t.date.isoformat(), t.quantity, t.price,
              t.amount) for t in snap.transactions],
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
        # Only the snapshot's own source: the same trade can come from Robinhood and a CSV
        txns = [Transaction(r["ticker"], r["kind"], date.fromisoformat(r["date"]),
                            r["quantity"], r["price"], r["amount"], r["ref"])
                for r in conn.execute("SELECT * FROM transactions WHERE source = ?"
                                      " ORDER BY date, rowid", (snap["source"],))]
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


def _utc_iso(dt: datetime) -> str:
    """News times are stored as UTC ISO strings so they sort and compare as text.
    Naive datetimes are taken to be UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def save_news(items: list[NewsItem], db_path: Path = DEFAULT_DB) -> int:
    """Insert new items; ones already stored (same ticker and URL) are left alone.
    Returns how many were new."""
    now = datetime.now().isoformat(timespec="seconds")
    with connect(db_path) as conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO news (ticker, url, source, published_at, headline, summary,"
            " publisher, kind, form, provider_sentiment, relevance, fetched_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(n.ticker, n.url, n.source, _utc_iso(n.published_at),
              n.headline, n.summary, n.publisher, n.kind, n.form, n.provider_sentiment,
              n.relevance, now) for n in items],
        )
        return conn.total_changes - before


def load_news(tickers: list[str] | None = None, since: datetime | None = None,
              db_path: Path = DEFAULT_DB) -> list[NewsItem]:
    """Stored news, newest first, optionally for some tickers and after a time."""
    sql, params = "SELECT * FROM news WHERE 1=1", []
    if tickers:
        sql += f" AND ticker IN ({', '.join('?' * len(tickers))})"
        params += [t.upper() for t in tickers]
    if since:
        sql += " AND published_at >= ?"
        params.append(_utc_iso(since))
    with connect(db_path) as conn:
        rows = conn.execute(sql + " ORDER BY published_at DESC", params).fetchall()
    return [NewsItem(
        ticker=r["ticker"], source=r["source"],
        published_at=datetime.fromisoformat(r["published_at"]), headline=r["headline"],
        url=r["url"], summary=r["summary"] or "", publisher=r["publisher"] or "",
        kind=r["kind"] or "article", form=r["form"] or "",
        provider_sentiment=r["provider_sentiment"], relevance=r["relevance"],
    ) for r in rows]


def save_sentiment(scores: list[SentimentScore], db_path: Path = DEFAULT_DB) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO sentiment (ticker, url, model, label, score, confidence,"
            " rationale, scored_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(s.ticker, s.url, s.model, s.label, s.score, s.confidence, s.rationale, now)
             for s in scores],
        )


def load_sentiment(tickers: list[str] | None = None, model: str | None = None,
                   db_path: Path = DEFAULT_DB) -> list[SentimentScore]:
    sql, params = "SELECT * FROM sentiment WHERE 1=1", []
    if tickers:
        sql += f" AND ticker IN ({', '.join('?' * len(tickers))})"
        params += [t.upper() for t in tickers]
    if model:
        sql += " AND model = ?"
        params.append(model)
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [SentimentScore(r["ticker"], r["url"], r["model"], r["label"], r["score"],
                           r["confidence"], r["rationale"] or "") for r in rows]
