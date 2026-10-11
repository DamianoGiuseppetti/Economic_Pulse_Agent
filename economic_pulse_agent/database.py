"""
SQLite schema + persistence helpers.

Deviations from the original spec (all deliberate, see README "Design notes"):

1. Uniqueness is (series_id, date), not series_id alone. `series_id UNIQUE`
   would allow exactly one row per series.
2. stock_data carries an `interval` column and is unique on
   (ticker, interval, date), so hourly and daily bars never mix.
3. A `fetch_log` table records which (key, start, end) ranges were fetched
   and when. Freshness alone is not enough for a cache hit: after fetching
   UNRATE for Sept 2025, a request for Sept 2026 is "fresh" but not covered.
4. Upserts use SQLite's native INSERT ... ON CONFLICT DO UPDATE instead of
   catching IntegrityError and issuing a second statement.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS economic_series (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id       TEXT NOT NULL,
    date            TEXT NOT NULL,
    value           REAL NOT NULL,
    source          TEXT NOT NULL,
    fetch_timestamp TEXT NOT NULL,
    UNIQUE (series_id, date)
);
CREATE INDEX IF NOT EXISTS idx_series_id_date ON economic_series (series_id, date);
CREATE INDEX IF NOT EXISTS idx_series_fetch_ts ON economic_series (fetch_timestamp);

CREATE TABLE IF NOT EXISTS stock_data (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    interval        TEXT NOT NULL,
    date            TEXT NOT NULL,
    open            REAL NOT NULL,
    high            REAL NOT NULL,
    low             REAL NOT NULL,
    close           REAL NOT NULL,
    volume          INTEGER NOT NULL,
    fetch_timestamp TEXT NOT NULL,
    UNIQUE (ticker, interval, date)
);
CREATE INDEX IF NOT EXISTS idx_stock_ticker_date ON stock_data (ticker, date);
CREATE INDEX IF NOT EXISTS idx_stock_fetch_ts ON stock_data (fetch_timestamp);

CREATE TABLE IF NOT EXISTS conversation_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT NOT NULL,
    turn_number        INTEGER NOT NULL,
    user_query         TEXT,
    agent_response     TEXT,
    macro_context_used TEXT,
    timestamp          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_session_turn ON conversation_log (session_id, turn_number);

-- One row per successful upstream fetch. kind = 'series' | 'stock'.
-- For stocks, key = "<ticker>|<interval>".
CREATE TABLE IF NOT EXISTS fetch_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT NOT NULL,
    key             TEXT NOT NULL,
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    fetch_timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fetch_log_key ON fetch_log (kind, key, fetch_timestamp);

-- FRED release metadata, split by how long each piece stays true:
--   series_release       which release a series belongs to (permanent)
--   first_release_dates  when an observation was first published (permanent)
--   release_schedule     next scheduled date of a release (stale once that date passes)
CREATE TABLE IF NOT EXISTS series_release (
    series_id    TEXT PRIMARY KEY,
    release_id   INTEGER NOT NULL,
    release_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS first_release_dates (
    series_id        TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    released_on      TEXT NOT NULL,
    PRIMARY KEY (series_id, observation_date)
);
CREATE TABLE IF NOT EXISTS release_schedule (
    release_id        INTEGER PRIMARY KEY,
    next_release_date TEXT,
    fetch_timestamp   TEXT NOT NULL
);

-- Whole fetch_macro_context results. key = JSON of (sorted keywords, start, end).
CREATE TABLE IF NOT EXISTS macro_context_cache (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    key             TEXT UNIQUE NOT NULL,
    payload         TEXT NOT NULL,
    fetch_timestamp TEXT NOT NULL
);
"""


def utc_now_iso() -> str:
    """Current UTC time as an ISO 8601 string with offset."""
    return datetime.now(timezone.utc).isoformat()


def initialize_database(db_path: str) -> sqlite3.Connection:
    """
    Create tables and indexes if they don't exist, return an open connection.

    Idempotent: safe to call on every startup. Schema errors propagate.
    """
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    connection.commit()
    return connection


# ---------------------------------------------------------------------------
# Cache bookkeeping
# ---------------------------------------------------------------------------

def _find_covering_fetch(
    connection: sqlite3.Connection,
    kind: str,
    key: str,
    start_date: str,
    end_date: str,
    ttl_seconds: int,
) -> dict:
    """
    Look for the most recent fetch of `key` whose range covers
    [start_date, end_date] and which is younger than ttl_seconds.

    Returns {"cached": bool, "age_seconds": int | None, "fetched_at": str | None}.
    """
    row = connection.execute(
        """
        SELECT fetch_timestamp FROM fetch_log
        WHERE kind = ? AND key = ? AND start_date <= ? AND end_date >= ?
        ORDER BY fetch_timestamp DESC LIMIT 1
        """,
        (kind, key, start_date, end_date),
    ).fetchone()
    if row is None:
        return {"cached": False, "age_seconds": None, "fetched_at": None}

    fetched_at = datetime.fromisoformat(row["fetch_timestamp"])
    age_seconds = int((datetime.now(timezone.utc) - fetched_at).total_seconds())
    return {"cached": age_seconds < ttl_seconds, "age_seconds": age_seconds, "fetched_at": row["fetch_timestamp"]}


def record_fetch(
    connection: sqlite3.Connection,
    kind: str,
    key: str,
    start_date: str,
    end_date: str,
    fetch_timestamp: str,
) -> None:
    """Remember that [start_date, end_date] for `key` was fetched at fetch_timestamp."""
    connection.execute(
        "INSERT INTO fetch_log (kind, key, start_date, end_date, fetch_timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        (kind, key, start_date, end_date, fetch_timestamp),
    )
    connection.commit()


def get_cache_status(
    connection: sqlite3.Connection,
    series_id: str,
    ttl_seconds: int,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Check whether series_id is cached for [start_date, end_date] within the TTL.

    Return shape:
      {"cached": bool, "data": list | None, "age_seconds": int | None,
       "fetched_at": str | None}
    `data` and `fetched_at` are only populated on a hit.
    """
    status = _find_covering_fetch(connection, "series", series_id, start_date, end_date, ttl_seconds)
    if not status["cached"]:
        return {"cached": False, "data": None, "age_seconds": status["age_seconds"], "fetched_at": None}
    return {
        "cached": True,
        "data": read_series_data(connection, series_id, start_date, end_date),
        "age_seconds": status["age_seconds"],
        "fetched_at": status["fetched_at"],
    }


def get_stock_cache_status(
    connection: sqlite3.Connection,
    ticker: str,
    interval: str,
    ttl_seconds: int,
    start_date: str,
    end_date: str,
) -> dict:
    """Same contract as get_cache_status, for stock bars."""
    key = f"{ticker}|{interval}"
    status = _find_covering_fetch(connection, "stock", key, start_date, end_date, ttl_seconds)
    if not status["cached"]:
        return {"cached": False, "data": None, "age_seconds": status["age_seconds"], "fetched_at": None}
    return {
        "cached": True,
        "data": read_stock_data(connection, ticker, interval, start_date, end_date),
        "age_seconds": status["age_seconds"],
        "fetched_at": status["fetched_at"],
    }


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def read_series_data(connection, series_id: str, start_date: str, end_date: str) -> list[dict]:
    """Rows for series_id with start_date <= date <= end_date, ascending."""
    rows = connection.execute(
        "SELECT date, value FROM economic_series "
        "WHERE series_id = ? AND date >= ? AND date <= ? ORDER BY date",
        (series_id, start_date, end_date),
    ).fetchall()
    return [{"date": r["date"], "value": r["value"]} for r in rows]


def read_stock_data(connection, ticker: str, interval: str, start_date: str, end_date: str) -> list[dict]:
    """
    Bars for ticker/interval inside [start_date, end_date], ascending.

    Intraday dates carry a time suffix ("2026-09-25T13:30:00"), so the upper
    bound compares against end_date + "~" ('~' sorts after any digit or 'T').
    """
    rows = connection.execute(
        "SELECT date, open, high, low, close, volume FROM stock_data "
        "WHERE ticker = ? AND interval = ? AND date >= ? AND date <= ? ORDER BY date",
        (ticker, interval, start_date, end_date + "~"),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def store_series_data(
    connection: sqlite3.Connection,
    series_id: str,
    data: list[dict],
    source: str,
    fetch_timestamp: str | None = None,
) -> int:
    """
    Upsert economic series rows. data: [{"date": "2023-01-01", "value": 3.7}, ...]

    (series_id, date) identifies an observation. On conflict the value is
    overwritten — FRED revises past data points, so the newest fetch wins.
    Returns the number of rows inserted or updated.
    """
    fetch_timestamp = fetch_timestamp or utc_now_iso()
    connection.executemany(
        """
        INSERT INTO economic_series (series_id, date, value, source, fetch_timestamp)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (series_id, date) DO UPDATE SET
            value = excluded.value,
            source = excluded.source,
            fetch_timestamp = excluded.fetch_timestamp
        """,
        [(series_id, row["date"], row["value"], source, fetch_timestamp) for row in data],
    )
    connection.commit()
    return len(data)


def store_stock_data(
    connection: sqlite3.Connection,
    ticker: str,
    data: list[dict],
    interval: str = "1d",
    fetch_timestamp: str | None = None,
) -> int:
    """
    Upsert OHLCV bars. data: [{"date": ..., "open": ..., "high": ..., "low": ...,
    "close": ..., "volume": ...}, ...]. Same overwrite-on-conflict policy as
    store_series_data (the latest bar of the day keeps changing until close).
    """
    fetch_timestamp = fetch_timestamp or utc_now_iso()
    connection.executemany(
        """
        INSERT INTO stock_data (ticker, interval, date, open, high, low, close, volume, fetch_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (ticker, interval, date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            fetch_timestamp = excluded.fetch_timestamp
        """,
        [
            (ticker, interval, r["date"], r["open"], r["high"], r["low"], r["close"], r["volume"], fetch_timestamp)
            for r in data
        ],
    )
    connection.commit()
    return len(data)
