"""Tool 2: get_stock_data — yfinance OHLCV with a SQLite cache in front."""

import sqlite3

import config
from data_fetchers import yfinance_fetcher
from database import (
    get_stock_cache_status,
    initialize_database,
    record_fetch,
    store_stock_data,
    utc_now_iso,
)
from models.schemas import StockDataOutput


def _default_ttl(interval: str) -> int:
    """Intraday bars go stale fast; daily-or-longer bars only once a day."""
    if interval in yfinance_fetcher.INTRADAY_INTERVALS:
        return config.CACHE_TTL_STOCK
    return config.CACHE_TTL_STOCK_DAILY


def get_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    db_connection: sqlite3.Connection | None = None,
    cache_ttl_seconds: int | None = None,
) -> StockDataOutput:
    """
    Return OHLCV bars for ticker in [start_date, end_date] (inclusive).

    cache_ttl_seconds=None picks a TTL from the interval (1h intraday, 24h
    otherwise). db_connection=None opens the default DB from config.
    """
    if db_connection is None:
        db_connection = initialize_database(config.DB_PATH)
    if cache_ttl_seconds is None:
        cache_ttl_seconds = _default_ttl(interval)

    cache = get_stock_cache_status(db_connection, ticker, interval, cache_ttl_seconds, start_date, end_date)
    if cache["cached"]:
        return StockDataOutput(
            ticker=ticker,
            interval=interval,
            data=cache["data"],
            source="yfinance",
            fetched_at=cache["fetched_at"],
            from_cache=True,
        )

    data = yfinance_fetcher.fetch_stock_data(ticker, start_date, end_date, interval)
    fetched_at = utc_now_iso()
    store_stock_data(db_connection, ticker, data, interval=interval, fetch_timestamp=fetched_at)
    record_fetch(db_connection, "stock", f"{ticker}|{interval}", start_date, end_date, fetched_at)

    return StockDataOutput(
        ticker=ticker,
        interval=interval,
        data=data,
        source="yfinance",
        fetched_at=fetched_at,
        from_cache=False,
    )
