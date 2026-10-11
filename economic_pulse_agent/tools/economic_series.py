"""Tool 1: get_economic_series — FRED series with a SQLite cache in front."""

import sqlite3

import config
from data_fetchers import fred_fetcher
from database import get_cache_status, record_fetch, store_series_data, utc_now_iso
from models.schemas import EconomicSeriesOutput


def get_economic_series(
    series_id: str,
    start_date: str,
    end_date: str,
    db_connection: sqlite3.Connection,
    cache_ttl_seconds: int = config.CACHE_TTL_FRED,
) -> EconomicSeriesOutput:
    """
    Return observations of a FRED series in [start_date, end_date].

    Cache hit = a fetch covering this exact range (or a wider one) happened
    within cache_ttl_seconds. Otherwise hit FRED, upsert rows, log the fetch.
    Exceptions from the fetcher propagate unchanged.
    """
    cache = get_cache_status(db_connection, series_id, cache_ttl_seconds, start_date, end_date)
    if cache["cached"]:
        return EconomicSeriesOutput(
            series_id=series_id,
            data=cache["data"],
            source="FRED",
            fetched_at=cache["fetched_at"],
            from_cache=True,
        )

    data = fred_fetcher.fetch_fred_series(series_id, start_date, end_date)
    fetched_at = utc_now_iso()
    store_series_data(db_connection, series_id, data, "FRED", fetch_timestamp=fetched_at)
    record_fetch(db_connection, "series", series_id, start_date, end_date, fetched_at)

    return EconomicSeriesOutput(
        series_id=series_id,
        data=data,
        source="FRED",
        fetched_at=fetched_at,
        from_cache=False,
    )
