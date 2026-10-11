"""
Release metadata for a FRED series: when its newest value was published and
when the next release is scheduled.

Not exposed to the model as its own tool: the agent's series tool attaches
this to its results so the model never has to guess release timing from
observation dates (UNRATE dated 2026-08-01 is August data, published
2026-09-04).

Latency: each piece is cached for as long as it stays true, and whatever is
missing is fetched in parallel, so a lookup costs at most two FRED round
trips (first time a series is seen), usually one, and none once cached:

  series -> release          permanent            (series_release)
  observation -> published   permanent            (first_release_dates)
  release -> next date       until that date passes, or CACHE_TTL_FRED
                                                   (release_schedule)
"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import config
from data_fetchers import fred_fetcher
from database import utc_now_iso
from models.schemas import ReleaseInfo


def _schedule_is_fresh(row, today: date, ttl_seconds: int) -> bool:
    if row is None:
        return False
    if row["next_release_date"] is not None and row["next_release_date"] < today.isoformat():
        return False  # that release has happened: a newer value (and a new next date) exists
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(row["fetch_timestamp"])).total_seconds()
    return age < ttl_seconds


def get_release_info(
    series_id: str,
    latest_observation: str,
    db_connection: sqlite3.Connection,
    today: date | None = None,
    cache_ttl_seconds: int = config.CACHE_TTL_FRED,
    http_timeout: float | None = None,
) -> ReleaseInfo:
    """
    Release info for series_id's latest observation. Fetcher errors propagate.
    http_timeout bounds each FRED request (callers on a time budget pass it).

    Network work runs in worker threads that only call the fetchers; all
    SQLite reads and writes stay on the calling thread.
    """
    today = today or date.today()
    db = db_connection

    release = db.execute("SELECT release_id, release_name FROM series_release WHERE series_id = ?",
                         (series_id,)).fetchone()
    first = db.execute("SELECT released_on FROM first_release_dates WHERE series_id = ? AND observation_date = ?",
                       (series_id, latest_observation)).fetchone()
    schedule = None
    if release:
        schedule = db.execute("SELECT next_release_date, fetch_timestamp FROM release_schedule WHERE release_id = ?",
                              (release["release_id"],)).fetchone()
    need_schedule = not _schedule_is_fresh(schedule, today, cache_ttl_seconds)
    all_cached = release is not None and first is not None and not need_schedule

    if not all_cached:
        def release_and_schedule():
            rel = dict(release) if release else fred_fetcher.fetch_series_release(series_id, timeout=http_timeout)
            next_date = (fred_fetcher.fetch_next_release_date(rel["release_id"], today.isoformat(), timeout=http_timeout)
                         if need_schedule else schedule["next_release_date"])
            return rel, next_date

        def first_release():
            return fred_fetcher.fetch_first_release_date(series_id, latest_observation, timeout=http_timeout)

        with ThreadPoolExecutor(max_workers=2) as pool:
            schedule_job = pool.submit(release_and_schedule)
            first_job = pool.submit(first_release) if first is None else None
            rel, next_date = schedule_job.result()
            released_on = first_job.result() if first_job else first["released_on"]

        now = utc_now_iso()
        db.execute("INSERT OR IGNORE INTO series_release (series_id, release_id, release_name) VALUES (?, ?, ?)",
                   (series_id, rel["release_id"], rel["release_name"]))
        if need_schedule:
            db.execute(
                """INSERT INTO release_schedule (release_id, next_release_date, fetch_timestamp) VALUES (?, ?, ?)
                   ON CONFLICT (release_id) DO UPDATE SET next_release_date = excluded.next_release_date,
                                                          fetch_timestamp = excluded.fetch_timestamp""",
                (rel["release_id"], next_date, now))
        if first is None and released_on:
            db.execute("INSERT OR IGNORE INTO first_release_dates (series_id, observation_date, released_on) "
                       "VALUES (?, ?, ?)", (series_id, latest_observation, released_on))
        db.commit()
        return ReleaseInfo(series_id=series_id, release_name=rel["release_name"],
                           latest_observation=latest_observation, latest_released_on=released_on,
                           next_release_date=next_date, from_cache=False)

    return ReleaseInfo(series_id=series_id, release_name=release["release_name"],
                       latest_observation=latest_observation, latest_released_on=first["released_on"],
                       next_release_date=schedule["next_release_date"], from_cache=True)
