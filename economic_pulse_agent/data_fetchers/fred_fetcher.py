"""
Low-level FRED API wrapper. No caching, no DB writes: fetch, parse, return.

Talks to the REST API directly with `requests` rather than through `fredapi`,
so each failure mode maps to a specific exception instead of fredapi's
catch-all ValueError.
"""

from datetime import date

import requests

import config
from resilience import retry_call


def _validate_date(value: str, name: str) -> None:
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD), got {value!r}") from None


def fetch_fred_series(series_id: str, start_date: str, end_date: str) -> list[dict]:
    """
    Fetch observations for series_id between start_date and end_date (inclusive).

    Returns [{"date": "2023-01-01", "value": 3.7}, ...] sorted ascending.
    FRED marks missing observations with "." (e.g. the October 2025 CPS gap
    during the government shutdown); those are dropped, not stored as NaN.

    Raises:
      RuntimeError    FRED_API_KEY not configured
      ValueError      bad dates, or series not found
      ConnectionError network failure or unexpected HTTP error (no retry)
    """
    if not config.FRED_API_KEY:
        raise RuntimeError(
            "FRED_API_KEY is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and put it in .env"
        )
    _validate_date(start_date, "start_date")
    _validate_date(end_date, "end_date")
    if start_date > end_date:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}")

    params = {
        "series_id": series_id,
        "observation_start": start_date,
        "observation_end": end_date,
        "api_key": config.FRED_API_KEY,
        "file_type": "json",
    }
    def attempt(timeout: float):
        try:
            response = requests.get(
                f"{config.FRED_BASE_URL}/series/observations",
                params=params,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise ConnectionError(f"FRED request failed for {series_id}: {exc}") from exc

        if response.status_code != 200:
            # FRED reports errors as {"error_code": 400, "error_message": "..."}.
            try:
                message = response.json().get("error_message", response.text)
            except ValueError:
                message = response.text
            if "api_key" in message:
                raise RuntimeError(f"FRED rejected the API key: {message}")
            if response.status_code in (400, 404) and "does not exist" in message:
                raise ValueError(f"Series {series_id} not found")
            if response.status_code == 400:
                raise ValueError(f"FRED rejected request for {series_id}: {message}")
            raise ConnectionError(f"FRED returned HTTP {response.status_code} for {series_id}: {message}")
        return response

    response = retry_call(attempt, source="fred")

    observations = response.json()["observations"]
    data = [
        {"date": obs["date"], "value": float(obs["value"])}
        for obs in observations
        if obs["value"] != "."
    ]
    return sorted(data, key=lambda row: row["date"])


def _fred_get(path: str, timeout: float | None = None, **params) -> dict:
    """GET a FRED endpoint; same error mapping as fetch_fred_series. timeout=None uses the config default."""
    if not config.FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is not set")
    def attempt(request_timeout: float):
        try:
            response = requests.get(
                f"{config.FRED_BASE_URL}/{path}",
                params={**params, "api_key": config.FRED_API_KEY, "file_type": "json"},
                timeout=request_timeout,
            )
        except requests.RequestException as exc:
            raise ConnectionError(f"FRED request failed ({path}): {exc}") from exc
        if response.status_code != 200:
            try:
                message = response.json().get("error_message", response.text)
            except ValueError:
                message = response.text
            if response.status_code == 400:
                raise ValueError(f"FRED rejected {path} request: {message}")
            raise ConnectionError(f"FRED returned HTTP {response.status_code} for {path}: {message}")
        return response

    response = retry_call(attempt, source="fred", max_timeout=timeout)
    return response.json()


def fetch_first_release_date(series_id: str, observation_date: str, timeout: float | None = None) -> str | None:
    """
    Date the observation for `observation_date` was first published
    (e.g. UNRATE 2026-08-01 -> "2026-09-04"). None if FRED has no record.

    Uses output_type=4 ("initial release only"). The real-time window starts at
    the observation date to stay under FRED's 2000-vintage cap on daily series.
    """
    body = _fred_get(
        "series/observations",
        timeout=timeout,
        series_id=series_id,
        observation_start=observation_date,
        observation_end=observation_date,
        output_type=4,
        realtime_start=observation_date,
        realtime_end="9999-12-31",
    )
    observations = body.get("observations", [])
    return observations[0]["realtime_start"] if observations else None


def fetch_series_release(series_id: str, timeout: float | None = None) -> dict:
    """
    The FRED release a series belongs to: {"release_id": 50, "release_name": "Employment Situation"}.
    Never changes for a series, so callers can store it permanently.
    """
    releases = _fred_get("series/release", timeout=timeout, series_id=series_id).get("releases", [])
    if not releases:
        raise ValueError(f"No FRED release found for {series_id}")
    return {"release_id": releases[0]["id"], "release_name": releases[0]["name"]}


def fetch_next_release_date(release_id: int, on_or_after: str, timeout: float | None = None) -> str | None:
    """First scheduled date of a release on or after `on_or_after` (FRED publishes the calendar ahead)."""
    dates = _fred_get(
        "release/dates",
        timeout=timeout,
        release_id=release_id,
        realtime_start=on_or_after,
        include_release_dates_with_no_data="true",
        sort_order="asc",
        limit=1,
    ).get("release_dates", [])
    return dates[0]["date"] if dates else None
