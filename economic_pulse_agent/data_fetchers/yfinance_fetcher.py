"""
Low-level yfinance wrapper. No caching, no DB writes: fetch, parse, return.

Uses Ticker.history() rather than yf.download(): download() returns
MultiIndex columns in recent yfinance versions and swallows per-ticker errors
into a printed message, which is the opposite of failing loudly.
"""

import math
from datetime import date, timedelta

import yfinance as yf
from curl_cffi.requests.exceptions import HTTPError as CurlHTTPError
from curl_cffi.requests.exceptions import RequestException as CurlRequestException
from yfinance.exceptions import YFException

from resilience import retry_call

# By default yfinance prints errors and returns an empty DataFrame. Make it raise.
yf.config.debug.hide_exceptions = False

INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


def fetch_stock_data(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> list[dict]:
    """
    Fetch OHLCV bars for ticker between start_date and end_date (both inclusive).

    Returns [{"date": "2023-01-03", "open": ..., "high": ..., "low": ...,
              "close": ..., "volume": int}, ...] sorted ascending.
    Daily-or-longer bars use "YYYY-MM-DD"; intraday bars use
    "YYYY-MM-DDTHH:MM:SS" in UTC.

    Raises:
      ValueError      bad dates, or no data for ticker (unknown/delisted)
      ConnectionError network failure
    """
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except (TypeError, ValueError):
        raise ValueError(f"Dates must be ISO (YYYY-MM-DD), got {start_date!r}, {end_date!r}") from None
    if start > end:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}")

    # yfinance's `end` is exclusive; add a day so our API is inclusive like FRED's.
    def attempt(timeout: float):
        try:
            return yf.Ticker(ticker).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval=interval,
                auto_adjust=False,
                timeout=timeout,
            )
        except CurlHTTPError as exc:
            # Yahoo answers 404 for symbols it doesn't know.
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                raise ValueError(f"Ticker {ticker} not found") from exc
            raise ConnectionError(f"Yahoo returned HTTP {status} for {ticker}: {exc}") from exc
        except CurlRequestException as exc:
            raise ConnectionError(f"yfinance request failed for {ticker}: {exc}") from exc
        except YFException as exc:
            # YFTickerMissingError, YFPricesMissingError, YFInvalidPeriodError, ...
            raise ValueError(f"Ticker {ticker} not found ({type(exc).__name__}: {exc})") from exc

    df = retry_call(attempt, source="yfinance")

    if df.empty:
        raise ValueError(
            f"Ticker {ticker} not found, or no trading days between {start_date} and {end_date}"
        )

    intraday = interval in INTRADAY_INTERVALS
    if intraday:
        df.index = df.index.tz_convert("UTC")

    data = []
    for ts, row in df.iterrows():
        # Skip bars yfinance returns with NaN prices (seen on holidays/partial days).
        if any(math.isnan(row[c]) for c in ("Open", "High", "Low", "Close")):
            continue
        data.append({
            "date": ts.strftime("%Y-%m-%dT%H:%M:%S") if intraday else ts.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": 0 if math.isnan(row["Volume"]) else int(row["Volume"]),
        })
    return sorted(data, key=lambda r: r["date"])
