"""
Tool definitions (LangChain StructuredTools) and the local executor that runs them.

Each tool's input is a Pydantic model. That model is the single source of
truth: it is the tool's `args_schema` (so its JSON Schema is what the model
sees, and LangChain validates what Claude sends against it before our code
runs).

execute_tool() is the error boundary Phase 1 deliberately didn't have: every
exception becomes a plain-text message written for the model (what went wrong
+ what to try instead), so the agent never crashes. The tool functions return
that text rather than raising. LangChain's string-in/string-out tool contract
has no is_error flag, so the structured outcome (is_error, from_cache,
error_type) goes through a thread-local side channel that the metrics
callback handler reads right after the tool finishes.
"""

import asyncio
import dataclasses
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

import config
from approval import ApprovalGate
from database import get_cache_status, get_stock_cache_status, initialize_database
from resilience import SourceUnavailableError
from resilience import call_state as _call  # per-tool-call deadline and stats, set in execute_tool
from resilience import seconds_left as _seconds_left
from tools.anomaly import SERIES_RULES, detect_anomalies, rule_for
from tools.compare_periods import compare_periods
from tools.economic_series import get_economic_series
from tools.macro_context import fetch_macro_context
from tools.narrative import build_narrative
from tools.release_info import get_release_info
from tools.stock_data import get_stock_data

TOOL_TIMEOUT_SECONDS = 10
# Tool results are re-sent on every loop iteration, so long series are
# downsampled. 120 points is ten years of monthly data.
MAX_POINTS = 120

ISO_DATE = r"^\d{4}-\d{2}-\d{2}$"

# Offered to the model when it asks for a series or ticker that doesn't exist.
COMMON_SERIES = {
    "UNRATE": "Unemployment rate, %, monthly",
    "PAYEMS": "Nonfarm payrolls, thousands, monthly",
    "ICSA": "Initial jobless claims, weekly",
    "CPIAUCSL": "CPI all items, index (inflation = YoY % change), monthly",
    "CPILFESL": "Core CPI (ex food & energy), index, monthly",
    "PCEPILFE": "Core PCE price index, monthly",
    "FEDFUNDS": "Effective federal funds rate, %, monthly",
    "DGS10": "10-year Treasury yield, %, daily",
    "DGS2": "2-year Treasury yield, %, daily",
    "T10Y2Y": "10y minus 2y Treasury spread, %, daily",
    "A191RL1Q225SBEA": "Real GDP growth, annualized %, quarterly",
    "GDPC1": "Real GDP level, quarterly",
    "UMCSENT": "U. Michigan consumer sentiment, monthly",
    "MORTGAGE30US": "30-year mortgage rate, %, weekly",
}
COMMON_TICKERS = {
    "^GSPC": "S&P 500 index",
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq-100 ETF",
    "^IXIC": "Nasdaq Composite",
    "^DJI": "Dow Jones Industrial Average",
    "^VIX": "CBOE volatility index",
    "TLT": "20+ year Treasury bond ETF",
}


# ---------------------------------------------------------------------------
# Input models (one per tool)
# ---------------------------------------------------------------------------

class EconomicSeriesInput(BaseModel):
    series_id: str = Field(
        description="FRED series ID, uppercase letters and digits only, e.g. UNRATE, CPIAUCSL, FEDFUNDS, DGS10.",
        pattern=r"^[A-Z0-9]{1,25}$",
    )
    start_date: str = Field(description="First date to include, YYYY-MM-DD.", pattern=ISO_DATE)
    end_date: str = Field(description="Last date to include, YYYY-MM-DD (inclusive).", pattern=ISO_DATE)


class StockDataInput(BaseModel):
    ticker: str = Field(description="Yahoo Finance symbol, e.g. ^GSPC (S&P 500 index), SPY, QQQ, ^VIX, AAPL.")
    start_date: str = Field(description="First date to include, YYYY-MM-DD.", pattern=ISO_DATE)
    end_date: str = Field(description="Last date to include, YYYY-MM-DD (inclusive).", pattern=ISO_DATE)
    interval: Literal["1d", "1wk", "1mo"] = Field(
        default="1d",
        description="Bar size. Use 1d for ranges under ~6 months, 1wk up to ~3 years, 1mo beyond that.",
    )


class ComparePeriodsInput(BaseModel):
    series_id: str = Field(description="FRED series ID, e.g. UNRATE.", pattern=r"^[A-Z0-9]{1,25}$")
    p1_start: str = Field(description="Baseline period start, YYYY-MM-DD.", pattern=ISO_DATE)
    p1_end: str = Field(description="Baseline period end, YYYY-MM-DD.", pattern=ISO_DATE)
    p2_start: str = Field(description="Comparison period start, YYYY-MM-DD.", pattern=ISO_DATE)
    p2_end: str = Field(description="Comparison period end, YYYY-MM-DD.", pattern=ISO_DATE)
    agg_method: Literal["mean", "last", "first", "min", "max"] = Field(
        default="mean", description="How to collapse each period's observations into one number."
    )


class MacroContextInput(BaseModel):
    keywords: list[str] = Field(description="Topics, e.g. ['inflation', 'FOMC'].", min_length=1)
    start_date: str = Field(description="Window start, YYYY-MM-DD.", pattern=ISO_DATE)
    end_date: str = Field(description="Window end, YYYY-MM-DD.", pattern=ISO_DATE)


class DetectAnomaliesInput(BaseModel):
    series_id: str = Field(
        description="FRED series ID, uppercase letters and digits only. Supported: " + ", ".join(SERIES_RULES) + ".",
        pattern=r"^[A-Z0-9]{1,25}$",
    )
    start_date: str = Field(description="First date of the window to check, YYYY-MM-DD.", pattern=ISO_DATE)
    end_date: str = Field(description="Last date of the window to check, YYYY-MM-DD (inclusive).", pattern=ISO_DATE)
    baseline_years: int = Field(
        default=10, ge=3, le=20,
        description="Years of earlier history each observation is compared against. Leave at 10 unless the user asks otherwise.",
    )


class GenerateNarrativeInput(BaseModel):
    series_id: str = Field(
        description="FRED series ID, uppercase letters and digits only. Supported: " + ", ".join(SERIES_RULES) + ".",
        pattern=r"^[A-Z0-9]{1,25}$",
    )
    start_date: str = Field(description="First date of the window to describe, YYYY-MM-DD.", pattern=ISO_DATE)
    end_date: str = Field(description="Last date of the window to describe, YYYY-MM-DD (inclusive).", pattern=ISO_DATE)


def _schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON Schema, minus the cosmetic 'title' keys."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


TOOL_SPECS: dict[str, tuple[type[BaseModel], str]] = {
    "get_economic_series": (
        EconomicSeriesInput,
        "Fetch observations of a FRED economic series (unemployment, inflation indexes, rates, GDP, ...) "
        "for a date range. Returns a summary (first/latest observation, min, max) plus the data points "
        "(downsampled if long). FRED publishes with a lag: if the range extends past the latest release, "
        "the result says which observation is the most recent one available. Request the whole range "
        "you need in one call.",
    ),
    "get_stock_data": (
        StockDataInput,
        "Fetch daily/weekly/monthly price bars for a stock, ETF or index from Yahoo Finance. Returns "
        "first/latest close, % change over the range, period high/low, and a list of [date, close] pairs "
        "(downsampled if long). Dates are trading days; weekends and holidays have no bars.",
    ),
    "compare_periods": (
        ComparePeriodsInput,
        "Compare a FRED series between two date windows: aggregates each window (mean by default) and "
        "returns both values, the absolute delta, the % delta relative to the first window, and a "
        "direction (up/down/flat). Use it for 'how did X change from A to B' questions.",
    ),
    "fetch_macro_context": (
        MacroContextInput,
        "Market sentiment and Fed policy context for a date window. Returns a rule-based sentiment "
        "(bullish/neutral/bearish, with confidence) built from VIX, the high-yield credit spread and "
        "consumer sentiment, each with its level and change over the window; extra indicators chosen "
        "by keyword (inflation -> 5y breakeven inflation, Fed/rates -> 2y Treasury yield, jobs -> "
        "initial claims); and Fed monetary-policy releases in the window, with the decision "
        "paragraphs of FOMC statements (only for roughly the last six months). Use it for 'why did "
        "markets move' questions and whenever Fed decisions matter. It has no news headlines.",
    ),
    "detect_anomalies": (
        DetectAnomaliesInput,
        "Tell whether a FRED series did something unusual in a date window, measured against that "
        "series' own previous years of history. Use it when the question is whether something was "
        "unusual, abnormal or an outlier; NOT for the values themselves (use get_economic_series). "
        "It scores each observation's change (not its level) and returns, for the unusual ones, the "
        "date, size of the move in words, a robust z-score and a percentile, plus how many "
        "observations were checked (n_flagged of n_scored) and whether anything was unusual at all "
        "(any_unusual). Supports: " + ", ".join(SERIES_RULES) + ". It reports how unusual a move was; "
        "it does not say why it happened or what to do.",
    ),
    "generate_narrative": (
        GenerateNarrativeInput,
        "A short factual brief about ONE FRED series over a window, for open questions like 'what is going "
        "on with unemployment / CPI / the 10-year yield'. Returns 3 to 5 ready-made sentences (latest "
        "value and when it was published, the change over the window, year-over-year inflation for price "
        "indexes, how many observations were unusual) plus caveats. It contains no raw data points and no "
        "causes. Use get_economic_series when the user wants the values themselves, detect_anomalies when "
        "the question is only whether something was unusual, and this tool for a general 'what is happening "
        "with X'. Supports: " + ", ".join(SERIES_RULES) + ".",
    ),
}

# The Phase 2 (raw SDK) tool definitions. No longer sent anywhere; kept as the
# reference that the LangChain tools' schemas are checked against (Gate 9).
LEGACY_TOOL_DEFS: list[dict] = [
    {"name": name, "description": description, "input_schema": _schema(model)}
    for name, (model, description) in TOOL_SPECS.items()
]


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------

def _downsample(rows: list, limit: int = MAX_POINTS) -> list:
    """Evenly spaced subset of rows that always keeps the first and last."""
    n = len(rows)
    if n <= limit:
        return rows
    idx = sorted({round(i * (n - 1) / (limit - 1)) for i in range(limit)})
    return [rows[i] for i in idx]


# Price indexes: the level means little, the year-over-year % change is inflation.
PRICE_INDEXES = {"CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE"}


def _direction(delta: float) -> str:
    return "up" if delta > 1e-9 else "down" if delta < -1e-9 else "unchanged"


def _change(from_date: str, from_value: float, to_date: str, to_value: float, digits: int = 2) -> dict:
    """
    Change between two points, with the direction spelled out.

    The model copies numbers reliably but can slip on deriving them (calling
    4.3 -> 4.1 "up"), so direction and % are computed here.
    """
    delta = to_value - from_value
    pct = (delta / from_value * 100) if from_value else None
    direction = _direction(delta)
    size = f" {abs(delta):.{digits}f}" + (f" ({pct:+.2f}%)" if pct is not None else "") if direction != "unchanged" else ""
    return {
        "from": {"date": from_date, "value": round(from_value, 4)},
        "to": {"date": to_date, "value": round(to_value, 4)},
        "delta": round(delta, 4),
        "pct_change": round(pct, 2) if pct is not None else None,
        "direction": direction,
        "summary": f"{direction}{size} from {from_value:g} on {from_date} to {to_value:g} on {to_date}",
    }


def _latest_available(series_id: str, conn, today: date) -> dict | None:
    """Most recent published observation (looks back two years from today)."""
    lookback = get_economic_series(series_id, (today - timedelta(days=730)).isoformat(), today.isoformat(), conn)
    if not lookback.data:
        return None
    last = lookback.data[-1]
    return {"date": last.date, "value": last.value}


def _release(series_id: str, latest_observation: str, conn, today: date) -> dict:
    """
    Release timing for the result. It's enrichment, so it must never sink the
    tool call: it gets whatever is left of the call's time budget (minus a
    margin), split across its up-to-3 FRED requests. A cached entry needs no
    network and returns immediately.
    """
    per_request = max(0.5, (_seconds_left() - 1.5) / 3)
    try:
        info = get_release_info(series_id, latest_observation, conn, today, http_timeout=per_request)
    except (ValueError, ConnectionError, RuntimeError) as exc:
        return {"unavailable": str(exc)}
    return {
        "release_name": info.release_name,
        "latest_observation": info.latest_observation,
        "latest_released_on": info.latest_released_on,
        "next_release_date": info.next_release_date,
    }


def _year_earlier(d: date) -> date:
    return d.replace(year=d.year - 1, day=min(d.day, 28)) if (d.month, d.day) == (2, 29) else d.replace(year=d.year - 1)


def _previous_month(d: date) -> date:
    return (d.replace(day=1) - timedelta(days=1)).replace(day=1)


def _rate_move(previous: float, current: float) -> str:
    """'up 0.05 pts from 3.30%': the model misread plain lists of rates as falling when they rose."""
    delta = round(current - previous, 2)
    if delta == 0:
        return f"unchanged from {previous:.2f}%"
    return f"{'up' if delta > 0 else 'down'} {abs(delta):.2f} pts from {previous:.2f}%"


def _history_start(start_date: str) -> str:
    """13 months before start: enough for a YoY rate on the first month and its previous-month comparison."""
    return _previous_month(_year_earlier(date.fromisoformat(start_date))).isoformat()


def _yoy_inflation(points: list[dict], history) -> dict | None:
    """
    Year-over-year % change (the inflation rate) for each monthly point, each
    with its move versus the previous month's rate spelled out.

    `history` must start at _history_start(first point) so the first point
    also gets a previous-month comparison. The caller fetches it together
    with the requested range: one FRED request instead of two.
    """
    by_date = {p.date: p.value for p in history}
    yoy_by_date = {}
    for p in history:
        prior = by_date.get(_year_earlier(date.fromisoformat(p.date)).isoformat())
        if prior:
            yoy_by_date[p.date] = round((p.value / prior - 1) * 100, 2)

    rows = []
    for p in points:
        if p["date"] not in yoy_by_date:
            continue
        previous_date = _previous_month(date.fromisoformat(p["date"])).isoformat()
        previous = yoy_by_date.get(previous_date)
        rows.append({
            "period": p["date"],
            "yoy_pct": yoy_by_date[p["date"]],
            "vs_previous_month": _rate_move(previous, yoy_by_date[p["date"]]) if previous is not None else None,
            "_previous": (previous_date, previous),
        })
    if not rows:
        return None

    latest = rows[-1]
    previous_date, previous = latest.pop("_previous")
    for row in rows:
        row.pop("_previous", None)
    result = {
        "latest_yoy_pct": latest["yoy_pct"],
        "latest_yoy_period": latest["period"],
        "yoy_by_month": _downsample(rows),
        "note": ("Year-over-year % change of the index = the inflation rate. Already computed, including "
                 "each month's move versus the previous month; quote these, don't recompute or re-compare."),
    }
    if previous is not None:
        delta = round(latest["yoy_pct"] - previous, 2)
        result["latest_vs_previous_month"] = {
            "previous_period": previous_date,
            "previous_yoy_pct": previous,
            "direction": _direction(delta),
            "summary": f"inflation {_rate_move(previous, latest['yoy_pct'])} ({previous_date}) "
                       f"to {latest['yoy_pct']:.2f}% ({latest['period']})",
        }
    return result


def _run_economic_series(args: EconomicSeriesInput, conn, today: date) -> tuple[dict, bool | None]:
    is_price_index = args.series_id in PRICE_INDEXES
    # Price indexes also need the prior 13 months for YoY inflation: fetch them in the same request.
    fetch_start = _history_start(args.start_date) if is_price_index else args.start_date
    out = get_economic_series(args.series_id, fetch_start, args.end_date, conn)
    points = [{"date": p.date, "value": p.value} for p in out.data if p.date >= args.start_date]
    result: dict = {"series_id": args.series_id, "requested": [args.start_date, args.end_date]}

    if not points:
        latest = _latest_available(args.series_id, conn, today)
        result["n_observations"] = 0
        result["note"] = (
            f"No {args.series_id} observations between {args.start_date} and {args.end_date}. "
            + (f"Latest published observation: {latest['date']} = {latest['value']}. " if latest else "")
            + "Data for that window has not been released yet or is outside the series' history."
        )
        if latest:
            result["release"] = _release(args.series_id, latest["date"], conn, today)
        return result, out.from_cache

    values = [p["value"] for p in points]
    low = min(points, key=lambda p: p["value"])
    high = max(points, key=lambda p: p["value"])
    result.update({
        "n_observations": len(points),
        "first": points[0],
        "latest": points[-1],
        "change": _change(points[0]["date"], points[0]["value"], points[-1]["date"], points[-1]["value"]),
        "min": low,
        "max": high,
    })
    if is_price_index:
        inflation = _yoy_inflation(points, out.data)
        if inflation:
            result["inflation"] = inflation
    # Last: the release lookup only gets the time the core data left over.
    result["release"] = _release(args.series_id, points[-1]["date"], conn, today)
    result["data"] = _downsample(points)
    if len(points) > MAX_POINTS:
        result["data_note"] = f"Downsampled from {len(points)} to {len(result['data'])} points."
    if args.end_date > points[-1]["date"] and args.end_date >= (today - timedelta(days=45)).isoformat():
        result["note"] = (
            f"Latest published observation is {points[-1]['date']}; later values are not released yet."
        )
    return result, out.from_cache


def _biggest_moves(bars) -> dict:
    """Largest peak-to-trough fall and trough-to-peak rise in closes, dates in chronological order."""
    peak = trough = bars[0]
    fall = {"pct": 0.0}
    rise = {"pct": 0.0}
    for bar in bars[1:]:
        if bar.close > peak.close:
            peak = bar
        if bar.close < trough.close:
            trough = bar
        drop = (bar.close / peak.close - 1) * 100
        gain = (bar.close / trough.close - 1) * 100
        if drop < fall["pct"]:
            fall = {"pct": round(drop, 2), "from": {"date": peak.date, "close": round(peak.close, 2)},
                    "to": {"date": bar.date, "close": round(bar.close, 2)}}
        if gain > rise["pct"]:
            rise = {"pct": round(gain, 2), "from": {"date": trough.date, "close": round(trough.close, 2)},
                    "to": {"date": bar.date, "close": round(bar.close, 2)}}
    for move, verb in ((fall, "fell"), (rise, "rose")):
        if "from" in move:
            move["summary"] = (f"{verb} {abs(move['pct']):.2f}% from {move['from']['close']:,} on "
                               f"{move['from']['date']} to {move['to']['close']:,} on {move['to']['date']}")
    return {"largest_decline": fall, "largest_rise": rise}


def _run_stock_data(args: StockDataInput, conn, today: date) -> tuple[dict, bool | None]:
    out = get_stock_data(args.ticker, args.start_date, args.end_date, interval=args.interval, db_connection=conn)
    bars = out.data
    closes = [[b.date, round(b.close, 2)] for b in bars]
    result = {
        "ticker": args.ticker,
        "interval": args.interval,
        "n_bars": len(bars),
        "first": {"date": bars[0].date, "close": round(bars[0].close, 2)},
        "latest": {"date": bars[-1].date, "close": round(bars[-1].close, 2)},
        "change": _change(bars[0].date, round(bars[0].close, 2), bars[-1].date, round(bars[-1].close, 2)),
        "period_high": {"date": max(bars, key=lambda b: b.high).date, "high": round(max(b.high for b in bars), 2)},
        "period_low": {"date": min(bars, key=lambda b: b.low).date, "low": round(min(b.low for b in bars), 2)},
        **_biggest_moves(bars),
        "closes": _downsample(closes),
        "note": "closes are in chronological order (oldest first).",
    }
    if len(closes) > MAX_POINTS:
        result["data_note"] = f"Downsampled from {len(closes)} to {len(result['closes'])} bars; use a coarser interval for long ranges."
    return result, out.from_cache


def _run_compare_periods(args: ComparePeriodsInput, conn, today: date) -> tuple[dict, bool | None]:
    out = compare_periods(
        args.series_id,
        {"start": args.p1_start, "end": args.p1_end},
        {"start": args.p2_start, "end": args.p2_end},
        agg_method=args.agg_method,
        db_connection=conn,
    )
    result = out.model_dump()
    result["summary"] = (
        f"{out.series_id} ({agg_method_label(out.agg_method)}) "
        f"{'was flat' if out.direction == 'flat' else 'went ' + out.direction}: "
        f"{out.period_1.aggregated_value:g} in {out.period_1.start}..{out.period_1.end} vs "
        f"{out.period_2.aggregated_value:g} in {out.period_2.start}..{out.period_2.end} "
        f"(delta {out.delta:+.2f}, {out.delta_pct:+.2f}%)"
    )
    return result, None


def agg_method_label(method: str) -> str:
    return {"mean": "average", "last": "last value", "first": "first value", "min": "minimum", "max": "maximum"}[method]


def _run_macro_context(args: MacroContextInput, conn, today: date) -> tuple[dict, bool | None]:
    out = fetch_macro_context(args.keywords, {"start": args.start_date, "end": args.end_date},
                              db_connection=conn, today=today)
    result = out.model_dump(exclude={"method", "fetched_at", "from_cache"})
    result["method"] = "rule-based composite of market indicators, not news analysis"
    return result, out.from_cache


def _signed_words(change: float, unit: str, period: str, transform: str) -> str:
    """The scored change in words, direction spelled out ('up 0.3 pts'): the model misreads signs."""
    size = round(abs(change), 2)
    if transform == "level":
        return "was flat at 0% annualized" if size == 0 else f"{'grew' if change > 0 else 'shrank'} {size:g}% annualized"
    if size == 0:
        return f"unchanged from the previous {period}"
    suffix = f"% from the previous {period}" if transform == "pct" else f" {unit}"
    return f"{_direction(change)} {size:g}{suffix}"


def _anomaly_point(p, out) -> dict:
    return {
        "date": p.date,
        "level": p.level,
        "change_words": _signed_words(p.change, out.unit, out.period, out.transform),
        "z": p.z,
        "percentile": p.percentile,
        "median_change": p.median_change,
        "sigma": p.sigma,
    }


def _anomaly_caveats(out) -> list[str]:
    caveats = ["FRED values are the latest revised data; a past release may have looked different."]
    if out.baseline_includes_2020:
        caveats.append("The baseline includes March to December 2020, when many series moved far outside their usual range.")
    if (out.baseline and out.baseline.sigma_method in ("none", "mixed")) or (out.latest and out.latest.z is None):
        caveats.append("z is unavailable for some observations because the baseline has no spread; use the percentile there.")
    if out.baseline_zero_share >= 0.3:
        caveats.append(
            f"{out.baseline_zero_share:.0%} of the baseline changes were exactly zero, so any real change scores a high z; "
            "read a high z here as 'it moved', and use the percentile for how large the move was."
        )
    if out.n_skipped:
        caveats.append(f"{out.n_skipped} observations were skipped for lack of enough earlier history to compare against.")
    return caveats


def _run_anomalies(args: DetectAnomaliesInput, conn, today: date) -> tuple[dict, bool | None]:
    out = detect_anomalies(args.series_id, args.start_date, args.end_date, args.baseline_years, conn)
    result: dict = {
        "series_id": out.series_id,
        "requested": [args.start_date, args.end_date],
        "transform": {"diff": "change in points", "pct": "% change from previous observation",
                      "level": "the growth rate itself"}[out.transform],
        "unit": out.unit,
    }
    if out.n_observations == 0:
        latest = _latest_available(args.series_id, conn, today)
        result["n_scored"] = 0
        result["note"] = (
            f"No {args.series_id} observations between {args.start_date} and {args.end_date}. "
            + (f"Latest published observation: {latest['date']} = {latest['value']}. " if latest else "")
            + "Data for that window has not been released yet or is outside the series' history."
        )
        if latest:
            result["release"] = _release(args.series_id, latest["date"], conn, today)
        return result, out.from_cache

    if out.n_scored == 0:
        # Nothing could be scored, so any_unusual is left out: "false" here would read as "nothing unusual".
        result.update({
            "n_scored": 0,
            "n_skipped": out.n_skipped,
            "note": f"None of the {out.n_skipped} observations in the window had at least 24 earlier changes to compare "
                    "against, so nothing was scored. This says nothing about whether the moves were unusual.",
            "caveats": _anomaly_caveats(out),
        })
        return result, out.from_cache

    result.update({
        "baseline": out.baseline.model_dump() if out.baseline else None,
        "n_scored": out.n_scored,
        "n_flagged": out.n_flagged,
        "n_skipped": out.n_skipped,
        "any_unusual": out.n_flagged > 0,
        "flag_rule": "flagged when |z| >= 3. z = distance from the usual change in standard-deviation-like units, signed; "
                     "percentile = share of baseline moves (by size, ignoring direction) smaller than this one, so they can differ",
        "flagged": [_anomaly_point(p, out) for p in out.flagged],
        "latest": _anomaly_point(out.latest, out) if out.latest else None,
        "caveats": _anomaly_caveats(out),
    })
    # Last: the release lookup only gets the time the core data left over.
    if out.latest:
        result["release"] = _release(args.series_id, out.latest.date, conn, today)
    return result, out.from_cache


def _run_narrative(args: GenerateNarrativeInput, conn, today: date) -> tuple[dict, bool | None]:
    """
    Deterministic brief: the sentences are assembled from the series and anomaly runners' own
    results (the same code paths the underlying tools use), so the numbers cannot disagree.
    """
    rule_for(args.series_id)  # ValueError listing the supported series
    if args.end_date < args.start_date:
        raise ValueError(f"end_date {args.end_date} is before start_date {args.start_date}")
    series, from_cache = _run_economic_series(
        EconomicSeriesInput(series_id=args.series_id, start_date=args.start_date, end_date=args.end_date), conn, today)
    result: dict = {"series_id": args.series_id, "requested": [args.start_date, args.end_date]}
    if series["n_observations"] == 0:
        result["narrative"] = [series["note"]]
        result["caveats"] = ["Nothing in the requested window has been published, so there is nothing to summarize."]
        if "release" in series:
            result["release"] = series["release"]
        return result, from_cache

    anomalies, _ = _run_anomalies(
        DetectAnomaliesInput(series_id=args.series_id, start_date=args.start_date, end_date=args.end_date), conn, today)
    out = build_narrative(args.series_id, series, anomalies)
    result.update(out.model_dump(exclude={"series_id"}))
    result["release"] = series["release"]
    return result, from_cache


RUNNERS = {
    "get_economic_series": _run_economic_series,
    "get_stock_data": _run_stock_data,
    "compare_periods": _run_compare_periods,
    "fetch_macro_context": _run_macro_context,
    "detect_anomalies": _run_anomalies,
    "generate_narrative": _run_narrative,
}


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

@dataclass
class ToolExecution:
    """Everything the agent needs to build a tool_result and log the call."""
    content: str            # JSON (success) or plain-text explanation (error)
    is_error: bool
    latency_ms: float
    from_cache: bool | None
    error_type: str | None = None
    attempts: int | None = None          # network attempts this call made (0 = served from cache)
    breaker_state: str | None = None     # worst circuit-breaker state it met: closed | half_open | open
    approval_required: bool | None = None   # an approval trigger fired for this call
    approval_decision: str | None = None    # approved | auto_approved | declined
    approval_reason: str | None = None


def _explain_error(name: str, tool_input: dict, exc: Exception, db_path: str, today: date) -> str:
    """Turn an exception into a message the model can act on."""
    msg = str(exc)
    if isinstance(exc, ValidationError):
        text = f"Invalid arguments for {name}: {exc.errors(include_url=False)}"
        if any("series_id" in err["loc"] for err in exc.errors()):
            catalog = "; ".join(f"{k} ({v})" for k, v in COMMON_SERIES.items())
            text += f". FRED series IDs are 1-25 uppercase letters/digits. Common series: {catalog}."
        return text
    if isinstance(exc, SourceUnavailableError):
        return msg
    if isinstance(exc, (FutureTimeout, TimeoutError)):
        return f"{name} timed out after {TOOL_TIMEOUT_SECONDS}s. The data source may be slow; try a shorter range or answer without it."
    if isinstance(exc, ConnectionError):
        return f"Data source unreachable ({msg}). This is a temporary network problem, not a bad request."
    if isinstance(exc, RuntimeError):
        return f"Data source is misconfigured: {msg}. Tell the user this tool is unavailable."
    if isinstance(exc, ValueError):
        if name in ("get_economic_series", "compare_periods") and ("not found" in msg or "rejected" in msg):
            catalog = "; ".join(f"{k} ({v})" for k, v in COMMON_SERIES.items())
            return f"{msg}. FRED series IDs are alphanumeric. Common series: {catalog}."
        if name == "get_stock_data":
            catalog = "; ".join(f"{k} ({v})" for k, v in COMMON_TICKERS.items())
            return f"{msg}. If the range is valid, the symbol may be wrong. Common symbols: {catalog}."
        if name == "compare_periods" and "No " in msg:
            series_id = tool_input.get("series_id", "")
            conn = initialize_database(db_path)
            try:
                latest = _latest_available(series_id, conn, today)
            except Exception:
                latest = None
            finally:
                conn.close()
            hint = f". Latest published {series_id} observation: {latest['date']} = {latest['value']}." if latest else ""
            return msg + hint
        return msg
    return f"Unexpected {type(exc).__name__}: {msg}"


def execute_tool(name: str, tool_input: dict, db_path: str, today: date | None = None) -> ToolExecution:
    """
    Validate tool_input, run the Phase 1 function with a 10 s timeout, and
    return a ToolExecution. Never raises.

    Each call opens its own SQLite connection inside the worker thread
    (sqlite3 connections are bound to the thread that made them).
    """
    today = today or date.today()
    t0 = time.perf_counter()

    def elapsed() -> float:
        return round((time.perf_counter() - t0) * 1000, 1)

    if name not in RUNNERS:
        return ToolExecution(f"Unknown tool {name!r}. Available: {sorted(RUNNERS)}", True, elapsed(), None, "UnknownTool")

    stats: dict = {"attempts": 0, "breaker_state": None}   # filled in by resilience.retry_call

    def work():
        _call.deadline = time.monotonic() + TOOL_TIMEOUT_SECONDS
        _call.stats = stats
        model = TOOL_SPECS[name][0]
        args = model.model_validate(tool_input)
        conn = initialize_database(db_path)
        try:
            return RUNNERS[name](args, conn, today)
        finally:
            conn.close()

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        result, from_cache = pool.submit(work).result(timeout=TOOL_TIMEOUT_SECONDS)
        return ToolExecution(json.dumps(result), False, elapsed(), from_cache,
                             attempts=stats["attempts"], breaker_state=stats["breaker_state"])
    except Exception as exc:  # the error boundary: nothing escapes into the loop
        return ToolExecution(_explain_error(name, tool_input, exc, db_path, today), True, elapsed(), None,
                             type(exc).__name__, attempts=stats["attempts"], breaker_state=stats["breaker_state"])
    finally:
        # Don't block on a timed-out worker; it finishes (or dies) on its own.
        pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------

# How a tool's structured outcome (is_error, from_cache, attempts, ...) reaches the metrics callback.
# LangChain's string-in/string-out tool contract has no side channel, and a thread-local is wrong once
# two tools run at once (and ContextVars don't survive: LangChain runs each tool in a *copy* of the
# caller's context). So the outcome is keyed by the tool run's own id, which the tool function reads
# from its child callbacks (parent_run_id) and the callback handler gets as run_id.
_executions: dict = {}
_executions_lock = threading.Lock()


def _record_execution(execution: ToolExecution, callbacks) -> None:
    run_id = getattr(callbacks, "parent_run_id", None)
    if run_id is not None:
        with _executions_lock:
            _executions[run_id] = execution


def pop_execution(run_id) -> ToolExecution | None:
    """The ToolExecution recorded for this tool run (read once); None if the tool function never ran."""
    with _executions_lock:
        return _executions.pop(run_id, None)


DECLINED_MESSAGE = (
    "The user declined this call ({reason}). Do not retry it. Answer with what you have and say what you "
    "could not check."
)


def _is_cold(name: str, tool_input: dict, db_path: str) -> bool | None:
    """True if the data this call fetches isn't in the cache (so it would hit the network); None if unknown."""
    conn = initialize_database(db_path)
    try:
        if name in ("get_economic_series", "detect_anomalies", "generate_narrative"):
            start, end = tool_input["start_date"], tool_input["end_date"]
            if name != "get_economic_series":
                # these also fetch the baseline years before the window (see tools/anomaly.py)
                years = tool_input.get("baseline_years", 10)
                start = (date.fromisoformat(start) - timedelta(days=int(years * 365.25) + 400)).isoformat()
            return not get_cache_status(conn, tool_input["series_id"], config.CACHE_TTL_FRED, start, end)["cached"]
        if name == "compare_periods":
            start = min(tool_input["p1_start"], tool_input["p2_start"])
            end = max(tool_input["p1_end"], tool_input["p2_end"])
            return not get_cache_status(conn, tool_input["series_id"], config.CACHE_TTL_FRED, start, end)["cached"]
        if name == "get_stock_data":
            interval = tool_input.get("interval", "1d")
            ttl = config.CACHE_TTL_STOCK_DAILY
            return not get_stock_cache_status(conn, tool_input["ticker"], interval, ttl,
                                              tool_input["start_date"], tool_input["end_date"])["cached"]
    except Exception:
        return None
    finally:
        conn.close()
    return None


def execute_gated(name: str, tool_input: dict, db_path: str, today: date | None, gate: ApprovalGate | None) -> ToolExecution:
    """
    execute_tool behind the approval gate. A call the policy flags is put to the gate's approve callback
    first; a decline returns a model-readable message instead of running the tool. Never raises.
    """
    decision = None
    if gate is not None:
        try:
            decision = gate.review(name, tool_input, _is_cold(name, tool_input, db_path))
        except Exception:  # a broken approval callback must not take the answer down; fail open, say so in the log
            decision = None
        if decision is not None and decision.declined:
            return ToolExecution(DECLINED_MESSAGE.format(reason=decision.reason), True, 0.0, None, "ApprovalDeclined",
                                 attempts=0, approval_required=True, approval_decision="declined",
                                 approval_reason=decision.reason)
    execution = execute_tool(name, tool_input, db_path, today)
    if decision is not None and decision.required:
        execution = dataclasses.replace(execution, approval_required=True, approval_decision=decision.decision,
                                        approval_reason=decision.reason)
    return execution


def build_tools(db_path: str, today: date | None = None, gate: ApprovalGate | None = None) -> list[StructuredTool]:
    """
    One StructuredTool per entry in TOOL_SPECS, bound to a database and a date.

    LangChain validates the model's arguments against args_schema before the
    function runs; a failed validation is turned into the same corrective
    message the Phase 2 executor produced, instead of raising out of the agent.
    """
    tools = []
    last = len(TOOL_SPECS) - 1
    for i, (name, (model, description)) in enumerate(TOOL_SPECS.items()):
        # `callbacks` is injected by LangChain (a child of this tool run's callback manager).
        def run(callbacks=None, _name=name, **tool_input) -> str:
            execution = execute_gated(_name, tool_input, db_path, today, gate)
            _record_execution(execution, callbacks)
            return execution.content

        async def arun(callbacks=None, _name=name, **tool_input) -> str:
            # The same blocking executor, off the event loop, so sibling tool calls overlap.
            execution = await asyncio.to_thread(execute_gated, _name, tool_input, db_path, today, gate)
            _record_execution(execution, callbacks)
            return execution.content

        def on_invalid(exc: ValidationError, _name=name) -> str:
            # No execution is recorded: the callback handler reads that as a ValidationError.
            return _explain_error(_name, {}, exc, db_path, today or date.today())

        tools.append(StructuredTool.from_function(
            func=run, coroutine=arun, name=name, description=description, args_schema=model,
            handle_validation_error=on_invalid,
            # Prompt caching: one breakpoint after the last tool definition caches the whole tool list.
            extras={"cache_control": {"type": "ephemeral"}} if i == last else None,
        ))
    return tools
