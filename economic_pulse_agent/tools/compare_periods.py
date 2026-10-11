"""Tool 3: compare_periods — aggregate a FRED series over two windows and diff them."""

import sqlite3
from statistics import mean

import config
from database import initialize_database
from models.schemas import ComparePeriodOutput, PeriodSummary
from tools.economic_series import get_economic_series

AGGREGATORS = {
    "mean": mean,
    "last": lambda values: values[-1],
    "first": lambda values: values[0],
    "min": min,
    "max": max,
}

# |delta| at or below this counts as "flat". Absolute, in the series' own
# units — fine for rates like UNRATE, too tight for levels like CPI.
FLAT_THRESHOLD = 0.01


def _summarize(series_id: str, period: dict, agg_method: str, db_connection) -> PeriodSummary:
    result = get_economic_series(series_id, period["start"], period["end"], db_connection)
    values = [point.value for point in result.data]
    if not values:
        # Common case: asking for a month that hasn't been released yet.
        raise ValueError(
            f"No {series_id} observations between {period['start']} and {period['end']} "
            f"(not yet released, or outside the series' history)"
        )
    return PeriodSummary(
        start=period["start"],
        end=period["end"],
        aggregated_value=AGGREGATORS[agg_method](values),
        n_observations=len(values),
    )


def compare_periods(
    series_id: str,
    period_1: dict,
    period_2: dict,
    agg_method: str = "mean",
    db_connection: sqlite3.Connection | None = None,
) -> ComparePeriodOutput:
    """
    Compare series_id between two periods, each {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}.

    delta = agg(period_2) - agg(period_1); delta_pct is relative to period_1.
    Raises ValueError for an unknown agg_method, an empty period, or a zero
    baseline (percentage change undefined).
    """
    if agg_method not in AGGREGATORS:
        raise ValueError(f"agg_method must be one of {sorted(AGGREGATORS)}, got {agg_method!r}")
    if db_connection is None:
        db_connection = initialize_database(config.DB_PATH)

    p1 = _summarize(series_id, period_1, agg_method, db_connection)
    p2 = _summarize(series_id, period_2, agg_method, db_connection)

    if p1.aggregated_value == 0:
        raise ValueError(f"{series_id} period_1 aggregate is 0; delta_pct is undefined")

    delta = p2.aggregated_value - p1.aggregated_value
    delta_pct = delta / p1.aggregated_value * 100
    if delta > FLAT_THRESHOLD:
        direction = "up"
    elif delta < -FLAT_THRESHOLD:
        direction = "down"
    else:
        direction = "flat"

    return ComparePeriodOutput(
        series_id=series_id,
        agg_method=agg_method,
        period_1=p1,
        period_2=p2,
        delta=delta,
        delta_pct=delta_pct,
        direction=direction,
    )
