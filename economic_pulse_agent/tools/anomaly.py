"""
Tool 5: detect_anomalies — was a FRED series' move unusual for that series?

Each observation in the window is scored against the series' own trailing
history. The statistic runs on *changes*, never levels: CPI or GDP trend
upward, so a z-score on the level would flag nearly every recent month. The
transform per series is a fixed table (SERIES_RULES); the model never picks it.

The score is a robust z (median / MAD) plus a percentile rank on magnitude.
Each observation's baseline is the window of changes strictly before it, so
no observation is scored against its own value or anything later.
"""

import sqlite3
import statistics
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import config
from database import initialize_database
from models.schemas import AnomalyOutput, AnomalyPoint, BaselineInfo
from tools.economic_series import get_economic_series

FLAG_Z = 3.0
MIN_BASELINE = 24
MAD_TO_SIGMA = 1.4826  # makes MAD comparable to a standard deviation when data are normal
FIRST_DIFF_MARGIN_DAYS = 400  # one extra period before the baseline, for the first difference
SHOCK_2020 = (date(2020, 3, 1), date(2020, 12, 31))
MAX_FLAGGED = 5


@dataclass(frozen=True)
class SeriesRule:
    transform: Literal["diff", "pct", "level"]
    unit: str
    period: str


SERIES_RULES: dict[str, SeriesRule] = {
    "UNRATE": SeriesRule("diff", "pts", "month"),
    "FEDFUNDS": SeriesRule("diff", "pts", "month"),
    "DGS10": SeriesRule("diff", "pts", "day"),
    "DGS2": SeriesRule("diff", "pts", "day"),
    "T10Y2Y": SeriesRule("diff", "pts", "day"),
    "MORTGAGE30US": SeriesRule("diff", "pts", "week"),
    "UMCSENT": SeriesRule("diff", "index pts", "month"),
    "PAYEMS": SeriesRule("diff", "thousand jobs", "month"),
    "CPIAUCSL": SeriesRule("pct", "%", "month"),
    "CPILFESL": SeriesRule("pct", "%", "month"),
    "PCEPILFE": SeriesRule("pct", "%", "month"),
    "GDPC1": SeriesRule("pct", "%", "quarter"),
    "ICSA": SeriesRule("pct", "%", "week"),
    "A191RL1Q225SBEA": SeriesRule("level", "% annualized", "quarter"),
}


def rule_for(series_id: str) -> SeriesRule:
    if series_id not in SERIES_RULES:
        raise ValueError(
            f"detect_anomalies does not support {series_id}. Supported series: {', '.join(SERIES_RULES)}. "
            "Use get_economic_series for the raw values of other series."
        )
    return SERIES_RULES[series_id]


def transform_series(points, rule: SeriesRule) -> list[tuple[str, float]]:
    """
    (date, change) pairs. diff: v[t]-v[t-1]; pct: (v[t]/v[t-1]-1)*100; level: v[t].
    The first observation has no change and is dropped (diff/pct). A pct change
    off a zero previous value is undefined and skipped.
    """
    if rule.transform == "level":
        return [(p.date, p.value) for p in points]
    changes = []
    for prev, cur in zip(points, points[1:]):
        if rule.transform == "diff":
            changes.append((cur.date, cur.value - prev.value))
        elif prev.value != 0:
            changes.append((cur.date, (cur.value / prev.value - 1) * 100))
    return changes


@dataclass(frozen=True)
class Score:
    z: float | None
    percentile: float
    median: float
    sigma: float
    method: Literal["mad", "stdev", "none"]


def score_one(change: float, baseline: list[float]) -> Score:
    """
    Robust z of `change` against `baseline`, plus its magnitude percentile.

    sigma = 1.4826 * MAD. Series that sit still (FEDFUNDS) can have a MAD of
    zero, so sigma falls back to the standard deviation, and if that is zero
    too, z is None and the percentile carries the message.
    """
    median = statistics.median(baseline)
    mad = statistics.median(abs(b - median) for b in baseline)
    sigma, method = MAD_TO_SIGMA * mad, "mad"
    if sigma == 0:
        sigma, method = (statistics.stdev(baseline) if len(baseline) > 1 else 0.0), "stdev"
    if sigma == 0:
        z, method = None, "none"
    else:
        z = (change - median) / sigma
    magnitude = abs(change)
    below = sum(1 for b in baseline if abs(b) < magnitude)
    ties = sum(1 for b in baseline if abs(b) == magnitude)
    percentile = 100 * (below + 0.5 * ties) / len(baseline)
    return Score(z, percentile, median, sigma, method)


def _years_earlier(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:  # Feb 29
        return d.replace(year=d.year - years, day=28)


def detect_anomalies(
    series_id: str,
    start_date: str,
    end_date: str,
    baseline_years: int = 10,
    db_connection: sqlite3.Connection | None = None,
) -> AnomalyOutput:
    """
    Score every observation in [start_date, end_date] against the changes in the
    `baseline_years` before it. Raises ValueError for an unsupported series or
    end_date < start_date; other exceptions come from the data fetch.
    """
    rule = rule_for(series_id)
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")
    conn = db_connection or initialize_database(config.DB_PATH)

    fetch_start = _years_earlier(start, baseline_years) - timedelta(days=FIRST_DIFF_MARGIN_DAYS)
    out = get_economic_series(series_id, fetch_start.isoformat(), end_date, conn)
    levels = {p.date: p.value for p in out.data}
    changes = transform_series(out.data, rule)
    change_dates = [d for d, _ in changes]
    change_values = [c for _, c in changes]

    scored: list[tuple[str, float, Score, int, int]] = []  # date, change, score, n_baseline, n_zero
    n_skipped = 0
    baselines: list[tuple[str, str]] = []
    for i, (d, change) in enumerate(changes):
        if not (start_date <= d <= end_date):
            continue
        lo = bisect_left(change_dates, _years_earlier(date.fromisoformat(d), baseline_years).isoformat())
        baseline = change_values[lo:i]  # strictly before this observation
        if len(baseline) < MIN_BASELINE:
            n_skipped += 1
            continue
        scored.append((d, change, score_one(change, baseline), len(baseline), sum(1 for b in baseline if b == 0)))
        baselines.append((change_dates[lo], change_dates[i - 1]))

    flagged_all = sorted(
        (s for s in scored if s[2].z is not None and abs(s[2].z) >= FLAG_Z),
        key=lambda s: abs(s[2].z), reverse=True,
    )

    def point(s) -> AnomalyPoint:
        d, change, sc, _, _ = s
        return AnomalyPoint(date=d, level=levels[d], change=round(change, 4),
                            z=round(sc.z, 2) if sc.z is not None else None,
                            percentile=round(sc.percentile, 1), median_change=round(sc.median, 4),
                            sigma=round(sc.sigma, 4))

    baseline_info = None
    includes_2020 = False
    zero_share = 0.0
    if scored:
        methods = {s[2].method for s in scored}
        baseline_info = BaselineInfo(
            years=baseline_years,
            n_min=min(s[3] for s in scored), n_max=max(s[3] for s in scored),
            start=min(b[0] for b in baselines), end=max(b[1] for b in baselines),
            sigma_method=methods.pop() if len(methods) == 1 else "mixed",
        )
        includes_2020 = (date.fromisoformat(baseline_info.start) <= SHOCK_2020[1]
                         and date.fromisoformat(baseline_info.end) >= SHOCK_2020[0])
        last = scored[-1]
        zero_share = last[4] / last[3]

    return AnomalyOutput(
        series_id=series_id, transform=rule.transform, unit=rule.unit, period=rule.period,
        start_date=start_date, end_date=end_date,
        n_observations=len(scored) + n_skipped, n_scored=len(scored), n_flagged=len(flagged_all),
        n_skipped=n_skipped,
        baseline=baseline_info,
        flagged=[point(s) for s in flagged_all[:MAX_FLAGGED]],
        latest=point(scored[-1]) if scored else None,
        baseline_includes_2020=includes_2020, baseline_zero_share=round(zero_share, 2),
        from_cache=out.from_cache,
    )
