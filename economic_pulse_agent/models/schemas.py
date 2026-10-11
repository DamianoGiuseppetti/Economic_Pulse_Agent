"""
Pydantic v2 output schemas for every tool.

Rows are typed models rather than bare dicts, so a missing field or a string
where a float belongs fails validation instead of reaching the agent.
model_dump() still produces exactly the dict shapes from the spec.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SeriesPoint(BaseModel):
    date: str
    value: float


class OHLCVBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int = Field(ge=0)


class EconomicSeriesOutput(BaseModel):
    series_id: str
    data: list[SeriesPoint]
    source: Literal["FRED"]
    fetched_at: str  # ISO 8601: when the underlying data was fetched from FRED
    from_cache: bool


class StockDataOutput(BaseModel):
    ticker: str
    interval: str
    data: list[OHLCVBar]
    source: Literal["yfinance"]
    fetched_at: str
    from_cache: bool


class PeriodSummary(BaseModel):
    start: str
    end: str
    aggregated_value: float
    n_observations: int = Field(ge=1)


class ComparePeriodOutput(BaseModel):
    series_id: str
    agg_method: Literal["mean", "last", "first", "min", "max"]
    period_1: PeriodSummary
    period_2: PeriodSummary
    delta: float      # period_2 - period_1
    delta_pct: float  # (delta / period_1) * 100
    direction: Literal["up", "down", "flat"]

    @model_validator(mode="after")
    def _check_arithmetic(self):
        # Guards against a future refactor silently breaking the math.
        expected = self.period_2.aggregated_value - self.period_1.aggregated_value
        if abs(self.delta - expected) > 1e-9:
            raise ValueError(f"delta {self.delta} != period_2 - period_1 ({expected})")
        return self


class IndicatorReading(BaseModel):
    """One FRED indicator measured over the context window."""
    name: str
    series_id: str
    start_date: str
    start_value: float
    end_date: str
    end_value: float
    change: float
    score: float | None = Field(default=None, ge=-1.0, le=1.0)  # None = context only, not in the composite
    reading: str


class FomcEvent(BaseModel):
    date: str
    title: str
    url: str
    statement_excerpt: str | None = None


class MacroContextOutput(BaseModel):
    keywords: list[str]
    window: dict[str, str]          # {"start": ..., "end": ...} actually covered
    context_summary: str
    sentiment: Literal["bullish", "neutral", "bearish"]
    confidence: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=-1.0, le=1.0)
    signals: list[IndicatorReading]      # scored: VIX, credit spread, consumer sentiment
    indicators: list[IndicatorReading]   # keyword-driven context, unscored
    fomc_events: list[FomcEvent]
    sources_unavailable: list[str]
    method: str
    fetched_at: str
    from_cache: bool


class ReleaseInfo(BaseModel):
    """When a FRED series' data comes out."""
    series_id: str
    release_name: str
    latest_observation: str               # period the newest value describes
    latest_released_on: str | None       # when that value was first published
    next_release_date: str | None        # next scheduled release, on or after today
    from_cache: bool


class AnomalyPoint(BaseModel):
    """One scored observation: its level, its change, and how unusual the change is."""
    date: str
    level: float
    change: float                  # the transformed value that was scored (diff / % change / growth rate)
    z: float | None                # robust z; None when the baseline has no spread at all
    percentile: float = Field(ge=0.0, le=100.0)  # of |change| within the baseline's |changes|
    median_change: float
    sigma: float


class BaselineInfo(BaseModel):
    years: int
    n_min: int
    n_max: int
    start: str
    end: str
    sigma_method: Literal["mad", "stdev", "none", "mixed"]


class AnomalyOutput(BaseModel):
    series_id: str
    transform: Literal["diff", "pct", "level"]
    unit: str
    period: str
    start_date: str
    end_date: str
    n_observations: int            # observations in the window (scored + skipped)
    n_scored: int
    n_flagged: int                 # all flags; `flagged` is capped
    n_skipped: int                 # too little history before them
    baseline: BaselineInfo | None  # None when nothing was scored
    flagged: list[AnomalyPoint]    # largest |z| first, capped
    latest: AnomalyPoint | None    # last scored observation, flagged or not
    baseline_includes_2020: bool
    baseline_zero_share: float     # share of exactly-zero changes in the latest observation's baseline
    from_cache: bool



class NarrativeOutput(BaseModel):
    """generate_narrative's text. `release` is added by the runner (it needs the DB and the clock)."""
    series_id: str
    narrative: list[str] = Field(min_length=1, max_length=5)
    caveats: list[str]
