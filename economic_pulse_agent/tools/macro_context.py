"""
Tool 4: fetch_macro_context — market-implied sentiment plus Fed policy events.

Sentiment is a transparent, rule-based composite of three FRED indicators,
not a language model's opinion:

  VIX (VIXCLS)                      fear gauge: low and falling = risk-on
  High-yield spread (BAMLH0A0HYM2)  credit stress: tight and narrowing = risk-on
  Consumer sentiment (UMCSENT)      household mood: high and rising = upbeat

Each indicator scores -1 (bearish) .. +1 (bullish) from its level at the end
of the window and its change across it; the composite is their mean.
Keywords add unscored context indicators (e.g. "inflation" -> 5-year
breakeven inflation) and FOMC press releases in the window come from the
Fed's RSS feed, with the decision paragraphs of any statement.

A source that fails is listed in `sources_unavailable` instead of aborting
the whole call; if no scored indicator is available at all, ValueError.
"""

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from statistics import mean, pstdev

import config
from data_fetchers import fed_fetcher
from database import initialize_database, utc_now_iso
from models.schemas import FomcEvent, IndicatorReading, MacroContextOutput
from tools.economic_series import get_economic_series

METHOD = (
    "Composite of VIX, high-yield credit spread and consumer sentiment; each scored -1..+1 "
    "from its end-of-window level and change over the window, then averaged. "
    "bullish > +0.2, bearish < -0.2. Confidence falls as the indicators disagree (spread of "
    "their scores) and as fewer of them are available."
)
MAX_STATEMENTS = 2


def _move(delta: float, fmt: str, unit: str) -> str:
    """'down 2.1 pts' rather than '-2.1 pts': signs are easy for a model to misread."""
    if abs(delta) < 1e-9:
        return "unchanged"
    return f"{'up' if delta > 0 else 'down'} {format(abs(delta), fmt)}{unit}"


def _clip(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _score_vix(start: float, end: float) -> float:
    # ~20 is the long-run average; a 25% jump in a window is a big risk-off move.
    level = _clip((20 - end) / 10)
    change = _clip(-((end / start - 1) * 100) / 25)
    return round(mean([level, change]), 3)


def _score_hy_spread(start: float, end: float) -> float:
    # Values are percent. ~4.5% is a normal spread; 40bp of widening is notable stress.
    level = _clip((4.5 - end) / 2)
    change = _clip(-((end - start) * 100) / 40)
    return round(mean([level, change]), 3)


def _score_umcsent(start: float, end: float) -> float:
    # Index ~85 is the long-run average; 5 points is a large monthly swing.
    level = _clip((end - 85) / 20)
    change = _clip((end - start) / 5)
    return round(mean([level, change]), 3)


# (label, series_id, days of look-back padding, scorer, reading formatter)
SCORED = [
    ("VIX (equity volatility)", "VIXCLS", 10, _score_vix,
     lambda s, e: f"VIX {e:.1f}, {_move(e - s, '.1f', ' pts')} over the window"),
    ("High-yield credit spread", "BAMLH0A0HYM2", 10, _score_hy_spread,
     lambda s, e: f"HY spread {e:.2f}%, {_move((e - s) * 100, '.0f', ' bp')} over the window"),
    ("Consumer sentiment (U. Michigan)", "UMCSENT", 70, _score_umcsent,
     lambda s, e: f"Consumer sentiment {e:.1f}, {_move(e - s, '.1f', ' pts')} over the window"),
]

# keyword triggers -> unscored context indicator
KEYWORD_INDICATORS = [
    (("inflation", "cpi", "price", "pce"), "5-year breakeven inflation", "T5YIE",
     lambda s, e: f"5y breakeven {e:.2f}%, {_move((e - s) * 100, '.0f', ' bp')} (market inflation expectations)"),
    (("fed", "fomc", "rate", "monetary", "powell"), "2-year Treasury yield", "DGS2",
     lambda s, e: f"2y yield {e:.2f}%, {_move((e - s) * 100, '.0f', ' bp')} (market view of Fed policy)"),
    (("job", "labor", "labour", "unemploy", "employ", "payroll"), "Initial jobless claims", "ICSA",
     lambda s, e: f"Initial claims {e:,.0f}, {_move(e - s, ',.0f', '')} over the window"),
]


def _measure(conn, name, series_id, start, end, pad_days, fmt, scorer=None) -> IndicatorReading:
    """Value at (or just before) the window start and end, plus the change."""
    padded_start = (date.fromisoformat(start) - timedelta(days=pad_days)).isoformat()
    points = get_economic_series(series_id, padded_start, end, conn).data
    if not points:
        raise ValueError(f"no {series_id} observations up to {end}")
    before_start = [p for p in points if p.date <= start]
    first = before_start[-1] if before_start else points[0]
    last = points[-1]
    return IndicatorReading(
        name=name, series_id=series_id,
        start_date=first.date, start_value=first.value,
        end_date=last.date, end_value=last.value,
        change=round(last.value - first.value, 4),
        score=scorer(first.value, last.value) if scorer else None,
        reading=fmt(first.value, last.value),
    )


def _fomc_events(start: str, end: str, unavailable: list[str], failures: list[str]) -> list[FomcEvent]:
    try:
        releases = fed_fetcher.fetch_fomc_press_releases()
    except (ConnectionError, ValueError) as exc:
        unavailable.append(f"FOMC press releases: {exc}")
        failures.append("fomc")
        return []
    if releases and start < releases[0]["date"]:
        unavailable.append(
            f"FOMC press releases before {releases[0]['date']} (the Fed feed only covers recent months)"
        )
    in_window = [r for r in releases if start <= r["date"] <= end]
    events = [FomcEvent(**r) for r in in_window]
    statements = [e for e in events if "FOMC statement" in e.title][-MAX_STATEMENTS:]
    for event in statements:
        try:
            event.statement_excerpt = fed_fetcher.fetch_statement_text(event.url)
        except (ConnectionError, ValueError) as exc:
            unavailable.append(f"Statement text {event.date}: {exc}")
            failures.append("statement")
    return events


def _summarize(sentiment, composite, signals, indicators, events, start, end) -> str:
    parts = [f"Market-implied sentiment {start} to {end} is {sentiment} (composite {composite:+.2f})."]
    upbeat = [s.name for s in signals if s.score > 0.2]
    gloomy = [s.name for s in signals if s.score < -0.2]
    if upbeat and gloomy:
        parts.append(f"Signals are mixed: {', '.join(upbeat)} upbeat vs {', '.join(gloomy)} gloomy.")
    parts.append("; ".join(s.reading for s in signals + indicators) + ".")
    decisions = [e for e in events if e.statement_excerpt]
    if decisions:
        first_sentence = lambda t: t.split(". ")[0].rstrip(".") + "."
        parts.append(" ".join(f"FOMC {e.date}: {first_sentence(e.statement_excerpt)}" for e in decisions))
    elif events:
        parts.append(f"{len(events)} Fed monetary-policy release(s) in the window.")
    else:
        parts.append("No Fed monetary-policy releases in the window.")
    return " ".join(parts)


def fetch_macro_context(
    keywords: list[str],
    date_range: dict,
    db_connection: sqlite3.Connection | None = None,
    cache_ttl_seconds: int = config.CACHE_TTL_MACRO,
    today: date | None = None,
) -> MacroContextOutput:
    """
    Sentiment and policy context for keywords over date_range {"start", "end"}.

    The window end is clamped to today; a window starting in the future
    raises ValueError. Results are cached whole for cache_ttl_seconds.
    """
    today = today or date.today()
    start, end = date_range["start"], min(date_range["end"], today.isoformat())
    if start > end:
        raise ValueError(f"Window {date_range['start']}..{date_range['end']} starts after today ({today})")
    if db_connection is None:
        db_connection = initialize_database(config.DB_PATH)

    cache_key = json.dumps([sorted(k.lower() for k in keywords), start, end])
    row = db_connection.execute(
        "SELECT payload, fetch_timestamp FROM macro_context_cache WHERE key = ?", (cache_key,)
    ).fetchone()
    if row:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(row["fetch_timestamp"])).total_seconds()
        if age < cache_ttl_seconds:
            cached = json.loads(row["payload"])
            return MacroContextOutput(**{**cached, "keywords": keywords, "from_cache": True})

    unavailable: list[str] = []
    failures: list[str] = []   # transient errors: don't cache a result that has them
    signals: list[IndicatorReading] = []
    for name, series_id, pad, scorer, fmt in SCORED:
        try:
            signals.append(_measure(db_connection, name, series_id, start, end, pad, fmt, scorer))
        except (ValueError, ConnectionError) as exc:
            unavailable.append(f"{name} ({series_id}): {exc}")
            failures.append(series_id)
    if not signals:
        raise ValueError(f"No sentiment indicators available for {start}..{end}: {unavailable}")

    lowered = " ".join(keywords).lower()
    indicators: list[IndicatorReading] = []
    for triggers, name, series_id, fmt in KEYWORD_INDICATORS:
        if any(t in lowered for t in triggers):
            try:
                indicators.append(_measure(db_connection, name, series_id, start, end, 10, fmt))
            except (ValueError, ConnectionError) as exc:
                unavailable.append(f"{name} ({series_id}): {exc}")
                failures.append(series_id)

    events = _fomc_events(start, end, unavailable, failures)

    scores = [s.score for s in signals]
    composite = round(mean(scores), 3)
    sentiment = "bullish" if composite > 0.2 else "bearish" if composite < -0.2 else "neutral"
    # Scores live in [-1, 1]; a spread (std dev) of 0.6+ means they point in opposite directions.
    agreement = 1 - min(1.0, pstdev(scores) / 0.6) if len(scores) > 1 else 0.5
    coverage = len(scores) / len(SCORED)
    confidence = round(agreement * (0.5 + 0.5 * coverage), 2)

    output = MacroContextOutput(
        keywords=keywords,
        window={"start": start, "end": end},
        context_summary=_summarize(sentiment, composite, signals, indicators, events, start, end),
        sentiment=sentiment,
        confidence=confidence,
        composite_score=composite,
        signals=signals,
        indicators=indicators,
        fomc_events=events,
        sources_unavailable=unavailable,
        method=METHOD,
        fetched_at=utc_now_iso(),
        from_cache=False,
    )
    if failures:
        return output
    db_connection.execute(
        """INSERT INTO macro_context_cache (key, payload, fetch_timestamp) VALUES (?, ?, ?)
           ON CONFLICT (key) DO UPDATE SET payload = excluded.payload, fetch_timestamp = excluded.fetch_timestamp""",
        (cache_key, output.model_dump_json(), output.fetched_at),
    )
    db_connection.commit()
    return output
