"""
Tool 6: generate_narrative — a short, templated, number-carrying brief about one series.

Deterministic on purpose: no model call. The agent's model is already the narrator; a
tool that made its own call would double the cost, put text into the answer outside
the grounding check, and spend a second model call from the iteration budget. So the
sentences are built here from results the series and anomaly tools already computed
(passed in by tools_def, which owns those code paths), and every number in them is
copied from those results. No sentence names a cause: this code knows only data.

Pure functions only. This module imports nothing from tools_def (tools_def imports it).
"""

from models.schemas import NarrativeOutput

MIN_OBSERVATIONS_FOR_CHANGE = 2
SHORT_WINDOW = 4


def _latest_sentence(series_id: str, series: dict) -> str:
    latest = series["latest"]
    release = series.get("release") or {}
    published = release.get("latest_released_on")
    tail = f", published {published}" if published else ""
    return f"The latest {series_id} observation in the window is {latest['value']:g}, for {latest['date']}{tail}."


def _change_sentence(series: dict) -> str | None:
    if series["n_observations"] < MIN_OBSERVATIONS_FOR_CHANGE:
        return None
    return f"Over the window it went {series['change']['summary']}."


def _inflation_sentence(series: dict) -> str | None:
    inflation = series.get("inflation")
    if not inflation:
        return None
    text = f"Year-over-year inflation was {inflation['latest_yoy_pct']:.2f}% in {inflation['latest_yoy_period']}"
    move = inflation.get("latest_vs_previous_month")
    return text + (f"; {move['summary']}." if move else ".")


def _unusual_sentence(anomalies: dict) -> str | None:
    if not anomalies.get("n_scored"):
        return None
    n_scored, n_flagged = anomalies["n_scored"], anomalies["n_flagged"]
    years = (anomalies.get("baseline") or {}).get("years", 10)
    if not n_flagged:
        return (f"None of the {n_scored} observations checked were unusual against the previous "
                f"{years} years of this series' changes.")
    top = anomalies["flagged"][0]
    z = "z unavailable" if top["z"] is None else f"z {top['z']:g}"
    return (f"{n_flagged} of {n_scored} observations were unusual; the largest was {top['date']}, "
            f"{top['change_words']} ({z}, percentile {top['percentile']:g}).")


def _limitation_sentence(series: dict, anomalies: dict) -> str | None:
    if series.get("note"):  # publication lag, from the series result
        return series["note"]
    n = series["n_observations"]
    if n < SHORT_WINDOW:
        return f"The window holds only {n} observation{'s' if n != 1 else ''}, so this is a short-window reading."
    return None


def build_narrative(series_id: str, series: dict, anomalies: dict) -> NarrativeOutput:
    """
    3 to 5 sentences from a get_economic_series result and a detect_anomalies result
    for the same window. Both must be non-empty-window results (the caller handles
    empty windows before calling this).
    """
    sentences = [
        _latest_sentence(series_id, series),
        _change_sentence(series),
        _inflation_sentence(series),
        _unusual_sentence(anomalies),
        _limitation_sentence(series, anomalies),
    ]
    caveats = list(anomalies.get("caveats") or [])
    if not anomalies.get("n_scored"):
        caveats.append(anomalies.get("note") or "No observation in the window could be scored for unusual moves.")
    return NarrativeOutput(
        series_id=series_id,
        narrative=[s for s in sentences if s],
        caveats=caveats,
    )
