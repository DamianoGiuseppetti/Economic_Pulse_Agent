"""
Entity memory, within one session: what was discussed (series, tickers, date windows), so a follow-up like
"and core CPI over the same period?" resolves without the user repeating the window.

Deterministic, no model call. After each turn, the *successful* tool calls of that turn are reduced to
(type, id, window) and upserted into entity_memory; before each turn a short "Recently discussed" block is
built from the session's most recent entities and goes into the prompt after the date line (outside the
cached block). It carries identifiers and windows only, never numbers: the model still fetches data with
its tools, so grounding still has tool results to check answers against. The text-only history replay in
agent.py is untouched.
"""

from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_memory (
    session_id   TEXT NOT NULL,
    entity_type  TEXT NOT NULL,
    entity_id    TEXT NOT NULL,
    last_start   TEXT,
    last_end     TEXT,
    last_turn    INTEGER NOT NULL,
    mentions     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (session_id, entity_type, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_entity_session_turn ON entity_memory (session_id, last_turn);
"""

MAX_ENTITIES = 8
MAX_BLOCK_CHARS = 500   # about 200 tokens: measured at roughly 2.2 characters per token for this text

SERIES_TOOLS = ("get_economic_series", "detect_anomalies", "generate_narrative")
PREFACE = "Recently discussed (to resolve 'that', 'same period'; still fetch data with tools, don't quote this): "


@dataclass(frozen=True)
class Entity:
    type: str        # series | ticker
    id: str
    start: str | None
    end: str | None


def extract_entities(tool_calls) -> list[Entity]:
    """
    One Entity per series or ticker the turn's tool calls touched successfully. Calls that errored or were
    declined create nothing. Several calls for one entity in a turn merge into one with the widest window.
    `tool_calls` are ToolCallRecord-like objects with .name, .input and .is_error.
    """
    merged: dict[tuple[str, str], Entity] = {}
    for call in tool_calls:
        if call.is_error:
            continue
        args = call.input
        try:
            if call.name in SERIES_TOOLS:
                entity = Entity("series", args["series_id"], args["start_date"], args["end_date"])
            elif call.name == "compare_periods":
                entity = Entity("series", args["series_id"], min(args["p1_start"], args["p2_start"]),
                                max(args["p1_end"], args["p2_end"]))
            elif call.name == "get_stock_data":
                entity = Entity("ticker", args["ticker"], args["start_date"], args["end_date"])
            else:
                continue
        except KeyError:
            continue
        key = (entity.type, entity.id)
        if key in merged:
            old = merged[key]
            entity = Entity(entity.type, entity.id, min(old.start, entity.start), max(old.end, entity.end))
        merged[key] = entity
    return list(merged.values())


def record(db, lock, session_id: str, turn: int, entities: list[Entity]) -> None:
    """Upsert this turn's entities: window and turn replaced, mentions counted once per turn."""
    if not entities:
        return
    with lock:
        db.executemany(
            """INSERT INTO entity_memory (session_id, entity_type, entity_id, last_start, last_end, last_turn, mentions)
               VALUES (?, ?, ?, ?, ?, ?, 1)
               ON CONFLICT (session_id, entity_type, entity_id) DO UPDATE SET
                   last_start = excluded.last_start, last_end = excluded.last_end,
                   last_turn = excluded.last_turn, mentions = mentions + 1""",
            [(session_id, e.type, e.id, e.start, e.end, turn) for e in entities],
        )
        db.commit()


def build_block(db, lock, session_id: str) -> str | None:
    """The 'Recently discussed' line for this session (most recent first), or None if nothing was discussed."""
    with lock:
        rows = db.execute(
            """SELECT entity_id, last_start, last_end, last_turn FROM entity_memory
               WHERE session_id = ? ORDER BY last_turn DESC, mentions DESC, entity_id LIMIT ?""",
            (session_id, MAX_ENTITIES)).fetchall()
    if not rows:
        return None
    items: list[str] = []
    for row in rows:
        # Month precision: dates cost about one token per two characters, and a window to the month is enough
        # to say "the same period" (the model fills in first and last day).
        item = f"{row['entity_id']} {str(row['last_start'])[:7]}..{str(row['last_end'])[:7]} (turn {row['last_turn']})"
        if len(PREFACE) + len("; ".join(items + [item])) + 1 > MAX_BLOCK_CHARS:
            break
        items.append(item)
    return PREFACE + "; ".join(items) + "."
