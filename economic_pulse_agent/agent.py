"""
Economic Pulse Agent: LangChain's AgentExecutor over the four Phase 1 tools.

    question -> AgentExecutor (ChatAnthropic + tools, tool-calling loop) -> answer
             -> grounding check (one optional revision call) -> SQLite log

Phase 2 hand-rolled this loop over the raw Anthropic SDK. What moved:
  - the while-loop and tool dispatch    -> create_tool_calling_agent + AgentExecutor
  - per-call logging, token accounting  -> MetricsCallbackHandler (below)
What did not: history replay, grounding, the SQLite schema, the tools' behavior.

Budgets: AgentExecutor has no "answer now, no more tools" switch (Phase 2 flipped
tool_choice to "none"). `max_iterations` is a hard stop instead; when it trips
the agent makes one final tool-free call to answer from the data gathered, or
returns a fixed limit message if that fails (exit_reason "max_iterations").
TOKEN_BUDGET is no longer enforced (kept as a reporting ceiling, see the Phase 3 report).

Usage:
    python agent.py                   # interactive chat
    python agent.py --examples        # run example questions, write examples.txt
"""

import argparse
import asyncio
import json
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from approval import ApprovalGate, ApprovalRequest
from grounding import find_unsupported, revision_request
from system_prompts import build_prompt_template, build_system_prompt
from tools_def import build_tools, pop_execution

import config
import entity_memory
from database import initialize_database, utc_now_iso

# Haiku 4.5 keeps costs low ($1 / $5 per million input / output tokens).
# It has no adaptive thinking or effort setting, so neither is sent.
MODEL = "claude-haiku-4-5"
MAX_TOKENS = 4096           # per response; answers target ~250 words
MAX_ITERATIONS = 5          # model calls per question: the only budget AgentExecutor enforces
HISTORY_TURNS = 10          # earlier Q&A pairs replayed as context (~300-400 tokens each)
# Kept as a reporting ceiling only, not enforced (AgentExecutor can't cut a run short on it).
TOKEN_BUDGET = 50_000       # input + output tokens per question

# Haiku 4.5 list prices, dollars per million tokens; cache reads cost 0.1x and cache writes 1.25x input.
HAIKU_USD_PER_MTOK = (1.0, 5.0)    # input, output
CACHE_READ_FACTOR, CACHE_WRITE_FACTOR = 0.1, 1.25


def turn_cost_usd(input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    """Cost of model calls. input_tokens already includes the cache reads and writes (LangChain's convention)."""
    inp, out = HAIKU_USD_PER_MTOK
    fresh = input_tokens - cache_read - cache_write
    return (fresh * inp + cache_read * inp * CACHE_READ_FACTOR + cache_write * inp * CACHE_WRITE_FACTOR
            + output_tokens * out) / 1e6


# What AgentExecutor returns as `output` when max_iterations stops the run.
EXECUTOR_STOPPED = "Agent stopped due to max iterations."

# Sent (tool-free) after max_iterations, so the model answers with what it has.
LIMIT_REACHED_REQUEST = (
    "You've used all the tool calls allowed for this question. Answer now using only the tool results above. "
    "Say clearly what you couldn't get, and don't state any figure that isn't in those results."
)



def with_final_notice(agent, max_iterations: int):
    """
    Tell the model, on its last allowed call, that no more tool calls will run.

    AgentExecutor lets the model call tools until max_iterations and then just stops, so a model that
    keeps asking for tools on its last turn never writes an answer (the limit path then summarizes). This
    wraps the agent so that call gets LIMIT_REACHED_REQUEST appended after the tool results. The number
    of model calls so far is the number of distinct model responses in the intermediate steps (parallel
    tool calls from one response count once). max_iterations itself is unchanged.
    """
    def add_notice(inputs: dict) -> dict:
        responses = {frozenset(tc["id"] for tc in action.message_log[0].tool_calls)
                     for action, _ in inputs["intermediate_steps"] if getattr(action, "message_log", None)}
        last_call = len(responses) >= max_iterations - 1
        return {**inputs, "final_notice": [HumanMessage(LIMIT_REACHED_REQUEST)] if last_call else []}

    return RunnableLambda(add_notice) | agent


AGENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_turn_metrics (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT NOT NULL,
    turn_number        INTEGER NOT NULL,
    model              TEXT NOT NULL,
    iterations         INTEGER NOT NULL,
    tool_calls         INTEGER NOT NULL,
    exit_reason        TEXT NOT NULL,
    stop_reason        TEXT,
    input_tokens       INTEGER NOT NULL,
    output_tokens      INTEGER NOT NULL,
    cache_read_tokens  INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL,
    latency_ms         REAL NOT NULL,
    timestamp          TEXT NOT NULL,
    grounding_revisions  INTEGER NOT NULL DEFAULT 0,
    grounding_unresolved INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_metrics_session ON agent_turn_metrics (session_id, turn_number);

CREATE TABLE IF NOT EXISTS tool_call_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    turn_number   INTEGER NOT NULL,
    iteration     INTEGER NOT NULL,
    tool_use_id   TEXT NOT NULL,
    tool_name     TEXT NOT NULL,
    input_json    TEXT NOT NULL,
    is_error      INTEGER NOT NULL,
    error_type    TEXT,
    result_text   TEXT,
    from_cache    INTEGER,
    is_duplicate  INTEGER NOT NULL,
    latency_ms    REAL NOT NULL,
    reasoning     TEXT,
    timestamp     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_log_session ON tool_call_log (session_id, turn_number);
"""


@dataclass
class ToolCallRecord:
    iteration: int
    name: str
    input: dict
    is_error: bool
    latency_ms: float
    from_cache: bool | None
    is_duplicate: bool


@dataclass
class AgentResult:
    """What one call to ask() produced."""
    answer: str
    session_id: str
    turn_number: int
    exit_reason: str   # answered | max_iterations | max_tokens | refusal | api_error
    iterations: int
    stop_reason: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    grounding_revisions: int = 0     # revision requests sent (0 or 1)
    grounding_unresolved: int = 0    # flagged sentences still in the final answer
    limit_summarized: bool = False   # max_iterations answer came from the final tool-free call


class MetricsCallbackHandler(BaseCallbackHandler):
    """
    Rebuilds what Phase 2's loop computed inline: per-tool-call rows in
    tool_call_log, plus the per-turn totals that end up in agent_turn_metrics.

    Must be passed in invoke(config={"callbacks": [...]}), NOT to the
    AgentExecutor constructor: constructor callbacks are not inherited by the
    LLM and tool runs inside it, so they would never see them.
    """

    raise_error = False  # a logging bug must not kill the user's answer
    # Under ainvoke LangChain would run a sync handler on a thread pool, out of order and concurrently
    # with its siblings. Inline keeps every callback on the event loop, in order, so the counters below
    # need no locks; only the database write is serialized (db_lock), because it is shared with other threads.
    run_inline = True

    def __init__(self, session_id: str, turn_number: int, db, db_lock=None):
        self.session_id, self.turn_number, self.db = session_id, turn_number, db
        self.db_lock = db_lock or threading.RLock()
        self.iterations = 0
        self.count_iterations = True       # off for the grounding revision call
        self.tool_calls: list[ToolCallRecord] = []
        self.macro_context_used: list[dict] = []
        self.input_tokens = self.output_tokens = 0
        self.cache_read_tokens = self.cache_write_tokens = 0
        self.stop_reason: str | None = None
        self.finished = False              # on_agent_finish fired
        self._seen_calls: set[str] = set()
        self._reasoning: str | None = None
        self._announced: list[dict] = []   # tool_calls of the last model response (carry the toolu_ ids)
        self._open: dict[UUID, dict] = {}

    # -- model calls: iteration count, tokens, the "reason" text before tool calls

    def on_chat_model_start(self, serialized, messages, **kwargs: Any) -> None:
        if self.count_iterations:
            self.iterations += 1

    def on_llm_end(self, response, **kwargs: Any) -> None:
        message = response.generations[0][0].message
        usage = message.usage_metadata or {}
        details = usage.get("input_token_details") or {}
        # LangChain's input_tokens already includes cache reads and writes.
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self.cache_read_tokens += details.get("cache_read", 0)
        self.cache_write_tokens += details.get("cache_creation", 0)
        self.stop_reason = (message.response_metadata or {}).get("stop_reason") or self.stop_reason
        self._reasoning = text_of(message.content) or None
        self._announced = list(getattr(message, "tool_calls", None) or [])

    # -- tool calls

    def on_tool_start(self, serialized, input_str, *, run_id: UUID, inputs: dict | None = None, **kwargs: Any) -> None:
        name = serialized.get("name") or kwargs.get("name") or "unknown"
        tool_input = inputs if inputs is not None else {"input": input_str}
        call_key = f"{name}:{json.dumps(tool_input, sort_keys=True, default=str)}"
        is_duplicate = call_key in self._seen_calls   # observed and logged, never blocked
        self._seen_calls.add(call_key)
        announced = next((c for c in self._announced if c["name"] == name and c["args"] == tool_input), None)
        if announced:
            self._announced.remove(announced)
        self._open[run_id] = {"name": name, "input": tool_input, "started": time.perf_counter(),
                              "is_duplicate": is_duplicate, "iteration": self.iterations,
                              "reasoning": self._reasoning,
                              "tool_call_id": announced["id"] if announced else str(run_id)}

    def on_tool_end(self, output, *, run_id: UUID, **kwargs: Any) -> None:
        content = output.content if isinstance(output, BaseMessage) else str(output)
        execution = pop_execution(run_id)
        if execution is None:
            # The tool function never ran: LangChain rejected the arguments (schema validation).
            self._finish_tool(run_id, content, is_error=True, error_type="ValidationError", from_cache=None)
            return
        self._finish_tool(run_id, content, is_error=execution.is_error, error_type=execution.error_type,
                          from_cache=execution.from_cache, attempts=execution.attempts,
                          breaker_state=execution.breaker_state, approval_required=execution.approval_required,
                          approval_decision=execution.approval_decision, approval_reason=execution.approval_reason)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        pop_execution(run_id)
        self._finish_tool(run_id, str(error), is_error=True, error_type=type(error).__name__, from_cache=None)

    def _finish_tool(self, run_id: UUID, content: str, is_error: bool, error_type: str | None,
                     from_cache: bool | None, attempts: int | None = None, breaker_state: str | None = None,
                     approval_required: bool | None = None, approval_decision: str | None = None,
                     approval_reason: str | None = None) -> None:
        call = self._open.pop(run_id, None)
        if call is None:
            return
        latency = round((time.perf_counter() - call["started"]) * 1000, 1)
        self.tool_calls.append(ToolCallRecord(call["iteration"], call["name"], call["input"], is_error,
                                              latency, from_cache, call["is_duplicate"]))
        if call["name"] == "fetch_macro_context" and not is_error:
            self.macro_context_used.append(json.loads(content))
        with self.db_lock:
            self.db.execute(
                """INSERT INTO tool_call_log (session_id, turn_number, iteration, tool_use_id, tool_name, input_json,
                       is_error, error_type, result_text, from_cache, is_duplicate, latency_ms, reasoning, timestamp,
                       attempts, breaker_state, approval_required, approval_decision, approval_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.session_id, self.turn_number, call["iteration"], call["tool_call_id"], call["name"],
                 json.dumps(call["input"], default=str), int(is_error), error_type, content[:4000],
                 None if from_cache is None else int(from_cache), int(call["is_duplicate"]), latency,
                 call["reasoning"], utc_now_iso(), attempts, breaker_state,
                 None if approval_required is None else int(approval_required), approval_decision, approval_reason),
            )
            self.db.commit()

    def on_agent_finish(self, finish, **kwargs: Any) -> None:
        self.finished = True


def text_of(content) -> str:
    """Plain text of a message's content, which is a str or a list of content blocks."""
    if isinstance(content, str):
        return content.strip()
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text").strip()


class EconomicPulseAgent:
    """
    Tool-calling agent over the four Phase 1 tools.

    One instance = one conversation (session_id). History lives in SQLite, so
    a new instance with the same session_id picks the conversation back up.
    """

    def __init__(
        self,
        db_path: str = config.DB_PATH,
        session_id: str | None = None,
        llm: BaseChatModel | None = None,
        model: str = MODEL,
        max_iterations: int = MAX_ITERATIONS,
        history_turns: int = HISTORY_TURNS,
        today: date | None = None,
        approve=None,
    ):
        self.db_path = db_path
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.model = model
        self.max_iterations = max_iterations
        self.history_turns = history_turns
        self.today = today or date.today()
        # The Anthropic SDK underneath retries 429/5xx/connection errors with
        # exponential backoff; 4 retries rides out short rate-limit windows.
        self.llm = llm or ChatAnthropic(model=model, max_tokens=MAX_TOKENS, max_retries=4)

        # One connection shared by the caller's thread, the event-loop thread and to_thread workers, so it
        # must allow cross-thread use; db_lock serializes everything that writes through it.
        self.db_lock = threading.RLock()
        if db_path == ":memory:":
            self.db = initialize_database(db_path)
        else:
            initialize_database(db_path).close()
            self.db = sqlite3.connect(db_path, check_same_thread=False)
            self.db.row_factory = sqlite3.Row
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_lock = threading.Lock()
        self.db.executescript(AGENT_SCHEMA)
        self.db.executescript(entity_memory.SCHEMA)
        # Databases created before the grounding columns existed.
        columns = {r["name"] for r in self.db.execute("PRAGMA table_info(agent_turn_metrics)")}
        for column in ("grounding_revisions", "grounding_unresolved"):
            if column not in columns:
                self.db.execute(f"ALTER TABLE agent_turn_metrics ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
        # Phase 5: retry/breaker (5A) and approval (5B) bookkeeping on tool_call_log (additive, nullable).
        tool_columns = {r["name"] for r in self.db.execute("PRAGMA table_info(tool_call_log)")}
        for column, decl in (("attempts", "INTEGER"), ("breaker_state", "TEXT"),
                             ("approval_required", "INTEGER"), ("approval_decision", "TEXT"), ("approval_reason", "TEXT")):
            if column not in tool_columns:
                self.db.execute(f"ALTER TABLE tool_call_log ADD COLUMN {column} {decl}")
        self.db.commit()

        # Approval gate: `approve(request) -> bool` is asked only when a cost/scope trigger fires. Default
        # auto-approves (tests, --examples, non-interactive runs); the CLI and the app pass a real one.
        self.gate = ApprovalGate(approve=approve, session_cost=self._session_cost)
        self._handler: MetricsCallbackHandler | None = None   # the turn in flight, for the session-cost trigger
        tools = build_tools(db_path, self.today, self.gate)
        self.executor = AgentExecutor(
            agent=with_final_notice(create_tool_calling_agent(self.llm, tools, build_prompt_template(self.today)),
                                    max_iterations),
            tools=tools,
            max_iterations=max_iterations,
            return_intermediate_steps=True,
        )

    # ------------------------------------------------------------------ public

    def ask(self, question: str) -> AgentResult:
        """
        Answer one question, calling tools as needed (synchronous wrapper over aask()).

        Runs aask() on a private event loop kept on a background thread, so it works from the CLI,
        from tests, and from inside a thread that already has a running loop (Streamlit): the caller
        just blocks. The same loop is reused for every question, which matters because the Anthropic
        async client keeps connections bound to the loop that opened them. Async callers should
        `await aask()` instead.
        """
        with self._loop_lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="agent-loop", daemon=True).start()
                self._loop = loop
        return asyncio.run_coroutine_threadsafe(self.aask(question), self._loop).result()

    async def aask(self, question: str) -> AgentResult:
        """Answer one question, calling tools as needed. Never raises for tool or transient API errors."""
        started = time.perf_counter()
        turn_number = self._next_turn_number()
        result = AgentResult(answer="", session_id=self.session_id, turn_number=turn_number,
                             exit_reason="answered", iterations=0)
        handler = MetricsCallbackHandler(self.session_id, turn_number, self.db, self.db_lock)
        self._handler = handler
        self.gate.begin_turn()
        history = self._load_history()
        steps: list = []

        try:
            block = entity_memory.build_block(self.db, self.db_lock, self.session_id)
            outcome = await self.executor.ainvoke(
                {"input": question, "chat_history": history,
                 "entity_context": [SystemMessage(content=block)] if block else []},
                config={"callbacks": [handler]})
            steps = outcome["intermediate_steps"]
            # The remaining model calls (limit summary, grounding revision) use the blocking client, so
            # they run off the loop; nothing else is in flight by now.
            await asyncio.to_thread(self._finish_answer, question, outcome, steps, handler, result, history)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
            raise  # misconfiguration: fail loudly, a retry won't fix it
        except anthropic.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after", "a minute")
            result.answer = f"I'm being rate limited by the model API right now. Please try again in {retry_after} seconds."
            result.exit_reason = "api_error"
        except anthropic.APIError as exc:
            result.answer = f"I couldn't reach the model API ({type(exc).__name__}). Please try again shortly."
            result.exit_reason = "api_error"

        result.iterations = handler.iterations
        result.stop_reason = handler.stop_reason
        result.tool_calls = handler.tool_calls
        result.input_tokens, result.output_tokens = handler.input_tokens, handler.output_tokens
        result.cache_read_tokens, result.cache_write_tokens = handler.cache_read_tokens, handler.cache_write_tokens
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        self._log_turn(question, result, handler.macro_context_used)
        entity_memory.record(self.db, self.db_lock, self.session_id, result.turn_number,
                             entity_memory.extract_entities(result.tool_calls))
        self._handler = None
        return result

    def _session_cost(self) -> float:
        """Model cost of this session so far: finished turns from the database plus the turn in flight."""
        with self.db_lock:
            row = self.db.execute(
                """SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0),
                          COALESCE(SUM(cache_read_tokens), 0), COALESCE(SUM(cache_write_tokens), 0)
                   FROM agent_turn_metrics WHERE session_id = ?""", (self.session_id,)).fetchone()
        cost = turn_cost_usd(*row)
        handler = self._handler
        if handler is not None:
            cost += turn_cost_usd(handler.input_tokens, handler.output_tokens, handler.cache_read_tokens,
                                  handler.cache_write_tokens)
        return cost

    def _finish_answer(self, question: str, outcome: dict, steps: list, handler: MetricsCallbackHandler,
                       result: AgentResult, history: list[BaseMessage]) -> None:
        self._interpret(question, outcome, handler, result, history)
        if result.exit_reason == "answered" or result.exit_reason == "max_iterations" and result.limit_summarized:
            self._enforce_grounding(question, result, steps, history, handler)

    run = ask  # the spec calls it run()

    def _interpret(self, question: str, outcome: dict, handler: MetricsCallbackHandler, result: AgentResult,
                   history: list[BaseMessage]) -> None:
        """Turn the executor's output into answer + exit_reason."""
        output = outcome["output"]
        text = text_of(output) if not isinstance(output, str) else output.strip()
        if text == EXECUTOR_STOPPED:
            result.exit_reason = "max_iterations"
            result.answer = self._limit_reached_answer(question, outcome["intermediate_steps"], history, handler, result)
        elif handler.stop_reason == "refusal":
            result.answer, result.exit_reason = text or "I can't help with that request.", "refusal"
        elif handler.stop_reason == "max_tokens":
            result.answer = text + "\n\n[Answer truncated: hit the response length limit.]"
            result.exit_reason = "max_tokens"
        else:
            result.answer = text

    def _limit_reached_answer(self, question: str, steps: list, history: list[BaseMessage],
                              handler: MetricsCallbackHandler, result: AgentResult) -> str:
        """
        max_iterations stopped the run before the model wrote an answer. Make one
        final tool-free call so it answers from the data already gathered (the
        closest thing to Phase 2's forced answer). If that call fails or comes
        back empty, fall back to a fixed limit message.
        """
        messages = [SystemMessage(content=build_system_prompt(self.today)), *history, HumanMessage(question),
                    *format_to_tool_messages(steps), HumanMessage(LIMIT_REACHED_REQUEST)]
        try:
            text = self._invoke_without_tools(messages, handler)
        except anthropic.APIError:
            text = ""
        if text and handler.stop_reason in ("end_turn", "stop_sequence"):
            result.limit_summarized = True
            return text
        gathered = sorted({action.tool for action, _ in steps})
        return (f"I reached my limit of {self.max_iterations} steps for a single question before I could "
                f"finish. I did gather data from: {', '.join(gathered) or 'no tools'}. "
                "Try a narrower question (one series or a shorter period) and I'll answer it directly.")

    def _invoke_without_tools(self, messages: list[BaseMessage], handler: MetricsCallbackHandler) -> str:
        """A plain llm.invoke (no tools bound). Its tokens are counted; it isn't an agent iteration."""
        handler.count_iterations = False
        try:
            return text_of(self.llm.invoke(messages, config={"callbacks": [handler]}).content)
        finally:
            handler.count_iterations = True

    # -------------------------------------------------------------- grounding

    def _enforce_grounding(self, question: str, result: AgentResult, steps: list, history: list[BaseMessage],
                           handler: MetricsCallbackHandler) -> None:
        """
        If the answer names causes no tool result supports, ask for one revision.

        Evidence = tool results from this turn + earlier answers (follow-ups
        may rely on context already established). One revision at most; any
        sentences still flagged afterwards are counted, not retried.
        """
        evidence = " ".join([str(observation) for _, observation in steps]
                            + [m.content for m in history if isinstance(m, AIMessage)])
        flagged = find_unsupported(result.answer, evidence)
        if not flagged:
            return

        result.grounding_revisions = 1
        # The revision call is a plain llm.invoke with no tools bound. The tool
        # turns still go in as messages so the model can keep the facts it used.
        messages = [SystemMessage(content=build_system_prompt(self.today)), *history, HumanMessage(question), *format_to_tool_messages(steps),
                    AIMessage(result.answer), HumanMessage(revision_request(flagged))]
        try:
            revised = self._invoke_without_tools(messages, handler)
        except anthropic.APIError:
            result.grounding_unresolved = len(flagged)  # keep the original answer
            return
        if handler.stop_reason in ("end_turn", "stop_sequence") and revised:
            result.answer = revised
        result.grounding_unresolved = len(find_unsupported(result.answer, evidence))

    # ---------------------------------------------------------------- history

    def _next_turn_number(self) -> int:
        with self.db_lock:
            row = self.db.execute("SELECT MAX(turn_number) FROM conversation_log WHERE session_id = ?",
                                  (self.session_id,)).fetchone()
        return (row[0] or 0) + 1

    def _load_history(self) -> list[BaseMessage]:
        """
        Last HISTORY_TURNS question/answer pairs as plain text messages.

        Only final answers are replayed, not tool calls or raw data: the
        answers already quote the numbers, which keeps follow-ups cheap and
        the context bounded no matter how long the session runs.
        """
        with self.db_lock:
            rows = self.db.execute(
                """SELECT user_query, agent_response FROM conversation_log
                   WHERE session_id = ? AND agent_response IS NOT NULL AND agent_response != ''
                   ORDER BY turn_number DESC LIMIT ?""",
                (self.session_id, self.history_turns),
            ).fetchall()
        messages: list[BaseMessage] = []
        for row in reversed(rows):
            messages.append(HumanMessage(content=row["user_query"]))
            messages.append(AIMessage(content=row["agent_response"]))
        return messages

    # ---------------------------------------------------------------- logging

    def _log_turn(self, question: str, result: AgentResult, macro_context_used: list[dict]) -> None:
        now = utc_now_iso()
        with self.db_lock:
            self.db.execute(
                """INSERT INTO conversation_log (session_id, turn_number, user_query, agent_response, macro_context_used, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self.session_id, result.turn_number, question, result.answer,
                 json.dumps(macro_context_used) if macro_context_used else None, now),
            )
            self.db.execute(
                """INSERT INTO agent_turn_metrics (session_id, turn_number, model, iterations, tool_calls, exit_reason,
                       stop_reason, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, latency_ms, timestamp,
                       grounding_revisions, grounding_unresolved)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.session_id, result.turn_number, self.model, result.iterations, len(result.tool_calls),
                 result.exit_reason, result.stop_reason, result.input_tokens, result.output_tokens, result.cache_read_tokens,
                 result.cache_write_tokens, result.latency_ms, now, result.grounding_revisions, result.grounding_unresolved),
            )
            self.db.commit()


# --------------------------------------------------------------------------- CLI

EXAMPLE_QUESTIONS = [
    "What is the current unemployment rate?",
    "What is inflation running at right now, based on CPI?",
    f"How has the S&P 500 performed so far in {date.today().year}?",
    "How did the 10-year Treasury yield change between Q1 2025 and Q1 2026?",
    "How has the S&P 500 reacted to unemployment changes since 2024?",
    "Has anything unusual happened in CPI over the past year?",
    "What is going on with unemployment lately?",
]


def _format_result(r: AgentResult) -> str:
    calls = ", ".join(f"{c.name}({json.dumps(c.input)})" + (" [error]" if c.is_error else "") for c in r.tool_calls) or "none"
    return (f"Tool calls: {calls}\n"
            f"[{r.exit_reason}; {r.iterations} iterations; {r.input_tokens} in / {r.output_tokens} out tokens; "
            f"{r.latency_ms / 1000:.1f}s]")


def write_examples(path: Path) -> None:
    agent = EconomicPulseAgent()
    blocks = []
    for q in EXAMPLE_QUESTIONS:
        r = agent.ask(q)
        print(f"Q: {q}\n{_format_result(r)}\n")
        blocks.append(f"Q: {q}\n\nA: {r.answer}\n\n{_format_result(r)}")
    header = (f"Economic Pulse Agent: example Q&A (generated {date.today().isoformat()}, model {MODEL}, "
              f"one session so later answers can use earlier context)\n")
    path.write_text(header + "\n\n" + ("\n\n" + "=" * 78 + "\n\n").join(blocks) + "\n")
    print(f"Wrote {path}")


def cli_approve(request: ApprovalRequest) -> bool:
    """Ask on the terminal. EOF or anything but y/yes declines."""
    print(f"\n[approval needed] {request.tool}({json.dumps(request.args)})\n  because {request.reason}")
    try:
        return input("  run it? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def chat() -> None:
    # Ask for approval only when someone is at the keyboard; piped input auto-approves.
    agent = EconomicPulseAgent(approve=cli_approve if sys.stdin.isatty() else None)
    print(f"Economic Pulse Agent (session {agent.session_id}). Ctrl-D to quit.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if q:
            r = agent.ask(q)
            print(f"\nagent> {r.answer}\n\n{_format_result(r)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", action="store_true", help="run example questions and write examples.txt")
    args = parser.parse_args()
    if args.examples:
        write_examples(Path(__file__).resolve().parent / "examples.txt")
    else:
        chat()
