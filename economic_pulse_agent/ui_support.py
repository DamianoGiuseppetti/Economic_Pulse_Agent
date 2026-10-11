"""
Helpers for the Streamlit app (app.py), kept free of Streamlit calls so they can be imported and tested.

Why this exists: the agent's approval hook is a blocking call, approve(request) -> bool, but Streamlit
reruns the whole script on every click, so the script thread can't sit inside a call and also draw buttons.
So a question runs on a worker thread (Job); when the approval gate asks, UIApprover parks that thread on an
Event and exposes the pending request; the script, polling from its own thread, draws Approve / Decline
buttons, a click calls decide(), and the worker resumes. An unanswered request declines after a timeout.
"""

import threading
from dataclasses import dataclass, field

from agent import AgentResult, turn_cost_usd
from approval import ApprovalRequest

APPROVAL_TIMEOUT_SECONDS = 300


class UIApprover:
    """Callable approve(request) -> bool for the agent's gate, answered from another thread via decide()."""

    def __init__(self, timeout: float = APPROVAL_TIMEOUT_SECONDS):
        self.timeout = timeout
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._decision = False
        self._pending: ApprovalRequest | None = None

    @property
    def pending(self) -> ApprovalRequest | None:
        with self._lock:
            return self._pending

    def __call__(self, request: ApprovalRequest) -> bool:
        with self._lock:
            self._pending, self._decision = request, False
            self._event.clear()
        answered = self._event.wait(self.timeout)
        with self._lock:
            self._pending = None
            return self._decision if answered else False

    def decide(self, approved: bool) -> None:
        with self._lock:
            if self._pending is None:
                return
            self._decision, self._pending = approved, None
        self._event.set()


@dataclass
class Job:
    """One question running on a worker thread."""
    question: str
    result: AgentResult | None = None
    error: BaseException | None = None
    thread: threading.Thread | None = field(default=None, repr=False)

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


def start_job(agent, question: str) -> Job:
    job = Job(question)

    def work():
        try:
            job.result = agent.ask(question)
        except BaseException as exc:  # shown in the chat, not lost in a thread
            job.error = exc

    job.thread = threading.Thread(target=work, name="agent-job", daemon=True)
    job.thread.start()
    return job


def grounding_status(r: AgentResult) -> str:
    if r.grounding_revisions == 0:
        return "passed"
    return "revised once, now clean" if r.grounding_unresolved == 0 else f"revised once, {r.grounding_unresolved} sentence(s) still unverified"


def cache_status(r: AgentResult) -> str:
    if r.cache_read_tokens:
        return f"hit ({r.cache_read_tokens:,} tokens read from the prompt cache)"
    if r.cache_write_tokens:
        return f"miss, wrote {r.cache_write_tokens:,} tokens to the cache"
    return "not used"


def turn_details(r: AgentResult) -> dict:
    """Everything the expander under an answer shows."""
    return {
        "tool_calls": [
            {"tool": c.name, "arguments": c.input,
             "status": "error" if c.is_error else "ok",
             "data": {None: "n/a", True: "from cache", False: "fetched"}[c.from_cache],
             "latency_ms": round(c.latency_ms), "iteration": c.iteration, "duplicate": c.is_duplicate}
            for c in r.tool_calls],
        "iterations": r.iterations,
        "tokens_in": r.input_tokens,
        "tokens_out": r.output_tokens,
        "cost_usd": round(turn_cost_usd(r.input_tokens, r.output_tokens, r.cache_read_tokens, r.cache_write_tokens), 4),
        "grounding": grounding_status(r),
        "prompt_cache": cache_status(r),
        "exit_reason": r.exit_reason,
        "seconds": round(r.latency_ms / 1000, 1),
    }
