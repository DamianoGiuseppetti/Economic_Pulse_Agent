"""
Approval gate for expensive tool calls (human in the loop, on cost and scope).

All six tools are read-only, so nothing here is about safety: the gate protects cost, latency and API
rate limits. The same hook is where a write-capable tool (send, place, delete) would attach: it asks
before the call runs and can refuse it.

ApprovalPolicy is a pure function of (tool, arguments, turn state, session state): no I/O, no model call.
ApprovalGate holds the per-turn counter and calls the injected approve(request) -> bool callback.
Defaults auto-approve, so tests, --examples and any non-interactive run behave as before; the CLI and the
Streamlit app inject a callback that really asks.

Triggers (thresholds in config.py):
  observations  a cold-cache fetch estimated above APPROVAL_MAX_OBSERVATIONS (from frequency x window)
  tool_calls    more than APPROVAL_MAX_TOOL_CALLS_PER_TURN calls already made this turn
  session_cost  cumulative session cost above APPROVAL_SESSION_COST_USD
"""

import math
import threading
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import config
from tools.anomaly import SERIES_RULES

PER_YEAR = {"day": 252, "week": 52, "month": 12, "quarter": 4}   # observations per year by frequency
STOCK_PER_YEAR = {"1d": 252, "1wk": 52, "1mo": 12}


@dataclass(frozen=True)
class ApprovalRequest:
    tool: str
    args: dict
    triggers: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class Decision:
    """What the gate decided about one call. required=False means no trigger fired (nothing was asked)."""
    required: bool = False
    decision: str | None = None      # approved | auto_approved | declined
    reason: str | None = None

    @property
    def declined(self) -> bool:
        return self.decision == "declined"


def _years(start: str, end: str) -> float:
    return max((date.fromisoformat(end) - date.fromisoformat(start)).days, 0) / 365.25


def estimate_observations(tool: str, args: dict) -> int | None:
    """
    Observations a fetch for these arguments would bring back, from frequency x window length.
    None when the tool doesn't fetch a series (macro context) or the arguments don't say.
    Unknown FRED series are assumed monthly; the 14 supported series use their real frequency.
    """
    try:
        if tool in ("get_economic_series", "detect_anomalies", "generate_narrative"):
            rule = SERIES_RULES.get(args["series_id"])
            per_year = PER_YEAR[rule.period] if rule else PER_YEAR["month"]
            return math.ceil(_years(args["start_date"], args["end_date"]) * per_year)
        if tool == "compare_periods":
            rule = SERIES_RULES.get(args["series_id"])
            per_year = PER_YEAR[rule.period] if rule else PER_YEAR["month"]
            years = _years(args["p1_start"], args["p1_end"]) + _years(args["p2_start"], args["p2_end"])
            return math.ceil(years * per_year)
        if tool == "get_stock_data":
            return math.ceil(_years(args["start_date"], args["end_date"]) * STOCK_PER_YEAR.get(args.get("interval", "1d"), 252))
    except (KeyError, ValueError):
        return None
    return None


@dataclass
class ApprovalPolicy:
    max_observations: int = field(default_factory=lambda: config.APPROVAL_MAX_OBSERVATIONS)
    max_tool_calls: int = field(default_factory=lambda: config.APPROVAL_MAX_TOOL_CALLS_PER_TURN)
    max_session_cost: float = field(default_factory=lambda: config.APPROVAL_SESSION_COST_USD)

    def check(self, tool: str, args: dict, *, calls_made: int, session_cost: float, cold: bool | None) -> ApprovalRequest | None:
        """None if the call may run unasked; otherwise the request to put to the user."""
        triggers, reasons = [], []
        estimate = estimate_observations(tool, args)
        if cold and estimate is not None and estimate > self.max_observations:
            triggers.append("observations")
            reasons.append(f"it would download about {estimate:,} observations that aren't cached "
                           f"(limit {self.max_observations:,})")
        if calls_made > self.max_tool_calls:
            triggers.append("tool_calls")
            reasons.append(f"{calls_made} tool calls were already made for this question (limit {self.max_tool_calls})")
        if session_cost > self.max_session_cost:
            triggers.append("session_cost")
            reasons.append(f"this session has already cost about ${session_cost:.3f} (limit ${self.max_session_cost:.2f})")
        if not triggers:
            return None
        return ApprovalRequest(tool, dict(args), tuple(triggers), "; ".join(reasons))


def auto_approve(request: ApprovalRequest) -> bool:
    return True


auto_approve.auto = True  # logged as auto_approved, not approved


class ApprovalGate:
    """Per-agent state for the policy: counts this turn's calls and asks the injected callback."""

    def __init__(self, policy: ApprovalPolicy | None = None, approve: Callable[[ApprovalRequest], bool] | None = None,
                 session_cost: Callable[[], float] | None = None):
        self.policy = policy or ApprovalPolicy()
        self.approve = approve or auto_approve
        self.session_cost = session_cost or (lambda: 0.0)
        self._calls_this_turn = 0
        self._count_lock = threading.Lock()
        self._ask_lock = threading.Lock()   # parallel tool calls must not interleave their prompts

    def begin_turn(self) -> None:
        with self._count_lock:
            self._calls_this_turn = 0

    def review(self, tool: str, args: dict, cold: bool | None) -> Decision:
        with self._count_lock:
            calls_made = self._calls_this_turn
            self._calls_this_turn += 1
        request = self.policy.check(tool, args, calls_made=calls_made, session_cost=self.session_cost(), cold=cold)
        if request is None:
            return Decision()
        with self._ask_lock:
            approved = bool(self.approve(request))
        if not approved:
            return Decision(True, "declined", request.reason)
        return Decision(True, "auto_approved" if getattr(self.approve, "auto", False) else "approved", request.reason)
