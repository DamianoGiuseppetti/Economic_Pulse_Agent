"""
Data-source resilience: bounded retries and a circuit breaker per source.

The model-API side already retries (ChatAnthropic max_retries). This is the data
side: FRED, Yahoo Finance and the Federal Reserve site.

Two rules shape it:
  - Everything lives inside the tool's 10 s budget. Each HTTP request gets a timeout derived
    from the time left (not a flat 30 s), and no retry or sleep runs past the deadline, so a
    retry can never outlive the tool timeout and leave an orphan thread writing to the cache.
  - Only transient failures (ConnectionError, TimeoutError) are retried and counted by the
    breaker. ValueError (bad request) means the source answered; RuntimeError (missing key)
    never reached the network. Neither counts as a failure.

A breaker "failure" is a call that still failed after its retries, not each attempt.
Cache hits never reach this module (they return before any fetch).
"""

import random
import threading
import time
from typing import Callable, TypeVar

import config

T = TypeVar("T")

SOURCES = {"fred": "FRED", "yfinance": "Yahoo Finance", "fed": "the Federal Reserve site"}

MAX_RETRIES = 2
BASE_BACKOFF_SECONDS = 0.3
MIN_ATTEMPT_SECONDS = 0.5     # a request given less time than this isn't worth sending
DEADLINE_MARGIN_SECONDS = 0.3  # left for the caller after the last request
FAILURE_THRESHOLD = 3
RESET_AFTER_SECONDS = 60.0

TRANSIENT = (ConnectionError, TimeoutError)


class SourceUnavailableError(ConnectionError):
    """The breaker for this source is open: the call was skipped without touching the network."""

    def __init__(self, source: str, retry_in: float):
        self.source, self.retry_in = source, retry_in
        super().__init__(
            f"{SOURCES.get(source, source)} has been failing, so requests to it are being skipped for "
            f"{max(1, round(retry_in))} more seconds. Answer from data you already have if it covers the question, "
            "say that data may be stale, and tell the user the live source is unavailable right now."
        )


class CircuitBreaker:
    """closed -> open after FAILURE_THRESHOLD consecutive failures -> half-open after the reset time (one probe) -> closed."""

    def __init__(self, source: str, *, failure_threshold: int = FAILURE_THRESHOLD,
                 reset_after: float = RESET_AFTER_SECONDS, clock: Callable[[], float] = time.monotonic):
        self.source, self.failure_threshold, self.reset_after, self.clock = source, failure_threshold, reset_after, clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state()

    def _state(self) -> str:
        if self._opened_at is None:
            return "closed"
        return "half_open" if self.clock() - self._opened_at >= self.reset_after else "open"

    def before_call(self) -> str:
        """Raise SourceUnavailableError if the call must be skipped; otherwise return the state it runs in."""
        with self._lock:
            state = self._state()
            if state == "closed":
                return "closed"
            if state == "half_open" and not self._probing:
                self._probing = True  # this call is the one probe
                return "half_open"
            remaining = self.reset_after - (self.clock() - self._opened_at) if state == "open" else 1.0
            raise SourceUnavailableError(self.source, max(remaining, 1.0))

    def on_success(self) -> None:
        with self._lock:
            self._failures, self._opened_at, self._probing = 0, None, False

    def on_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._probing or self._failures >= self.failure_threshold:
                self._opened_at = self.clock()  # (re)open; a failed probe restarts the wait
            self._probing = False

    def on_neutral(self) -> None:
        """The call said nothing about the source's health (e.g. missing API key): free a probe slot, change nothing."""
        with self._lock:
            self._probing = False


BREAKERS: dict[str, CircuitBreaker] = {name: CircuitBreaker(name) for name in SOURCES}


def reset_breakers() -> None:
    for breaker in BREAKERS.values():
        breaker.on_success()


# Per-call state of the tool running on this worker thread: its deadline (a time.monotonic()
# timestamp) and a stats dict that retry_call fills in (attempts, worst breaker state).
call_state = threading.local()


def seconds_left() -> float:
    return getattr(call_state, "deadline", float("inf")) - time.monotonic()


_STATE_RANK = {"closed": 0, "half_open": 1, "open": 2}


def _note(attempts: int, state: str) -> None:
    stats = getattr(call_state, "stats", None)
    if stats is None:
        return
    stats["attempts"] = stats.get("attempts", 0) + attempts
    if _STATE_RANK[state] >= _STATE_RANK.get(stats.get("breaker_state") or "closed", 0):
        stats["breaker_state"] = state


def retry_call(
    fn: Callable[[float], T],
    *,
    source: str,
    deadline: float | None = None,
    max_timeout: float | None = None,
    breaker: CircuitBreaker | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> T:
    """
    Call fn(timeout) with bounded retries.

    `timeout` is the HTTP timeout for that attempt: the smaller of max_timeout (default
    config.HTTP_TIMEOUT_SECONDS) and the time left before `deadline` (a monotonic timestamp;
    default is the running tool's deadline, else none). Retries only on TRANSIENT exceptions,
    at most MAX_RETRIES, exponential backoff with jitter, and never past the deadline: if
    there's no time for another attempt the last error is raised immediately.
    """
    breaker = breaker or BREAKERS[source]
    if deadline is None:
        deadline = getattr(call_state, "deadline", float("inf"))
    cap = max_timeout or config.HTTP_TIMEOUT_SECONDS

    try:
        state = breaker.before_call()
    except SourceUnavailableError:
        _note(0, "open")
        raise

    attempts = 0
    last_error: Exception | None = None
    try:
        while True:
            remaining = deadline - clock()
            timeout = min(cap, remaining - DEADLINE_MARGIN_SECONDS)
            if attempts == 0:
                if remaining <= 0:
                    raise TimeoutError(f"no time left to call {source}")
                timeout = max(timeout, MIN_ATTEMPT_SECONDS)
            attempts += 1
            try:
                result = fn(timeout)
            except SourceUnavailableError:
                raise
            except TRANSIENT as exc:
                last_error = exc
                delay = BASE_BACKOFF_SECONDS * (2 ** (attempts - 1)) * (0.5 + jitter())
                time_left = deadline - clock()
                if attempts > MAX_RETRIES or time_left < delay + DEADLINE_MARGIN_SECONDS + MIN_ATTEMPT_SECONDS:
                    breaker.on_failure()
                    raise
                sleep(delay)
                continue
            except RuntimeError:
                breaker.on_neutral()
                raise
            except Exception:
                breaker.on_success()  # e.g. ValueError: the source answered, the request was bad
                raise
            breaker.on_success()
            return result
    finally:
        # What the log shows: the state the call ran in, or "open" if this call's failure tripped the breaker.
        after = breaker.state
        _note(attempts, state if after == "closed" else after)
