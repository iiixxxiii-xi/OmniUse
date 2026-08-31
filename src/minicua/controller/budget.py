"""Execution budget: bound the agent loop across steps, failures, tokens, cost and time.

The agent consults its :class:`Budget` before every step. Limits are declared up
front (``max_steps``, ``max_failures``, ``max_tokens``, ``max_cost_usd``,
``timeout_seconds``); counters are mutated by the loop as it goes. Exhaustion is
checked with :meth:`Budget.exhausted` and the *reason* (for a clean terminal
result) with :meth:`Budget.exhaustion_reason`.

``timeout_seconds`` uses a monotonic clock via :meth:`Budget.start` so wall-clock
changes (NTP, DST) cannot affect the deadline. ``now`` is injectable for tests.
"""

import time

from pydantic import BaseModel, Field


class Budget(BaseModel):
    """Bounds + runtime counters for one agent run."""

    # -- limits -------------------------------------------------------------
    max_steps: int = Field(default=100, ge=1)
    max_failures: int = Field(default=3, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, ge=0.0)
    timeout_seconds: float | None = Field(default=None, gt=0.0)

    # -- runtime counters ---------------------------------------------------
    steps: int = 0
    failures: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    started_at: float | None = None

    # -- clock --------------------------------------------------------------

    def start(self, now: float | None = None) -> None:
        """Begin the timeout clock (idempotent; ``now`` injectable for tests)."""
        self.started_at = now if now is not None else time.monotonic()

    # -- recording ----------------------------------------------------------

    def record_step(self) -> None:
        self.steps += 1

    def record_failure(self) -> None:
        self.failures += 1

    def reset_failures(self) -> None:
        """Clear the failure count (called when an action succeeds)."""
        self.failures = 0

    def record_tokens(self, n: int) -> None:
        self.tokens += n

    def record_cost(self, usd: float) -> None:
        self.cost_usd += usd

    # -- exhaustion checks --------------------------------------------------

    def steps_exhausted(self) -> bool:
        return self.steps >= self.max_steps

    def failures_exhausted(self) -> bool:
        return self.failures >= self.max_failures

    def tokens_exhausted(self) -> bool:
        return self.max_tokens is not None and self.tokens >= self.max_tokens

    def cost_exhausted(self) -> bool:
        return self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd

    def timed_out(self, now: float | None = None) -> bool:
        if self.timeout_seconds is None or self.started_at is None:
            return False
        t = now if now is not None else time.monotonic()
        return (t - self.started_at) >= self.timeout_seconds

    def exhausted(self, now: float | None = None) -> bool:
        """True when *any* limit is reached."""
        return (
            self.steps_exhausted()
            or self.failures_exhausted()
            or self.timed_out(now)
            or self.tokens_exhausted()
            or self.cost_exhausted()
        )

    def exhaustion_reason(self, now: float | None = None) -> str | None:
        """The first limit hit (priority order), or ``None`` when not exhausted."""
        if self.steps_exhausted():
            return "max_steps"
        if self.failures_exhausted():
            return "max_failures"
        if self.timed_out(now):
            return "timeout"
        if self.tokens_exhausted():
            return "max_tokens"
        if self.cost_exhausted():
            return "max_cost"
        return None
