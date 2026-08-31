"""Retry utilities with exponential backoff (shared by browser + controller)."""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")

logger = logging.getLogger("minicua.core.retry")


class RetryPolicy(BaseModel):
    """Configuration for retrying a transient operation."""

    max_attempts: int = Field(default=3, ge=1)
    base_delay: float = Field(default=0.5, ge=0.0)
    max_delay: float = Field(default=10.0, ge=0.0)
    jitter: bool = True


def exponential_backoff(attempt: int, policy: RetryPolicy) -> float:
    """Delay in seconds for a 0-based attempt, capped at ``max_delay``."""
    delay = policy.base_delay * (2 ** attempt)
    delay = min(delay, policy.max_delay)
    if policy.jitter:
        delay *= random.uniform(0.5, 1.5)
    return delay


async def async_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[Exception], bool],
    logger_: logging.Logger | None = None,
) -> T:
    """Run ``fn``, retrying transient failures with exponential backoff.

    ``is_retryable`` classifies an exception: transient (True) or permanent
    (False). Permanent failures are re-raised immediately without retrying.
    """
    log = logger_ or logger
    last_exc: Exception | None = None
    for attempt in range(policy.max_attempts):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if not is_retryable(exc) or attempt == policy.max_attempts - 1:
                raise
            delay = exponential_backoff(attempt, policy)
            log.warning(
                "retryable failure (attempt %d/%d): %s; retrying in %.2fs",
                attempt + 1,
                policy.max_attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None  # pragma: no cover - loop always raises first
    raise last_exc
