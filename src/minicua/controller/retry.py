"""Model-call retry: classify failures and retry transient ones with backoff.

Built on :mod:`minicua.core.retry` (which already provides exponential backoff),
this module adds the model-specific classification the controller needs:

* :func:`classify_model_error` — maps any exception (typed :class:`ModelError` or
  a raw SDK/transport exception) to a machine-readable category.
* :func:`is_retryable_model_error` — True only for *transient* failures worth a
  blind retry (rate limit / timeout / server). A format error
  (:class:`ModelInvalidResponseError`) is *not* blind-retryable — the agent loop
  handles it by requerying with feedback, which is a different mechanism.
* :func:`retry_model_call` — wraps a model ``generate`` call with
  :func:`minicua.core.retry.async_retry` using that classifier.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from minicua.controller.llm import ModelError
from minicua.core.retry import RetryPolicy, async_retry

logger = logging.getLogger("minicua.controller.retry")

T = TypeVar("T")

#: Default backoff for model calls: up to 3 attempts, exponential, jittered.
MODEL_RETRY_POLICY = RetryPolicy(max_attempts=3, base_delay=0.5, max_delay=10.0, jitter=True)

#: Categories that warrant a blind retry (transient provider/transport issues).
RETRYABLE_CATEGORIES = frozenset({"rate_limit", "timeout", "server"})


def classify_model_error(exc: Exception) -> str:
    """Return a machine-readable category for a model/transport exception.

    Typed :class:`ModelError` subclasses carry their own category; raw exceptions
    (e.g. from an SDK/HTTP layer) are classified by inspecting their message.
    """
    if isinstance(exc, ModelError):
        return exc.category
    msg = str(exc).lower()
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "rate limit" in msg or "too many requests" in msg or "429" in msg:
        return "rate_limit"
    if any(t in msg for t in ("connection", "network", "refused", "reset", "503", "502", "500")):
        return "server"
    if any(t in msg for t in ("401", "403", "api key", "unauthorized", "authentication")):
        return "auth"
    return "unknown"


def is_retryable_model_error(exc: Exception) -> bool:
    """True only for transient model failures worth a blind retry."""
    if isinstance(exc, ModelError):
        return exc.retryable
    return classify_model_error(exc) in RETRYABLE_CATEGORIES


async def retry_model_call(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy = MODEL_RETRY_POLICY,
    logger_: logging.Logger | None = None,
) -> T:
    """Run a model ``generate`` call, retrying transient failures with backoff."""
    return await async_retry(
        fn,
        policy=policy,
        is_retryable=is_retryable_model_error,
        logger_=logger_ or logger,
    )
