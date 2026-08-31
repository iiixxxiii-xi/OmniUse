"""Task 4.3: budget tracking + model-call retry (classification + backoff)."""

import pytest

from minicua.controller.budget import Budget
from minicua.controller.llm import (
    ModelAuthError,
    ModelInvalidResponseError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from minicua.controller.retry import (
    classify_model_error,
    is_retryable_model_error,
    retry_model_call,
)
from minicua.core.retry import RetryPolicy


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


def test_budget_max_steps_stops():
    b = Budget(max_steps=3)
    assert not b.exhausted()
    b.record_step()
    b.record_step()
    assert not b.steps_exhausted()
    b.record_step()
    assert b.steps_exhausted()
    assert b.exhausted()


def test_budget_max_failures_stops():
    b = Budget(max_failures=2)
    b.record_failure()
    b.record_failure()
    assert b.failures_exhausted()
    assert b.exhausted()


def test_budget_reset_failures():
    b = Budget(max_failures=2)
    b.record_failure()
    b.reset_failures()
    assert not b.failures_exhausted()


def test_budget_max_tokens_stops():
    b = Budget(max_tokens=100)
    b.record_tokens(90)
    assert not b.tokens_exhausted()
    b.record_tokens(20)
    assert b.tokens_exhausted()


def test_budget_max_cost_stops():
    b = Budget(max_cost_usd=0.5)
    b.record_cost(0.4)
    assert not b.cost_exhausted()
    b.record_cost(0.2)
    assert b.cost_exhausted()


def test_budget_timeout():
    b = Budget(timeout_seconds=10.0)
    b.start(now=1000.0)
    assert not b.timed_out(now=1005.0)
    assert b.timed_out(now=1010.0)
    assert b.exhausted(now=1010.0)


def test_budget_not_exhausted_initially():
    b = Budget()
    assert not b.exhausted()
    assert b.exhaustion_reason() is None


def test_budget_exhaustion_reason_priority():
    b = Budget(max_steps=1, max_failures=1)
    b.record_step()
    b.record_failure()
    assert b.exhaustion_reason() == "max_steps"


def test_budget_unbounded_limits_never_exhaust():
    b = Budget()  # only max_steps/max_failures bounded; tokens/cost/timeout unbounded
    assert not b.tokens_exhausted()
    assert not b.cost_exhausted()
    assert not b.timed_out()


# --------------------------------------------------------------------------- #
# Model-call retry
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_retry_model_call_retries_transient():
    policy = RetryPolicy(max_attempts=3, base_delay=0.0, jitter=False)
    attempts = 0

    async def fn() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ModelRateLimitError("rate limited")
        return "ok"

    result = await retry_model_call(fn, policy=policy)
    assert result == "ok"
    assert attempts == 2


@pytest.mark.asyncio
async def test_retry_model_call_does_not_retry_permanent():
    policy = RetryPolicy(max_attempts=3, base_delay=0.0, jitter=False)
    attempts = 0

    async def fn() -> str:
        nonlocal attempts
        attempts += 1
        raise ModelAuthError("bad key")

    with pytest.raises(ModelAuthError):
        await retry_model_call(fn, policy=policy)
    assert attempts == 1


@pytest.mark.asyncio
async def test_retry_model_call_exhausts_attempts():
    policy = RetryPolicy(max_attempts=2, base_delay=0.0, jitter=False)
    attempts = 0

    async def fn() -> str:
        nonlocal attempts
        attempts += 1
        raise ModelTimeoutError("timeout")

    with pytest.raises(ModelTimeoutError):
        await retry_model_call(fn, policy=policy)
    assert attempts == 2


def test_classify_model_error_uses_category():
    assert classify_model_error(ModelRateLimitError("x")) == "rate_limit"
    assert classify_model_error(ModelTimeoutError("x")) == "timeout"
    assert classify_model_error(ModelAuthError("x")) == "auth"
    assert classify_model_error(ModelInvalidResponseError("x")) == "invalid_response"


def test_classify_generic_exceptions():
    assert classify_model_error(TimeoutError("timed out")) == "timeout"
    assert classify_model_error(RuntimeError("HTTP 429 Too Many Requests")) == "rate_limit"
    assert classify_model_error(RuntimeError("connection reset by peer")) == "server"
    assert classify_model_error(RuntimeError("401 Unauthorized")) == "auth"
    assert classify_model_error(RuntimeError("something else entirely")) == "unknown"


def test_is_retryable_model_error():
    assert is_retryable_model_error(ModelRateLimitError("x")) is True
    assert is_retryable_model_error(ModelTimeoutError("x")) is True
    assert is_retryable_model_error(ModelAuthError("x")) is False
    # A format error is requeryable, not blind-retryable.
    assert is_retryable_model_error(ModelInvalidResponseError("x")) is False
    # Generic transient exceptions are classified retryable.
    assert is_retryable_model_error(TimeoutError("timed out")) is True
