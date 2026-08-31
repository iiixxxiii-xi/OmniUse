import pytest

from minicua.core.retry import RetryPolicy, exponential_backoff, async_retry


def test_exponential_backoff_grows():
    policy = RetryPolicy(base_delay=1.0, max_delay=100.0, jitter=False)
    assert exponential_backoff(0, policy) == 1.0
    assert exponential_backoff(1, policy) == 2.0
    assert exponential_backoff(2, policy) == 4.0


def test_exponential_backoff_capped():
    policy = RetryPolicy(base_delay=10.0, max_delay=15.0, jitter=False)
    assert exponential_backoff(5, policy) == 15.0


@pytest.mark.asyncio
async def test_async_retry_succeeds_after_transient_failures():
    policy = RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0, jitter=False)
    calls = []

    async def fn():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("transient")
        return "ok"

    result = await async_retry(fn, policy=policy, is_retryable=lambda e: True)
    assert result == "ok"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_async_retry_stops_on_non_retryable():
    policy = RetryPolicy(max_attempts=5, base_delay=0.0, max_delay=0.0, jitter=False)
    calls = []

    async def fn():
        calls.append(1)
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        await async_retry(fn, policy=policy, is_retryable=lambda e: False)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_async_retry_raises_after_max_attempts():
    policy = RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0, jitter=False)
    calls = []

    async def fn():
        calls.append(1)
        raise RuntimeError("always fails")

    with pytest.raises(RuntimeError):
        await async_retry(fn, policy=policy, is_retryable=lambda e: True)
    assert len(calls) == 3
