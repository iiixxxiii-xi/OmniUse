import asyncio
import socket

import pytest

from minicua.browser.session import BrowserSession, is_retryable_navigation_error
from minicua.core.errors import BrowserError, BrowserStartupError, NavigationError


def _free_port() -> int:
    """Reserve an ephemeral free port and release it, so a connection is refused."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- Task 1.1: lifecycle --------------------------------------------------


@pytest.mark.asyncio
async def test_session_start_and_close():
    s = BrowserSession(headless=True)
    await s.start()
    assert s.context is not None
    await s.close()


@pytest.mark.asyncio
async def test_session_as_context_manager():
    async with BrowserSession(headless=True) as s:
        assert s.context is not None
    assert s.context is None


@pytest.mark.asyncio
async def test_close_is_idempotent():
    s = BrowserSession(headless=True)
    await s.start()
    await s.close()
    await s.close()  # second close must be a no-op, not raise


@pytest.mark.asyncio
async def test_navigate_before_start_raises_clear_error():
    s = BrowserSession(headless=True)
    with pytest.raises(BrowserError):
        await s.navigate("data:text/html,<h1>x</h1>")


# --- Task 1.2: storage_state ----------------------------------------------


@pytest.mark.asyncio
async def test_storage_state_roundtrip(tmp_path):
    s = BrowserSession(headless=True)
    await s.start()
    await s.context.add_cookies([{"name": "k", "value": "v", "url": "https://example.com"}])
    path = tmp_path / "state.json"
    await s.save_storage_state(path)
    await s.close()
    assert path.exists()

    s2 = BrowserSession(headless=True, storage_state=path)
    await s2.start()
    cookies = await s2.context.cookies("https://example.com")
    assert any(c["name"] == "k" and c["value"] == "v" for c in cookies)
    await s2.close()


# --- Task 1.3: navigate ---------------------------------------------------


@pytest.mark.asyncio
async def test_navigate_and_read_state():
    s = BrowserSession(headless=True)
    await s.start()
    await s.navigate("data:text/html,<title>MyTitle</title>")
    assert s.url.startswith("data:text/html")
    assert await s.get_title() == "MyTitle"
    await s.close()


def test_classify_retryable_errors():
    assert is_retryable_navigation_error(asyncio.TimeoutError()) is True
    assert is_retryable_navigation_error(RuntimeError("net::ERR_CONNECTION_REFUSED")) is True
    assert is_retryable_navigation_error(RuntimeError("page.goto: Timeout 100ms exceeded")) is True


def test_classify_non_retryable_errors():
    assert is_retryable_navigation_error(RuntimeError("Cannot navigate to invalid URL")) is False
    assert is_retryable_navigation_error(RuntimeError("net::ERR_ABORTED")) is False


@pytest.mark.asyncio
async def test_navigate_unreachable_raises_retryable():
    s = BrowserSession(headless=True, max_navigation_retries=0, navigation_timeout_ms=3000)
    await s.start()
    with pytest.raises(NavigationError) as exc_info:
        await s.navigate(f"http://127.0.0.1:{_free_port()}/")
    assert exc_info.value.retryable is True
    await s.close()


@pytest.mark.asyncio
async def test_navigate_invalid_url_raises_non_retryable():
    s = BrowserSession(headless=True, max_navigation_retries=2)
    await s.start()
    with pytest.raises(NavigationError) as exc_info:
        await s.navigate("http://[")
    assert exc_info.value.retryable is False
    await s.close()
