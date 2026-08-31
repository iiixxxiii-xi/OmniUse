import pytest

from minicua.core.errors import (
    CUAError,
    StaleElementError,
    PageChangedError,
    CrashError,
    LoopDetected,
    BrowserError,
    BrowserStartupError,
    NavigationError,
    BrowserConnectionError,
    BrowserTimeoutError,
)


def test_errors_have_messages():
    assert str(StaleElementError(index=5)) == "Element index 5 is stale"
    assert str(PageChangedError(before="a.com", after="b.com")) == "Page changed from a.com to b.com"


def test_loop_detected_is_soft():
    err = LoopDetected(repeat_count=5)
    assert err.repeat_count == 5


def test_browser_errors_are_cua_errors():
    for cls in (BrowserError, BrowserStartupError, NavigationError, BrowserConnectionError, BrowserTimeoutError):
        assert issubclass(cls, CUAError)


def test_navigation_error_carries_retryable():
    err = NavigationError(url="http://x", reason="net::ERR_CONNECTION_REFUSED", retryable=True)
    assert err.retryable is True
    assert err.url == "http://x"
    assert "ERR_CONNECTION_REFUSED" in str(err)

    err2 = NavigationError(url="http://x", reason="invalid url", retryable=False)
    assert err2.retryable is False


def test_connection_and_timeout_are_retryable():
    assert BrowserConnectionError().retryable is True
    assert BrowserTimeoutError().retryable is True


def test_default_cua_error_not_retryable():
    assert CUAError().retryable is False
