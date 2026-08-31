"""Task 7.1: eval getters — read final browser state for a declarative evaluator.

Each getter is a defensive async function ``(session, **config) -> Any`` that
never raises on a transient page problem (it degrades to ``None`` / ``False``).
The registry maps a declarative name to the callable, and ``get_getter`` raises a
typed :class:`GetterError` for an unknown name so a bad task JSON fails loudly
rather than silently scoring 0.
"""

import pytest

from minicua.eval.errors import GetterError
from minicua.eval.getters import (
    GETTERS,
    cookie_exists,
    element_attribute,
    element_count,
    element_exists,
    element_text,
    get_getter,
    local_storage,
    page_text,
    page_title,
    page_url,
)


async def _serve(session, html: str, url: str = "http://minicua.test/"):
    """Serve inline HTML on a synthetic origin so cookies/localStorage are available."""
    async def handler(route):
        await route.fulfill(status=200, content_type="text/html", body=html)

    await session.context.route("http://minicua.test/**", handler)
    await session.page.goto(url)
    return url


# --------------------------------------------------------------------------- #
# URL / title / page text
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_page_url_getter(session):
    await session.page.goto("data:text/html,<div>x</div>")
    assert await page_url(session) == session.page.url


@pytest.mark.asyncio
async def test_page_title_getter(session):
    await session.page.set_content("<html><head><title>T</title></head><body></body></html>")
    assert await page_title(session) == "T"


@pytest.mark.asyncio
async def test_page_text_getter(session):
    await session.page.set_content("<div>hello world</div>")
    assert "hello world" in await page_text(session)


# --------------------------------------------------------------------------- #
# element getters
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_element_exists_getter(session):
    await session.page.set_content("<div id=b>x</div>")
    assert await element_exists(session, selector="#b") is True
    assert await element_exists(session, selector="#none") is False


@pytest.mark.asyncio
async def test_element_text_getter(session):
    await session.page.set_content("<div id=b>hello</div>")
    assert await element_text(session, selector="#b") == "hello"


@pytest.mark.asyncio
async def test_element_text_getter_missing_returns_none(session):
    await session.page.set_content("<div id=b>hello</div>")
    assert await element_text(session, selector="#none") is None


@pytest.mark.asyncio
async def test_element_attribute_getter(session):
    await session.page.set_content("<input id=b value=alice>")
    assert await element_attribute(session, selector="#b", attribute="value") == "alice"


@pytest.mark.asyncio
async def test_element_attribute_getter_missing_returns_none(session):
    await session.page.set_content("<input id=b value=alice>")
    assert await element_attribute(session, selector="#none", attribute="value") is None


@pytest.mark.asyncio
async def test_element_count_getter(session):
    await session.page.set_content("<ul><li>a</li><li>b</li><li>c</li></ul>")
    assert await element_count(session, selector="li") == 3
    assert await element_count(session, selector="p") == 0


# --------------------------------------------------------------------------- #
# cookie / localStorage getters
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cookie_exists_getter(session):
    await _serve(session, "<div>x</div>")
    await session.context.add_cookies([{"name": "session", "value": "abc", "url": "http://minicua.test"}])
    assert await cookie_exists(session, name="session") is True
    assert await cookie_exists(session, name="missing") is False


@pytest.mark.asyncio
async def test_local_storage_getter(session):
    await _serve(session, "<div>x</div>")
    await session.page.evaluate("localStorage.setItem('token', 'xyz')")
    assert await local_storage(session, key="token") == "xyz"
    assert await local_storage(session, key="missing") is None


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_registry_contains_expected_getters():
    for name in (
        "page_url",
        "page_title",
        "page_text",
        "element_exists",
        "element_text",
        "element_attribute",
        "element_count",
        "cookie_exists",
        "local_storage",
        "screenshot",
    ):
        assert name in GETTERS


def test_get_getter_unknown_raises():
    with pytest.raises(GetterError):
        get_getter("does_not_exist")
