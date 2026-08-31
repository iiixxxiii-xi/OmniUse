"""Task 3.2: grounding — map a model-emitted index back to a real DOM element."""

import pytest

from minicua.action.grounding import ground, to_locator
from minicua.core.errors import StaleElementError
from minicua.perception.dom import DOMElement


def test_ground_by_index_returns_element():
    el = DOMElement(index=1, tag="button", text="登录", xpath="//button[1]")
    selector_map = {1: el}
    assert ground(1, selector_map) is el


def test_ground_by_index_preserves_identity():
    selector_map = {
        1: DOMElement(index=1, tag="button", text="a", xpath="//button[1]"),
        2: DOMElement(index=2, tag="a", text="b", xpath="//a[1]"),
    }
    assert ground(2, selector_map).text == "b"


def test_ground_missing_index_raises_stale():
    with pytest.raises(StaleElementError):
        ground(99, {1: DOMElement(index=1, tag="button", text="x", xpath="//button")})


def test_ground_zero_index_raises_stale():
    # Selector maps are 1-based (serializer start_index=1); 0 is never valid.
    with pytest.raises(StaleElementError):
        ground(0, {1: DOMElement(index=1, tag="button", text="x", xpath="//button")})


def test_ground_negative_index_raises_stale():
    with pytest.raises(StaleElementError):
        ground(-1, {1: DOMElement(index=1, tag="button", text="x", xpath="//button")})


@pytest.mark.asyncio
async def test_to_locator_builds_xpath_locator(session):
    await session.page.set_content("<button id='b'>go</button>")
    el = DOMElement(index=1, tag="button", text="go", xpath="//*[@id='b']")
    loc = to_locator(el, session.page)
    assert await loc.count() == 1
    assert (await loc.inner_text()) == "go"


@pytest.mark.asyncio
async def test_to_locator_missing_xpath_raises_stale(session):
    el = DOMElement(index=1, tag="button", text="go", xpath=None)
    with pytest.raises(StaleElementError):
        to_locator(el, session.page)
