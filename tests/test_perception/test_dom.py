"""Task 2.1: DOMElement / BrowserState pydantic models."""

import pytest
from pydantic import ValidationError

from minicua.perception.dom import BrowserState, DOMElement, ScrollInfo, Viewport


def test_dom_element_defaults():
    el = DOMElement(index=1, tag="button", text="登录")
    assert el.index == 1
    assert el.tag == "button"
    assert el.text == "登录"
    assert el.stable_hash == ""


def test_dom_element_optional_fields_have_sane_defaults():
    el = DOMElement(index=2, tag="input", text="")
    assert el.role is None
    assert el.xpath is None
    assert el.ax_name is None
    assert el.attributes == {}
    assert el.interactive is True
    assert el.visible is True
    assert el.disabled is False


def test_dom_element_full_construction():
    el = DOMElement(
        index=3,
        tag="a",
        text="首页",
        role="link",
        xpath="//a[1]",
        stable_hash="abc123",
        ax_name="Home",
        attributes={"href": "/", "title": "go home"},
        disabled=False,
    )
    assert el.xpath == "//a[1]"
    assert el.ax_name == "Home"
    assert el.attributes["href"] == "/"


def test_dom_element_rejects_empty_tag():
    with pytest.raises(ValidationError):
        DOMElement(index=1, tag="")


def test_dom_element_rejects_negative_index():
    with pytest.raises(ValidationError):
        DOMElement(index=-1, tag="button")


def test_browser_state_selector_map():
    st = BrowserState(
        url="x",
        dom_text="[1] <button>登录</button>",
        selector_map={1: DOMElement(index=1, tag="button", text="登录")},
    )
    assert st.selector_map[1].tag == "button"


def test_browser_state_defaults():
    st = BrowserState(url="x")
    assert st.url == "x"
    assert st.title == ""
    assert st.dom_text == ""
    assert st.selector_map == {}
    assert st.screenshot is None
    assert st.viewport is None
    assert st.scroll is None


def test_browser_state_requires_url():
    with pytest.raises(ValidationError):
        BrowserState()


def test_browser_state_carries_viewport_and_scroll():
    st = BrowserState(
        url="x",
        viewport=Viewport(width=1280, height=800),
        scroll=ScrollInfo(x=0, y=120, scroll_height=2000, client_height=800),
    )
    assert st.viewport.width == 1280
    assert st.scroll.y == 120
    assert st.scroll.scroll_height == 2000
