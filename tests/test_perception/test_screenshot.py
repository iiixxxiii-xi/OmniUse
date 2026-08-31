"""Task 2.4: screenshot capture + use_vision three-state policy."""

import base64

import pytest

from minicua.perception.extract import extract_state
from minicua.perception.screenshot import capture, should_capture


# --- should_capture (pure) ---------------------------------------------------


def test_should_capture_dom_only_never_captures():
    assert should_capture("dom_only", False) is False
    assert should_capture("dom_only", True) is False


def test_should_capture_vision_always_captures():
    assert should_capture("vision", True) is True
    assert should_capture("vision", False) is True


def test_should_capture_auto_follows_model_capability():
    assert should_capture("auto", True) is True
    assert should_capture("auto", False) is False


def test_should_capture_unknown_mode_raises():
    with pytest.raises(ValueError):
        should_capture("bogus", True)


# --- capture (integration) ---------------------------------------------------


@pytest.mark.asyncio
async def test_capture_returns_png_base64(session):
    await session.page.set_content("<h1>hello</h1>")
    b64 = await capture(session.page)
    assert b64
    data = base64.b64decode(b64)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


# --- extract_state vision wiring --------------------------------------------


@pytest.mark.asyncio
async def test_extract_state_dom_only_has_no_screenshot(session):
    await session.page.set_content("<button>ok</button>")
    state = await extract_state(session.page, use_vision="dom_only")
    assert state.screenshot is None
    assert "[1]" in state.dom_text


@pytest.mark.asyncio
async def test_extract_state_vision_captures_screenshot_and_dom(session):
    await session.page.set_content("<button>ok</button>")
    state = await extract_state(session.page, use_vision="vision")
    assert state.screenshot is not None
    assert state.dom_text  # DOM still present alongside vision


@pytest.mark.asyncio
async def test_extract_state_screenshot_failure_falls_back_to_dom(session, monkeypatch):
    await session.page.set_content("<button>ok</button>")

    async def fake_capture(page):
        return None

    monkeypatch.setattr("minicua.perception.extract.capture", fake_capture)
    state = await extract_state(session.page, use_vision="vision")
    assert state.screenshot is None
    assert "[1]" in state.dom_text
    assert state.selector_map  # DOM is never blocked by screenshot failure
