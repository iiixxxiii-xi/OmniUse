"""Task 2.3: extract selector_map + BrowserState from a live Playwright page."""

import pytest

from minicua.perception.extract import extract_state


@pytest.mark.asyncio
async def test_extract_from_page(session):
    await session.page.set_content("<button>登录</button><input type=text placeholder=用户名>")
    state = await extract_state(session.page)
    assert state.selector_map[1].tag == "button"
    assert "[1]" in state.dom_text


@pytest.mark.asyncio
async def test_extract_empty_page(session):
    await session.page.set_content("<div></div>")
    state = await extract_state(session.page)
    assert state.selector_map == {}
    assert state.dom_text == ""


@pytest.mark.asyncio
async def test_extract_no_interactive_elements(session):
    await session.page.set_content("<h1>Only a heading</h1>")
    state = await extract_state(session.page)
    assert state.selector_map == {}
    assert "Only a heading" in state.dom_text


@pytest.mark.asyncio
async def test_extract_skips_hidden_elements(session):
    await session.page.set_content("<button style='display:none'>hidden</button><button>shown</button>")
    state = await extract_state(session.page)
    assert len(state.selector_map) == 1
    assert state.selector_map[1].text == "shown"


@pytest.mark.asyncio
async def test_extract_marks_disabled(session):
    await session.page.set_content("<button disabled>go</button>")
    state = await extract_state(session.page)
    assert state.selector_map[1].disabled is True


@pytest.mark.asyncio
async def test_extract_captures_url_title_viewport(session):
    await session.page.set_content(
        "<html><head><title>MyTitle</title></head><body><button>ok</button></body></html>"
    )
    state = await extract_state(session.page)
    assert state.title == "MyTitle"
    assert state.url != ""
    assert state.viewport is not None and state.viewport.width > 0


@pytest.mark.asyncio
async def test_extract_input_attributes(session):
    await session.page.set_content("<input type=text placeholder=用户名 name=user>")
    state = await extract_state(session.page)
    el = state.selector_map[1]
    assert el.attributes["type"] == "text"
    assert el.attributes["placeholder"] == "用户名"


@pytest.mark.asyncio
async def test_extract_assigns_indexes_in_document_order(session):
    await session.page.set_content("<button>First</button><a href='/'>Second</a><button>Third</button>")
    state = await extract_state(session.page)
    assert state.selector_map[1].text == "First"
    assert state.selector_map[2].text == "Second"
    assert state.selector_map[3].text == "Third"


@pytest.mark.asyncio
async def test_extract_distinct_xpaths_for_siblings(session):
    await session.page.set_content("<button>A</button><button>B</button>")
    state = await extract_state(session.page)
    assert state.selector_map[1].xpath != state.selector_map[2].xpath


@pytest.mark.asyncio
async def test_extract_populates_ax_name_from_aria_label(session):
    await session.page.set_content("<button aria-label='Submit form'>Go</button>")
    state = await extract_state(session.page)
    assert state.selector_map[1].ax_name == "Submit form"


@pytest.mark.asyncio
async def test_extract_ax_name_falls_back_to_text(session):
    await session.page.set_content("<button>Plain</button>")
    state = await extract_state(session.page)
    assert state.selector_map[1].ax_name == "Plain"
