"""Task 3.4: executor — run actions against a live Playwright page, structured results."""

import pytest

from minicua.action.executor import execute
from minicua.action.models import (
    Action,
    ActionError,
    ClickParams,
    DoneParams,
    NavigateParams,
    PressParams,
    ScrollParams,
    SwitchTabParams,
    TypeParams,
    WaitParams,
)
from minicua.perception.extract import extract_state


# --------------------------------------------------------------------------- #
# click
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_execute_click(session):
    await session.page.set_content(
        "<button id=b onclick=\"this.textContent='clicked'\">go</button>"
    )
    state = await extract_state(session.page)
    res = await execute(Action(name="click", params=ClickParams(index=1)), session.page, state)
    assert res.success is True
    assert await session.page.inner_text("#b") == "clicked"


@pytest.mark.asyncio
async def test_click_stale_index_returns_structured_error(session):
    await session.page.set_content("<button>go</button>")
    state = await extract_state(session.page)
    res = await execute(Action(name="click", params=ClickParams(index=99)), session.page, state)
    assert res.success is False
    assert res.error_code == ActionError.STALE_ELEMENT
    assert res.retryable is True


@pytest.mark.asyncio
async def test_click_page_changed_returns_not_found(session):
    await session.page.set_content("<button id=b>go</button>")
    state = await extract_state(session.page)
    # Page changes after perception; the stale xpath no longer resolves.
    await session.page.set_content("<div>changed</div>")
    res = await execute(Action(name="click", params=ClickParams(index=1)), session.page, state)
    assert res.success is False
    assert res.error_code == ActionError.ELEMENT_NOT_FOUND
    assert res.retryable is True


@pytest.mark.asyncio
async def test_click_disabled_element_returns_error(session):
    await session.page.set_content("<button id=b disabled>go</button>")
    state = await extract_state(session.page)
    res = await execute(Action(name="click", params=ClickParams(index=1)), session.page, state)
    assert res.success is False
    assert res.error_code == ActionError.ELEMENT_DISABLED


@pytest.mark.asyncio
async def test_click_blocked_by_overlay_returns_error(session):
    await session.page.set_content(
        "<button id=b style='position:absolute;top:0;left:0;width:100px;height:100px'>go</button>"
        "<div id=overlay style='position:absolute;top:0;left:0;width:300px;height:300px'>cover</div>"
    )
    state = await extract_state(session.page)
    res = await execute(Action(name="click", params=ClickParams(index=1)), session.page, state)
    assert res.success is False
    assert res.error_code == ActionError.CLICK_BLOCKED


@pytest.mark.asyncio
async def test_click_by_coordinate_fallback(session):
    await session.page.set_content(
        "<button id=b onclick=\"this.textContent='clicked'\" style='position:absolute;top:0;left:0;width:100px;height:100px'>go</button>"
    )
    state = await extract_state(session.page)
    res = await execute(
        Action(name="click", params=ClickParams(index=1, coordinate_x=10, coordinate_y=10)),
        session.page,
        state,
    )
    assert res.success is True
    assert await session.page.inner_text("#b") == "clicked"


# --------------------------------------------------------------------------- #
# type
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_execute_type(session):
    await session.page.set_content("<input id=t type=text>")
    state = await extract_state(session.page)
    res = await execute(Action(name="type", params=TypeParams(index=1, text="hello")), session.page, state)
    assert res.success is True
    assert await session.page.input_value("#t") == "hello"


@pytest.mark.asyncio
async def test_execute_type_append_when_clear_false(session):
    await session.page.set_content("<input id=t type=text value='abc'>")
    state = await extract_state(session.page)
    res = await execute(
        Action(name="type", params=TypeParams(index=1, text="X", clear=False)), session.page, state
    )
    assert res.success is True
    assert await session.page.input_value("#t") == "abcX"


@pytest.mark.asyncio
async def test_type_non_editable_element_returns_error(session):
    await session.page.set_content("<button id=b>go</button>")
    state = await extract_state(session.page)
    res = await execute(Action(name="type", params=TypeParams(index=1, text="x")), session.page, state)
    assert res.success is False
    assert res.error_code == ActionError.ELEMENT_NOT_EDITABLE


@pytest.mark.asyncio
async def test_type_disabled_element_returns_error(session):
    await session.page.set_content("<input id=t disabled>")
    state = await extract_state(session.page)
    res = await execute(Action(name="type", params=TypeParams(index=1, text="x")), session.page, state)
    assert res.success is False
    assert res.error_code == ActionError.ELEMENT_DISABLED


# --------------------------------------------------------------------------- #
# scroll / navigate / go_back / switch_tab / press / wait / done
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_execute_scroll_down(session):
    await session.page.set_content("<div style='height:2000px'>tall</div>")
    state = await extract_state(session.page)
    res = await execute(Action(name="scroll", params=ScrollParams(direction="down")), session.page, state)
    assert res.success is True
    assert await session.page.evaluate("() => window.scrollY") > 0


@pytest.mark.asyncio
async def test_execute_navigate(session):
    res = await execute(
        Action(name="navigate", params=NavigateParams(url="data:text/html,<h1>hi</h1>")),
        session.page,
        None,
    )
    assert res.success is True
    assert "data:text/html" in session.page.url


@pytest.mark.asyncio
async def test_navigate_invalid_url_returns_error(session):
    res = await execute(
        Action(name="navigate", params=NavigateParams(url="://bad")), session.page, None
    )
    assert res.success is False
    assert res.error_code == ActionError.NAVIGATION_FAILED


@pytest.mark.asyncio
async def test_execute_go_back(session):
    await session.page.goto("data:text/html,<h1>one</h1>")
    await session.page.goto("data:text/html,<h1>two</h1>")
    res = await execute(Action(name="go_back", params=None), session.page, None)
    assert res.success is True
    assert "one" in await session.page.content()


@pytest.mark.asyncio
async def test_switch_tab_success(session):
    await session.context.new_page()
    res = await execute(Action(name="switch_tab", params=SwitchTabParams(index=0)), session.page, None)
    assert res.success is True
    assert res.metadata["tab_index"] == 0


@pytest.mark.asyncio
async def test_switch_tab_out_of_range_returns_error(session):
    res = await execute(Action(name="switch_tab", params=SwitchTabParams(index=5)), session.page, None)
    assert res.success is False
    assert res.error_code == ActionError.TAB_NOT_FOUND


@pytest.mark.asyncio
async def test_execute_press(session):
    await session.page.set_content("<input id=t>")
    await session.page.focus("#t")
    res = await execute(Action(name="press", params=PressParams(keys="a")), session.page, None)
    assert res.success is True
    assert await session.page.input_value("#t") == "a"


@pytest.mark.asyncio
async def test_execute_wait(session):
    res = await execute(Action(name="wait", params=WaitParams(seconds=0.01)), session.page, None)
    assert res.success is True


@pytest.mark.asyncio
async def test_execute_done(session):
    res = await execute(
        Action(name="done", params=DoneParams(success=True, submission="answer")),
        session.page,
        None,
    )
    assert res.success is True
    assert res.extracted == "answer"


@pytest.mark.asyncio
async def test_execute_done_failure_mirrors_params(session):
    res = await execute(Action(name="done", params=DoneParams(success=False)), session.page, None)
    assert res.success is False


@pytest.mark.asyncio
async def test_unknown_action_returns_error(session):
    fake = Action.model_construct(name="nope", params=None)
    res = await execute(fake, session.page, None)
    assert res.success is False
    assert res.error_code == ActionError.UNKNOWN_ACTION
