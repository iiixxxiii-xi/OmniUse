"""ChatRunner: turn a natural-language instruction into a browser run (no evaluator).

The runner is the thin wrapper behind ``minicua chat`` — it reuses the existing
:class:`~minicua.controller.agent.Agent`, :class:`~minicua.browser.session.BrowserSession`,
and the inline-HTML fixture serving from the eval runner, and returns a
human-readable :class:`ChatRun` (what happened + final URL + summary) instead of a
scored :class:`~minicua.eval.runner.EvalResult`.
"""

import pytest

from minicua.chat import ChatRunner
from minicua.controller.agent import StopReason
from minicua.controller.llm import FakeModel

_FIXTURE_URL = "http://minicua.local/"

_BUTTON_HTML = (
    "<button id=btn onclick=\"document.getElementById('out').textContent='clicked'\">go</button>"
    "<div id=out></div>"
)

_INPUT_HTML = "<input id=q type=text><div id=out></div>"


@pytest.mark.asyncio
async def test_chat_runner_click_then_done(session):
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 1}},
            {"name": "done", "params": {"success": True, "submission": "clicked the button"}},
        ]
    )
    result = await ChatRunner(model).run("click the button", html=_BUTTON_HTML, session=session)

    assert result.final_url == _FIXTURE_URL
    assert result.stop_reason == StopReason.DONE.value
    assert result.steps == 2
    assert result.error is None
    assert result.submission == "clicked the button"

    assert len(result.actions) == 2
    assert result.actions[0].name == "click"
    assert result.actions[0].success is True
    assert result.actions[1].name == "done"

    assert "clicked element #1" in result.summary
    assert "finished: clicked the button" in result.summary


@pytest.mark.asyncio
async def test_chat_runner_type_describes_text(session):
    model = FakeModel(
        responses=[
            {"name": "type", "params": {"index": 1, "text": "hello world"}},
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await ChatRunner(model).run("type hello into the box", html=_INPUT_HTML, session=session)

    assert result.actions[0].name == "type"
    assert result.actions[0].success is True
    assert "typed 'hello world' into element #1" in result.summary


@pytest.mark.asyncio
async def test_chat_runner_navigate_describes_url(session):
    url = "data:text/html,<div>hi</div>"
    model = FakeModel(
        responses=[
            {"name": "navigate", "params": {"url": url}},
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await ChatRunner(model).run("go somewhere", session=session)

    assert result.final_url == url
    assert f"navigated to {url}" in result.summary


@pytest.mark.asyncio
async def test_chat_runner_initial_url(session):
    url = "data:text/html,<div>hello world</div>"
    model = FakeModel(responses=[{"name": "done", "params": {"success": True}}])
    result = await ChatRunner(model).run("verify the page", initial_url=url, session=session)

    assert result.final_url == url
    assert result.stop_reason == StopReason.DONE.value


@pytest.mark.asyncio
async def test_chat_runner_model_error_is_structured(session):
    # A model with no scripted responses fails the run but returns a ChatRun, never raises.
    result = await ChatRunner(FakeModel(responses=[])).run("do something", session=session)

    assert result.stop_reason == StopReason.MODEL_ERROR.value
    assert result.error is not None
    assert result.actions == []
    assert result.summary == ""
