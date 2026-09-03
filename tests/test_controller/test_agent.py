"""Task 4.2: the agent step loop — perceive → think → act → observe → repeat."""

import pytest

from minicua.browser.session import BrowserSession
from minicua.controller.agent import Agent, StopReason
from minicua.controller.llm import FakeModel, ImageBlock, ModelOutput, ModelRateLimitError, TextBlock
from minicua.core.errors import BrowserError
from minicua.core.retry import RetryPolicy
from minicua.perception.dom import BrowserState


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_runs_to_done(session):
    model = FakeModel(
        responses=[
            {"name": "navigate", "params": {"url": "data:text/html,<button id=b>ok</button>"}},
            {"name": "click", "params": {"index": 1}},
            {"name": "done", "params": {"success": True, "submission": "clicked ok"}},
        ]
    )
    agent = Agent(session=session, model=model, max_steps=10)
    result = await agent.run(task="click the ok button")

    assert result.done is True
    assert result.success is True
    assert result.submission == "clicked ok"
    assert result.stop_reason == StopReason.DONE
    assert result.steps == 3


@pytest.mark.asyncio
async def test_agent_done_failure_reflects_success_false(session):
    model = FakeModel(responses=[{"name": "done", "params": {"success": False, "submission": "nope"}}])
    agent = Agent(session=session, model=model)
    result = await agent.run(task="impossible task")
    assert result.done is True
    assert result.success is False
    assert result.submission == "nope"


# --------------------------------------------------------------------------- #
# budget / termination
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_stops_at_max_steps(session):
    model = FakeModel(responses=[{"name": "wait", "params": {"seconds": 0.01}}] * 100)
    agent = Agent(session=session, model=model, max_steps=2)
    result = await agent.run(task="never finishes")
    assert result.done is False
    assert result.stop_reason == StopReason.MAX_STEPS
    assert result.steps == 2


@pytest.mark.asyncio
async def test_agent_stops_at_max_failures(session):
    model = FakeModel(responses=[{"name": "click", "params": {"index": 999}}] * 10)
    agent = Agent(session=session, model=model, max_failures=2, max_steps=100)
    result = await agent.run(task="click a nonexistent element")
    assert result.done is False
    assert result.stop_reason == StopReason.MAX_FAILURES


# --------------------------------------------------------------------------- #
# model retry + requery
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_recovers_from_transient_model_error(session):
    model = FakeModel(
        responses=[ModelRateLimitError("rate limited"), {"name": "done", "params": {"success": True}}]
    )
    agent = Agent(
        session=session,
        model=model,
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0.0, jitter=False),
    )
    result = await agent.run(task="x")
    assert result.done is True
    assert result.success is True


@pytest.mark.asyncio
async def test_agent_requeries_on_empty_tool_calls(session):
    # First response has no tool calls -> format error -> requery; second is valid.
    model = FakeModel(
        responses=[ModelOutput(thought="let me reconsider..."), {"name": "done", "params": {"success": True}}]
    )
    agent = Agent(session=session, model=model)
    result = await agent.run(task="x")
    assert result.done is True
    assert result.success is True
    assert len(model.calls) == 2  # one requery happened
    assert result.steps == 1  # requery is within a single step


@pytest.mark.asyncio
async def test_agent_terminates_when_requery_budget_exhausted(session):
    model = FakeModel(responses=[ModelOutput()] * 10)  # always empty
    agent = Agent(session=session, model=model, max_requeries=2)
    result = await agent.run(task="x")
    assert result.done is False
    assert result.stop_reason == StopReason.INVALID_RESPONSE


# --------------------------------------------------------------------------- #
# action failure feedback
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_feeds_action_error_back_to_model(session):
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 999}},  # stale -> structured error
            {"name": "done", "params": {"success": True}},
        ]
    )
    agent = Agent(session=session, model=model, max_steps=10)
    result = await agent.run(task="click something")
    assert result.done is True  # the loop continues past a failed action

    # The second model call's messages must carry the structured error as feedback.
    second_call_messages = model.calls[1][0]
    observations = [m.content for m in second_call_messages if m.role == "user"]
    assert any("not in browser state" in text for text in observations)


# --------------------------------------------------------------------------- #
# perception -> model vision wiring
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_passes_screenshot_as_image_block_in_vision_mode(session, monkeypatch):
    async def fake_extract_state(page, *, use_vision="dom_only", model_supports_vision=False):
        return BrowserState(url="http://x", title="t", dom_text="[1] button", screenshot="aGVsbG8=")

    monkeypatch.setattr("minicua.controller.agent.extract_state", fake_extract_state)
    model = FakeModel(responses=[{"name": "done", "params": {"success": True}}])
    agent = Agent(session=session, model=model, use_vision="vision")
    await agent.run(task="x")

    state_messages = [m for m in model.calls[0][0] if m.role == "user"]
    content = state_messages[0].content
    assert isinstance(content, list)
    assert any(isinstance(b, ImageBlock) and b.image_base64 == "aGVsbG8=" for b in content)
    assert any(isinstance(b, TextBlock) and "button" in b.text for b in content)


@pytest.mark.asyncio
async def test_agent_dom_only_uses_text_only_content(session, monkeypatch):
    async def fake_extract_state(page, *, use_vision="dom_only", model_supports_vision=False):
        return BrowserState(url="http://x", title="t", dom_text="[1] button", screenshot=None)

    monkeypatch.setattr("minicua.controller.agent.extract_state", fake_extract_state)
    model = FakeModel(responses=[{"name": "done", "params": {"success": True}}])
    agent = Agent(session=session, model=model, use_vision="dom_only")
    await agent.run(task="x")

    state_messages = [m for m in model.calls[0][0] if m.role == "user"]
    assert isinstance(state_messages[0].content, str)


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_step_returns_step_result(session):
    model = FakeModel(responses=[{"name": "done", "params": {"success": True}}])
    agent = Agent(session=session, model=model)
    step = await agent.step()
    assert step.is_done is True
    assert step.success is True


@pytest.mark.asyncio
async def test_agent_requires_started_session():
    not_started = BrowserSession(headless=True)
    agent = Agent(session=not_started, model=FakeModel(responses=[{"name": "done", "params": {}}]))
    with pytest.raises(BrowserError):
        await agent.run(task="x")


@pytest.mark.asyncio
async def test_agent_surfaces_unexpected_error(session):
    # A non-ModelError exception is not retryable and not requeryable; the run
    # surfaces it as a structured ERROR result instead of crashing the caller.
    model = FakeModel(responses=[RuntimeError("boom")])
    agent = Agent(session=session, model=model)
    result = await agent.run(task="x")
    assert result.done is False
    assert result.stop_reason == StopReason.ERROR
    assert "RuntimeError" in result.error


# --------------------------------------------------------------------------- #
# completion verifier (done-before-termination)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_verifier_rejects_premature_done(session):
    # A claimed success is a claim, not ground truth: the first done is rejected
    # by the verifier, the model is fed the rejection and keeps going, and the
    # second done (now genuinely satisfied) is accepted.
    calls = []

    def verifier():
        calls.append(1)
        return (len(calls) >= 2, "goal not met yet")

    model = FakeModel(
        responses=[
            {"name": "done", "params": {"success": True}},
            {"name": "done", "params": {"success": True}},
        ]
    )
    agent = Agent(session=session, model=model, verifier=verifier, max_steps=10)
    result = await agent.run(task="finish")

    assert result.done is True
    assert result.success is True
    assert result.steps == 2  # first done rejected, second accepted
    assert len(calls) == 2

    # The rejection must have been fed back to the model between the two dones.
    second_call_messages = model.calls[1][0]
    observations = [m.content for m in second_call_messages if m.role == "user"]
    assert any("rejected" in text for text in observations)


@pytest.mark.asyncio
async def test_agent_verifier_accepts_done(session):
    def verifier():
        return True, ""

    model = FakeModel(responses=[{"name": "done", "params": {"success": True}}])
    agent = Agent(session=session, model=model, verifier=verifier)
    result = await agent.run(task="finish")
    assert result.done is True
    assert result.success is True
    assert result.steps == 1


@pytest.mark.asyncio
async def test_agent_async_verifier(session):
    async def verifier():
        return True, ""

    model = FakeModel(responses=[{"name": "done", "params": {"success": True}}])
    agent = Agent(session=session, model=model, verifier=verifier)
    result = await agent.run(task="finish")
    assert result.done is True
    assert result.success is True


@pytest.mark.asyncio
async def test_agent_verifier_exception_degrades_to_accept(session):
    # A broken verifier must never hang or crash the loop: it degrades to
    # accepting the agent's own done claim.
    def verifier():
        raise RuntimeError("verifier broken")

    model = FakeModel(responses=[{"name": "done", "params": {"success": True}}])
    agent = Agent(session=session, model=model, verifier=verifier)
    result = await agent.run(task="finish")
    assert result.done is True
    assert result.success is True


@pytest.mark.asyncio
async def test_agent_verifier_rejected_then_budget_exhausted(session):
    # A verifier that always rejects keeps the agent in the loop until the step
    # budget runs out, at which point the run ends as MAX_STEPS (not a false DONE).
    def verifier():
        return False, "still not done"

    model = FakeModel(responses=[{"name": "done", "params": {"success": True}}] * 10)
    agent = Agent(session=session, model=model, verifier=verifier, max_steps=3)
    result = await agent.run(task="finish")
    assert result.done is False
    assert result.stop_reason == StopReason.MAX_STEPS
    assert result.steps == 3
