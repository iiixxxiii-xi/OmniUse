"""Task 4.1: ChatModel abstraction — protocol, fake model, adapter skeletons."""

import pytest

from minicua.controller.llm import (
    AnthropicModel,
    FakeModel,
    ImageBlock,
    Message,
    ModelAuthError,
    ModelError,
    ModelInvalidResponseError,
    ModelNotConfiguredError,
    ModelOutput,
    ModelRateLimitError,
    ModelUsage,
    OpenAIModel,
    ScriptExhaustedError,
    TextBlock,
    ToolCall,
)


# --------------------------------------------------------------------------- #
# ModelOutput / ModelUsage data models
# --------------------------------------------------------------------------- #


def test_model_output_holds_thought_and_tool_calls():
    out = ModelOutput(
        thought="I should click the button",
        tool_calls=[ToolCall(name="click", arguments={"index": 1})],
    )
    assert out.thought == "I should click the button"
    assert out.tool_calls[0].name == "click"
    assert out.tool_calls[0].arguments == {"index": 1}


def test_model_output_defaults_empty_tool_calls():
    out = ModelOutput()
    assert out.thought is None
    assert out.tool_calls == []


def test_model_usage_total_tokens():
    usage = ModelUsage(input_tokens=10, output_tokens=5)
    assert usage.total_tokens == 15
    assert usage.cost_usd == 0.0


# --------------------------------------------------------------------------- #
# FakeModel — scripted returns, no real API call
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fake_model_returns_scripted_tool_call():
    m = FakeModel(responses=[{"name": "done", "params": {"success": True}}])
    out = await m.generate(messages=[], tools=[])
    assert [c.name for c in out.tool_calls] == ["done"]
    assert out.tool_calls[0].arguments == {"success": True}


@pytest.mark.asyncio
async def test_fake_model_sequential_responses():
    m = FakeModel(
        responses=[
            {"name": "navigate", "params": {"url": "data:text/html,<button>x</button>"}},
            {"name": "done", "params": {}},
        ]
    )
    first = await m.generate([], [])
    second = await m.generate([], [])
    assert first.tool_calls[0].name == "navigate"
    assert second.tool_calls[0].name == "done"


@pytest.mark.asyncio
async def test_fake_model_multiple_tool_calls_per_step():
    m = FakeModel(
        responses=[[{"name": "click", "params": {"index": 1}}, {"name": "click", "params": {"index": 2}}]]
    )
    out = await m.generate([], [])
    assert [c.name for c in out.tool_calls] == ["click", "click"]


@pytest.mark.asyncio
async def test_fake_model_accepts_model_output_directly():
    scripted = ModelOutput(thought="done now", tool_calls=[ToolCall(name="done", arguments={})])
    m = FakeModel(responses=[scripted])
    out = await m.generate([], [])
    assert out.thought == "done now"
    assert out.tool_calls[0].name == "done"


@pytest.mark.asyncio
async def test_fake_model_scripted_usage():
    m = FakeModel(
        responses=[{"name": "done", "params": {}, "usage": {"input_tokens": 10, "output_tokens": 5}}]
    )
    out = await m.generate([], [])
    assert out.usage is not None
    assert out.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_fake_model_exhausted_raises():
    m = FakeModel(responses=[{"name": "done", "params": {}}])
    await m.generate([], [])
    with pytest.raises(ScriptExhaustedError):
        await m.generate([], [])


@pytest.mark.asyncio
async def test_fake_model_scripted_exception():
    m = FakeModel(responses=[ModelRateLimitError("rate limited"), {"name": "done", "params": {}}])
    with pytest.raises(ModelRateLimitError):
        await m.generate([], [])
    out = await m.generate([], [])  # next scripted response is consumed
    assert out.tool_calls[0].name == "done"


@pytest.mark.asyncio
async def test_fake_model_records_calls():
    m = FakeModel(responses=[{"name": "done", "params": {}}])
    msgs = [Message(role="system", content="hi")]
    tools = [{"type": "function", "function": {"name": "done"}}]
    await m.generate(msgs, tools)
    assert len(m.calls) == 1
    assert m.calls[0][0][0].content == "hi"
    assert m.calls[0][1][0]["function"]["name"] == "done"


def test_fake_model_vision_flag_defaults_false():
    assert FakeModel().supports_vision is False
    assert FakeModel(supports_vision=True).supports_vision is True


# --------------------------------------------------------------------------- #
# Model error taxonomy
# --------------------------------------------------------------------------- #


def test_model_error_taxonomy_retryable_flags():
    assert ModelRateLimitError("rate limited").retryable is True
    assert ModelAuthError("bad key").retryable is False
    assert ModelInvalidResponseError("bad schema").retryable is False


def test_model_errors_are_cua_errors():
    for cls in (ModelRateLimitError, ModelAuthError, ModelInvalidResponseError, ModelNotConfiguredError):
        err = cls("boom")
        assert isinstance(err, ModelError)
        assert isinstance(err, Exception)
        assert err.category  # every model error carries a category


# --------------------------------------------------------------------------- #
# Message content blocks (vision support)
# --------------------------------------------------------------------------- #


def test_message_accepts_vision_content_blocks():
    msg = Message(
        role="user",
        content=[TextBlock(text="DOM snapshot"), ImageBlock(image_base64="aGVsbG8=")],
    )
    assert msg.content[0].type == "text"
    assert msg.content[0].text == "DOM snapshot"
    assert msg.content[1].type == "image"
    assert msg.content[1].image_base64 == "aGVsbG8="


def test_message_defaults_to_plain_text():
    assert Message(role="user", content="hi").content == "hi"
    assert Message(role="system").content == ""


# --------------------------------------------------------------------------- #
# Real model adapter skeletons (real API calls deferred to integration stage)
# --------------------------------------------------------------------------- #


def test_anthropic_model_skeleton_config():
    m = AnthropicModel(model="claude-sonnet-4-5", supports_vision=True)
    assert m.supports_vision is True
    assert m.model == "claude-sonnet-4-5"


def test_openai_model_skeleton_config():
    m = OpenAIModel(model="gpt-4o", supports_vision=False)
    assert m.supports_vision is False
    assert m.model == "gpt-4o"


@pytest.mark.asyncio
async def test_anthropic_model_generate_not_configured():
    m = AnthropicModel()
    with pytest.raises(ModelNotConfiguredError):
        await m.generate([], [])


@pytest.mark.asyncio
async def test_openai_model_generate_not_configured():
    m = OpenAIModel()
    with pytest.raises(ModelNotConfiguredError):
        await m.generate([], [])
