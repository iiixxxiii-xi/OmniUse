"""OpenAI-compatible adapter: message translation, tool-call parsing, vision
multimodal format, and error classification.

The adapter is driven through a *fake client* (a stand-in for
``openai.AsyncOpenAI``) that captures the exact messages/tools sent to the SDK,
so the conversion logic is tested against the real wire shape without a network.
"""

import asyncio

import pytest

from minicua.controller.llm import (
    ImageBlock,
    Message,
    ModelAuthError,
    ModelError,
    ModelInvalidResponseError,
    ModelNotConfiguredError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
    OpenAIModel,
    TextBlock,
    ToolCall,
    classify_openai_error,
    compute_cost_usd,
    parse_openai_choice,
    to_openai_messages,
)


# --------------------------------------------------------------------------- #
# Fake client + canned responses
# --------------------------------------------------------------------------- #


class _FakeCompletions:
    def __init__(self, handler):
        self._handler = handler
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._handler(kwargs)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class RecordingClient:
    """Stand-in for ``openai.AsyncOpenAI``: records each ``create`` call."""

    def __init__(self, handler=None):
        self.completions = _FakeCompletions(handler or (lambda kw: _response()))
        self.chat = _FakeChat(self.completions)

    @property
    def calls(self):
        return self.completions.calls


class _FakeHTTPError(Exception):
    """An exception carrying a ``status_code``, like ``openai.APIStatusError``."""

    def __init__(self, status_code, message=""):
        self.status_code = status_code
        super().__init__(message)


def _response(*, content=None, tool_calls=None, usage=None):
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "tool_calls": tool_calls or [],
                }
            }
        ],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _raises(exc):
    def handler(kwargs):
        raise exc

    return handler


# --------------------------------------------------------------------------- #
# to_openai_messages
# --------------------------------------------------------------------------- #


def test_to_openai_messages_text_passthrough():
    msgs = [
        Message(role="system", content="be helpful"),
        Message(role="user", content="click the button"),
    ]
    assert to_openai_messages(msgs) == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "click the button"},
    ]


def test_to_openai_messages_vision_multimodal_format():
    msgs = [
        Message(
            role="user",
            content=[
                TextBlock(text="DOM snapshot"),
                ImageBlock(image_base64="aGVsbG8="),
            ],
        )
    ]
    out = to_openai_messages(msgs)
    assert out[0]["content"] == [
        {"type": "text", "text": "DOM snapshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
    ]


def test_to_openai_messages_text_only_block_list_stays_structured():
    msgs = [Message(role="user", content=[TextBlock(text="just text")])]
    assert to_openai_messages(msgs) == [{"role": "user", "content": [{"type": "text", "text": "just text"}]}]


# --------------------------------------------------------------------------- #
# parse_openai_choice
# --------------------------------------------------------------------------- #


def test_parse_openai_choice_tool_calls_and_usage():
    choice = {
        "message": {
            "content": "thinking...",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "click", "arguments": '{"index": 3}'}},
            ],
        }
    }
    out = parse_openai_choice(choice, usage={"prompt_tokens": 12, "completion_tokens": 4})
    assert out.thought == "thinking..."
    assert out.tool_calls == [ToolCall(name="click", arguments={"index": 3})]
    assert out.usage.total_tokens == 16


def test_parse_openai_choice_malformed_arguments_becomes_empty_dict():
    choice = {"message": {"content": None, "tool_calls": [{"function": {"name": "done", "arguments": "not-json"}}]}}
    out = parse_openai_choice(choice)
    assert out.tool_calls[0].name == "done"
    assert out.tool_calls[0].arguments == {}


def test_parse_openai_choice_non_dict_arguments_becomes_empty_dict():
    choice = {"message": {"tool_calls": [{"function": {"name": "done", "arguments": '["a", "b"]'}}]}}
    out = parse_openai_choice(choice)
    assert out.tool_calls[0].arguments == {}


def test_parse_openai_choice_no_tool_calls():
    out = parse_openai_choice({"message": {"content": "no tools here"}})
    assert out.thought == "no tools here"
    assert out.tool_calls == []


def test_parse_openai_choice_missing_usage_is_none():
    out = parse_openai_choice({"message": {"content": None}})
    assert out.usage is None
    assert out.thought is None


def test_parse_openai_choice_vision_list_content_extracts_text_thought():
    choice = {"message": {"content": [{"type": "text", "text": "I see a button"}]}}
    out = parse_openai_choice(choice)
    assert out.thought == "I see a button"


# --------------------------------------------------------------------------- #
# OpenAIModel.generate with a fake client
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_generate_calls_client_with_translated_messages_and_tools():
    client = RecordingClient(
        handler=lambda kw: _response(
            content="I'll click",
            tool_calls=[{"function": {"name": "click", "arguments": '{"index": 1}'}}],
        )
    )
    m = OpenAIModel(model="deepseek-chat", api_key="sk-test", client=client, supports_vision=False)
    out = await m.generate(
        [Message(role="system", content="hi"), Message(role="user", content="go")],
        [{"type": "function", "function": {"name": "click", "parameters": {}}}],
    )
    assert out.thought == "I'll click"
    assert out.tool_calls[0].name == "click"
    assert out.tool_calls[0].arguments == {"index": 1}

    kw = client.calls[0]
    assert kw["model"] == "deepseek-chat"
    assert kw["messages"] == [
        {"role": "system", "content": "hi"},
        {"role": "user", "content": "go"},
    ]
    assert kw["tools"][0]["function"]["name"] == "click"


@pytest.mark.asyncio
async def test_generate_sends_vision_blocks_to_sdk():
    client = RecordingClient(handler=lambda kw: _response(content=None, tool_calls=[]))
    m = OpenAIModel(model="qwen3-vl-flash", api_key="sk-test", client=client, supports_vision=True)
    await m.generate(
        [Message(role="user", content=[TextBlock(text="DOM"), ImageBlock(image_base64="aGVsbG8=")])],
        [],
    )
    sent = client.calls[0]["messages"][0]["content"]
    assert sent[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}


@pytest.mark.asyncio
async def test_generate_omits_empty_tools():
    client = RecordingClient(handler=lambda kw: _response(content=None, tool_calls=[]))
    m = OpenAIModel(model="x", api_key="sk-test", client=client)
    await m.generate([Message(role="user", content="hi")], [])
    assert client.calls[0]["tools"] is None


# --------------------------------------------------------------------------- #
# Cost computation from token usage (pricing table)
# --------------------------------------------------------------------------- #


def test_compute_cost_usd_uses_pricing_table():
    assert compute_cost_usd("deepseek-chat", 1_000_000, 0) == pytest.approx(0.28)
    assert compute_cost_usd("deepseek-chat", 0, 1_000_000) == pytest.approx(0.42)
    assert compute_cost_usd("qwen3-vl-flash", 0, 1_000_000) == pytest.approx(0.40)


def test_compute_cost_usd_prefix_matches_variant_ids():
    # A versioned/sized id still resolves to its base model's price.
    assert compute_cost_usd("qwen3-vl-flash-32b", 1_000_000, 0) == pytest.approx(0.05)


def test_compute_cost_usd_unknown_model_is_zero():
    assert compute_cost_usd("no-such-model", 1_000_000, 1_000_000) == 0.0


@pytest.mark.asyncio
async def test_generate_computes_cost_for_deepseek():
    client = RecordingClient(
        handler=lambda kw: _response(usage={"prompt_tokens": 1000, "completion_tokens": 500})
    )
    m = OpenAIModel(model="deepseek-chat", api_key="sk-test", client=client, supports_vision=False)
    out = await m.generate([Message(role="user", content="hi")], [])
    assert out.usage is not None
    assert out.usage.input_tokens == 1000
    assert out.usage.output_tokens == 500
    # 1000 * $0.28/M + 500 * $0.42/M
    assert out.usage.cost_usd == pytest.approx((1000 * 0.28 + 500 * 0.42) / 1_000_000)


@pytest.mark.asyncio
async def test_generate_computes_cost_for_qwen3_vl():
    client = RecordingClient(
        handler=lambda kw: _response(usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
    )
    m = OpenAIModel(model="qwen3-vl-flash", api_key="sk-test", client=client, supports_vision=True)
    out = await m.generate([Message(role="user", content="hi")], [])
    # $0.05/M input + $0.40/M output
    assert out.usage.cost_usd == pytest.approx(0.05 + 0.40)


@pytest.mark.asyncio
async def test_generate_uses_explicit_api_cost_when_present():
    client = RecordingClient(
        handler=lambda kw: _response(usage={"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.1234})
    )
    m = OpenAIModel(model="deepseek-chat", api_key="sk-test", client=client, supports_vision=False)
    out = await m.generate([Message(role="user", content="hi")], [])
    assert out.usage.cost_usd == pytest.approx(0.1234)


@pytest.mark.asyncio
async def test_generate_unknown_model_cost_is_zero():
    client = RecordingClient(
        handler=lambda kw: _response(usage={"prompt_tokens": 100, "completion_tokens": 10})
    )
    m = OpenAIModel(model="some-unknown-model", api_key="sk-test", client=client)
    out = await m.generate([Message(role="user", content="hi")], [])
    assert out.usage.cost_usd == 0.0


# --------------------------------------------------------------------------- #
# Error classification (via generate)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_generate_classifies_429_as_rate_limit():
    m = OpenAIModel(model="x", api_key="sk", client=RecordingClient(_raises(_FakeHTTPError(429))))
    with pytest.raises(ModelRateLimitError):
        await m.generate([], [])


@pytest.mark.asyncio
async def test_generate_classifies_500_as_server_error():
    m = OpenAIModel(model="x", api_key="sk", client=RecordingClient(_raises(_FakeHTTPError(500))))
    with pytest.raises(ModelServerError):
        await m.generate([], [])


@pytest.mark.asyncio
async def test_generate_classifies_401_and_403_as_auth():
    for code in (401, 403):
        m = OpenAIModel(model="x", api_key="sk", client=RecordingClient(_raises(_FakeHTTPError(code))))
        with pytest.raises(ModelAuthError):
            await m.generate([], [])


@pytest.mark.asyncio
async def test_generate_classifies_other_4xx_as_model_error():
    m = OpenAIModel(model="x", api_key="sk", client=RecordingClient(_raises(_FakeHTTPError(400))))
    with pytest.raises(ModelError):
        await m.generate([], [])


@pytest.mark.asyncio
async def test_generate_classifies_timeout():
    m = OpenAIModel(model="x", api_key="sk", client=RecordingClient(_raises(asyncio.TimeoutError("timed out"))))
    with pytest.raises(ModelTimeoutError):
        await m.generate([], [])


@pytest.mark.asyncio
async def test_generate_empty_choices_is_invalid_response():
    m = OpenAIModel(model="x", api_key="sk", client=RecordingClient(lambda kw: {"choices": []}))
    with pytest.raises(ModelInvalidResponseError):
        await m.generate([], [])


@pytest.mark.asyncio
async def test_generate_without_key_raises_not_configured():
    m = OpenAIModel()
    with pytest.raises(ModelNotConfiguredError):
        await m.generate([], [])


# --------------------------------------------------------------------------- #
# classify_openai_error (direct edge cases)
# --------------------------------------------------------------------------- #


def test_classify_timeout_by_exception_typename():
    class APITimeoutError(Exception):
        pass

    assert isinstance(classify_openai_error(APITimeoutError("timed out")), ModelTimeoutError)


def test_classify_connection_error_as_server():
    assert isinstance(classify_openai_error(_FakeHTTPError(502)), ModelServerError)
