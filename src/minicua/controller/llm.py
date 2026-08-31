"""ChatModel abstraction: the boundary between the agent loop and a language model.

The controller depends only on the :class:`ChatModel` protocol — anything that
can turn a message list + tool list into a :class:`ModelOutput` is a model. This
decouples the loop from any single vendor and lets tests drive the loop with a
deterministic :class:`FakeModel` (no network, no API key, zero cost).

Data model:

* :class:`Message` — a single chat message (system / user / assistant / tool).
* :class:`ToolCall` — one model-emitted tool invocation (``name`` + ``arguments``).
* :class:`ModelOutput` — a single model response: optional ``thought`` plus a
  (possibly empty) list of tool calls, plus optional token/cost ``usage``.
* :class:`ModelUsage` — input/output token counts and cost, for budget tracking.

Errors:

Model failures are typed, so the controller can *classify* them and decide
whether to retry (transient: rate limit / timeout / server) or stop (permanent:
auth / not-configured / budget exceeded). Format problems are a distinct error
(:class:`ModelInvalidResponseError`) that the loop handles by *requerying* the
model, not by blind retry.

Real adapters (:class:`AnthropicModel`, :class:`OpenAIModel`) are skeletons:
they carry their configuration (model id, api key, vision support) but their
``generate`` raises :class:`ModelNotConfiguredError` until the integration stage
wires the vendor SDKs.
"""

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from minicua.core.errors import CUAError

# --------------------------------------------------------------------------- #
# Message / tool-call / output data models
# --------------------------------------------------------------------------- #

MessageRole = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """A single chat message. ``content`` is plain text (vision blocks come later)."""

    role: MessageRole
    content: str = ""


class ToolCall(BaseModel):
    """One model-emitted tool invocation, in a wire-format-agnostic shape."""

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelUsage(BaseModel):
    """Token counts and cost for a single model call (for budget tracking)."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelOutput(BaseModel):
    """A single model response: ``thought`` plus a list of ``tool_calls``.

    The controller turns each :class:`ToolCall` into a validated
    :class:`~minicua.action.models.Action`. An empty ``tool_calls`` is treated as
    a format error by the loop (see :mod:`minicua.controller.agent`).
    """

    thought: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: ModelUsage | None = None


# --------------------------------------------------------------------------- #
# ChatModel protocol
# --------------------------------------------------------------------------- #


class ChatModel(Protocol):
    """The interface the controller depends on for any model.

    ``tools`` uses the OpenAI function-calling shape produced by
    :meth:`~minicua.action.registry.ActionRegistry.to_tools`; adapters convert to
    their vendor format internally.
    """

    supports_vision: bool

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> ModelOutput: ...


# --------------------------------------------------------------------------- #
# Model error taxonomy
# --------------------------------------------------------------------------- #


class ModelError(CUAError):
    """Base class for model failures. ``category`` is a machine-readable label."""

    category: str = "model_error"


class ModelRateLimitError(ModelError):
    """The provider rate-limited the request. Transient — back off and retry."""

    category = "rate_limit"
    retryable = True


class ModelTimeoutError(ModelError):
    """The request timed out. Transient — retry with backoff."""

    category = "timeout"
    retryable = True


class ModelServerError(ModelError):
    """The provider returned a 5xx / transient server error. Retryable."""

    category = "server"
    retryable = True


class ModelInvalidResponseError(ModelError):
    """The model returned something unparseable / malformed.

    Not blind-retryable: the loop feeds the error back and *requeries* instead.
    """

    category = "invalid_response"


class ModelAuthError(ModelError):
    """Authentication / authorization failed. Permanent — do not retry."""

    category = "auth"


class ModelNotConfiguredError(ModelError):
    """A real adapter whose SDK wiring is deferred to the integration stage."""

    category = "not_configured"


class ModelBudgetExceededError(ModelError):
    """The request exceeded a provider-side budget (e.g. context length)."""

    category = "budget_exceeded"


class ScriptExhaustedError(ModelError):
    """A :class:`FakeModel` ran out of scripted responses. Permanent (test-only)."""

    category = "script_exhausted"


# --------------------------------------------------------------------------- #
# FakeModel — deterministic, scripted, no network
# --------------------------------------------------------------------------- #


class FakeModel:
    """A scripted model for tests: returns canned responses in order.

    ``responses`` is a list of items, each consumed by one ``generate`` call:

    * a ``dict`` like ``{"name": "click", "params": {"index": 1}}`` — a single
      tool call (``params`` and ``arguments`` are both accepted);
    * a ``dict`` with ``"thought"`` / ``"tool_calls"`` / ``"usage"`` — a full
      :class:`ModelOutput`;
    * a ``list`` of ``{"name", "params"}`` dicts — multiple tool calls in one step;
    * a :class:`ModelOutput` — used as-is;
    * an ``Exception`` — raised by ``generate`` (to script transient failures).

    Every ``generate`` call is recorded in ``calls`` for assertions.
    """

    def __init__(self, responses: list[Any] | None = None, *, supports_vision: bool = False) -> None:
        self.supports_vision = supports_vision
        self._responses: list[Any] = [self._normalize(r) for r in (responses or [])]
        self.calls: list[tuple[tuple[Message, ...], tuple[dict[str, Any], ...]]] = []

    @staticmethod
    def _normalize(item: Any) -> Any:
        if isinstance(item, (BaseException, ModelOutput)):
            return item
        if isinstance(item, dict) and "tool_calls" in item:
            calls = [
                ToolCall(name=c["name"], arguments=c.get("arguments", c.get("params", {})))
                for c in item["tool_calls"]
            ]
            usage = item.get("usage")
            return ModelOutput(
                thought=item.get("thought"),
                tool_calls=calls,
                usage=ModelUsage(**usage) if usage else None,
            )
        if isinstance(item, dict) and "name" in item:
            params = item.get("params") or item.get("arguments") or {}
            return ModelOutput(
                tool_calls=[ToolCall(name=item["name"], arguments=params)],
                usage=ModelUsage(**item["usage"]) if item.get("usage") else None,
            )
        if isinstance(item, (list, tuple)):
            calls = [
                ToolCall(name=d["name"], arguments=d.get("params", d.get("arguments", {})))
                for d in item
            ]
            return ModelOutput(tool_calls=calls)
        raise TypeError(f"unsupported FakeModel response: {item!r}")

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> ModelOutput:
        self.calls.append((tuple(messages), tuple(tools)))
        if not self._responses:
            raise ScriptExhaustedError("FakeModel has no scripted responses left")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


# --------------------------------------------------------------------------- #
# Real adapter skeletons (SDK wiring deferred to the integration stage)
# --------------------------------------------------------------------------- #


class AnthropicModel:
    """Anthropic Claude adapter skeleton.

    Carries configuration (model id, api key, vision support, max tokens). The
    real Messages API call is deferred to the integration stage; ``generate``
    raises :class:`ModelNotConfiguredError` until then.
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-5",
        api_key: str | None = None,
        supports_vision: bool = True,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.supports_vision = supports_vision
        self.max_tokens = max_tokens

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> ModelOutput:
        raise ModelNotConfiguredError(
            "AnthropicModel.generate() is a skeleton; the real Anthropic API call "
            "is deferred to the integration stage"
        )


class OpenAIModel:
    """OpenAI adapter skeleton (see :class:`AnthropicModel`)."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: str | None = None,
        supports_vision: bool = True,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.supports_vision = supports_vision
        self.max_tokens = max_tokens

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> ModelOutput:
        raise ModelNotConfiguredError(
            "OpenAIModel.generate() is a skeleton; the real OpenAI API call "
            "is deferred to the integration stage"
        )
