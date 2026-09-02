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

import asyncio
import json
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field

from minicua.core.errors import CUAError

# --------------------------------------------------------------------------- #
# Message / tool-call / output data models
# --------------------------------------------------------------------------- #

MessageRole = Literal["system", "user", "assistant", "tool"]


class TextBlock(BaseModel):
    """A text content block (the non-vision branch of a message's content)."""

    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """An image content block: raw base64-encoded bytes (no ``data:`` prefix).

    The adapter prepends the ``data:image/png;base64,`` prefix when translating
    to a provider's multimodal wire format.
    """

    type: Literal["image"] = "image"
    image_base64: str


ContentBlock = Annotated[TextBlock | ImageBlock, Field(discriminator="type")]


class Message(BaseModel):
    """A single chat message.

    ``content`` is plain text, or a list of :class:`ContentBlock` (text and/or
    image) for multimodal (vision) messages. Plain-text usage stays fully
    backward compatible.
    """

    role: MessageRole
    content: str | list[ContentBlock] = ""


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


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict or an object (both shapes occur: SDK objects and
    plain-dict test doubles)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _explicit_cost(usage: Any) -> float:
    """Read a provider-supplied cost from ``usage``, or ``0.0`` when absent.

    The OpenAI-compatible ``usage`` object normally carries only token counts;
    some gateways additionally return a monetary ``cost`` (a few spell it
    ``cost_usd`` / ``total_cost``). When present, that figure is authoritative
    and the adapter should not recompute it from the pricing table.
    """
    for key in ("cost", "cost_usd", "total_cost"):
        value = _get(usage, key, None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Translate CUA :class:`Message` objects into the OpenAI chat-completion shape.

    Plain-text content is passed through as a string; multimodal content (a list
    of :class:`ContentBlock`) becomes the OpenAI ``image_url`` part format with a
    ``data:image/png;base64,`` prefix on each image.
    """
    converted: list[dict[str, Any]] = []
    for m in messages:
        content: str | list[dict[str, Any]]
        if isinstance(m.content, str):
            content = m.content
        else:
            parts: list[dict[str, Any]] = []
            for block in m.content:
                if block.type == "text":
                    parts.append({"type": "text", "text": block.text})
                elif block.type == "image":
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{block.image_base64}"},
                        }
                    )
            content = parts
        converted.append({"role": m.role, "content": content})
    return converted


def parse_openai_choice(choice: Any, usage: Any | None = None) -> ModelOutput:
    """Convert an OpenAI chat-completion ``choice`` into a :class:`ModelOutput`.

    ``choice`` may be a dict or an SDK object; ``usage`` is ``response.usage``
    (dict or object). Tool calls are unwrapped from the OpenAI ``function`` shape
    and their JSON-string arguments parsed back into a dict.
    """
    message = _get(choice, "message", None) or {}

    content = _get(message, "content", None)
    if isinstance(content, str):
        thought = content or None
    elif isinstance(content, list):
        thought = "".join(
            _get(b, "text", "") or "" for b in content if _get(b, "type", None) == "text"
        ) or None
    else:
        thought = None

    tool_calls: list[ToolCall] = []
    for tc in _get(message, "tool_calls", None) or []:
        fn = _get(tc, "function", None) or {}
        name = _get(fn, "name", "") or ""
        raw_arguments = _get(fn, "arguments", "{}") or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        tool_calls.append(ToolCall(name=name, arguments=arguments))

    usage_obj: ModelUsage | None = None
    if usage is not None:
        usage_obj = ModelUsage(
            input_tokens=int(_get(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(_get(usage, "completion_tokens", 0) or 0),
            cost_usd=_explicit_cost(usage),
        )

    return ModelOutput(thought=thought, tool_calls=tool_calls, usage=usage_obj)


def classify_openai_error(exc: Exception) -> ModelError:
    """Map an OpenAI SDK / transport exception to a typed :class:`ModelError`.

    Classification is attribute-based (``status_code``, exception type name,
    message) so it works with real ``openai`` exceptions and plain test doubles:

    * 401/403 → :class:`ModelAuthError` (permanent).
    * 429     → :class:`ModelRateLimitError` (transient).
    * 5xx     → :class:`ModelServerError` (transient).
    * other 4xx → :class:`ModelError` (permanent).
    * timeout → :class:`ModelTimeoutError` (transient).
    * connection/network → :class:`ModelServerError` (transient).
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status in (401, 403):
            return ModelAuthError(f"authentication failed (HTTP {status}): {exc}")
        if status == 429:
            return ModelRateLimitError(f"rate limited (HTTP 429): {exc}")
        if status >= 500:
            return ModelServerError(f"provider server error (HTTP {status}): {exc}")
        if status >= 400:
            return ModelError(f"request rejected (HTTP {status}): {exc}")

    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if (
        "timeout" in name
        or isinstance(exc, (asyncio.TimeoutError, TimeoutError))
        or "timeout" in msg
        or "timed out" in msg
    ):
        return ModelTimeoutError(f"request timed out: {exc}")
    if "connection" in name or any(t in msg for t in ("connection", "network", "refused", "reset")):
        return ModelServerError(f"connection failure: {exc}")
    return ModelError(f"OpenAI API call failed: {exc}")


# --------------------------------------------------------------------------- #
# Token cost pricing (USD per 1M tokens)
# --------------------------------------------------------------------------- #

#: Published list prices, USD per 1M tokens, as ``(input, output)``. ``input`` is
#: the cache-miss rate (cached input is typically ~10x cheaper but providers don't
#: report the cache split in the ``usage`` object, so cache-miss is the safe
#: upper bound). Centralized so a provider price change is a one-line edit.
#:
#: * ``deepseek-chat`` / ``deepseek-reasoner`` — api.deepseek.com (Oct 2025 USD list).
#: * ``qwen3-vl-*`` — Alibaba Cloud Model Studio / DashScope compatible-mode,
#:   entry (≤32K input) tier.
#: * ``gpt-4o`` / ``gpt-4o-mini`` — OpenAI standard list prices.
PRICING_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "deepseek-chat": (0.28, 0.42),
    "deepseek-reasoner": (0.55, 2.19),
    "qwen3-vl-flash": (0.05, 0.40),
    "qwen3-vl-plus": (0.35, 0.35),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def _match_pricing(model: str) -> tuple[float, float] | None:
    """Return ``(input, output)`` USD-per-1M for ``model``.

    Exact name first, then longest-prefix match, so ``qwen3-vl-flash`` and any
    variant such as ``qwen3-vl-flash-32b`` both resolve to the same entry.
    """
    if model in PRICING_USD_PER_MILLION:
        return PRICING_USD_PER_MILLION[model]
    prefixes = [k for k in PRICING_USD_PER_MILLION if model.startswith(k)]
    if not prefixes:
        return None
    return PRICING_USD_PER_MILLION[max(prefixes, key=len)]


def compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost of a model call from its list price; ``0.0`` for an unpriced model.

    ``input_tokens`` / ``output_tokens`` are the counts from the provider's
    ``usage`` object. Models without a pricing entry cost nothing (unknown models
    are tolerated rather than raising, so a misconfigured model name can't crash
    an eval run).
    """
    prices = _match_pricing(model)
    if prices is None:
        return 0.0
    input_usd, output_usd = prices
    return (input_tokens * input_usd + output_tokens * output_usd) / 1_000_000.0


class OpenAIModel:
    """OpenAI-compatible chat adapter (DeepSeek, Qwen/DashScope, OpenAI, …).

    Backs :meth:`generate` with the ``openai`` SDK's :class:`~openai.AsyncOpenAI`
    client, created lazily so the adapter can be constructed without a key (for
    tests) and injected with a ``client`` double. ``base_url`` + ``api_key``
    select the endpoint; ``supports_vision`` gates multimodal content.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        supports_vision: bool = True,
        max_tokens: int = 4096,
        timeout_seconds: float = 120.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.supports_vision = supports_vision
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise ModelNotConfiguredError(
                    f"OpenAIModel({self.model!r}) needs an api_key (or an injected client)"
                )
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout_seconds,
            )
        return self._client

    async def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> ModelOutput:
        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=to_openai_messages(messages),
                tools=list(tools) if tools else None,
                max_tokens=self.max_tokens,
            )
        except ModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - any SDK/transport error is classified
            raise classify_openai_error(exc) from exc

        choices = _get(response, "choices", None) or []
        if not choices:
            raise ModelInvalidResponseError("OpenAI returned an empty choices list")
        output = parse_openai_choice(choices[0], _get(response, "usage", None))
        # When the provider didn't return a monetary cost, derive it from the
        # token counts via the pricing table (real DeepSeek / Qwen runs need this).
        if output.usage is not None and output.usage.cost_usd == 0.0:
            output.usage.cost_usd = compute_cost_usd(
                self.model, output.usage.input_tokens, output.usage.output_tokens
            )
        return output
