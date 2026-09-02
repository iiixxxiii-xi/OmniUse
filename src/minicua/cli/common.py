"""Shared helpers for the CLI commands."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from minicua.controller.llm import ChatModel, FakeModel, OpenAIModel

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

#: The two agent operating modes selectable via ``--mode``.
VALID_MODES = ("browser", "desktop")


def require_vision_model(model: ChatModel, mode: str) -> None:
    """Raise :class:`ValueError` if ``mode`` is desktop and ``model`` lacks vision.

    Desktop mode perceives purely through screenshots, so a text-only model
    cannot drive it. Callers turn this into a clean CLI error + exit code.
    """
    if mode == "desktop" and not getattr(model, "supports_vision", False):
        raise ValueError(
            "desktop mode requires a vision model (e.g. --model dashscope/qwen3-vl-flash); "
            "the selected model does not support vision"
        )


def load_script(path: str | Path) -> list:
    """Load scripted :class:`FakeModel` responses from a JSON file.

    Accepts a JSON list of response dicts (``[{"name": "click", "params":
    {"index": 1}}, ...]``) or a single response object. Raises ``OSError`` /
    ``ValueError`` on a missing / malformed file; callers turn those into a clean
    CLI error + exit code.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("script file must be a JSON list (or single object) of responses")


def resolve_model(model_id: str) -> ChatModel:
    """Map a ``--model`` string to a :class:`ChatModel` instance.

    ``fake`` (default) needs no API key. ``deepseek/<id>`` and ``dashscope/<id>``
    (alias ``qwen/<id>``) select the OpenAI-compatible adapter and read their
    credentials from the environment (``.env`` via python-dotenv). DeepSeek is
    text-only (DOM); DashScope/Qwen is vision-capable.
    """
    model_id = (model_id or "").strip()
    if model_id in ("", "fake", "FakeModel", "mock", "MockModel"):
        return FakeModel()

    if model_id.startswith("deepseek/"):
        name = model_id.split("/", 1)[1]
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("deepseek model selected but DEEPSEEK_API_KEY is not set (check .env)")
        return OpenAIModel(
            model=name,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
            api_key=api_key,
            supports_vision=False,
            # DeepSeek V4 models are reasoning models: their chain-of-thought
            # tokens share the same output budget as tool calls. A low ceiling
            # truncates the response before the model emits any tool call
            # ("no tool calls"), so give the reasoning head room to breathe.
            max_tokens=8192,
            # Disable the reasoning/thinking head for V4 models: with thinking
            # on, the model narrates its plan into ``content`` instead of
            # actually emitting tool calls on long multi-step tasks. Combine
            # with a forced tool call so it can never "narrate instead of act".
            extra_body={"thinking": {"type": "disabled"}} if "v4" in name else None,
            tool_choice="required" if "v4" in name else None,
        )

    if model_id.startswith(("dashscope/", "qwen/")):
        name = model_id.split("/", 1)[1]
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("dashscope/qwen model selected but DASHSCOPE_API_KEY is not set (check .env)")
        return OpenAIModel(
            model=name,
            base_url=os.environ.get("DASHSCOPE_BASE_URL", DASHSCOPE_BASE_URL),
            api_key=api_key,
            supports_vision=True,
        )

    raise ValueError(
        f"unrecognized model '{model_id}'; use 'fake', 'deepseek/<id>', 'dashscope/<id>', or 'qwen/<id>'"
    )
