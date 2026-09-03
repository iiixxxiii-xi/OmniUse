"""CLI model selection: deepseek / dashscope (qwen) / fake."""

import pytest

from minicua.cli.common import resolve_model
from minicua.controller.llm import FakeModel, OpenAIModel


def test_resolve_model_fake_default():
    assert isinstance(resolve_model("fake"), FakeModel)
    assert isinstance(resolve_model(""), FakeModel)
    assert isinstance(resolve_model("mock"), FakeModel)


def test_resolve_model_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deep")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    m = resolve_model("deepseek/deepseek-chat")
    assert isinstance(m, OpenAIModel)
    assert m.model == "deepseek-chat"
    assert m.api_key == "sk-deep"
    assert m.base_url == "https://api.deepseek.com"
    assert m.supports_vision is False


def test_resolve_model_deepseek_default_base_url(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deep")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    m = resolve_model("deepseek/deepseek-chat")
    assert m.base_url == "https://api.deepseek.com"


def test_resolve_model_deepseek_vision(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deep")
    m = resolve_model("deepseek/deepseek-v4-flash-vision-exp")
    assert isinstance(m, OpenAIModel)
    assert m.model == "deepseek-v4-flash-vision-exp"
    assert m.supports_vision is True
    # reasoning head off (so tool_choice="required" is accepted) + forced tool call
    assert m.extra_body == {"thinking": {"type": "disabled"}}
    assert m.tool_choice == "required"


def test_resolve_model_deepseek_v4_text_not_vision(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deep")
    m = resolve_model("deepseek/deepseek-v4-flash")
    assert m.supports_vision is False
    assert m.extra_body == {"thinking": {"type": "disabled"}}
    assert m.tool_choice == "required"


def test_resolve_model_dashscope_vision(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dash")
    m = resolve_model("dashscope/qwen3-vl-flash")
    assert isinstance(m, OpenAIModel)
    assert m.model == "qwen3-vl-flash"
    assert m.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert m.supports_vision is True


def test_resolve_model_qwen_alias(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dash")
    m = resolve_model("qwen/qwen3-vl-flash")
    assert m.model == "qwen3-vl-flash"
    assert m.supports_vision is True


def test_resolve_model_unknown_raises():
    with pytest.raises(ValueError):
        resolve_model("bogus/x")


def test_resolve_model_missing_deepseek_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError):
        resolve_model("deepseek/deepseek-chat")


def test_resolve_model_missing_dashscope_key_raises(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        resolve_model("dashscope/qwen3-vl-flash")
