"""Task 3.3: ActionRegistry — register actions and generate LLM tool schemas."""

from minicua.action.models import ClickParams
from minicua.action.registry import ActionRegistry, get_default_registry, register_action


def test_registry_generates_tools_for_builtin_actions():
    reg = ActionRegistry()
    tools = reg.to_tools()
    names = [t["function"]["name"] for t in tools]
    assert "click" in names
    assert "type" in names
    assert "done" in names


def test_registry_has_all_nine_default_actions():
    reg = ActionRegistry()
    assert len(reg) == 9


def test_registry_tool_schema_is_openai_function_format():
    reg = ActionRegistry()
    click = next(t for t in reg.to_tools() if t["function"]["name"] == "click")
    assert click["type"] == "function"
    params = click["function"]["parameters"]
    assert params["type"] == "object"
    assert "index" in params["properties"]
    assert "coordinate_x" in params["properties"]
    assert "coordinate_y" in params["properties"]
    # index is optional now (coordinate-only clicks are valid), so it is not required.
    assert "index" not in params.get("required", [])


def test_registry_tool_schema_strips_titles():
    reg = ActionRegistry()
    click = next(t for t in reg.to_tools() if t["function"]["name"] == "click")
    params = click["function"]["parameters"]
    assert "title" not in params
    assert "title" not in params["properties"]["index"]


def test_registry_anthropic_tool_format():
    reg = ActionRegistry()
    tools = reg.to_anthropic_tools()
    click = next(t for t in tools if t["name"] == "click")
    assert "input_schema" in click
    assert click["input_schema"]["type"] == "object"


def test_register_custom_action_via_method_decorator():
    reg = ActionRegistry(default=False)

    @reg.action("custom", ClickParams, "click a custom thing")
    async def custom(params, page):
        return "ok"

    assert "custom" in reg
    assert any(t["function"]["name"] == "custom" for t in reg.to_tools())


def test_register_action_standalone_decorator_targets_default_registry():
    reg = get_default_registry()
    name = "__test_custom_action__"
    try:

        @register_action(name, ClickParams, "test-only action")
        async def _fn(params, page):
            return "ok"

        assert name in reg
    finally:
        reg.discard(name)


def test_get_default_registry_is_singleton():
    assert get_default_registry() is get_default_registry()


def test_registry_get_and_contains():
    reg = ActionRegistry()
    assert "click" in reg
    assert "nope" not in reg
    assert reg.get("click").param_model is ClickParams
