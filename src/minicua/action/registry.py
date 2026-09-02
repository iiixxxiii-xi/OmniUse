"""Action registry: the set of actions the model may call, and their LLM tool schemas.

The registry is the single source of truth for *what* actions exist, their
parameter schemas (for tool-calling / JSON schema), and — when a handler is
attached — *how* to run them. The nine built-in actions are derived from
:data:`minicua.action.models.PARAM_MODELS` with human-readable descriptions;
the concrete Playwright handlers are attached by
:mod:`minicua.action.executor` via :func:`register_action`.

Tool schemas are produced in two wire formats:

* :meth:`ActionRegistry.to_tools` — OpenAI function-calling format.
* :meth:`ActionRegistry.to_anthropic_tools` — Anthropic tool-use format.
"""

import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from minicua.action.models import PARAM_MODELS, action_param_model

logger = logging.getLogger("minicua.action.registry")

# Human-readable descriptions surfaced to the model in tool schemas.
ACTION_DESCRIPTIONS: dict[str, str] = {
    "click": (
        "Click an interactive element on the page by its index (from the current "
        "browser state selector map). Optionally fall back to raw coordinates."
    ),
    "type": (
        "Type text into an editable element (input/textarea/contenteditable) by its "
        "index. Clears existing text first unless clear=False."
    ),
    "select": (
        "Select an option from a native <select> dropdown by its visible label "
        "(one of the option labels listed in the page state)."
    ),
    "scroll": "Scroll the viewport in a direction, optionally by a pixel amount.",
    "navigate": "Navigate the current tab to a URL.",
    "go_back": "Go back one entry in the current tab's history.",
    "switch_tab": "Switch focus to a browser tab by its 0-based index.",
    "press": "Send a keyboard key or chord (e.g. 'Enter', 'Control+A').",
    "wait": "Wait for a number of seconds (lets the page settle).",
    "done": "Signal task completion with an optional textual submission.",
}


class RegisteredAction(BaseModel):
    """A single registered action: metadata + optional execution handler."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    param_model: type[BaseModel]
    func: Callable[..., Any] | None = None


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip pydantic ``title``/``$defs`` noise from a JSON schema recursively.

    ``description`` is preserved (it carries field guidance the model needs);
    ``title`` is dropped because it is redundant with the field/action name.
    """
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("title", "$defs"):
            continue
        if isinstance(value, dict):
            cleaned[key] = _clean_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [_clean_schema(x) if isinstance(x, dict) else x for x in value]
        else:
            cleaned[key] = value
    return cleaned


class ActionRegistry:
    """Ordered collection of registered actions with schema generation."""

    def __init__(self, *, default: bool = True) -> None:
        self._actions: dict[str, RegisteredAction] = {}
        if default:
            for name, model in PARAM_MODELS.items():
                self._actions[name] = RegisteredAction(
                    name=name,
                    description=ACTION_DESCRIPTIONS.get(name, ""),
                    param_model=model,
                )

    # -- registration ------------------------------------------------------

    def register(
        self,
        name: str,
        param_model: type[BaseModel],
        description: str | None = None,
        func: Callable[..., Any] | None = None,
    ) -> None:
        """Register (or replace) an action, returning nothing."""
        self._actions[name] = RegisteredAction(
            name=name,
            description=description if description is not None else ACTION_DESCRIPTIONS.get(name, ""),
            param_model=param_model,
            func=func,
        )

    def action(
        self,
        name: str,
        param_model: type[BaseModel],
        description: str | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form of :meth:`register`, used as ``@reg.action(...)``."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(name, param_model, description, func)
            return func

        return decorator

    def discard(self, name: str) -> None:
        """Remove an action from the registry (idempotent)."""
        self._actions.pop(name, None)

    # -- lookup ------------------------------------------------------------

    def get(self, name: str) -> RegisteredAction | None:
        """Return the registered action for ``name`` or ``None`` if unknown."""
        return self._actions.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._actions

    def __len__(self) -> int:
        return len(self._actions)

    def names(self) -> list[str]:
        """Action names in registration order."""
        return list(self._actions)

    # -- schema generation -------------------------------------------------

    def to_tools(self) -> list[dict[str, Any]]:
        """OpenAI function-calling tool list for all registered actions."""
        tools: list[dict[str, Any]] = []
        for action in self._actions.values():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": action.name,
                        "description": action.description,
                        "parameters": _clean_schema(action.param_model.model_json_schema()),
                    },
                }
            )
        return tools

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        """Anthropic tool-use format for all registered actions."""
        tools: list[dict[str, Any]] = []
        for action in self._actions.values():
            tools.append(
                {
                    "name": action.name,
                    "description": action.description,
                    "input_schema": _clean_schema(action.param_model.model_json_schema()),
                }
            )
        return tools


# --------------------------------------------------------------------------- #
# Default registry (lazy singleton)
# --------------------------------------------------------------------------- #

_default_registry: ActionRegistry | None = None


def get_default_registry() -> ActionRegistry:
    """Return the process-wide default :class:`ActionRegistry` (lazy singleton)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ActionRegistry()
    return _default_registry


def register_action(
    name: str,
    param_model: type[BaseModel] | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering a handler into the default registry.

    ``param_model`` may be omitted to look it up from :func:`action_param_model`.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        model = param_model if param_model is not None else action_param_model(name)
        get_default_registry().register(name, model, description, func)
        return func

    return decorator
