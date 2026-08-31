"""Desktop action space: coordinate-grounded actions + a registry + executor.

Desktop mode has **no DOM**, so grounding is by *screen coordinates* (plus a shell
action) instead of the browser's ``index -> DOMElement``. This module is the
desktop twin of :mod:`minicua.action`: it defines pydantic parameter schemas, a
:class:`DesktopAction` union, and an :class:`~minicua.action.registry.ActionRegistry`
populated with concrete handlers that drive a :class:`~minicua.desktop.env.DesktopEnvironment`.

The desktop actions are a *second* action space: they live in their own registry
(:func:`get_desktop_registry`), so browser mode keeps its nine DOM actions
untouched while desktop mode exposes the coordinate + shell surface. Executing a
desktop action always returns a structured :class:`ActionResult` (reusing the
browser layer's model), never an unhandled exception.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from minicua.action.models import ActionResult, ActionError, DoneParams
from minicua.action.registry import ActionRegistry

logger = logging.getLogger("minicua.desktop.actions")

# --------------------------------------------------------------------------- #
# Parameter schemas
# --------------------------------------------------------------------------- #


class DesktopClickParams(BaseModel):
    """Click the left mouse button at screen coordinates (x, y)."""

    x: int = Field(ge=0, description="Screen x coordinate.")
    y: int = Field(ge=0, description="Screen y coordinate.")


class DesktopMoveParams(BaseModel):
    """Move the mouse pointer to screen coordinates (x, y) without clicking."""

    x: int = Field(ge=0, description="Screen x coordinate.")
    y: int = Field(ge=0, description="Screen y coordinate.")


class DesktopDoubleClickParams(BaseModel):
    """Double-click the left mouse button at screen coordinates (x, y)."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)


class DesktopRightClickParams(BaseModel):
    """Right-click at screen coordinates (x, y)."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)


class DesktopDragParams(BaseModel):
    """Drag the left mouse button from (x1, y1) to (x2, y2)."""

    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)


class DesktopTypeTextParams(BaseModel):
    """Type text into the currently focused window / field."""

    text: str = Field(min_length=1, description="Text to type.")

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("text must not be blank")
        return text


class DesktopPressParams(BaseModel):
    """Press a single key (e.g. 'enter', 'tab', 'a')."""

    key: str = Field(min_length=1, description="Key name to press.")

    @field_validator("key")
    @classmethod
    def _key_not_blank(cls, key: str) -> str:
        if not key.strip():
            raise ValueError("key must not be blank")
        return key


class DesktopHotkeyParams(BaseModel):
    """Press a key chord together (e.g. ['ctrl', 'c'])."""

    keys: list[str] = Field(min_length=1, description="Keys to press simultaneously.")

    @field_validator("keys")
    @classmethod
    def _keys_nonempty(cls, keys: list[str]) -> list[str]:
        if any(not k.strip() for k in keys):
            raise ValueError("hotkey keys must be non-empty strings")
        return keys


class DesktopScrollParams(BaseModel):
    """Scroll the mouse wheel; positive scrolls down, negative scrolls up."""

    amount: int = Field(description="Scroll amount in wheel clicks (signed).")


class DesktopShellParams(BaseModel):
    """Run a shell command and return its stdout / stderr / exit code."""

    command: str = Field(min_length=1, description="Shell command to run.")

    @field_validator("command")
    @classmethod
    def _command_not_blank(cls, command: str) -> str:
        if not command.strip():
            raise ValueError("command must not be blank")
        return command


# --------------------------------------------------------------------------- #
# Action union
# --------------------------------------------------------------------------- #

DesktopActionName = Literal[
    "click", "move_to", "double_click", "right_click", "drag",
    "type_text", "press", "hotkey", "scroll", "shell", "done",
]

#: name -> parameter schema, the desktop single source of truth.
DESKTOP_PARAM_MODELS: dict[str, type[BaseModel]] = {
    "click": DesktopClickParams,
    "move_to": DesktopMoveParams,
    "double_click": DesktopDoubleClickParams,
    "right_click": DesktopRightClickParams,
    "drag": DesktopDragParams,
    "type_text": DesktopTypeTextParams,
    "press": DesktopPressParams,
    "hotkey": DesktopHotkeyParams,
    "scroll": DesktopScrollParams,
    "shell": DesktopShellParams,
    "done": DoneParams,
}

DESKTOP_ACTION_NAMES: frozenset[str] = frozenset(DESKTOP_PARAM_MODELS)

DesktopActionParams = (
    DesktopClickParams
    | DesktopMoveParams
    | DesktopDoubleClickParams
    | DesktopRightClickParams
    | DesktopDragParams
    | DesktopTypeTextParams
    | DesktopPressParams
    | DesktopHotkeyParams
    | DesktopScrollParams
    | DesktopShellParams
    | DoneParams
)


class DesktopAction(BaseModel):
    """A single validated desktop action: ``name`` discriminates ``params``."""

    name: DesktopActionName
    params: DesktopActionParams | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_params_from_dict(cls, data: Any) -> Any:
        if isinstance(data, dict):
            model = DESKTOP_PARAM_MODELS.get(data.get("name"))
            raw = data.get("params")
            if model is not None and isinstance(raw, dict):
                return {**data, "params": model.model_validate(raw)}
        return data

    @model_validator(mode="after")
    def _validate_params_match_name(self) -> "DesktopAction":
        expected = DESKTOP_PARAM_MODELS[self.name]
        if self.params is None:
            raise ValueError(f"action {self.name!r} requires params of type {expected.__name__}")
        if not isinstance(self.params, expected):
            raise ValueError(
                f"action {self.name!r} expects {expected.__name__} params, got {type(self.params).__name__}"
            )
        return self


def desktop_param_model(name: str) -> type[BaseModel]:
    """Return the parameter schema class for a desktop action name (raises on unknown)."""
    try:
        return DESKTOP_PARAM_MODELS[name]
    except KeyError:
        raise KeyError(f"unknown desktop action name: {name!r}") from None


# --------------------------------------------------------------------------- #
# Human-readable descriptions (surfaced to the model in tool schemas)
# --------------------------------------------------------------------------- #

DESKTOP_ACTION_DESCRIPTIONS: dict[str, str] = {
    "click": "Click the left mouse button at screen coordinates (x, y).",
    "move_to": "Move the mouse pointer to (x, y) without clicking.",
    "double_click": "Double-click the left mouse button at (x, y).",
    "right_click": "Right-click at screen coordinates (x, y).",
    "drag": "Drag the mouse from (x1, y1) to (x2, y2), holding the left button.",
    "type_text": "Type text into the currently focused window.",
    "press": "Press a single keyboard key (e.g. 'enter', 'tab').",
    "hotkey": "Press a key chord together (e.g. ['ctrl', 'c']).",
    "scroll": "Scroll the mouse wheel (positive scrolls down, negative up).",
    "shell": "Run a shell command and return its stdout, stderr, and exit code.",
    "done": "Signal task completion with an optional textual submission.",
}


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #


class _DesktopActionFailure(Exception):
    """Internal signal carrying a structured :class:`ActionResult`."""

    def __init__(self, result: ActionResult) -> None:
        self.result = result
        super().__init__(result.error or "desktop action failed")


def _input_failure(what: str, exc: Exception) -> _DesktopActionFailure:
    return _DesktopActionFailure(
        ActionResult.fail(f"{what} failed: {exc}", error_code=ActionError.INPUT_FAILED)
    )


async def _click(params: DesktopClickParams, env: Any, state: Any = None) -> ActionResult:
    try:
        env.click(params.x, params.y)
    except Exception as exc:  # noqa: BLE001
        raise _input_failure(f"click at ({params.x}, {params.y})", exc) from exc
    return ActionResult.ok(f"Clicked at ({params.x}, {params.y})", x=params.x, y=params.y)


async def _move_to(params: DesktopMoveParams, env: Any, state: Any = None) -> ActionResult:
    try:
        env.move_to(params.x, params.y)
    except Exception as exc:  # noqa: BLE001
        raise _input_failure(f"move to ({params.x}, {params.y})", exc) from exc
    return ActionResult.ok(f"Moved to ({params.x}, {params.y})", x=params.x, y=params.y)


async def _double_click(params: DesktopDoubleClickParams, env: Any, state: Any = None) -> ActionResult:
    try:
        env.double_click(params.x, params.y)
    except Exception as exc:  # noqa: BLE001
        raise _input_failure(f"double-click at ({params.x}, {params.y})", exc) from exc
    return ActionResult.ok(f"Double-clicked at ({params.x}, {params.y})", x=params.x, y=params.y)


async def _right_click(params: DesktopRightClickParams, env: Any, state: Any = None) -> ActionResult:
    try:
        env.right_click(params.x, params.y)
    except Exception as exc:  # noqa: BLE001
        raise _input_failure(f"right-click at ({params.x}, {params.y})", exc) from exc
    return ActionResult.ok(f"Right-clicked at ({params.x}, {params.y})", x=params.x, y=params.y)


async def _drag(params: DesktopDragParams, env: Any, state: Any = None) -> ActionResult:
    try:
        env.drag(params.x1, params.y1, params.x2, params.y2)
    except Exception as exc:  # noqa: BLE001
        raise _input_failure(
            f"drag from ({params.x1}, {params.y1}) to ({params.x2}, {params.y2})", exc
        ) from exc
    return ActionResult.ok(
        f"Dragged from ({params.x1}, {params.y1}) to ({params.x2}, {params.y2})",
        x1=params.x1, y1=params.y1, x2=params.x2, y2=params.y2,
    )


async def _type_text(params: DesktopTypeTextParams, env: Any, state: Any = None) -> ActionResult:
    try:
        env.type_text(params.text)
    except Exception as exc:  # noqa: BLE001
        raise _input_failure("type text", exc) from exc
    return ActionResult.ok(f"Typed {params.text!r}")


async def _press(params: DesktopPressParams, env: Any, state: Any = None) -> ActionResult:
    try:
        env.press(params.key)
    except Exception as exc:  # noqa: BLE001
        raise _input_failure(f"press {params.key!r}", exc) from exc
    return ActionResult.ok(f"Pressed {params.key}", key=params.key)


async def _hotkey(params: DesktopHotkeyParams, env: Any, state: Any = None) -> ActionResult:
    try:
        env.hotkey(*params.keys)
    except Exception as exc:  # noqa: BLE001
        raise _input_failure(f"hotkey {'+'.join(params.keys)!r}", exc) from exc
    return ActionResult.ok(f"Pressed hotkey {'+'.join(params.keys)}", keys=params.keys)


async def _scroll(params: DesktopScrollParams, env: Any, state: Any = None) -> ActionResult:
    try:
        env.scroll(params.amount)
    except Exception as exc:  # noqa: BLE001
        raise _input_failure(f"scroll by {params.amount}", exc) from exc
    return ActionResult.ok(f"Scrolled by {params.amount}", amount=params.amount)


async def _shell(params: DesktopShellParams, env: Any, state: Any = None) -> ActionResult:
    result = env.run_shell(params.command)
    if result.timed_out:
        return ActionResult.fail(
            f"shell command {params.command!r} timed out",
            error_code=ActionError.SHELL_TIMEOUT,
            retryable=True,
        )
    if result.returncode != 0:
        return ActionResult.fail(
            f"shell command {params.command!r} exited with code {result.returncode}: {result.stderr}",
            error_code=ActionError.SHELL_FAILED,
        )
    return ActionResult.ok(
        f"Ran command (exit 0)",
        command=params.command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


async def _done(params: DoneParams, env: Any, state: Any = None) -> ActionResult:
    return ActionResult(success=params.success, extracted=params.submission)


# --------------------------------------------------------------------------- #
# Registry (lazy singleton)
# --------------------------------------------------------------------------- #

_desktop_registry: ActionRegistry | None = None


def get_desktop_registry() -> ActionRegistry:
    """Return the process-wide desktop :class:`ActionRegistry` (lazy singleton)."""
    global _desktop_registry
    if _desktop_registry is None:
        reg = ActionRegistry(default=False)
        for name, model in DESKTOP_PARAM_MODELS.items():
            func = _HANDLERS[name]
            reg.register(name, model, DESKTOP_ACTION_DESCRIPTIONS.get(name, ""), func)
        _desktop_registry = reg
    return _desktop_registry


_HANDLERS: dict[str, Any] = {
    "click": _click,
    "move_to": _move_to,
    "double_click": _double_click,
    "right_click": _right_click,
    "drag": _drag,
    "type_text": _type_text,
    "press": _press,
    "hotkey": _hotkey,
    "scroll": _scroll,
    "shell": _shell,
    "done": _done,
}


async def execute_desktop(
    action: DesktopAction,
    env: Any,
    state: Any = None,
) -> ActionResult:
    """Execute ``action`` against ``env`` and return a structured :class:`ActionResult`.

    Never raises for action failures: input errors (mouse/keyboard), non-zero shell
    exits, timeouts, and unexpected handler exceptions are all returned as
    structured results with an ``error_code`` and ``retryable`` flag.
    """
    registry = get_desktop_registry()
    registered = registry.get(action.name)
    if registered is None or registered.func is None:
        logger.warning("unknown desktop action %r", action.name)
        return ActionResult.fail(f"unknown action {action.name!r}", error_code=ActionError.UNKNOWN_ACTION)

    try:
        result: Any = await registered.func(action.params, env, state)
    except _DesktopActionFailure as exc:
        logger.warning("desktop action %s failed: %s", action.name, exc.result.error)
        return exc.result
    except Exception as exc:  # noqa: BLE001 - convert unexpected errors to structured result
        logger.exception("unexpected error executing desktop action %s", action.name)
        return ActionResult.fail(f"{type(exc).__name__}: {exc}", error_code=ActionError.EXECUTION_FAILED)

    if not isinstance(result, ActionResult):
        logger.error("desktop action %s handler returned %r, expected ActionResult", action.name, type(result).__name__)
        return ActionResult.fail(
            f"action {action.name!r} returned an invalid result", error_code=ActionError.EXECUTION_FAILED
        )
    return result
