"""Action layer data models: per-action parameter schemas and the unified ``Action``.

The model emits one of a fixed set of actions, each with a pydantic-validated
parameter payload. ``Action.name`` is the discriminator: it both selects the
handler in the executor and pins which parameter schema ``params`` must match.
A single ``Action`` model (rather than nine loose dicts) gives the controller a
type-safe boundary — every action it receives is guaranteed to be well-formed.

``ActionResult`` is the structured outcome returned by the executor for every
action, *including* failures. Failures are data (``success=False`` + ``error`` +
``error_code`` + ``retryable``), never raised exceptions, so the controller can
decide whether to re-perceive, retry, or surface the problem to the model.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------- #
# Parameter schemas
# --------------------------------------------------------------------------- #


class ClickParams(BaseModel):
    """Click an interactive element by its grounding index (or raw coordinates)."""

    index: int = Field(ge=1, description="Element index from the current browser state selector map.")
    coordinate_x: int | None = Field(
        default=None, ge=0, description="Fallback: x coordinate (CSS px) when index grounding is unavailable."
    )
    coordinate_y: int | None = Field(
        default=None, ge=0, description="Fallback: y coordinate (CSS px) when index grounding is unavailable."
    )


class TypeParams(BaseModel):
    """Type text into an editable element (clears existing text by default)."""

    index: int = Field(ge=1, description="Element index of the input/textarea/contenteditable to type into.")
    text: str = Field(min_length=1, description="Text to type.")
    clear: bool = Field(default=True, description="Clear existing content before typing.")


class ScrollParams(BaseModel):
    """Scroll the viewport by a direction, optionally a pixel amount."""

    direction: Literal["up", "down", "left", "right"] = Field(
        default="down", description="Scroll direction."
    )
    amount: int | None = Field(
        default=None, ge=0, description="Scroll amount in CSS px; defaults to one viewport when omitted."
    )


class NavigateParams(BaseModel):
    """Navigate the current tab to a URL."""

    url: str = Field(min_length=1, description="Target URL.")


class GoBackParams(BaseModel):
    """Go back one entry in the current tab's history. No parameters."""


class SwitchTabParams(BaseModel):
    """Switch focus to a tab by its 0-based index."""

    index: int = Field(ge=0, description="0-based tab index.")


class PressParams(BaseModel):
    """Send a keyboard key / chord (e.g. 'Enter', 'Control+A')."""

    keys: str = Field(min_length=1, description="Key or key combination to press.")


class WaitParams(BaseModel):
    """Wait a number of seconds (no-op used to let the page settle)."""

    seconds: float = Field(default=1.0, gt=0.0, description="Seconds to wait.")


class DoneParams(BaseModel):
    """Signal task completion with an optional textual submission."""

    success: bool = Field(default=True, description="Whether the task succeeded.")
    submission: str | None = Field(default=None, description="Optional final answer / result text.")


# --------------------------------------------------------------------------- #
# Action union
# --------------------------------------------------------------------------- #

ActionName = Literal[
    "click", "type", "scroll", "navigate", "go_back",
    "switch_tab", "press", "wait", "done",
]

#: name -> parameter schema, the single source of truth for the action set.
PARAM_MODELS: dict[str, type[BaseModel]] = {
    "click": ClickParams,
    "type": TypeParams,
    "scroll": ScrollParams,
    "navigate": NavigateParams,
    "go_back": GoBackParams,
    "switch_tab": SwitchTabParams,
    "press": PressParams,
    "wait": WaitParams,
    "done": DoneParams,
}

ACTION_NAMES: frozenset[str] = frozenset(PARAM_MODELS)

#: Actions whose parameter payload may legitimately be ``None``.
_NO_PARAM_ACTIONS: frozenset[str] = frozenset({"go_back"})

ActionParams = (
    ClickParams
    | TypeParams
    | ScrollParams
    | NavigateParams
    | GoBackParams
    | SwitchTabParams
    | PressParams
    | WaitParams
    | DoneParams
)


class Action(BaseModel):
    """A single validated action: ``name`` discriminates the ``params`` schema."""

    name: ActionName
    params: ActionParams | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_params_from_dict(cls, data: Any) -> Any:
        # Allow ``Action(name="click", params={"index": 1})`` — coerce a raw dict
        # to the name-specific model before the union field is validated, so the
        # controller can feed us LLM JSON output directly.
        if isinstance(data, dict):
            model = PARAM_MODELS.get(data.get("name"))
            raw = data.get("params")
            if model is not None and isinstance(raw, dict):
                return {**data, "params": model.model_validate(raw)}
        return data

    @model_validator(mode="after")
    def _validate_params_match_name(self) -> "Action":
        expected = PARAM_MODELS[self.name]
        if self.params is None:
            if self.name not in _NO_PARAM_ACTIONS:
                raise ValueError(f"action {self.name!r} requires params of type {expected.__name__}")
            return self
        if not isinstance(self.params, expected):
            raise ValueError(
                f"action {self.name!r} expects {expected.__name__} params, got {type(self.params).__name__}"
            )
        return self


def action_param_model(name: str) -> type[BaseModel]:
    """Return the parameter schema class for an action name (raises on unknown)."""
    try:
        return PARAM_MODELS[name]
    except KeyError:
        raise KeyError(f"unknown action name: {name!r}") from None


# --------------------------------------------------------------------------- #
# Structured execution result
# --------------------------------------------------------------------------- #


class ActionError(str, Enum):
    """Machine-readable failure codes for :class:`ActionResult`."""

    UNKNOWN_ACTION = "unknown_action"
    INVALID_PARAMS = "invalid_params"
    STALE_ELEMENT = "stale_element"
    ELEMENT_NOT_FOUND = "element_not_found"
    ELEMENT_DISABLED = "element_disabled"
    ELEMENT_NOT_VISIBLE = "element_not_visible"
    ELEMENT_NOT_EDITABLE = "element_not_editable"
    CLICK_BLOCKED = "click_blocked"
    NAVIGATION_FAILED = "navigation_failed"
    GO_BACK_FAILED = "go_back_failed"
    TAB_NOT_FOUND = "tab_not_found"
    PRESS_FAILED = "press_failed"
    EXECUTION_FAILED = "execution_failed"


class ActionResult(BaseModel):
    """Structured outcome of an action execution (success OR failure)."""

    success: bool
    error: str | None = None
    error_code: ActionError | None = None
    retryable: bool = False
    extracted: str | None = None  # human-readable summary of what happened
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, extracted: str | None = None, **metadata: Any) -> "ActionResult":
        """Build a success result (optionally with extra metadata)."""
        return cls(success=True, extracted=extracted, metadata=metadata)

    @classmethod
    def fail(
        cls,
        error: str,
        *,
        error_code: ActionError,
        retryable: bool = False,
    ) -> "ActionResult":
        """Build a structured failure result."""
        return cls(success=False, error=error, error_code=error_code, retryable=retryable)
