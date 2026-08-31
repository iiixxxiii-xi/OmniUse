"""Action executor: run a validated :class:`Action` against a live Playwright page.

Every action returns an :class:`ActionResult` — *including* failures. Execution
never raises to the caller: grounding problems (stale index, element disabled /
hidden / gone), blocked clicks, non-editable targets, and navigation/tab errors
are all converted into structured results with a machine-readable
:class:`ActionError` code and a ``retryable`` flag, so the controller can decide
whether to re-perceive, retry, or surface the problem to the model.

Grounding follows the project's primary signal: ``index -> DOMElement`` via the
perception ``selector_map`` (see :mod:`minicua.action.grounding`). Raw coordinates
are a fallback used only for ``click`` when index grounding is unavailable.
"""

import logging
from typing import Any

from playwright.async_api import Locator, Page, TimeoutError as PlaywrightTimeoutError

from minicua.action.grounding import ground, to_locator
from minicua.action.models import (
    Action,
    ActionResult,
    ActionError,
    ClickParams,
    DoneParams,
    GoBackParams,
    NavigateParams,
    PressParams,
    ScrollParams,
    SwitchTabParams,
    TypeParams,
    WaitParams,
)
from minicua.action.registry import get_default_registry, register_action
from minicua.core.errors import StaleElementError
from minicua.perception.dom import BrowserState, DOMElement

logger = logging.getLogger("minicua.action.executor")

DEFAULT_ACTION_TIMEOUT_MS = 5_000
NAVIGATION_TIMEOUT_MS = 30_000

# Input types that accept free text (excludes checkbox/radio/submit/button/file).
_EDITABLE_INPUT_TYPES = frozenset(
    {
        "text", "search", "email", "url", "tel", "password", "number",
        "date", "datetime-local", "month", "time", "week",
    }
)


class _ActionFailure(Exception):
    """Internal signal carrying a structured :class:`ActionResult`.

    Raised by handlers and converted back to the result by :func:`execute`; never
    escapes the executor's public API.
    """

    def __init__(self, result: ActionResult) -> None:
        self.result = result
        super().__init__(result.error or "action failed")


def _is_navigation_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in ("timeout", "net::err", "connection", "resolv"))


def _is_editable(element: DOMElement) -> bool:
    """Whether an element can accept typed text."""
    if element.tag == "textarea":
        return True
    if element.tag == "input":
        itype = (element.attributes.get("type") or "text").lower()
        return itype in _EDITABLE_INPUT_TYPES
    if element.role == "textbox":
        return True
    if element.attributes.get("contenteditable") == "true":
        return True
    return False


async def _ground(index: int, page: Page, state: BrowserState | None) -> tuple[Locator, DOMElement]:
    """Resolve ``index`` to a live, visible, enabled locator; raises ``_ActionFailure`` otherwise."""
    selector_map = state.selector_map if state is not None else {}
    try:
        element = ground(index, selector_map)
    except StaleElementError:
        raise _ActionFailure(
            ActionResult.fail(
                f"element index {index} not in browser state - page may have changed; "
                "re-perceive before retrying",
                error_code=ActionError.STALE_ELEMENT,
                retryable=True,
            )
        ) from None

    if element.disabled:
        raise _ActionFailure(
            ActionResult.fail(f"element index {index} is disabled", error_code=ActionError.ELEMENT_DISABLED)
        )

    try:
        locator = to_locator(element, page)
    except StaleElementError:
        raise _ActionFailure(
            ActionResult.fail(
                f"element index {index} has no locatable xpath",
                error_code=ActionError.STALE_ELEMENT,
                retryable=True,
            )
        ) from None

    count = await locator.count()
    if count == 0:
        raise _ActionFailure(
            ActionResult.fail(
                f"element index {index} no longer on page - page may have changed",
                error_code=ActionError.ELEMENT_NOT_FOUND,
                retryable=True,
            )
        )
    if count > 1:
        raise _ActionFailure(
            ActionResult.fail(
                f"element index {index} resolves to {count} elements",
                error_code=ActionError.ELEMENT_NOT_FOUND,
                retryable=True,
            )
        )
    if not await locator.is_visible():
        raise _ActionFailure(
            ActionResult.fail(f"element index {index} is not visible", error_code=ActionError.ELEMENT_NOT_VISIBLE)
        )
    return locator, element


# --------------------------------------------------------------------------- #
# Handlers (registered into the default registry)
# --------------------------------------------------------------------------- #


@register_action("click", ClickParams)
async def _click(params: ClickParams, page: Page, state: BrowserState | None = None) -> ActionResult:
    if params.coordinate_x is not None or params.coordinate_y is not None:
        if params.coordinate_x is None or params.coordinate_y is None:
            raise _ActionFailure(
                ActionResult.fail(
                    "both coordinate_x and coordinate_y are required for a coordinate click",
                    error_code=ActionError.INVALID_PARAMS,
                )
            )
        await page.mouse.click(params.coordinate_x, params.coordinate_y)
        return ActionResult.ok(
            f"Clicked at ({params.coordinate_x}, {params.coordinate_y})",
            click_x=params.coordinate_x,
            click_y=params.coordinate_y,
        )

    locator, _ = await _ground(params.index, page, state)
    try:
        await locator.click(timeout=DEFAULT_ACTION_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        raise _ActionFailure(
            ActionResult.fail(
                f"click on element {params.index} blocked or not actionable",
                error_code=ActionError.CLICK_BLOCKED,
                retryable=True,
            )
        ) from None
    return ActionResult.ok(f"Clicked element {params.index}")


@register_action("type", TypeParams)
async def _type(params: TypeParams, page: Page, state: BrowserState | None = None) -> ActionResult:
    locator, element = await _ground(params.index, page, state)
    if not _is_editable(element):
        raise _ActionFailure(
            ActionResult.fail(
                f"element index {params.index} (<{element.tag}>) is not editable",
                error_code=ActionError.ELEMENT_NOT_EDITABLE,
            )
        )
    try:
        if params.clear:
            await locator.fill(params.text)
        else:
            # Append rather than replace: move the caret to the end first so the
            # text is inserted after existing content.
            await locator.press("End")
            await locator.press_sequentially(params.text)
    except PlaywrightTimeoutError:
        raise _ActionFailure(
            ActionResult.fail(
                f"typing into element {params.index} timed out (not actionable)",
                error_code=ActionError.EXECUTION_FAILED,
                retryable=True,
            )
        ) from None
    return ActionResult.ok(f"Typed into element {params.index}")


@register_action("scroll", ScrollParams)
async def _scroll(params: ScrollParams, page: Page, state: BrowserState | None = None) -> ActionResult:
    amount = params.amount
    if amount is None:
        if state is not None and state.viewport is not None and state.viewport.height > 0:
            amount = state.viewport.height
        else:
            amount = await page.evaluate("() => window.innerHeight") or 0

    dx = dy = 0
    if params.direction == "down":
        dy = amount
    elif params.direction == "up":
        dy = -amount
    elif params.direction == "right":
        dx = amount
    elif params.direction == "left":
        dx = -amount

    await page.evaluate("({x, y}) => window.scrollBy(x, y)", {"x": dx, "y": dy})
    return ActionResult.ok(f"Scrolled {params.direction}", direction=params.direction, amount=amount)


@register_action("navigate", NavigateParams)
async def _navigate(params: NavigateParams, page: Page, state: BrowserState | None = None) -> ActionResult:
    try:
        await page.goto(params.url, timeout=NAVIGATION_TIMEOUT_MS)
    except Exception as exc:
        raise _ActionFailure(
            ActionResult.fail(
                f"navigation to {params.url!r} failed: {exc}",
                error_code=ActionError.NAVIGATION_FAILED,
                retryable=_is_navigation_retryable(exc),
            )
        ) from exc
    return ActionResult.ok(f"Navigated to {params.url}", url=params.url)


@register_action("go_back", GoBackParams)
async def _go_back(params: GoBackParams | None, page: Page, state: BrowserState | None = None) -> ActionResult:
    try:
        await page.go_back(timeout=NAVIGATION_TIMEOUT_MS)
    except Exception as exc:
        raise _ActionFailure(
            ActionResult.fail(f"go_back failed: {exc}", error_code=ActionError.GO_BACK_FAILED)
        ) from exc
    return ActionResult.ok("Went back")


@register_action("switch_tab", SwitchTabParams)
async def _switch_tab(params: SwitchTabParams, page: Page, state: BrowserState | None = None) -> ActionResult:
    pages = page.context.pages
    if params.index < 0 or params.index >= len(pages):
        raise _ActionFailure(
            ActionResult.fail(
                f"tab index {params.index} not found ({len(pages)} tabs)",
                error_code=ActionError.TAB_NOT_FOUND,
            )
        )
    await pages[params.index].bring_to_front()
    return ActionResult.ok(f"Switched to tab {params.index}", tab_index=params.index, tab_count=len(pages))


@register_action("press", PressParams)
async def _press(params: PressParams, page: Page, state: BrowserState | None = None) -> ActionResult:
    try:
        await page.keyboard.press(params.keys)
    except Exception as exc:
        raise _ActionFailure(
            ActionResult.fail(f"press {params.keys!r} failed: {exc}", error_code=ActionError.PRESS_FAILED)
        ) from exc
    return ActionResult.ok(f"Pressed {params.keys}", keys=params.keys)


@register_action("wait", WaitParams)
async def _wait(params: WaitParams, page: Page, state: BrowserState | None = None) -> ActionResult:
    await page.wait_for_timeout(int(params.seconds * 1000))
    return ActionResult.ok(f"Waited {params.seconds}s", seconds=params.seconds)


@register_action("done", DoneParams)
async def _done(params: DoneParams, page: Page, state: BrowserState | None = None) -> ActionResult:
    return ActionResult(success=params.success, extracted=params.submission)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


async def execute(
    action: Action,
    page: Page,
    state: BrowserState | None = None,
) -> ActionResult:
    """Execute ``action`` on ``page`` and return a structured :class:`ActionResult`.

    Never raises for action failures: grounding errors, blocked/disabled/hidden
    elements, invalid targets, and unexpected handler exceptions are all returned
    as structured results with an ``error_code`` and ``retryable`` flag.
    """
    registry = get_default_registry()
    registered = registry.get(action.name)
    if registered is None or registered.func is None:
        logger.warning("unknown action %r", action.name)
        return ActionResult.fail(f"unknown action {action.name!r}", error_code=ActionError.UNKNOWN_ACTION)

    try:
        result: Any = await registered.func(action.params, page, state)
    except _ActionFailure as exc:
        logger.warning("action %s failed: %s", action.name, exc.result.error)
        return exc.result
    except Exception as exc:  # noqa: BLE001 - convert unexpected errors to structured result
        logger.exception("unexpected error executing action %s", action.name)
        return ActionResult.fail(
            f"{type(exc).__name__}: {exc}", error_code=ActionError.EXECUTION_FAILED
        )

    if not isinstance(result, ActionResult):
        logger.error("action %s handler returned %r, expected ActionResult", action.name, type(result).__name__)
        return ActionResult.fail(
            f"action {action.name!r} returned an invalid result", error_code=ActionError.EXECUTION_FAILED
        )

    logger.info("action %s succeeded", action.name)
    return result
