"""Stale-element recovery: re-ground an element whose index moved between perceptions.

When an action fails with ``STALE_ELEMENT`` / ``ELEMENT_NOT_FOUND``, the index the
model emitted no longer names the element it meant. Rather than failing outright,
we re-perceive and try to *relocalize* the old element into the fresh selector map
by matching identity signals in order of strength:

1. ``stable_hash`` — strongest: derived from xpath + identity attributes + text.
2. ``xpath`` — the element's exact DOM path.
3. ``ax_name`` — its accessible name (aria-label / label / title).

Only when every signal misses do we give up (``None``) and let the caller escalate
to re-perceiving + prompting the model to change strategy.
"""

import logging
from typing import Any

from playwright.async_api import Page

from minicua.action.models import Action
from minicua.perception.dom import BrowserState, DOMElement
from minicua.perception.extract import extract_state

logger = logging.getLogger("minicua.recovery.stale")

# Actions that reference a DOM element by ``index`` (vs. a tab index or nothing).
_RELOCALIZABLE_ACTIONS = frozenset({"click", "type"})


def _find_by_stable_hash(old: DOMElement, new_map: dict[int, DOMElement]) -> int | None:
    if not old.stable_hash:
        return None
    for index, element in new_map.items():
        if element.stable_hash and element.stable_hash == old.stable_hash:
            return index
    return None


def _find_by_xpath(old: DOMElement, new_map: dict[int, DOMElement]) -> int | None:
    if not old.xpath:
        return None
    for index, element in new_map.items():
        if element.xpath and element.xpath == old.xpath:
            return index
    return None


def _find_by_ax_name(old: DOMElement, new_map: dict[int, DOMElement]) -> int | None:
    if not old.ax_name:
        return None
    for index, element in new_map.items():
        if element.ax_name and element.ax_name == old.ax_name:
            return index
    return None


def relocalize(old: DOMElement, new_map: dict[int, DOMElement]) -> int | None:
    """Find ``old``'s new index in ``new_map``, or ``None`` if it is gone.

    Degradation order: ``stable_hash`` → ``xpath`` → ``ax_name`` → give up. Each
    signal is only consulted when the element carries a non-empty value for it, so
    an empty ``stable_hash`` never falsely matches another empty ``stable_hash``.
    """
    for finder in (_find_by_stable_hash, _find_by_xpath, _find_by_ax_name):
        index = finder(old, new_map)
        if index is not None:
            logger.debug("relocalized stale element %s -> index %s", old.index, index)
            return index
    logger.warning("could not relocalize stale element index %s", old.index)
    return None


def relocalize_action(
    action: Action,
    old: DOMElement,
    new_map: dict[int, DOMElement],
) -> Action | None:
    """Return a copy of ``action`` with its index updated to ``old``'s new index.

    ``None`` when the action does not reference a DOM element index (``go_back``,
    ``scroll``, ``switch_tab`` …) or when ``old`` cannot be relocalized.
    """
    if action.name not in _RELOCALIZABLE_ACTIONS or action.params is None:
        return None
    new_index = relocalize(old, new_map)
    if new_index is None:
        return None
    return action.model_copy(update={"params": action.params.model_copy(update={"index": new_index})})


def _action_index(action: Action) -> int | None:
    if action.name not in _RELOCALIZABLE_ACTIONS or action.params is None:
        return None
    params: Any = action.params
    index = getattr(params, "index", None)
    return index if isinstance(index, int) else None


async def recover_stale(
    action: Action,
    old_state: BrowserState,
    page: Page,
) -> tuple[Action, BrowserState] | None:
    """Re-perceive the page and relocalize ``action``'s stale element.

    Returns ``(relocalized_action, fresh_state)`` so the caller can re-execute the
    action against the fresh selector map, or ``None`` when the element is truly
    gone (the caller should then re-perceive and prompt the model to change course).
    """
    old_index = _action_index(action)
    if old_index is None:
        return None
    old_element = old_state.selector_map.get(old_index)
    if old_element is None:
        logger.warning("stale index %s has no old element to relocalize", old_index)
        return None

    fresh_state = await extract_state(page)
    relocalized = relocalize_action(action, old_element, fresh_state.selector_map)
    if relocalized is None:
        return None
    return relocalized, fresh_state
