"""Grounding: turn a model-emitted index into a concrete DOM element / locator.

The model references page elements by the 1-based ``index`` assigned by the
perception serializer (see :mod:`minicua.perception.serializer`). Grounding is
the bridge from that symbolic handle back to a real Playwright locator:

* :func:`ground` — pure ``index -> DOMElement`` lookup over the current
  ``selector_map``. Missing indexes raise :class:`StaleElementError` (the page
  snapshot the model saw is gone).
* :func:`to_locator` — ``DOMElement -> Locator`` via the element's XPath. A
  ``DOMElement`` without an XPath cannot be grounded (again ``StaleElementError``).

Index is the *primary* grounding signal because it is stable across re-serializations
and cheap to emit. Raw coordinates are a fallback the executor applies only when
index grounding is unavailable (see :mod:`minicua.action.executor`).
"""

import logging

from playwright.async_api import Locator, Page

from minicua.core.errors import StaleElementError
from minicua.perception.dom import DOMElement

logger = logging.getLogger("minicua.action.grounding")


def ground(index: int, selector_map: dict[int, DOMElement]) -> DOMElement:
    """Resolve a model-emitted ``index`` to a :class:`DOMElement`.

    Raises :class:`StaleElementError` when the index is absent from the current
    selector map — i.e. the page snapshot has changed since the model last saw
    it and the caller should re-perceive rather than guess.
    """
    element = selector_map.get(index)
    if element is None:
        logger.warning("grounding failed: index %s not in selector map", index)
        raise StaleElementError(index)
    return element


def to_locator(element: DOMElement, page: Page) -> Locator:
    """Build a Playwright :class:`Locator` for ``element`` from its XPath.

    Raises :class:`StaleElementError` if the element carries no XPath (it cannot
    be re-located on the live page).
    """
    if not element.xpath:
        logger.warning("grounding failed: element %s has no xpath", element.index)
        raise StaleElementError(element.index)
    return page.locator(f"xpath={element.xpath}")
