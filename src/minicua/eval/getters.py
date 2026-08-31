"""Eval getters: read the *final* browser state a task left behind.

A getter is the first stage of the declarative evaluator pipeline (getter →
metric → conj). It answers one concrete question about the browser — "what is the
current URL?", "does element ``#result`` exist?", "what text does it hold?",
"is this cookie set?" — and returns a plain value for the metric to compare.

Every getter is **defensive**: a page that has navigated away, an element that
vanished, or a security origin that forbids localStorage access degrades to a
safe sentinel (``None`` / ``False`` / ``""``) instead of raising, so one flaky
probe can never crash an eval run. An *unknown getter name*, by contrast, is a
task-JSON authoring error and raises :class:`GetterError` loudly via
:func:`get_getter`.

Getters are registered in :data:`GETTERS` so a new task only needs to name one —
no code change.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from minicua.browser.session import BrowserSession
from minicua.eval.errors import GetterError
from minicua.perception.screenshot import capture

logger = logging.getLogger("minicua.eval.getters")

# A getter is an async callable ``(session, **config) -> Any`` where ``config`` is
# the declarative options (selector / attribute / name / key / url ...).
Getter = Callable[..., Awaitable[Any]]


async def page_url(session: BrowserSession, **extra: Any) -> str | None:
    """Current URL of the active tab (``None`` if the page is gone)."""
    try:
        return await session.get_url()
    except Exception as exc:  # noqa: BLE001 - degrade to None, never raise
        logger.warning("page_url getter failed: %s", exc)
        return None


async def page_title(session: BrowserSession, **extra: Any) -> str | None:
    """Title of the active tab."""
    try:
        return await session.get_title()
    except Exception as exc:  # noqa: BLE001
        logger.warning("page_title getter failed: %s", exc)
        return None


async def page_text(session: BrowserSession, **extra: Any) -> str | None:
    """``document.body.innerText`` of the active tab (``None`` on failure)."""
    try:
        text = await session.page.evaluate("() => document.body ? document.body.innerText : ''")
        return text if isinstance(text, str) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("page_text getter failed: %s", exc)
        return None


async def element_exists(session: BrowserSession, *, selector: str | None = None, **extra: Any) -> bool:
    """Whether a CSS selector matches at least one element (``False`` on failure)."""
    if not selector:
        return False
    try:
        return await session.page.locator(selector).count() > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("element_exists(%r) failed: %s", selector, exc)
        return False


async def element_text(session: BrowserSession, *, selector: str | None = None, **extra: Any) -> str | None:
    """Trimmed inner text of the first element matching ``selector`` (``None`` if absent)."""
    if not selector:
        return None
    try:
        locator = session.page.locator(selector).first
        if await locator.count() == 0:
            return None
        return (await locator.inner_text()).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("element_text(%r) failed: %s", selector, exc)
        return None


async def element_attribute(
    session: BrowserSession,
    *,
    selector: str | None = None,
    attribute: str | None = None,
    **extra: Any,
) -> str | None:
    """A named attribute of the first element matching ``selector`` (``None`` if absent)."""
    if not selector or not attribute:
        return None
    try:
        locator = session.page.locator(selector).first
        if await locator.count() == 0:
            return None
        return await locator.get_attribute(attribute)
    except Exception as exc:  # noqa: BLE001
        logger.warning("element_attribute(%r, %r) failed: %s", selector, attribute, exc)
        return None


async def cookie_exists(
    session: BrowserSession,
    *,
    name: str | None = None,
    url: str | None = None,
    **extra: Any,
) -> bool:
    """Whether a cookie named ``name`` is present (optionally scoped to ``url``)."""
    if not name:
        return False
    try:
        cookies = await session.context.cookies(url) if url else await session.context.cookies()
        return any(c.get("name") == name for c in cookies)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cookie_exists(%r) failed: %s", name, exc)
        return False


async def local_storage(session: BrowserSession, *, key: str | None = None, **extra: Any) -> str | None:
    """A ``localStorage`` value by key (``None`` if absent or origin-forbidden)."""
    if not key:
        return None
    try:
        return await session.page.evaluate("(k) => window.localStorage.getItem(k)", key)
    except Exception as exc:  # noqa: BLE001 - data:/about:blank forbid localStorage
        logger.warning("local_storage(%r) failed: %s", key, exc)
        return None


async def screenshot(session: BrowserSession, **extra: Any) -> str | None:
    """Base64 PNG screenshot of the active tab (``None`` on failure)."""
    try:
        return await capture(session.page)
    except Exception as exc:  # noqa: BLE001
        logger.warning("screenshot getter failed: %s", exc)
        return None


#: name -> getter callable, the single declarative surface for tasks.
GETTERS: dict[str, Getter] = {
    "page_url": page_url,
    "page_title": page_title,
    "page_text": page_text,
    "element_exists": element_exists,
    "element_text": element_text,
    "element_attribute": element_attribute,
    "cookie_exists": cookie_exists,
    "local_storage": local_storage,
    "screenshot": screenshot,
}


def get_getter(name: str) -> Getter:
    """Return the getter for a declarative name, raising :class:`GetterError` if unknown."""
    try:
        return GETTERS[name]
    except KeyError:
        raise GetterError(f"unknown getter {name!r}") from None
