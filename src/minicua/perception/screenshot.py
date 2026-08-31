"""Screenshot capture + the ``use_vision`` three-state policy.

Vision is an auxiliary signal: DOM remains the primary perception channel, so a
text-only model can operate with ``use_vision="dom_only"``. The policy is a pure
function so it can be unit-tested in isolation:

* ``dom_only`` — never capture (text-only models).
* ``vision``    — always capture.
* ``auto``      — capture iff the model supports vision.

``capture`` never raises: on timeout or any screenshot error it logs a warning
and returns ``None``, letting the caller degrade to ``dom_only`` gracefully.
"""

import base64
import logging
from typing import Literal

from playwright.async_api import Page

logger = logging.getLogger("minicua.perception.screenshot")

VisionMode = Literal["dom_only", "vision", "auto"]

#: Default screenshot timeout in milliseconds.
DEFAULT_SCREENSHOT_TIMEOUT_MS = 5_000


def should_capture(mode: str, model_supports_vision: bool) -> bool:
    """Decide whether to capture a screenshot for the given vision mode."""
    if mode == "dom_only":
        return False
    if mode == "vision":
        return True
    if mode == "auto":
        return bool(model_supports_vision)
    raise ValueError(f"unknown vision mode: {mode!r}")


async def capture(page: Page, *, timeout_ms: int = DEFAULT_SCREENSHOT_TIMEOUT_MS) -> str | None:
    """Capture a viewport screenshot as base64 PNG, or ``None`` on failure."""
    try:
        data = await page.screenshot(type="png", timeout=timeout_ms)
    except Exception as exc:  # noqa: BLE001 - degrade to dom_only on any failure
        logger.warning("screenshot capture failed: %s", exc)
        return None
    return base64.b64encode(data).decode("ascii")
