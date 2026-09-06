"""Desktop perception: a screenshot-only state snapshot.

Desktop mode has **no DOM**, so the only perception signal is a full-screen
screenshot handed to a vision model (``qwen3-vl-flash``) as an image block. This
module is the desktop twin of :mod:`minicua.perception.extract`: it builds a
small, defensive :class:`DesktopState` from a :class:`~minicua.desktop.env.DesktopEnvironment`.

Perception never raises: a failed screenshot degrades to ``None`` and a failed
screen-size read degrades to ``(0, 0)``, so the agent loop always has a state to
hand the model.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from minicua.desktop.a11y import extract_a11y_tree

logger = logging.getLogger("minicua.desktop.perception")


class DesktopState(BaseModel):
    """The complete desktop perception snapshot for one agent step."""

    screenshot: str | None = None  # base64-encoded PNG, or None when unavailable
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    a11y_tree: str = ""  # linearized accessibility tree (name → position)


def extract_desktop_state(env: Any) -> DesktopState:
    """Build a :class:`DesktopState` from ``env`` (screenshot + screen size).

    Both reads are defensive: a screenshot grab failure yields ``screenshot=None``
    and a size failure yields ``(0, 0)``, so this never raises.
    """
    screenshot = env.screenshot()
    if screenshot is not None and not isinstance(screenshot, str):
        logger.warning("desktop screenshot was not a string; treating as None")
        screenshot = None

    width, height = 0, 0
    try:
        size = env.screen_size()
        if isinstance(size, (tuple, list)) and len(size) == 2:
            width, height = int(size[0]), int(size[1])
    except Exception as exc:  # noqa: BLE001 - degrade rather than crash the loop
        logger.warning("desktop screen size read failed: %s", exc)

    a11y_tree = ""
    try:
        scale = env.scale_factor() if hasattr(env, "scale_factor") else 1.0
        a11y_tree = extract_a11y_tree(scale=scale)
    except Exception as exc:  # noqa: BLE001 - degrade rather than crash the loop
        logger.warning("desktop a11y tree extraction failed: %s", exc)

    return DesktopState(screenshot=screenshot, width=width, height=height, a11y_tree=a11y_tree)
