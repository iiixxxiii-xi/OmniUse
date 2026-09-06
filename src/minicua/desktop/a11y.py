"""Windows accessibility-tree extraction for desktop perception.

Enumerates the desktop icons and top-level windows via UIA (``pywinauto``) and
linearizes them into a compact ``name @ center-position`` list so the agent can
target UI elements **by name** instead of reading tiny labels from a screenshot —
mirroring OSWorld's ``a11y_tree`` observation.

Positions are divided by ``scale`` so they live in the same coordinate space as
the (downscaled) screenshot the model sees; the harness scales clicks back up.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("minicua.desktop.a11y")


def extract_a11y_tree(scale: float = 1.0, max_items: int = 80) -> str:
    """Return a linearized accessibility tree (``- [kind] name @ (x, y)``).

    ``scale`` divides native pixel coordinates to match a downscaled screenshot;
    ``max_items`` bounds the list to keep the token budget sane. Returns ``""``
    when UIA is unavailable or enumeration fails (never raises).
    """
    try:
        from pywinauto import Desktop
    except Exception as exc:  # pragma: no cover - UIA is Windows-only
        logger.warning("pywinauto unavailable: %s", exc)
        return ""

    try:
        desktop = Desktop(backend="uia")
        windows = list(desktop.windows())
    except Exception as exc:
        logger.warning("desktop UIA enumeration failed: %s", exc)
        return ""

    lines: list[str] = []

    # Desktop icons (Progman → SysListView32 → items) — the primary "open app"
    # target. Each item's center is its clickable point.
    for w in windows:
        if w.class_name() != "Progman":
            continue
        try:
            for child in w.children():
                if child.class_name() != "SysListView32":
                    continue
                for item in child.children():
                    name = _text(item)
                    if not name:
                        continue
                    center = _center(item, scale)
                    if center is None:
                        continue
                    lines.append(f"- [icon] {name} @ {center}")
        except Exception as exc:  # noqa: BLE001 - best-effort enumeration
            logger.debug("desktop icon enumeration failed: %s", exc)
        break

    # Top-level windows with a visible title — for switching to / verifying an app.
    for w in windows:
        title = _text(w)
        if not title or title == "Program Manager":
            continue
        center = _center(w, scale)
        if center is None:
            continue
        lines.append(f"- [window] {title} @ {center}")

    if len(lines) > max_items:
        lines = lines[:max_items]
    return "\n".join(lines)


def _text(element) -> str:
    try:
        return (element.window_text() or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _center(element, scale: float) -> tuple[int, int] | None:
    try:
        r = element.rectangle()
    except Exception:  # noqa: BLE001
        return None
    if r.right - r.left <= 0 or r.bottom - r.top <= 0:
        return None
    s = scale if scale > 0 else 1.0
    return int((r.left + r.right) / 2 / s), int((r.top + r.bottom) / 2 / s)
