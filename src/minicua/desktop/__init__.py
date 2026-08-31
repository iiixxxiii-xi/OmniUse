"""Desktop mode: operate the whole machine (mouse / keyboard / shell + screenshots).

This package is the desktop twin of the browser stack — :class:`DesktopEnvironment`
(the OS-level session), :func:`extract_desktop_state` (screenshot perception), and
a coordinate-grounded desktop action space (:func:`execute_desktop` +
:func:`get_desktop_registry`). It reuses the browser layer's :class:`ActionResult`,
:class:`ActionRegistry`, and the :class:`~minicua.controller.agent.Agent` loop.
"""

from minicua.desktop.actions import (
    DESKTOP_ACTION_NAMES,
    DESKTOP_PARAM_MODELS,
    DesktopAction,
    DesktopClickParams,
    DesktopDoubleClickParams,
    DesktopDragParams,
    DesktopHotkeyParams,
    DesktopMoveParams,
    DesktopPressParams,
    DesktopRightClickParams,
    DesktopScrollParams,
    DesktopShellParams,
    DesktopTypeTextParams,
    desktop_param_model,
    execute_desktop,
    get_desktop_registry,
)
from minicua.desktop.env import DesktopEnvironment, ShellResult
from minicua.desktop.perception import DesktopState, extract_desktop_state

__all__ = [
    "DESKTOP_ACTION_NAMES",
    "DESKTOP_PARAM_MODELS",
    "DesktopAction",
    "DesktopClickParams",
    "DesktopDoubleClickParams",
    "DesktopDragParams",
    "DesktopEnvironment",
    "DesktopHotkeyParams",
    "DesktopMoveParams",
    "DesktopPressParams",
    "DesktopRightClickParams",
    "DesktopScrollParams",
    "DesktopShellParams",
    "DesktopState",
    "DesktopTypeTextParams",
    "ShellResult",
    "desktop_param_model",
    "execute_desktop",
    "extract_desktop_state",
    "get_desktop_registry",
]
