"""Desktop environment: the OS-level counterpart to :class:`BrowserSession`.

``DesktopEnvironment`` wraps the real desktop input/screenshot stack — PyAutoGUI
(mouse + keyboard) and ``mss`` (fast full-screen grab, with a PyAutoGUI/PIL
fallback) — plus a ``subprocess``-backed shell. It is deliberately a *thin,
synchronous* facade: every method either delegates to the injected backend or
returns a structured value, and nothing here raises an unhandled exception to the
agent loop (screenshots degrade to ``None``; shell failures are data).

The real PyAutoGUI / ``mss`` backends are imported lazily inside methods, so a
test or a browser-only run never pulls those dependencies until a desktop action
actually executes. Every backend is also injectable (``controller`` / ``screen`` /
``screenshot_fn`` / ``runner``) so the whole layer is unit-testable on a
headless machine.
"""

import base64
import io
import logging
import subprocess
from typing import Any, Callable

from pydantic import BaseModel, Field

logger = logging.getLogger("minicua.desktop.env")

#: A subprocess.run-compatible callable (injectable for tests).
SubprocessRunner = Callable[..., Any]


class ShellResult(BaseModel):
    """Structured outcome of a shell command (never raised as an exception)."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class DesktopEnvironment:
    """Control the whole machine: screenshot, mouse, keyboard, scroll, and shell.

    Parameters
    ----------
    controller:
        A pyautogui-like object (``click``/``moveTo``/``doubleClick``/``rightClick``/
        ``dragTo``/``typewrite``/``press``/``hotkey``/``scroll``/``size``). Defaults
        to the real ``pyautogui`` module (imported lazily).
    screenshot_fn:
        A ``() -> PIL.Image | None`` callable used for screenshots. Defaults to the
        real ``mss`` + PIL capture (with a pyautogui fallback).
    runner:
        A ``subprocess.run``-compatible callable for ``run_shell``. Defaults to the
        real ``subprocess.run``.
    shell_timeout:
        Default per-command timeout (seconds) for ``run_shell``.
    """

    #: Native screenshots wider than this are downscaled before reaching the
    #: vision model (desktop icons are tiny at e.g. 2560x1600); mouse coordinates
    #: are scaled back up before execution so clicks land on the right pixel.
    _SCREENSHOT_MAX_WIDTH = 1280

    def __init__(
        self,
        *,
        controller: Any = None,
        screenshot_fn: Callable[[], Any] | None = None,
        runner: SubprocessRunner | None = None,
        shell_timeout: float = 30.0,
    ) -> None:
        self._controller = controller
        self._screenshot_fn = screenshot_fn
        self._runner = runner
        self.shell_timeout = shell_timeout
        self._scale: float | None = None

    def _scale_factor(self) -> float:
        """Downscale factor between native pixels and the model-facing screenshot."""
        if self._scale is None:
            w, _ = self._get_controller().size()
            self._scale = (w / self._SCREENSHOT_MAX_WIDTH) if w > self._SCREENSHOT_MAX_WIDTH else 1.0
        return self._scale

    def scale_factor(self) -> float:
        """Public accessor for the native→model screenshot downscale factor."""
        return self._scale_factor()

    def _to_native(self, x: int, y: int) -> tuple[int, int]:
        """Scale model-space coordinates back up to native screen pixels."""
        s = self._scale_factor()
        return int(x * s), int(y * s)

    # -- backend accessors --------------------------------------------------

    def _get_controller(self) -> Any:
        if self._controller is None:
            import pyautogui

            self._controller = pyautogui
        return self._controller

    # -- screenshot ---------------------------------------------------------

    def screenshot(self) -> str | None:
        """Capture the full screen as base64-encoded PNG, or ``None`` on failure.

        Never raises: any grab or encode failure is logged and degrades to ``None``
        so perception can continue without a screenshot.
        """
        try:
            img = self._screenshot_fn() if self._screenshot_fn is not None else self._capture_default()
        except Exception as exc:  # noqa: BLE001 - degrade rather than crash the loop
            logger.warning("desktop screenshot grab failed: %s", exc)
            return None
        if img is None:
            return None
        try:
            s = self._scale_factor()
            if s > 1.0:
                from PIL import Image

                w, h = img.size
                img = img.resize((int(w / s), int(h / s)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:  # noqa: BLE001
            logger.warning("desktop screenshot encode failed: %s", exc)
            return None

    def _capture_default(self) -> Any:
        """Grab the primary monitor with ``mss`` (fast), falling back to pyautogui."""
        try:
            import mss
            from PIL import Image

            with mss.mss() as sct:
                monitor = sct.monitors[1]  # primary monitor (mss convention)
                shot = sct.grab(monitor)
                return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        except Exception:  # noqa: BLE001 - fall back to pyautogui's own grab
            return self._get_controller().screenshot()

    def screen_size(self) -> tuple[int, int]:
        """Return the model-facing screen size as ``(width, height)`` (downscaled)."""
        w, h = self._get_controller().size()
        s = self._scale_factor()
        return int(w / s), int(h / s)

    # -- mouse --------------------------------------------------------------

    def click(self, x: int, y: int) -> None:
        nx, ny = self._to_native(x, y)
        self._get_controller().click(nx, ny)

    def move_to(self, x: int, y: int) -> None:
        nx, ny = self._to_native(x, y)
        self._get_controller().moveTo(nx, ny)

    def double_click(self, x: int, y: int) -> None:
        nx, ny = self._to_native(x, y)
        self._get_controller().doubleClick(nx, ny)

    def right_click(self, x: int, y: int) -> None:
        nx, ny = self._to_native(x, y)
        self._get_controller().rightClick(nx, ny)

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> None:
        nx1, ny1 = self._to_native(x1, y1)
        nx2, ny2 = self._to_native(x2, y2)
        ctrl = self._get_controller()
        ctrl.moveTo(nx1, ny1)
        ctrl.dragTo(nx2, ny2, button="left")

    # -- keyboard -----------------------------------------------------------

    def type_text(self, text: str) -> None:
        if text.isascii():
            self._get_controller().typewrite(text)
            return
        # pyautogui.typewrite only types ASCII; for unicode (Chinese) copy the
        # text to the clipboard and paste via Ctrl+V so non-ASCII lands correctly.
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value $args[0]", text],
            capture_output=True,
            timeout=15,
        )
        self._get_controller().hotkey("ctrl", "v")

    def press(self, key: str) -> None:
        self._get_controller().press(key)

    def hotkey(self, *keys: str) -> None:
        self._get_controller().hotkey(*keys)

    def scroll(self, amount: int) -> None:
        """Scroll the mouse wheel; positive ``amount`` scrolls down."""
        self._get_controller().scroll(-amount)

    # -- shell --------------------------------------------------------------

    def run_shell(self, command: str, *, timeout: float | None = None) -> ShellResult:
        """Run ``command`` in a shell and return a structured :class:`ShellResult`.

        Encoding is forced to UTF-8 with ``errors="replace"`` so non-decodable bytes
        never raise; a timeout produces ``timed_out=True`` rather than an exception.
        """
        runner = self._runner if self._runner is not None else subprocess.run
        effective_timeout = self.shell_timeout if timeout is None else timeout
        try:
            proc = runner(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = (exc.stderr or "") + f"\n[command timed out after {effective_timeout}s]"
            return ShellResult(
                returncode=-1,
                stdout=exc.stdout or "",
                stderr=stderr,
                timed_out=True,
            )
        except Exception as exc:  # noqa: BLE001 - shell failures are data, never exceptions
            return ShellResult(returncode=-1, stderr=f"{type(exc).__name__}: {exc}")
        return ShellResult(
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            timed_out=False,
        )
