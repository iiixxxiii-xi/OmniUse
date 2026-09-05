"""SSH-driven VM desktop environment (for OSWorld-style evaluation).

``SSHVmEnvironment`` implements the same surface as
:class:`~minicua.desktop.env.DesktopEnvironment` — ``screenshot``, ``screen_size``,
mouse, keyboard, scroll and shell — but every call is forwarded over SSH to a
short ``pyautogui`` snippet running *inside* the guest with ``DISPLAY`` set. The
agent loop can therefore drive a headless VM exactly as it drives a local
desktop, with no other changes.

The SSH client is a persistent :class:`paramiko.SSHClient` (reused across calls)
so a task's many small actions don't pay a fresh handshake each time.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from typing import Any

logger = logging.getLogger("minicua.desktop.ssh_vm")

#: A subprocess.run-compatible callable (injectable for tests).
SubprocessRunner = Any


class SSHVmEnvironment:
    """Control a remote VM's desktop over SSH (pyautogui runs inside the guest)."""

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        *,
        port: int = 22,
        display: str = ":0",
        screen_size: tuple[int, int] = (1920, 1080),
        connect_timeout: float = 15.0,
    ) -> None:
        self._host = host
        self._user = user
        self._password = password
        self._port = port
        self._display = display
        self._screen_size = screen_size
        self._real_screen_size: tuple[int, int] | None = None
        self._connect_timeout = connect_timeout
        self._client = self._connect()
        self._display = self._detect_display()

    def _connect(self):
        import paramiko

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            self._host,
            port=self._port,
            username=self._user,
            password=self._password,
            timeout=self._connect_timeout,
        )
        return client

    def _detect_display(self) -> str:
        """Return a reachable X display, probing common candidates.

        The guest's display number shifts across reboots (observed :0 -> :1 after
        a gdm re-login), so a hardcoded ``DISPLAY`` goes stale and every pyautogui
        call fails with "Can't connect to display". Probe the caller's hint first,
        then :0..:4, and fall back to the hint if none respond.
        """
        candidates = [self._display] if self._display != "auto" else []
        candidates += [f":{d}" for d in range(5) if f":{d}" not in candidates]
        for disp in candidates:
            _, out, err = self._client.exec_command(
                f"DISPLAY={disp} python3 -c 'import pyautogui' 2>/dev/null"
            )
            out.read()
            err.read()
            if out.channel.recv_exit_status() == 0:
                if disp != self._display:
                    logger.info("X display %r unreachable; using %r", self._display, disp)
                return disp
        return self._display

    def _run(self, code: str) -> str:
        """Run ``code`` inside the guest's Python (DISPLAY set); return stdout."""
        cmd = f"DISPLAY={self._display} python3 -c {shlex.quote(code)}"
        try:
            _, out, _ = self._client.exec_command(cmd)
            return out.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001 - a dead SSH channel is recoverable
            logger.warning("SSH command failed (%s); reconnecting", exc)
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = self._connect()
            _, out, _ = self._client.exec_command(cmd)
            return out.read().decode("utf-8", "replace")

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass

    # -- screenshot ---------------------------------------------------------

    def screenshot(self) -> str | None:
        """Return the guest screen as base64 PNG, or ``None`` on failure."""
        code = (
            "import pyautogui, base64, io;"
            "img=pyautogui.screenshot();"
            "buf=io.BytesIO();img.save(buf,'PNG');"
            "print(base64.b64encode(buf.getvalue()).decode())"
        )
        out = self._run(code)
        return out.strip() or None

    def screen_size(self) -> tuple[int, int]:
        """Return the guest's *real* screen size, queried over SSH (cached).

        A headless VM's native resolution often differs from the constructor hint
        (e.g. this VM is 730x624, not 1920x1080), and a hardcoded size makes every
        coordinate the model predicts drift off-target. Query pyautogui once and
        cache it; fall back to the hint only if the query fails.
        """
        if self._real_screen_size is not None:
            return self._real_screen_size
        out = self._run("import pyautogui;s=pyautogui.size();print(s.width,s.height)")
        try:
            w, h = out.strip().split()
            self._real_screen_size = (int(w), int(h))
        except (ValueError, AttributeError):
            logger.warning(
                "failed to query guest screen size (%r); using hint %s",
                out,
                self._screen_size,
            )
            self._real_screen_size = self._screen_size
        return self._real_screen_size

    # -- mouse --------------------------------------------------------------

    def click(self, x: int, y: int) -> None:
        self._run(f"import pyautogui;pyautogui.click({x},{y})")

    def move_to(self, x: int, y: int) -> None:
        self._run(f"import pyautogui;pyautogui.moveTo({x},{y})")

    def double_click(self, x: int, y: int) -> None:
        self._run(f"import pyautogui;pyautogui.doubleClick({x},{y})")

    def right_click(self, x: int, y: int) -> None:
        self._run(f"import pyautogui;pyautogui.rightClick({x},{y})")

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self._run(f"import pyautogui;pyautogui.moveTo({x1},{y1});pyautogui.dragTo({x2},{y2},button='left')")

    # -- keyboard -----------------------------------------------------------

    def type_text(self, text: str) -> None:
        self._run(f"import pyautogui;pyautogui.typewrite({text!r})")

    def press(self, key: str) -> None:
        self._run(f"import pyautogui;pyautogui.press({key!r})")

    def hotkey(self, *keys: str) -> None:
        self._run(f"import pyautogui;pyautogui.hotkey({', '.join(repr(k) for k in keys)})")

    def scroll(self, amount: int) -> None:
        # positive amount scrolls down (matches DesktopEnvironment semantics)
        self._run(f"import pyautogui;pyautogui.scroll({-amount})")

    # -- shell --------------------------------------------------------------

    def run_shell(self, command: str, *, timeout: float | None = 30.0) -> Any:
        """Run a command over SSH and return a small structured result."""
        from minicua.desktop.env import ShellResult

        try:
            _, out, err = self._client.exec_command(command, timeout=timeout)
            return ShellResult(
                returncode=out.channel.recv_exit_status(),
                stdout=out.read().decode("utf-8", "replace"),
                stderr=err.read().decode("utf-8", "replace"),
            )
        except Exception as exc:  # noqa: BLE001 - shell failures are data
            return ShellResult(returncode=-1, stderr=f"{type(exc).__name__}: {exc}")
