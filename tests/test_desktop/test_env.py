"""DesktopEnvironment: PyAutoGUI/mss/shell facade, dependency-injected for tests."""

import base64
import io
import subprocess

from PIL import Image

from minicua.desktop.env import DesktopEnvironment, ShellResult


class FakeController:
    """A pyautogui-like object that records calls instead of touching the mouse."""

    def __init__(self, size=(1920, 1080)):
        self._size = size
        self.calls = []

    def size(self):
        return self._size

    def click(self, x, y):
        self.calls.append(("click", x, y))

    def moveTo(self, x, y):
        self.calls.append(("moveTo", x, y))

    def doubleClick(self, x, y):
        self.calls.append(("doubleClick", x, y))

    def rightClick(self, x, y):
        self.calls.append(("rightClick", x, y))

    def dragTo(self, x, y, **kwargs):
        self.calls.append(("dragTo", x, y))

    def typewrite(self, text):
        self.calls.append(("typewrite", text))

    def press(self, key):
        self.calls.append(("press", key))

    def hotkey(self, *keys):
        self.calls.append(("hotkey", keys))

    def scroll(self, amount):
        self.calls.append(("scroll", amount))


# --------------------------------------------------------------------------- #
# screenshot
# --------------------------------------------------------------------------- #


def test_screenshot_returns_base64_png():
    env = DesktopEnvironment(screenshot_fn=lambda: Image.new("RGB", (2, 2), "red"))
    b64 = env.screenshot()
    assert b64 is not None
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_screenshot_returns_none_when_grab_fails():
    def boom():
        raise RuntimeError("no screen")

    env = DesktopEnvironment(screenshot_fn=boom)
    assert env.screenshot() is None


def test_screenshot_returns_none_when_grab_returns_none():
    env = DesktopEnvironment(screenshot_fn=lambda: None)
    assert env.screenshot() is None


# --------------------------------------------------------------------------- #
# screen size + mouse / keyboard delegation
# --------------------------------------------------------------------------- #


def test_screen_size_delegates_to_controller():
    env = DesktopEnvironment(controller=FakeController(size=(800, 600)))
    assert env.screen_size() == (800, 600)


def test_mouse_actions_delegate_coordinates():
    ctrl = FakeController(size=(800, 600))  # small screen → no downscale, coords pass through
    env = DesktopEnvironment(controller=ctrl)
    env.click(10, 20)
    env.move_to(30, 40)
    env.double_click(50, 60)
    env.right_click(70, 80)
    assert ctrl.calls == [
        ("click", 10, 20),
        ("moveTo", 30, 40),
        ("doubleClick", 50, 60),
        ("rightClick", 70, 80),
    ]


def test_drag_moves_then_drags_to():
    ctrl = FakeController(size=(800, 600))  # small screen → no downscale
    env = DesktopEnvironment(controller=ctrl)
    env.drag(1, 2, 3, 4)
    assert ctrl.calls == [("moveTo", 1, 2), ("dragTo", 3, 4)]


def test_mouse_actions_scale_coordinates_back_to_native():
    ctrl = FakeController(size=(2560, 1600))
    env = DesktopEnvironment(controller=ctrl)
    env.click(640, 400)  # model-space (1280-wide screenshot) → native ×2
    assert ctrl.calls == [("click", 1280, 800)]
    assert env.screen_size() == (1280, 800)


def test_keyboard_actions_delegate():
    ctrl = FakeController()
    env = DesktopEnvironment(controller=ctrl)
    env.type_text("hello")
    env.press("enter")
    env.hotkey("ctrl", "c")
    assert ctrl.calls == [
        ("typewrite", "hello"),
        ("press", "enter"),
        ("hotkey", ("ctrl", "c")),
    ]


def test_scroll_positive_amount_scrolls_down():
    ctrl = FakeController()
    env = DesktopEnvironment(controller=ctrl)
    env.scroll(3)
    assert ctrl.calls == [("scroll", -3)]


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #


def test_run_shell_success_returns_structured_result():
    def runner(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="hello\n", stderr="")

    r = DesktopEnvironment(runner=runner).run_shell("echo hello")
    assert r.returncode == 0
    assert r.stdout == "hello\n"
    assert r.stderr == ""
    assert r.timed_out is False


def test_run_shell_nonzero_exit_is_data_not_exception():
    def runner(*args, **kwargs):
        return subprocess.CompletedProcess(args, 3, stdout="", stderr="oops")

    r = DesktopEnvironment(runner=runner).run_shell("false")
    assert r.returncode == 3
    assert r.stderr == "oops"


def test_run_shell_timeout_is_structured():
    def runner(*args, **kwargs):
        raise subprocess.TimeoutExpired("cmd", 1.0)

    r = DesktopEnvironment(runner=runner).run_shell("sleep 5")
    assert r.timed_out is True
    assert r.returncode == -1
    assert "timed out" in r.stderr


def test_run_shell_uses_utf8_replace_and_timeout():
    captured = {}

    def runner(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    DesktopEnvironment(runner=runner, shell_timeout=5.0).run_shell("x")
    assert captured.get("shell") is True
    assert captured.get("encoding") == "utf-8"
    assert captured.get("errors") == "replace"
    assert captured.get("timeout") == 5.0


def test_run_shell_oserror_is_structured():
    def runner(*args, **kwargs):
        raise OSError("no such file")

    r = DesktopEnvironment(runner=runner).run_shell("not-a-command")
    assert r.returncode == -1
    assert "no such file" in r.stderr


def test_shell_result_model():
    r = ShellResult(returncode=0, stdout="out", stderr="", timed_out=False)
    assert r.returncode == 0
