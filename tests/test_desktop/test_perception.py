"""Desktop perception: screenshot-only state (no DOM)."""

from minicua.desktop.perception import DesktopState, extract_desktop_state


class FakeEnv:
    def __init__(self, *, screenshot="c2NyZWVu", size=(1280, 720)):
        self._screenshot = screenshot
        self._size = size

    def screenshot(self):
        return self._screenshot

    def screen_size(self):
        return self._size


def test_extract_desktop_state_returns_screenshot_and_size():
    state = extract_desktop_state(FakeEnv(screenshot="aGVsbG8=", size=(1024, 768)))
    assert isinstance(state, DesktopState)
    assert state.screenshot == "aGVsbG8="
    assert state.width == 1024
    assert state.height == 768


def test_extract_desktop_state_degrades_when_screenshot_is_none():
    state = extract_desktop_state(FakeEnv(screenshot=None, size=(1024, 768)))
    assert state.screenshot is None
    assert state.width == 1024


def test_extract_desktop_state_degrades_when_size_fails():
    class Broken(FakeEnv):
        def screen_size(self):
            raise RuntimeError("no display")

    state = extract_desktop_state(Broken(screenshot="aGVsbG8="))
    assert state.screenshot == "aGVsbG8="
    assert state.width == 0
    assert state.height == 0
