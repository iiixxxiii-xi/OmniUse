"""Desktop action space: pydantic models + executor with structured results."""

import pytest
from pydantic import ValidationError

from minicua.action.models import ActionError
from minicua.desktop.actions import (
    DESKTOP_ACTION_NAMES,
    DesktopAction,
    DesktopClickParams,
    DesktopDragParams,
    DesktopHotkeyParams,
    DesktopScrollParams,
    DesktopShellParams,
    DesktopTypeTextParams,
    execute_desktop,
    get_desktop_registry,
)
from minicua.desktop.env import ShellResult


class FakeEnv:
    """A desktop environment double that records calls and never touches hardware."""

    def __init__(self, *, shell=ShellResult(returncode=0, stdout="ok"), screen=(1920, 1080)):
        self.shell_result = shell
        self.screen = screen
        self.calls = []

    def screen_size(self):
        return self.screen

    def screenshot(self):
        return "c2NyZWVuc2hvdA=="

    def click(self, x, y):
        self.calls.append(("click", x, y))

    def move_to(self, x, y):
        self.calls.append(("move_to", x, y))

    def double_click(self, x, y):
        self.calls.append(("double_click", x, y))

    def right_click(self, x, y):
        self.calls.append(("right_click", x, y))

    def drag(self, x1, y1, x2, y2):
        self.calls.append(("drag", x1, y1, x2, y2))

    def type_text(self, text):
        self.calls.append(("type_text", text))

    def press(self, key):
        self.calls.append(("press", key))

    def hotkey(self, *keys):
        self.calls.append(("hotkey", keys))

    def scroll(self, amount):
        self.calls.append(("scroll", amount))

    def run_shell(self, command, *, timeout=None):
        self.calls.append(("shell", command))
        return self.shell_result


# --------------------------------------------------------------------------- #
# param models
# --------------------------------------------------------------------------- #


def test_click_params_reject_negative_coordinates():
    with pytest.raises(ValidationError):
        DesktopClickParams(x=-1, y=0)
    with pytest.raises(ValidationError):
        DesktopClickParams(x=0, y=-5)


def test_drag_params_validate_all_corners():
    with pytest.raises(ValidationError):
        DesktopDragParams(x1=-1, y1=0, x2=0, y2=0)


def test_type_text_requires_nonempty():
    with pytest.raises(ValidationError):
        DesktopTypeTextParams(text="")


def test_shell_requires_nonempty_command():
    with pytest.raises(ValidationError):
        DesktopShellParams(command="  ")


def test_hotkey_requires_at_least_one_key():
    with pytest.raises(ValidationError):
        DesktopHotkeyParams(keys=[])


def test_scroll_allows_negative_amount():
    assert DesktopScrollParams(amount=-3).amount == -3


def test_desktop_action_parses_params_from_dict():
    a = DesktopAction(name="click", params={"x": 5, "y": 6})
    assert isinstance(a.params, DesktopClickParams)
    assert a.params.x == 5


def test_desktop_action_rejects_mismatched_params():
    with pytest.raises(ValidationError):
        DesktopAction(name="click", params=DesktopTypeTextParams(text="x"))


def test_desktop_action_rejects_unknown_name():
    with pytest.raises(ValidationError):
        DesktopAction(name="fly", params=None)


def test_desktop_action_names_cover_the_full_space():
    assert DESKTOP_ACTION_NAMES == {
        "click", "move_to", "double_click", "right_click", "drag",
        "type_text", "press", "hotkey", "scroll", "shell", "done",
    }


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_desktop_registry_contains_all_actions():
    reg = get_desktop_registry()
    assert set(reg.names()) == set(DESKTOP_ACTION_NAMES)
    assert reg.get("click").func is not None


def test_desktop_registry_emits_openai_tools():
    reg = get_desktop_registry()
    names = [t["function"]["name"] for t in reg.to_tools()]
    assert "shell" in names
    assert "move_to" in names


# --------------------------------------------------------------------------- #
# executor
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_execute_click_dispatches_coordinates():
    env = FakeEnv()
    res = await execute_desktop(DesktopAction(name="click", params={"x": 100, "y": 200}), env)
    assert res.success is True
    assert env.calls == [("click", 100, 200)]
    assert res.metadata["x"] == 100 and res.metadata["y"] == 200


@pytest.mark.asyncio
async def test_execute_type_text_and_hotkey():
    env = FakeEnv()
    await execute_desktop(DesktopAction(name="type_text", params={"text": "hi"}), env)
    await execute_desktop(DesktopAction(name="hotkey", params={"keys": ["ctrl", "c"]}), env)
    assert ("type_text", "hi") in env.calls
    assert ("hotkey", ("ctrl", "c")) in env.calls


@pytest.mark.asyncio
async def test_execute_shell_success():
    env = FakeEnv(shell=ShellResult(returncode=0, stdout="done"))
    res = await execute_desktop(DesktopAction(name="shell", params={"command": "echo hi"}), env)
    assert res.success is True
    assert res.metadata["returncode"] == 0


@pytest.mark.asyncio
async def test_execute_shell_surfaces_stdout_in_extracted():
    env = FakeEnv(shell=ShellResult(returncode=0, stdout="poster_party_night.webp\n"))
    res = await execute_desktop(DesktopAction(name="shell", params={"command": "ls"}), env)
    assert res.success is True
    # The model must see the command's stdout in the observation, not just
    # "Ran command (exit 0)" — otherwise shell-driven tasks go blind and loop.
    assert "poster_party_night.webp" in (res.extracted or "")


@pytest.mark.asyncio
async def test_execute_shell_nonzero_returns_shell_failed():
    env = FakeEnv(shell=ShellResult(returncode=2, stderr="nope"))
    res = await execute_desktop(DesktopAction(name="shell", params={"command": "false"}), env)
    assert res.success is False
    assert res.error_code == ActionError.SHELL_FAILED


@pytest.mark.asyncio
async def test_execute_shell_timeout_is_retryable():
    env = FakeEnv(shell=ShellResult(returncode=-1, timed_out=True))
    res = await execute_desktop(DesktopAction(name="shell", params={"command": "sleep"}), env)
    assert res.success is False
    assert res.error_code == ActionError.SHELL_TIMEOUT
    assert res.retryable is True


@pytest.mark.asyncio
async def test_execute_input_failure_is_structured():
    env = FakeEnv()

    def boom(x, y):
        raise OSError("no mouse")

    env.click = boom
    res = await execute_desktop(DesktopAction(name="click", params={"x": 0, "y": 0}), env)
    assert res.success is False
    assert res.error_code == ActionError.INPUT_FAILED


@pytest.mark.asyncio
async def test_execute_done_mirrors_params():
    env = FakeEnv()
    res = await execute_desktop(DesktopAction(name="done", params={"success": True, "submission": "x"}), env)
    assert res.success is True
    assert res.extracted == "x"


@pytest.mark.asyncio
async def test_execute_unknown_action_returns_error():
    env = FakeEnv()
    fake = DesktopAction.model_construct(name="bogus", params=None)
    res = await execute_desktop(fake, env)
    assert res.success is False
    assert res.error_code == ActionError.UNKNOWN_ACTION
