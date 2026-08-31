"""ChatRunner in desktop mode: reuse the Agent loop with a desktop environment."""

import pytest

from minicua.chat.runner import ChatRunner, _describe_action
from minicua.controller.agent import StopReason
from minicua.controller.llm import FakeModel
from minicua.desktop.actions import (
    DesktopAction,
    DesktopClickParams,
    DesktopShellParams,
)


class FakeDesktopEnv:
    def screenshot(self):
        return "c2NyZWVu"

    def screen_size(self):
        return (1920, 1080)

    def run_shell(self, command, *, timeout=None):
        from minicua.desktop.env import ShellResult

        return ShellResult(returncode=0, stdout="ok")

    def __getattr__(self, name):
        # Mouse / keyboard methods are never asserted here; no-op them.
        return lambda *a, **k: None


@pytest.mark.asyncio
async def test_chat_runner_desktop_mode_runs_to_done():
    model = FakeModel(
        responses=[
            {"name": "shell", "params": {"command": "echo hi"}},
            {"name": "done", "params": {"success": True, "submission": "did it"}},
        ]
    )
    runner = ChatRunner(model, mode="desktop")
    result = await runner.run("run a command", environment=FakeDesktopEnv())

    assert result.stop_reason == StopReason.DONE.value
    assert result.final_url == ""
    assert result.submission == "did it"
    assert len(result.actions) == 2
    assert result.actions[0].name == "shell"
    assert result.actions[0].success is True


def test_describe_desktop_click_uses_coordinates():
    action = DesktopAction(name="click", params=DesktopClickParams(x=11, y=22))
    label = _describe_action(action, None)
    assert "clicked at (11, 22)" in label


def test_describe_desktop_shell():
    action = DesktopAction(name="shell", params=DesktopShellParams(command="ls -la"))
    label = _describe_action(action, None)
    assert "ls -la" in label
