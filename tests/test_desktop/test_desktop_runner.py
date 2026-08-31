"""Desktop task runner: run an instruction to completion with a desktop backend."""

import pytest

from minicua.controller.llm import FakeModel
from minicua.desktop.runner import DesktopRunResult, run_desktop


class FakeDesktopEnv:
    def screenshot(self):
        return "c2NyZWVu"

    def screen_size(self):
        return (1920, 1080)

    def run_shell(self, command, *, timeout=None):
        from minicua.desktop.env import ShellResult

        return ShellResult(returncode=0, stdout="ok")

    def __getattr__(self, name):
        return lambda *a, **k: None


@pytest.mark.asyncio
async def test_run_desktop_returns_structured_result():
    model = FakeModel(responses=[{"name": "done", "params": {"success": True, "submission": "answer"}}])
    result = await run_desktop("finish the task", model, environment=FakeDesktopEnv())
    assert isinstance(result, DesktopRunResult)
    assert result.success is True
    assert result.submission == "answer"
    assert result.stop_reason == "done"


@pytest.mark.asyncio
async def test_run_desktop_reflects_failure():
    model = FakeModel(responses=[{"name": "done", "params": {"success": False}}])
    result = await run_desktop("impossible", model, environment=FakeDesktopEnv())
    assert result.success is False
