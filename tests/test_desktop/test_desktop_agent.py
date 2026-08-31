"""Agent desktop mode: reuse the perceive→think→act loop with a desktop backend."""

import pytest

from minicua.controller.agent import Agent
from minicua.controller.llm import FakeModel, ImageBlock, TextBlock


class FakeDesktopEnv:
    """A desktop environment double for the agent loop (no real mouse/screen)."""

    def __init__(self, *, screenshot="aGVsbG8=", size=(1920, 1080)):
        self._screenshot = screenshot
        self._size = size
        self.calls = []

    def screenshot(self):
        return self._screenshot

    def screen_size(self):
        return self._size

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
        from minicua.desktop.env import ShellResult

        return ShellResult(returncode=0, stdout="ok")


@pytest.mark.asyncio
async def test_desktop_agent_runs_to_done():
    env = FakeDesktopEnv()
    model = FakeModel(
        responses=[
            {"name": "shell", "params": {"command": "echo hi"}},
            {"name": "click", "params": {"x": 10, "y": 20}},
            {"name": "done", "params": {"success": True, "submission": "done"}},
        ]
    )
    agent = Agent(mode="desktop", environment=env, model=model, max_steps=10)
    result = await agent.run(task="do something on the desktop")

    assert result.done is True
    assert result.success is True
    assert result.submission == "done"
    assert result.steps == 3
    assert ("shell", "echo hi") in env.calls
    assert ("click", 10, 20) in env.calls


@pytest.mark.asyncio
async def test_desktop_agent_does_not_require_browser_session():
    model = FakeModel(responses=[{"name": "done", "params": {"success": True}}])
    agent = Agent(mode="desktop", environment=FakeDesktopEnv(), model=model)
    result = await agent.run(task="x")
    assert result.done is True
    assert result.success is True


@pytest.mark.asyncio
async def test_desktop_agent_passes_screenshot_as_image_block():
    env = FakeDesktopEnv(screenshot="c2NyZWVu")
    model = FakeModel(responses=[{"name": "done", "params": {"success": True}}])
    agent = Agent(mode="desktop", environment=env, model=model)
    await agent.run(task="x")

    state_messages = [m for m in model.calls[0][0] if m.role == "user"]
    content = state_messages[0].content
    assert isinstance(content, list)
    assert any(isinstance(b, ImageBlock) and b.image_base64 == "c2NyZWVu" for b in content)
    assert any(isinstance(b, TextBlock) and "Screen" in b.text for b in content)


@pytest.mark.asyncio
async def test_desktop_agent_uses_desktop_tool_schema():
    env = FakeDesktopEnv()
    model = FakeModel(responses=[{"name": "done", "params": {"success": True}}])
    agent = Agent(mode="desktop", environment=env, model=model)
    await agent.run(task="x")

    tools = model.calls[0][1]
    names = [t["function"]["name"] for t in tools]
    assert "shell" in names
    assert "move_to" in names
    assert "drag" in names
    # Browser-only actions must not leak into the desktop tool set.
    assert "navigate" not in names
    assert "type" not in names


def test_desktop_system_prompt_mentions_desktop_and_coordinates():
    agent = Agent(mode="desktop", environment=FakeDesktopEnv(), model=FakeModel())
    prompt = agent._system_prompt()
    assert "desktop" in prompt.lower()
    assert "coordinate" in prompt.lower()


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        Agent(mode="tablet", model=FakeModel())
