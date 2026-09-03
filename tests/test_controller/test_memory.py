"""Task-level memory wired into the agent: remember writes, recall injects."""

import pytest

from minicua.controller.agent import Agent
from minicua.controller.llm import FakeModel
from minicua.state.memory import TaskMemory


@pytest.mark.asyncio
async def test_agent_remember_writes_to_memory(session, tmp_path):
    await session.page.set_content("<button id=b>go</button>")
    memory = TaskMemory(tmp_path / "mem.json")
    model = FakeModel(
        responses=[
            {"name": "remember", "params": {"text": "login button is at bottom"}},
            {"name": "done", "params": {"success": True}},
        ]
    )
    agent = Agent(session=session, model=model, memory=memory)
    result = await agent.run(task="x")
    assert result.done is True
    assert [f.text for f in memory.recall()] == ["login button is at bottom"]


@pytest.mark.asyncio
async def test_agent_injects_prior_memory_into_context(session):
    await session.page.set_content("<button id=b>go</button>")
    memory = TaskMemory()
    memory.remember("the submit button is disabled until checked")
    model = FakeModel(responses=[{"name": "done", "params": {"success": True}}])
    agent = Agent(session=session, model=model, memory=memory)
    await agent.run(task="x")

    # The memory fact must have been injected into the first model call's messages.
    first_call = model.calls[0][0]
    joined = "\n".join(str(m.content) for m in first_call if m.role == "user")
    assert "the submit button is disabled until checked" in joined


@pytest.mark.asyncio
async def test_agent_remember_without_memory_is_a_structured_failure(session):
    await session.page.set_content("<button id=b>go</button>")
    model = FakeModel(
        responses=[
            {"name": "remember", "params": {"text": "fact"}},
            {"name": "done", "params": {"success": True}},
        ]
    )
    agent = Agent(session=session, model=model)  # no memory configured
    result = await agent.run(task="x")
    assert result.done is True
    # The remember action failed (no memory) rather than crashing the run.
    assert result.history[0].results[0].success is False
