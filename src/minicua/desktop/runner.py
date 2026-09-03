"""Desktop task runner: run a single instruction to completion (no browser, no evaluator).

Desktop mode has no DOM to score against, so this is a thin wrapper around the
:class:`~minicua.controller.agent.Agent` — it composes a :class:`DesktopEnvironment`
with the existing perceive→think→act loop and returns a flat, structured result.
The only "success" signal is the agent's own ``done`` action.
"""

import logging
from typing import Any, Callable

from pydantic import BaseModel, Field

from minicua.controller.agent import Agent
from minicua.controller.llm import ChatModel
from minicua.desktop.env import DesktopEnvironment

logger = logging.getLogger("minicua.desktop.runner")


class DesktopRunResult(BaseModel):
    """The outcome of one desktop instruction (mirrors :class:`EvalResult` minus eval)."""

    task_id: str = ""
    success: bool = False
    steps: int = 0
    stop_reason: str = ""
    submission: str | None = None
    error: str | None = None


async def run_desktop(
    instruction: str,
    model: ChatModel,
    *,
    environment: Any = None,
    max_steps: int = 50,
    use_vision: str = "vision",
    verifier: Callable[[], Any] | None = None,
) -> DesktopRunResult:
    """Run ``instruction`` against the desktop and return a structured result.

    ``environment`` may be injected for tests; otherwise a real
    :class:`DesktopEnvironment` is used. A run never raises for a *task* failure
    (a model error or a budget limit becomes ``success=False`` + ``error``).

    ``verifier`` is an optional completion verifier ``() -> (ok, feedback)`` (sync
    or async) that independently checks the *actual* environment state before a
    ``done`` is accepted — the OSWorld-style answer to premature "false done".
    """
    env = environment if environment is not None else DesktopEnvironment()
    agent = Agent(
        mode="desktop",
        environment=env,
        model=model,
        task=instruction,
        max_steps=max_steps,
        use_vision=use_vision,
        verifier=verifier,
    )
    result = await agent.run(instruction)
    return DesktopRunResult(
        success=result.done and result.success is True,
        steps=result.steps,
        stop_reason=result.stop_reason.value,
        submission=result.submission,
        error=result.error,
    )
