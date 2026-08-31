"""Eval runner: turn a task + model into a scored :class:`EvalResult`.

The runner is the "closed loop" of the eval stage — it composes everything built
so far:

1. start a browser (or reuse a caller-supplied session);
2. set the page up from the task's ``html`` fixture or ``initial_url``;
3. drive the :class:`~minicua.controller.agent.Agent` with the model;
4. score the *final* browser state with the declarative evaluator;
5. synthesize an :class:`EventLog` from the :class:`AgentResult` (so the six
   metrics have a single, auditable source);
6. return a flat :class:`EvalResult` (and close the browser, if it owns it).

A run never raises for a *task* failure: a model error, a budget limit, or a
failed evaluator all produce a structured :class:`EvalResult` with
``success=False`` and an ``error``/``stop_reason``.
"""

import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from minicua.browser.session import BrowserSession
from minicua.controller.agent import Agent, AgentResult, StopReason
from minicua.controller.llm import ChatModel
from minicua.eval.evaluator import evaluate
from minicua.eval.metrics_aggregate import aggregate
from minicua.eval.task import TaskDef
from minicua.state.events import (
    ActionEvent,
    EventLog,
    ModelCallEvent,
    ObservationEvent,
    RecoveryEvent,
    StepEvent,
)

logger = logging.getLogger("minicua.eval.runner")

#: Synthetic origin used to serve inline ``html`` fixtures, so a self-contained
#: task gets a real origin (cookies / localStorage / URL checks all work).
_FIXTURE_URL = "http://minicua.local/"


async def _serve_fixture(session: BrowserSession, html: str) -> None:
    """Serve an inline ``html`` fixture on a real origin (cookies/localStorage work)."""
    async def handler(route: Any) -> None:
        await route.fulfill(status=200, content_type="text/html", body=html)

    await session.context.route(_FIXTURE_URL + "**", handler)
    await session.page.goto(_FIXTURE_URL)


class EvalResult(BaseModel):
    """The outcome of one task run: evaluator score + flat run metrics."""

    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    success: bool
    threshold: float = 0.5
    stop_reason: str = ""
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    recoveries: int = 0
    page_changes: int = 0
    submission: str | None = None
    error: str | None = None
    # The raw event log feeds the six-metric aggregate; excluded from JSON dumps.
    event_log: EventLog = Field(default_factory=EventLog, exclude=True)


class SuiteResult(BaseModel):
    """A whole task set's results plus its six aggregate metrics."""

    results: list[EvalResult] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def success_rate(self) -> float:
        return (self.n_passed / self.n_total) if self.n_total else 0.0


def event_log_from_result(result: AgentResult, latency_seconds: float = 0.0) -> EventLog:
    """Synthesize an :class:`EventLog` from an :class:`AgentResult`.

    The controller keeps a rich :class:`AgentResult` rather than an event log;
    this reconstructs the typed event stream so eval metrics read from one source.
    The first event carries ``ts=0.0`` and the last ``ts=latency_seconds``, so the
    aggregate's ``max(ts) - min(ts)`` recovers the run's wall-clock latency.
    """
    log = EventLog()
    log.append(StepEvent(step=0, phase="perceive", ts=0.0))

    total_actions = sum(len(rec.actions) for rec in result.history)
    if result.cost_usd or result.tokens:
        log.append(
            ModelCallEvent(
                step=0,
                ts=0.0,
                output_tokens=result.tokens,
                cost_usd=result.cost_usd,
                n_tool_calls=total_actions,
            )
        )

    for rec in result.history:
        log.append(StepEvent(step=rec.step, phase="act", ts=0.0))
        for i, action in enumerate(rec.actions):
            action_result = rec.results[i] if i < len(rec.results) else None
            log.append(
                ActionEvent(
                    step=rec.step,
                    ts=0.0,
                    name=action.name,
                    params=action.params.model_dump() if action.params is not None else {},
                    success=action_result.success if action_result is not None else None,
                    error=action_result.error if action_result is not None else None,
                )
            )
        for _ in range(rec.recoveries):
            log.append(RecoveryEvent(step=rec.step, ts=0.0, kind="stale"))

    log.append(ObservationEvent(step=result.steps, ts=latency_seconds, content=""))
    return log


def _to_eval_result(
    task: TaskDef,
    agent_result: AgentResult,
    score: float,
    latency_seconds: float,
    event_log: EventLog,
) -> EvalResult:
    tool_calls = sum(1 for e in event_log.events if isinstance(e, ActionEvent))
    return EvalResult(
        task_id=task.id,
        score=score,
        success=score >= task.threshold,
        threshold=task.threshold,
        stop_reason=agent_result.stop_reason.value,
        steps=agent_result.steps,
        tool_calls=tool_calls,
        tokens=agent_result.tokens,
        cost_usd=agent_result.cost_usd,
        latency_seconds=latency_seconds,
        recoveries=agent_result.recoveries,
        page_changes=agent_result.page_changes,
        submission=agent_result.submission,
        error=agent_result.error,
        event_log=event_log,
    )


async def run_task(
    task: TaskDef,
    model: ChatModel,
    *,
    session: BrowserSession | None = None,
    max_steps: int | None = None,
    use_vision: str = "dom_only",
) -> EvalResult:
    """Run one task: setup page → agent loop → declarative evaluator → result."""
    owns_session = session is None
    session = session or BrowserSession(headless=True)
    try:
        await session.start()
        if task.html is not None:
            await _serve_fixture(session, task.html)
        elif task.initial_url:
            await session.navigate(task.initial_url)

        agent = Agent(
            session=session,
            model=model,
            task=task.instruction,
            max_steps=max_steps or task.max_steps,
            use_vision=use_vision,
        )
        start = time.monotonic()
        agent_result = await agent.run(task.instruction)
        latency_seconds = time.monotonic() - start

        score = await evaluate(session, task.evaluator)
        event_log = event_log_from_result(agent_result, latency_seconds=latency_seconds)
        return _to_eval_result(task, agent_result, score, latency_seconds, event_log)
    finally:
        if owns_session:
            await session.close()


async def run_suite(
    tasks: list[TaskDef],
    model: ChatModel,
    *,
    max_steps: int | None = None,
    use_vision: str = "dom_only",
) -> SuiteResult:
    """Run every task against the same model and aggregate the six metrics."""
    results: list[EvalResult] = []
    for task in tasks:
        results.append(await run_task(task, model, max_steps=max_steps, use_vision=use_vision))
    metrics = aggregate([r.event_log for r in results], [r.score for r in results])
    return SuiteResult(results=results, metrics=metrics)
