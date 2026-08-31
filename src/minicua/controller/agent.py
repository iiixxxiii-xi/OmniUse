"""The agent control loop: perceive → think → act → observe → repeat.

The loop is deliberately small and exception-driven. Each :meth:`Agent.step`
runs one full cycle:

1. **perceive** — :func:`minicua.perception.extract.extract_state` snapshots the
   page (DOM-first; a screenshot only when the vision policy allows it).
2. **think** — the model (via :class:`~minicua.controller.llm.ChatModel`) turns
   the message history + tool schemas into a :class:`ModelOutput` (thought +
   tool calls). Transient model failures are retried with backoff; a malformed
   response is *requeried* (the error is fed back and the model asked again).
3. **act** — each tool call is validated into an :class:`~minicua.action.models.Action`
   and run via :func:`~minicua.action.executor.execute`. Failures are data
   (structured :class:`ActionResult`), never exceptions.
4. **observe** — results are appended to the history as feedback for the next step.

Termination is *data*: a ``done`` action ends the run and its ``success`` /
``submission`` become the final result. Every other exit is a budget limit or a
classified model failure, both captured as a :class:`StopReason`.

The agent does **not** own the browser session — the caller starts/closes it. A
run is bounded by :class:`Budget` (steps / failures / tokens / cost / timeout).
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from minicua.action.executor import execute
from minicua.action.models import Action, ActionError, ActionResult
from minicua.action.registry import ActionRegistry, get_default_registry
from minicua.browser.crash_watchdog import CrashWatchdog
from minicua.browser.session import BrowserSession
from minicua.controller.budget import Budget
from minicua.controller.llm import (
    ChatModel,
    Message,
    ModelError,
    ModelInvalidResponseError,
    ModelOutput,
)
from minicua.controller.retry import MODEL_RETRY_POLICY, retry_model_call
from minicua.core.errors import BrowserError, CrashError
from minicua.core.retry import RetryPolicy
from minicua.perception.dom import BrowserState
from minicua.perception.extract import extract_state
from minicua.recovery.crash import STORAGE_STATE_FILENAME, RecoveryCheckpoint, recover, save_checkpoint
from minicua.recovery.loop import LoopDetector
from minicua.recovery.page_change import PageFingerprint, page_changed
from minicua.recovery.stale import recover_stale

logger = logging.getLogger("minicua.controller.agent")

# Action failures that indicate the page moved under the model and are worth a
# re-perceive + relocalize before escalating.
_STALE_ERROR_CODES = frozenset({ActionError.STALE_ELEMENT, ActionError.ELEMENT_NOT_FOUND})

# --------------------------------------------------------------------------- #
# Result models
# --------------------------------------------------------------------------- #


class StopReason(str, Enum):
    """Why a run ended (a ``done`` action, a budget limit, or a model failure)."""

    DONE = "done"
    MAX_STEPS = "max_steps"
    MAX_FAILURES = "max_failures"
    TIMEOUT = "timeout"
    MAX_TOKENS = "max_tokens"
    MAX_COST = "max_cost"
    MODEL_ERROR = "model_error"
    INVALID_RESPONSE = "invalid_response"
    ERROR = "error"


class StepRecord(BaseModel):
    """One complete perceive→think→act cycle, for observability."""

    step: int
    thought: str | None = None
    actions: list[Action] = Field(default_factory=list)
    results: list[ActionResult] = Field(default_factory=list)
    page_changed: bool = False
    recoveries: int = 0


class StepResult(BaseModel):
    """The outcome of a single :meth:`Agent.step`."""

    is_done: bool = False
    success: bool | None = None
    submission: str | None = None
    thought: str | None = None
    actions: list[Action] = Field(default_factory=list)
    results: list[ActionResult] = Field(default_factory=list)
    page_changed: bool = False
    recoveries: int = 0


class AgentResult(BaseModel):
    """The final result of a full :meth:`Agent.run`."""

    done: bool
    success: bool | None = None
    submission: str | None = None
    steps: int = 0
    failures: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    stop_reason: StopReason
    error: str | None = None
    history: list[StepRecord] = Field(default_factory=list)
    recoveries: int = 0
    page_changes: int = 0


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a browser automation agent. Complete the task by calling tools, one step at a time.\n"
    "Task: {task}\n"
    "Reference page elements by their [index] from the current page elements list.\n"
    "When the task is finished, call the 'done' tool."
)


def _render_state(url: str, title: str, dom_text: str) -> str:
    lines = [f"URL: {url}"]
    if title:
        lines.append(f"Title: {title}")
    if dom_text:
        lines.append("Page elements:")
        lines.append(dom_text)
    else:
        lines.append("Page elements: (none)")
    return "\n".join(lines)


def _render_observation(results: list[ActionResult]) -> str:
    lines: list[str] = []
    for r in results:
        if r.success:
            lines.append(f"Action succeeded: {r.extracted or 'ok'}")
        else:
            lines.append(f"Action failed: {r.error or 'unknown error'} (code={r.error_code})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class Agent:
    """Drive a task to completion over a bounded perceive→think→act loop."""

    def __init__(
        self,
        session: BrowserSession,
        model: ChatModel,
        *,
        task: str = "",
        max_steps: int = 100,
        max_failures: int = 3,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
        timeout_seconds: float | None = None,
        use_vision: str = "dom_only",
        max_requeries: int = 2,
        max_actions_per_step: int = 10,
        registry: ActionRegistry | None = None,
        retry_policy: RetryPolicy | None = None,
        enable_recovery: bool = True,
        checkpoint_dir: str | Path | None = None,
        loop_detection: bool = True,
        loop_window: int = 10,
        loop_threshold: int = 5,
        crash_watchdog: CrashWatchdog | None = None,
    ) -> None:
        self.session = session
        self.model = model
        self.task = task
        self.use_vision = use_vision
        self.max_requeries = max_requeries
        self.max_actions_per_step = max_actions_per_step
        self.registry = registry or get_default_registry()
        self.retry_policy = retry_policy or MODEL_RETRY_POLICY

        self.enable_recovery = enable_recovery
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.loop_detector = LoopDetector(window=loop_window, threshold=loop_threshold) if loop_detection else None
        self._watchdog = crash_watchdog or CrashWatchdog()
        self.recoveries = 0
        self.page_changes = 0

        self._budget_config = dict(
            max_steps=max_steps,
            max_failures=max_failures,
            max_tokens=max_tokens,
            max_cost_usd=max_cost_usd,
            timeout_seconds=timeout_seconds,
        )
        self.budget = self._new_budget()
        self._tools: list[dict[str, Any]] = self.registry.to_tools()
        self._messages: list[Message] = []
        self._history: list[StepRecord] = []

    def _new_budget(self) -> Budget:
        return Budget(**self._budget_config)

    # -- lifecycle ----------------------------------------------------------

    def _require_page(self):
        page = self.session.page
        if page is None:
            raise BrowserError("browser session is not started (call start() first)")
        return page

    def _system_prompt(self) -> str:
        return _SYSTEM_PROMPT_TEMPLATE.format(task=self.task or "(no task)")

    # -- recovery ----------------------------------------------------------

    def _page_fingerprint(self, state: BrowserState) -> PageFingerprint:
        return PageFingerprint.from_browser_state(state.url, state.dom_text, len(state.selector_map))

    async def _maybe_recover_from_crash(self) -> None:
        """Rebuild the session if the watchdog flagged a crash, else no-op."""
        if not self._watchdog.crashed:
            return
        if not self.enable_recovery or self._checkpoint_dir is None:
            self._watchdog.crashed = False
            raise CrashError(
                f"browser crashed and recovery is unavailable "
                f"(enable_recovery={self.enable_recovery}, checkpoint_dir={self._checkpoint_dir})"
            )
        logger.warning("browser crash detected; recovering from %s", self._checkpoint_dir)
        result = await recover(self.session, self._checkpoint_dir)
        self._watchdog.crashed = False
        self._watchdog.attach(self.session.context)
        if result.checkpoint is not None and result.checkpoint.task:
            self.task = result.checkpoint.task
        self.recoveries += 1
        logger.info("session recovered; resuming task %r", self.task)

    async def _save_checkpoint(self) -> None:
        """Persist storage_state + task checkpoint for a later crash recovery."""
        if self._checkpoint_dir is None:
            return
        try:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            await self.session.save_storage_state(self._checkpoint_dir / STORAGE_STATE_FILENAME)
        except Exception:  # noqa: BLE001 - checkpointing must never crash the loop
            logger.warning("failed to save storage_state checkpoint", exc_info=True)
        try:
            save_checkpoint(self._checkpoint_dir, RecoveryCheckpoint(task=self.task, step=self.budget.steps))
        except Exception:  # noqa: BLE001
            logger.warning("failed to save recovery checkpoint", exc_info=True)

    async def run(self, task: str | None = None) -> AgentResult:
        """Run the loop until ``done``, a budget limit, or a classified model failure."""
        if task is not None:
            self.task = task
        self._require_page()
        self._watchdog.attach(self.session.context)
        self.budget = self._new_budget()
        self.budget.start()
        self._messages = [Message(role="system", content=self._system_prompt())]
        self._history = []
        self.recoveries = 0
        self.page_changes = 0
        await self._save_checkpoint()

        try:
            while True:
                if self.budget.exhausted():
                    reason = self.budget.exhaustion_reason() or StopReason.ERROR.value
                    return self._result(StopReason(reason))
                step = await self.step()
                if step.is_done:
                    return AgentResult(
                        done=True,
                        success=step.success,
                        submission=step.submission,
                        steps=self.budget.steps,
                        failures=self.budget.failures,
                        tokens=self.budget.tokens,
                        cost_usd=self.budget.cost_usd,
                        stop_reason=StopReason.DONE,
                        history=self._history,
                        recoveries=self.recoveries,
                        page_changes=self.page_changes,
                    )
        except ModelInvalidResponseError as exc:
            return self._result(StopReason.INVALID_RESPONSE, error=str(exc))
        except ModelError as exc:
            return self._result(StopReason.MODEL_ERROR, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - surface unexpected failures as a structured result
            logger.exception("unexpected error during agent run")
            return self._result(StopReason.ERROR, error=f"{type(exc).__name__}: {exc}")
        finally:
            # The caller (e.g. the eval runner) closes the session after the run;
            # an intentional close must not be misread as a crash.
            self._watchdog.detach()

    def _result(self, reason: StopReason, error: str | None = None) -> AgentResult:
        return AgentResult(
            done=False,
            stop_reason=reason,
            error=error,
            steps=self.budget.steps,
            failures=self.budget.failures,
            tokens=self.budget.tokens,
            cost_usd=self.budget.cost_usd,
            history=self._history,
            recoveries=self.recoveries,
            page_changes=self.page_changes,
        )

    # -- one step -----------------------------------------------------------

    async def step(self) -> StepResult:
        """Run one perceive→think→act cycle and return its structured outcome."""
        page = self._require_page()
        if not self._messages:
            self._messages = [Message(role="system", content=self._system_prompt())]

        # Recover from a browser crash before perceiving (fresh page + task state).
        await self._maybe_recover_from_crash()
        page = self._require_page()

        # perceive
        state = await extract_state(
            page,
            use_vision=self.use_vision,
            model_supports_vision=self.model.supports_vision,
        )
        self._messages.append(Message(role="user", content=_render_state(state.url, state.title, state.dom_text)))

        # think (with transient retry + requery on malformed output)
        output, actions = await self._think()
        self._record_usage(output)
        if output.thought:
            self._messages.append(Message(role="assistant", content=output.thought))

        # act (with stale-element recovery + page-change guard)
        multi_action = len(actions) > 1
        fingerprint_before = self._page_fingerprint(state) if multi_action else None

        results: list[ActionResult] = []
        is_done = False
        success: bool | None = None
        submission: str | None = None
        page_changed_this_step = False
        recovered_this_step = 0
        for action in actions:
            result = await execute(action, page, state)

            # Stale-element recovery: re-perceive + relocalize, then re-execute
            # with the fresh index rather than failing outright.
            if (
                self.enable_recovery
                and not result.success
                and result.retryable
                and result.error_code in _STALE_ERROR_CODES
            ):
                recovered = await recover_stale(action, state, page)
                if recovered is not None:
                    logger.info("relocalized stale action %s; retrying with fresh index", action.name)
                    recovered_this_step += 1
                    action, state = recovered
                    result = await execute(action, page, state)

            results.append(result)
            if action.name == "done":
                is_done = True
                success = result.success
                submission = result.extracted
                break  # done terminates; ignore any further actions
            if result.success:
                self.budget.reset_failures()
            else:
                self.budget.record_failure()

            # Page-change guard: abort the remaining multi-action queue when the
            # page moved under us (e.g. a navigate re-rendered the whole DOM).
            if multi_action and not is_done:
                after = await extract_state(page)
                if page_changed(fingerprint_before, self._page_fingerprint(after)):
                    page_changed_this_step = True
                    self.page_changes += 1
                    self._messages.append(
                        Message(
                            role="user",
                            content="Page changed mid-step; remaining actions aborted. Re-perceive before continuing.",
                        )
                    )
                    break

        # Loop detection: record this step's actions + page, then inject a soft
        # nudge for the next model call (never blocks an action).
        if self.loop_detector is not None:
            for action in actions:
                params = action.params.model_dump() if action.params is not None else {}
                self.loop_detector.record_action(action.name, params)
            self.loop_detector.record_page_state(state.url, state.dom_text, len(state.selector_map))
            nudge = self.loop_detector.nudge_message()
            if nudge:
                self._messages.append(Message(role="user", content=nudge))

        # observe
        if not is_done and results:
            self._messages.append(Message(role="user", content=_render_observation(results)))

        self.budget.record_step()
        self._history.append(
            StepRecord(
                step=self.budget.steps,
                thought=output.thought,
                actions=actions,
                results=results,
                page_changed=page_changed_this_step,
                recoveries=recovered_this_step,
            )
        )
        self.recoveries += recovered_this_step
        await self._save_checkpoint()

        return StepResult(
            is_done=is_done,
            success=success,
            submission=submission,
            thought=output.thought,
            actions=actions,
            results=results,
            page_changed=page_changed_this_step,
            recoveries=recovered_this_step,
        )

    # -- think (retry + requery) -------------------------------------------

    async def _think(self) -> tuple[ModelOutput, list[Action]]:
        for attempt in range(self.max_requeries + 1):
            output = await self._call_model()
            try:
                return output, self._parse_actions(output)
            except ModelInvalidResponseError as exc:
                if attempt >= self.max_requeries:
                    raise
                logger.warning("malformed model output (requery %d/%d): %s", attempt + 1, self.max_requeries, exc)
                self._messages.append(
                    Message(
                        role="user",
                        content=(
                            f"Your previous response was invalid: {exc}. "
                            "Respond again with valid tool calls (a 'name' and matching 'params')."
                        ),
                    )
                )
        # Unreachable — the loop always raises on the final attempt.
        raise ModelInvalidResponseError("model failed to produce a valid response")

    async def _call_model(self) -> ModelOutput:
        return await retry_model_call(
            lambda: self.model.generate(self._messages, self._tools),
            policy=self.retry_policy,
            logger_=logger,
        )

    def _parse_actions(self, output: ModelOutput) -> list[Action]:
        if not output.tool_calls:
            raise ModelInvalidResponseError("model returned no tool calls")
        if len(output.tool_calls) > self.max_actions_per_step:
            raise ModelInvalidResponseError(
                f"model returned {len(output.tool_calls)} tool calls, "
                f"exceeding max_actions_per_step={self.max_actions_per_step}"
            )
        actions: list[Action] = []
        for tc in output.tool_calls:
            try:
                actions.append(Action(name=tc.name, params=tc.arguments))
            except (ValidationError, ValueError, KeyError) as exc:
                raise ModelInvalidResponseError(f"invalid tool call {tc.name!r}: {exc}") from exc
        return actions

    def _record_usage(self, output: ModelOutput) -> None:
        if output.usage is not None:
            self.budget.record_tokens(output.usage.total_tokens)
            self.budget.record_cost(output.usage.cost_usd)
