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

import inspect
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from minicua.action.executor import execute
from minicua.action.models import Action, ActionError, ActionResult
from minicua.action.registry import ActionRegistry, get_default_registry
from minicua.browser.crash_watchdog import CrashWatchdog
from minicua.browser.session import BrowserSession
from minicua.controller.budget import Budget
from minicua.controller.llm import (
    ChatModel,
    ImageBlock,
    Message,
    ModelError,
    ModelInvalidResponseError,
    ModelOutput,
    TextBlock,
)
from minicua.controller.retry import MODEL_RETRY_POLICY, retry_model_call
from minicua.core.errors import BrowserError, CrashError
from minicua.core.retry import RetryPolicy
from minicua.desktop.actions import DesktopAction, execute_desktop, get_desktop_registry
from minicua.desktop.perception import DesktopState, extract_desktop_state
from minicua.perception.dom import BrowserState
from minicua.perception.extract import extract_state
from minicua.recovery.crash import STORAGE_STATE_FILENAME, RecoveryCheckpoint, recover, save_checkpoint
from minicua.recovery.loop import LoopDetector
from minicua.state.memory import TaskMemory
from minicua.recovery.page_change import PageFingerprint, page_changed
from minicua.recovery.stale import recover_stale

logger = logging.getLogger("minicua.controller.agent")

# Action failures that indicate the page moved under the model and are worth a
# re-perceive + relocalize before escalating.
_STALE_ERROR_CODES = frozenset({ActionError.STALE_ELEMENT, ActionError.ELEMENT_NOT_FOUND})

# After this many consecutive failed actions, inject a "replan" nudge so the
# model changes strategy instead of burning the failure budget on the same wrong
# action (mirrors Browser Use's ``planning_replan_on_stall``).
_REPLAN_ON_STALL_THRESHOLD = 2

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
    actions: list[Action | DesktopAction] = Field(default_factory=list)
    results: list[ActionResult] = Field(default_factory=list)
    page_changed: bool = False
    recoveries: int = 0


class StepResult(BaseModel):
    """The outcome of a single :meth:`Agent.step`."""

    is_done: bool = False
    success: bool | None = None
    submission: str | None = None
    thought: str | None = None
    actions: list[Action | DesktopAction] = Field(default_factory=list)
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
    recovery_attempts: int = 0
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

_DESKTOP_SYSTEM_PROMPT_TEMPLATE = (
    "You are a desktop automation agent. Complete the task by calling tools, one step at a time.\n"
    "Task: {task}\n"
    "Look at the screenshot first — it is your primary signal for what is on screen. "
    "Control the computer using screen coordinates (x, y), keyboard actions, and shell commands.\n"
    "To open an application, find its icon in the screenshot and click (or double-click) it. "
    "Do not use shell commands (e.g. tasklist) to search for running processes instead of looking at the screen.\n"
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


def _build_state_message(state: BrowserState) -> Message:
    """Render a perceived state into a user message (text, plus an image block
    when a screenshot was captured)."""
    text = _render_state(state.url, state.title, state.dom_text)
    if state.screenshot:
        return Message(
            role="user",
            content=[TextBlock(text=text), ImageBlock(image_base64=state.screenshot)],
        )
    return Message(role="user", content=text)


def _build_desktop_state_message(state: DesktopState) -> Message:
    """Render a desktop perception snapshot into a user message (image + size).

    Desktop has no DOM, so the screenshot is the primary signal; screen size is
    the only textual context (helps the model sanity-check coordinates).
    """
    text = f"Screen: {state.width}x{state.height}"
    if state.screenshot:
        return Message(
            role="user",
            content=[TextBlock(text=text), ImageBlock(image_base64=state.screenshot)],
        )
    return Message(role="user", content=text)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


class Agent:
    """Drive a task to completion over a bounded perceive→think→act loop."""

    def __init__(
        self,
        session: BrowserSession | None = None,
        model: ChatModel | None = None,
        *,
        mode: str = "browser",
        environment: Any = None,
        task: str = "",
        max_steps: int = 100,
        max_failures: int = 3,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
        timeout_seconds: float | None = None,
        use_vision: str = "dom_only",
        max_requeries: int = 2,
        max_actions_per_step: int = 25,
        replan_on_stall: bool = True,
        registry: ActionRegistry | None = None,
        retry_policy: RetryPolicy | None = None,
        enable_recovery: bool = True,
        recovery: bool = True,
        checkpoint_dir: str | Path | None = None,
        loop_detection: bool = True,
        loop_window: int = 10,
        loop_threshold: int = 5,
        crash_watchdog: CrashWatchdog | None = None,
        memory: TaskMemory | None = None,
        verifier: Callable[[], Any] | None = None,
        fault_injector: Callable[[Any], Any] | None = None,
    ) -> None:
        if mode not in ("browser", "desktop"):
            raise ValueError(f"unknown agent mode {mode!r}; expected 'browser' or 'desktop'")
        self.session = session
        self.model = model
        self.mode = mode
        self.environment = environment
        self.task = task
        self.use_vision = use_vision
        self.max_requeries = max_requeries if recovery else 0
        self.max_actions_per_step = max_actions_per_step
        self.replan_on_stall = replan_on_stall
        self._memory = memory
        self._verifier = verifier
        self.fault_injector = fault_injector
        if registry is None:
            registry = get_desktop_registry() if self.mode == "desktop" else get_default_registry()
        self.registry = registry
        self.retry_policy = retry_policy or MODEL_RETRY_POLICY

        # ``recovery`` is the master switch: False strips the agent down to a bare
        # ReAct loop (no stale relocalization, page-change guard, loop detection,
        # crash recovery, or malformed-output requery). The finer-grained flags
        # stay available for partial control, but ``recovery=False`` folds them
        # all off.
        self.recovery = recovery
        self.enable_recovery = enable_recovery and recovery
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.loop_detector = (
            LoopDetector(window=loop_window, threshold=loop_threshold)
            if (loop_detection and recovery)
            else None
        )
        self._watchdog = crash_watchdog or CrashWatchdog()
        self.recoveries = 0
        self.recovery_attempts = 0
        self.page_changes = 0
        # The last perception the model saw, kept so stale recovery can fall back
        # to a *previous* step's element when the model references an index that
        # no longer exists in the current selector map.
        self._last_seen_state: BrowserState | None = None

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

    def _is_desktop(self) -> bool:
        return self.mode == "desktop"

    def _require_page(self):
        if self.session is None:
            raise BrowserError("browser mode requires a BrowserSession")
        page = self.session.page
        if page is None:
            raise BrowserError("browser session is not started (call start() first)")
        return page

    def _require_environment(self):
        """Return the active environment: the Playwright page (browser) or the desktop env."""
        if self._is_desktop():
            if self.environment is None:
                raise BrowserError("desktop mode requires a DesktopEnvironment")
            return self.environment
        return self._require_page()

    def _system_prompt(self) -> str:
        template = _DESKTOP_SYSTEM_PROMPT_TEMPLATE if self._is_desktop() else _SYSTEM_PROMPT_TEMPLATE
        return template.format(task=self.task or "(no task)")

    # -- mode hooks (overridden behavior for desktop vs. browser) -----------

    async def _before_perceive(self) -> None:
        """Recover from a browser crash before perceiving (desktop: no-op)."""
        if not self._is_desktop():
            await self._maybe_recover_from_crash()

    async def _perceive(self):
        """Snapshot the environment: DOM-first (browser) or screenshot-only (desktop)."""
        if self._is_desktop():
            return extract_desktop_state(self.environment)
        page = self._require_page()
        return await extract_state(
            page,
            use_vision=self.use_vision,
            model_supports_vision=self.model.supports_vision if self.model is not None else False,
        )

    async def _execute(self, action: Action, state: Any) -> ActionResult:
        """Run one validated action against the active environment."""
        if action.name == "remember":
            return self._remember(action)
        if self._is_desktop():
            return await execute_desktop(action, self.environment, state)
        page = self._require_page()
        return await execute(action, page, state)

    def _remember(self, action: Action) -> ActionResult:
        """Persist a ``remember`` action into task-level memory (no browser work)."""
        if self._memory is None:
            return ActionResult.fail(
                "task-level memory is not configured",
                error_code=ActionError.EXECUTION_FAILED,
            )
        params = action.params
        self._memory.remember(params.text, params.tags)
        return ActionResult.ok(f"Remembered: {params.text}")

    async def _verify_completion(self) -> tuple[bool, str]:
        """Run the external completion verifier on a claimed success.

        Returns ``(ok, feedback)``. The verifier inspects the *actual* environment
        state, independent of the model's ``done`` claim, so a premature ``done``
        that declares success while the goal is unmet is caught here. A verifier
        that raises (or is absent) degrades to ``(True, "")`` so a broken verifier
        can never hang or crash the loop.
        """
        if self._verifier is None:
            return True, ""
        try:
            verdict = self._verifier()
            if inspect.isawaitable(verdict):
                verdict = await verdict
            ok, feedback = verdict
            return bool(ok), str(feedback or "")
        except Exception as exc:  # noqa: BLE001 - verifier failure is data, not a hang
            logger.warning("completion verifier failed (%s); accepting the agent's done", exc)
            return True, ""

    def _build_state_message(self, state: Any) -> Message:
        """Render a perceived state into a user message (mode-aware)."""
        if self._is_desktop():
            return _build_desktop_state_message(state)
        return _build_state_message(state)

    def _memory_message(self) -> Message | None:
        """Return a user message carrying prior task-level memory, or ``None``."""
        if self._memory is None or len(self._memory) == 0:
            return None
        facts = "\n".join(f"- {fact.text}" for fact in self._memory.recall())
        return Message(
            role="user",
            content=f"Relevant memory from previous tasks:\n{facts}",
        )

    def _supports_stale_recovery(self) -> bool:
        """Stale-element relocalization is browser-only (index grounding)."""
        return not self._is_desktop()

    def _supports_page_change_guard(self) -> bool:
        """Page-change detection is browser-only (DOM fingerprinting)."""
        return not self._is_desktop()

    def _action_model(self) -> type[Action] | type[DesktopAction]:
        """Return the action union the model's tool calls validate against."""
        if self._is_desktop():
            return DesktopAction
        return Action

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
        self.recovery_attempts += 1
        result = await recover(self.session, self._checkpoint_dir)
        self._watchdog.crashed = False
        self._watchdog.attach(self.session.context)
        if result.checkpoint is not None and result.checkpoint.task:
            self.task = result.checkpoint.task
        self.recoveries += 1
        logger.info("session recovered; resuming task %r", self.task)

    async def _save_checkpoint(self) -> None:
        """Persist storage_state + task checkpoint for a later crash recovery."""
        if self._checkpoint_dir is None or self._is_desktop():
            return  # desktop has no browser storage_state to checkpoint
        try:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            await self.session.save_storage_state(self._checkpoint_dir / STORAGE_STATE_FILENAME)
        except Exception:  # noqa: BLE001 - checkpointing must never crash the loop
            logger.warning("failed to save storage_state checkpoint", exc_info=True)
        try:
            save_checkpoint(self._checkpoint_dir, RecoveryCheckpoint(task=self.task, step=self.budget.steps))
        except Exception:  # noqa: BLE001
            logger.warning("failed to save recovery checkpoint", exc_info=True)

    async def _reobserve_and_replan(
        self, action: Action, failed: ActionResult
    ) -> tuple[list[Any], Any] | None:
        """Re-observe the page and ask the model to re-plan from the fresh DOM.

        The last rung of the stale-element ladder: when relocalization cannot
        re-ground the action's index (e.g. the index never existed — a hallucinated
        index), re-perceive a fresh :class:`BrowserState`, feed it back to the model
        with a hint that its last action failed, and let the model emit a new plan
        grounded on the fresh DOM.

        Returns ``(new_actions, fresh_state)`` when the model produced a fresh plan,
        or ``None`` when it could not (malformed output / no tool calls), in which
        case the caller degrades to the ordinary failure path.
        """
        fresh_state = await self._perceive()
        self._messages.append(self._build_state_message(fresh_state))
        self._messages.append(
            Message(
                role="user",
                content=(
                    f"Your last action {action.name!r} failed: "
                    f"{failed.error or 'unknown error'} (code={failed.error_code}). "
                    "The page may have changed. Re-plan using the fresh page elements above."
                ),
            )
        )
        try:
            output, new_actions = await self._think()
        except ModelInvalidResponseError:
            logger.warning("re-plan produced no valid tool calls; giving up recovery")
            return None
        self._record_usage(output)
        if output.thought or output.reasoning_content or output.tool_calls:
            self._messages.append(
                Message(
                    role="assistant",
                    content=output.thought or "",
                    reasoning_content=output.reasoning_content,
                )
            )
        if not new_actions:
            return None
        return new_actions, fresh_state

    async def run(self, task: str | None = None) -> AgentResult:
        """Run the loop until ``done``, a budget limit, or a classified model failure."""
        if task is not None:
            self.task = task
        self._require_environment()
        if not self._is_desktop():
            self._watchdog.attach(self.session.context)
        self.budget = self._new_budget()
        self.budget.start()
        self._messages = [Message(role="system", content=self._system_prompt())]
        memory_msg = self._memory_message()
        if memory_msg is not None:
            self._messages.append(memory_msg)
        self._history = []
        self.recoveries = 0
        self.recovery_attempts = 0
        self.page_changes = 0
        self._last_seen_state = None
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
                        recovery_attempts=self.recovery_attempts,
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
            if not self._is_desktop():
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
            recovery_attempts=self.recovery_attempts,
            page_changes=self.page_changes,
        )

    # -- one step -----------------------------------------------------------

    async def step(self) -> StepResult:
        """Run one perceive→think→act cycle and return its structured outcome."""
        if not self._messages:
            self._messages = [Message(role="system", content=self._system_prompt())]

        # Recover from a browser crash before perceiving (desktop: no-op).
        await self._before_perceive()

        # perceive
        state = await self._perceive()
        perceived_state = state
        self._messages.append(self._build_state_message(state))

        # think (with transient retry + requery on malformed output)
        output, actions = await self._think()
        self._record_usage(output)
        if output.thought or output.reasoning_content or output.tool_calls:
            self._messages.append(
                Message(
                    role="assistant",
                    content=output.thought or "",
                    reasoning_content=output.reasoning_content,
                )
            )

        # Harness-level fault injection (eval-only): perturb the DOM after the
        # model committed to actions, so a stale index is exercised deterministically.
        if self.fault_injector is not None:
            await self.fault_injector(
                session=self.session, state=state, actions=actions, step=self.budget.steps
            )

        # act (with stale-element recovery + page-change guard; browser only)
        multi_action = len(actions) > 1
        fingerprint_before = (
            self._page_fingerprint(state)
            if multi_action and self._supports_page_change_guard() and self.recovery
            else None
        )

        results: list[ActionResult] = []
        is_done = False
        success: bool | None = None
        submission: str | None = None
        page_changed_this_step = False
        recovered_this_step = 0
        for action in actions:
            result = await self._execute(action, state)

            # Stale-element recovery: re-perceive + relocalize, then re-execute
            # with the fresh index rather than failing outright (index grounding
            # is browser-only, so this is a no-op in desktop mode).
            if (
                self.enable_recovery
                and self._supports_stale_recovery()
                and not result.success
                and result.retryable
                and result.error_code in _STALE_ERROR_CODES
            ):
                self.recovery_attempts += 1
                page = self._require_page()
                recovered = await recover_stale(
                    action, state, page, previous_state=self._last_seen_state
                )
                if recovered is not None:
                    logger.info("relocalized stale action %s; retrying with fresh index", action.name)
                    action, state = recovered
                    result = await self._execute(action, state)
                    if result.success:
                        recovered_this_step += 1
                else:
                    # Relocalization failed (e.g. a hallucinated index with no old
                    # element to re-ground). Escalate: re-observe the page and let
                    # the model re-plan against the fresh DOM.
                    replanned = await self._reobserve_and_replan(action, result)
                    if replanned is not None:
                        new_actions, new_state = replanned
                        logger.info(
                            "re-planned %d action(s) after unrelocalizable stale index",
                            len(new_actions),
                        )
                        # Record the failed action for observability, then execute
                        # the fresh plan. A re-planned action that itself fails is
                        # left to the next step (no recursive re-plan). The recovery
                        # only counts as successful if one of the re-planned actions
                        # actually executed successfully.
                        results.append(result)
                        re_planned_succeeded = False
                        for new_action in new_actions:
                            new_result = await self._execute(new_action, new_state)
                            results.append(new_result)
                            if new_action.name == "done":
                                is_done = True
                                success = new_result.success
                                submission = new_result.extracted
                                if new_result.success:
                                    re_planned_succeeded = True
                                break
                            if new_result.success:
                                re_planned_succeeded = True
                                self.budget.reset_failures()
                            else:
                                self.budget.record_failure()
                        if re_planned_succeeded:
                            recovered_this_step += 1
                        # Abandon any remaining original actions: they were grounded
                        # on the pre-replan DOM, which has since changed.
                        break

            results.append(result)
            if action.name == "done":
                # A claimed success is a claim, not ground truth: run the external
                # verifier (when configured) before accepting it. A premature done
                # that fails verification is rejected and fed back so the model
                # keeps working; the rejected done still ends the action queue.
                accept = True
                if result.success and self._verifier is not None:
                    ok, feedback = await self._verify_completion()
                    if not ok:
                        accept = False
                        result = ActionResult.fail(
                            f"completion not verified: {feedback or 'goal unmet'}",
                            error_code=ActionError.EXECUTION_FAILED,
                        )
                        results[-1] = result
                        self._messages.append(
                            Message(
                                role="user",
                                content=(
                                    "Your 'done' was rejected: the task goal is not yet met. "
                                    f"Current state: {feedback}. Continue until it is complete, "
                                    "then call 'done' again."
                                ),
                            )
                        )
                        self.budget.record_failure()
                if accept:
                    is_done = True
                    success = result.success
                    submission = result.extracted
                break  # done always ends the action queue
            if result.success:
                self.budget.reset_failures()
            else:
                self.budget.record_failure()
                # Replan-on-stall: after a couple of consecutive failures, push
                # the model to re-examine the page and change strategy instead of
                # burning the remaining failure budget on the same wrong action.
                if (
                    self.recovery
                    and self.replan_on_stall
                    and self.budget.failures == _REPLAN_ON_STALL_THRESHOLD
                ):
                    self._messages.append(
                        Message(
                            role="user",
                            content=(
                                "REPLAN SUGGESTED: you've had several consecutive failed "
                                "actions. The page may have changed or your approach may be "
                                "wrong. Re-examine the current page state and try a different "
                                "strategy rather than repeating the same action."
                            ),
                        )
                    )

            # Page-change guard: abort the remaining multi-action queue when the
            # page moved under us (e.g. a navigate re-rendered the whole DOM).
            if multi_action and not is_done and self._supports_page_change_guard() and self.recovery:
                after = await self._perceive()
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
            if self._is_desktop():
                self.loop_detector.record_page_state("", "", 0)
            else:
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
        self._last_seen_state = perceived_state
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

    def _parse_actions(self, output: ModelOutput) -> list[Any]:
        if not output.tool_calls:
            raise ModelInvalidResponseError("model returned no tool calls")
        if len(output.tool_calls) > self.max_actions_per_step:
            raise ModelInvalidResponseError(
                f"model returned {len(output.tool_calls)} tool calls, "
                f"exceeding max_actions_per_step={self.max_actions_per_step}"
            )
        model = self._action_model()
        actions: list[Any] = []
        for tc in output.tool_calls:
            try:
                actions.append(model(name=tc.name, params=tc.arguments))
            except (ValidationError, ValueError, KeyError) as exc:
                raise ModelInvalidResponseError(f"invalid tool call {tc.name!r}: {exc}") from exc
        return actions

    def _record_usage(self, output: ModelOutput) -> None:
        if output.usage is not None:
            self.budget.record_tokens(output.usage.total_tokens)
            self.budget.record_cost(output.usage.cost_usd)
