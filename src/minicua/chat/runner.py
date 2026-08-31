"""Conversational browser runner: a natural-language instruction → a browser run.

This is the thin wrapper behind ``minicua chat``. It reuses the existing
:class:`~minicua.controller.agent.Agent` (perceive → think → act loop),
:class:`~minicua.browser.session.BrowserSession` (Playwright persistence), and the
inline-HTML fixture serving already used by the eval runner — it does **not**
reimplement them. The only new work is composing those pieces and turning the raw
:class:`~minicua.controller.agent.AgentResult` into a human-readable
:class:`ChatRun`: *what* the agent did, the *final URL*, and a one-line-per-action
summary.

There is deliberately **no evaluator** here: chat mode reports what happened and
leaves pass/fail judgment to the human watching the browser.
"""

import logging

from pydantic import BaseModel, Field

from minicua.action.models import Action, ActionResult
from minicua.browser.session import BrowserSession
from minicua.controller.agent import Agent, AgentResult
from minicua.controller.llm import ChatModel
from minicua.eval.runner import _serve_fixture

logger = logging.getLogger("minicua.chat.runner")


class ChatAction(BaseModel):
    """One action the agent took, flattened for display/testing."""

    step: int
    name: str
    description: str
    success: bool | None = None
    error: str | None = None


class ChatRun(BaseModel):
    """The outcome of one conversational instruction (no evaluator)."""

    instruction: str
    final_url: str
    summary: str
    steps: int = 0
    stop_reason: str = ""
    submission: str | None = None
    error: str | None = None
    actions: list[ChatAction] = Field(default_factory=list)


def _describe_action(action: Action, result: ActionResult | None) -> str:
    """Human-readable one-liner for an action, e.g. ``typed 'hi' into element #2``."""
    name = action.name
    p = action.params

    if name == "click":
        if p is not None and (
            getattr(p, "coordinate_x", None) is not None or getattr(p, "coordinate_y", None) is not None
        ):
            label = f"clicked at ({p.coordinate_x}, {p.coordinate_y})"
        else:
            label = f"clicked element #{getattr(p, 'index', '?')}"
    elif name == "type":
        label = f"typed {getattr(p, 'text', '')!r} into element #{getattr(p, 'index', '?')}"
    elif name == "navigate":
        label = f"navigated to {getattr(p, 'url', '?')}"
    elif name == "scroll":
        label = f"scrolled {getattr(p, 'direction', 'down')}"
    elif name == "press":
        label = f"pressed {getattr(p, 'keys', '?')}"
    elif name == "wait":
        label = f"waited {getattr(p, 'seconds', '?')}s"
    elif name == "go_back":
        label = "went back"
    elif name == "switch_tab":
        label = f"switched to tab {getattr(p, 'index', '?')}"
    elif name == "done":
        submission = getattr(p, "submission", None)
        label = "finished" + (f": {submission}" if submission else "")
    else:
        label = name

    if result is not None and not result.success and result.error:
        label += f" [FAILED: {result.error}]"
    return label


def build_summary(agent_result: AgentResult) -> tuple[list[ChatAction], str]:
    """Flatten an :class:`AgentResult` into structured actions + a summary string."""
    actions: list[ChatAction] = []
    lines: list[str] = []
    for rec in agent_result.history:
        for i, action in enumerate(rec.actions):
            result = rec.results[i] if i < len(rec.results) else None
            desc = _describe_action(action, result)
            actions.append(
                ChatAction(
                    step=rec.step,
                    name=action.name,
                    description=desc,
                    success=result.success if result is not None else None,
                    error=result.error if result is not None else None,
                )
            )
            lines.append(f"{rec.step}. {desc}")
    return actions, "\n".join(lines)


class ChatRunner:
    """Run one natural-language instruction in a fresh browser session (no evaluator).

    Holds the model + budget/vision configuration; each :meth:`run` spins up its
    own :class:`BrowserSession`, drives the :class:`Agent`, and returns a
    :class:`ChatRun`. The session is always closed (when owned) — even on error —
    so no browser process is left behind.
    """

    def __init__(
        self,
        model: ChatModel,
        *,
        max_steps: int = 20,
        use_vision: str = "dom_only",
        headless: bool = True,
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.use_vision = use_vision
        self.headless = headless

    async def run(
        self,
        instruction: str,
        *,
        html: str | None = None,
        initial_url: str | None = None,
        session: BrowserSession | None = None,
    ) -> ChatRun:
        """Run ``instruction`` and return what happened (never raises for task failure)."""
        owns_session = session is None
        session = session or BrowserSession(headless=self.headless)
        try:
            await session.start()
            if html is not None:
                await _serve_fixture(session, html)
            elif initial_url:
                await session.navigate(initial_url)

            agent = Agent(
                session=session,
                model=self.model,
                task=instruction,
                max_steps=self.max_steps,
                use_vision=self.use_vision,
            )
            agent_result = await agent.run(instruction)
            final_url = await session.get_url()
            actions, summary = build_summary(agent_result)

            return ChatRun(
                instruction=instruction,
                final_url=final_url,
                summary=summary,
                steps=agent_result.steps,
                stop_reason=agent_result.stop_reason.value,
                submission=agent_result.submission,
                error=agent_result.error,
                actions=actions,
            )
        finally:
            if owns_session:
                await session.close()
