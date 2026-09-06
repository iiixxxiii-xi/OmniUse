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
from minicua.desktop.env import DesktopEnvironment
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
    """Human-readable one-liner for an action, e.g. ``typed 'hi' into element #2``.

    Handles both browser actions (index/DOM grounding) and desktop actions
    (coordinate/shell grounding); overlapping names (``click``/``scroll``/``press``)
    are disambiguated by which parameter attributes the action carries.
    """
    name = action.name
    p = action.params

    if name == "click":
        if p is not None and hasattr(p, "x"):
            label = f"clicked at ({p.x}, {p.y})"
        elif p is not None and (
            getattr(p, "coordinate_x", None) is not None or getattr(p, "coordinate_y", None) is not None
        ):
            label = f"clicked at ({p.coordinate_x}, {p.coordinate_y})"
        else:
            label = f"clicked element #{getattr(p, 'index', '?')}"
    elif name == "type":
        label = f"typed {getattr(p, 'text', '')!r} into element #{getattr(p, 'index', '?')}"
    elif name == "type_text":
        label = f"typed {getattr(p, 'text', '')!r}"
    elif name == "navigate":
        label = f"navigated to {getattr(p, 'url', '?')}"
    elif name == "scroll":
        if p is not None and hasattr(p, "direction"):
            label = f"scrolled {getattr(p, 'direction', 'down')}"
        else:
            label = f"scrolled by {getattr(p, 'amount', 0)}"
    elif name == "press":
        if p is not None and hasattr(p, "key"):
            label = f"pressed {getattr(p, 'key', '?')}"
        else:
            label = f"pressed {getattr(p, 'keys', '?')}"
    elif name == "move_to":
        label = f"moved to ({getattr(p, 'x', '?')}, {getattr(p, 'y', '?')})"
    elif name == "double_click":
        label = f"double-clicked at ({getattr(p, 'x', '?')}, {getattr(p, 'y', '?')})"
    elif name == "right_click":
        label = f"right-clicked at ({getattr(p, 'x', '?')}, {getattr(p, 'y', '?')})"
    elif name == "drag":
        label = (
            f"dragged from ({getattr(p, 'x1', '?')}, {getattr(p, 'y1', '?')}) "
            f"to ({getattr(p, 'x2', '?')}, {getattr(p, 'y2', '?')})"
        )
    elif name == "hotkey":
        label = f"pressed hotkey {'+'.join(getattr(p, 'keys', []) or [])}"
    elif name == "shell":
        label = f"ran {getattr(p, 'command', '?')!r}"
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
        max_steps: int = 50,
        use_vision: str = "dom_only",
        headless: bool = True,
        mode: str = "browser",
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.use_vision = use_vision
        self.headless = headless
        self.mode = mode

    async def run(
        self,
        instruction: str,
        *,
        html: str | None = None,
        initial_url: str | None = None,
        session: BrowserSession | None = None,
        environment: DesktopEnvironment | None = None,
    ) -> ChatRun:
        """Run ``instruction`` and return what happened (never raises for task failure)."""
        mode = self._detect_mode(instruction) if self.mode == "auto" else self.mode
        if mode == "desktop":
            return await self._run_desktop(instruction, environment)
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

    @staticmethod
    def _detect_mode(instruction: str) -> str:
        """Pick browser vs desktop from a URL / web keyword, else desktop."""
        text = instruction.lower()
        browser_hints = (
            "http://", "https://", "www.", ".com", ".cn", ".org", ".io", ".net",
            "网页", "网站", "浏览器", "页面", "网址", "链接",
        )
        return "browser" if any(h in text for h in browser_hints) else "desktop"

    async def _run_desktop(
        self,
        instruction: str,
        environment: DesktopEnvironment | None,
    ) -> ChatRun:
        """Run ``instruction`` against the desktop (no browser, no URL)."""
        env = environment if environment is not None else DesktopEnvironment()
        agent = Agent(
            mode="desktop",
            environment=env,
            model=self.model,
            task=instruction,
            max_steps=self.max_steps,
            use_vision="vision",
        )
        agent_result = await agent.run(instruction)
        actions, summary = build_summary(agent_result)
        return ChatRun(
            instruction=instruction,
            final_url="",
            summary=summary,
            steps=agent_result.steps,
            stop_reason=agent_result.stop_reason.value,
            submission=agent_result.submission,
            error=agent_result.error,
            actions=actions,
        )
