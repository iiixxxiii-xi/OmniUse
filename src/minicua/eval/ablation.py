"""Baseline-vs-full recovery ablation.

Run the *same* task set twice — once as a bare ReAct loop (``recovery=False``)
and once with the full recovery ladder (``recovery=True``) — then report the
success-rate, invalid-action, and recovery-success deltas that support the
"基础 ReAct Agent vs full" comparison.

``run_ablation`` drives :func:`minicua.eval.runner.run_suite` twice with a
:class:`ModelFactory` (a zero-arg callable returning a :class:`ChatModel`), so a
stateful test double (e.g. :class:`FakeModel`) gets a fresh script per pass while
a real model can return the same stateless instance.
"""

from collections.abc import Callable

from pydantic import BaseModel

from minicua.controller.llm import ChatModel
from minicua.eval.runner import SuiteResult, run_suite
from minicua.eval.task import TaskDef

#: A zero-arg callable producing a fresh :class:`ChatModel` for one suite run.
ModelFactory = Callable[[], ChatModel]


def recovery_success_rate(suite: SuiteResult) -> float:
    """Successful recovery events per recovery attempt across a suite (0.0 if none)."""
    attempts = sum(r.recovery_attempts for r in suite.results)
    successes = sum(r.recoveries for r in suite.results)
    return (successes / attempts) if attempts else 0.0


class AblationResult(BaseModel):
    """``baseline`` (recovery off) vs ``full`` (recovery on) comparison."""

    baseline: SuiteResult
    full: SuiteResult

    @property
    def success_rate_delta(self) -> float:
        """Task success-rate improvement (full − baseline)."""
        return self.full.success_rate - self.baseline.success_rate

    @property
    def invalid_action_delta(self) -> float:
        """Invalid-action-rate reduction (baseline − full; positive = improvement)."""
        baseline_rate = self.baseline.metrics.get("invalid_action_rate", 0.0)
        full_rate = self.full.metrics.get("invalid_action_rate", 0.0)
        return baseline_rate - full_rate

    @property
    def recovery_success_rate(self) -> float:
        """Recovery-event success rate in full mode (baseline has no recovery)."""
        return recovery_success_rate(self.full)

    def comparison(self) -> dict[str, float]:
        """Flat, report-ready metrics for both modes and their deltas."""
        return {
            "baseline_success_rate": self.baseline.success_rate,
            "full_success_rate": self.full.success_rate,
            "success_rate_delta": self.success_rate_delta,
            "baseline_invalid_action_rate": self.baseline.metrics.get("invalid_action_rate", 0.0),
            "full_invalid_action_rate": self.full.metrics.get("invalid_action_rate", 0.0),
            "invalid_action_delta": self.invalid_action_delta,
            "baseline_recovery_success_rate": recovery_success_rate(self.baseline),
            "full_recovery_success_rate": self.recovery_success_rate,
        }


async def run_ablation(
    tasks: list[TaskDef],
    model_factory: ModelFactory,
    *,
    max_steps: int | None = None,
    use_vision: str = "dom_only",
) -> AblationResult:
    """Run ``tasks`` as a bare ReAct loop and with full recovery, then compare.

    ``model_factory`` is called once per run so a stateful model (FakeModel) is
    not exhausted between the baseline and full passes.
    """
    baseline = await run_suite(
        tasks, model_factory(), max_steps=max_steps, use_vision=use_vision, recovery=False
    )
    full = await run_suite(
        tasks, model_factory(), max_steps=max_steps, use_vision=use_vision, recovery=True
    )
    return AblationResult(baseline=baseline, full=full)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_ablation_markdown(
    ablation: AblationResult,
    title: str = "CUA Recovery Ablation Report",
) -> str:
    """Render a human-readable baseline-vs-full comparison as markdown."""
    c = ablation.comparison()
    lines = [
        f"# {title}",
        "",
        f"{ablation.baseline.n_total} tasks, baseline (recovery off) vs full (recovery on).",
        "",
        "| Metric | Baseline | Full | Delta |",
        "| ------ | -------- | ---- | ----- |",
        f"| Success rate | {_pct(c['baseline_success_rate'])} | {_pct(c['full_success_rate'])} "
        f"| {c['success_rate_delta']:+.1%} |",
        f"| Invalid action rate | {_pct(c['baseline_invalid_action_rate'])} | "
        f"{_pct(c['full_invalid_action_rate'])} | {c['invalid_action_delta']:+.1%} |",
        f"| Recovery success rate | {_pct(c['baseline_recovery_success_rate'])} | "
        f"{_pct(c['full_recovery_success_rate'])} | — |",
        "",
    ]
    return "\n".join(lines)
