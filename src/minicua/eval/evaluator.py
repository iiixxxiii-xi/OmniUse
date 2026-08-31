"""Declarative evaluator: getter → metric → conj, composed from a task JSON.

The evaluator is the OSWorld-inspired "declarative judgement" that decides
whether a task succeeded. A task's ``evaluator`` is a :class:`EvaluatorSpec`
with three parallel lists plus an aggregation operator:

* ``func`` — one metric name per check (e.g. ``"exact_match"``).
* ``result`` — one getter config per check (e.g. ``{"getter": "element_text",
  "selector": "#result"}``); the ``getter`` key selects the getter, every other
  key is passed through as that getter's options.
* ``expected`` — one rule dict per check, fed to the metric.
* ``conj`` — ``"and"`` (every check must pass) or ``"or"`` (any check passes).

The three lists decouple *where to read state* (getter), *how to compare it*
(metric), and *what to compare against* (options/expected) — so a new task is
pure JSON and never a code change. ``evaluate`` runs each getter → metric pair,
short-circuiting on the ``conj`` (``and`` bails on the first miss, ``or`` on the
first hit) and otherwise returning the mean (``and``) or max (``or``) score.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from minicua.browser.session import BrowserSession
from minicua.eval.errors import EvaluatorError
from minicua.eval.getters import get_getter
from minicua.eval.metrics import get_metric

logger = logging.getLogger("minicua.eval.evaluator")

Conj = Literal["and", "or"]


class EvaluatorSpec(BaseModel):
    """A declarative evaluator: ``func``/``result``/``expected`` + ``conj``."""

    func: str | list[str]
    conj: Conj = "and"
    result: dict[str, Any] | list[dict[str, Any]]
    expected: dict[str, Any] | list[dict[str, Any]]

    @model_validator(mode="after")
    def _check_arity(self) -> "EvaluatorSpec":
        if not self.funcs:
            raise ValueError("evaluator 'func' must name at least one metric")
        if len(self.funcs) != len(self.results):
            raise ValueError(
                f"evaluator arity mismatch: {len(self.funcs)} funcs vs {len(self.results)} results"
            )
        if len(self.funcs) != len(self.expecteds):
            raise ValueError(
                f"evaluator arity mismatch: {len(self.funcs)} funcs vs {len(self.expecteds)} expecteds"
            )
        return self

    @property
    def funcs(self) -> list[str]:
        return [self.func] if isinstance(self.func, str) else list(self.func)

    @property
    def results(self) -> list[dict[str, Any]]:
        return [self.result] if isinstance(self.result, dict) else list(self.result)

    @property
    def expecteds(self) -> list[dict[str, Any]]:
        return [self.expected] if isinstance(self.expected, dict) else list(self.expected)


async def evaluate(session: BrowserSession, spec: EvaluatorSpec | dict[str, Any]) -> float:
    """Run a declarative evaluator against the current browser state; return 0..1."""
    if isinstance(spec, dict):
        spec = EvaluatorSpec.model_validate(spec)

    scores: list[float] = []
    for metric_name, result_cfg, expected in zip(spec.funcs, spec.results, spec.expecteds):
        getter_name = result_cfg.get("getter")
        if not getter_name:
            raise EvaluatorError(f"evaluator result entry is missing 'getter': {result_cfg!r}")

        getter = get_getter(getter_name)
        getter_cfg = {k: v for k, v in result_cfg.items() if k != "getter"}
        metric = get_metric(metric_name)

        try:
            actual = await getter(session, **getter_cfg)
        except Exception as exc:  # noqa: BLE001 - a hard getter failure scores 0, never crashes
            logger.warning("getter %r failed during evaluation: %s", getter_name, exc)
            scores.append(0.0)
        else:
            scores.append(float(metric(actual, expected)))

        # Short-circuit on the conjunction (OSWorld semantics).
        if spec.conj == "and" and scores[-1] == 0.0:
            return 0.0
        if spec.conj == "or" and scores[-1] == 1.0:
            return 1.0

    if spec.conj == "and":
        return sum(scores) / len(scores)
    return max(scores)
