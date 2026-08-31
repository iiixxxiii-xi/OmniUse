"""Error taxonomy for the eval layer.

Eval failures split into three families, each a distinct :class:`CUAError` so a
caller (the runner, the CLI) can tell *why* a task scored as it did:

* :class:`GetterError` — the declarative evaluator referenced a getter that does
  not exist (a task-JSON bug, not a page problem).
* :class:`MetricError` — a metric name is unknown, or a metric's rule was
  malformed (missing ``expected``, bad ``pattern``).
* :class:`EvaluatorError` — an evaluator spec is structurally invalid (mismatched
  ``func``/``result``/``expected`` arity, unknown ``conj``, empty spec).
* :class:`TaskDefinitionError` — a task JSON is missing / malformed / unloadable.
"""

from minicua.core.errors import CUAError


class EvalError(CUAError):
    """Base class for all eval-layer failures."""


class GetterError(EvalError):
    """A declarative evaluator referenced an unknown getter."""


class MetricError(EvalError):
    """A declarative evaluator referenced an unknown metric, or a bad rule."""


class EvaluatorError(EvalError):
    """A declarative evaluator spec is structurally invalid."""


class TaskDefinitionError(EvalError):
    """A task definition (JSON) is missing, malformed, or fails validation."""
