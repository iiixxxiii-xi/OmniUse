"""Eval metrics: pure comparison functions ``(actual, rule) -> float``.

A metric is the second stage of the declarative evaluator pipeline (getter →
metric → conj). It compares the value a getter produced (``actual``) against a
declarative ``rule`` and returns a score in ``[0.0, 1.0]``. Metrics are total:
``None`` ``actual`` (a failed getter) scores ``0.0``. A malformed *rule* —
missing its required key — is a task-authoring error and raises
:class:`MetricError`.

The rule is a plain dict so a task JSON fully determines the comparison:

* ``{"expected": ...}`` — exact / substring / membership / boolean targets.
* ``{"pattern": ...}`` — regex(es) to search.
* ``{"include": [...], "exclude": [...]}`` — substring must/must-not lists.

Metrics are registered in :data:`METRICS`; :func:`get_metric` resolves a
declarative name, raising :class:`MetricError` for an unknown name.
"""

import re
from collections.abc import Callable, Sequence
from typing import Any

from minicua.eval.errors import MetricError

#: metric callable signature: (actual value, rule dict) -> score in [0.0, 1.0].
Metric = Callable[[Any, dict[str, Any]], float]


def _require(rule: dict[str, Any], *keys: str) -> None:
    """Ensure ``rule`` carries the keys this metric needs, else raise."""
    for key in keys:
        if key not in rule:
            raise MetricError(f"metric rule is missing required key {key!r}: {rule!r}")


def _score(cond: bool) -> float:
    return 1.0 if cond else 0.0


def exact_match(actual: Any, rule: dict[str, Any]) -> float:
    """``actual == expected`` (optionally case-insensitive for strings)."""
    _require(rule, "expected")
    expected = rule["expected"]
    if actual is None:
        return 0.0
    if rule.get("ignore_case") and isinstance(actual, str) and isinstance(expected, str):
        return _score(actual.casefold() == expected.casefold())
    return _score(actual == expected)


def contains(actual: Any, rule: dict[str, Any]) -> float:
    """Substring containment.

    Accepts either ``{"expected": str}`` (single substring), ``{"expected":
    [str, ...]}`` (all must be present), or ``{"include": [...], "exclude":
    [...]}`` (all includes present, all excludes absent).
    """
    if actual is None:
        return 0.0
    text = str(actual)

    include: list[str] = []
    exclude: list[str] = []
    if "include" in rule or "exclude" in rule:
        include = [str(x) for x in rule.get("include", [])]
        exclude = [str(x) for x in rule.get("exclude", [])]
    elif "expected" in rule:
        expected = rule["expected"]
        include = [str(x) for x in expected] if isinstance(expected, (list, tuple)) else [str(expected)]
    else:
        raise MetricError(f"contains rule needs 'expected' or 'include': {rule!r}")

    if not all(s in text for s in include):
        return 0.0
    if any(s in text for s in exclude):
        return 0.0
    return 1.0


def regex_match(actual: Any, rule: dict[str, Any]) -> float:
    """Every regex in ``pattern`` (str or list) must search-match ``actual``."""
    _require(rule, "pattern")
    if actual is None:
        return 0.0
    text = str(actual)
    patterns = rule["pattern"]
    if isinstance(patterns, (str, re.Pattern)):
        patterns = [patterns]
    for pattern in patterns:
        if re.search(pattern, text) is None:
            return 0.0
    return 1.0


def count_eq(actual: Any, rule: dict[str, Any]) -> float:
    """Number of matches equals ``expected`` (a sequence's length, else the value)."""
    _require(rule, "expected")
    expected = rule["expected"]
    if actual is None:
        return 0.0
    if isinstance(actual, (list, tuple, set, dict)):
        n: Any = len(actual)
    else:
        n = actual
    return _score(n == expected)


def element_exists_metric(actual: Any, rule: dict[str, Any]) -> float:
    """Boolean equality: ``bool(actual) == bool(expected)`` (presence checks)."""
    _require(rule, "expected")
    return _score(bool(actual) == bool(rule["expected"]))


def match_in_list(actual: Any, rule: dict[str, Any]) -> float:
    """``actual`` is one of ``expected`` (a list of allowed values)."""
    _require(rule, "expected")
    if actual is None:
        return 0.0
    expected = rule["expected"]
    if not isinstance(expected, (list, tuple, set)):
        raise MetricError(f"match_in_list 'expected' must be a list: {rule!r}")
    return _score(actual in expected)


def is_in_list(actual: Any, rule: dict[str, Any]) -> float:
    """``expected`` appears within ``actual`` (a list/sequence of values)."""
    _require(rule, "expected")
    if actual is None:
        return 0.0
    expected = rule["expected"]
    if isinstance(actual, str):
        return _score(expected in actual)
    if isinstance(actual, Sequence):
        return _score(expected in list(actual))
    return 0.0


#: name -> metric callable, the declarative comparison surface for tasks.
METRICS: dict[str, Metric] = {
    "exact_match": exact_match,
    "contains": contains,
    "regex_match": regex_match,
    "count_eq": count_eq,
    "element_exists_metric": element_exists_metric,
    "match_in_list": match_in_list,
    "is_in_list": is_in_list,
}


def get_metric(name: str) -> Metric:
    """Return the metric for a declarative name, raising :class:`MetricError` if unknown."""
    try:
        return METRICS[name]
    except KeyError:
        raise MetricError(f"unknown metric {name!r}") from None
