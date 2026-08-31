"""Task 7.2: eval metrics — pure comparison functions ``(actual, rule) -> float``.

Metrics are the second stage of the declarative pipeline. Each is a pure,
total function returning a score in ``[0.0, 1.0]``. A ``None`` ``actual`` is a
safe ``0.0`` (a failed getter should score a miss, never crash). A malformed
*rule* (missing a required key) is a task-authoring bug and raises
:class:`MetricError`, so bad JSON fails loudly instead of silently passing.
"""

import pytest

from minicua.eval.errors import MetricError
from minicua.eval.metrics import (
    METRICS,
    contains,
    count_eq,
    element_exists_metric,
    exact_match,
    get_metric,
    is_in_list,
    match_in_list,
    regex_match,
)


# --------------------------------------------------------------------------- #
# exact_match
# --------------------------------------------------------------------------- #


def test_exact_match():
    assert exact_match("a.com", {"expected": "a.com"}) == 1.0
    assert exact_match("b.com", {"expected": "a.com"}) == 0.0


def test_exact_match_ignore_case():
    assert exact_match("A.COM", {"expected": "a.com", "ignore_case": True}) == 1.0
    assert exact_match("A.COM", {"expected": "a.com"}) == 0.0


def test_exact_match_none_scores_zero():
    assert exact_match(None, {"expected": "a.com"}) == 0.0


def test_exact_match_missing_expected_raises():
    with pytest.raises(MetricError):
        exact_match("a.com", {})


# --------------------------------------------------------------------------- #
# contains
# --------------------------------------------------------------------------- #


def test_contains_substring():
    assert contains("hello world", {"expected": "world"}) == 1.0
    assert contains("hello", {"expected": "world"}) == 0.0


def test_contains_all_substrings():
    assert contains("alpha beta gamma", {"expected": ["alpha", "gamma"]}) == 1.0
    assert contains("alpha beta", {"expected": ["alpha", "gamma"]}) == 0.0


def test_contains_include_exclude():
    assert contains("welcome user", {"include": ["welcome"], "exclude": ["admin"]}) == 1.0
    assert contains("welcome admin", {"include": ["welcome"], "exclude": ["admin"]}) == 0.0


def test_contains_none_scores_zero():
    assert contains(None, {"expected": "x"}) == 0.0


# --------------------------------------------------------------------------- #
# regex_match
# --------------------------------------------------------------------------- #


def test_regex_match():
    assert regex_match("price $42", {"pattern": r"\$\d+"}) == 1.0
    assert regex_match("no price", {"pattern": r"\$\d+"}) == 0.0


def test_regex_match_all_patterns():
    assert regex_match("a1 b2", {"pattern": [r"a\d", r"b\d"]}) == 1.0
    assert regex_match("a1", {"pattern": [r"a\d", r"b\d"]}) == 0.0


def test_regex_match_missing_pattern_raises():
    with pytest.raises(MetricError):
        regex_match("x", {})


# --------------------------------------------------------------------------- #
# count_eq
# --------------------------------------------------------------------------- #


def test_count_eq_list_length():
    assert count_eq([1, 2, 3], {"expected": 3}) == 1.0
    assert count_eq([1, 2], {"expected": 3}) == 0.0


def test_count_eq_scalar():
    assert count_eq(3, {"expected": 3}) == 1.0
    assert count_eq(2, {"expected": 3}) == 0.0


# --------------------------------------------------------------------------- #
# element_exists_metric / match_in_list / is_in_list
# --------------------------------------------------------------------------- #


def test_element_exists_metric():
    assert element_exists_metric(True, {"expected": True}) == 1.0
    assert element_exists_metric(False, {"expected": True}) == 0.0
    assert element_exists_metric(False, {"expected": False}) == 1.0


def test_match_in_list():
    assert match_in_list("red", {"expected": ["red", "blue"]}) == 1.0
    assert match_in_list("green", {"expected": ["red", "blue"]}) == 0.0


def test_is_in_list():
    assert is_in_list(["a", "x"], {"expected": "x"}) == 1.0
    assert is_in_list(["a", "b"], {"expected": "x"}) == 0.0


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_registry_contains_expected_metrics():
    for name in (
        "exact_match",
        "contains",
        "regex_match",
        "count_eq",
        "element_exists_metric",
        "match_in_list",
        "is_in_list",
    ):
        assert name in METRICS


def test_get_metric_unknown_raises():
    with pytest.raises(MetricError):
        get_metric("does_not_exist")
