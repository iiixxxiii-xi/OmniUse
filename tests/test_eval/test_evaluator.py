"""Task 7.3: declarative evaluator — getter → metric → conj composition.

The evaluator turns a declarative ``EvaluatorSpec`` (``func`` / ``conj`` /
``result`` / ``expected``) into a single 0..1 score by, for each pair, running
the named getter to fetch ``actual``, comparing it via the named metric against
``expected``, and aggregating by ``conj`` (``and`` = all pass → mean; ``or`` =
any pass → max). New tasks are pure JSON — no code change.
"""

import pytest
from pydantic import ValidationError

from minicua.eval.errors import EvaluatorError, GetterError, MetricError
from minicua.eval.evaluator import EvaluatorSpec, evaluate


@pytest.mark.asyncio
async def test_evaluator_and(session):
    await session.page.set_content("<div id=ok>done</div>")
    spec = {
        "func": ["element_exists_metric", "exact_match"],
        "conj": "and",
        "result": [
            {"getter": "element_exists", "selector": "#ok"},
            {"getter": "element_text", "selector": "#ok"},
        ],
        "expected": [{"expected": True}, {"expected": "done"}],
    }
    assert await evaluate(session, spec) == 1.0


@pytest.mark.asyncio
async def test_evaluator_and_fails_when_one_fails(session):
    await session.page.set_content("<div id=ok>done</div>")
    spec = {
        "func": ["element_exists_metric", "exact_match"],
        "conj": "and",
        "result": [
            {"getter": "element_exists", "selector": "#ok"},
            {"getter": "element_text", "selector": "#ok"},
        ],
        "expected": [{"expected": True}, {"expected": "wrong"}],
    }
    assert await evaluate(session, spec) == 0.0


@pytest.mark.asyncio
async def test_evaluator_or_passes_when_any_passes(session):
    await session.page.set_content("<div id=ok>done</div>")
    spec = {
        "func": ["element_exists_metric", "exact_match"],
        "conj": "or",
        "result": [
            {"getter": "element_exists", "selector": "#none"},
            {"getter": "element_text", "selector": "#ok"},
        ],
        "expected": [{"expected": True}, {"expected": "done"}],
    }
    assert await evaluate(session, spec) == 1.0


@pytest.mark.asyncio
async def test_evaluator_or_all_fail(session):
    await session.page.set_content("<div id=ok>done</div>")
    spec = {
        "func": ["element_exists_metric", "exact_match"],
        "conj": "or",
        "result": [
            {"getter": "element_exists", "selector": "#none"},
            {"getter": "element_text", "selector": "#none"},
        ],
        "expected": [{"expected": True}, {"expected": "done"}],
    }
    assert await evaluate(session, spec) == 0.0


@pytest.mark.asyncio
async def test_evaluator_single_string_func(session):
    # func/result/expected may be given as scalars (not wrapped in a list).
    await session.page.set_content("<div id=ok>done</div>")
    spec = {
        "func": "exact_match",
        "conj": "and",
        "result": {"getter": "element_text", "selector": "#ok"},
        "expected": {"expected": "done"},
    }
    assert await evaluate(session, spec) == 1.0


@pytest.mark.asyncio
async def test_evaluator_missing_element_scores_zero(session):
    # A getter that degrades to None feeds a metric that scores 0.0 (not a crash).
    await session.page.set_content("<div id=ok>done</div>")
    spec = {
        "func": "exact_match",
        "conj": "and",
        "result": {"getter": "element_text", "selector": "#none"},
        "expected": {"expected": "done"},
    }
    assert await evaluate(session, spec) == 0.0


# --------------------------------------------------------------------------- #
# spec validation
# --------------------------------------------------------------------------- #


def test_spec_normalizes_scalars_to_lists():
    spec = EvaluatorSpec(
        func="exact_match",
        result={"getter": "element_text", "selector": "#x"},
        expected={"expected": "y"},
    )
    assert spec.funcs == ["exact_match"]
    assert len(spec.results) == 1
    assert len(spec.expecteds) == 1


def test_spec_arity_mismatch_raises():
    with pytest.raises(ValidationError):
        EvaluatorSpec(
            func=["a", "b"],
            result=[{"getter": "page_url"}],
            expected=[{"expected": "x"}],
        )


def test_spec_empty_func_raises():
    with pytest.raises(ValidationError):
        EvaluatorSpec(func=[], result=[], expected=[])


def test_spec_invalid_conj_raises():
    with pytest.raises(ValidationError):
        EvaluatorSpec(
            func="exact_match",
            conj="xor",
            result={"getter": "page_url"},
            expected={"expected": "x"},
        )


@pytest.mark.asyncio
async def test_evaluator_unknown_getter_raises(session):
    spec = {
        "func": "exact_match",
        "result": {"getter": "does_not_exist"},
        "expected": {"expected": "x"},
    }
    with pytest.raises(GetterError):
        await evaluate(session, spec)


@pytest.mark.asyncio
async def test_evaluator_unknown_metric_raises(session):
    spec = {
        "func": "does_not_exist",
        "result": {"getter": "page_url"},
        "expected": {"expected": "x"},
    }
    with pytest.raises(MetricError):
        await evaluate(session, spec)


@pytest.mark.asyncio
async def test_evaluator_result_without_getter_raises(session):
    spec = {
        "func": "exact_match",
        "result": {"selector": "#x"},
        "expected": {"expected": "x"},
    }
    with pytest.raises(EvaluatorError):
        await evaluate(session, spec)
