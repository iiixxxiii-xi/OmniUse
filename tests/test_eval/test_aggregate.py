"""Task 7.4: six-metric aggregate, extracted from event logs + evaluator results.

The aggregate collapses a set of runs (each an :class:`EventLog`) plus their
evaluator scores into the six summary metrics the project reports:

* ``task_success`` — mean evaluator score (0..1).
* ``avg_tool_calls`` — mean number of action events per run.
* ``token_cost`` — total USD cost across model-call events.
* ``latency`` — total run duration (max ts − min ts per log, summed).
* ``recovery_rate`` — recovery events per action event.
* ``invalid_action_rate`` — failed action events per action event.

Every metric has a divide-by-zero guard (empty input → 0.0), so an empty suite
or a run with no actions still yields a well-formed result dict.
"""

from minicua.eval.metrics_aggregate import SIX_METRICS, aggregate
from minicua.state.events import (
    ActionEvent,
    EventLog,
    ModelCallEvent,
    RecoveryEvent,
    StepEvent,
)


def _two_logs() -> tuple[EventLog, EventLog]:
    # Run 1: 3 actions (2 ok, 1 failed), 1 recovery, cost 0.50, latency 2.0s.
    log1 = EventLog()
    log1.append(StepEvent(step=1, ts=0.0, phase="act"))
    log1.append(ModelCallEvent(step=1, ts=0.0, cost_usd=0.50, output_tokens=10))
    log1.append(ActionEvent(step=1, ts=0.0, name="click", success=True))
    log1.append(ActionEvent(step=1, ts=0.0, name="type", success=False))
    log1.append(ActionEvent(step=1, ts=0.0, name="click", success=True))
    log1.append(RecoveryEvent(step=1, ts=0.0, kind="stale"))
    log1.append(StepEvent(step=1, ts=2.0, phase="done"))

    # Run 2: 1 action (ok), cost 0.25, latency 1.0s.
    log2 = EventLog()
    log2.append(StepEvent(step=1, ts=0.0))
    log2.append(ModelCallEvent(step=1, ts=0.0, cost_usd=0.25))
    log2.append(ActionEvent(step=1, ts=0.0, name="done", success=True))
    log2.append(StepEvent(step=1, ts=1.0))
    return log1, log2


def test_aggregate_six_metrics():
    log1, log2 = _two_logs()
    m = aggregate([log1, log2], [1.0, 0.0])

    assert set(m.keys()) == SIX_METRICS
    assert m["task_success"] == 0.5
    assert m["avg_tool_calls"] == 2.0  # (3 + 1) / 2
    assert m["token_cost"] == 0.75  # 0.50 + 0.25
    assert m["latency"] == 3.0  # 2.0 + 1.0
    assert m["recovery_rate"] == 0.25  # 1 recovery / 4 actions
    assert m["invalid_action_rate"] == 0.25  # 1 failed / 4 actions


def test_aggregate_empty_guards():
    m = aggregate([], [])
    assert m["task_success"] == 0.0
    assert m["avg_tool_calls"] == 0.0
    assert m["token_cost"] == 0.0
    assert m["latency"] == 0.0
    assert m["recovery_rate"] == 0.0
    assert m["invalid_action_rate"] == 0.0


def test_aggregate_no_actions_guards_rates():
    log = EventLog()
    log.append(StepEvent(step=1))
    m = aggregate([log], [1.0])
    assert m["recovery_rate"] == 0.0
    assert m["invalid_action_rate"] == 0.0
    assert m["avg_tool_calls"] == 0.0
    assert m["task_success"] == 1.0


def test_aggregate_single_log_all_success():
    log = EventLog()
    log.append(StepEvent(step=1, ts=0.0))
    log.append(ActionEvent(step=1, ts=0.0, name="done", success=True))
    log.append(StepEvent(step=1, ts=5.0))
    m = aggregate([log], [1.0])
    assert m["task_success"] == 1.0
    assert m["invalid_action_rate"] == 0.0
    assert m["latency"] == 5.0
