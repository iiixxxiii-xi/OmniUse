"""Six-metric aggregation over a set of event logs + evaluator results.

Given the :class:`~minicua.state.events.EventLog` for each run plus that run's
evaluator score, :func:`aggregate` collapses the whole suite into the six
headline metrics the project reports. Every metric is derived from the typed
events, so the numbers are auditable from the raw log:

* ``task_success`` — mean evaluator score across runs (0..1).
* ``avg_tool_calls`` — mean :class:`ActionEvent` count per run.
* ``token_cost`` — total ``cost_usd`` summed over :class:`ModelCallEvent`.
* ``latency`` — total run duration (per-log ``max(ts) − min(ts)``, summed).
* ``recovery_rate`` — :class:`RecoveryEvent` count per action event.
* ``invalid_action_rate`` — failed (``success is False``) actions per action.

Every ratio and mean is guarded against a zero denominator, so an empty suite
or a run that produced no actions still returns a complete, well-formed dict.
"""

from collections.abc import Sequence

from minicua.state.events import ActionEvent, EventLog, ModelCallEvent, RecoveryEvent

#: The six aggregate metrics, in report order.
SIX_METRICS = frozenset(
    {
        "task_success",
        "avg_tool_calls",
        "token_cost",
        "latency",
        "recovery_rate",
        "invalid_action_rate",
    }
)


def aggregate(event_logs: Sequence[EventLog], results: Sequence[float]) -> dict[str, float]:
    """Collapse a suite's event logs + evaluator scores into six summary metrics."""
    n_logs = len(event_logs)
    total_actions = 0
    total_failed = 0
    total_recoveries = 0
    total_cost = 0.0
    total_latency = 0.0

    for log in event_logs:
        events = log.replay()
        timestamps = [e.ts for e in events]
        if timestamps:
            total_latency += max(timestamps) - min(timestamps)
        for event in events:
            if isinstance(event, ActionEvent):
                total_actions += 1
                if event.success is False:
                    total_failed += 1
            elif isinstance(event, RecoveryEvent):
                total_recoveries += 1
            elif isinstance(event, ModelCallEvent):
                total_cost += event.cost_usd

    return {
        "task_success": (sum(results) / len(results)) if results else 0.0,
        "avg_tool_calls": (total_actions / n_logs) if n_logs else 0.0,
        "token_cost": total_cost,
        "latency": total_latency,
        "recovery_rate": (total_recoveries / total_actions) if total_actions else 0.0,
        "invalid_action_rate": (total_failed / total_actions) if total_actions else 0.0,
    }
