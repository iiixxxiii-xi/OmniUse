"""Trajectory recording: a replayable, task-tagged JSONL of the steps an agent took.

Where the event log is the fine-grained *source of truth*, a trajectory is the
*replayable* per-step picture used for debugging and evaluation: each line carries
a ``task_id`` plus the full step (screenshot, thought, actions, results,
observation, recovery note). Replaying a trajectory reconstructs exactly what the
agent saw and did, offline, without a browser or model.

Screenshots are base64 PNG inlined into the record (the same shape
:class:`~minicua.perception.dom.BrowserState.screenshot` already carries), so a
trajectory file is self-contained.
"""

import json
import time
from pathlib import Path

from pydantic import BaseModel, Field

from minicua.action.models import Action, ActionResult
from minicua.state.events import Event
from minicua.state.io import append_jsonl, read_jsonl


class TrajectoryStep(BaseModel):
    """One replayable step: the full perceive → think → act → observe picture."""

    step: int = Field(default=0, ge=0)
    ts: float = Field(default_factory=time.time, ge=0)
    url: str | None = None
    screenshot: str | None = None  # base64 PNG, inlined for a self-contained file
    thought: str | None = None
    actions: list[Action] = Field(default_factory=list)
    results: list[ActionResult] = Field(default_factory=list)
    observation: str | None = None
    recovery: str | None = None


class TrajectoryRecorder:
    """Records a run's steps and writes them as a task-tagged JSONL trajectory.

    :class:`TrajectoryRecorder` accepts either a full :class:`TrajectoryStep`
    (the normal path) or a raw :class:`~minicua.state.events.Event` (coerced to a
    minimal step) for compatibility with event-log producers. It can be backed by
    a file (each ``record`` appends + flushes one line) or buffered in memory and
    flushed with :meth:`dump`.
    """

    def __init__(self, task_id: str, path: str | Path | None = None) -> None:
        self.task_id = task_id
        self._path = Path(path) if path is not None else None
        self._steps: list[TrajectoryStep] = []

    @property
    def steps(self) -> list[TrajectoryStep]:
        """The steps recorded so far (in memory)."""
        return self._steps

    def record(self, step: TrajectoryStep | Event) -> None:
        """Record one step, appending it in memory and (when file-backed) to disk."""
        normalized = self._coerce(step)
        self._steps.append(normalized)
        if self._path is not None:
            append_jsonl(self._path, self._serialize(normalized))

    def dump(self, path: str | Path) -> None:
        """Write all recorded steps to ``path`` as JSONL (one line per step)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            for step in self._steps:
                fh.write(self._serialize(step) + "\n")

    def replay(self, path: str | Path | None = None) -> list[TrajectoryStep]:
        """Reconstruct steps from ``path`` (default: the recorder's own file).

        Corrupt / blank lines are skipped so a torn write at the tail is never
        fatal. Without a file, returns the in-memory steps.
        """
        source = Path(path) if path is not None else self._path
        if source is None:
            return list(self._steps)
        steps: list[TrajectoryStep] = []
        for obj in read_jsonl(source):
            step_data = {k: v for k, v in obj.items() if k != "task_id"}
            steps.append(TrajectoryStep.model_validate(step_data))
        return steps

    @staticmethod
    def _coerce(step: TrajectoryStep | Event) -> TrajectoryStep:
        if isinstance(step, TrajectoryStep):
            return step
        return TrajectoryStep(step=step.step, ts=step.ts)

    def _serialize(self, step: TrajectoryStep) -> str:
        payload = {"task_id": self.task_id, **step.model_dump(mode="json")}
        return json.dumps(payload, ensure_ascii=False)
