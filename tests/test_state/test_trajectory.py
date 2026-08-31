"""Task 6.3: trajectory recording — per-step screenshot + action + observation + result.

A trajectory is a replayable, task-tagged JSONL of the steps an agent took, used
for debugging and evaluation. Each line carries a ``task_id`` plus the full
per-step picture (screenshot, thought, actions, results, observation, recovery),
so a whole run can be replayed offline.
"""

import json

from minicua.action.models import ActionResult
from minicua.state.events import StepEvent
from minicua.state.trajectory import TrajectoryRecorder, TrajectoryStep


def test_trajectory_writes_jsonl(tmp_path):
    rec = TrajectoryRecorder(task_id="t1")
    rec.record(StepEvent(step=1, ts=0.0, phase="act"))
    path = tmp_path / "traj.jsonl"
    rec.dump(path)
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["task_id"] == "t1"


def test_trajectory_step_roundtrip(tmp_path):
    step = TrajectoryStep(
        step=1,
        ts=0.0,
        url="data:text/html,<button>x</button>",
        screenshot="b64",
        thought="click it",
        actions=[],
        results=[ActionResult.ok("clicked")],
        observation="clicked",
    )
    rec = TrajectoryRecorder(task_id="t1")
    rec.record(step)
    path = tmp_path / "traj.jsonl"
    rec.dump(path)

    replayed = TrajectoryRecorder(task_id="t1").replay(path)
    assert len(replayed) == 1
    assert replayed[0].step == 1
    assert replayed[0].screenshot == "b64"
    assert replayed[0].results[0].success is True


def test_trajectory_record_event_is_coerced():
    rec = TrajectoryRecorder(task_id="t1")
    rec.record(StepEvent(step=3, ts=1.0, phase="done"))
    assert rec.steps[0].step == 3


def test_trajectory_dump_creates_parent_dir(tmp_path):
    rec = TrajectoryRecorder(task_id="t1")
    rec.record(StepEvent(step=1, ts=0.0, phase="act"))
    path = tmp_path / "deep" / "traj.jsonl"
    rec.dump(path)
    assert path.is_file()


def test_trajectory_file_backed_append_flushes(tmp_path):
    path = tmp_path / "traj.jsonl"
    rec = TrajectoryRecorder(task_id="t1", path=path)
    rec.record(TrajectoryStep(step=1, ts=0.0, screenshot="b64"))
    assert json.loads(path.read_text(encoding="utf-8").strip())["task_id"] == "t1"


def test_trajectory_replay_skips_corrupt_lines(tmp_path):
    path = tmp_path / "traj.jsonl"
    path.write_text(
        '{"task_id":"t1","step":1,"ts":0.0}\n'
        "{corrupt}\n"
        '{"task_id":"t1","step":2,"ts":0.1}\n',
        encoding="utf-8",
    )
    replayed = TrajectoryRecorder(task_id="t1").replay(path)
    assert [s.step for s in replayed] == [1, 2]
