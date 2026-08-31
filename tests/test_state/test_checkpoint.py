"""Task 6.2: checkpoint — full state persistence with atomic write + corruption fallback.

A checkpoint captures the complete resumable state: message stream, step count,
executed actions, the event log, task state, and a reference to the
``storage_state`` file. ``save`` is atomic (temp file + fsync + rename) so a
crash mid-write never leaves a half-written checkpoint; ``load`` / ``load_or_none``
recover from the most recent one, degrading gracefully on corruption.
"""

import json

import pytest

from minicua.action.models import Action
from minicua.controller.llm import Message
from minicua.state.checkpoint import Checkpoint, CheckpointError
from minicua.state.events import EventLog, StepEvent


def test_checkpoint_roundtrip(tmp_path):
    cp = Checkpoint(step=5, event_log=EventLog(), task_state={"goal": "x"})
    path = tmp_path / "ckpt"
    cp.save(path)
    cp2 = Checkpoint.load(path)
    assert cp2.step == 5
    assert cp2.task_state["goal"] == "x"


def test_checkpoint_roundtrip_full_state(tmp_path):
    log = EventLog()
    log.append(StepEvent(step=1, ts=0.0, phase="act"))
    cp = Checkpoint(
        task="book a flight",
        step=7,
        messages=[Message(role="user", content="go")],
        actions=[Action(name="click", params={"index": 1})],
        event_log=log,
        task_state={"goal": "x", "n": 3},
        storage_state="state.json",
    )
    path = tmp_path / "ckpt.json"
    cp.save(path)

    cp2 = Checkpoint.load(path)
    assert cp2.task == "book a flight"
    assert cp2.step == 7
    assert cp2.messages == [Message(role="user", content="go")]
    assert cp2.actions[0].name == "click"
    assert cp2.actions[0].params.index == 1
    assert cp2.event_log.events[0].step == 1
    assert cp2.storage_state == "state.json"


def test_checkpoint_save_is_atomic_no_leftover_tmp(tmp_path):
    cp = Checkpoint(step=1)
    path = tmp_path / "ckpt"
    cp.save(path)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_checkpoint_save_creates_parent_dir(tmp_path):
    path = tmp_path / "a" / "b" / "ckpt"
    Checkpoint(step=1).save(path)
    assert path.is_file()


def test_checkpoint_load_missing_raises(tmp_path):
    with pytest.raises(CheckpointError):
        Checkpoint.load(tmp_path / "nope")


def test_checkpoint_load_corrupt_raises(tmp_path):
    path = tmp_path / "ckpt"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CheckpointError):
        Checkpoint.load(path)


def test_checkpoint_load_wrong_schema_raises(tmp_path):
    path = tmp_path / "ckpt"
    path.write_text(json.dumps({"step": -1}), encoding="utf-8")
    with pytest.raises(CheckpointError):
        Checkpoint.load(path)


def test_checkpoint_load_or_none_missing(tmp_path):
    assert Checkpoint.load_or_none(tmp_path / "nope") is None


def test_checkpoint_load_or_none_corrupt(tmp_path):
    path = tmp_path / "ckpt"
    path.write_text("{oops", encoding="utf-8")
    assert Checkpoint.load_or_none(path) is None


# --------------------------------------------------------------------------- #
# interoperability with the recovery layer
# --------------------------------------------------------------------------- #


def test_recovery_checkpoint_loads_as_state_checkpoint(tmp_path):
    """A recovery-layer checkpoint (task + step) is a valid state Checkpoint."""
    from minicua.recovery.crash import RecoveryCheckpoint, save_checkpoint

    save_checkpoint(tmp_path, RecoveryCheckpoint(task="x", step=3))
    cp = Checkpoint.load(tmp_path / "checkpoint.json")
    assert cp.task == "x"
    assert cp.step == 3


def test_state_checkpoint_loads_via_recovery(tmp_path):
    """A full state checkpoint is a superset the recovery layer can still read."""
    from minicua.recovery.crash import load_checkpoint

    Checkpoint(task="y", step=9).save(tmp_path / "checkpoint.json")
    rc = load_checkpoint(tmp_path)
    assert rc is not None
    assert rc.task == "y"
    assert rc.step == 9

