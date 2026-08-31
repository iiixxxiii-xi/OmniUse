"""Task 6.1: event log — append-only JSONL, the single source of truth for a run.

The event log records every step as a sequence of typed events (``model_call`` /
``step`` / ``action`` / ``observation`` / ``recovery``). ``append`` is
crash-safe: each event is written as one JSON line and flushed, so a crash never
loses already-committed events. ``replay`` reconstructs the stream, tolerating
corrupt lines so a partial write at the tail never makes the whole log unreadable.
"""

import json

import pytest
from pydantic import ValidationError

from minicua.state.events import (
    ActionEvent,
    EventLog,
    ModelCallEvent,
    ObservationEvent,
    RecoveryEvent,
    StepEvent,
    event_from_json,
    event_to_json,
)


# --------------------------------------------------------------------------- #
# event model basics
# --------------------------------------------------------------------------- #


def test_step_event_defaults():
    ev = StepEvent(step=1, ts=0.0, phase="act")
    assert ev.type == "step"
    assert ev.step == 1
    assert ev.ts == 0.0
    assert ev.phase == "act"


def test_event_types_have_distinct_discriminator():
    events = [
        ModelCallEvent(step=1, input_tokens=10, output_tokens=3),
        StepEvent(step=1, phase="act"),
        ActionEvent(step=1, name="click", params={"index": 1}),
        ObservationEvent(step=1, content="clicked"),
        RecoveryEvent(step=1, kind="stale", detail="relocalized"),
    ]
    assert [e.type for e in events] == ["model_call", "step", "action", "observation", "recovery"]


def test_action_event_roundtrips_through_json():
    ev = ActionEvent(step=2, ts=1.5, name="type", params={"index": 1, "text": "hi"}, success=True)
    assert event_from_json(event_to_json(ev)) == ev


def test_invalid_action_event_rejected():
    with pytest.raises(ValidationError):
        ActionEvent(step=1, name="", params={})


# --------------------------------------------------------------------------- #
# in-memory EventLog
# --------------------------------------------------------------------------- #


def test_event_log_append_and_dump():
    log = EventLog()
    log.append(StepEvent(step=1, ts=0.0, phase="act"))
    assert len(log.events) == 1
    d = log.model_dump()
    assert d["events"][0]["step"] == 1


def test_event_log_replay_in_memory():
    log = EventLog()
    log.append(ModelCallEvent(step=1, input_tokens=5, output_tokens=2))
    log.append(ActionEvent(step=1, name="click", params={"index": 1}))
    replayed = log.replay()
    assert [e.type for e in replayed] == ["model_call", "action"]


def test_event_log_to_jsonl_roundtrips():
    log = EventLog()
    log.append(StepEvent(step=1, ts=0.0, phase="perceive"))
    log.append(ActionEvent(step=1, name="done", params={"success": True}))
    text = log.to_jsonl()
    lines = text.splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "step"
    assert json.loads(lines[1])["name"] == "done"


# --------------------------------------------------------------------------- #
# file-backed EventLog (append + flush, replay)
# --------------------------------------------------------------------------- #


def test_file_backed_append_writes_line_immediately(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path=path)
    log.append(StepEvent(step=1, ts=0.0, phase="act"))
    # The line must already be on disk (flushed) — no explicit dump() needed.
    assert json.loads(path.read_text(encoding="utf-8").strip())["step"] == 1


def test_file_backed_replay_reads_back(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path=path)
    log.append(StepEvent(step=1, ts=0.0, phase="act"))
    log.append(ActionEvent(step=2, name="click", params={"index": 1}))
    replayed = EventLog(path=path).replay()
    assert [e.step for e in replayed] == [1, 2]
    assert replayed[1].name == "click"


def test_replay_missing_file_returns_empty(tmp_path):
    assert EventLog(path=tmp_path / "nope.jsonl").replay() == []


def test_replay_skips_corrupt_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"type":"step","ts":0.0,"step":1,"phase":"act"}\n'
        "{this is not valid json}\n"
        '{"type":"action","ts":0.1,"step":1,"name":"click","params":{}}\n',
        encoding="utf-8",
    )
    replayed = EventLog(path=path).replay()
    assert len(replayed) == 2
    assert replayed[0].type == "step"
    assert replayed[1].type == "action"


def test_file_backed_append_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "dir" / "events.jsonl"
    log = EventLog(path=path)
    log.append(StepEvent(step=1, ts=0.0, phase="act"))
    assert path.is_file()
