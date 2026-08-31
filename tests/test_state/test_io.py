"""Durability primitives: atomic write, tolerant JSON read, append/read JSONL.

These are the building blocks the event log, checkpoint, and trajectory recorder
(and the recovery layer) all share. They must survive crashes, tolerate corrupt
data, and never leak file handles.
"""

import json

from minicua.state.io import append_jsonl, atomic_write_text, read_json_or_none, read_jsonl


def test_atomic_write_and_read_json(tmp_path):
    path = tmp_path / "data.json"
    atomic_write_text(path, json.dumps({"a": 1}))
    assert read_json_or_none(path) == {"a": 1}


def test_atomic_write_overwrites_existing(tmp_path):
    path = tmp_path / "data.json"
    atomic_write_text(path, '{"old": true}')
    atomic_write_text(path, '{"new": true}')
    assert read_json_or_none(path) == {"new": True}


def test_atomic_write_leaves_no_temp_file(tmp_path):
    path = tmp_path / "data.json"
    atomic_write_text(path, '{"a": 1}')
    assert [p for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_read_json_or_none_missing(tmp_path):
    assert read_json_or_none(tmp_path / "nope.json") is None


def test_read_json_or_none_corrupt(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{oops", encoding="utf-8")
    assert read_json_or_none(path) is None


def test_append_jsonl_and_read_jsonl(tmp_path):
    path = tmp_path / "log.jsonl"
    append_jsonl(path, json.dumps({"i": 1}))
    append_jsonl(path, json.dumps({"i": 2}))
    assert read_jsonl(path) == [{"i": 1}, {"i": 2}]


def test_read_jsonl_skips_blank_and_corrupt_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"i": 1}\n\n  \nnot json\n{"i": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"i": 1}, {"i": 2}]


def test_read_jsonl_missing_returns_empty(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_append_jsonl_creates_parent_dir(tmp_path):
    path = tmp_path / "x" / "y" / "log.jsonl"
    append_jsonl(path, json.dumps({"i": 1}))
    assert path.is_file()
