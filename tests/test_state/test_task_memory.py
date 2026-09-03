"""TaskMemory: file-backed, cross-task memory store."""

from minicua.state.memory import TaskMemory


def test_remember_and_recall_all(tmp_path):
    m = TaskMemory(tmp_path / "mem.json")
    m.remember("login button is at the page bottom")
    m.remember("checkout flow has 3 steps")
    assert len(m) == 2
    assert [f.text for f in m.recall()] == [
        "login button is at the page bottom",
        "checkout flow has 3 steps",
    ]


def test_recall_matches_substring_and_tags(tmp_path):
    m = TaskMemory(tmp_path / "mem.json")
    m.remember("submit is disabled until checkbox", tags=["form", "login"])
    m.remember("shipping is free over $50", tags=["checkout"])
    # match by text substring
    assert len(m.recall("checkbox")) == 1
    # match by tag
    assert len(m.recall("checkout")) == 1
    # no match
    assert m.recall("nonexistent") == []


def test_remember_dedupes(tmp_path):
    m = TaskMemory(tmp_path / "mem.json")
    m.remember("same fact")
    m.remember("same fact")
    assert len(m) == 1


def test_persists_across_instances(tmp_path):
    path = tmp_path / "mem.json"
    TaskMemory(path).remember("persistent fact")
    reloaded = TaskMemory(path)
    assert [f.text for f in reloaded.recall()] == ["persistent fact"]


def test_in_memory_when_no_path():
    m = TaskMemory()
    m.remember("ephemeral")
    assert len(m) == 1
    assert [f.text for f in m.recall()] == ["ephemeral"]
