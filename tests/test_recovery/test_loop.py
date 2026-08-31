"""Task 5.3: loop detection — action repetition + page stagnation.

:class:`minicua.recovery.loop.LoopDetector` is a *soft* detector: it produces a
nudge message for the model but never blocks an action. It watches two signals
over a rolling window — repeated actions (hashed) and consecutive stagnant page
fingerprints — and exempts ``wait`` / ``done`` / ``go_back``, which legitimately
repeat or do not change the page.
"""

from minicua.recovery.loop import LoopDetector


def test_detects_action_repetition():
    d = LoopDetector(window=10, threshold=5)
    for _ in range(6):
        d.record_action("click", {"index": 1})
    assert d.is_looping() is True
    assert d.max_repetition_count == 6


def test_detects_stagnation():
    d = LoopDetector(window=10, threshold=5)
    for _ in range(6):
        d.record_page_state("a.com", "<button>x</button>", 1)
    assert d.stagnant() is True


def test_below_threshold_is_not_looping():
    d = LoopDetector(window=10, threshold=5)
    for _ in range(4):
        d.record_action("click", {"index": 1})
    assert d.is_looping() is False


def test_exempt_actions_do_not_count_as_loops():
    d = LoopDetector(window=10, threshold=3)
    for _ in range(10):
        d.record_action("wait", {"seconds": 1})
        d.record_action("done", {})
        d.record_action("go_back")
    assert d.is_looping() is False
    assert d.max_repetition_count == 0


def test_different_actions_do_not_count_as_loops():
    d = LoopDetector(window=10, threshold=3)
    for i in range(6):
        d.record_action("click", {"index": i})
    assert d.is_looping() is False


def test_stagnation_resets_when_page_changes():
    d = LoopDetector(window=10, threshold=5)
    for _ in range(3):
        d.record_page_state("a.com", "<button>x</button>", 1)
    d.record_page_state("a.com", "<button>y</button>", 1)  # content changes
    assert d.stagnant() is False


def test_nudge_message_when_looping():
    d = LoopDetector(window=10, threshold=2)
    for _ in range(3):
        d.record_action("click", {"index": 1})
    msg = d.nudge_message()
    assert msg is not None
    assert "repeated" in msg


def test_nudge_message_when_stagnant():
    d = LoopDetector(window=10, threshold=2)
    for _ in range(3):
        d.record_page_state("a.com", "<button>x</button>", 1)
    msg = d.nudge_message()
    assert msg is not None
    assert "unchanged" in msg or "stagnant" in msg


def test_nudge_message_none_when_idle():
    d = LoopDetector(window=10, threshold=5)
    d.record_action("click", {"index": 1})
    d.record_page_state("a.com", "<button>x</button>", 1)
    assert d.nudge_message() is None
