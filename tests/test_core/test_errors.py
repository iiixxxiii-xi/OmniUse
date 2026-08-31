import pytest

from minicua.core.errors import StaleElementError, PageChangedError, CrashError, LoopDetected


def test_errors_have_messages():
    assert str(StaleElementError(index=5)) == "Element index 5 is stale"
    assert str(PageChangedError(before="a.com", after="b.com")) == "Page changed from a.com to b.com"


def test_loop_detected_is_soft():
    err = LoopDetected(repeat_count=5)
    assert err.repeat_count == 5
