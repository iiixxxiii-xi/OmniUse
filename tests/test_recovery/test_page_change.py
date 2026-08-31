"""Task 5.2: page-change detection — a lightweight URL + DOM fingerprint.

:class:`minicua.recovery.page_change.PageFingerprint` condenses the page into
(url, element_count, text_hash). :func:`page_changed` compares two fingerprints to
detect that the page moved under the agent — the signal used to abort a stale
multi-action queue before it clicks on something that is no longer there.
"""

import hashlib

from minicua.recovery.page_change import PageFingerprint, page_changed


def _fp(url: str, text: str, count: int | None = None) -> PageFingerprint:
    return PageFingerprint(
        url=url,
        element_count=count if count is not None else 1,
        text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    )


def test_page_changed_when_url_changes():
    a = _fp("a.com", "<button>1</button>")
    b = _fp("b.com", "<button>1</button>")
    assert page_changed(a, b) is True


def test_page_not_changed_when_identical():
    a = _fp("a.com", "<button>1</button>")
    c = _fp("a.com", "<button>1</button>")
    assert page_changed(a, c) is False


def test_page_changed_when_dom_text_changes_same_url():
    a = _fp("a.com", "<button>1</button>")
    b = _fp("a.com", "<button>2</button>")
    assert page_changed(a, b) is True


def test_page_changed_when_element_count_changes():
    a = PageFingerprint(url="a.com", element_count=1, text_hash="x" * 16)
    b = PageFingerprint(url="a.com", element_count=2, text_hash="x" * 16)
    assert page_changed(a, b) is True


def test_from_browser_state_hashes_dom_text():
    fp = PageFingerprint.from_browser_state(
        url="a.com",
        dom_text="<button>hi</button>",
        element_count=1,
    )
    expected = hashlib.sha256(b"<button>hi</button>").hexdigest()[:16]
    assert fp.text_hash == expected
    assert fp.element_count == 1
    assert fp.url == "a.com"


def test_fingerprint_is_hashable_and_comparable():
    a = _fp("a.com", "<button>1</button>")
    b = _fp("a.com", "<button>1</button>")
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
