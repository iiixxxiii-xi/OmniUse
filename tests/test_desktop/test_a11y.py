"""Desktop accessibility-tree extraction (name → clickable-center linearization)."""

from minicua.desktop.a11y import _center, _text, extract_a11y_tree


class _Rect:
    def __init__(self, left, top, right, bottom):
        self.left, self.top, self.right, self.bottom = left, top, right, bottom


class _FakeElement:
    def __init__(self, name="", rect=None):
        self._name = name
        self._rect = rect

    def window_text(self):
        return self._name

    def rectangle(self):
        if self._rect is None:
            raise RuntimeError("no rect")
        return self._rect


def test_center_scales_coordinates():
    el = _FakeElement(rect=_Rect(0, 740, 114, 843))
    # center = ((0+114)/2, (740+843)/2) = (57, 791.5) → /2 → (28, 395)
    assert _center(el, scale=2.0) == (28, 395)


def test_center_skips_empty_rect():
    assert _center(_FakeElement(rect=_Rect(10, 10, 10, 10)), 1.0) is None


def test_text_strips_and_handles_missing():
    assert _text(_FakeElement("  hello ")) == "hello"
    assert _text(_FakeElement()) == ""


def test_extract_returns_empty_when_uia_unavailable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pywinauto":
            raise ImportError("no pywinauto")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert extract_a11y_tree() == ""
