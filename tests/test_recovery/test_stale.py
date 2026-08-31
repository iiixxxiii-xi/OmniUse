"""Task 5.1: stale-element recovery — relocalize an old element into a fresh selector map.

The core is :func:`minicua.recovery.stale.relocalize`, which re-grounds an element
whose index changed between perceptions by matching, in order: ``stable_hash`` →
``xpath`` → ``ax_name`` → give up (``None``). This is the first rung of the graded
recovery ladder — before re-perceiving and prompting the model to change strategy.
"""

from minicua.action.models import Action, ClickParams
from minicua.perception.dom import DOMElement
from minicua.recovery.stale import relocalize, relocalize_action


def _el(index, *, stable_hash="", ax_name=None, xpath=None, tag="button", text="x"):
    return DOMElement(
        index=index, tag=tag, text=text, stable_hash=stable_hash, ax_name=ax_name, xpath=xpath
    )


# --------------------------------------------------------------------------- #
# relocalize — graded degradation
# --------------------------------------------------------------------------- #


def test_relocalize_by_stable_hash():
    old = _el(1, stable_hash="abc", xpath="//button")
    new_map = {3: _el(3, stable_hash="abc", xpath="//button")}
    assert relocalize(old, new_map) == 3


def test_relocalize_by_xpath_fallback():
    old = _el(1, stable_hash="abc", xpath="//button[@id='save']")
    new_map = {4: _el(4, stable_hash="xyz", xpath="//button[@id='save']")}
    assert relocalize(old, new_map) == 4


def test_relocalize_by_ax_name_fallback():
    old = _el(1, stable_hash="abc", ax_name="Login")
    new_map = {5: _el(5, stable_hash="xyz", ax_name="Login")}
    assert relocalize(old, new_map) == 5


def test_relocalize_prefers_stable_hash_over_xpath_and_ax_name():
    # When multiple signals collide, the strongest signal (stable_hash) wins.
    old = _el(1, stable_hash="abc", xpath="//button[@id='save']", ax_name="Login")
    new_map = {
        2: _el(2, stable_hash="wrong", xpath="//button[@id='save']", ax_name="Login"),
        3: _el(3, stable_hash="abc", xpath="//button[@id='other']", ax_name="Other"),
    }
    assert relocalize(old, new_map) == 3


def test_relocalize_returns_none_when_no_match():
    old = _el(1, stable_hash="abc", xpath="//button[@id='save']", ax_name="Login")
    new_map = {9: _el(9, stable_hash="xyz", xpath="//button[@id='other']", ax_name="Other")}
    assert relocalize(old, new_map) is None


def test_relocalize_empty_stable_hash_does_not_false_match():
    # An empty stable_hash must not match another empty stable_hash — only xpath /
    # ax_name (real identity) may disambiguate.
    old = _el(1, stable_hash="", xpath="//button[@id='save']", ax_name=None)
    new_map = {7: _el(7, stable_hash="", xpath="//button[@id='other']", ax_name=None)}
    assert relocalize(old, new_map) is None


def test_relocalize_empty_map_returns_none():
    assert relocalize(_el(1, stable_hash="abc"), {}) is None


# --------------------------------------------------------------------------- #
# relocalize_action — backfill a fresh index into a click/type action
# --------------------------------------------------------------------------- #


def test_relocalize_action_updates_click_index():
    action = Action(name="click", params=ClickParams(index=1))
    old = _el(1, stable_hash="abc")
    new_map = {8: _el(8, stable_hash="abc")}
    new_action = relocalize_action(action, old, new_map)
    assert new_action is not None
    assert new_action.name == "click"
    assert new_action.params.index == 8
    # original action is not mutated
    assert action.params.index == 1


def test_relocalize_action_returns_none_when_no_match():
    action = Action(name="click", params=ClickParams(index=1))
    assert relocalize_action(action, _el(1, stable_hash="abc"), {}) is None


def test_relocalize_action_returns_none_for_non_index_actions():
    # go_back has no DOM element index — it is never relocalizable.
    action = Action(name="go_back")
    old = _el(1, stable_hash="abc")
    assert relocalize_action(action, old, {1: old}) is None
