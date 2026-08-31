"""Task 3.1: pydantic action models (param schemas + Action union + ActionResult)."""

import pytest
from pydantic import ValidationError

from minicua.action.models import (
    ACTION_NAMES,
    Action,
    ActionResult,
    ActionError,
    ClickParams,
    DoneParams,
    GoBackParams,
    NavigateParams,
    PressParams,
    ScrollParams,
    SwitchTabParams,
    TypeParams,
    WaitParams,
    action_param_model,
)


def test_click_action_constructed_from_instance():
    a = Action(name="click", params=ClickParams(index=1))
    assert a.name == "click"
    assert a.params.index == 1


def test_action_parses_params_from_dict():
    a = Action(name="click", params={"index": 3})
    assert a.name == "click"
    assert isinstance(a.params, ClickParams)
    assert a.params.index == 3


def test_action_rejects_mismatched_params_type():
    with pytest.raises(ValidationError):
        Action(name="click", params=TypeParams(index=1, text="x"))


def test_action_rejects_unknown_name():
    with pytest.raises(ValidationError):
        Action(name="fly", params=None)


def test_action_requires_params_for_click():
    with pytest.raises(ValidationError):
        Action(name="click", params=None)


def test_go_back_action_allows_no_params():
    a = Action(name="go_back", params=None)
    assert a.name == "go_back"
    assert a.params is None


def test_action_model_dump_roundtrip():
    a = Action(name="type", params=TypeParams(index=2, text="hello"))
    assert a.model_dump() == {"name": "type", "params": {"index": 2, "text": "hello", "clear": True}}


def test_click_params_requires_positive_index():
    with pytest.raises(ValidationError):
        ClickParams(index=0)


def test_type_params_requires_text():
    with pytest.raises(ValidationError):
        TypeParams(index=1, text="")


def test_scroll_params_default_and_validation():
    assert ScrollParams().direction == "down"
    with pytest.raises(ValidationError):
        ScrollParams(direction="sideways")


def test_navigate_params_requires_url():
    with pytest.raises(ValidationError):
        NavigateParams(url="")


def test_done_params_defaults_and_submission():
    assert DoneParams().success is True
    assert DoneParams().submission is None
    assert DoneParams(success=False, submission="the answer").submission == "the answer"


def test_press_params_requires_keys():
    with pytest.raises(ValidationError):
        PressParams(keys="")


def test_wait_params_positive_seconds():
    assert WaitParams().seconds == 1.0
    with pytest.raises(ValidationError):
        WaitParams(seconds=0)


def test_switch_tab_params_zero_based_index():
    assert SwitchTabParams(index=0).index == 0


def test_action_param_model_lookup():
    assert action_param_model("click") is ClickParams
    assert action_param_model("go_back") is GoBackParams


def test_all_nine_actions_registered():
    assert ACTION_NAMES == {
        "click", "type", "scroll", "navigate", "go_back",
        "switch_tab", "press", "wait", "done",
    }


def test_action_result_structured_success():
    r = ActionResult(success=True, extracted="Clicked button")
    assert r.success is True
    assert r.error is None
    assert r.retryable is False


def test_action_result_structured_failure():
    r = ActionResult(
        success=False,
        error="element not found - page may have changed",
        error_code=ActionError.STALE_ELEMENT,
        retryable=True,
    )
    assert r.success is False
    assert r.retryable is True
    assert r.error_code == ActionError.STALE_ELEMENT
