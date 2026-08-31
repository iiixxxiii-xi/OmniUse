"""Task 2.2: DOM linearization + stable index assignment (pure function)."""

from minicua.perception.serializer import serialize_dom


def test_serialize_assigns_indexes():
    nodes = [
        {"tag": "div", "text": "", "interactive": False, "attrs": {}},
        {"tag": "button", "text": "登录", "interactive": True, "attrs": {}},
        {"tag": "input", "text": "", "interactive": True, "attrs": {"type": "text"}},
    ]
    text, selector_map = serialize_dom(nodes)
    assert "[1]" in text and "[2]" in text
    assert selector_map[1].tag == "button"
    assert selector_map[2].tag == "input"


def test_serialize_skips_non_interactive_empty_nodes():
    nodes = [{"tag": "div", "text": "", "interactive": False, "attrs": {}}]
    text, selector_map = serialize_dom(nodes)
    assert text == ""
    assert selector_map == {}


def test_serialize_emits_context_text_without_index():
    nodes = [
        {"tag": "h1", "text": "欢迎", "interactive": False, "attrs": {}},
        {"tag": "button", "text": "登录", "interactive": True, "attrs": {}},
    ]
    text, selector_map = serialize_dom(nodes)
    assert "<h1> 欢迎" in text
    assert "[1] <button> 登录" in text
    assert selector_map == {1: selector_map[1]}


def test_serialize_empty_input():
    text, selector_map = serialize_dom([])
    assert text == ""
    assert selector_map == {}


def test_serialize_is_deterministic():
    nodes = [
        {"tag": "button", "text": "A", "interactive": True, "attrs": {}},
        {"tag": "input", "text": "", "interactive": True, "attrs": {"type": "text"}},
    ]
    a = serialize_dom(nodes)
    b = serialize_dom(nodes)
    assert a == b


def test_serialize_computes_xpath_and_stable_hash():
    nodes = [{"tag": "button", "text": "登录", "interactive": True, "attrs": {}}]
    _, selector_map = serialize_dom(nodes)
    el = selector_map[1]
    assert el.xpath == "//button"
    assert el.stable_hash != ""


def test_serialize_xpath_counts_same_tag_siblings():
    nodes = [
        {"tag": "button", "text": "A", "interactive": True, "attrs": {}},
        {"tag": "button", "text": "B", "interactive": True, "attrs": {}},
    ]
    _, selector_map = serialize_dom(nodes)
    assert selector_map[1].xpath == "//button[1]"
    assert selector_map[2].xpath == "//button[2]"


def test_serialize_nested_children_get_path_aware_xpath():
    nodes = [
        {
            "tag": "div",
            "text": "",
            "interactive": False,
            "attrs": {},
            "children": [{"tag": "button", "text": "OK", "interactive": True, "attrs": {}}],
        }
    ]
    _, selector_map = serialize_dom(nodes)
    assert selector_map[1].xpath == "//div/button"


def test_serialize_uses_provided_xpath():
    nodes = [{"tag": "button", "text": "x", "interactive": True, "attrs": {}, "xpath": "//form/button"}]
    _, selector_map = serialize_dom(nodes)
    assert selector_map[1].xpath == "//form/button"


def test_serialize_skips_invisible_interactive_elements():
    nodes = [{"tag": "button", "text": "hidden", "interactive": True, "attrs": {}, "visible": False}]
    text, selector_map = serialize_dom(nodes)
    assert selector_map == {}
    assert "hidden" not in text


def test_serialize_marks_disabled_elements():
    nodes = [{"tag": "button", "text": "go", "interactive": True, "attrs": {}, "disabled": True}]
    text, selector_map = serialize_dom(nodes)
    assert selector_map[1].disabled is True
    assert "(disabled)" in text


def test_serialize_respects_start_index():
    nodes = [{"tag": "button", "text": "A", "interactive": True, "attrs": {}}]
    _, selector_map = serialize_dom(nodes, start_index=5)
    assert 5 in selector_map
    assert 1 not in selector_map


def test_serialize_preserves_provided_ax_name():
    nodes = [{"tag": "button", "text": "Go", "interactive": True, "attrs": {}, "ax_name": "Submit"}]
    _, selector_map = serialize_dom(nodes)
    assert selector_map[1].ax_name == "Submit"
