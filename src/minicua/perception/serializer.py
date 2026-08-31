"""Linearize a normalized DOM tree into text with stable per-element indexes.

The serializer is the single source of truth for index assignment, xpath
derivation and ``stable_hash`` computation. It is a pure function of its input
(no I/O, no Playwright), so it is trivially unit-testable and deterministic:

* Interactive, visible elements are assigned a monotonically increasing
  ``index`` (the grounding handle the model references) in document order.
* Invisible interactive elements are skipped entirely (they cannot be acted on).
* Disabled interactive elements keep an index but are marked ``(disabled)`` so
  the model understands the state without trying to act on them.
* Non-interactive elements that carry text are emitted as plain context lines
  (no index) so text-only models retain page context.
"""

import hashlib
import logging
from typing import Any

from minicua.perception.dom import DOMElement

logger = logging.getLogger("minicua.perception.serializer")

# Attributes surfaced in the linearized text (for the model's benefit).
_DISPLAY_ATTRS = frozenset(
    {
        "type", "name", "id", "placeholder", "value", "aria-label", "title",
        "alt", "href", "role", "checked", "selected", "required", "readonly",
        "aria-expanded", "aria-checked", "aria-selected", "aria-disabled",
        "min", "max", "step", "maxlength", "pattern",
    }
)

# Identity attributes used for the stable hash. Deliberately narrower than
# ``_DISPLAY_ATTRS``: state attributes (``value``, ``checked``, ``selected``,
# ``aria-*`` state) change across steps and would destabilize the hash.
_IDENTITY_ATTRS = frozenset(
    {"id", "name", "type", "placeholder", "aria-label", "title", "alt", "href", "role"}
)


def _build_xpath(segments: list[str]) -> str:
    """Join path segments into an absolute XPath expression."""
    return "//" + "/".join(segments)


def _compute_stable_hash(xpath: str, tag: str, text: str, attrs: dict[str, Any]) -> str:
    identity = "|".join(
        f"{k}={v}" for k, v in sorted(attrs.items()) if k in _IDENTITY_ATTRS and v not in (None, "")
    )
    material = f"{xpath}|{tag}|{identity}|{text}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _render_attrs(attrs: dict[str, Any]) -> str:
    parts = [f"{k}={v}" for k, v in attrs.items() if k in _DISPLAY_ATTRS and v not in (None, "")]
    return " " + " ".join(parts) if parts else ""


def _render_interactive(index: int, tag: str, attrs: dict[str, Any], text: str, disabled: bool) -> str:
    line = f"[{index}] <{tag}{_render_attrs(attrs)}>"
    if text:
        line += f" {text}"
    if disabled:
        line += " (disabled)"
    return line


def _render_context(tag: str, text: str) -> str:
    return f"<{tag}> {text}"


def serialize_dom(
    nodes: list[dict[str, Any]],
    *,
    start_index: int = 1,
) -> tuple[str, dict[int, DOMElement]]:
    """Linearize ``nodes`` into ``(text, selector_map)``.

    ``nodes`` is a normalized tree: each node is a ``dict`` with ``tag``,
    ``text``, ``interactive`` and optional ``attrs`` / ``children`` / ``role`` /
    ``visible`` / ``disabled`` / ``xpath`` keys. Interactive nodes get a stable
    1-based index assigned in document order; ``selector_map`` maps index to the
    corresponding :class:`DOMElement`.
    """
    lines: list[str] = []
    selector_map: dict[int, DOMElement] = {}
    next_index = [start_index]

    def walk(node_list: list[dict[str, Any]], path_segments: list[str]) -> None:
        # Count same-tag siblings up front so positional xpath segments include
        # an index only when it is actually disambiguating.
        tag_counts: dict[str, int] = {}
        for n in node_list:
            tag = str(n.get("tag") or "").lower()
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        seen: dict[str, int] = {}
        for node in node_list:
            tag = str(node.get("tag") or "").lower()
            interactive = bool(node.get("interactive", False))
            visible = bool(node.get("visible", True))
            disabled = bool(node.get("disabled", False))
            text = str(node.get("text") or "").strip()
            attrs = {k: v for k, v in (node.get("attrs") or {}).items()}
            role = node.get("role")
            provided_xpath = node.get("xpath")
            children = node.get("children") or []

            if not tag:
                walk(children, path_segments)
                continue

            seen[tag] = seen.get(tag, 0) + 1
            position = seen[tag]
            count = tag_counts.get(tag, 1)
            segment = tag if count <= 1 else f"{tag}[{position}]"

            if interactive and visible:
                index = next_index[0]
                next_index[0] += 1
                xpath = str(provided_xpath) if provided_xpath else _build_xpath(path_segments + [segment])
                stable_hash = _compute_stable_hash(xpath, tag, text, attrs)
                element = DOMElement(
                    index=index,
                    tag=tag,
                    text=text,
                    role=role,
                    xpath=xpath,
                    stable_hash=stable_hash,
                    ax_name=node.get("ax_name"),
                    attributes=attrs,
                    interactive=True,
                    visible=True,
                    disabled=disabled,
                )
                selector_map[index] = element
                lines.append(_render_interactive(index, tag, attrs, text, disabled))
            elif not interactive and text:
                lines.append(_render_context(tag, text))
            # Invisible interactive elements are intentionally dropped.

            walk(children, path_segments + [segment])

    walk(nodes, [])
    return "\n".join(lines), selector_map
