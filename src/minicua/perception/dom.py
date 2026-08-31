"""Perception data models: a serialized DOM element and the full browser state.

``DOMElement`` is the grounding atom — its ``index`` is what the model emits to
reference a page element, and ``xpath`` / ``stable_hash`` are what later stages
(action grounding, stale-element recovery) use to turn that index back into a
concrete Playwright locator. ``BrowserState`` is the complete perception payload
handed to the model each step: URL, linearized DOM text, the ``selector_map``,
and optional viewport / scroll / screenshot metadata.
"""

from pydantic import BaseModel, Field


class DOMElement(BaseModel):
    """A single interactive page element, identified by a stable ``index``."""

    index: int = Field(ge=0)
    tag: str = Field(min_length=1)
    text: str = ""
    role: str | None = None
    xpath: str | None = None
    stable_hash: str = ""
    ax_name: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    interactive: bool = True
    visible: bool = True
    disabled: bool = False


class Viewport(BaseModel):
    """Visible viewport size in CSS pixels."""

    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)


class ScrollInfo(BaseModel):
    """Current scroll position and document extent in CSS pixels."""

    x: int = Field(default=0, ge=0)
    y: int = Field(default=0, ge=0)
    scroll_height: int = Field(default=0, ge=0)
    client_height: int = Field(default=0, ge=0)


class BrowserState(BaseModel):
    """The complete perception snapshot for one agent step."""

    url: str
    title: str = ""
    dom_text: str = ""
    selector_map: dict[int, DOMElement] = Field(default_factory=dict)
    screenshot: str | None = None  # base64-encoded PNG, or None when unavailable
    viewport: Viewport | None = None
    scroll: ScrollInfo | None = None
