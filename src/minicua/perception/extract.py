"""Extract a :class:`BrowserState` from a live Playwright page.

DOM is the primary perception signal: we inject a small JavaScript probe that
walks the real DOM, classifies elements as interactive / visible / disabled,
computes a stable XPath for each interactive element (against the real DOM, so
it stays usable for grounding) and returns a normalized list. That list is then
linearized by :func:`minicua.perception.serializer.serialize_dom`, which assigns
the stable ``index`` and computes ``stable_hash``.

All page reads are defensive: any single probe (DOM, viewport, scroll, title)
that fails degrades to a safe default instead of raising, so perception never
crashes the agent loop.
"""

import logging
from typing import Any

from playwright.async_api import Page

from minicua.perception.dom import BrowserState, ScrollInfo, Viewport
from minicua.perception.screenshot import capture, should_capture
from minicua.perception.serializer import serialize_dom

logger = logging.getLogger("minicua.perception.extract")

# JavaScript probe: returns a flat list (document order) of "notable" elements.
# * Interactive elements carry tag/text/attrs/xpath/role/disabled.
# * Non-interactive elements with direct text are returned as context (no xpath)
#   so text-only models retain page structure, but only when they are NOT nested
#   inside an interactive element (avoids duplicating an ancestor's label).
_EXTRACT_JS = r"""
() => {
  const INTERACTIVE_TAGS = new Set(['a','button','input','select','textarea','option','summary','iframe','video','audio']);
  const INTERACTIVE_ROLES = new Set(['button','link','checkbox','radio','textbox','combobox','searchbox','menuitem','menuitemcheckbox','menuitemradio','tab','switch','slider','spinbutton','option','listbox','treeitem','menu','gridcell','rowheader','columnheader']);
  const DISPLAY_ATTRS = ['type','name','id','placeholder','value','aria-label','title','alt','href','role','checked','selected','required','readonly','aria-expanded','aria-checked','aria-selected','aria-disabled','min','max','step','maxlength','pattern'];

  function isVisible(el) {
    if (el.tagName === 'INPUT' && el.type === 'hidden') return false;
    if (el.getAttribute('aria-hidden') === 'true') return false;
    const style = window.getComputedStyle(el);
    if (style.display === 'none') return false;
    if (style.visibility === 'hidden' || style.visibility === 'collapse') return false;
    if (style.opacity === '0') return false;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) return false;
    return true;
  }

  function isDisabled(el) {
    if (el.disabled) return true;
    if (el.getAttribute('aria-disabled') === 'true') return true;
    return false;
  }

  function roleOf(el) {
    return (el.getAttribute('role') || '').trim().toLowerCase() || null;
  }

  function isInteractive(el) {
    if (!isVisible(el)) return false;
    const tag = el.tagName.toLowerCase();
    if (tag === 'a' && !el.hasAttribute('href')) return false;
    if (INTERACTIVE_TAGS.has(tag)) return true;
    const role = roleOf(el);
    if (role && INTERACTIVE_ROLES.has(role)) return true;
    if (el.isContentEditable) return true;
    if (el.hasAttribute('onclick') || el.hasAttribute('onchange') || el.hasAttribute('onsubmit')) return true;
    const tabindex = el.getAttribute('tabindex');
    if (tabindex !== null && Number(tabindex) >= 0) return true;
    return false;
  }

  function getLabel(el) {
    const tag = el.tagName.toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') {
      const v = el.value;
      if (v) return String(v).trim();
      const p = el.getAttribute('placeholder');
      if (p) return p.trim();
      const al = el.getAttribute('aria-label');
      if (al) return al.trim();
      const t = el.getAttribute('title');
      if (t) return t.trim();
      return '';
    }
    const al = el.getAttribute('aria-label');
    if (al) return al.trim();
    const t = el.getAttribute('title');
    if (t) return t.trim();
    const alt = el.getAttribute('alt');
    if (alt) return alt.trim();
    return (el.innerText || el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 200);
  }

  function directText(el) {
    let s = '';
    for (const c of el.childNodes) {
      if (c.nodeType === Node.TEXT_NODE) s += c.textContent;
    }
    return s.trim().replace(/\s+/g, ' ');
  }

  function getXPath(el) {
    if (el.id) return '//*[@id="' + el.id.replace(/"/g, '\\"') + '"]';
    const segs = [];
    let cur = el;
    while (cur && cur.nodeType === 1) {
      let pos = 1, total = 1;
      let prev = cur.previousElementSibling;
      while (prev) { if (prev.tagName === cur.tagName) { pos++; total++; } prev = prev.previousElementSibling; }
      let next = cur.nextElementSibling;
      while (next) { if (next.tagName === cur.tagName) total++; next = next.nextElementSibling; }
      segs.unshift(cur.tagName.toLowerCase() + (total > 1 ? '[' + pos + ']' : ''));
      cur = cur.parentElement;
    }
    return '//' + segs.join('/');
  }

  function getAttrs(el) {
    const out = {};
    for (const a of DISPLAY_ATTRS) {
      const v = el.getAttribute(a);
      if (v !== null && v !== '') out[a] = v;
    }
    return out;
  }

  const result = [];
  const root = document.body || document.documentElement;

  function walk(el, insideInteractive) {
    for (const child of el.children) {
      const interactive = isInteractive(child);
      if (interactive) {
        result.push({
          tag: child.tagName.toLowerCase(),
          text: getLabel(child),
          interactive: true,
          role: roleOf(child),
          disabled: isDisabled(child),
          visible: true,
          attrs: getAttrs(child),
          xpath: getXPath(child),
        });
        walk(child, true);
      } else {
        if (!insideInteractive) {
          const txt = directText(child);
          if (txt) {
            result.push({
              tag: child.tagName.toLowerCase(),
              text: txt,
              interactive: false,
              role: roleOf(child),
              disabled: false,
              visible: true,
              attrs: {},
              xpath: null,
            });
          }
        }
        walk(child, insideInteractive);
      }
    }
  }

  walk(root, false);
  return result;
}
"""


async def _safe_evaluate(page: Page, script: str, default: Any) -> Any:
    try:
        return await page.evaluate(script)
    except Exception as exc:  # noqa: BLE001 - degrade rather than crash the loop
        logger.warning("page.evaluate failed: %s", exc)
        return default


async def _read_viewport(page: Page) -> Viewport | None:
    data = await _safe_evaluate(
        page,
        "() => ({width: window.innerWidth, height: window.innerHeight})",
        None,
    )
    if not isinstance(data, dict):
        return None
    try:
        return Viewport(width=int(data.get("width", 0)), height=int(data.get("height", 0)))
    except (TypeError, ValueError):
        return None


async def _read_scroll(page: Page) -> ScrollInfo | None:
    data = await _safe_evaluate(
        page,
        "() => ({x: window.scrollX, y: window.scrollY, "
        "scroll_height: document.documentElement.scrollHeight, "
        "client_height: document.documentElement.clientHeight})",
        None,
    )
    if not isinstance(data, dict):
        return None
    try:
        return ScrollInfo(
            x=int(data.get("x", 0)),
            y=int(data.get("y", 0)),
            scroll_height=int(data.get("scroll_height", 0)),
            client_height=int(data.get("client_height", 0)),
        )
    except (TypeError, ValueError):
        return None


async def extract_state(
    page: Page,
    *,
    use_vision: str = "dom_only",
    model_supports_vision: bool = False,
) -> BrowserState:
    """Build a :class:`BrowserState` from ``page`` (DOM-first, screenshot optional).

    DOM is always extracted; a screenshot is added only when the ``use_vision``
    policy says so. Any failure (DOM probe, viewport, scroll, screenshot)
    degrades to a safe default instead of raising, so perception never crashes
    the agent loop.
    """
    url = page.url or ""
    title = await _safe_evaluate(page, "() => document.title", "")
    if not isinstance(title, str):
        title = ""

    raw_nodes = await _safe_evaluate(page, _EXTRACT_JS, [])
    if not isinstance(raw_nodes, list):
        logger.warning("DOM probe returned non-list (%s); treating as empty", type(raw_nodes).__name__)
        raw_nodes = []

    dom_text, selector_map = serialize_dom(raw_nodes)

    screenshot: str | None = None
    if should_capture(use_vision, model_supports_vision):
        screenshot = await capture(page)

    return BrowserState(
        url=url,
        title=title,
        dom_text=dom_text,
        selector_map=selector_map,
        screenshot=screenshot,
        viewport=await _read_viewport(page),
        scroll=await _read_scroll(page),
    )
