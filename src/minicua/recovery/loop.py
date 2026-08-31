"""Loop detection: action repetition + page stagnation (soft nudges).

A :class:`LoopDetector` watches two signals over a rolling window and emits a
*soft* nudge message — it never blocks an action. The model is free to keep going
if the repetition is genuine progress. Signals:

* **action repetition** — the same normalized action hashed many times in a window
  (e.g. clicking the same index over and over).
* **page stagnation** — the page fingerprint is identical across consecutive steps,
  meaning the agent's actions are not having an effect.

``wait`` / ``done`` / ``go_back`` are exempt from repetition counting because they
legitimately repeat without indicating a loop. Borrows the design of Browser Use's
``ActionLoopDetector``.
"""

import hashlib
import json
import logging
from typing import Any

from minicua.recovery.page_change import PageFingerprint

logger = logging.getLogger("minicua.recovery.loop")

#: Actions that may legitimately repeat / not change the page.
EXEMPT_ACTIONS = frozenset({"wait", "done", "go_back"})


def _normalize_action(action_name: str, params: dict[str, Any] | None) -> str:
    params = params or {}
    if action_name == "click":
        # Element identity only — two clicks on the same index are the same action.
        return f"click|{params.get('index')}"
    if action_name == "type":
        return f"type|{params.get('index')}|{str(params.get('text', '')).strip().lower()}"
    if action_name == "navigate":
        return f"navigate|{params.get('url', '')}"
    if action_name == "scroll":
        return f"scroll|{params.get('direction', 'down')}"
    filtered = {k: v for k, v in sorted(params.items()) if v is not None}
    return f"{action_name}|{json.dumps(filtered, sort_keys=True, default=str)}"


def compute_action_hash(action_name: str, params: dict[str, Any] | None) -> str:
    """A stable hash of a normalized action, used to detect repetition."""
    return hashlib.sha256(_normalize_action(action_name, params).encode("utf-8")).hexdigest()[:12]


class LoopDetector:
    """Tracks action repetition and page stagnation over a rolling window."""

    def __init__(self, window: int = 10, threshold: int = 5) -> None:
        self.window = window
        self.threshold = threshold
        self._action_hashes: list[str] = []
        self._page_fingerprints: list[PageFingerprint] = []
        self.consecutive_stagnant_pages: int = 0
        self.max_repetition_count: int = 0
        self.most_repeated_hash: str | None = None

    # -- recording ----------------------------------------------------------

    def record_action(self, action_name: str, params: dict[str, Any] | None = None) -> None:
        """Record an executed action and update repetition statistics."""
        if action_name in EXEMPT_ACTIONS:
            return
        h = compute_action_hash(action_name, params)
        self._action_hashes.append(h)
        if len(self._action_hashes) > self.window:
            self._action_hashes = self._action_hashes[-self.window :]
        self._update_repetition_stats()

    def record_page_state(self, url: str, dom_text: str, element_count: int) -> None:
        """Record the current page fingerprint and update the stagnation count."""
        fp = PageFingerprint.from_browser_state(url, dom_text, element_count)
        if self._page_fingerprints and self._page_fingerprints[-1] == fp:
            self.consecutive_stagnant_pages += 1
        else:
            self.consecutive_stagnant_pages = 0
        self._page_fingerprints.append(fp)
        if len(self._page_fingerprints) > 5:
            self._page_fingerprints = self._page_fingerprints[-5:]

    def _update_repetition_stats(self) -> None:
        if not self._action_hashes:
            self.max_repetition_count = 0
            self.most_repeated_hash = None
            return
        counts: dict[str, int] = {}
        for h in self._action_hashes:
            counts[h] = counts.get(h, 0) + 1
        self.most_repeated_hash = max(counts, key=lambda k: counts[k])
        self.max_repetition_count = counts[self.most_repeated_hash]

    # -- detection ----------------------------------------------------------

    def is_looping(self) -> bool:
        """Whether a single action has repeated at least ``threshold`` times."""
        return self.max_repetition_count >= self.threshold

    def stagnant(self) -> bool:
        """Whether the page fingerprint has been identical for ``threshold`` steps."""
        return self.consecutive_stagnant_pages >= self.threshold

    def nudge_message(self) -> str | None:
        """An escalating awareness nudge, or ``None`` when no loop is detected."""
        messages: list[str] = []
        if self.is_looping():
            messages.append(
                f"Heads up: you have repeated a similar action {self.max_repetition_count} times "
                f"in the last {len(self._action_hashes)} actions. If this is intentional and making "
                "progress, carry on. If not, try a different approach."
            )
        if self.stagnant():
            messages.append(
                f"The page has been unchanged across {self.consecutive_stagnant_pages} consecutive "
                "actions. Your actions might not be having the intended effect; consider a different "
                "element or approach."
            )
        if not messages:
            return None
        logger.info("loop nudge: repetition=%d stagnant=%d", self.max_repetition_count, self.consecutive_stagnant_pages)
        return "\n\n".join(messages)
