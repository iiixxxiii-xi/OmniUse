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
        # Browser clicks carry an element index; desktop clicks carry (x, y)
        # coordinates. Normalize each so a stuck click in either mode is detected.
        if params.get("index") is not None:
            return f"click|{params.get('index')}"
        return f"click|{params.get('x')},{params.get('y')}"
    if action_name == "type":
        return f"type|{params.get('index')}|{str(params.get('text', '')).strip().lower()}"
    if action_name == "navigate":
        return f"navigate|{params.get('url', '')}"
    if action_name == "scroll":
        return f"scroll|{params.get('direction', 'down')}"
    if action_name == "shell":
        # Normalize by the command's first word (ls/cat/mv/find/...), so a
        # "similar-variation loop" — the same command repeated with different
        # flags, pipes, or trailing args — is still detected as a loop instead
        # of being masked by tiny command diffs.
        cmd = str(params.get("command", "")).strip()
        first = cmd.split()[0] if cmd else ""
        return f"shell|{first}"
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
        self._nudges: int = 0

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
        """An escalating awareness nudge, or ``None`` when no loop is detected.

        The message grows more forceful as the loop persists (``self._nudges``),
        so a strong model that ignores the first soft hint gets a direct,
        actionable instruction to stop and re-decide instead of spinning forever.
        """
        messages: list[str] = []
        if self.is_looping():
            messages.append(
                f"WARNING: you have repeated the same action {self.max_repetition_count} times "
                f"in the last {len(self._action_hashes)} actions — this looks like a loop. "
                "Re-check the current state: if the task goal is already satisfied, call done now. "
                "Otherwise, take a genuinely different action."
            )
        if self.stagnant():
            messages.append(
                f"The page has been unchanged across {self.consecutive_stagnant_pages} consecutive "
                "actions. Your actions are not having the intended effect; re-observe and change "
                "approach."
            )
        if not messages:
            return None
        self._nudges += 1
        if self._nudges >= 3:
            messages.append(
                "You have been warned repeatedly about a loop and are still repeating the same "
                "action. STOP: if the goal is met, call done immediately; otherwise pick a "
                "completely different action right now."
            )
        logger.info("loop nudge: repetition=%d stagnant=%d nudges=%d", self.max_repetition_count, self.consecutive_stagnant_pages, self._nudges)
        return "\n\n".join(messages)
