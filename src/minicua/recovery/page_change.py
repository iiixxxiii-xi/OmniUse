"""Page-change detection: a lightweight fingerprint of URL + DOM.

The model sometimes returns several actions in one step. Those actions are all
grounded against the *same* perception snapshot; if one of them navigates (or a
prior action re-renders the page), the remaining actions are stale. Comparing a
:class:`PageFingerprint` before and after each action detects this and lets the
controller abort the rest of the queue (the Browser Use ``multi_act`` guard).
"""

import hashlib

from pydantic import BaseModel, ConfigDict, Field


class PageFingerprint(BaseModel):
    """A cheap, hashable snapshot of the page: URL + element count + DOM text hash."""

    model_config = ConfigDict(frozen=True)

    url: str
    element_count: int = Field(ge=0)
    text_hash: str = Field(min_length=1)

    @staticmethod
    def from_browser_state(url: str, dom_text: str, element_count: int) -> "PageFingerprint":
        """Build a fingerprint from a perception snapshot's URL + DOM text."""
        text_hash = hashlib.sha256(dom_text.encode("utf-8", errors="replace")).hexdigest()[:16]
        return PageFingerprint(url=url, element_count=element_count, text_hash=text_hash)


def page_changed(before: PageFingerprint, after: PageFingerprint) -> bool:
    """Whether the page moved between two fingerprints (URL, DOM text, or element count)."""
    return before != after
