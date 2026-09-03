"""Task-level memory: a persistent, file-backed store of facts the agent carries
across tasks.

Mirrors Claude Code's MEMORY.md (an agent-written, transparent scratchpad):
facts are plain strings appended over time and surfaced again when a later task
mentions them. Unlike a trajectory log (which records *what happened*), this
stores *what was learned* — e.g. "the login form on this site is at the page
bottom, not the top".

The store is a single JSON file: one object per fact, written atomically so a
crash mid-write can't corrupt it.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field


class MemoryFact(BaseModel):
    """A single remembered fact with an optional relevance tag."""

    text: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class TaskMemory:
    """Append-only, file-backed memory shared across tasks."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._facts: list[MemoryFact] = []
        if self._path is not None and self._path.exists():
            self._load()

    # -- persistence ----------------------------------------------------

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._facts = [MemoryFact.model_validate(f) for f in raw.get("facts", [])]
        except (json.JSONDecodeError, OSError):
            self._facts = []

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"facts": [f.model_dump() for f in self._facts]}, ensure_ascii=False)
        # Atomic write: temp file + replace, so a crash never leaves a torn file.
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, self._path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # -- API ------------------------------------------------------------

    def remember(self, text: str, tags: list[str] | None = None) -> MemoryFact:
        """Record a fact (deduplicated by exact text) and persist it."""
        fact = MemoryFact(text=text, tags=tags or [])
        if all(f.text != text for f in self._facts):
            self._facts.append(fact)
            self._save()
        return fact

    def recall(self, query: str | None = None, limit: int | None = None) -> list[MemoryFact]:
        """Return facts relevant to ``query`` (substring match on text/tags).

        ``query=None`` returns all facts, newest last. ``limit`` caps the count.
        """
        facts = self._facts
        if query:
            q = query.lower()
            facts = [
                f
                for f in facts
                if q in f.text.lower() or any(q in t.lower() for t in f.tags)
            ]
        if limit is not None:
            facts = facts[-limit:]
        return facts

    def all_facts(self) -> list[MemoryFact]:
        return list(self._facts)

    def __len__(self) -> int:
        return len(self._facts)
