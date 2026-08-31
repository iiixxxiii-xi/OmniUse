"""Full-state checkpoint: the resumable snapshot of a run.

A checkpoint captures everything needed to resume a task after a crash — the
message stream, the step count, the actions already executed, the event log, an
arbitrary ``task_state`` blob, and a path reference to the Playwright
``storage_state`` (so cookies / localStorage survive). It is a superset of the
recovery layer's minimal :class:`~minicua.recovery.crash.RecoveryCheckpoint`
(task + step): the recovery layer shares this module's durability primitives
(atomic write, corruption-tolerant load) via :mod:`minicua.state.io`.

``save`` is atomic (temp file + fsync + rename), so a crash mid-write can never
leave a half-written checkpoint. ``load`` raises :class:`CheckpointError` on a
missing / corrupt / ill-formed checkpoint; ``load_or_none`` degrades to ``None``
for callers (like recovery) that prefer to keep going without one.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from minicua.action.models import Action
from minicua.controller.llm import Message
from minicua.core.errors import CUAError
from minicua.state.events import EventLog
from minicua.state.io import atomic_write_text, read_json_or_none

logger = logging.getLogger("minicua.state.checkpoint")


class CheckpointError(CUAError):
    """A checkpoint is missing, corrupt, or failed schema validation."""


class Checkpoint(BaseModel):
    """The complete resumable state of an agent run."""

    task: str = ""
    step: int = Field(default=0, ge=0)
    messages: list[Message] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    event_log: EventLog = Field(default_factory=EventLog)
    task_state: dict[str, Any] = Field(default_factory=dict)
    storage_state: str | None = None
    created_at: float = Field(default_factory=time.time, ge=0)

    def save(self, path: str | Path) -> None:
        """Atomically persist this checkpoint to ``path`` (creates parent dirs)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.model_dump(mode="json"), ensure_ascii=False)
        atomic_write_text(target, payload)

    @classmethod
    def load(cls, path: str | Path) -> "Checkpoint":
        """Load the most recent checkpoint at ``path``, raising on any problem."""
        checkpoint = cls.load_or_none(path)
        if checkpoint is None:
            raise CheckpointError(f"no valid checkpoint at {path}")
        return checkpoint

    @classmethod
    def load_or_none(cls, path: str | Path) -> "Checkpoint | None":
        """Load a checkpoint, returning ``None`` on missing / corrupt / invalid data."""
        data = read_json_or_none(Path(path))
        if data is None or not isinstance(data, dict):
            return None
        try:
            return cls.model_validate(data)
        except (ValidationError, ValueError) as exc:
            logger.warning("invalid checkpoint at %s: %s", path, exc)
            return None
