"""Append-only event log — the single source of truth for an agent run.

Every step of the loop emits typed events that, appended in order, reconstruct
the full run: a ``model_call`` (what the model thought and how much it cost), a
``step`` marker, each ``action`` the model took, the ``observation`` fed back to
it, and any ``recovery`` that intervened (stale relocalize, page-change abort,
loop nudge, crash rebuild).

Events are a pydantic discriminated union on ``type``, so the log is
self-describing: a later reader (Stage 7's six-metric aggregator, a replay tool)
can re-validate every line against the exact schema that wrote it.

:class:`EventLog` is the append-only container. It can be used purely in-memory
(for tests and short-lived runs) or backed by a JSONL file: ``append`` then
writes one line and flushes it, so a crash never loses a committed event, and
``replay`` reads the stream back (tolerating a torn/corrupt tail line).
"""

import time
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from minicua.state.io import append_jsonl, read_jsonl

# --------------------------------------------------------------------------- #
# Event models (discriminated on ``type``)
# --------------------------------------------------------------------------- #

EventType = Literal["model_call", "step", "action", "observation", "recovery"]

StepPhase = Literal["perceive", "think", "act", "observe", "done"]


class BaseEvent(BaseModel):
    """Common fields for every event in the log."""

    ts: float = Field(default_factory=time.time, ge=0, description="Unix timestamp (seconds).")
    step: int = Field(default=0, ge=0, description="1-based step number this event belongs to.")


class ModelCallEvent(BaseEvent):
    """One model response: its thought and the usage it consumed."""

    type: Literal["model_call"] = "model_call"
    thought: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    n_tool_calls: int = Field(default=0, ge=0)


class StepEvent(BaseEvent):
    """A step boundary marker (optionally tagged with the loop phase)."""

    type: Literal["step"] = "step"
    phase: StepPhase = "act"


class ActionEvent(BaseEvent):
    """One action the model chose and its execution outcome."""

    type: Literal["action"] = "action"
    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    success: bool | None = None
    error: str | None = None


class ObservationEvent(BaseEvent):
    """The observation (state or result feedback) fed back to the model."""

    type: Literal["observation"] = "observation"
    content: str = ""


class RecoveryEvent(BaseEvent):
    """A recovery intervention (stale / page-change / loop / crash)."""

    type: Literal["recovery"] = "recovery"
    kind: Literal["stale", "page_change", "loop", "crash"]
    detail: str = ""


Event = Annotated[
    Union[ModelCallEvent, StepEvent, ActionEvent, ObservationEvent, RecoveryEvent],
    Field(discriminator="type"),
]

_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)


def event_to_json(event: Event) -> str:
    """Serialize an event to a single-line JSON string (JSONL-safe)."""
    return _event_adapter.dump_json(event).decode("utf-8")


def event_from_json(text: str) -> Event:
    """Deserialize a JSON line back into a validated :class:`Event`."""
    return _event_adapter.validate_json(text)


# --------------------------------------------------------------------------- #
# EventLog
# --------------------------------------------------------------------------- #


class EventLog(BaseModel):
    """An append-only, optionally file-backed sequence of typed events."""

    events: list[Event] = Field(default_factory=list)
    path: Path | None = Field(default=None, exclude=True)

    def append(self, event: Event) -> None:
        """Append an event in memory and, when file-backed, write + flush one line."""
        self.events.append(event)
        if self.path is not None:
            append_jsonl(self.path, event_to_json(event))

    def replay(self) -> list[Event]:
        """Reconstruct the event stream.

        When file-backed, reads and re-validates the JSONL from disk (tolerating
        corrupt lines); otherwise returns the in-memory events.
        """
        if self.path is not None:
            return [_event_adapter.validate_python(obj) for obj in read_jsonl(self.path)]
        return list(self.events)

    def to_jsonl(self) -> str:
        """Serialize all in-memory events to a JSONL string (one event per line)."""
        return "\n".join(event_to_json(e) for e in self.events)
