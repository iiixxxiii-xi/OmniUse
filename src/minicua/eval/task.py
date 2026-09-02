"""Task definitions: the declarative JSON a browser task is written in.

A task is pure data — ``id`` + ``instruction`` + a page setup (``initial_url``
or an inline ``html`` fixture) + a declarative :class:`EvaluatorSpec`. Writing a
new task never touches code; adding one JSON file to the tasks directory is
enough. :func:`load_tasks` reads a directory (or single file) of task JSONs,
validating each into a :class:`TaskDef` and — by default — *skipping* corrupt
files with a warning so one bad file never sinks an entire suite. ``strict=True``
turns that skip into a :class:`TaskDefinitionError`.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from minicua.eval.errors import TaskDefinitionError
from minicua.eval.evaluator import EvaluatorSpec

logger = logging.getLogger("minicua.eval.task")


class TaskDef(BaseModel):
    """A single declarative browser task."""

    id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    initial_url: str | None = None
    html: str | None = None  # inline HTML fixture (page.set_content)
    evaluator: EvaluatorSpec
    vision_required: bool = False  # True when the answer lives only in the screenshot
    max_steps: int = Field(default=20, ge=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


def load_task_file(path: str | Path) -> TaskDef:
    """Load and validate one task JSON file, raising :class:`TaskDefinitionError` on any problem."""
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise TaskDefinitionError(f"could not read task file {target}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TaskDefinitionError(f"invalid JSON in task file {target}: {exc}") from exc
    if not isinstance(data, dict):
        raise TaskDefinitionError(f"task file {target} must contain a JSON object")
    try:
        return TaskDef.model_validate(data)
    except ValidationError as exc:
        raise TaskDefinitionError(f"invalid task definition in {target}: {exc}") from exc


def load_tasks(path: str | Path, *, strict: bool = False) -> list[TaskDef]:
    """Load task JSON(s) from a directory (``*.json``) or a single file.

    ``strict=True`` raises :class:`TaskDefinitionError` on the first invalid
    file; the default skips invalid files with a warning so a whole suite still
    runs. A missing path always raises.
    """
    target = Path(path)
    if target.is_dir():
        tasks: list[TaskDef] = []
        for file in sorted(target.glob("*.json")):
            try:
                tasks.append(load_task_file(file))
            except TaskDefinitionError as exc:
                if strict:
                    raise
                logger.warning("skipping invalid task file %s: %s", file, exc)
        return tasks
    if target.is_file():
        return [load_task_file(target)]
    raise TaskDefinitionError(f"task path not found: {target}")
