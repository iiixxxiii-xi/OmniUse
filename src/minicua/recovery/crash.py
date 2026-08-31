"""Crash recovery: rebuild a browser session and restore task state.

When the browser crashes or the CDP connection is lost, the controller calls
:func:`recover`: close the dead session, repoint it at the last saved
``storage_state`` (so cookies / localStorage survive), restart it, and reload the
task checkpoint (task goal + step count). The result is a fresh, live page with
the agent's progress intact — no re-login, no re-planning from zero.

Checkpoints are small JSON files; ``storage_state`` is a Playwright
``context.storage_state`` export. Both live under a caller-supplied directory so
the controller can save them every step and recover to the most recent one.
"""

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from minicua.browser.session import BrowserSession

logger = logging.getLogger("minicua.recovery.crash")

STORAGE_STATE_FILENAME = "state.json"
CHECKPOINT_FILENAME = "checkpoint.json"


class RecoveryCheckpoint(BaseModel):
    """The minimal task state needed to resume after a crash."""

    task: str = ""
    step: int = Field(default=0, ge=0)


class RecoveryResult(BaseModel):
    """What :func:`recover` did, for observability."""

    restarted: bool
    storage_state_loaded: bool
    checkpoint: RecoveryCheckpoint | None = None


def save_checkpoint(checkpoint_dir: str | Path, checkpoint: RecoveryCheckpoint) -> None:
    """Persist ``checkpoint`` to ``checkpoint_dir / CHECKPOINT_FILENAME``."""
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CHECKPOINT_FILENAME
    path.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
    logger.info("saved recovery checkpoint to %s (step=%d)", path, checkpoint.step)


def load_checkpoint(checkpoint_dir: str | Path) -> RecoveryCheckpoint | None:
    """Load a checkpoint, or ``None`` if it is missing / corrupt (never raises)."""
    path = Path(checkpoint_dir) / CHECKPOINT_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return RecoveryCheckpoint.model_validate(data)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("could not load recovery checkpoint %s: %s", path, exc)
        return None


async def recover(session: BrowserSession, checkpoint_dir: str | Path) -> RecoveryResult:
    """Rebuild a (possibly crashed) ``session`` from ``checkpoint_dir``.

    Closes the existing session (idempotent — safe even if the context already
    died), points it at the saved ``storage_state`` if present, restarts it, and
    restores the checkpoint. Returns a :class:`RecoveryResult`.
    """
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)

    state_path = directory / STORAGE_STATE_FILENAME
    storage_state_loaded = state_path.is_file()
    if storage_state_loaded:
        session.config.storage_state = str(state_path)

    logger.info("recovering session (storage_state=%s)", storage_state_loaded)
    await session.close()
    await session.start()

    checkpoint = load_checkpoint(directory)
    return RecoveryResult(
        restarted=True,
        storage_state_loaded=storage_state_loaded,
        checkpoint=checkpoint,
    )
