"""Task 5.4: crash recovery — rebuild a session from storage_state + checkpoint.

:func:`minicua.recovery.crash.recover` closes the (possibly crashed) session,
repoints it at the saved ``storage_state``, restarts it, and restores the task
state from a checkpoint file. This is the last rung of the recovery ladder: when
the browser itself dies, we rebuild and keep going from where we left off.
"""

import pytest

from minicua.recovery.crash import (
    RecoveryCheckpoint,
    load_checkpoint,
    recover,
    save_checkpoint,
)


# --------------------------------------------------------------------------- #
# checkpoint save/load
# --------------------------------------------------------------------------- #


def test_checkpoint_roundtrip(tmp_path):
    save_checkpoint(tmp_path, RecoveryCheckpoint(task="book a flight", step=7))
    loaded = load_checkpoint(tmp_path)
    assert loaded is not None
    assert loaded.task == "book a flight"
    assert loaded.step == 7


def test_load_checkpoint_missing_returns_none(tmp_path):
    assert load_checkpoint(tmp_path) is None


def test_load_checkpoint_corrupt_returns_none(tmp_path):
    (tmp_path / "checkpoint.json").write_text("{not valid json", encoding="utf-8")
    assert load_checkpoint(tmp_path) is None


# --------------------------------------------------------------------------- #
# recover
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recover_restarts_session(session, tmp_path):
    await session.save_storage_state(tmp_path / "state.json")
    result = await recover(session, checkpoint_dir=tmp_path)
    assert session.page is not None
    assert result.restarted is True
    assert result.storage_state_loaded is True


@pytest.mark.asyncio
async def test_recover_restores_checkpoint(session, tmp_path):
    await session.save_storage_state(tmp_path / "state.json")
    save_checkpoint(tmp_path, RecoveryCheckpoint(task="click ok", step=3))
    result = await recover(session, checkpoint_dir=tmp_path)
    assert result.checkpoint is not None
    assert result.checkpoint.task == "click ok"
    assert result.checkpoint.step == 3


@pytest.mark.asyncio
async def test_recover_without_any_checkpoint(session, tmp_path):
    result = await recover(session, checkpoint_dir=tmp_path)
    assert session.page is not None
    assert result.storage_state_loaded is False
    assert result.checkpoint is None


@pytest.mark.asyncio
async def test_recover_after_session_already_closed(session, tmp_path):
    # Simulate a crash aftermath: the context was torn down before recover runs.
    await session.save_storage_state(tmp_path / "state.json")
    await session.close()
    assert session.page is None
    result = await recover(session, checkpoint_dir=tmp_path)
    assert session.page is not None
    assert result.restarted is True
