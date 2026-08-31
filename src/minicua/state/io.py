"""Durability primitives for the state layer.

The event log, checkpoint, and trajectory recorder all persist to disk, and the
recovery layer reuses the same primitives — so crash safety and corruption
handling live in exactly one place.

* :func:`atomic_write_text` — write via a temp file in the same directory, fsync,
  then ``os.replace`` (atomic on POSIX and Windows). A crash mid-write can never
  leave a half-written file, and the original is preserved until the replacement
  is durable.
* :func:`read_json_or_none` — parse a JSON file, returning ``None`` on a missing
  or corrupt file instead of raising.
* :func:`append_jsonl` — append one JSON line and flush it, so already-committed
  events survive a crash (the "single source of truth" log is never partially
  buffered in userspace).
* :func:`read_jsonl` — read a JSONL file back, skipping blank and corrupt lines
  so a torn write at the tail never makes the whole log unreadable.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("minicua.state.io")


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically write ``text`` to ``path`` (temp file + fsync + rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json_or_none(path: str | Path) -> Any:
    """Parse a JSON file, or return ``None`` if missing / corrupt (never raises)."""
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read %s: %s", target, exc)
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("corrupt JSON in %s: %s", target, exc)
        return None


def append_jsonl(path: str | Path, line: str) -> None:
    """Append ``line`` to a JSONL file, adding a newline, and flush (crash-safe)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(line)
        if not line.endswith("\n"):
            fh.write("\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError as exc:  # some filesystems reject fsync on this descriptor
            logger.warning("fsync unsupported for %s: %s", target, exc)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read all JSON objects from a JSONL file, skipping blank / corrupt lines."""
    target = Path(path)
    if not target.is_file():
        return []
    out: list[dict[str, Any]] = []
    with open(target, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skipping corrupt JSONL line in %s: %s", target, exc)
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out
