"""Shared helpers for the CLI commands."""

import json
from pathlib import Path


def load_script(path: str | Path) -> list:
    """Load scripted :class:`FakeModel` responses from a JSON file.

    Accepts a JSON list of response dicts (``[{"name": "click", "params":
    {"index": 1}}, ...]``) or a single response object. Raises ``OSError`` /
    ``ValueError`` on a missing / malformed file; callers turn those into a clean
    CLI error + exit code.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("script file must be a JSON list (or single object) of responses")
