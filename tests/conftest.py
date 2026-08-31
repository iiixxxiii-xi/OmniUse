"""Shared pytest fixtures for minicua."""

import os

# Keep Playwright browsers on D: (see project memory: files live on D:, not C:).
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "D:/playwright-browsers")
