"""Shared pytest fixtures for minicua."""

import os

import pytest

from minicua.browser.session import BrowserSession

# Keep Playwright browsers on D: (see project memory: files live on D:, not C:).
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "D:/playwright-browsers")


@pytest.fixture
async def session():
    """A fresh, isolated headless browser session per test.

    Function-scoped so each test starts from a clean page / context (no cookie
    or localStorage leakage between tests). Cleanup is guaranteed via finally.
    """
    s = BrowserSession(headless=True)
    await s.start()
    try:
        yield s
    finally:
        await s.close()
