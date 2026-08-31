"""Detect browser crashes and connection loss.

Wires Playwright CDP/context events into a single ``on_crash`` callback so the
controller can rebuild the session and recover. The core signal handlers are
plain methods (unit-testable); ``attach`` connects them to a live context.
"""

import logging
from typing import Any, Callable

logger = logging.getLogger("minicua.browser.crash")


class CrashWatchdog:
    def __init__(self) -> None:
        self.on_crash: Callable[[str], None] = lambda msg: None
        self.crashed: bool = False

    async def _handle_target_crashed(self, target_id: str) -> None:
        """Handle a CDP ``Target.targetCrashed`` event."""
        self.crashed = True
        msg = f"target {target_id} crashed"
        logger.error(msg)
        self.on_crash(msg)

    def _on_connection_lost(self) -> None:
        """Handle browser context / CDP connection loss."""
        self.crashed = True
        msg = "browser connection lost"
        logger.error(msg)
        self.on_crash(msg)

    def attach(self, context: Any) -> None:
        """Wire real events from a Playwright context to this watchdog."""
        context.on("close", lambda: self._on_connection_lost())
