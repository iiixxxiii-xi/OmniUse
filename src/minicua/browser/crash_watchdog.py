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
        self._context: Any = None
        self._handler: Callable[[], None] | None = None

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
        self._context = context
        self._handler = self._on_connection_lost
        context.on("close", self._handler)

    def detach(self) -> None:
        """Stop listening to the attached context (an intentional close is not a crash)."""
        if self._context is not None and self._handler is not None:
            try:
                self._context.remove_listener("close", self._handler)
            except Exception:  # noqa: BLE001 - context may already be gone
                pass
        self._context = None
        self._handler = None
