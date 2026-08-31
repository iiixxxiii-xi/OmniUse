"""Conversational browser mode: run a natural-language instruction in the browser.

The only public surface is :class:`ChatRunner` (which composes the existing
:class:`~minicua.controller.agent.Agent` and :class:`~minicua.browser.session.BrowserSession`)
and its :class:`ChatRun` result.
"""

from minicua.chat.runner import ChatAction, ChatRun, ChatRunner, build_summary

__all__ = ["ChatAction", "ChatRun", "ChatRunner", "build_summary"]
