import pytest

from minicua.browser.crash_watchdog import CrashWatchdog
from minicua.browser.session import BrowserSession


@pytest.mark.asyncio
async def test_watchdog_emits_on_crash():
    wd = CrashWatchdog()
    events = []
    wd.on_crash = lambda msg: events.append(msg)
    await wd._handle_target_crashed("tab-1")
    assert events == ["target tab-1 crashed"]
    assert wd.crashed is True


def test_watchdog_emits_on_connection_lost():
    wd = CrashWatchdog()
    events = []
    wd.on_crash = lambda msg: events.append(msg)
    wd._on_connection_lost()
    assert events == ["browser connection lost"]
    assert wd.crashed is True


@pytest.mark.asyncio
async def test_watchdog_attached_close_fires_connection_lost():
    s = BrowserSession(headless=True)
    await s.start()
    wd = CrashWatchdog()
    events = []
    wd.on_crash = lambda msg: events.append(msg)
    wd.attach(s.context)
    await s.close()
    assert events == ["browser connection lost"]


@pytest.mark.asyncio
async def test_watchdog_detach_stops_close_events():
    s = BrowserSession(headless=True)
    await s.start()
    wd = CrashWatchdog()
    events = []
    wd.on_crash = lambda msg: events.append(msg)
    wd.attach(s.context)
    wd.detach()
    await s.close()
    assert events == []  # an intentional close must not look like a crash
