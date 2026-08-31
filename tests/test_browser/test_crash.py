import pytest

from minicua.browser.crash_watchdog import CrashWatchdog


@pytest.mark.asyncio
async def test_watchdog_emits_on_crash():
    wd = CrashWatchdog()
    events = []
    wd.on_crash = lambda msg: events.append(msg)
    await wd._handle_target_crashed("tab-1")
    assert events == ["target tab-1 crashed"]
    assert wd.crashed is True
