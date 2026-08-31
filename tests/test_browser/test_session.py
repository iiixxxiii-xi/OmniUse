import pytest

from minicua.browser.session import BrowserSession


@pytest.mark.asyncio
async def test_session_start_and_close():
    s = BrowserSession(headless=True)
    await s.start()
    assert s.context is not None
    await s.close()


@pytest.mark.asyncio
async def test_storage_state_roundtrip(tmp_path):
    s = BrowserSession(headless=True)
    await s.start()
    await s.context.add_cookies([{"name": "k", "value": "v", "url": "https://example.com"}])
    path = tmp_path / "state.json"
    await s.save_storage_state(path)
    await s.close()

    s2 = BrowserSession(headless=True, storage_state=path)
    await s2.start()
    cookies = await s2.context.cookies("https://example.com")
    assert any(c["name"] == "k" and c["value"] == "v" for c in cookies)
    await s2.close()


@pytest.mark.asyncio
async def test_navigate_and_read_state():
    s = BrowserSession(headless=True)
    await s.start()
    await s.navigate("data:text/html,<title>MyTitle</title>")
    assert s.url.startswith("data:text/html")
    assert await s.get_title() == "MyTitle"
    await s.close()
