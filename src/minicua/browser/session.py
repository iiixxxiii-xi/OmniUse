from playwright.async_api import async_playwright


class BrowserSession:
    def __init__(self, headless: bool = True, user_data_dir: str | None = None):
        self.headless = headless
        self.user_data_dir = user_data_dir

    async def start(self) -> None:
        self._pw = await async_playwright().start()
        self.context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=self.user_data_dir or "", headless=self.headless
        )
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()

    async def close(self) -> None:
        await self.context.close()
        await self._pw.stop()
