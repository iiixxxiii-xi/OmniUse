"""Persistent Playwright browser session with storage_state and a safe lifecycle."""

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext, Page, async_playwright
from pydantic import BaseModel, Field

from minicua.core.errors import BrowserError, BrowserStartupError, NavigationError
from minicua.core.retry import RetryPolicy, async_retry

logger = logging.getLogger("minicua.browser.session")

# Transient network errors worth retrying (vs. permanent ones like invalid URL).
RETRYABLE_NET_ERROR_CODES = frozenset(
    {
        "ERR_CONNECTION_REFUSED",
        "ERR_CONNECTION_RESET",
        "ERR_CONNECTION_ABORTED",
        "ERR_CONNECTION_TIMED_OUT",
        "ERR_TIMED_OUT",
        "ERR_NETWORK_CHANGED",
        "ERR_INTERNET_DISCONNECTED",
        "ERR_NAME_RESOLUTION_FAILED",
        "ERR_PROXY_CONNECTION_FAILED",
    }
)


def is_retryable_navigation_error(exc: Exception) -> bool:
    """Classify a navigation exception as transient (retryable) or permanent."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    msg = str(exc)
    if "Timeout" in msg or "timeout" in msg:
        return True
    return any(code in msg for code in RETRYABLE_NET_ERROR_CODES)


class BrowserSessionConfig(BaseModel):
    """Validated configuration for a :class:`BrowserSession`."""

    headless: bool = True
    user_data_dir: str | None = None
    storage_state: str | Path | None = None
    navigation_timeout_ms: int = Field(default=30_000, ge=0)
    max_navigation_retries: int = Field(default=2, ge=0)
    retry_base_delay_ms: int = Field(default=500, ge=0)
    retry_max_delay_ms: int = Field(default=5_000, ge=0)


class BrowserSession:
    """A persistent Playwright Chromium session.

    Manages a persistent context (so cookies / localStorage / storage_state
    survive across steps) and exposes a small navigation surface. All failure
    paths are classified (retryable vs permanent) and logged.
    """

    def __init__(self, config: BrowserSessionConfig | None = None, **kwargs: Any) -> None:
        self.config = config or BrowserSessionConfig(**kwargs)
        self._pw: Any = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self._owns_user_data_dir = False
        self._user_data_dir: str | None = None

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self.context is not None:
            return  # already started (idempotent)
        logger.info("starting browser session (headless=%s)", self.config.headless)
        self._pw = await async_playwright().start()
        try:
            self.context = await self._pw.chromium.launch_persistent_context(
                user_data_dir=self._resolve_user_data_dir(),
                headless=self.config.headless,
            )
        except Exception as exc:
            await self._stop_playwright()
            self._discard_user_data_dir()
            raise BrowserStartupError(f"failed to launch persistent context: {exc}") from exc
        # ``launch_persistent_context`` does not accept ``storage_state`` (only
        # ``new_context`` does), so restore persisted cookies / localStorage
        # explicitly after the persistent context is up.
        try:
            if self.config.storage_state:
                await self._load_storage_state(self.config.storage_state)
        except Exception:
            await self.close()
            raise
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        logger.info("browser session started")

    async def close(self) -> None:
        if self.context is not None:
            try:
                await self.context.close()
            except Exception:
                logger.exception("error closing browser context")
            finally:
                self.context = None
                self.page = None
        await self._stop_playwright()
        self._discard_user_data_dir()
        logger.info("browser session closed")

    async def __aenter__(self) -> "BrowserSession":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def _stop_playwright(self) -> None:
        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:
                logger.exception("error stopping playwright driver")
            finally:
                self._pw = None

    def _resolve_user_data_dir(self) -> str:
        if self.config.user_data_dir:
            return self.config.user_data_dir
        self._owns_user_data_dir = True
        self._user_data_dir = tempfile.mkdtemp(prefix="minicua-")
        return self._user_data_dir

    def _discard_user_data_dir(self) -> None:
        if self._owns_user_data_dir and self._user_data_dir:
            shutil.rmtree(self._user_data_dir, ignore_errors=True)
        self._user_data_dir = None
        self._owns_user_data_dir = False

    # -- navigation --------------------------------------------------------

    @property
    def url(self) -> str:
        self._require_page()
        return self.page.url

    async def get_url(self) -> str:
        self._require_page()
        return self.page.url

    async def get_title(self) -> str:
        self._require_page()
        return await self.page.title()

    async def navigate(self, url: str) -> None:
        self._require_page()
        policy = RetryPolicy(
            max_attempts=self.config.max_navigation_retries + 1,
            base_delay=self.config.retry_base_delay_ms / 1000.0,
            max_delay=self.config.retry_max_delay_ms / 1000.0,
        )
        logger.info("navigating to %s", url)
        try:
            await async_retry(
                lambda: self.page.goto(url, timeout=self.config.navigation_timeout_ms),
                policy=policy,
                is_retryable=is_retryable_navigation_error,
                logger_=logger,
            )
        except Exception as exc:
            raise NavigationError(
                url=url,
                reason=str(exc),
                retryable=is_retryable_navigation_error(exc),
            ) from exc

    # -- state -------------------------------------------------------------

    async def save_storage_state(self, path: str | Path) -> None:
        self._require_context()
        await self.context.storage_state(path=str(path))
        logger.info("saved storage_state to %s", path)

    async def _load_storage_state(self, path: str | Path) -> None:
        """Restore cookies + localStorage from a Playwright storage_state file.

        ``launch_persistent_context`` cannot consume ``storage_state`` directly,
        so we parse the JSON file and apply it manually.
        """
        state_path = Path(path)
        if not state_path.is_file():
            raise BrowserStartupError(f"storage_state file not found: {state_path}")
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise BrowserStartupError(f"invalid storage_state file {state_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise BrowserStartupError(f"invalid storage_state file {state_path}: expected a JSON object")

        cookies = data.get("cookies") or []
        if cookies:
            await self.context.add_cookies(cookies)
            logger.info("restored %d cookie(s) from storage_state", len(cookies))

        for origin in data.get("origins") or []:
            origin_url = origin.get("origin")
            entries = origin.get("localStorage") or []
            if origin_url and entries:
                self._install_local_storage(origin_url, entries)

    def _install_local_storage(self, origin_url: str, entries: list[dict[str, Any]]) -> None:
        """Inject an init script that replays localStorage keys for an origin."""
        payload = {entry["name"]: entry["value"] for entry in entries}
        script = (
            "if (window.location.origin === " + json.dumps(origin_url) + ") {\n"
            "  const data = " + json.dumps(payload) + ";\n"
            "  for (const [key, value] of Object.entries(data)) {\n"
            "    try { window.localStorage.setItem(key, value); } catch (e) {}\n"
            "  }\n"
            "}"
        )
        self.context.add_init_script(script)
        logger.info("installed localStorage init script for %s (%d keys)", origin_url, len(payload))

    def _require_page(self) -> None:
        if self.page is None:
            raise BrowserError("browser session is not started (call start() first)")

    def _require_context(self) -> None:
        if self.context is None:
            raise BrowserError("browser session is not started (call start() first)")
