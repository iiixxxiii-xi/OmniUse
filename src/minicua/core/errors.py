"""Error taxonomy for minicua.

Every error carries a `retryable` flag: callers use it to decide whether an
operation is worth retrying (transient) or must be surfaced / escalated
(permanent). Subclasses that represent transient conditions set it to True.
"""


class CUAError(Exception):
    """Base class for all minicua errors. Not retryable by default."""

    retryable: bool = False


class StaleElementError(CUAError):
    def __init__(self, index: int | None = None):
        self.index = index
        super().__init__(f"Element index {index} is stale")


class PageChangedError(CUAError):
    def __init__(self, before: str, after: str):
        super().__init__(f"Page changed from {before} to {after}")


class CrashError(CUAError): ...


class LoopDetected(CUAError):
    def __init__(self, repeat_count: int):
        self.repeat_count = repeat_count
        super().__init__(f"Loop detected: {repeat_count} repeats")


# --- Browser-session errors -------------------------------------------------


class BrowserError(CUAError):
    """Base class for browser-session failures."""


class BrowserStartupError(BrowserError):
    """Browser / Playwright driver failed to start."""


class NavigationError(BrowserError):
    """Navigation failed. `retryable` reflects whether a retry may succeed."""

    def __init__(self, url: str, reason: str, retryable: bool = False):
        self.url = url
        self.reason = reason
        self.retryable = retryable
        super().__init__(f"Navigation to {url!r} failed: {reason}")


class BrowserConnectionError(BrowserError):
    """CDP / browser connection was lost. Transient."""

    retryable = True

    def __init__(self, message: str = "browser connection lost"):
        super().__init__(message)


class BrowserTimeoutError(BrowserError):
    """A CDP call timed out. Transient."""

    retryable = True

    def __init__(self, message: str = "browser operation timed out"):
        super().__init__(message)
