class CUAError(Exception): ...


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
