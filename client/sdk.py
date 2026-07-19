from __future__ import annotations

from collections.abc import Iterator


class AnalystClientError(Exception):
    def __init__(self, message: str, status_code=None, request_id=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class DocumentAnalystClient:
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        raise NotImplementedError("Task 3.1: implement the client constructor")

    def ask(self, question: str) -> str:
        raise NotImplementedError("Task 3.1: implement ask()")

    def ask_streaming(self, question: str) -> Iterator[str]:
        raise NotImplementedError("Task 3.1: implement ask_streaming()")

    def health_check(self) -> bool:
        raise NotImplementedError("Task 3.1: implement health_check()")
