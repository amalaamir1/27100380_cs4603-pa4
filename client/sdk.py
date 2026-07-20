from __future__ import annotations

import json
import os
import time
import requests
from collections.abc import Iterator
from typing import Any

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
        if not endpoint_name or not endpoint_name.strip():
            raise ValueError("endpoint_name must be a non-empty string")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        resolved_host = host or os.environ.get("DATABRICKS_HOST")
        resolved_token = token or os.environ.get("DATABRICKS_TOKEN")

        if not resolved_host:
            raise ValueError(
                "Missing Databricks host. Pass host=... or set DATABRICKS_HOST."
            )
        if not resolved_token:
            raise ValueError(
                "Missing Databricks token. Pass token=... or set DATABRICKS_TOKEN."
            )

        self.endpoint_name = endpoint_name.strip()
        self.host = resolved_host.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)

        self._invocation_url = (
            f"{self.host}/serving-endpoints/"
            f"{self.endpoint_name}/invocations"
        )
        self._status_url = (
            f"{self.host}/api/2.0/serving-endpoints/{self.endpoint_name}"
        )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {resolved_token}",
                "Content-Type": "application/json",
            }
        )

    def ask(self, question: str) -> str:
        """Send one question and return the final answer text."""
        self._validate_question(question)

        response = self._request(
            "POST",
            self._invocation_url,
            json_body=self._payload(question),
        )

        try:
            data = response.json()
        except ValueError as exc:
            raise AnalystClientError(
                "The endpoint returned invalid JSON.",
                status_code=response.status_code,
                request_id=self._request_id(response),
            ) from exc

        answer = self._extract_answer(data)
        if not answer:
            raise AnalystClientError(
                "The endpoint response did not contain an assistant answer.",
                status_code=response.status_code,
                request_id=self._request_id(response),
            )
        return answer


    def ask_streaming(self, question: str) -> Iterator[str]:
        """Yield SSE text deltas, or the complete answer once if not streamed."""
        self._validate_question(question)

        response = self._request(
            "POST",
            self._invocation_url,
            json_body=self._payload(question),
            stream=True,
            extra_headers={"Accept": "text/event-stream"},
        )

        content_type = response.headers.get("Content-Type", "").lower()

        # A models-from-code LangChain endpoint commonly returns one normal JSON
        # completion even when the caller requests SSE. That is a valid stream
        # outcome: yield the complete answer as a single chunk.
        if "text/event-stream" not in content_type:
            try:
                data = response.json()
            except ValueError as exc:
                raise AnalystClientError(
                    "The endpoint returned neither SSE nor valid JSON.",
                    status_code=response.status_code,
                    request_id=self._request_id(response),
                ) from exc

            answer = self._extract_answer(data)
            if not answer:
                raise AnalystClientError(
                    "The endpoint response did not contain an assistant answer.",
                    status_code=response.status_code,
                    request_id=self._request_id(response),
                )
            yield answer
            return

        yielded = False
        fallback_answer = ""
        stream_started = time.perf_counter()

        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue

                raw_event = line[len("data:") :].strip()
                if not raw_event or raw_event == "[DONE]":
                    continue

                try:
                    event: Any = json.loads(raw_event)
                except json.JSONDecodeError:
                    yielded = True
                    yield raw_event
                    continue

                chunk = self._extract_stream_chunk(event)
                if chunk:
                    yielded = True
                    yield chunk
                    continue

                complete = self._extract_answer(event)
                if complete:
                    fallback_answer = complete
        except requests.Timeout as exc:
            elapsed = time.perf_counter() - stream_started
            raise TimeoutError(
                f"Streaming request to endpoint '{self.endpoint_name}' "
                f"timed out after {elapsed:.2f} seconds "
                f"(configured timeout: {self.timeout:.2f}s)."
            ) from exc
        except requests.RequestException as exc:
            raise AnalystClientError(
                f"Streaming connection failed: {exc}",
                request_id=self._request_id(response),
            ) from exc
        finally:
            response.close()

        if not yielded and fallback_answer:
            yield fallback_answer
        elif not yielded:
            raise AnalystClientError(
                "The endpoint stream ended without an answer.",
                status_code=response.status_code,
                request_id=self._request_id(response),
            )

    def health_check(self) -> bool:
        """Return True only when the serving endpoint reports READY."""
        try:
            response = self._request("GET", self._status_url)
            state = response.json().get("state", {})
            return state.get("ready") == "READY"
        except (AnalystClientError, TimeoutError, ValueError):
            return False

    @staticmethod
    def _validate_question(question: str) -> None:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

    @staticmethod
    def _payload(question: str) -> dict[str, list[dict[str, str]]]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": question.strip(),
                }
            ]
        }

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        stream: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> requests.Response:
        started = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.request(
                    method,
                    url,
                    json=json_body,
                    headers=extra_headers,
                    timeout=self.timeout,
                    stream=stream,
                )
            except requests.Timeout as exc:
                elapsed = time.perf_counter() - started
                raise TimeoutError(
                    f"Request to endpoint '{self.endpoint_name}' timed out "
                    f"after {elapsed:.2f} seconds "
                    f"(configured timeout: {self.timeout:.2f}s)."
                ) from exc
            except requests.RequestException as exc:
                raise AnalystClientError(
                    f"Could not reach endpoint '{self.endpoint_name}': {exc}"
                ) from exc

            if response.status_code not in self._RETRYABLE_STATUS_CODES:
                break

            if attempt >= self.max_retries:
                break

            delay = self._retry_delay(response, attempt)
            response.close()
            time.sleep(delay)

        if not response.ok:
            self._raise_http_error(response)

        return response

    @staticmethod
    def _retry_delay(response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return float(2**attempt)

    def _raise_http_error(self, response: requests.Response) -> None:
        request_id = self._request_id(response)
        message = self._error_message(response)
        response.close()
        raise AnalystClientError(
            message,
            status_code=response.status_code,
            request_id=request_id,
        )

    @staticmethod
    def _request_id(response: requests.Response) -> str | None:
        for header in (
            "x-databricks-request-id",
            "x-request-id",
            "request-id",
        ):
            value = response.headers.get(header)
            if value:
                return value

        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            value = body.get("request_id") or body.get("requestId")
            if value:
                return str(value)
        return None

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = None

        if isinstance(body, dict):
            for key in ("message", "error", "error_code"):
                value = body.get(key)
                if value:
                    return str(value)

        text = response.text.strip()
        if text:
            return text[:1000]
        return f"Databricks endpoint returned HTTP {response.status_code}."

    @classmethod
    def _extract_answer(cls, data: Any) -> str:
        if isinstance(data, list):
            if not data:
                return ""
            data = data[0]

        if not isinstance(data, dict):
            return ""

        predictions = data.get("predictions")
        if predictions is not None:
            return cls._extract_answer(predictions)

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                return cls._content_text(message.get("content"))

        messages = data.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                if isinstance(last.get("kwargs"), dict):
                    last = last["kwargs"]
                return cls._content_text(last.get("content"))
            return cls._content_text(getattr(last, "content", ""))

        return cls._content_text(data.get("final_answer"))

    @classmethod
    def _extract_stream_chunk(cls, event: Any) -> str:
        if not isinstance(event, dict):
            return ""

        choices = event.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta", {})
            if isinstance(delta, dict):
                return cls._content_text(delta.get("content"))

        for key in ("delta", "text", "token"):
            value = event.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                text = cls._content_text(value.get("content"))
                if text:
                    return text
        return ""

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces: list[str] = []
            for item in content:
                if isinstance(item, str):
                    pieces.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    pieces.append(item["text"])
            return "".join(pieces)
        return ""

    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
