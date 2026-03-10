"""LM Studio client that targets the OpenAI-compatible local API.

TODO: add streaming support if the agent loop begins to benefit from incremental text.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
import os
import time
from typing import Any, Mapping

import httpx

from zork_agent.llm.base import (
    BaseLLMClient,
    LLMConfigurationError,
    LLMHTTPError,
    LLMRequestError,
    LLMResponseFormatError,
)
from zork_agent.types import LLMChatRequest, LLMResponse, TokenUsage

TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LMStudioClient(BaseLLMClient):
    """Small wrapper around LM Studio's OpenAI-compatible endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        default_model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        retry_backoff_seconds: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ):
        # TODO: inject a shared client only if profiling shows connection setup overhead matters.
        resolved_base_url = _normalize_base_url(base_url or os.getenv("LMSTUDIO_BASE_URL", ""))
        if not resolved_base_url:
            raise LLMConfigurationError(
                "LM Studio base URL is required. Set it in config or LMSTUDIO_BASE_URL."
            )

        super().__init__(default_model=default_model or os.getenv("LMSTUDIO_MODEL_NAME"))
        self.base_url = resolved_base_url
        self.api_key = api_key or os.getenv("LMSTUDIO_API_KEY", "lm-studio")
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else _env_float(
            "LMSTUDIO_TIMEOUT_SECONDS",
            60.0,
        )
        self.max_retries = max_retries if max_retries is not None else _env_int("LMSTUDIO_MAX_RETRIES", 2)
        if self.max_retries < 0:
            raise LLMConfigurationError("LM Studio max_retries must be >= 0.")
        if retry_backoff_seconds < 0:
            raise LLMConfigurationError("LM Studio retry_backoff_seconds must be >= 0.")
        self.retry_backoff_seconds = retry_backoff_seconds
        self.transport = transport

    def list_models(self) -> list[str]:
        """Fetch model ids from `GET /v1/models`."""

        payload = self._request_json(method="GET", path="/models")
        data = payload.get("data")
        if not isinstance(data, list):
            raise LLMResponseFormatError("LM Studio `/models` response did not include a `data` list.")
        return [str(item["id"]) for item in data if isinstance(item, MappingABC) and "id" in item]

    def complete_chat(self, request: LLMChatRequest) -> LLMResponse:
        """Call `POST /v1/chat/completions` and normalize the response."""

        model_name = request.model or self.default_model
        if not model_name:
            raise LLMConfigurationError(
                "No model was provided for the LM Studio request. Set config.llm.model_name or "
                "LMSTUDIO_MODEL_NAME."
            )

        payload = {
            "model": model_name,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        started_at = time.monotonic()
        raw = self._request_json(
            method="POST",
            path="/chat/completions",
            json_payload=payload,
            timeout_seconds=request.timeout_seconds,
        )
        latency_seconds = time.monotonic() - started_at
        return self._parse_chat_completion_payload(raw, fallback_model=model_name, latency_seconds=latency_seconds)

    def _headers(self) -> dict[str, str]:
        """Construct request headers."""

        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _client(self, timeout_seconds: float) -> httpx.Client:
        """Construct a short-lived HTTP client."""

        return httpx.Client(timeout=timeout_seconds, transport=self.transport)

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        json_payload: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Execute an HTTP request with retries and return the parsed JSON payload."""

        request_url = f"{self.base_url}{path}"
        request_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        attempts = self.max_retries + 1

        for attempt in range(attempts):
            try:
                with self._client(request_timeout) as client:
                    response = client.request(
                        method,
                        request_url,
                        headers=self._headers(),
                        json=json_payload,
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self.max_retries:
                    raise LLMRequestError(
                        f"LM Studio {method} {request_url} failed after {attempts} attempt(s): {exc}"
                    ) from exc
                self._sleep_before_retry(attempt)
                continue

            if response.status_code >= 400:
                retryable = response.status_code in TRANSIENT_STATUS_CODES
                backend_message = self._extract_error_message(response)
                error = LLMHTTPError(
                    (
                        f"LM Studio returned HTTP {response.status_code} for {request_url}."
                        + (f" Detail: {backend_message}" if backend_message else "")
                    ),
                    status_code=response.status_code,
                    request_url=request_url,
                    response_body=response.text,
                    retryable=retryable,
                )
                if retryable and attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise error

            try:
                payload = response.json()
            except ValueError as exc:
                raise LLMResponseFormatError("LM Studio returned invalid JSON.") from exc

            if not isinstance(payload, dict):
                raise LLMResponseFormatError("LM Studio response payload must be a JSON object.")
            return payload

        raise LLMRequestError(f"LM Studio request exhausted retries for {request_url}.")

    def _parse_chat_completion_payload(
        self,
        payload: dict[str, Any],
        *,
        fallback_model: str,
        latency_seconds: float,
    ) -> LLMResponse:
        """Parse a chat completion response into the shared response type."""

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseFormatError("LM Studio chat completion payload did not include any choices.")

        first_choice = choices[0]
        if not isinstance(first_choice, MappingABC):
            raise LLMResponseFormatError("LM Studio choice entry must be a JSON object.")

        message = first_choice.get("message")
        content: Any | None = None
        if isinstance(message, MappingABC):
            content = message.get("content")
        elif "text" in first_choice:
            content = first_choice.get("text")

        text = self._extract_text(content)
        model_name = str(payload.get("model") or fallback_model)
        usage = TokenUsage.from_payload(payload.get("usage"))
        return LLMResponse(
            text=text,
            model=model_name,
            latency_seconds=latency_seconds,
            usage=usage,
            raw_payload=payload,
        )

    def _extract_text(self, content: Any) -> str:
        """Normalize content returned by the OpenAI-compatible response shape."""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, MappingABC):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif item.get("type") == "output_text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif item.get("type") == "text" and isinstance(item.get("content"), str):
                        parts.append(item["content"])
            if parts:
                return "\n".join(parts)

        raise LLMResponseFormatError("LM Studio chat completion payload did not include text content.")

    def _extract_error_message(self, response: httpx.Response) -> str:
        """Extract a compact backend error string from an error response."""

        try:
            payload = response.json()
        except ValueError:
            return response.text.strip()[:200]
        if not isinstance(payload, MappingABC):
            return response.text.strip()[:200]
        error_payload = payload.get("error")
        if isinstance(error_payload, MappingABC):
            message = error_payload.get("message")
            if isinstance(message, str):
                return " ".join(message.split()).strip()
        if isinstance(payload.get("message"), str):
            return " ".join(str(payload["message"]).split()).strip()
        return response.text.strip()[:200]

    def _sleep_before_retry(self, attempt: int) -> None:
        """Sleep using a small exponential backoff before the next retry."""

        if self.retry_backoff_seconds <= 0:
            return
        time.sleep(self.retry_backoff_seconds * (2**attempt))


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment with a fallback default."""

    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise LLMConfigurationError(f"Environment variable {name} must be a float.") from exc


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment with a fallback default."""

    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise LLMConfigurationError(f"Environment variable {name} must be an integer.") from exc


def _normalize_base_url(base_url: str) -> str:
    """Normalize an LM Studio base URL, appending `/v1` when only the host is supplied."""

    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return ""
    if not normalized.startswith(("http://", "https://")):
        raise LLMConfigurationError("LM Studio base URL must start with http:// or https://.")
    if normalized.endswith("/v1"):
        return normalized
    if normalized.count("/") <= 2:
        return f"{normalized}/v1"
    return normalized
