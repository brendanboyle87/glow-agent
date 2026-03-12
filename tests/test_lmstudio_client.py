"""Tests for the LM Studio OpenAI-compatible client.

TODO: add streamed response tests if streaming is implemented later.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from zork_agent.config import PromptConfig
from zork_agent.llm.base import LLMHTTPError, LLMResponseFormatError
from zork_agent.llm.lmstudio_client import LMStudioClient
from zork_agent.llm.prompts import PromptManager
from zork_agent.types import LLMChatRequest


def _build_prompt_manager(tmp_path: Path) -> PromptManager:
    """Create a small prompt manager for LLM helper tests."""

    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "system.txt").write_text("System for {game_id}", encoding="utf-8")
    (prompt_dir / "action.txt").write_text(
        "Observation: {observation}\nCandidates: {action_candidates}",
        encoding="utf-8",
    )
    (prompt_dir / "trajectory.txt").write_text(
        "Episode {episode_id}\nSeed {seed}\n{trajectory_excerpt}",
        encoding="utf-8",
    )
    (prompt_dir / "reflection.txt").write_text(
        "State: {state_summary}\nActions: {recent_actions}",
        encoding="utf-8",
    )
    (prompt_dir / "selection.txt").write_text("Frontier: {frontier_snapshot}", encoding="utf-8")
    config = PromptConfig(
        directory=prompt_dir,
        system_file="system.txt",
        action_proposal_file="action.txt",
        trajectory_analysis_file="trajectory.txt",
        local_reflection_file="reflection.txt",
        state_selection_file="selection.txt",
    )
    return PromptManager(config)


def test_lmstudio_client_parses_chat_completion_response() -> None:
    """The client should return text, model metadata, usage, and raw payload."""

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "model": "qwen/test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "look\nopen mailbox"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 7,
                    "total_tokens": 19,
                },
            },
        )
    )
    client = LMStudioClient(
        base_url="http://localhost:1234/v1",
        default_model="qwen/test",
        retry_backoff_seconds=0.0,
        transport=transport,
    )

    response = client.complete_chat(
        LLMChatRequest.from_prompts(
            system_prompt="You are helpful.",
            user_prompt="Introduce yourself.",
            model=None,
            temperature=0.1,
            max_tokens=64,
        )
    )

    assert response.text == "look\nopen mailbox"
    assert response.model == "qwen/test"
    assert response.latency_seconds >= 0.0
    assert response.usage is not None
    assert response.usage.total_tokens == 19
    assert response.raw_payload["id"] == "chatcmpl-123"


def test_lmstudio_client_includes_response_format_in_payload() -> None:
    """Structured-output requests should pass response_format through to LM Studio."""

    captured_request: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "model": "qwen/test",
                "choices": [{"message": {"role": "assistant", "content": "{\"ok\":true}"}}],
            },
        )

    client = LMStudioClient(
        base_url="http://localhost:1234/v1",
        default_model="qwen/test",
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )

    client.complete_chat(
        LLMChatRequest.from_prompts(
            system_prompt="System",
            user_prompt="Return JSON.",
            model=None,
            temperature=0.0,
            max_tokens=64,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "simple",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                    },
                },
            },
        )
    )

    payload = captured_request["payload"]
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["name"] == "simple"


def test_generate_action_candidates_renders_prompts_and_uses_env_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Helper methods should render prompts and fall back to env-provided defaults."""

    captured_request: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "model": "env-model",
                "choices": [{"message": {"role": "assistant", "content": "look\ninventory"}}],
            },
        )

    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LMSTUDIO_MODEL_NAME", "env-model")
    manager = _build_prompt_manager(tmp_path)
    client = LMStudioClient(retry_backoff_seconds=0.0, transport=httpx.MockTransport(handler))

    response = client.generate_action_candidates(
        prompt_manager=manager,
        game_id="zork1",
        observation="You are west of the house.",
        inventory="Empty.",
        score=0,
        moves=0,
        action_candidates=3,
        model=None,
        temperature=0.2,
        max_tokens=80,
    )

    payload = captured_request["payload"]

    assert response.model == "env-model"
    assert response.text == "look\ninventory"
    assert captured_request["url"] == "http://localhost:1234/v1/chat/completions"
    assert payload["model"] == "env-model"
    assert payload["messages"][0]["content"] == "System for zork1"
    assert "Candidates: 3" in payload["messages"][1]["content"]


def test_lmstudio_client_retries_transient_http_errors() -> None:
    """Transient HTTP errors should be retried before succeeding."""

    call_count = {"value": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        call_count["value"] += 1
        if call_count["value"] == 1:
            return httpx.Response(503, json={"error": {"message": "server busy"}})
        return httpx.Response(
            200,
            json={
                "model": "retry-model",
                "choices": [{"message": {"role": "assistant", "content": "retry success"}}],
            },
        )

    client = LMStudioClient(
        base_url="http://localhost:1234/v1",
        default_model="retry-model",
        max_retries=1,
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(handler),
    )

    response = client.complete_chat(
        LLMChatRequest.from_prompts(
            system_prompt="System",
            user_prompt="Hello",
            model=None,
            temperature=0.0,
            max_tokens=32,
        )
    )

    assert call_count["value"] == 2
    assert response.text == "retry success"


def test_lmstudio_client_raises_http_error_for_non_retryable_failure() -> None:
    """Non-retryable HTTP failures should raise a useful exception."""

    client = LMStudioClient(
        base_url="http://localhost:1234/v1",
        default_model="broken-model",
        max_retries=0,
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(400, json={"error": {"message": "bad request"}})
        ),
    )

    with pytest.raises(LLMHTTPError) as exc_info:
        client.complete_chat(
            LLMChatRequest.from_prompts(
                system_prompt="System",
                user_prompt="Hello",
                model=None,
                temperature=0.0,
                max_tokens=32,
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.retryable is False
    assert "HTTP 400" in str(exc_info.value)
    assert "bad request" in str(exc_info.value)


def test_lmstudio_client_raises_format_error_for_bad_payload() -> None:
    """Malformed payloads should raise a response-format exception."""

    client = LMStudioClient(
        base_url="http://localhost:1234/v1",
        default_model="bad-payload",
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"choices": []})),
    )

    with pytest.raises(LLMResponseFormatError):
        client.complete_chat(
            LLMChatRequest.from_prompts(
                system_prompt="System",
                user_prompt="Hello",
                model=None,
                temperature=0.0,
                max_tokens=32,
            )
        )


def test_lmstudio_client_normalizes_host_only_base_url() -> None:
    """Supplying only the host/port should normalize to the OpenAI-compatible `/v1` base path."""

    client = LMStudioClient(
        base_url="http://localhost:1234",
        default_model="normalized-model",
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": [{"id": "normalized-model"}]})),
    )

    assert client.base_url == "http://localhost:1234/v1"
    assert client.list_models() == ["normalized-model"]


def test_lmstudio_client_uses_backend_error_message_for_plain_text_body() -> None:
    """HTTP errors should surface compact backend detail even when the body is plain text."""

    client = LMStudioClient(
        base_url="http://localhost:1234",
        default_model="broken-model",
        max_retries=0,
        retry_backoff_seconds=0.0,
        transport=httpx.MockTransport(lambda _: httpx.Response(500, text="server exploded")),
    )

    with pytest.raises(LLMHTTPError, match="server exploded"):
        client.list_models()
