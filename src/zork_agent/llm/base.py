"""Abstract interfaces and shared helpers for local LLM backends.

TODO: extend this interface only after a second backend forces a cleaner contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from zork_agent.types import ChatMessage, LLMChatRequest, LLMResponse

if TYPE_CHECKING:
    from zork_agent.llm.prompts import PromptManager


class LLMError(Exception):
    """Base exception for local LLM client errors."""


class LLMConfigurationError(LLMError):
    """Raised when a client is missing required configuration."""


class LLMRequestError(LLMError):
    """Raised when an HTTP request cannot be completed successfully."""


class LLMHTTPError(LLMRequestError):
    """Raised when a backend returns an HTTP error response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        request_url: str,
        response_body: str,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.request_url = request_url
        self.response_body = response_body
        self.retryable = retryable


class LLMResponseFormatError(LLMError):
    """Raised when a backend response does not match the expected schema."""



class BaseLLMClient(ABC):
    """Abstract base for local model clients."""

    def __init__(self, *, default_model: str | None = None):
        # TODO: centralize backend settings if more local providers are added.
        self.default_model = default_model

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return model identifiers exposed by the backend."""

    @abstractmethod
    def complete_chat(self, request: LLMChatRequest) -> LLMResponse:
        """Execute one normalized chat completion request."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        """Execute a chat completion request from an existing message list."""

        request = LLMChatRequest(
            messages=list(messages),
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        return self.complete_chat(request)

    def generate_action_candidates(
        self,
        *,
        prompt_manager: "PromptManager",
        game_id: str,
        observation: str,
        inventory: str,
        score: int,
        moves: int,
        action_candidates: int,
        model: str | None,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        """Render the action proposal prompts and execute a completion."""

        # TODO: move prompt-specific data validation closer to the policy layer if needed.
        return self._complete_from_prompts(
            system_prompt=prompt_manager.render_system(game_id=game_id),
            user_prompt=prompt_manager.render_action_proposal(
                observation=observation,
                inventory=inventory,
                score=score,
                moves=moves,
                action_candidates=action_candidates,
            ),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            metadata={"task": "generate_action_candidates"},
        )

    def summarize_trajectory(
        self,
        *,
        prompt_manager: "PromptManager",
        game_id: str,
        episode_id: str,
        seed: int,
        trajectory_excerpt: str,
        model: str | None,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        """Render the trajectory analysis prompts and execute a completion."""

        return self._complete_from_prompts(
            system_prompt=prompt_manager.render_system(game_id=game_id),
            user_prompt=prompt_manager.render_trajectory_analysis(
                episode_id=episode_id,
                seed=seed,
                trajectory_excerpt=trajectory_excerpt,
            ),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            metadata={"task": "summarize_trajectory"},
        )

    def reflect_on_rollouts(
        self,
        *,
        prompt_manager: "PromptManager",
        game_id: str,
        state_summary: str,
        recent_actions: str,
        model: str | None,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        """Render the rollout reflection prompts and execute a completion."""

        return self._complete_from_prompts(
            system_prompt=prompt_manager.render_system(game_id=game_id),
            user_prompt=prompt_manager.render_local_reflection(
                state_summary=state_summary,
                recent_actions=recent_actions,
            ),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            metadata={"task": "reflect_on_rollouts"},
        )

    def score_state_for_revisit(
        self,
        *,
        prompt_manager: "PromptManager",
        game_id: str,
        frontier_snapshot: str,
        model: str | None,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        """Render the state-selection prompts and execute a completion."""

        return self._complete_from_prompts(
            system_prompt=prompt_manager.render_system(game_id=game_id),
            user_prompt=prompt_manager.render_state_selection(frontier_snapshot=frontier_snapshot),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            metadata={"task": "score_state_for_revisit"},
        )

    def _complete_from_prompts(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        model: str | None,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None,
        metadata: dict[str, str],
    ) -> LLMResponse:
        """Build a normalized request from prompts and execute it."""

        request = LLMChatRequest.from_prompts(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            metadata=metadata,
        )
        return self.complete_chat(request)
