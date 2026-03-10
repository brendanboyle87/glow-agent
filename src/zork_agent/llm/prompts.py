"""Prompt template loading and light rendering helpers.

TODO: add richer template validation if prompt complexity grows past plain format strings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zork_agent.config import PromptConfig


class _SafeFormatDict(dict[str, Any]):
    """Dictionary that leaves missing keys visibly unresolved."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class PromptManager:
    """Load prompt templates from the configured prompt directory."""

    def __init__(self, config: PromptConfig):
        # TODO: cache invalidation is unnecessary for now because files are tiny.
        self.config = config
        self.directory = config.directory

    def load_named_prompt(self, name: str) -> str:
        """Load an arbitrary prompt file by name."""

        path = self.directory / name
        return path.read_text(encoding="utf-8")

    def render_template(self, template: str, **values: Any) -> str:
        """Render an in-memory prompt template with Python format placeholders."""

        return template.format_map(_SafeFormatDict(values))

    def load_system_prompt(self) -> str:
        """Load the shared system prompt."""

        return self.load_named_prompt(self.config.system_file)

    def render(self, name: str, **values: Any) -> str:
        """Render a prompt file using Python format syntax."""

        template = self.load_named_prompt(name)
        return self.render_template(template, **values)

    def render_system(self, **values: Any) -> str:
        """Render the shared system prompt."""

        return self.render(self.config.system_file, **values)

    def render_action_proposal(
        self,
        *,
        observation: str,
        inventory: str,
        score: int,
        moves: int,
        action_candidates: int,
        generation_mode: str = "open",
        recent_trajectory_context: str = "",
        valid_actions: list[str] | None = None,
    ) -> str:
        """Render the action proposal prompt."""

        valid_actions = valid_actions or []
        valid_actions_block = (
            "Valid actions:\n" + "\n".join(f"- {action}" for action in valid_actions)
            if valid_actions
            else "Valid actions:\n(unavailable)"
        )
        recent_trajectory_block = recent_trajectory_context.strip() or "(none)"
        return self.render(
            self.config.action_proposal_file,
            observation=observation,
            inventory=inventory,
            score=score,
            moves=moves,
            action_candidates=action_candidates,
            generation_mode=generation_mode,
            recent_trajectory_context=recent_trajectory_block,
            valid_actions_block=valid_actions_block,
        )

    def render_trajectory_analysis(
        self,
        *,
        episode_id: str,
        seed: int,
        trajectory_excerpt: str,
        grounding_constraints: str = "",
    ) -> str:
        """Render the trajectory analysis prompt."""

        return self.render(
            self.config.trajectory_analysis_file,
            episode_id=episode_id,
            seed=seed,
            trajectory_excerpt=trajectory_excerpt,
            grounding_constraints=grounding_constraints.strip() or "(none)",
        )

    def render_local_reflection(
        self,
        *,
        state_summary: str,
        recent_actions: str,
        grounding_constraints: str = "",
    ) -> str:
        """Render the local reflection prompt."""

        return self.render(
            self.config.local_reflection_file,
            state_summary=state_summary,
            recent_actions=recent_actions,
            grounding_constraints=grounding_constraints.strip() or "(none)",
        )

    def render_state_selection(self, *, frontier_snapshot: str) -> str:
        """Render the state-selection prompt."""

        return self.render(self.config.state_selection_file, frontier_snapshot=frontier_snapshot)

    def available_prompt_paths(self) -> list[Path]:
        """Return the expected prompt files for quick inspection."""

        return [
            self.directory / self.config.system_file,
            self.directory / self.config.action_proposal_file,
            self.directory / self.config.trajectory_analysis_file,
            self.directory / self.config.local_reflection_file,
            self.directory / self.config.state_selection_file,
        ]
