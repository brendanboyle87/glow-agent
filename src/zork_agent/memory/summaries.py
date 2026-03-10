"""Deterministic summary helpers for trajectories and replay candidates.

These summaries are engineering approximations intended for logging, storage, and
LLM prompt context. They are intentionally explicit rather than "smart."

TODO: keep deterministic summaries even if model-generated notes are added later.
"""

from __future__ import annotations

from zork_agent.types import StateCandidate, Trajectory, TrajectoryStep


class SummaryBuilder:
    """Create short deterministic summaries from trajectories and candidate states."""

    def summarize_steps(self, steps: list[TrajectoryStep]) -> str:
        """Build a small text summary from raw trajectory steps."""

        if not steps:
            return "No steps recorded."

        unique_states = {step.world_state_hash for step in steps}
        last_action = steps[-1].action
        final_score = steps[-1].score
        return (
            f"Visited {len(unique_states)} distinct states across {len(steps)} steps. "
            f"Final score: {final_score}. Last action: {last_action}."
        )

    def summarize_trajectory(self, trajectory: Trajectory) -> str:
        """Build a short summary for a full or partial trajectory."""

        if not trajectory.steps:
            return f"Episode {trajectory.episode_id} has no recorded steps."

        return (
            f"Episode {trajectory.episode_id}: {self.summarize_steps(trajectory.steps)} "
            f"Total reward: {trajectory.total_reward():.2f}."
        )

    def summarize_state_candidate(
        self,
        *,
        observation: str,
        score: int | float,
        depth: int,
        recent_gain: float,
        inventory_text: str = "",
        valid_actions: list[str] | None = None,
    ) -> str:
        """Build a compact state summary string for prompt consumption."""

        # TODO: expose truncation rules if prompt lengths start mattering.
        actions = ", ".join((valid_actions or [])[:4]) or "unknown"
        inventory = inventory_text or "unknown"
        return (
            f"depth={depth} score={score} recent_gain={recent_gain:.2f} "
            f"inventory={inventory} next_actions={actions} observation={observation}"
        )

    def summarize_candidate(self, candidate: StateCandidate) -> str:
        """Summarize a replayable state candidate."""

        if candidate.summary_text:
            return candidate.summary_text

        return self.summarize_state_candidate(
            observation=candidate.observation,
            score=candidate.score,
            depth=candidate.depth,
            recent_gain=candidate.recent_gain,
            inventory_text=candidate.inventory_text,
            valid_actions=candidate.valid_actions,
        )
