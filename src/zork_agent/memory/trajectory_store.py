"""Storage helpers for episode trajectories and replay targets.

This module keeps persistence deliberately simple: one JSONL record per trajectory
step. Sub-trajectories are derived as prefixes rather than requiring a second
storage format.

TODO: add manifest files only once experiments accumulate more sidecar artifacts.
"""

from __future__ import annotations

from pathlib import Path

from zork_agent.memory.summaries import SummaryBuilder
from zork_agent.types import (
    SavedNode,
    StateCandidate,
    Trajectory,
    TrajectoryStep,
    WorldStateSnapshot,
    derive_state_family_key,
)
from zork_agent.utils.serialization import read_jsonl, write_jsonl


class TrajectoryStore:
    """Write and read JSONL trajectories under a root directory."""

    def __init__(self, root_dir: Path):
        # TODO: consider sharding directories by date when archives become large.
        self.root_dir = root_dir
        self.summary_builder = SummaryBuilder()

    def path_for_episode(self, episode_id: str) -> Path:
        """Return the JSONL path for one episode."""

        return self.root_dir / f"{episode_id}.jsonl"

    def path_for_prefix(self, episode_id: str, step_index: int) -> Path:
        """Return the JSONL path used for a stored prefix or exported sub-trajectory."""

        return self.root_dir / f"{episode_id}.prefix-{step_index:03d}.jsonl"

    def write_steps(self, episode_id: str, steps: list[TrajectoryStep]) -> Path:
        """Persist one episode trajectory from raw steps."""

        return self.write_trajectory(Trajectory(episode_id=episode_id, steps=steps))

    def write_trajectory(self, trajectory: Trajectory, *, path: Path | None = None) -> Path:
        """Persist a full trajectory to JSONL."""

        trajectory.validate_integrity()
        target_path = path or self.path_for_episode(trajectory.episode_id)
        write_jsonl(target_path, trajectory.to_records())
        return target_path

    def write_subtrajectory(self, trajectory: Trajectory, up_to_step_index: int) -> Path:
        """Persist a replayable prefix of a trajectory to JSONL."""

        prefix = trajectory.prefix(up_to_step_index)
        return self.write_trajectory(prefix, path=self.path_for_prefix(prefix.episode_id, up_to_step_index))

    def read_steps(self, path: Path) -> list[TrajectoryStep]:
        """Load steps back from a saved trajectory."""

        return self.read_trajectory(path).steps

    def read_trajectory(self, path: Path) -> Trajectory:
        """Load one full trajectory from JSONL."""

        return Trajectory.from_records(read_jsonl(path), source_path=path)

    def read_episode(self, episode_id: str) -> Trajectory:
        """Load a stored episode trajectory by episode id."""

        return self.read_trajectory(self.path_for_episode(episode_id))

    def read_subtrajectory(self, path: Path, up_to_step_index: int) -> Trajectory:
        """Load a stored trajectory and return the requested replayable prefix."""

        return self.read_trajectory(path).prefix(up_to_step_index)

    def state_candidate_for_step(
        self,
        trajectory: Trajectory,
        step_index: int,
        *,
        novelty: float = 0.0,
        native_state: tuple[object, ...] | None = None,
    ) -> StateCandidate:
        """Build a replay target candidate for a chosen trajectory node."""

        matching_steps = [step for step in trajectory.steps if step.step_index == step_index]
        if not matching_steps:
            raise IndexError(f"Trajectory step index out of range: {step_index}")
        step = matching_steps[0]
        recent_gain = self._recent_gain_for_step(trajectory, step)
        snapshot = step.world_state_snapshot or {}
        replay_actions = (
            list(snapshot.get("replay_actions", []))
            if isinstance(snapshot.get("replay_actions"), list)
            else trajectory.replay_actions(step_index)
        )
        saved_node = self.saved_node_for_step(
            trajectory,
            step_index,
            native_state=native_state,
        )
        summary_text = self.summary_builder.summarize_state_candidate(
            observation=step.observation,
            score=step.score,
            depth=step.step_index + 1,
            recent_gain=recent_gain,
            inventory_text=step.inventory_text,
            valid_actions=step.valid_actions,
        )
        candidate_metadata = dict(step.metadata)
        candidate_metadata.setdefault(
            "state_family_key",
            derive_state_family_key(
                observation=step.observation,
                inventory_text=step.inventory_text,
                valid_actions=step.valid_actions,
                last_action=step.action,
                summary_text=summary_text,
            ),
        )
        candidate_metadata.setdefault(
            "oscillating_pair_member",
            bool(step.loop_detected or step.inverse_of_previous or step.repeated_pair_count > 0),
        )
        candidate_metadata.setdefault("trivial_reversible_change", bool(step.metadata.get("loop_no_progress", False)))
        return StateCandidate.from_trajectory_step(
            step,
            replay_actions=replay_actions,
            novelty=novelty,
            recent_gain=recent_gain,
            summary_text=summary_text,
            saved_node=saved_node,
            metadata=candidate_metadata,
        )

    def saved_node_for_step(
        self,
        trajectory: Trajectory,
        step_index: int,
        *,
        native_state: tuple[object, ...] | None = None,
        summary_text: str | None = None,
    ) -> SavedNode:
        """Build a Jericho-first saved frontier node for one trajectory step."""

        matching_steps = [step for step in trajectory.steps if step.step_index == step_index]
        if not matching_steps:
            raise IndexError(f"Trajectory step index out of range: {step_index}")
        step = matching_steps[0]
        snapshot = step.world_state_snapshot or {}
        replay_actions = (
            list(snapshot.get("replay_actions", []))
            if isinstance(snapshot.get("replay_actions"), list)
            else trajectory.replay_actions(step_index)
        )
        stored_native_state: tuple[object, ...] | None = None
        if native_state is not None:
            stored_native_state = native_state
        elif isinstance(snapshot.get("native_state"), list):
            stored_native_state = tuple(snapshot["native_state"])
        summary = summary_text or self.summary_builder.summarize_state_candidate(
            observation=step.observation,
            score=step.score,
            depth=step.step_index + 1,
            recent_gain=self._recent_gain_for_step(trajectory, step),
            inventory_text=step.inventory_text,
            valid_actions=step.valid_actions,
        )
        return SavedNode(
            state_id=f"{step.episode_id}:{step.step_index}:{step.world_state_hash}",
            episode_id=step.episode_id,
            step_index=step.step_index,
            native_state=stored_native_state,
            action_prefix=replay_actions,
            score=step.score,
            observation=step.observation,
            inventory_text=step.inventory_text,
            world_state_hash=step.world_state_hash,
            valid_actions=list(step.valid_actions),
            summary_text=summary,
            metadata=dict(step.metadata),
        )

    def saved_node_from_snapshot(
        self,
        *,
        episode_id: str,
        step_index: int,
        observation: str,
        score: int,
        inventory_text: str,
        world_state_hash: str,
        action_prefix: list[str],
        snapshot: WorldStateSnapshot | None,
        valid_actions: list[str] | None = None,
        summary_text: str = "",
        metadata: dict[str, object] | None = None,
    ) -> SavedNode:
        """Build a saved node from a live world-state snapshot."""

        return SavedNode(
            state_id=f"{episode_id}:{step_index}:{world_state_hash}",
            episode_id=episode_id,
            step_index=step_index,
            native_state=snapshot.native_state if snapshot is not None else None,
            action_prefix=list(action_prefix),
            score=score,
            observation=observation,
            inventory_text=inventory_text,
            world_state_hash=world_state_hash,
            valid_actions=list(valid_actions or []),
            summary_text=summary_text,
            metadata={
                "state_family_key": derive_state_family_key(
                    observation=observation,
                    inventory_text=inventory_text,
                    valid_actions=list(valid_actions or []),
                    summary_text=summary_text,
                ),
                **dict(metadata or {}),
            },
        )

    def _recent_gain_for_step(self, trajectory: Trajectory, step: TrajectoryStep) -> float:
        """Compute a simple score-gain heuristic for one trajectory step."""

        previous_steps = [candidate for candidate in trajectory.steps if candidate.step_index < step.step_index]
        if not previous_steps:
            return float(step.score)
        previous = previous_steps[-1]
        return float(step.score - previous.score)
