"""Archive ingestion and derived-value refresh for GLoW archived states.

The archive is independent from the bounded trajectory frontier:
- ingestion walks all explored trajectories
- achieved value is refreshed from the retained frontier as a derived field
- projected potential is refreshed from frontier analysis as a derived field

This keeps the archive symbolic, typed, and inspectable without introducing a
learned archive model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any

from zork_agent.memory.frontier import TrajectoryFrontier
from zork_agent.memory.summaries import SummaryBuilder
from zork_agent.types import (
    ArchivedState,
    EpisodeTrajectory,
    FrontierInsight,
    ReplayMetadata,
    TextGameState,
    TrajectoryStep,
    derive_state_cluster_id,
)


class ArchiveUpdater:
    """Own archive identity, ingestion, and derived-field refresh logic."""

    def __init__(self, summary_builder: SummaryBuilder | None = None):
        self.summary_builder = summary_builder or SummaryBuilder()

    def archived_state_for_live_state(
        self,
        *,
        state: TextGameState,
        episode_id: str,
        provenance_trajectory_id: str,
        provenance_timestep: int,
    ) -> ArchivedState:
        """Build an initial archived state from a live Jericho/root state."""

        state_id, identity_strategy = archive_state_identity_from_fields(
            observation=state.observation,
            inventory_text=state.inventory_text,
            valid_actions=state.valid_actions,
            world_state_hash=state.world_state_hash,
            fallback_namespace=f"{episode_id}:root",
        )
        snapshot = state.world_state_snapshot
        replay_metadata = ReplayMetadata(
            restore_strategy=snapshot.restore_strategy if snapshot is not None else "action_replay_fallback",
            replay_actions=list(snapshot.replay_actions) if snapshot is not None else [],
            world_state_hash=state.world_state_hash,
            native_snapshot_reference=(
                state.world_state_hash
                if snapshot is not None and snapshot.native_state is not None
                else None
            ),
            native_snapshot=snapshot.native_state if snapshot is not None else None,
            metadata={"source": "initial_root"},
        )
        archived_state = ArchivedState(
            state_id=state_id,
            provenance_trajectory_id=provenance_trajectory_id,
            provenance_timestep=provenance_timestep,
            achieved_value=0.0,
            identity_strategy=identity_strategy,
            first_seen_episode_id=episode_id,
            first_seen_timestep=provenance_timestep,
            last_seen_episode_id=episode_id,
            last_seen_timestep=provenance_timestep,
            provenance_trajectory_ids=[provenance_trajectory_id],
            score_at_state=state.score,
            observation_summary=_truncate_text(state.observation, 200),
            inventory_summary=_truncate_text(state.inventory_text, 120),
            valid_action_summary=_valid_action_summary(state.valid_actions),
            state_cluster_id=state.state_cluster_id
            or derive_state_cluster_id(
                observation=state.observation,
                inventory_text=state.inventory_text,
                valid_actions=state.valid_actions,
            ),
            replay_metadata=replay_metadata,
            native_snapshot_reference=replay_metadata.native_snapshot_reference,
            native_snapshot=replay_metadata.native_snapshot,
            metadata={
                "observation": state.observation,
                "inventory_text": state.inventory_text,
                "valid_actions": list(state.valid_actions),
                "score": state.score,
                "done": state.done,
                "world_state_hash": state.world_state_hash,
            },
        )
        archived_state.validate_invariants()
        return archived_state

    def ingest_episode_trajectories(
        self,
        archive_by_state_id: dict[str, ArchivedState],
        trajectories: list[EpisodeTrajectory],
    ) -> dict[str, ArchivedState]:
        """Upsert every state encountered on a batch of explored trajectories."""

        merged = dict(archive_by_state_id)
        for trajectory in trajectories:
            merged = self.ingest_episode_trajectory(merged, trajectory)
        return merged

    def ingest_episode_trajectory(
        self,
        archive_by_state_id: dict[str, ArchivedState],
        trajectory: EpisodeTrajectory,
    ) -> dict[str, ArchivedState]:
        """Upsert every step from one explored trajectory into the archive."""

        merged = dict(archive_by_state_id)
        source_episode_id = str(trajectory.metadata.get("source_episode_id", trajectory.episode_id))
        for step in trajectory.steps:
            candidate = self._archived_state_for_step(
                trajectory=trajectory,
                step=step,
                source_episode_id=source_episode_id,
            )
            existing = merged.get(candidate.state_id)
            merged[candidate.state_id] = self._merge_archived_state(existing, candidate)
        return merged

    def refresh_achieved_values_from_frontier(
        self,
        archive_by_state_id: dict[str, ArchivedState],
        trajectory_frontier: TrajectoryFrontier,
    ) -> dict[str, ArchivedState]:
        """Refresh achieved value and frontier support from retained frontier trajectories."""

        refreshed = {
            state_id: self._reset_frontier_derived_fields(state)
            for state_id, state in archive_by_state_id.items()
        }
        for trajectory in trajectory_frontier.trajectories_for_analysis():
            source_episode_id = str(trajectory.metadata.get("source_episode_id", trajectory.episode_id))
            supported_state_ids: set[str] = set()
            for step in trajectory.steps:
                state_id, _identity_strategy = archive_state_identity_for_step(
                    step,
                    trajectory_id=trajectory.trajectory_id,
                )
                if state_id not in refreshed:
                    candidate = self._archived_state_for_step(
                        trajectory=trajectory,
                        step=step,
                        source_episode_id=source_episode_id,
                    )
                    refreshed[state_id] = candidate
                archived_state = refreshed[state_id]
                archived_state.achieved_value = max(
                    float(archived_state.achieved_value),
                    float(step.cumulative_reward),
                )
                if state_id not in supported_state_ids:
                    archived_state.frontier_support_count += 1
                    if trajectory.trajectory_id not in archived_state.supporting_frontier_trajectory_ids:
                        archived_state.supporting_frontier_trajectory_ids.append(trajectory.trajectory_id)
                    supported_state_ids.add(state_id)
        return refreshed

    def apply_projected_potential_from_frontier_analysis(
        self,
        archive_by_state_id: dict[str, ArchivedState],
        frontier_insight: FrontierInsight | None,
    ) -> dict[str, ArchivedState]:
        """Refresh projected potential from the latest frontier analysis."""

        refreshed = dict(archive_by_state_id)
        for archived_state in refreshed.values():
            archived_state.projected_potential_value = None

        if frontier_insight is None:
            return refreshed

        for annotation in frontier_insight.candidate_critical_states:
            supported_state_ids = {
                annotation.critical_state_id,
                *annotation.supporting_state_ids,
            }
            for state_id in supported_state_ids:
                archived_state = refreshed.get(state_id)
                if archived_state is None:
                    continue
                candidate_value = max(float(annotation.potential_value), float(archived_state.achieved_value))
                if (
                    archived_state.projected_potential_value is None
                    or candidate_value > archived_state.projected_potential_value
                ):
                    archived_state.projected_potential_value = candidate_value
        return refreshed

    def note_selection(self, archived_state: ArchivedState) -> ArchivedState:
        """Increment the archive selection count for one state."""

        archived_state.selection_count += 1
        return archived_state

    def note_restore_result(self, archived_state: ArchivedState, *, success: bool) -> ArchivedState:
        """Record one restore attempt outcome for an archived state."""

        if success:
            archived_state.visit_count += 1
            archived_state.restore_success_count += 1
        else:
            archived_state.restore_failure_count += 1
        return archived_state

    def note_revisit_progress(
        self,
        archived_state: ArchivedState,
        *,
        achieved_progress: bool,
        exploratory_progress: bool,
        branch_commit_allowed: bool,
        best_branch_actions: Sequence[str] | None = None,
        depth_progression_metadata: Mapping[str, Any] | None = None,
    ) -> ArchivedState:
        """Record whether revisiting one archived state produced reusable progress."""

        metadata = dict(archived_state.metadata)
        no_achievement_count = _metadata_int(metadata, "no_achievement_revisit_count")
        no_achievement_streak = _metadata_int(metadata, "no_achievement_revisit_streak")
        nonproductive_count = _metadata_int(metadata, "nonproductive_revisit_count")
        nonproductive_streak = _metadata_int(metadata, "nonproductive_revisit_streak")

        if achieved_progress:
            no_achievement_streak = 0
            nonproductive_streak = 0
        else:
            no_achievement_count += 1
            no_achievement_streak += 1
            if exploratory_progress:
                nonproductive_streak = 0
            else:
                nonproductive_count += 1
                nonproductive_streak += 1

        metadata.update(
            {
                "no_achievement_revisit_count": no_achievement_count,
                "no_achievement_revisit_streak": no_achievement_streak,
                "nonproductive_revisit_count": nonproductive_count,
                "nonproductive_revisit_streak": nonproductive_streak,
                "last_revisit_achieved_progress": bool(achieved_progress),
                "last_revisit_exploratory_progress": bool(exploratory_progress),
                "last_revisit_branch_commit_allowed": bool(branch_commit_allowed),
                "last_revisit_best_branch_actions": list(best_branch_actions or []),
            }
        )
        if depth_progression_metadata is not None:
            metadata.update(dict(depth_progression_metadata))
        archived_state.metadata = metadata
        return archived_state

    def upsert_archived_state(
        self,
        archive_by_state_id: dict[str, ArchivedState],
        archived_state: ArchivedState,
    ) -> dict[str, ArchivedState]:
        """Insert one archived state into the persistent archive map."""

        merged = dict(archive_by_state_id)
        merged[archived_state.state_id] = self._merge_archived_state(
            merged.get(archived_state.state_id),
            archived_state,
        )
        return merged

    def sorted_states(self, archive_by_state_id: Mapping[str, ArchivedState]) -> list[ArchivedState]:
        """Return archived states in a deterministic selection-friendly order."""

        return sorted(
            archive_by_state_id.values(),
            key=lambda state: (
                -float(state.achieved_value),
                -float(state.projected_potential_value or state.achieved_value),
                -int(state.frontier_support_count),
                state.selection_count,
                state.provenance_trajectory_id,
                state.provenance_timestep,
                state.state_id,
            ),
        )

    def _archived_state_for_step(
        self,
        *,
        trajectory: EpisodeTrajectory,
        step: TrajectoryStep,
        source_episode_id: str,
    ) -> ArchivedState:
        """Build one archive state from a single encountered trajectory step."""

        state_id, identity_strategy = archive_state_identity_for_step(
            step,
            trajectory_id=trajectory.trajectory_id,
        )
        replay_metadata = replay_metadata_for_step(trajectory, step)
        archived_state = ArchivedState(
            state_id=state_id,
            provenance_trajectory_id=trajectory.trajectory_id,
            provenance_timestep=step.step_index,
            achieved_value=0.0,
            identity_strategy=identity_strategy,
            first_seen_episode_id=source_episode_id,
            first_seen_timestep=step.step_index,
            last_seen_episode_id=source_episode_id,
            last_seen_timestep=step.step_index,
            provenance_trajectory_ids=[trajectory.trajectory_id],
            score_at_state=step.score,
            observation_summary=_truncate_text(step.observation, 200),
            inventory_summary=_truncate_text(step.inventory_text, 120),
            valid_action_summary=_valid_action_summary(step.valid_actions),
            state_cluster_id=step.state_cluster_id
            or derive_state_cluster_id(
                observation=step.observation,
                inventory_text=step.inventory_text,
                valid_actions=step.valid_actions,
            ),
            replay_metadata=replay_metadata,
            native_snapshot_reference=step.native_snapshot_reference or replay_metadata.native_snapshot_reference,
            native_snapshot=replay_metadata.native_snapshot,
            metadata={
                "observation": step.observation,
                "inventory_text": step.inventory_text,
                "valid_actions": list(step.valid_actions),
                "score": step.score,
                "done": step.done,
                "world_state_hash": step.world_state_hash,
                "trajectory_root_state_id": trajectory.root_state_id,
                "selected_from_archive_state_id": trajectory.selected_from_archive_state_id,
            },
        )
        archived_state.validate_invariants()
        return archived_state

    def _merge_archived_state(
        self,
        existing: ArchivedState | None,
        candidate: ArchivedState,
    ) -> ArchivedState:
        """Merge a newly encountered state into the persistent archive entry."""

        if existing is None:
            return candidate

        merged = ArchivedState.from_record(existing.to_record(include_native_snapshot=True))
        merged.last_seen_episode_id = candidate.last_seen_episode_id or existing.last_seen_episode_id
        merged.last_seen_timestep = max(existing.last_seen_timestep, candidate.last_seen_timestep)
        if not merged.first_seen_episode_id:
            merged.first_seen_episode_id = candidate.first_seen_episode_id
        merged.provenance_trajectory_ids = _ordered_unique(
            [*existing.provenance_trajectory_ids, *candidate.provenance_trajectory_ids]
        )
        if candidate.native_snapshot is not None:
            merged.native_snapshot = candidate.native_snapshot
        if candidate.native_snapshot_reference:
            merged.native_snapshot_reference = candidate.native_snapshot_reference
        if candidate.replay_metadata.native_snapshot is not None:
            merged.replay_metadata = candidate.replay_metadata
        elif not merged.replay_metadata.replay_actions and candidate.replay_metadata.replay_actions:
            merged.replay_metadata = candidate.replay_metadata
        merged.score_at_state = candidate.score_at_state
        if candidate.observation_summary:
            merged.observation_summary = candidate.observation_summary
        if candidate.inventory_summary:
            merged.inventory_summary = candidate.inventory_summary
        if candidate.valid_action_summary:
            merged.valid_action_summary = candidate.valid_action_summary
        if candidate.state_cluster_id:
            merged.state_cluster_id = candidate.state_cluster_id
        merged.metadata = {**dict(existing.metadata), **dict(candidate.metadata)}
        if candidate.identity_strategy and not merged.identity_strategy:
            merged.identity_strategy = candidate.identity_strategy
        merged.validate_invariants()
        return merged

    def _reset_frontier_derived_fields(self, archived_state: ArchivedState) -> ArchivedState:
        """Clear frontier-derived cache fields before recomputing them."""

        refreshed = ArchivedState.from_record(archived_state.to_record(include_native_snapshot=True))
        refreshed.achieved_value = 0.0
        refreshed.frontier_support_count = 0
        refreshed.supporting_frontier_trajectory_ids = []
        return refreshed


def archive_state_identity_for_step(
    step: TrajectoryStep,
    *,
    trajectory_id: str,
) -> tuple[str, str]:
    """Return a stable archive identity for one trajectory step.

    Identity strategy:
    - prefer the environment/native `world_state_hash` when it is present
    - otherwise fall back to a normalized text signature over observation, inventory,
      and valid actions
    """

    return archive_state_identity_from_fields(
        observation=step.observation,
        inventory_text=step.inventory_text,
        valid_actions=step.valid_actions,
        world_state_hash=step.world_state_hash,
        fallback_namespace=f"{trajectory_id}:{step.step_index}",
    )


def archive_state_identity_from_fields(
    *,
    observation: str,
    inventory_text: str,
    valid_actions: list[str],
    world_state_hash: str,
    fallback_namespace: str,
) -> tuple[str, str]:
    """Return a stable archive state id and the strategy used to derive it."""

    normalized_hash = world_state_hash.strip()
    if normalized_hash and normalized_hash != "unknown":
        return f"archive:{normalized_hash}", "world_state_hash"

    normalized_observation = " ".join(observation.split()).strip().lower()
    normalized_inventory = " ".join(inventory_text.split()).strip().lower()
    normalized_actions = "|".join(sorted(" ".join(action.split()).strip().lower() for action in valid_actions if action.strip()))
    fallback_payload = "||".join([normalized_observation, normalized_inventory, normalized_actions, fallback_namespace])
    digest = hashlib.sha1(fallback_payload.encode("utf-8")).hexdigest()[:16]
    return f"archive:fallback:{digest}", "normalized_state_signature"


def replay_metadata_for_step(trajectory: EpisodeTrajectory, step: TrajectoryStep) -> ReplayMetadata:
    """Build replay metadata for one state encountered on a trajectory."""

    snapshot = step.world_state_snapshot or {}
    restore_strategy = str(snapshot.get("restore_strategy", trajectory.replay_metadata.restore_strategy))
    replay_actions = (
        [str(item) for item in snapshot.get("replay_actions", [])]
        if isinstance(snapshot.get("replay_actions"), list)
        else [candidate.action for candidate in trajectory.steps if candidate.step_index <= step.step_index]
    )
    native_snapshot = snapshot.get("native_state")
    normalized_native_snapshot: tuple[Any, ...] | None = None
    if isinstance(native_snapshot, list):
        normalized_native_snapshot = tuple(native_snapshot)
    elif isinstance(native_snapshot, tuple):
        normalized_native_snapshot = tuple(native_snapshot)
    return ReplayMetadata(
        restore_strategy=restore_strategy,
        replay_actions=replay_actions,
        world_state_hash=step.world_state_hash,
        native_snapshot_reference=step.native_snapshot_reference or trajectory.replay_metadata.native_snapshot_reference,
        native_snapshot=normalized_native_snapshot,
        metadata={
            **dict(trajectory.replay_metadata.metadata),
            "trajectory_id": trajectory.trajectory_id,
            "step_index": step.step_index,
        },
    )


def _ordered_unique(values: list[str]) -> list[str]:
    """Return unique strings while preserving the original order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _truncate_text(value: str, limit: int) -> str:
    """Truncate text conservatively for archive summaries."""

    normalized = " ".join(value.split()).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _valid_action_summary(valid_actions: list[str]) -> str:
    """Build a compact valid-action summary string."""

    return ", ".join(valid_actions[:8])


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    """Read one metadata counter as a non-negative integer."""

    raw_value = metadata.get(key, 0)
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return 0
