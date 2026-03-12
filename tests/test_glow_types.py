"""Tests for typed GLoW-style artifact models.

TODO: expand these tests when the global/local world-model pipeline starts writing real artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zork_agent.types import (
    ActionBiasRecord,
    AdvantageHint,
    AdvantagePoint,
    AffordanceRecord,
    ArchivedState,
    CriticalStateAnnotation,
    EpisodeTrajectory,
    FrontierInsight,
    FrontierTrajectoryEntry,
    LocalWorldModel,
    ReplayMetadata,
    SimilarityMetadata,
    SubgoalRecord,
    Trajectory,
    TrajectoryStep,
)


def _legacy_trajectory() -> Trajectory:
    """Create a small legacy trajectory fixture used by the typed adapters."""

    return Trajectory(
        episode_id="episode-typed-001",
        steps=[
            TrajectoryStep(
                episode_id="episode-typed-001",
                step_index=0,
                action="open mailbox",
                observation="Opening the small mailbox reveals a leaflet.",
                reward=0.0,
                done=False,
                score=0,
                moves=1,
                world_state_hash="state-mailbox-open",
                inventory_text="Inventory empty.",
                valid_actions=["take leaflet", "close mailbox", "north"],
                state_cluster_id="region:mailbox",
                region_novelty_score=0.8,
                native_snapshot_reference="snap-000",
                world_state_snapshot={
                    "restore_strategy": "native_snapshot",
                    "replay_actions": ["open mailbox"],
                    "native_state": ("ram", "mailbox-open"),
                },
            ),
            TrajectoryStep(
                episode_id="episode-typed-001",
                step_index=1,
                action="take leaflet",
                observation="Taken.",
                reward=1.0,
                cumulative_reward=1.0,
                done=False,
                score=1,
                moves=2,
                world_state_hash="state-leaflet-taken",
                inventory_text="leaflet",
                valid_actions=["north", "south", "read leaflet"],
                loop_detected=False,
                state_cluster_id="region:mailbox",
                region_novelty_score=0.6,
                native_snapshot_reference="snap-001",
            ),
        ],
        source_path=Path("/tmp/episode-typed-001.jsonl"),
        metadata={"seed": 11},
    )


def test_trajectory_step_round_trip_with_typed_fields() -> None:
    """Trajectory steps should round-trip cumulative reward and native snapshot references."""

    step = TrajectoryStep(
        episode_id="episode-001",
        step_index=3,
        action="read leaflet",
        observation="It says welcome to Zork.",
        reward=0.5,
        cumulative_reward=1.5,
        done=False,
        score=2,
        moves=4,
        world_state_hash="hash-leaflet",
        inventory_text="leaflet",
        valid_actions=["north", "south"],
        native_snapshot_reference="native-snap-003",
        world_state_snapshot={
            "restore_strategy": "native_snapshot",
            "replay_actions": ["open mailbox", "take leaflet", "read leaflet"],
            "native_state": ("ram", "leaflet-read"),
        },
    )

    restored = TrajectoryStep.from_record(step.to_record())

    assert restored.cumulative_reward == 1.5
    assert restored.native_snapshot_reference == "native-snap-003"
    assert restored.world_state_snapshot == {
        "restore_strategy": "native_snapshot",
        "replay_actions": ["open mailbox", "take leaflet", "read leaflet"],
        "native_state": ("ram", "leaflet-read"),
    }


def test_episode_trajectory_adapts_legacy_trajectory_and_round_trips() -> None:
    """The existing JSONL trajectory should adapt into the typed episode representation."""

    legacy = _legacy_trajectory()

    typed_episode = EpisodeTrajectory.from_legacy_trajectory(
        legacy,
        root_state_id="archive:mailbox-root",
        selected_from_archive_state_id="archive:selected-mailbox",
    )
    restored = EpisodeTrajectory.from_record(typed_episode.to_record())

    assert restored.episode_id == legacy.episode_id
    assert restored.root_state_id == "archive:mailbox-root"
    assert restored.selected_from_archive_state_id == "archive:selected-mailbox"
    assert [step.cumulative_reward for step in restored.steps] == [0.0, 1.0]
    assert restored.max_cumulative_reward_achieved == 1.0
    assert restored.final_score == 1
    assert restored.final_done is False
    assert restored.summary_fields.unique_world_state_count == 2
    assert "leaflet" in restored.summary_fields.discovered_object_tokens

    projected = restored.to_legacy_trajectory()

    assert projected.episode_id == legacy.episode_id
    assert [step.action for step in projected.steps] == ["open mailbox", "take leaflet"]


def test_episode_trajectory_rejects_bad_cumulative_reward_sequence() -> None:
    """Episode trajectories should fail fast when cumulative reward does not match the steps."""

    with pytest.raises(ValueError, match="cumulative_reward mismatch"):
        EpisodeTrajectory(
            episode_id="episode-001",
            root_state_id="root-state",
            steps=[
                TrajectoryStep(
                    episode_id="episode-001",
                    step_index=0,
                    action="look",
                    observation="field",
                    reward=1.0,
                    cumulative_reward=0.0,
                    done=False,
                    score=1,
                    moves=1,
                    world_state_hash="hash-1",
                )
            ],
            max_cumulative_reward_achieved=0.0,
            final_score=1,
            final_done=False,
        ).validate_invariants()


def test_frontier_trajectory_entry_round_trip() -> None:
    """Frontier trajectory entries should serialize without losing diversity metadata."""

    entry = FrontierTrajectoryEntry(
        trajectory_id="episode-001",
        value=5.5,
        novelty_score=1.2,
        diversity_score=0.8,
        state_cluster_ids=["region:mailbox", "region:forest"],
        state_family_keys=["toggle:mailbox"],
        bottleneck_state_ids=["archive:window"],
        bottleneck_step_indices=[4, 8],
        inserted_at="2026-03-10T12:00:00Z",
        insertion_order=7,
        metadata={"frontier": "global"},
    )

    restored = FrontierTrajectoryEntry.from_record(entry.to_record())

    assert restored == entry


def test_archived_state_round_trip_with_similarity_and_snapshot() -> None:
    """Archived states should carry replay metadata, similarity metadata, and native snapshot refs."""

    archived_state = ArchivedState(
        state_id="archive:window-001",
        provenance_trajectory_id="episode-001",
        provenance_timestep=9,
        achieved_value=7.5,
        visit_count=2,
        selection_count=1,
        replay_metadata=ReplayMetadata(
            restore_strategy="native_snapshot",
            replay_actions=["open mailbox", "take leaflet", "east"],
            world_state_hash="hash-window",
            native_snapshot_reference="snap-window-001",
            native_snapshot=("ram", "window-open"),
            metadata={"cluster": "region:house-field"},
        ),
        similarity_metadata=SimilarityMetadata(
            embedding_model="test-embed",
            embedding_id="emb-001",
            similarity_key="window",
            nearest_state_ids=["archive:window-002"],
            similarity_score=0.93,
        ),
        native_snapshot_reference="snap-window-001",
        native_snapshot=("ram", "window-open"),
        metadata={"tag": "critical"},
    )

    restored = ArchivedState.from_record(archived_state.to_record(include_native_snapshot=True))

    assert restored.state_id == archived_state.state_id
    assert restored.replay_metadata.native_snapshot == ("ram", "window-open")
    assert restored.similarity_metadata is not None
    assert restored.similarity_metadata.embedding_id == "emb-001"
    assert restored.native_snapshot_reference == "snap-window-001"
    assert restored.native_snapshot == ("ram", "window-open")


def test_frontier_insight_round_trip_with_critical_states() -> None:
    """Global frontier insights should round-trip nested critical-state annotations."""

    critical_state = CriticalStateAnnotation(
        critical_state_id="archive:window-001",
        achieved_value=5.0,
        potential_value=12.0,
        textual_rationale="Window entry appears to unlock the house interior.",
        source_frontier_trajectory_ids=["episode-001", "episode-003"],
        confidence=0.9,
        support_count=2,
        supporting_state_ids=["archive:window-001", "archive:window-002"],
    )
    insight = FrontierInsight(
        analysis_id="analysis-001",
        frontier_trajectory_ids=["episode-001", "episode-002"],
        inferred_bottlenecks=["house-entry"],
        partial_solutions=["open window"],
        missing_prerequisites=["reach house exterior"],
        candidate_critical_states=[critical_state],
        generated_at="2026-03-10T12:30:00Z",
    )

    restored = FrontierInsight.from_record(insight.to_record())

    assert restored.analysis_id == "analysis-001"
    assert restored.candidate_critical_states[0].critical_state_id == "archive:window-001"
    assert restored.candidate_critical_states[0].potential_value == 12.0


def test_local_world_model_round_trip_and_root_invariant() -> None:
    """Local world models should round-trip and enforce a shared root state across hints."""

    world_model = LocalWorldModel(
        root_state_id="archive:window-001",
        accumulated_advantage_hints=[
            AdvantageHint(
                root_state_id="archive:window-001",
                compared_trajectory_ids=["episode-001", "episode-002"],
                key_state_action_points=[
                    AdvantagePoint(
                        state_id="archive:window-001",
                        action="open window",
                        inferred_advantage=2.5,
                        outcome_delta="revealed house entry",
                        textual_rationale="This branch exposes the house interior.",
                        support_trajectory_ids=["episode-001"],
                        confidence=0.8,
                    )
                ],
                intermediate_advantage_summaries=["Window access creates a structural shortcut."],
                action_preferences=["open window", "enter window"],
                action_avoidances=["close window"],
                textual_reasoning="Opening the window preserves access to the house.",
                confidence=0.85,
            )
        ],
        discovered_subgoals=[
            SubgoalRecord(
                subgoal_id="subgoal:enter-house",
                description="Reach the house interior through the window.",
                supporting_state_ids=["archive:window-001"],
                confidence=0.9,
            )
        ],
        inferred_affordances=[
            AffordanceRecord(
                object_text="window",
                affordance="enterable",
                supporting_action="open window",
                confidence=0.8,
            )
        ],
        action_priors=[ActionBiasRecord(action="open window", weight=1.5, rationale="structural access", support_count=2)],
        action_antipriors=[ActionBiasRecord(action="close window", weight=-1.0, rationale="removes access", support_count=1)],
        last_updated_timestamp="2026-03-10T12:45:00Z",
    )

    restored = LocalWorldModel.from_record(world_model.to_record())

    assert restored.root_state_id == "archive:window-001"
    assert restored.accumulated_advantage_hints[0].action_preferences == ["open window", "enter window"]
    assert restored.discovered_subgoals[0].subgoal_id == "subgoal:enter-house"
    assert restored.inferred_affordances[0].affordance == "enterable"

    bad_world_model = LocalWorldModel(
        root_state_id="archive:window-001",
        accumulated_advantage_hints=[
            AdvantageHint(root_state_id="archive:forest-002")
        ],
    )
    with pytest.raises(ValueError, match="share the same root_state_id"):
        bad_world_model.validate_invariants()
