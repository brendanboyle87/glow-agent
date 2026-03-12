"""Tests for JSONL serialization and trajectory persistence helpers.

TODO: add corruption and partial-write tests if logs become critical experiment artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zork_agent.memory.trajectory_store import TrajectoryStore
from zork_agent.types import EpisodeTrajectory, Trajectory, TrajectoryStep
from zork_agent.utils.serialization import SerializationError, append_jsonl, read_jsonl, write_jsonl


def _build_trajectory() -> Trajectory:
    """Create a small trajectory fixture used by persistence tests."""

    return Trajectory(
        episode_id="episode-001",
        steps=[
            TrajectoryStep(
                episode_id="episode-001",
                step_index=0,
                action="look",
                observation="You are standing in an open field.",
                reward=0.0,
                done=False,
                score=0,
                moves=1,
                inventory_text="Inventory empty.",
                valid_actions=["look", "open mailbox"],
                world_state_hash="state-1",
                world_state_snapshot={"restore_strategy": "action_replay_fallback", "replay_actions": ["look"]},
            ),
            TrajectoryStep(
                episode_id="episode-001",
                step_index=1,
                action="open mailbox",
                observation="Opening the small mailbox reveals a leaflet.",
                reward=0.0,
                done=False,
                score=1,
                moves=2,
                inventory_text="Inventory empty.",
                valid_actions=["read leaflet", "look"],
                world_state_hash="state-2",
                world_state_snapshot={
                    "restore_strategy": "action_replay_fallback",
                    "replay_actions": ["look", "open mailbox"],
                },
            ),
        ],
    )


def test_write_and_read_jsonl_round_trip(tmp_path: Path) -> None:
    """JSONL helpers should preserve ordered records."""

    path = tmp_path / "records.jsonl"
    records = [{"step": 0, "action": "look"}, {"step": 1, "action": "open mailbox"}]

    write_jsonl(path, records)

    assert read_jsonl(path) == records


def test_append_jsonl_adds_one_record(tmp_path: Path) -> None:
    """Appending should preserve existing data and add one new record."""

    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"step": 0}])
    append_jsonl(path, {"step": 1})

    assert read_jsonl(path) == [{"step": 0}, {"step": 1}]


def test_trajectory_store_persists_and_restores_trajectory_round_trip(tmp_path: Path) -> None:
    """Trajectory persistence should round-trip typed steps through JSONL."""

    store = TrajectoryStore(tmp_path / "trajectories")
    trajectory = _build_trajectory()

    path = store.write_trajectory(trajectory)
    restored = store.read_trajectory(path)

    assert restored.episode_id == "episode-001"
    assert len(restored.steps) == 2
    assert restored.steps[1].action == "open mailbox"
    assert restored.steps[1].world_state_snapshot == {
        "restore_strategy": "action_replay_fallback",
        "replay_actions": ["look", "open mailbox"],
    }


def test_trajectory_store_writes_replayable_prefix(tmp_path: Path) -> None:
    """Sub-trajectories should be written as replayable prefixes."""

    store = TrajectoryStore(tmp_path / "trajectories")
    trajectory = _build_trajectory()

    prefix_path = store.write_subtrajectory(trajectory, up_to_step_index=0)
    restored_prefix = store.read_trajectory(prefix_path)

    assert restored_prefix.episode_id == "episode-001"
    assert [step.action for step in restored_prefix.steps] == ["look"]


def test_trajectory_store_adapts_legacy_episode_to_typed_episode_trajectory(tmp_path: Path) -> None:
    """Stored legacy trajectories should adapt cleanly into the typed episode artifact."""

    store = TrajectoryStore(tmp_path / "trajectories")
    trajectory = _build_trajectory()
    path = store.write_trajectory(trajectory)

    typed_episode = store.read_episode_trajectory(path, root_state_id="archive:root-state")

    assert isinstance(typed_episode, EpisodeTrajectory)
    assert typed_episode.root_state_id == "archive:root-state"
    assert typed_episode.max_cumulative_reward_achieved == 0.0
    assert typed_episode.steps[0].cumulative_reward == 0.0
    assert typed_episode.steps[1].cumulative_reward == 0.0


def test_trajectory_store_derives_reversible_state_family_for_replay_candidates(tmp_path: Path) -> None:
    """Replay candidates derived from stored steps should carry a stable family key."""

    store = TrajectoryStore(tmp_path / "trajectories")
    trajectory = Trajectory(
        episode_id="episode-001",
        steps=[
            TrajectoryStep(
                episode_id="episode-001",
                step_index=0,
                action="open mailbox",
                observation="Opening the small mailbox reveals a leaflet.",
                reward=0.0,
                done=False,
                score=0,
                moves=1,
                inventory_text="Inventory empty.",
                valid_actions=["close mailbox", "take leaflet", "look"],
                world_state_hash="state-mailbox-open",
                loop_detected=True,
                inverse_of_previous=True,
                repeated_pair_count=1,
                loop_penalty=1.75,
                metadata={"loop_no_progress": True},
            )
        ],
    )

    candidate = store.state_candidate_for_step(trajectory, 0)

    assert candidate.state_family_key == "toggle:mailbox"
    assert candidate.oscillating_pair_member is True
    assert candidate.trivial_reversible_change is True


def test_trajectory_step_from_record_normalizes_legacy_snapshot_fields() -> None:
    """Legacy snapshot payloads should be canonicalized on read."""

    step = TrajectoryStep.from_record(
        {
            "episode_id": "episode-001",
            "step_index": 3,
            "action": "open mailbox",
            "observation": "Leaflet visible.",
            "reward": 1.0,
            "done": False,
            "score": 1,
            "moves": 4,
            "inventory_text": "lamp, leaflet",
            "valid_actions": "read leaflet",
            "world_state_hash": "hash-mailbox",
            "world_state_snapshot": {
                "restore_strategy": "replay_only",
                "replay_actions": "open mailbox",
                "native_state": ["ram", "stack", 4],
            },
            "metadata": [],
        }
    )

    assert step.valid_actions == ["read leaflet"]
    assert step.metadata == {}
    assert step.world_state_snapshot == {
        "restore_strategy": "action_replay_fallback",
        "replay_actions": ["open mailbox"],
        "native_state": ("ram", "stack", 4),
    }


def test_read_jsonl_raises_clear_error_for_invalid_json(tmp_path: Path) -> None:
    """Malformed JSONL should fail with line-number context."""

    path = tmp_path / "broken.jsonl"
    path.write_text('{"ok": 1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(SerializationError, match="broken.jsonl:2"):
        read_jsonl(path)


def test_read_jsonl_rejects_non_object_records(tmp_path: Path) -> None:
    """JSONL records must be object-shaped for trajectory persistence."""

    path = tmp_path / "array-record.jsonl"
    path.write_text('[1, 2, 3]\n', encoding="utf-8")

    with pytest.raises(SerializationError, match="Expected a JSON object"):
        read_jsonl(path)


def test_trajectory_store_rejects_inconsistent_episode_ids(tmp_path: Path) -> None:
    """Trajectory persistence should reject steps from multiple episodes."""

    store = TrajectoryStore(tmp_path / "trajectories")
    trajectory = Trajectory(
        episode_id="episode-001",
        steps=[
            TrajectoryStep(
                episode_id="episode-001",
                step_index=0,
                action="look",
                observation="field",
                reward=0.0,
                done=False,
                score=0,
                moves=1,
                world_state_hash="state-1",
            ),
            TrajectoryStep(
                episode_id="episode-002",
                step_index=1,
                action="open mailbox",
                observation="leaflet",
                reward=0.0,
                done=False,
                score=1,
                moves=2,
                world_state_hash="state-2",
            ),
        ],
    )

    with pytest.raises(ValueError, match="multiple episode ids"):
        store.write_trajectory(trajectory)


def test_trajectory_store_rejects_unsorted_step_indices(tmp_path: Path) -> None:
    """Trajectory persistence should reject unsorted step indices."""

    store = TrajectoryStore(tmp_path / "trajectories")
    trajectory = Trajectory(
        episode_id="episode-001",
        steps=[
            TrajectoryStep(
                episode_id="episode-001",
                step_index=1,
                action="open mailbox",
                observation="leaflet",
                reward=0.0,
                done=False,
                score=1,
                moves=2,
                world_state_hash="state-2",
            ),
            TrajectoryStep(
                episode_id="episode-001",
                step_index=0,
                action="look",
                observation="field",
                reward=0.0,
                done=False,
                score=0,
                moves=1,
                world_state_hash="state-1",
            ),
        ],
    )

    with pytest.raises(ValueError, match="sorted by step_index"):
        store.write_trajectory(trajectory)
