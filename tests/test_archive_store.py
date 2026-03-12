"""Tests for the independent persistent archive subsystem."""

from __future__ import annotations

from pathlib import Path

from zork_agent.memory.archive_store import ArchiveStore
from zork_agent.memory.archive_updater import ArchiveUpdater, archive_state_identity_from_fields
from zork_agent.memory.frontier import TrajectoryFrontier, TrajectoryFrontierConfig
from zork_agent.types import EpisodeTrajectory, ReplayMetadata, TrajectoryStep


def _episode_trajectory(
    episode_id: str,
    *,
    rewards: list[float],
    world_state_hashes: list[str],
) -> EpisodeTrajectory:
    """Build a small typed trajectory fixture for archive tests."""

    cumulative_reward = 0.0
    steps: list[TrajectoryStep] = []
    for index, reward in enumerate(rewards):
        cumulative_reward += reward
        steps.append(
            TrajectoryStep(
                episode_id=episode_id,
                step_index=index,
                action=f"action-{index}",
                observation=f"Observation {index} in {episode_id}.",
                reward=reward,
                cumulative_reward=cumulative_reward,
                done=index == len(rewards) - 1,
                score=int(cumulative_reward),
                moves=index + 1,
                world_state_hash=world_state_hashes[index],
                inventory_text="leaflet" if index >= 1 else "",
                valid_actions=["look", "north", "open window"],
                state_cluster_id=f"cluster-{index}",
                native_snapshot_reference=f"{episode_id}:snap:{index}",
                world_state_snapshot={
                    "restore_strategy": "native_snapshot",
                    "replay_actions": [f"action-{step_index}" for step_index in range(index + 1)],
                    "native_state": [episode_id, index],
                },
            )
        )
    trajectory = EpisodeTrajectory(
        episode_id=episode_id,
        root_state_id=f"{episode_id}:root:{world_state_hashes[0]}",
        steps=steps,
        max_cumulative_reward_achieved=max(step.cumulative_reward for step in steps),
        final_score=steps[-1].score,
        final_done=steps[-1].done,
        replay_metadata=ReplayMetadata(
            restore_strategy="native_snapshot",
            replay_actions=[step.action for step in steps],
            world_state_hash=world_state_hashes[0],
            native_snapshot_reference=steps[0].native_snapshot_reference,
        ),
        metadata={"source_episode_id": "episode-root"},
    )
    trajectory.validate_invariants()
    return trajectory


def test_archive_ingests_states_from_all_explored_trajectories_not_just_frontier(tmp_path: Path) -> None:
    """All explored trajectories should populate the archive, even if some are not retained."""

    updater = ArchiveUpdater()
    frontier = TrajectoryFrontier(TrajectoryFrontierConfig(max_size=1))
    low = _episode_trajectory(
        "traj-low",
        rewards=[0.0, 1.0],
        world_state_hashes=["low-root", "low-leaflet"],
    )
    high = _episode_trajectory(
        "traj-high",
        rewards=[0.0, 5.0],
        world_state_hashes=["high-root", "high-egg"],
    )

    archive_by_state_id = updater.ingest_episode_trajectories({}, [low, high])
    frontier.insert(low)
    frontier.insert(high)
    archive_by_state_id = updater.refresh_achieved_values_from_frontier(archive_by_state_id, frontier)

    assert "archive:low-root" in archive_by_state_id
    assert "archive:low-leaflet" in archive_by_state_id
    assert "archive:high-root" in archive_by_state_id
    assert "archive:high-egg" in archive_by_state_id
    assert len(frontier.snapshot_entries()) == 1
    assert frontier.snapshot_entries()[0].trajectory_id == "traj-high"
    assert archive_by_state_id["archive:low-leaflet"].frontier_support_count == 0
    assert archive_by_state_id["archive:high-egg"].frontier_support_count == 1


def test_archive_refreshes_achieved_value_from_current_frontier_support() -> None:
    """Achieved value should be a frontier-derived cache, not a local permanent maximum."""

    updater = ArchiveUpdater()
    lower = _episode_trajectory(
        "traj-one",
        rewards=[0.0, 2.0],
        world_state_hashes=["shared-root", "shared-window"],
    )
    higher = _episode_trajectory(
        "traj-two",
        rewards=[0.0, 5.0],
        world_state_hashes=["shared-root", "shared-window"],
    )
    archive_by_state_id = updater.ingest_episode_trajectories({}, [lower, higher])
    frontier = TrajectoryFrontier(TrajectoryFrontierConfig(max_size=1))
    frontier.insert(lower)
    archive_by_state_id = updater.refresh_achieved_values_from_frontier(archive_by_state_id, frontier)

    assert archive_by_state_id["archive:shared-root"].achieved_value == 0.0
    assert archive_by_state_id["archive:shared-window"].achieved_value == 2.0

    frontier = TrajectoryFrontier(TrajectoryFrontierConfig(max_size=1))
    frontier.insert(higher)
    archive_by_state_id = updater.refresh_achieved_values_from_frontier(archive_by_state_id, frontier)

    assert archive_by_state_id["archive:shared-window"].achieved_value == 5.0
    assert archive_by_state_id["archive:shared-window"].supporting_frontier_trajectory_ids == ["traj-two"]


def test_archive_store_round_trips_independent_snapshot(tmp_path: Path) -> None:
    """Archive snapshots should persist independently of trajectory/frontier artifacts."""

    updater = ArchiveUpdater()
    archive_store = ArchiveStore(tmp_path / "archive")
    trajectory = _episode_trajectory(
        "traj-archive",
        rewards=[0.0, 1.0],
        world_state_hashes=["archive-root", "archive-leaflet"],
    )
    archive_by_state_id = updater.ingest_episode_trajectory({}, trajectory)
    states = updater.sorted_states(archive_by_state_id)

    path = archive_store.write_states("episode-archive", states)
    restored = archive_store.read_episode("episode-archive")

    assert path.exists()
    assert [state.state_id for state in restored] == [state.state_id for state in states]
    assert restored[0].replay_metadata.replay_actions


def test_archive_identity_falls_back_to_deterministic_normalized_signature() -> None:
    """Missing engine hashes should use a deterministic normalized fallback state id."""

    first_id, first_strategy = archive_state_identity_from_fields(
        observation=" A forest path winds among trees. ",
        inventory_text="Leaflet",
        valid_actions=["west", "look", "north"],
        world_state_hash="unknown",
        fallback_namespace="episode-a:3",
    )
    second_id, second_strategy = archive_state_identity_from_fields(
        observation="A forest path winds among trees.",
        inventory_text="Leaflet",
        valid_actions=["north", "look", "west"],
        world_state_hash="unknown",
        fallback_namespace="episode-a:3",
    )

    assert first_strategy == "normalized_state_signature"
    assert second_strategy == "normalized_state_signature"
    assert first_id == second_id
    assert first_id.startswith("archive:fallback:")
