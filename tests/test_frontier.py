"""Tests for heuristic frontier bookkeeping and native snapshot retention.

TODO: add policy-level tests once the selector starts using richer frontier signals.
"""

from __future__ import annotations

import pytest

from zork_agent.memory.archive_updater import ArchiveUpdater
from zork_agent.memory.frontier import (
    FrontierQueue,
    FrontierScoringConfig,
    TrajectoryFrontier,
    TrajectoryFrontierConfig,
)
from zork_agent.types import EpisodeTrajectory, FrontierEntry, ReplayMetadata, SavedNode, StateCandidate, TrajectoryStep


def _saved_node(state_id: str, *, native_state: tuple[object, ...] | None) -> SavedNode:
    """Create a small saved node fixture."""

    return SavedNode(
        state_id=state_id,
        episode_id="episode-001",
        step_index=0,
        native_state=native_state,
        action_prefix=["look"],
        score=1,
        observation="West of House.",
        inventory_text="lamp",
        summary_text=f"summary-{state_id}",
    )


def _episode_trajectory(
    episode_id: str,
    *,
    rewards: list[float],
    scores: list[int],
    world_state_hashes: list[str],
    state_cluster_ids: list[str] | None = None,
) -> EpisodeTrajectory:
    """Create a small typed episode trajectory fixture for frontier tests."""

    cumulative_reward = 0.0
    steps: list[TrajectoryStep] = []
    cluster_ids = state_cluster_ids or ["" for _ in rewards]
    for index, reward in enumerate(rewards):
        cumulative_reward += reward
        steps.append(
            TrajectoryStep(
                episode_id=episode_id,
                step_index=index,
                action=f"action-{index}",
                observation=f"Observation {index}",
                reward=reward,
                cumulative_reward=cumulative_reward,
                done=index == len(rewards) - 1,
                score=scores[index],
                moves=index + 1,
                world_state_hash=world_state_hashes[index],
                state_cluster_id=cluster_ids[index],
                valid_actions=["look", "north"],
                world_state_snapshot={
                    "restore_strategy": "action_replay_fallback",
                    "replay_actions": [f"action-{step_index}" for step_index in range(index + 1)],
                },
            )
        )

    trajectory = EpisodeTrajectory(
        episode_id=episode_id,
        root_state_id=f"{episode_id}:root:{world_state_hashes[0]}",
        steps=steps,
        max_cumulative_reward_achieved=max(step.cumulative_reward for step in steps) if steps else 0.0,
        final_score=steps[-1].score if steps else 0,
        final_done=steps[-1].done if steps else False,
        replay_metadata=ReplayMetadata(
            replay_actions=[step.action for step in steps],
            world_state_hash=world_state_hashes[0],
        ),
    )
    trajectory.validate_invariants()
    return trajectory


def test_frontier_ranks_entries_by_explicit_weighted_priority() -> None:
    """Ranking should follow the configured heuristic formula."""

    frontier = FrontierQueue(
        max_size=3,
        scoring=FrontierScoringConfig(
            score_weight=1.0,
            novelty_weight=2.0,
            recent_gain_weight=1.0,
            depth_weight=0.1,
            loop_penalty_weight=1.0,
            reversible_state_penalty_weight=1.0,
            revisit_saturation_penalty_weight=1.0,
            snapshot_retention_limit=3,
        ),
    )
    frontier.add(FrontierEntry(state_id="baseline", score=4.0, depth=1))
    frontier.add(FrontierEntry(state_id="novel-branch", score=2.0, novelty=1.0, recent_gain=1.0, depth=2))
    frontier.add(FrontierEntry(state_id="deep-low-value", score=1.0, novelty=0.1, recent_gain=0.0, depth=8))

    snapshot = frontier.snapshot()

    assert [entry.state_id for entry in snapshot] == ["novel-branch", "baseline", "deep-low-value"]
    assert snapshot[0].priority > snapshot[1].priority > snapshot[2].priority


def test_frontier_deduplicates_near_identical_entries_and_keeps_better_one() -> None:
    """Near-identical entries should collapse to the higher-priority candidate."""

    frontier = FrontierQueue(max_size=3)
    frontier.add(
        FrontierEntry(
            state_id="older",
            score=1.0,
            depth=2,
            world_state_hash="same-state",
            observation="You are standing in an open field west of a white house.",
        )
    )
    frontier.add(
        FrontierEntry(
            state_id="better",
            score=3.0,
            depth=1,
            world_state_hash="same-state",
            observation="You are standing in an open field west of a white house.",
            recent_gain=1.0,
        )
    )

    snapshot = frontier.snapshot()

    assert len(snapshot) == 1
    assert snapshot[0].state_id == "better"
    assert snapshot[0].priority > 0


def test_frontier_penalizes_reversible_loop_candidates() -> None:
    """Loop penalties should reduce frontier priority for otherwise similar states."""

    frontier = FrontierQueue(max_size=3)
    frontier.add(
        FrontierEntry(
            state_id="stable",
            score=1.0,
            depth=2,
            novelty=0.5,
            recent_gain=0.0,
            loop_penalty=0.0,
        )
    )
    frontier.add(
        FrontierEntry(
            state_id="oscillating",
            score=1.0,
            depth=2,
            novelty=0.5,
            recent_gain=0.0,
            loop_penalty=2.0,
        )
    )

    snapshot = frontier.snapshot()

    assert [entry.state_id for entry in snapshot] == ["stable", "oscillating"]
    assert snapshot[0].priority > snapshot[1].priority


def test_frontier_mailbox_toggle_family_collapses_after_repeated_zero_gain_revisits() -> None:
    """Repeated dead-end revisits should saturate a reversible mailbox-open/closed family."""

    frontier = FrontierQueue(
        max_size=4,
        scoring=FrontierScoringConfig(
            novelty_weight=2.0,
            recent_gain_weight=1.0,
            loop_penalty_weight=1.0,
            reversible_state_penalty_weight=1.0,
            revisit_saturation_penalty_weight=1.0,
            family_revisit_saturation_weight=1.0,
            snapshot_retention_limit=2,
        ),
    )
    mailbox_open = FrontierEntry(
        state_id="mailbox-open",
        score=0.0,
        depth=2,
        novelty=1.0,
        observation="Opening the small mailbox reveals a leaflet.",
        inventory_text="Inventory empty.",
        valid_actions=["close mailbox", "take leaflet", "look"],
        state_family_key="toggle:mailbox",
        oscillating_pair_member=True,
        trivial_reversible_change=True,
    )
    mailbox_closed = FrontierEntry(
        state_id="mailbox-closed",
        score=0.0,
        depth=2,
        novelty=1.0,
        observation="The small mailbox is closed.",
        inventory_text="Inventory empty.",
        valid_actions=["open mailbox", "look"],
        state_family_key="toggle:mailbox",
        oscillating_pair_member=True,
        trivial_reversible_change=True,
    )
    novel_room = FrontierEntry(
        state_id="forest-path",
        score=0.0,
        depth=3,
        novelty=0.9,
        recent_gain=0.0,
        observation="You are in a forest path beside a stream.",
        inventory_text="Inventory empty.",
        valid_actions=["north", "south", "examine stream"],
        state_family_key="state:forest|path|stream",
    )

    frontier.add(mailbox_open)
    frontier.add(mailbox_closed)
    frontier.add(novel_room)

    frontier.note_revisit_outcome(mailbox_open, made_progress=False)
    frontier.note_revisit_outcome(mailbox_closed, made_progress=False)
    frontier.note_revisit_outcome(mailbox_open, made_progress=False)

    snapshot = frontier.snapshot()

    assert snapshot[0].state_id == "forest-path"
    mailbox_entries = {entry.state_id: entry for entry in snapshot if entry.state_id.startswith("mailbox-")}
    assert mailbox_entries["mailbox-open"].revisit_saturation_penalty > 0.0
    assert mailbox_entries["mailbox-open"].family_no_progress_revisit_count >= 2
    assert mailbox_entries["mailbox-open"].effective_novelty < mailbox_entries["mailbox-open"].novelty
    assert mailbox_entries["mailbox-closed"].priority < snapshot[0].priority


def test_frontier_real_novel_states_still_rank_well() -> None:
    """Non-reversible novel states should retain strong effective novelty."""

    frontier = FrontierQueue(
        max_size=3,
        scoring=FrontierScoringConfig(
            novelty_weight=2.0,
            reversible_toggle_novelty_scale=0.1,
            trivial_observation_novelty_scale=0.25,
            oscillating_pair_novelty_scale=0.1,
        ),
    )
    frontier.add(
        FrontierEntry(
            state_id="toggle-mailbox",
            score=0.0,
            depth=1,
            novelty=1.2,
            observation="Opening the small mailbox reveals a leaflet.",
            valid_actions=["close mailbox", "take leaflet"],
            state_family_key="toggle:mailbox",
            oscillating_pair_member=True,
            trivial_reversible_change=True,
        )
    )
    frontier.add(
        FrontierEntry(
            state_id="living-room",
            score=0.0,
            depth=2,
            novelty=0.9,
            recent_gain=0.0,
            observation="You are in the living room. A trophy case stands here.",
            valid_actions=["open case", "north", "examine trophy case"],
            state_family_key="state:case|living|room|trophy",
        )
    )

    snapshot = frontier.snapshot()

    assert snapshot[0].state_id == "living-room"
    assert snapshot[0].effective_novelty > snapshot[1].effective_novelty


def test_frontier_score_gaining_states_dominate_reversible_novelty() -> None:
    """Real score-gaining states should outrank flashy but reversible novelty."""

    frontier = FrontierQueue(
        max_size=3,
        scoring=FrontierScoringConfig(
            score_weight=1.5,
            recent_gain_weight=1.0,
            novelty_weight=1.0,
            reversible_state_penalty_weight=1.0,
            snapshot_retention_limit=2,
        ),
    )
    frontier.add(
        FrontierEntry(
            state_id="mailbox-open",
            score=0.0,
            depth=1,
            novelty=1.5,
            observation="Opening the small mailbox reveals a leaflet.",
            valid_actions=["close mailbox", "take leaflet"],
            state_family_key="toggle:mailbox",
            oscillating_pair_member=True,
            trivial_reversible_change=True,
        )
    )
    frontier.add(
        FrontierEntry(
            state_id="egg-room",
            score=5.0,
            depth=4,
            novelty=0.4,
            recent_gain=5.0,
            observation="You are beneath the egg. The score has increased.",
            valid_actions=["take egg", "down"],
            state_family_key="state:egg|score",
        )
    )

    snapshot = frontier.snapshot()

    assert snapshot[0].state_id == "egg-room"
    assert snapshot[0].priority > snapshot[1].priority


def test_frontier_top_k_replay_targets_return_candidates_with_and_without_snapshots() -> None:
    """Top-k retrieval should preserve whether a replay target has a native snapshot."""

    frontier = FrontierQueue(max_size=5, scoring=FrontierScoringConfig(snapshot_retention_limit=2))
    frontier.add_candidate(
        StateCandidate(
            state_id="state-1",
            episode_id="episode-001",
            step_index=1,
            observation="A mailbox is here.",
            score=1.0,
            depth=2,
            replay_actions=["look", "open mailbox"],
            novelty=0.2,
            recent_gain=1.0,
            summary_text="mailbox branch",
            saved_node=_saved_node("state-1", native_state=("native", "mailbox")),
        )
    )
    frontier.add_candidate(
        StateCandidate(
            state_id="state-2",
            episode_id="episode-001",
            step_index=2,
            observation="You read the leaflet.",
            score=2.0,
            depth=3,
            replay_actions=["look", "open mailbox", "read leaflet"],
            novelty=0.4,
            recent_gain=1.0,
            summary_text="leaflet branch",
            saved_node=_saved_node("state-2", native_state=None),
        )
    )

    targets = frontier.top_k_replay_targets(2)

    assert [target.state_id for target in targets] == ["state-2", "state-1"]
    assert targets[0].saved_node is not None
    assert targets[0].saved_node.has_native_state is False
    assert targets[1].saved_node is not None
    assert targets[1].saved_node.has_native_state is True


def test_frontier_prunes_native_snapshots_beyond_retention_limit() -> None:
    """Only the top configured entries should keep native snapshot payloads."""

    frontier = FrontierQueue(max_size=4, scoring=FrontierScoringConfig(snapshot_retention_limit=1))
    frontier.add(
        FrontierEntry(
            state_id="top",
            score=5.0,
            depth=1,
            saved_node=_saved_node("top", native_state=("native", "top")),
        )
    )
    frontier.add(
        FrontierEntry(
            state_id="second",
            score=3.0,
            depth=1,
            saved_node=_saved_node("second", native_state=("native", "second")),
        )
    )
    frontier.add(
        FrontierEntry(
            state_id="no-snapshot",
            score=1.0,
            depth=1,
            saved_node=_saved_node("no-snapshot", native_state=None),
        )
    )

    snapshot = frontier.snapshot()

    assert snapshot[0].saved_node is not None
    assert snapshot[0].saved_node.has_native_state is True
    assert snapshot[1].saved_node is not None
    assert snapshot[1].saved_node.has_native_state is False
    assert snapshot[2].saved_node is not None
    assert snapshot[2].saved_node.has_native_state is False


def test_frontier_tie_breaking_is_stable_for_equal_priority_entries() -> None:
    """Equal-priority entries should sort deterministically by episode, step, and state id."""

    frontier = FrontierQueue(max_size=4)
    frontier.add(FrontierEntry(state_id="state-b", episode_id="episode-002", step_index=1, score=1.0, depth=1))
    frontier.add(FrontierEntry(state_id="state-a", episode_id="episode-001", step_index=2, score=1.0, depth=1))
    frontier.add(FrontierEntry(state_id="state-c", episode_id="episode-001", step_index=0, score=1.0, depth=1))

    assert [entry.state_id for entry in frontier.snapshot()] == ["state-c", "state-a", "state-b"]


def test_frontier_rejects_invalid_configuration_and_nonfinite_priority() -> None:
    """Frontier config and priorities should fail early for invalid values."""

    with pytest.raises(ValueError, match="similarity_threshold"):
        FrontierScoringConfig(similarity_threshold=1.5)

    frontier = FrontierQueue(max_size=2)
    with pytest.raises(ValueError, match="priority must be finite"):
        frontier.add(FrontierEntry(state_id="bad", score=float("nan"), depth=1))


def test_trajectory_frontier_collects_bottleneck_ids_from_generator_inputs() -> None:
    """Generator-based bottleneck/state-id collection should materialize string values, not generator reprs."""

    trajectory = _episode_trajectory(
        "traj-generator",
        rewards=[0.0, 1.0],
        scores=[0, 1],
        world_state_hashes=["root", "leaflet"],
    )
    trajectory.summary_fields.bottleneck_step_indices = [0, 1]
    frontier = TrajectoryFrontier(TrajectoryFrontierConfig(max_size=2))

    entry = frontier.insert(trajectory)

    assert "archive:root" in entry.bottleneck_state_ids
    assert "archive:leaflet" in entry.bottleneck_state_ids
    assert all("generator object" not in value for value in entry.bottleneck_state_ids)


def test_frontier_deprioritizes_movement_only_forest_clusters() -> None:
    """Near-identical forest movement states should decay behind object-bearing states."""

    frontier = FrontierQueue(
        max_size=4,
        scoring=FrontierScoringConfig(
            novelty_weight=2.0,
            movement_penalty_weight=1.0,
            room_text_only_penalty_weight=1.0,
            cluster_revisit_saturation_weight=1.0,
        ),
    )
    forest_a = FrontierEntry(
        state_id="forest-a",
        score=0.0,
        depth=3,
        novelty=1.0,
        observation="Forest path among trees.",
        inventory_text="Inventory empty.",
        valid_actions=["go around trees", "west", "look"],
        state_cluster_id="region:forest",
        region_novelty_score=0.1,
        movement_penalty=1.5,
        room_text_only_gain=1.0,
    )
    forest_b = FrontierEntry(
        state_id="forest-b",
        score=0.0,
        depth=3,
        novelty=1.0,
        observation="A forest path winds among trees.",
        inventory_text="Inventory empty.",
        valid_actions=["go around trees", "west", "look"],
        state_cluster_id="region:forest",
        region_novelty_score=0.05,
        movement_penalty=1.5,
        room_text_only_gain=1.0,
    )
    mailbox = FrontierEntry(
        state_id="mailbox-leaflet",
        score=0.0,
        depth=2,
        novelty=0.8,
        recent_gain=0.0,
        observation="Opening the small mailbox reveals a leaflet.",
        inventory_text="Inventory empty.",
        valid_actions=["take leaflet", "close mailbox", "look"],
        state_cluster_id="region:mailbox",
        region_novelty_score=1.0,
        affordance_gain=1,
    )

    frontier.add(forest_a)
    frontier.add(forest_b)
    frontier.add(mailbox)
    frontier.note_revisit_outcome(forest_a, made_progress=False)
    frontier.note_revisit_outcome(forest_b, made_progress=False)

    snapshot = frontier.snapshot()

    assert snapshot[0].state_id == "mailbox-leaflet"
    forest_entries = [entry for entry in snapshot if entry.state_cluster_id == "region:forest"]
    assert all(entry.effective_novelty < entry.novelty for entry in forest_entries)
    assert all(entry.revisit_saturation_penalty > 0.0 for entry in forest_entries)


def test_trajectory_frontier_retains_top_k_complete_trajectories_by_value() -> None:
    """The paper-faithful frontier should keep the highest-value complete trajectories."""

    frontier = TrajectoryFrontier(TrajectoryFrontierConfig(max_size=2))
    low = _episode_trajectory(
        "episode-low",
        rewards=[0.0, 1.0],
        scores=[0, 1],
        world_state_hashes=["hash-a", "hash-b"],
    )
    high = _episode_trajectory(
        "episode-high",
        rewards=[0.0, 2.0],
        scores=[0, 2],
        world_state_hashes=["hash-c", "hash-d"],
    )
    mid = _episode_trajectory(
        "episode-mid",
        rewards=[0.0, 1.5],
        scores=[0, 1],
        world_state_hashes=["hash-e", "hash-f"],
    )

    frontier.insert(low)
    frontier.insert(high)
    frontier.insert(mid)

    retained_ids = [entry.trajectory_id for entry in frontier.snapshot_entries()]

    assert retained_ids == ["episode-high", "episode-mid"]
    assert [trajectory.trajectory_id for trajectory in frontier.top_k_trajectories(2)] == retained_ids


def test_trajectory_frontier_tie_breaking_is_stable_by_insertion_order() -> None:
    """Equal-valued trajectories should retain stable ordering by insertion order."""

    frontier = TrajectoryFrontier(TrajectoryFrontierConfig(max_size=3))
    first = _episode_trajectory(
        "episode-first",
        rewards=[0.0, 1.0],
        scores=[0, 1],
        world_state_hashes=["hash-first-0", "hash-first-1"],
    )
    second = _episode_trajectory(
        "episode-second",
        rewards=[0.0, 1.0],
        scores=[0, 1],
        world_state_hashes=["hash-second-0", "hash-second-1"],
    )

    frontier.insert(first)
    frontier.insert(second)

    assert [entry.trajectory_id for entry in frontier.snapshot_entries()] == ["episode-first", "episode-second"]


def test_frontier_achieved_value_refresh_uses_the_independent_archive() -> None:
    """Achieved-value refresh should come from the archive updater, not the frontier itself."""

    frontier = TrajectoryFrontier(TrajectoryFrontierConfig(max_size=3))
    archive_updater = ArchiveUpdater()
    lower = _episode_trajectory(
        "episode-one",
        rewards=[0.0, 2.0],
        scores=[0, 2],
        world_state_hashes=["hash-root", "hash-window"],
        state_cluster_ids=["region:field", "region:window"],
    )
    higher = _episode_trajectory(
        "episode-two",
        rewards=[0.0, 5.0],
        scores=[0, 5],
        world_state_hashes=["hash-root", "hash-window"],
        state_cluster_ids=["region:field", "region:window"],
    )

    frontier.insert(lower)
    frontier.insert(higher)
    archive_by_state_id = archive_updater.ingest_episode_trajectories({}, [lower, higher])
    archive_by_state_id = archive_updater.refresh_achieved_values_from_frontier(archive_by_state_id, frontier)

    assert archive_by_state_id["archive:hash-root"].achieved_value == 0.0
    assert archive_by_state_id["archive:hash-window"].achieved_value == 5.0
    assert archive_by_state_id["archive:hash-window"].frontier_support_count == 2
    assert archive_by_state_id["archive:hash-window"].supporting_frontier_trajectory_ids == [
        "episode-two",
        "episode-one",
    ]


def test_trajectory_frontier_analysis_view_returns_complete_trajectories() -> None:
    """Frontier analysis retrieval should expose the retained full trajectories."""

    frontier = TrajectoryFrontier(TrajectoryFrontierConfig(max_size=2))
    first = _episode_trajectory(
        "episode-analysis-1",
        rewards=[0.0, 1.0, 1.0],
        scores=[0, 1, 2],
        world_state_hashes=["hash-1", "hash-2", "hash-3"],
    )
    second = _episode_trajectory(
        "episode-analysis-2",
        rewards=[0.0, 3.0],
        scores=[0, 3],
        world_state_hashes=["hash-4", "hash-5"],
    )

    frontier.insert(first, novelty_score=0.2, diversity_score=0.4)
    frontier.insert(second, novelty_score=0.1, diversity_score=0.1)

    analysis_payload = frontier.trajectories_for_analysis()

    assert [trajectory.trajectory_id for trajectory in analysis_payload] == ["episode-analysis-2", "episode-analysis-1"]
    assert analysis_payload[0].steps[-1].cumulative_reward == 3.0
