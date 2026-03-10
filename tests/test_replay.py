"""Tests for Jericho-first restoration utilities.

TODO: add higher-level branch rollout tests once local exploration starts expanding frontier nodes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zork_agent.config import (
    ExperimentConfig,
    LLMConfig,
    LoggingConfig,
    PathsConfig,
    PolicyConfig,
    ProjectConfig,
    PromptConfig,
    RuntimeConfig,
)
from zork_agent.env.jericho_env import JerichoEnv
from zork_agent.env.replay import ReplaySession, restore_saved_node
from zork_agent.types import ReplayDivergenceReason, RestoreMode, SavedNode, TrajectoryStep
from zork_agent.utils.serialization import write_jsonl


class _FakeInventoryItem:
    """Small object that mimics Jericho inventory items."""

    def __init__(self, name: str):
        self.name = name


class NativeSnapshotBackend:
    """Fake Jericho backend with working native save/restore."""

    def __init__(self, story_file: str, seed: int | None = None):
        self.story_file = story_file
        self.seed = seed
        self.closed = False
        self.saved_state: tuple[object, ...] | None = None
        self._snapshot_id = "west"
        self._state_table = {
            "west": {
                "observation": "West of House.",
                "inventory": [_FakeInventoryItem("lamp")],
                "valid_actions": ["look", "open mailbox"],
                "score": 0,
                "moves": 0,
            },
            "mailbox": {
                "observation": "Opening the small mailbox reveals a leaflet.",
                "inventory": [_FakeInventoryItem("lamp"), _FakeInventoryItem("leaflet")],
                "valid_actions": ["read leaflet", "look"],
                "score": 1,
                "moves": 1,
            },
        }
        self._apply_snapshot("west")

    def reset(self) -> tuple[str, dict[str, int]]:
        self._apply_snapshot("west")
        return self.observation, {"score": self.score, "moves": self.moves}

    def step(self, action: str) -> tuple[str, float, bool, dict[str, int]]:
        if action.strip().lower() == "open mailbox":
            self._apply_snapshot("mailbox")
            reward = 1.0
            done = False
        else:
            reward = 0.0
            done = False
        return self.observation, reward, done, {"score": self.score, "moves": self.moves}

    def get_score(self) -> int:
        return self.score

    def get_moves(self) -> int:
        return self.moves

    def get_inventory(self) -> list[_FakeInventoryItem]:
        return self.inventory

    def get_valid_actions(self) -> list[str]:
        return list(self.valid_actions)

    def get_world_state_hash(self) -> str:
        return f"hash-{self._snapshot_id}"

    def get_state(self) -> tuple[object, ...]:
        return ("native", self._snapshot_id)

    def set_state(self, state: tuple[object, ...]) -> None:
        self.saved_state = state
        self._apply_snapshot(str(state[1]))

    def close(self) -> None:
        self.closed = True

    def _apply_snapshot(self, snapshot_id: str) -> None:
        payload = self._state_table[snapshot_id]
        self._snapshot_id = snapshot_id
        self.observation = str(payload["observation"])
        self.inventory = list(payload["inventory"])
        self.valid_actions = list(payload["valid_actions"])
        self.score = int(payload["score"])
        self.moves = int(payload["moves"])


class BrokenNativeRestoreBackend(NativeSnapshotBackend):
    """Fake backend whose `set_state()` leaves the env in the wrong observation."""

    def set_state(self, state: tuple[object, ...]) -> None:
        self.saved_state = state
        self._apply_snapshot("west")


def _build_config(tmp_path: Path, *, use_live_jericho: bool) -> ProjectConfig:
    """Create a minimal project config for replay tests."""

    rom_path = tmp_path / "games" / "zork1.z5"
    rom_path.parent.mkdir(parents=True, exist_ok=True)
    rom_path.write_text("fake rom", encoding="utf-8")
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "system.txt",
        "action.txt",
        "trajectory.txt",
        "reflection.txt",
        "selection.txt",
    ):
        (prompt_dir / name).write_text(name, encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    return ProjectConfig(
        runtime=RuntimeConfig(
            jericho_game_path=rom_path,
            use_live_jericho=use_live_jericho,
            docker_notes=["Replay tests use fake Jericho backends."],
        ),
        llm=LLMConfig(),
        prompts=PromptConfig(
            directory=prompt_dir,
            system_file="system.txt",
            action_proposal_file="action.txt",
            trajectory_analysis_file="trajectory.txt",
            local_reflection_file="reflection.txt",
            state_selection_file="selection.txt",
        ),
        policy=PolicyConfig(snapshot_retention_limit=2),
        paths=PathsConfig(
            artifacts_dir=artifacts_dir,
            trajectory_dir=artifacts_dir / "trajectories",
            archive_dir=artifacts_dir / "archive",
            log_dir=artifacts_dir / "logs",
            summary_dir=artifacts_dir / "summaries",
        ),
        logging=LoggingConfig(),
        experiment=ExperimentConfig(game_id="zork1", stub_episode_length=4),
    )


def _build_saved_node(tmp_path: Path) -> SavedNode:
    """Create a saved node from a live fake-Jericho run."""

    env = JerichoEnv(_build_config(tmp_path, use_live_jericho=True), env_factory=NativeSnapshotBackend)
    env.reset(seed=7)
    transition = env.step("open mailbox")
    snapshot = transition.world_state_snapshot
    assert snapshot is not None
    env.close()
    return SavedNode(
        state_id="episode-001:0:hash-mailbox",
        episode_id="episode-001",
        step_index=0,
        native_state=snapshot.native_state,
        action_prefix=list(snapshot.replay_actions),
        score=transition.score,
        observation=transition.observation,
        inventory_text=transition.inventory_text,
        world_state_hash=transition.world_state_hash,
        valid_actions=list(transition.valid_actions),
        summary_text="mailbox branch",
    )


def test_replay_session_loads_steps_and_summary(tmp_path: Path) -> None:
    """Replay sessions should load trajectory steps and format a summary."""

    path = tmp_path / "episode.jsonl"
    steps = [
        TrajectoryStep(
            episode_id="episode-001",
            step_index=0,
            action="look",
            observation="West of House.",
            reward=0.0,
            done=False,
            score=0,
            moves=1,
            inventory_text="lamp",
            valid_actions=["look", "open mailbox"],
            world_state_hash="hash-west",
            world_state_snapshot={"restore_strategy": "action_replay_fallback", "replay_actions": ["look"]},
        ),
        TrajectoryStep(
            episode_id="episode-001",
            step_index=1,
            action="open mailbox",
            observation="Opening the small mailbox reveals a leaflet.",
            reward=1.0,
            done=False,
            score=1,
            moves=2,
            inventory_text="lamp, leaflet",
            valid_actions=["read leaflet", "look"],
            world_state_hash="hash-mailbox",
            world_state_snapshot={
                "restore_strategy": "action_replay_fallback",
                "replay_actions": ["look", "open mailbox"],
            },
        ),
    ]
    write_jsonl(path, [step.to_record() for step in steps])

    replay = ReplaySession.from_path(path)

    assert replay.episode_id == "episode-001"
    assert replay.total_reward() == 1.0
    assert replay.final_score() == 1
    assert "preview_actions=[look, open mailbox]" in replay.format_summary()


def test_replay_session_uses_actual_step_index_not_list_position() -> None:
    """Replay helpers should address stored `step_index`, not assume dense list positions."""

    replay = ReplaySession.from_steps(
        [
            TrajectoryStep(
                episode_id="episode-001",
                step_index=2,
                action="look",
                observation="West of House.",
                reward=0.0,
                done=False,
                score=0,
                moves=1,
                inventory_text="lamp",
                valid_actions=["look", "open mailbox"],
                world_state_hash="hash-west",
            ),
            TrajectoryStep(
                episode_id="episode-001",
                step_index=4,
                action="open mailbox",
                observation="Opening the small mailbox reveals a leaflet.",
                reward=1.0,
                done=False,
                score=1,
                moves=2,
                inventory_text="lamp, leaflet",
                valid_actions=["read leaflet", "look"],
                world_state_hash="hash-mailbox",
                world_state_snapshot={"replay_actions": ["look", "open mailbox"]},
            ),
        ]
    )

    selected_steps = replay.steps_up_to(4)
    saved_node = replay.saved_node_at(4)

    assert [step.step_index for step in selected_steps] == [2, 4]
    assert saved_node is not None
    assert saved_node.step_index == 4
    assert saved_node.action_prefix == ["look", "open mailbox"]


def test_native_snapshot_restore_succeeds_when_set_state_validates(tmp_path: Path) -> None:
    """Native `set_state()` should be the preferred restoration path."""

    saved_node = _build_saved_node(tmp_path)
    env = JerichoEnv(_build_config(tmp_path, use_live_jericho=True), env_factory=NativeSnapshotBackend)
    env.reset(seed=9)

    result = restore_saved_node(env, saved_node)

    assert result.success is True
    assert result.restore_mode is RestoreMode.NATIVE_SNAPSHOT
    assert result.native_restore_attempted is True
    assert result.replayed_action_count == 0
    assert result.divergence_reason is ReplayDivergenceReason.NONE
    assert result.final_observation == "Opening the small mailbox reveals a leaflet."
    assert result.validation is not None
    assert result.validation.is_valid is True


def test_restore_falls_back_to_action_replay_when_native_validation_fails(tmp_path: Path) -> None:
    """Failed native validation should trigger reset-plus-replay fallback."""

    saved_node = _build_saved_node(tmp_path)
    env = JerichoEnv(_build_config(tmp_path, use_live_jericho=True), env_factory=BrokenNativeRestoreBackend)
    env.reset(seed=11)

    result = restore_saved_node(env, saved_node)

    assert result.success is True
    assert result.restore_mode is RestoreMode.ACTION_REPLAY_FALLBACK
    assert result.native_restore_attempted is True
    assert result.replayed_action_count == 1
    assert result.final_score == 1
    assert result.final_observation == "Opening the small mailbox reveals a leaflet."


def test_restore_to_step_without_native_snapshot_uses_action_replay(tmp_path: Path) -> None:
    """Trajectory-based restoration without a native snapshot should use replay fallback."""

    session = ReplaySession.from_steps(
        [
            TrajectoryStep(
                episode_id="episode-002",
                step_index=0,
                action="look",
                observation="You are standing in an open field west of a white house, near a small mailbox.",
                reward=0.0,
                done=False,
                score=0,
                moves=1,
                inventory_text="You are empty-handed.",
                valid_actions=["look", "open mailbox", "read leaflet", "quit"],
                world_state_hash="stub-west-of-house",
                world_state_snapshot={
                    "restore_strategy": "action_replay_fallback",
                    "replay_actions": ["look"],
                },
            )
        ]
    )
    env = JerichoEnv(_build_config(tmp_path, use_live_jericho=False))

    result = session.restore_to_step(env, 0)

    assert result.success is True
    assert result.restore_mode is RestoreMode.ACTION_REPLAY_FALLBACK
    assert result.replayed_action_count == 1
    assert result.final_observation == "You are standing in an open field west of a white house, near a small mailbox."


def test_empty_restore_returns_reset_state(tmp_path: Path) -> None:
    """Restoring without a saved node should just return the reset state."""

    session = ReplaySession.from_steps([])
    env = JerichoEnv(_build_config(tmp_path, use_live_jericho=False))

    result = session.restore(env)

    assert result.success is True
    assert result.restore_mode is RestoreMode.RESET_ONLY
    assert result.replayed_action_count == 0
    assert result.final_observation.startswith("Stub reset:")


def test_replay_session_rejects_unsorted_or_duplicate_step_indices() -> None:
    """Replay input should be validated before restore logic relies on it."""

    with pytest.raises(ValueError, match="sorted by step_index"):
        ReplaySession.from_steps(
            [
                TrajectoryStep(
                    episode_id="episode-001",
                    step_index=1,
                    action="open mailbox",
                    observation="leaflet",
                    reward=0.0,
                    done=False,
                    score=1,
                    moves=2,
                    world_state_hash="hash-2",
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
                    world_state_hash="hash-1",
                ),
            ]
        )
