"""Tests for the Jericho environment wrapper.

TODO: add live Jericho integration tests in Linux CI if the project adopts containerized test runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


class FakeInventoryItem:
    """Simple object that mimics Jericho inventory items with a name attribute."""

    def __init__(self, name: str):
        self.name = name


class FakeJerichoBackend:
    """Minimal fake Jericho backend for unit tests."""

    def __init__(self, story_file: str, seed: int | None = None):
        self.story_file = story_file
        self.seed = seed
        self.closed = False
        self.score = 0
        self.moves = 0
        self.valid_actions = ["look", "open mailbox"]
        self.inventory = [FakeInventoryItem("lamp")]
        self.observation = "West of House."
        self.saved_state: tuple[Any, ...] | None = None

    def reset(self) -> tuple[str, dict[str, int]]:
        self.score = 0
        self.moves = 0
        self.valid_actions = ["look", "open mailbox"]
        self.inventory = [FakeInventoryItem("lamp")]
        self.observation = "West of House."
        return self.observation, {"score": self.score, "moves": self.moves}

    def step(self, action: str) -> tuple[str, float, bool, dict[str, int]]:
        self.moves += 1
        normalized = action.strip().lower()
        if normalized == "open mailbox":
            self.observation = "Opening the small mailbox reveals a leaflet."
            self.inventory = [FakeInventoryItem("lamp"), FakeInventoryItem("leaflet")]
            self.valid_actions = ["read leaflet", "look"]
            done = False
            reward = 1.0
        elif normalized == "read leaflet":
            self.observation = "It says welcome."
            self.valid_actions = ["look"]
            done = True
            reward = 0.0
        else:
            self.observation = "Nothing obvious happens."
            done = False
            reward = 0.0
        return self.observation, reward, done, {"score": self.score, "moves": self.moves}

    def get_score(self) -> int:
        return self.score

    def get_moves(self) -> int:
        return self.moves

    def get_inventory(self) -> list[FakeInventoryItem]:
        return self.inventory

    def get_valid_actions(self) -> list[str]:
        return list(self.valid_actions)

    def get_world_state_hash(self) -> str:
        return f"hash-{self.moves}"

    def get_state(self) -> tuple[Any, ...]:
        return ("ram", "stack", self.moves)

    def set_state(self, state: tuple[Any, ...]) -> None:
        self.saved_state = state

    def close(self) -> None:
        self.closed = True


class MissingOptionalJerichoBackend:
    """Fake backend without Jericho's optional helper methods."""

    def __init__(self, story_file: str, seed: int | None = None):
        self.story_file = story_file
        self.seed = seed
        self.moves = 0
        self.closed = False

    def reset(self) -> tuple[str, dict[str, int]]:
        self.moves = 0
        return "Start.", {"score": 2, "moves": 0}

    def step(self, action: str) -> tuple[str, float, bool, dict[str, int]]:
        self.moves += 1
        return f"Did {action}.", 0.5, False, {"score": 2, "moves": self.moves}

    def close(self) -> None:
        self.closed = True


def _build_config(tmp_path: Path, *, use_live_jericho: bool) -> ProjectConfig:
    """Create a minimal project config for environment tests."""

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
            docker_notes=["Linux-only Jericho runtime."],
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
        policy=PolicyConfig(),
        paths=PathsConfig(
            artifacts_dir=artifacts_dir,
            trajectory_dir=artifacts_dir / "trajectories",
            archive_dir=artifacts_dir / "archive",
            log_dir=artifacts_dir / "logs",
            summary_dir=artifacts_dir / "summaries",
        ),
        logging=LoggingConfig(),
        experiment=ExperimentConfig(game_id="zork1", stub_episode_length=3),
    )


def test_live_wrapper_collects_inventory_actions_and_snapshot(tmp_path: Path) -> None:
    """The wrapper should normalize Jericho helpers into typed state and step objects."""

    env = JerichoEnv(_build_config(tmp_path, use_live_jericho=True), env_factory=FakeJerichoBackend)

    initial_state = env.reset(seed=7)

    assert initial_state.observation == "West of House."
    assert initial_state.inventory_text == "lamp"
    assert initial_state.valid_actions == ["look", "open mailbox"]
    assert initial_state.world_state_snapshot is not None
    assert initial_state.world_state_snapshot.restore_strategy == "native_snapshot"
    assert initial_state.world_state_snapshot.native_state == ("ram", "stack", 0)

    transition = env.step("open mailbox")

    assert transition.action == "open mailbox"
    assert transition.step_index == 0
    assert transition.reward == 1.0
    assert transition.inventory_text == "lamp, leaflet"
    assert transition.valid_actions == ["read leaflet", "look"]
    assert transition.world_state_hash == "hash-1"
    assert transition.world_state_snapshot is not None
    assert transition.world_state_snapshot.replay_actions == ["open mailbox"]


def test_restore_snapshot_uses_set_state_when_available(tmp_path: Path) -> None:
    """Exact Jericho snapshots should restore through `set_state()` when exposed."""

    backend = FakeJerichoBackend("unused")
    env = JerichoEnv(
        _build_config(tmp_path, use_live_jericho=True),
        env_factory=lambda story_file, seed=None: backend,
    )
    env.reset(seed=1)
    transition = env.step("open mailbox")
    snapshot = transition.world_state_snapshot

    assert snapshot is not None

    restored = env.restore_snapshot(snapshot)

    assert backend.saved_state == snapshot.jericho_state
    assert restored.observation == snapshot.observation
    assert restored.world_state_snapshot is not None


def test_wrapper_gracefully_handles_missing_optional_jericho_methods(tmp_path: Path) -> None:
    """Optional Jericho helpers should degrade cleanly when unavailable."""

    env = JerichoEnv(
        _build_config(tmp_path, use_live_jericho=True),
        env_factory=MissingOptionalJerichoBackend,
    )

    initial_state = env.reset(seed=2)
    transition = env.step("look")

    assert initial_state.inventory_text == "Inventory unavailable."
    assert initial_state.valid_actions == []
    assert initial_state.world_state_snapshot is not None
    assert initial_state.world_state_snapshot.restore_strategy == "action_replay_fallback"
    assert transition.score == 2
    assert transition.valid_actions == []


def test_stub_mode_uses_action_replay_fallback_snapshots(tmp_path: Path) -> None:
    """The deterministic stub should expose replay-fallback snapshots without Jericho."""

    env = JerichoEnv(_build_config(tmp_path, use_live_jericho=False))

    initial_state = env.reset(seed=3)
    transition = env.step("look")

    assert initial_state.valid_actions == ["look", "open mailbox", "read leaflet", "quit"]
    assert transition.world_state_snapshot is not None
    assert transition.world_state_snapshot.restore_strategy == "action_replay_fallback"
    assert transition.step_index == 0
