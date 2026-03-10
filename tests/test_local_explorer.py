"""Tests for shallow local branching from restored states.

TODO: add longer-horizon tests once the local explorer starts feeding evaluator metrics.
"""

from __future__ import annotations

from pathlib import Path

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
from zork_agent.llm.prompts import PromptManager
from zork_agent.policy.action_generator import ActionGenerator
from zork_agent.policy.local_explorer import LocalExplorer
from zork_agent.types import ActionProposal, BranchTerminationReason, TextGameState


class _FakeInventoryItem:
    """Small object that mimics Jericho inventory items."""

    def __init__(self, name: str):
        self.name = name


class BranchingBackend:
    """Fake backend with native snapshots and a small local branching structure."""

    def __init__(self, story_file: str, seed: int | None = None):
        self.story_file = story_file
        self.seed = seed
        self.closed = False
        self.set_state_calls = 0
        self._snapshot_id = "west"
        self._state_table = {
            "west": {
                "observation": "West of House.",
                "inventory": [_FakeInventoryItem("lamp")],
                "valid_actions": ["open mailbox", "examine house", "look"],
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
            "leaflet": {
                "observation": "The leaflet welcomes you to Zork.",
                "inventory": [_FakeInventoryItem("lamp"), _FakeInventoryItem("leaflet")],
                "valid_actions": ["look"],
                "score": 2,
                "moves": 2,
            },
            "house": {
                "observation": "The white house is boarded shut.",
                "inventory": [_FakeInventoryItem("lamp")],
                "valid_actions": ["look"],
                "score": 0,
                "moves": 1,
            },
        }
        self._apply_snapshot("west")

    def reset(self) -> tuple[str, dict[str, int]]:
        self._apply_snapshot("west")
        return self.observation, {"score": self.score, "moves": self.moves}

    def step(self, action: str) -> tuple[str, float, bool, dict[str, int]]:
        normalized = action.strip().lower()
        reward = 0.0
        done = False
        if self._snapshot_id == "west":
            if normalized == "open mailbox":
                self._apply_snapshot("mailbox")
                reward = 1.0
            elif normalized == "examine house":
                self._apply_snapshot("house")
        elif self._snapshot_id == "mailbox":
            if normalized == "read leaflet":
                self._apply_snapshot("leaflet")
                reward = 1.0
                done = True
        return self.observation, reward, done, {"score": self.score, "moves": self.moves}

    def get_score(self) -> int:
        return self.score

    def get_moves(self) -> int:
        return self.moves

    def get_inventory(self) -> list[_FakeInventoryItem]:
        return list(self.inventory)

    def get_valid_actions(self) -> list[str]:
        return list(self.valid_actions)

    def get_world_state_hash(self) -> str:
        return f"hash-{self._snapshot_id}"

    def get_state(self) -> tuple[object, ...]:
        return ("native", self._snapshot_id)

    def set_state(self, state: tuple[object, ...]) -> None:
        self.set_state_calls += 1
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


class OscillatingBackend(BranchingBackend):
    """Fake backend that exposes an explicit open/close oscillation path."""

    def __init__(self, story_file: str, seed: int | None = None):
        super().__init__(story_file, seed=seed)
        self._state_table["mailbox"]["valid_actions"] = ["close mailbox", "read leaflet", "look"]

    def step(self, action: str) -> tuple[str, float, bool, dict[str, int]]:
        normalized = action.strip().lower()
        if self._snapshot_id == "mailbox" and normalized == "close mailbox":
            self._apply_snapshot("west")
            return self.observation, 0.0, False, {"score": self.score, "moves": self.moves}
        return super().step(action)


class InventoryGainBackend(BranchingBackend):
    """Fake backend where taking an object changes inventory without changing score."""

    def __init__(self, story_file: str, seed: int | None = None):
        super().__init__(story_file, seed=seed)
        self._state_table = {
            "west": {
                "observation": "West of House. A brass lamp is on the ground.",
                "inventory": [],
                "valid_actions": ["take lamp", "look"],
                "score": 0,
                "moves": 0,
            },
            "lamp": {
                "observation": "You are now carrying the brass lamp.",
                "inventory": [_FakeInventoryItem("lamp")],
                "valid_actions": ["drop lamp", "look"],
                "score": 0,
                "moves": 1,
            },
        }
        self._apply_snapshot("west")

    def step(self, action: str) -> tuple[str, float, bool, dict[str, int]]:
        normalized = action.strip().lower()
        if self._snapshot_id == "west" and normalized == "take lamp":
            self._apply_snapshot("lamp")
            return self.observation, 0.0, False, {"score": self.score, "moves": self.moves}
        return super().step(action)


class WanderingForestBackend(BranchingBackend):
    """Fake backend where movement only changes low-value forest text."""

    def __init__(self, story_file: str, seed: int | None = None):
        super().__init__(story_file, seed=seed)
        self._state_table = {
            "forest-a": {
                "observation": "Forest path among trees.",
                "inventory": [],
                "valid_actions": ["go around trees", "west", "look"],
                "score": 0,
                "moves": 0,
            },
            "forest-b": {
                "observation": "A forest path winds among trees.",
                "inventory": [],
                "valid_actions": ["go around trees", "west", "look"],
                "score": 0,
                "moves": 1,
            },
            "forest-c": {
                "observation": "You are in a dim forest clearing.",
                "inventory": [],
                "valid_actions": ["go around trees", "west", "look"],
                "score": 0,
                "moves": 2,
            },
        }
        self._apply_snapshot("forest-a")

    def step(self, action: str) -> tuple[str, float, bool, dict[str, int]]:
        normalized = action.strip().lower()
        if self._snapshot_id == "forest-a" and normalized == "go around trees":
            self._apply_snapshot("forest-b")
        elif self._snapshot_id == "forest-b" and normalized == "go around trees":
            self._apply_snapshot("forest-a")
        elif normalized == "west":
            self._apply_snapshot("forest-c")
        return self.observation, 0.0, False, {"score": self.score, "moves": self.moves}

    def reset(self) -> tuple[str, dict[str, int]]:
        self._apply_snapshot("forest-a")
        return self.observation, {"score": self.score, "moves": self.moves}


def _build_config(tmp_path: Path) -> ProjectConfig:
    """Create a minimal config for local-explorer tests."""

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
            use_live_jericho=True,
            docker_notes=["Local explorer tests use a fake backend."],
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
        policy=PolicyConfig(rollout_count=2, rollout_depth=2, action_candidates=3),
        paths=PathsConfig(
            artifacts_dir=artifacts_dir,
            trajectory_dir=artifacts_dir / "trajectories",
            archive_dir=artifacts_dir / "archive",
            log_dir=artifacts_dir / "logs",
            summary_dir=artifacts_dir / "summaries",
        ),
        logging=LoggingConfig(),
        experiment=ExperimentConfig(game_id="zork1"),
    )


def test_local_explorer_restores_the_same_base_state_for_each_branch(tmp_path: Path) -> None:
    """Each branch should start from the same restored node before rolling out."""

    config = _build_config(tmp_path)
    env = JerichoEnv(config, env_factory=BranchingBackend)
    prompt_manager = PromptManager(config.prompts)
    action_generator = ActionGenerator(config, prompt_manager, llm_client=None)
    explorer = LocalExplorer(action_generator, env=env)

    env.reset(seed=3)
    base_state = env.step("open mailbox").to_state()

    result = explorer.explore_from_state(
        base_state,
        branch_count=2,
        branch_horizon=1,
        temperature=0.05,
        action_candidate_count=2,
    )

    assert result.branch_count == 2
    assert result.branch_horizon == 1
    assert result.temperature == 0.05
    assert result.action_candidate_count == 2
    assert result.best_branch_index == 0
    assert result.best_branch is not None
    assert result.best_branch.actions_taken == ["read leaflet"]
    assert result.best_branch.branch_progress_score > config.policy.branch_commit_min_progress_score
    assert result.branch_commit_allowed is True
    assert env._env is not None
    assert env._env.set_state_calls == 2

    first_branch, second_branch = result.branches
    assert first_branch.score_change == 1
    assert first_branch.total_reward == 1.0
    assert first_branch.terminated is True
    assert first_branch.termination_reason is BranchTerminationReason.TERMINATED
    assert first_branch.new_room_or_object_detected is True

    assert second_branch.actions_taken == ["look"]
    assert second_branch.score_change == 0
    assert second_branch.appears_stuck is True
    assert second_branch.termination_reason is BranchTerminationReason.HORIZON_REACHED


def test_local_explorer_reduces_to_one_branch_without_a_replayable_snapshot(tmp_path: Path) -> None:
    """Without a snapshot, local exploration should avoid pretending it can restore many branches."""

    config = _build_config(tmp_path)
    env = JerichoEnv(config, env_factory=BranchingBackend)
    prompt_manager = PromptManager(config.prompts)
    action_generator = ActionGenerator(config, prompt_manager, llm_client=None)
    explorer = LocalExplorer(action_generator, env=env)

    base_state = env.reset(seed=5)
    snapshotless_state = TextGameState(
        observation=base_state.observation,
        inventory_text=base_state.inventory_text,
        valid_actions=list(base_state.valid_actions),
        score=base_state.score,
        moves=base_state.moves,
        done=base_state.done,
        world_state_hash=base_state.world_state_hash,
    )

    result = explorer.explore_from_state(snapshotless_state, branch_count=3, branch_horizon=1)

    assert result.branch_count == 1
    assert "reduced to one branch" in result.comparison_notes
    assert len(result.branches) == 1


def test_local_explorer_penalizes_open_close_oscillation_when_scoring_branches(tmp_path: Path) -> None:
    """Oscillating open/close loops should lose to productive branches."""

    config = _build_config(tmp_path)
    env = JerichoEnv(config, env_factory=OscillatingBackend)
    prompt_manager = PromptManager(config.prompts)
    action_generator = ActionGenerator(config, prompt_manager, llm_client=None)
    explorer = LocalExplorer(action_generator, env=env)

    def scripted_proposals(
        state: TextGameState,
        *,
        recent_trajectory_context: str | None = None,
        candidate_count: int | None = None,
        temperature: float | None = None,
        recent_actions: list[str] | None = None,
        recent_loop_results=None,
        state_action_history=None,
    ) -> list[ActionProposal]:
        if "mailbox" in state.observation.lower():
            return [
                ActionProposal(action="close mailbox", rank=1),
                ActionProposal(action="read leaflet", rank=2),
            ]
        return [
            ActionProposal(action="open mailbox", rank=1),
            ActionProposal(action="look", rank=2),
        ]

    action_generator.propose_actions = scripted_proposals  # type: ignore[method-assign]

    env.reset(seed=7)
    base_state = env.step("open mailbox").to_state()
    result = explorer.explore_from_state(base_state, branch_count=2, branch_horizon=2)

    assert result.best_branch_index == 1
    assert result.best_branch is not None
    assert result.best_branch.actions_taken == ["read leaflet"]

    oscillating_branch = result.branches[0]
    productive_branch = result.branches[1]

    assert oscillating_branch.actions_taken == ["close mailbox", "open mailbox"]
    assert oscillating_branch.oscillation_penalty_total > 0.0
    assert oscillating_branch.loop_event_count >= 1
    assert oscillating_branch.branch_progress_score <= config.policy.branch_commit_min_progress_score
    assert result.branch_commit_allowed is True
    assert productive_branch.oscillation_penalty_total == 0.0
    assert productive_branch.score_change > oscillating_branch.score_change


def test_local_explorer_allows_inventory_gain_branch_with_zero_score(tmp_path: Path) -> None:
    """Inventory gain should count as meaningful branch progress even without score gain."""

    config = _build_config(tmp_path)
    env = JerichoEnv(config, env_factory=InventoryGainBackend)
    prompt_manager = PromptManager(config.prompts)
    action_generator = ActionGenerator(config, prompt_manager, llm_client=None)
    explorer = LocalExplorer(action_generator, env=env)

    base_state = env.reset(seed=11)
    result = explorer.explore_from_state(base_state, branch_count=1, branch_horizon=1)

    assert result.best_branch is not None
    assert result.best_branch.actions_taken == ["take lamp"]
    assert result.best_branch.score_change == 0
    assert result.best_branch.inventory_changed is True
    assert result.best_branch.persistent_inventory_gain_count == 1
    assert result.best_branch.branch_progress_score > config.policy.branch_commit_min_progress_score
    assert result.branch_commit_allowed is True


def test_local_explorer_rejects_zero_score_movement_only_branch(tmp_path: Path) -> None:
    """Movement-only room-text variation should not clear the branch commit gate."""

    config = _build_config(tmp_path)
    env = JerichoEnv(config, env_factory=WanderingForestBackend)
    prompt_manager = PromptManager(config.prompts)
    action_generator = ActionGenerator(config, prompt_manager, llm_client=None)
    explorer = LocalExplorer(action_generator, env=env)

    def scripted_proposals(
        state: TextGameState,
        *,
        recent_trajectory_context: str | None = None,
        candidate_count: int | None = None,
        temperature: float | None = None,
        recent_actions: list[str] | None = None,
        recent_loop_results=None,
        state_action_history=None,
    ) -> list[ActionProposal]:
        return [
            ActionProposal(action="go around trees", rank=1),
            ActionProposal(action="west", rank=2),
        ]

    action_generator.propose_actions = scripted_proposals  # type: ignore[method-assign]

    base_state = env.reset(seed=17)
    result = explorer.explore_from_state(base_state, branch_count=1, branch_horizon=2)

    assert result.best_branch is not None
    assert result.best_branch.actions_taken == ["go around trees", "go around trees"]
    assert result.best_branch.score_change == 0
    assert result.best_branch.affordance_gain == 0
    assert result.best_branch.movement_penalty_total > 0.0
    assert result.best_branch.movement_only_action_count == 2
    assert result.best_branch.movement_action_ratio == 1.0
    assert result.branch_commit_allowed is False
    assert (
        "did not clear threshold" in result.commit_rejection_reason
        or "movement-dominated" in result.commit_rejection_reason
        or "no score gain, inventory gain, exit gain, or sufficient affordance gain"
        in result.commit_rejection_reason
    )


def test_local_explorer_aborts_low_value_movement_cycle_early(tmp_path: Path) -> None:
    """Movement-only cycles between one or two clusters should fail fast inside a branch."""

    config = _build_config(tmp_path)
    config.policy.branch_fail_fast_min_movement_actions = 3
    env = JerichoEnv(config, env_factory=WanderingForestBackend)
    prompt_manager = PromptManager(config.prompts)
    action_generator = ActionGenerator(config, prompt_manager, llm_client=None)
    explorer = LocalExplorer(action_generator, env=env)

    def scripted_proposals(
        state: TextGameState,
        *,
        recent_trajectory_context: str | None = None,
        candidate_count: int | None = None,
        temperature: float | None = None,
        recent_actions: list[str] | None = None,
        recent_loop_results=None,
        state_action_history=None,
    ) -> list[ActionProposal]:
        return [
            ActionProposal(action="go around trees", rank=1),
            ActionProposal(action="west", rank=2),
        ]

    action_generator.propose_actions = scripted_proposals  # type: ignore[method-assign]

    base_state = env.reset(seed=23)
    result = explorer.explore_from_state(base_state, branch_count=1, branch_horizon=8)

    assert result.best_branch is not None
    assert result.best_branch.termination_reason is BranchTerminationReason.LOOP_ABORTED
    assert len(result.best_branch.actions_taken) < 8
    assert result.best_branch.movement_action_ratio == 1.0
    assert result.branch_commit_allowed is False
