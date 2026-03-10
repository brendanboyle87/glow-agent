"""Tests for episode execution and batch evaluation.

TODO: add live-container integration coverage if Jericho-backed CI is introduced later.
"""

from __future__ import annotations

import json
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
from zork_agent.experiment.episode_runner import EpisodeRunner
from zork_agent.experiment.evaluator import Evaluator
from zork_agent.memory.trajectory_store import TrajectoryStore
from zork_agent.types import (
    ActionProposal,
    BranchTerminationReason,
    LocalBranchOutcome,
    LocalExplorationResult,
    TextGameState,
    TextGameTransition,
)


def _build_config(tmp_path: Path) -> ProjectConfig:
    """Create a stub-friendly config for episode/evaluator tests."""

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
            use_live_jericho=False,
            docker_notes=["Episode runner tests use the deterministic stub env."],
        ),
        llm=LLMConfig(offline_stub=True),
        prompts=PromptConfig(
            directory=prompt_dir,
            system_file="system.txt",
            action_proposal_file="action.txt",
            trajectory_analysis_file="trajectory.txt",
            local_reflection_file="reflection.txt",
            state_selection_file="selection.txt",
        ),
        policy=PolicyConfig(
            rollout_count=2,
            rollout_depth=2,
            frontier_max_size=8,
            snapshot_retention_limit=2,
            action_candidates=3,
            state_selection_mode="heuristic",
        ),
        paths=PathsConfig(
            artifacts_dir=artifacts_dir,
            trajectory_dir=artifacts_dir / "trajectories",
            archive_dir=artifacts_dir / "archive",
            log_dir=artifacts_dir / "logs",
            summary_dir=artifacts_dir / "summaries",
        ),
        logging=LoggingConfig(),
        experiment=ExperimentConfig(
            game_id="zork1",
            seed=13,
            batch_size=2,
            max_steps=4,
            max_replay_attempts=1,
            frontier_refresh_cadence=1,
            local_exploration_cadence=2,
            stub_episode_length=10,
        ),
    )


def test_episode_runner_writes_trajectory_and_summary_artifacts(tmp_path: Path) -> None:
    """A stub-mode episode should execute the full baseline loop and persist artifacts."""

    config = _build_config(tmp_path)
    config.ensure_output_directories()
    runner = EpisodeRunner(config=config, llm_client=None)

    result = runner.run_episode(episode_id="episode-runner-smoke")

    assert result.step_count == 4
    assert result.trajectory_path.exists()
    assert result.summary_path is not None
    assert result.summary_path.exists()
    assert result.metadata["replay_attempt_count"] == 1
    assert result.metadata["local_exploration_count"] == 1
    assert result.metadata["frontier_size"] >= 1

    stored_trajectory = TrajectoryStore(config.paths.trajectory_dir).read_episode("episode-runner-smoke")
    assert len(stored_trajectory.steps) == result.step_count
    assert [step.step_index for step in stored_trajectory.steps] == [0, 1, 2, 3]
    assert stored_trajectory.steps[-1].metadata["action_source"] in {
        "action_generator",
        "best_local_branch",
        "best_local_branch_plan",
    }

    summary_payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary_payload["episode_id"] == "episode-runner-smoke"
    assert summary_payload["step_count"] == 4
    assert summary_payload["metadata"]["final_guidance"]


def test_evaluator_reports_mean_and_std_and_persists_batch_summary(tmp_path: Path) -> None:
    """Batch evaluation should run multiple seeds and emit mean/std statistics."""

    config = _build_config(tmp_path)
    config.ensure_output_directories()
    evaluator = Evaluator(config=config, llm_client=None)

    results = evaluator.run_batch(episode_count=2)
    summary = evaluator.summarize_results(results)

    assert len(results) == 2
    assert [result.seed for result in results] == [13, 14]
    assert summary.episode_count == 2
    assert summary.seed_values == [13, 14]
    assert summary.mean_steps == 4.0
    assert summary.std_steps == 0.0
    assert summary.summary_path is not None
    assert summary.summary_path.exists()

    summary_payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    assert summary_payload["episode_count"] == 2
    assert summary_payload["seed_values"] == [13, 14]
    assert len(summary_payload["trajectory_paths"]) == 2


def test_episode_runner_commits_multiple_actions_from_best_local_branch(tmp_path: Path) -> None:
    """A winning local branch should contribute a short action prefix to the real episode."""

    config = _build_config(tmp_path)
    config.ensure_output_directories()
    config.experiment.max_steps = 4
    config.experiment.local_exploration_cadence = 2
    config.experiment.max_replay_attempts = 1
    config.experiment.branch_commit_steps = 2
    runner = EpisodeRunner(config=config, llm_client=None)

    def fake_propose_actions(*args, **kwargs) -> list[ActionProposal]:
        return [ActionProposal(action="look", source="stub")]

    def fake_explore_from_state(*args, **kwargs) -> LocalExplorationResult:
        base_state = args[0]
        return LocalExplorationResult(
            base_state=base_state,
            branch_count=1,
            branch_horizon=2,
            temperature=0.2,
            action_candidate_count=3,
            branches=[
                LocalBranchOutcome(
                    branch_index=0,
                    actions_taken=["open mailbox", "read leaflet"],
                    total_reward=1.0,
                    score_change=1,
                    final_score=1,
                    final_observation="Welcome to the scaffold. The leaflet mostly confirms this is a placeholder.",
                    terminated=False,
                    termination_reason=BranchTerminationReason.HORIZON_REACHED,
                    new_room_or_object_detected=True,
                    appears_stuck=False,
                    branch_progress_score=2.0,
                    branch_commit_allowed=True,
                )
            ],
            best_branch_index=0,
            branch_commit_allowed=True,
            comparison_notes="forced best branch for regression coverage",
        )

    runner.action_generator.propose_actions = fake_propose_actions  # type: ignore[method-assign]
    runner.local_explorer.explore_from_state = fake_explore_from_state  # type: ignore[method-assign]

    result = runner.run_episode(episode_id="episode-runner-branch-commit")
    stored_trajectory = TrajectoryStore(config.paths.trajectory_dir).read_episode("episode-runner-branch-commit")

    assert result.step_count == 4
    assert [step.action for step in stored_trajectory.steps] == [
        "look",
        "look",
        "open mailbox",
        "read leaflet",
    ]
    assert [step.metadata["action_source"] for step in stored_trajectory.steps] == [
        "action_generator",
        "action_generator",
        "best_local_branch",
        "best_local_branch_plan",
    ]


def test_episode_runner_extends_branch_commit_through_first_durable_gain(tmp_path: Path) -> None:
    """A winning branch should commit through the first durable gain even past the nominal prefix size."""

    config = _build_config(tmp_path)
    config.ensure_output_directories()
    config.experiment.max_steps = 7
    config.experiment.local_exploration_cadence = 2
    config.experiment.max_replay_attempts = 1
    config.experiment.branch_commit_steps = 2
    runner = EpisodeRunner(config=config, llm_client=None)

    def fake_propose_actions(*args, **kwargs) -> list[ActionProposal]:
        return [ActionProposal(action="look", source="stub")]

    def fake_explore_from_state(*args, **kwargs) -> LocalExplorationResult:
        base_state = args[0]
        return LocalExplorationResult(
            base_state=base_state,
            branch_count=1,
            branch_horizon=5,
            temperature=0.2,
            action_candidate_count=3,
            branches=[
                LocalBranchOutcome(
                    branch_index=0,
                    actions_taken=["north", "up", "open egg", "shake egg", "take canary"],
                    total_reward=0.0,
                    score_change=0,
                    final_score=0,
                    final_observation="A jeweled canary is now in your inventory.",
                    terminated=False,
                    termination_reason=BranchTerminationReason.HORIZON_REACHED,
                    new_room_or_object_detected=True,
                    appears_stuck=False,
                    persistent_inventory_gain_count=1,
                    branch_progress_score=3.0,
                    branch_commit_allowed=True,
                    first_durable_gain_action_index=4,
                    first_durable_gain_action="take canary",
                )
            ],
            best_branch_index=0,
            branch_commit_allowed=True,
            comparison_notes="forced durable gain late in the branch",
        )

    runner.action_generator.propose_actions = fake_propose_actions  # type: ignore[method-assign]
    runner.local_explorer.explore_from_state = fake_explore_from_state  # type: ignore[method-assign]

    result = runner.run_episode(episode_id="episode-runner-branch-commit-gain")
    stored_trajectory = TrajectoryStore(config.paths.trajectory_dir).read_episode("episode-runner-branch-commit-gain")

    assert result.step_count == 7
    assert [step.action for step in stored_trajectory.steps] == [
        "look",
        "look",
        "north",
        "up",
        "open egg",
        "shake egg",
        "take canary",
    ]
    assert stored_trajectory.steps[-1].metadata["action_source"] == "best_local_branch_plan"


def test_episode_runner_rejects_zero_value_local_branch_commit(tmp_path: Path) -> None:
    """Local branches below the progress threshold should never be promoted into the main episode."""

    config = _build_config(tmp_path)
    config.ensure_output_directories()
    config.experiment.max_steps = 4
    config.experiment.local_exploration_cadence = 2
    config.experiment.max_replay_attempts = 1
    runner = EpisodeRunner(config=config, llm_client=None)

    def fake_propose_actions(*args, **kwargs) -> list[ActionProposal]:
        return [ActionProposal(action="look", source="stub")]

    def fake_explore_from_state(*args, **kwargs) -> LocalExplorationResult:
        base_state = args[0]
        return LocalExplorationResult(
            base_state=base_state,
            branch_count=1,
            branch_horizon=2,
            temperature=0.2,
            action_candidate_count=3,
            branches=[
                LocalBranchOutcome(
                    branch_index=0,
                    actions_taken=["open mailbox", "close mailbox"],
                    total_reward=0.0,
                    score_change=0,
                    final_score=0,
                    final_observation="You are standing in an open field west of a white house, near a small mailbox.",
                    terminated=False,
                    termination_reason=BranchTerminationReason.HORIZON_REACHED,
                    new_room_or_object_detected=False,
                    appears_stuck=True,
                    oscillation_penalty_total=2.75,
                    loop_event_count=1,
                    branch_progress_score=0.0,
                    branch_commit_allowed=False,
                    commit_rejection_reason="best branch progress_score 0.00 did not clear threshold 1.00",
                )
            ],
            best_branch_index=0,
            branch_commit_allowed=False,
            commit_rejection_reason="best branch progress_score 0.00 did not clear threshold 1.00",
            comparison_notes="forced zero-progress branch for rejection coverage",
        )

    runner.action_generator.propose_actions = fake_propose_actions  # type: ignore[method-assign]
    runner.local_explorer.explore_from_state = fake_explore_from_state  # type: ignore[method-assign]

    result = runner.run_episode(episode_id="episode-runner-branch-reject")
    stored_trajectory = TrajectoryStore(config.paths.trajectory_dir).read_episode("episode-runner-branch-reject")

    assert result.step_count == 4
    assert [step.action for step in stored_trajectory.steps] == ["look", "look", "look", "look"]
    assert {step.metadata["action_source"] for step in stored_trajectory.steps} == {"action_generator"}
    assert result.metadata["branch_commit_rejection_count"] == 1
    assert result.metadata["branch_commit_allowed"] is False
    assert "did not clear threshold" in result.metadata["commit_rejection_reason"]


def test_episode_runner_skips_local_exploration_during_post_gain_cooldown(tmp_path: Path) -> None:
    """Score gains should start a cooldown that suppresses immediate revisit/local-exploration probes."""

    class CooldownEnv:
        """Tiny env that scores once, then keeps offering plain movement."""

        def __init__(self) -> None:
            self._step = 0

        def reset(self, seed: int | None = None) -> TextGameState:
            self._step = 0
            return TextGameState(
                observation="Mailbox scene.",
                inventory_text="Inventory empty.",
                valid_actions=["take treasure", "north"],
                score=0,
                moves=0,
                done=False,
                world_state_hash="mailbox-start",
            )

        def step(self, action: str) -> TextGameTransition:
            self._step += 1
            if action == "take treasure":
                return TextGameTransition(
                    action=action,
                    step_index=self._step - 1,
                    observation="You are carrying a treasure.",
                    reward=1.0,
                    done=False,
                    score=1,
                    moves=self._step,
                    inventory_text="You are carrying a treasure.",
                    valid_actions=["north"],
                    world_state_hash="treasure-state",
                )
            return TextGameTransition(
                action=action,
                step_index=self._step - 1,
                observation="A plain hallway.",
                reward=0.0,
                done=False,
                score=1,
                moves=self._step,
                inventory_text="You are carrying a treasure.",
                valid_actions=["north"],
                world_state_hash=f"hall-{self._step}",
            )

        def close(self) -> None:
            return None

    config = _build_config(tmp_path)
    config.ensure_output_directories()
    config.experiment.max_steps = 4
    config.experiment.local_exploration_cadence = 2
    config.experiment.local_exploration_post_gain_cooldown_steps = 3
    config.experiment.max_replay_attempts = 3
    runner = EpisodeRunner(config=config, llm_client=None)
    runner.env = CooldownEnv()  # type: ignore[assignment]
    runner.local_explorer.env = runner.env  # type: ignore[assignment]

    def fake_propose_actions(state: TextGameState, **kwargs) -> list[ActionProposal]:
        if "treasure" in state.inventory_text.lower():
            return [ActionProposal(action="north", source="stub")]
        return [ActionProposal(action="take treasure", source="stub")]

    explore_call_count = 0

    def fake_explore_from_state(*args, **kwargs) -> LocalExplorationResult:
        nonlocal explore_call_count
        explore_call_count += 1
        base_state = args[0]
        return LocalExplorationResult(
            base_state=base_state,
            branch_count=1,
            branch_horizon=1,
            temperature=0.2,
            action_candidate_count=1,
            branches=[],
            best_branch_index=None,
            branch_commit_allowed=False,
            commit_rejection_reason="no branches",
            comparison_notes="should not be called during cooldown",
        )

    runner.action_generator.propose_actions = fake_propose_actions  # type: ignore[method-assign]
    runner.local_explorer.explore_from_state = fake_explore_from_state  # type: ignore[method-assign]

    result = runner.run_episode(episode_id="episode-runner-cooldown")

    assert result.final_score == 1
    assert explore_call_count == 0
    assert result.metadata["local_exploration_count"] == 0


def test_episode_runner_fails_fast_on_stalled_movement_basin(tmp_path: Path) -> None:
    """Movement-only no-progress episodes should stop early instead of burning the whole budget."""

    class MovementBasinEnv:
        """Small fake env that only wanders among forest descriptions."""

        def __init__(self) -> None:
            self._step = 0

        def reset(self, seed: int | None = None) -> TextGameState:
            self._step = 0
            return TextGameState(
                observation="Forest path among trees.",
                inventory_text="Inventory empty.",
                valid_actions=["go around trees", "west"],
                score=0,
                moves=0,
                done=False,
                world_state_hash="forest-0",
            )

        def step(self, action: str) -> TextGameTransition:
            self._step += 1
            observation = (
                "A forest path winds among trees."
                if self._step % 2
                else "Forest path among trees."
            )
            return TextGameTransition(
                action=action,
                step_index=self._step - 1,
                observation=observation,
                reward=0.0,
                done=False,
                score=0,
                moves=self._step,
                inventory_text="Inventory empty.",
                valid_actions=["go around trees", "west"],
                world_state_hash=f"forest-{self._step % 2}",
            )

        def close(self) -> None:
            return None

    config = _build_config(tmp_path)
    config.ensure_output_directories()
    config.experiment.max_steps = 10
    config.experiment.max_replay_attempts = 0
    config.experiment.fail_fast_no_durable_gain_steps = 4
    config.experiment.fail_fast_same_cluster_movement_steps = 3
    runner = EpisodeRunner(config=config, llm_client=None)
    runner.env = MovementBasinEnv()  # type: ignore[assignment]
    runner.local_explorer.env = runner.env  # type: ignore[assignment]

    def fake_propose_actions(*args, **kwargs) -> list[ActionProposal]:
        return [ActionProposal(action="go around trees", source="stub")]

    runner.action_generator.propose_actions = fake_propose_actions  # type: ignore[method-assign]

    result = runner.run_episode(episode_id="episode-runner-fail-fast")

    assert result.step_count < config.experiment.max_steps
    assert "movement wandering persisted" in result.metadata["episode_fail_fast_reason"]
    assert result.metadata["consecutive_same_cluster_movement_steps"] >= 3
