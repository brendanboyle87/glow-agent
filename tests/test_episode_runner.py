"""Tests for GLoW episode execution and batch evaluation.

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
from zork_agent.llm.base import BaseLLMClient
from zork_agent.memory.archive_store import ArchiveStore
from zork_agent.memory.frontier import FrontierEntry
from zork_agent.memory.trajectory_store import TrajectoryStore
from zork_agent.types import (
    ActionProposal,
    ArchiveStateSelectionResult,
    ArchivedState,
    BranchTerminationReason,
    FrontierAnalysisResult,
    FrontierInsight,
    LocalBranchOutcome,
    LocalExplorationResult,
    LocalWorldModel,
    MARInferenceResult,
    ReplayDivergenceReason,
    ReplayMetadata,
    ReplayResult,
    RestoreMode,
    RunnerMode,
    StateSelectionResult,
    StrategicGuidance,
    StrategicMode,
    TextGameState,
    TextGameTransition,
    TrajectoryStep,
    WorldStateSnapshot,
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
        "frontier_analysis.txt",
        "mar_advantage.txt",
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
            frontier_analysis_file="frontier_analysis.txt",
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
            runner_mode=RunnerMode.LEGACY,
        ),
    )


class _FakeLLMClient(BaseLLMClient):
    """Minimal fake LLM client for runner integration tests."""

    def __init__(self):
        super().__init__(default_model="fake-model")

    def list_models(self) -> list[str]:
        return ["fake-model"]

    def complete_chat(self, request):
        raise AssertionError("This fake LLM client should not be called in the mocked runner test.")


class _FakeGlowEnv:
    """Minimal env stub for the GLoW runner integration path."""

    def __init__(self):
        self.closed = False

    def reset(self, seed: int | None = None) -> TextGameState:
        return TextGameState(
            observation="West of House.",
            inventory_text="You are empty-handed.",
            valid_actions=["open mailbox", "north"],
            score=0,
            moves=0,
            done=False,
            world_state_hash="root-hash",
            world_state_snapshot=WorldStateSnapshot(
                step_index=0,
                observation="West of House.",
                done=False,
                score=0,
                moves=0,
                inventory_text="You are empty-handed.",
                valid_actions=["open mailbox", "north"],
                world_state_hash="root-hash",
                restore_strategy="native_snapshot",
                replay_actions=[],
                native_state=("root",),
            ),
        )

    def close(self) -> None:
        self.closed = True


def test_episode_runner_writes_trajectory_and_summary_artifacts(tmp_path: Path) -> None:
    """A stub-mode episode should execute the full loop and persist artifacts."""

    config = _build_config(tmp_path)
    config.ensure_output_directories()
    runner = EpisodeRunner(config=config, llm_client=None)

    result = runner.run_episode(episode_id="episode-runner-smoke")

    assert result.step_count == 4
    assert result.trajectory_path.exists()
    assert result.summary_path is not None
    assert result.summary_path.exists()
    assert result.metrics_path is None
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
    """Batch evaluation should emit run-level GLoW metrics and persist them."""

    config = _build_config(tmp_path)
    config.ensure_output_directories()
    config.experiment.runner_mode = RunnerMode.GLOW_FAITHFUL
    evaluator = Evaluator(config=config, llm_client=None)

    results = evaluator.run_batch(episode_count=2)
    summary = evaluator.summarize_results(results)

    assert len(results) == 2
    assert [result.seed for result in results] == [13, 14]
    assert summary.episode_count == 2
    assert summary.seed_values == [13, 14]
    assert summary.summary_path is not None
    assert Path(summary.summary_path).exists()
    assert len(summary.episode_metric_paths) == 2
    assert len(summary.episode_summary_paths) == 2
    assert len(summary.trajectory_paths) == 2

    summary_payload = json.loads(Path(summary.summary_path).read_text(encoding="utf-8"))
    assert summary_payload["episode_count"] == 2
    assert summary_payload["seed_values"] == [13, 14]
    assert len(summary_payload["trajectory_paths"]) == 2
    assert len(summary_payload["episode_metric_paths"]) == 2
    assert "mean_environment_interactions" in summary_payload
    assert "total_frontier_analysis_count" in summary_payload
    assert "total_mar_update_count" in summary_payload


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


def test_episode_runner_commits_through_last_meaningful_progress_and_trims_churn(tmp_path: Path) -> None:
    """A winning branch should commit through the last real gain and stop before post-gain churn."""

    config = _build_config(tmp_path)
    config.ensure_output_directories()
    config.experiment.max_steps = 8
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
            branch_horizon=8,
            temperature=0.2,
            action_candidate_count=3,
            branches=[
                LocalBranchOutcome(
                    branch_index=0,
                    actions_taken=[
                        "north",
                        "up",
                        "open egg",
                        "take egg",
                        "take on egg",
                        "take canary",
                        "close egg",
                        "open egg",
                    ],
                    total_reward=0.0,
                    score_change=5,
                    final_score=5,
                    final_observation="A jeweled canary is now in your inventory.",
                    terminated=False,
                    termination_reason=BranchTerminationReason.HORIZON_REACHED,
                    new_room_or_object_detected=True,
                    appears_stuck=False,
                    persistent_inventory_gain_count=3,
                    branch_progress_score=7.0,
                    branch_commit_allowed=True,
                    first_durable_gain_action_index=3,
                    first_durable_gain_action="take egg",
                    last_durable_progress_action_index=3,
                    last_durable_progress_action="take egg",
                    last_meaningful_progress_action_index=5,
                    last_meaningful_progress_action="take canary",
                )
            ],
            best_branch_index=0,
            branch_commit_allowed=True,
            comparison_notes="forced durable gain chain followed by churn",
        )

    runner.action_generator.propose_actions = fake_propose_actions  # type: ignore[method-assign]
    runner.local_explorer.explore_from_state = fake_explore_from_state  # type: ignore[method-assign]

    result = runner.run_episode(episode_id="episode-runner-branch-commit-gain")
    stored_trajectory = TrajectoryStore(config.paths.trajectory_dir).read_episode("episode-runner-branch-commit-gain")

    assert result.step_count == 8
    assert [step.action for step in stored_trajectory.steps] == [
        "look",
        "look",
        "north",
        "up",
        "open egg",
        "take egg",
        "look",
        "look",
    ]
    assert stored_trajectory.steps[5].metadata["action_source"] == "best_local_branch_plan"


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


def test_episode_runner_allows_off_cadence_revisit_for_preferred_cluster(tmp_path: Path) -> None:
    """Strategic exploit guidance for another cluster should bypass the normal cadence gate."""

    runner = EpisodeRunner(config=_build_config(tmp_path), llm_client=None)
    runner.frontier.add(
        FrontierEntry(
            state_id="house-node",
            score=1.0,
            depth=1,
            state_cluster_id="house",
            world_state_hash="house",
            summary_text="Window branch",
        )
    )

    should_run = runner._should_run_local_exploration(  # type: ignore[attr-defined]
        1,
        0,
        0,
        strategic_guidance=StrategicGuidance(
            mode=StrategicMode.EXPLOIT,
            reason="Known house window route is stronger.",
            preferred_cluster_id="house",
        ),
        current_cluster_id="tree",
    )

    assert should_run is True


def test_episode_runner_prefers_frontier_entry_from_strategic_cluster(tmp_path: Path) -> None:
    """Strategic guidance should override selection toward the preferred replay cluster."""

    runner = EpisodeRunner(config=_build_config(tmp_path), llm_client=None)
    runner.frontier.add(
        FrontierEntry(
            state_id="tree-node",
            score=2.0,
            depth=1,
            state_cluster_id="tree",
            world_state_hash="tree",
            summary_text="Tree branch",
        )
    )
    runner.frontier.add(
        FrontierEntry(
            state_id="house-node",
            score=1.0,
            depth=1,
            state_cluster_id="house",
            world_state_hash="house",
            summary_text="Window branch",
        )
    )
    base_result = StateSelectionResult(
        selected=runner.frontier.top_k(1)[0],
        reason="Heuristic frontier ordering picked the current tree node.",
        selection_mode=runner.state_selector.selection_mode,
    )

    overridden = runner._apply_strategic_selection_override(  # type: ignore[attr-defined]
        base_result,
        strategic_guidance=StrategicGuidance(
            mode=StrategicMode.EXPLOIT,
            reason="Route back to the house window.",
            preferred_cluster_id="house",
        ),
    )

    assert overridden.selected is not None
    assert overridden.selected.state_id == "house-node"
    assert "Strategic override selected" in overridden.reason


def test_episode_runner_treats_off_cluster_replay_branch_as_planning_only(tmp_path: Path) -> None:
    """Replay-derived branches from another cluster should not rewrite the live episode timeline."""

    runner = EpisodeRunner(config=_build_config(tmp_path), llm_client=None)
    origin_state = TextGameState(
        observation="Up a Tree",
        inventory_text="You are carrying a jewel-encrusted egg.",
        valid_actions=["down", "take egg"],
        score=5,
        moves=10,
        world_state_hash="tree-state",
        state_cluster_id="region:title:up-a-tree",
    )
    selected_entry = FrontierEntry(
        state_id="house-node",
        score=0.0,
        depth=1,
        state_cluster_id="region:house-field",
        world_state_hash="house-field",
        inventory_text="Inventory empty.",
        summary_text="House exterior with a promising window.",
    )

    allowed, reason = runner._validate_branch_commit_origin(  # type: ignore[attr-defined]
        origin_state=origin_state,
        selected_entry=selected_entry,
    )

    assert allowed is False
    assert "planning-only off-cluster restore" in reason


def test_episode_runner_switches_to_explore_after_same_cluster_stall(tmp_path: Path) -> None:
    """Repeated no-progress steps in one cluster should temporarily override exploit mode."""

    runner = EpisodeRunner(config=_build_config(tmp_path), llm_client=None)
    stalled_state = TextGameState(
        observation="Behind House. The window is open and a leaflet lies here.",
        inventory_text="You are empty-handed.",
        valid_actions=["take leaflet", "close window", "east", "west"],
        state_cluster_id="region:window",
        world_state_hash="window",
    )

    overridden = runner._apply_stall_recovery_guidance_override(  # type: ignore[attr-defined]
        strategic_guidance=StrategicGuidance(
            mode=StrategicMode.EXPLOIT,
            reason="Local object follow-ups still seem available.",
            preferred_cluster_id="region:window",
            try_actions=["take leaflet", "close window"],
        ),
        current_state=stalled_state,
        consecutive_no_durable_gain_steps=5,
    )

    assert overridden.mode is StrategicMode.EXPLORE
    assert overridden.try_actions == ["east", "west"]
    assert "take leaflet" in overridden.avoid_actions
    assert "stall recovery" in overridden.reason.lower()


def test_glow_runner_executes_global_local_loop_and_persists_best_trajectory(tmp_path: Path) -> None:
    """The GLoW-style runner should wire selection, local rollouts, MAR, and frontier analysis together."""

    config = _build_config(tmp_path)
    config.ensure_output_directories()
    config.experiment.runner_mode = RunnerMode.GLOW_FAITHFUL
    config.experiment.max_steps = 3
    config.experiment.glow_local_branches_per_root = 2
    config.experiment.glow_frontier_analysis_frequency = 1
    runner = EpisodeRunner(config=config, llm_client=_FakeLLMClient())
    runner.env = _FakeGlowEnv()  # type: ignore[assignment]
    runner.local_explorer.env = runner.env

    selection_calls: list[int] = []
    analysis_calls: list[int] = []

    def fake_select_archive_state(*, archive_states, frontier_insight, trajectory_frontier, mode=None, selection_id=None):
        selection_calls.append(len(archive_states))
        return ArchiveStateSelectionResult(
            selected_archived_state=archive_states[0],
            chosen_replay_method="native_snapshot",
            achieved_contribution=0.0,
            potential_contribution=0.0,
            rationale="Initial root selection.",
            selection_mode=config.policy.archive_state_selection_mode,
            artifact_directory=str(tmp_path / "selection-artifact"),
        )

    def fake_frontier_analysis(trajectory_frontier, *, archive_states=None, top_k=None, analysis_id=None):
        analysis_calls.append(len(trajectory_frontier))
        return FrontierAnalysisResult(
            analysis_id=analysis_id or "analysis-001",
            insight=FrontierInsight(
                analysis_id=analysis_id or "analysis-001",
                frontier_trajectory_ids=[entry.trajectory_id for entry in trajectory_frontier.top_k_entries(2)],
                partial_solutions=["Leaflet pickup looks promising."],
                inferred_bottlenecks=["House entry not yet reached."],
            ),
            prompt_input="frontier prompt",
            artifact_directory=str(tmp_path / "frontier-analysis"),
        )

    def fake_explore_from_state(restored_state, **kwargs):
        mar_result = MARInferenceResult(
            root_state_id="root-state-1",
            compared_branch_ids=["b0", "b1"],
            prompt_input="mar prompt",
            artifact_directory=str(tmp_path / "mar-artifact"),
        )
        local_world_model = LocalWorldModel(root_state_id="root-state-1")
        return LocalExplorationResult(
            base_state=restored_state,
            branch_count=2,
            branch_horizon=2,
            temperature=0.2,
            action_candidate_count=3,
            root_state_id="root-state-1",
            best_branch_index=1,
            branch_commit_allowed=True,
            comparison_notes="mocked glow local comparison",
            mar_inference=mar_result,
            updated_local_world_model=local_world_model,
            branches=[
                LocalBranchOutcome(
                    branch_index=0,
                    actions_taken=["open mailbox", "take leaflet"],
                    total_reward=0.0,
                    score_change=0,
                    final_score=0,
                    final_observation="A leaflet is here.",
                    terminated=False,
                    termination_reason=BranchTerminationReason.HORIZON_REACHED,
                    branch_progress_score=1.0,
                    trajectory_steps=[
                        TrajectoryStep(
                            episode_id="branch-0",
                            step_index=0,
                            action="open mailbox",
                            observation="Opening the mailbox reveals a leaflet.",
                            reward=0.0,
                            cumulative_reward=0.0,
                            done=False,
                            score=0,
                            moves=1,
                            world_state_hash="mailbox-open",
                            inventory_text="You are empty-handed.",
                            valid_actions=["take leaflet", "north"],
                        ),
                        TrajectoryStep(
                            episode_id="branch-0",
                            step_index=1,
                            action="take leaflet",
                            observation="Taken.",
                            reward=0.0,
                            cumulative_reward=0.0,
                            done=False,
                            score=0,
                            moves=2,
                            world_state_hash="leaflet-taken",
                            inventory_text="You are carrying a leaflet.",
                            valid_actions=["north"],
                        ),
                    ],
                ),
                LocalBranchOutcome(
                    branch_index=1,
                    actions_taken=["north"],
                    total_reward=1.0,
                    score_change=1,
                    final_score=1,
                    final_observation="North of House.",
                    terminated=False,
                    termination_reason=BranchTerminationReason.HORIZON_REACHED,
                    branch_progress_score=2.0,
                    trajectory_steps=[
                        TrajectoryStep(
                            episode_id="branch-1",
                            step_index=0,
                            action="north",
                            observation="North of House.",
                            reward=1.0,
                            cumulative_reward=1.0,
                            done=False,
                            score=1,
                            moves=1,
                            world_state_hash="north-house",
                            inventory_text="You are empty-handed.",
                            valid_actions=["east", "west"],
                        )
                    ],
                ),
            ],
        )

    runner.state_selector.select_archive_state = fake_select_archive_state  # type: ignore[method-assign]
    runner.frontier_analyzer.analyze_frontier = fake_frontier_analysis  # type: ignore[method-assign]
    runner.local_explorer.explore_from_state = fake_explore_from_state  # type: ignore[method-assign]
    runner._restore_archived_state = lambda archived_state: ReplayResult(  # type: ignore[method-assign]
        success=True,
        restore_mode=RestoreMode.NATIVE_SNAPSHOT,
        divergence_reason=ReplayDivergenceReason.NONE,
        final_observation="West of House.",
        final_score=0,
        replayed_action_count=0,
        target_step_index=archived_state.provenance_timestep,
        final_state=runner.env.reset(),
        restored_node_id=archived_state.state_id,
        message="mocked restore",
    )

    result = runner.run_episode(episode_id="glow-runner-smoke")
    stored_trajectory = TrajectoryStore(config.paths.trajectory_dir).read_episode("glow-runner-smoke")

    assert result.metadata["runner_mode"] == RunnerMode.GLOW_FAITHFUL.value
    assert result.final_score == 1
    assert [step.action for step in stored_trajectory.steps] == ["north"]
    assert result.summary_path is not None and result.summary_path.exists()
    assert result.metrics_path is not None and result.metrics_path.exists()
    assert selection_calls == [1]
    assert analysis_calls == [2]
    assert Path(result.metadata["archive_snapshot_path"]).exists()
    assert result.metadata["selection_artifact_directories"] == [str(tmp_path / "selection-artifact")]
    assert result.metadata["mar_artifact_directories"] == [str(tmp_path / "mar-artifact")]
    assert result.metadata["frontier_analysis_artifact_directories"] == [str(tmp_path / "frontier-analysis")]

    metrics_payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["frontier_analysis_count"] == 1
    assert metrics_payload["mar_update_count"] == 1
    assert metrics_payload["selection_artifact_directories"] == [str(tmp_path / "selection-artifact")]
    assert metrics_payload["metadata"]["unique_selected_root_count"] == 1
    assert metrics_payload["metadata"]["top_selected_roots"] == [
        {
            "state_id": "root-state-1",
            "count": 1,
            "summary": "West of House.",
        }
    ]
    assert metrics_payload["metadata"]["max_score_over_time"] == [1]
    assert metrics_payload["metadata"]["score_milestones"] == [
        {
            "cycle_index": 1,
            "score": 1,
            "trajectory_id": "glow-runner-smoke-cycle-001-branch-01",
            "root_state_id": "root-state-1",
        }
    ]
    assert metrics_payload["metadata"]["commit_allowed_count"] == 1
    assert metrics_payload["metadata"]["committed_branch_positive_progress_count"] == 1
    assert metrics_payload["metadata"]["committed_branch_score_gain_count"] == 1
    assert metrics_payload["metadata"]["top_committed_actions"] == [{"action": "north", "count": 1}]
    assert metrics_payload["metadata"]["frontier_analysis_success_count"] == 1
    assert metrics_payload["metadata"]["frontier_analysis_fallback_count"] == 0
    assert metrics_payload["metadata"]["selected_root_details"] == [
        {
            "state_id": "root-state-1",
            "summary": "West of House.",
            "region_label": "west-house-mailbox",
            "first_seen_episode_id": "glow-runner-smoke",
            "provenance_trajectory_id": "glow-runner-smoke-root",
            "selection_count": 1,
        }
    ]
    assert metrics_payload["metadata"]["top_selected_regions"] == [
        {"region_label": "west-house-mailbox", "count": 1}
    ]
    assert metrics_payload["metadata"]["shadow_commit_threshold"] == 0.5
    assert metrics_payload["metadata"]["shadow_threshold_would_commit_count"] == 0
    assert metrics_payload["metadata"]["threshold_blocked_exploratory_best_branch_count"] == 0


def test_glow_runner_loads_prior_archive_snapshot_for_cross_run_selection(tmp_path: Path) -> None:
    """A fresh GLoW runner should preload the latest archive snapshot before selection."""

    config = _build_config(tmp_path)
    config.ensure_output_directories()
    config.experiment.runner_mode = RunnerMode.GLOW_FAITHFUL
    config.experiment.max_steps = 1
    runner = EpisodeRunner(config=config, llm_client=_FakeLLMClient())
    runner.env = _FakeGlowEnv()  # type: ignore[assignment]
    runner.local_explorer.env = runner.env

    prior_run_state = ArchivedState(
        state_id="archive:mailbox-open",
        provenance_trajectory_id="glow-persistence-run-1-cycle-001-branch-00",
        provenance_timestep=1,
        achieved_value=2.0,
        first_seen_episode_id="glow-persistence-run-1",
        last_seen_episode_id="glow-persistence-run-1",
        provenance_trajectory_ids=["glow-persistence-run-1-cycle-001-branch-00"],
        score_at_state=0,
        projected_potential_value=3.0,
        observation_summary="Opening the mailbox reveals a leaflet.",
        inventory_summary="You are empty-handed.",
        valid_action_summary="take leaflet, north",
        state_cluster_id="house-mailbox",
        replay_metadata=ReplayMetadata(
            restore_strategy="action_replay_fallback",
            replay_actions=["open mailbox"],
            world_state_hash="mailbox-open",
        ),
        metadata={
            "observation": "Opening the mailbox reveals a leaflet.",
            "inventory_text": "You are empty-handed.",
            "valid_actions": ["take leaflet", "north"],
            "score": 0,
            "done": False,
            "world_state_hash": "mailbox-open",
        },
    )
    loaded_snapshot_path = ArchiveStore(config.paths.archive_dir).write_states(
        "glow-persistence-run-1",
        [prior_run_state],
    )

    selection_inputs: list[list[tuple[str, str]]] = []
    restore_calls: list[tuple[str, str]] = []

    def fake_select_archive_state(*, archive_states, frontier_insight, trajectory_frontier, mode=None, selection_id=None):
        selection_inputs.append(
            [(state.state_id, state.first_seen_episode_id) for state in archive_states]
        )
        selected = next(
            state for state in archive_states if state.first_seen_episode_id == "glow-persistence-run-1"
        )
        return ArchiveStateSelectionResult(
            selected_archived_state=selected,
            chosen_replay_method="action_replay_fallback",
            achieved_contribution=selected.achieved_value,
            potential_contribution=float(selected.projected_potential_value or 0.0),
            rationale="Selected the prior-run mailbox state.",
            selection_mode=config.policy.archive_state_selection_mode,
            artifact_directory=str(tmp_path / "selection-artifact"),
        )

    def fake_frontier_analysis(trajectory_frontier, *, archive_states=None, top_k=None, analysis_id=None):
        return FrontierAnalysisResult(
            analysis_id=analysis_id or "analysis-cross-run",
            insight=FrontierInsight(
                analysis_id=analysis_id or "analysis-cross-run",
                frontier_trajectory_ids=[entry.trajectory_id for entry in trajectory_frontier.top_k_entries(2)],
            ),
            prompt_input="frontier prompt",
            artifact_directory=str(tmp_path / "frontier-analysis"),
        )

    def fake_explore_from_state(restored_state, **kwargs):
        return LocalExplorationResult(
            base_state=restored_state,
            branch_count=1,
            branch_horizon=1,
            temperature=0.2,
            action_candidate_count=3,
            root_state_id="archive:mailbox-open",
            best_branch_index=0,
            branch_commit_allowed=False,
            commit_rejection_reason="planning only",
            branches=[
                LocalBranchOutcome(
                    branch_index=0,
                    actions_taken=["take leaflet"],
                    total_reward=0.0,
                    score_change=0,
                    final_score=0,
                    final_observation="Taken.",
                    terminated=False,
                    termination_reason=BranchTerminationReason.HORIZON_REACHED,
                    branch_progress_score=1.0,
                    trajectory_steps=[
                        TrajectoryStep(
                            episode_id="branch-0",
                            step_index=0,
                            action="take leaflet",
                            observation="Taken.",
                            reward=0.0,
                            cumulative_reward=0.0,
                            done=False,
                            score=0,
                            moves=2,
                            world_state_hash="leaflet-taken",
                            inventory_text="You are carrying a leaflet.",
                            valid_actions=["north"],
                        )
                    ],
                )
            ],
        )

    runner.state_selector.select_archive_state = fake_select_archive_state  # type: ignore[method-assign]
    runner.frontier_analyzer.analyze_frontier = fake_frontier_analysis  # type: ignore[method-assign]
    runner.local_explorer.explore_from_state = fake_explore_from_state  # type: ignore[method-assign]

    def fake_restore_archived_state(archived_state: ArchivedState) -> ReplayResult:
        restore_calls.append((archived_state.state_id, archived_state.first_seen_episode_id))
        return ReplayResult(
            success=True,
            restore_mode=RestoreMode.ACTION_REPLAY_FALLBACK,
            divergence_reason=ReplayDivergenceReason.NONE,
            final_observation="Opening the mailbox reveals a leaflet.",
            final_score=0,
            replayed_action_count=1,
            target_step_index=archived_state.provenance_timestep,
            final_state=runner.env.reset(),
            restored_node_id=archived_state.state_id,
            message="mocked cross-run restore",
        )

    runner._restore_archived_state = fake_restore_archived_state  # type: ignore[method-assign]

    result = runner.run_episode(episode_id="glow-persistence-run-2")

    assert selection_inputs
    assert ("archive:mailbox-open", "glow-persistence-run-1") in selection_inputs[0]
    assert restore_calls == [("archive:mailbox-open", "glow-persistence-run-1")]
    assert result.metadata["loaded_archive_snapshot_path"] == str(loaded_snapshot_path)
    assert result.metadata["loaded_archive_state_count"] == 1
    assert result.metadata["loaded_archive_prior_run_state_count"] == 1

    metrics_payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["metadata"]["loaded_archive_snapshot_path"] == str(loaded_snapshot_path)
    assert metrics_payload["metadata"]["loaded_archive_state_count"] == 1
    assert metrics_payload["metadata"]["loaded_archive_prior_run_state_count"] == 1


def test_glow_runner_skips_frontier_analysis_when_disabled(tmp_path: Path) -> None:
    """The no-global-analysis ablation should skip frontier-analysis calls cleanly."""

    config = _build_config(tmp_path)
    config.ensure_output_directories()
    config.experiment.runner_mode = RunnerMode.GLOW_FAITHFUL
    config.experiment.enable_global_frontier_analysis = False
    config.experiment.max_steps = 2
    runner = EpisodeRunner(config=config, llm_client=_FakeLLMClient())
    runner.env = _FakeGlowEnv()  # type: ignore[assignment]
    runner.local_explorer.env = runner.env

    analysis_calls: list[int] = []

    def fake_select_archive_state(*, archive_states, frontier_insight, trajectory_frontier, mode=None, selection_id=None):
        return ArchiveStateSelectionResult(
            selected_archived_state=archive_states[0],
            chosen_replay_method="native_snapshot",
            achieved_contribution=0.0,
            potential_contribution=0.0,
            rationale="Initial root selection.",
            selection_mode=config.policy.archive_state_selection_mode,
        )

    def fake_frontier_analysis(*args, **kwargs):
        analysis_calls.append(1)
        raise AssertionError("frontier analysis should be disabled")

    def fake_explore_from_state(restored_state, **kwargs):
        return LocalExplorationResult(
            base_state=restored_state,
            branch_count=1,
            branch_horizon=1,
            temperature=0.2,
            action_candidate_count=3,
            root_state_id="root-state-1",
            best_branch_index=0,
            branch_commit_allowed=False,
            commit_rejection_reason="no durable gain",
            branches=[
                LocalBranchOutcome(
                    branch_index=0,
                    actions_taken=["open mailbox"],
                    total_reward=0.0,
                    score_change=0,
                    final_score=0,
                    final_observation="Opening the mailbox reveals a leaflet.",
                    terminated=False,
                    termination_reason=BranchTerminationReason.HORIZON_REACHED,
                    trajectory_steps=[
                        TrajectoryStep(
                            episode_id="branch-0",
                            step_index=0,
                            action="open mailbox",
                            observation="Opening the mailbox reveals a leaflet.",
                            reward=0.0,
                            cumulative_reward=0.0,
                            done=False,
                            score=0,
                            moves=1,
                            world_state_hash="mailbox-open",
                            inventory_text="You are empty-handed.",
                            valid_actions=["take leaflet", "north"],
                        )
                    ],
                )
            ],
        )

    runner.state_selector.select_archive_state = fake_select_archive_state  # type: ignore[method-assign]
    runner.frontier_analyzer.analyze_frontier = fake_frontier_analysis  # type: ignore[method-assign]
    runner.local_explorer.explore_from_state = fake_explore_from_state  # type: ignore[method-assign]
    runner._restore_archived_state = lambda archived_state: ReplayResult(  # type: ignore[method-assign]
        success=True,
        restore_mode=RestoreMode.NATIVE_SNAPSHOT,
        divergence_reason=ReplayDivergenceReason.NONE,
        final_observation="West of House.",
        final_score=0,
        replayed_action_count=0,
        target_step_index=archived_state.provenance_timestep,
        final_state=runner.env.reset(),
        restored_node_id=archived_state.state_id,
        message="mocked restore",
    )

    result = runner.run_episode(episode_id="glow-no-frontier-analysis")

    assert analysis_calls == []
    assert result.metrics_path is not None
    metrics_payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["frontier_analysis_count"] == 0


def test_revisit_progress_classification_treats_repeated_local_affordance_as_stale(tmp_path: Path) -> None:
    """Repeated non-scoring local affordances should not reset stale-root counters."""

    runner = EpisodeRunner(config=_build_config(tmp_path), llm_client=None)

    repeated_local_branch = LocalBranchOutcome(
        branch_index=0,
        actions_taken=["north", "take leaves"],
        total_reward=0.0,
        score_change=0,
        final_score=0,
        final_observation="Forest path.",
        terminated=False,
        termination_reason=BranchTerminationReason.HORIZON_REACHED,
        durable_progress=True,
        metadata={"repeated_known_local_affordance": True},
    )

    progress = runner._classify_revisit_progress([repeated_local_branch])  # type: ignore[attr-defined]

    assert progress == {
        "achieved_progress": False,
        "score_progress": False,
        "fresh_durable_progress": False,
        "repeated_local_affordance_only_progress": True,
        "exploratory_progress": False,
    }


def test_revisit_progress_classification_keeps_fresh_durable_progress(tmp_path: Path) -> None:
    """Fresh durable gains should still count as achieved revisit progress."""

    runner = EpisodeRunner(config=_build_config(tmp_path), llm_client=None)

    fresh_branch = LocalBranchOutcome(
        branch_index=0,
        actions_taken=["take lamp"],
        total_reward=0.0,
        score_change=0,
        final_score=0,
        final_observation="You are carrying a lamp.",
        terminated=False,
        termination_reason=BranchTerminationReason.HORIZON_REACHED,
        durable_progress=True,
        metadata={"repeated_known_local_affordance": False},
    )

    progress = runner._classify_revisit_progress([fresh_branch])  # type: ignore[attr-defined]

    assert progress == {
        "achieved_progress": True,
        "score_progress": False,
        "fresh_durable_progress": True,
        "repeated_local_affordance_only_progress": False,
        "exploratory_progress": False,
    }
