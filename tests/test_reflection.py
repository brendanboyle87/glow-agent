"""Tests for compact operational reflection over branches and trajectories.

TODO: add prompt-budget regression tests if reflection summaries become longer.
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
from zork_agent.llm.base import BaseLLMClient
from zork_agent.llm.prompts import PromptManager
from zork_agent.policy.reflection import ReflectionEngine
from zork_agent.types import (
    BranchTerminationReason,
    LLMChatRequest,
    LLMResponse,
    LocalBranchOutcome,
    LocalExplorationResult,
    ReflectionMode,
    TextGameState,
    TrajectoryStep,
)


class FakeLLMClient(BaseLLMClient):
    """Small fake LLM client for reflection tests."""

    def __init__(self, text: str):
        super().__init__(default_model="fake-model")
        self.text = text
        self.last_request: LLMChatRequest | None = None

    def list_models(self) -> list[str]:
        return ["fake-model"]

    def complete_chat(self, request: LLMChatRequest) -> LLMResponse:
        self.last_request = request
        return LLMResponse(
            text=self.text,
            model=request.model or "fake-model",
            latency_seconds=0.01,
            raw_payload={"messages": request.messages},
        )


def _build_config(tmp_path: Path) -> ProjectConfig:
    """Create a minimal config for reflection tests."""

    rom_path = tmp_path / "games" / "zork1.z5"
    rom_path.parent.mkdir(parents=True, exist_ok=True)
    rom_path.write_text("fake rom", encoding="utf-8")
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "system.txt").write_text("System for {game_id}", encoding="utf-8")
    (prompt_dir / "action.txt").write_text("Action prompt", encoding="utf-8")
    (prompt_dir / "trajectory.txt").write_text(
        "Episode {episode_id}\nSeed {seed}\n{trajectory_excerpt}\nGrounding constraints:\n{grounding_constraints}",
        encoding="utf-8",
    )
    (prompt_dir / "reflection.txt").write_text(
        "State summary:\n{state_summary}\nRecent actions:\n{recent_actions}\nGrounding constraints:\n{grounding_constraints}",
        encoding="utf-8",
    )
    (prompt_dir / "selection.txt").write_text("Frontier: {frontier_snapshot}", encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    return ProjectConfig(
        runtime=RuntimeConfig(jericho_game_path=rom_path, use_live_jericho=False),
        llm=LLMConfig(model_name="fake-model"),
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
        experiment=ExperimentConfig(game_id="zork1"),
    )


def _local_exploration_result() -> LocalExplorationResult:
    """Build a small local exploration fixture with one good and one bad branch."""

    base_state = TextGameState(
        observation="West of House.",
        inventory_text="lamp",
        valid_actions=["open mailbox", "examine house", "look"],
        score=0,
        moves=0,
        world_state_hash="hash-west",
    )
    return LocalExplorationResult(
        base_state=base_state,
        branch_count=2,
        branch_horizon=2,
        temperature=0.1,
        action_candidate_count=3,
        branches=[
            LocalBranchOutcome(
                branch_index=0,
                actions_taken=["open mailbox", "read leaflet"],
                total_reward=1.0,
                score_change=1,
                final_score=1,
                final_observation="The leaflet welcomes you to Zork.",
                terminated=True,
                termination_reason=BranchTerminationReason.TERMINATED,
                new_room_or_object_detected=True,
                appears_stuck=False,
                new_affordance_count=1,
                persistent_affordance_gain=1,
                notable_observation_changes=["score increased by 1", "new tokens: leaflet, zork"],
                metadata={
                    "movement_results": [],
                    "action_events": [
                        {
                            "action": "open mailbox",
                            "score_delta": 0,
                            "inventory_gained": False,
                            "persistent_affordance_gain": 1,
                            "persistent_exit_gain_count": 0,
                            "revealed_new_object": True,
                            "loop_detected": False,
                            "loop_penalty": 0.0,
                            "movement_penalty": 0.0,
                        },
                        {
                            "action": "read leaflet",
                            "score_delta": 1,
                            "inventory_gained": False,
                            "persistent_affordance_gain": 0,
                            "persistent_exit_gain_count": 0,
                            "revealed_new_object": False,
                            "loop_detected": False,
                            "loop_penalty": 0.0,
                            "movement_penalty": 0.0,
                        },
                    ],
                },
            ),
            LocalBranchOutcome(
                branch_index=1,
                actions_taken=["look", "look"],
                total_reward=0.0,
                score_change=0,
                final_score=0,
                final_observation="West of House.",
                terminated=False,
                termination_reason=BranchTerminationReason.HORIZON_REACHED,
                new_room_or_object_detected=False,
                appears_stuck=True,
                notable_observation_changes=["branch appears stuck"],
            ),
        ],
        best_branch_index=0,
        comparison_notes="compare good and bad local branches",
    )


def test_reflection_engine_parses_llm_rollout_guidance(tmp_path: Path) -> None:
    """Labeled LLM output should be normalized into compact operational guidance."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient(
        "try_actions: open mailbox; read leaflet\n"
        "salient_objects: mailbox; leaflet\n"
        "avoid_actions: repeated look\n"
        "discovered_affordances: mailbox -> open; leaflet -> read\n"
        "short_guidance_text: prioritize mailbox and leaflet interactions"
    )
    engine = ReflectionEngine(config, prompt_manager, llm_client)

    result = engine.reflect_rollouts(_local_exploration_result())

    assert result.mode is ReflectionMode.LLM
    assert result.guidance.try_actions == ["open mailbox", "read leaflet"]
    assert result.guidance.salient_objects[:2] == ["mailbox", "leaflet"]
    assert result.guidance.avoid_actions == ["look"]
    assert result.guidance.discovered_affordances == ["mailbox -> open", "leaflet -> read"]
    assert "Try: open mailbox; read leaflet." in result.normalized_guidance_text
    assert result.prompt_context == result.normalized_guidance_text
    assert llm_client.last_request is not None
    assert "State summary:" in llm_client.last_request.messages[1]["content"]
    assert "Grounding constraints:" in llm_client.last_request.messages[1]["content"]


def test_reflection_engine_falls_back_to_heuristics_for_rollouts(tmp_path: Path) -> None:
    """Malformed model output should fall back to deterministic branch heuristics."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient("not sure, maybe the second one?")
    engine = ReflectionEngine(config, prompt_manager, llm_client)

    result = engine.reflect_rollouts(_local_exploration_result())

    assert result.mode is ReflectionMode.FALLBACK
    assert "open mailbox" in result.guidance.try_actions
    assert "read leaflet" in result.guidance.try_actions
    assert "look" in result.guidance.avoid_actions[0]
    assert "mailbox -> open" in result.guidance.discovered_affordances
    assert result.raw_output == "not sure, maybe the second one?"


def test_reflection_engine_keeps_episode_runner_trajectory_path_working(tmp_path: Path) -> None:
    """The backward-compatible `reflect(steps)` path should still return compact text."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    engine = ReflectionEngine(config, prompt_manager, llm_client=None)
    steps = [
        TrajectoryStep(
            episode_id="episode-001",
            step_index=0,
            action="open mailbox",
            observation="A leaflet is visible.",
            reward=1.0,
            done=False,
            score=1,
            moves=1,
            world_state_hash="hash-mailbox",
        ),
        TrajectoryStep(
            episode_id="episode-001",
            step_index=1,
            action="read leaflet",
            observation="The leaflet welcomes you to Zork.",
            reward=0.0,
            done=False,
            score=1,
            moves=2,
            world_state_hash="hash-leaflet",
        ),
    ]

    result = engine.reflect_trajectory(steps)

    assert result.mode is ReflectionMode.FALLBACK
    assert "open mailbox" in result.guidance.try_actions
    assert "mailbox -> open" in result.guidance.discovered_affordances
    assert engine.reflect(steps) == result.normalized_guidance_text


def test_reflection_engine_does_not_promote_bulk_inventory_commands(tmp_path: Path) -> None:
    """Bulk inventory churn should not survive into grounded try-actions."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    engine = ReflectionEngine(config, prompt_manager, llm_client=None)

    result = engine.reflect_rollouts(
        LocalExplorationResult(
            base_state=TextGameState(observation="A room.", inventory_text="", valid_actions=["take all", "look"]),
            branch_count=1,
            branch_horizon=3,
            temperature=0.1,
            action_candidate_count=3,
            branches=[
                LocalBranchOutcome(
                    branch_index=0,
                    actions_taken=["take all", "put down all", "take all"],
                    total_reward=0.0,
                    score_change=0,
                    final_score=0,
                    final_observation="You are carrying a pile of objects.",
                    terminated=False,
                    termination_reason=BranchTerminationReason.HORIZON_REACHED,
                    inventory_changed=True,
                    persistent_inventory_gain_count=2,
                    persistent_inventory_loss_count=1,
                    branch_progress_score=0.0,
                    metadata={
                        "action_events": [
                            {
                                "action": "take all",
                                "score_delta": 0,
                                "inventory_gained": True,
                                "persistent_affordance_gain": 0,
                                "persistent_exit_gain_count": 0,
                                "revealed_new_object": False,
                                "loop_detected": False,
                                "loop_penalty": 0.0,
                                "movement_penalty": 0.0,
                            }
                        ]
                    },
                )
            ],
            best_branch_index=0,
        )
    )

    assert "take all" not in result.guidance.try_actions


def test_reflection_engine_strips_meta_template_leakage(tmp_path: Path) -> None:
    """Prompt/template leakage should be removed before guidance enters memory."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient(
        "objective: keep the response concise\n"
        "format: use the requested schema\n"
        "try_actions: open mailbox; read leaflet; open window\n"
        "avoid_actions: repeated look; task: follow the format\n"
        "salient_objects: mailbox; leaflet; prompt\n"
        "discovered_affordances: mailbox -> open; leaflet -> read; window -> open\n"
        "hypotheses: leaflet may be readable after opening mailbox; format: one item per line\n"
        "short_guidance_text: task: follow the format. leaflet likely matters after opening the mailbox.\n"
    )
    engine = ReflectionEngine(config, prompt_manager, llm_client)

    result = engine.reflect_rollouts(_local_exploration_result())

    assert result.mode is ReflectionMode.LLM
    assert result.guidance.try_actions == ["open mailbox", "read leaflet"]
    assert result.guidance.avoid_actions == ["look"]
    assert result.guidance.salient_objects == ["mailbox", "leaflet"]
    assert result.guidance.discovered_affordances == ["mailbox -> open", "leaflet -> read"]
    normalized = result.normalized_guidance_text.lower()
    assert "objective:" not in normalized
    assert "task:" not in normalized
    assert "format:" not in normalized
    assert "follow the format" not in normalized
    assert "open window" not in normalized
    assert "prompt" not in normalized


def test_reflection_engine_keeps_take_leaflet_in_try_and_out_of_avoid(tmp_path: Path) -> None:
    """A productive object interaction should not be blacklisted just because another branch wandered."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient(
        "try_actions: open mailbox; take leaflet\n"
        "avoid_actions: take leaflet; go around trees\n"
        "salient_objects: mailbox; leaflet\n"
        "discovered_affordances: mailbox -> open; leaflet -> take\n"
        "short_guidance_text: maybe avoid take leaflet after the mailbox"
    )
    engine = ReflectionEngine(config, prompt_manager, llm_client)
    base_state = TextGameState(
        observation="West of House.",
        inventory_text="",
        valid_actions=["open mailbox", "go around trees", "look"],
        score=0,
        moves=0,
        world_state_hash="hash-west",
    )
    exploration = LocalExplorationResult(
        base_state=base_state,
        branch_count=2,
        branch_horizon=2,
        temperature=0.1,
        action_candidate_count=3,
        branches=[
            LocalBranchOutcome(
                branch_index=0,
                actions_taken=["open mailbox", "take leaflet"],
                total_reward=0.0,
                score_change=0,
                final_score=0,
                final_observation="You are now carrying the leaflet.",
                terminated=False,
                termination_reason=BranchTerminationReason.HORIZON_REACHED,
                inventory_changed=True,
                persistent_inventory_gain_count=1,
                new_affordance_count=1,
                persistent_affordance_gain=1,
                affordance_gain=1,
                new_room_or_object_detected=True,
                novel_object_count=1,
                appears_stuck=False,
                durable_progress=True,
                metadata={
                    "movement_results": [],
                    "action_events": [
                        {
                            "action": "open mailbox",
                            "score_delta": 0,
                            "inventory_gained": False,
                            "persistent_affordance_gain": 1,
                            "persistent_exit_gain_count": 0,
                            "revealed_new_object": True,
                            "loop_detected": False,
                            "loop_penalty": 0.0,
                            "movement_penalty": 0.0,
                        },
                        {
                            "action": "take leaflet",
                            "score_delta": 0,
                            "inventory_gained": True,
                            "persistent_affordance_gain": 0,
                            "persistent_exit_gain_count": 0,
                            "revealed_new_object": False,
                            "loop_detected": False,
                            "loop_penalty": 0.0,
                            "movement_penalty": 0.0,
                        },
                    ],
                },
            ),
            LocalBranchOutcome(
                branch_index=1,
                actions_taken=["go around trees", "go around trees"],
                total_reward=0.0,
                score_change=0,
                final_score=0,
                final_observation="Forest path among trees.",
                terminated=False,
                termination_reason=BranchTerminationReason.HORIZON_REACHED,
                appears_stuck=True,
                movement_penalty_total=2.0,
                metadata={
                    "movement_results": [
                        {"total_penalty": 1.0, "movement_repeat_count": 1},
                        {"total_penalty": 1.0, "movement_repeat_count": 2},
                    ]
                },
            ),
        ],
        best_branch_index=0,
        branch_commit_allowed=True,
    )

    result = engine.reflect_rollouts(exploration)

    assert result.guidance.try_actions == ["open mailbox", "take leaflet"]
    assert "take leaflet" not in result.guidance.avoid_actions
    assert "go around trees" in result.guidance.avoid_actions
    assert "take leaflet" in result.guidance.unsupported_items_removed


def test_reflection_engine_drops_close_mailbox_from_try_when_it_only_reflects_local_churn(
    tmp_path: Path,
) -> None:
    """A close-toggle should not survive as a try-action when only reveal/carry actions were productive."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient(
        "try_actions: close mailbox; take leaflet\n"
        "avoid_actions: repeated look\n"
        "salient_objects: mailbox; leaflet\n"
        "discovered_affordances: mailbox -> open; leaflet -> take\n"
        "short_guidance_text: keep working the mailbox before leaving"
    )
    engine = ReflectionEngine(config, prompt_manager, llm_client)
    base_state = TextGameState(
        observation="West of House.",
        inventory_text="",
        valid_actions=["open mailbox", "close mailbox", "look"],
        score=0,
        moves=0,
        world_state_hash="hash-west",
    )
    exploration = LocalExplorationResult(
        base_state=base_state,
        branch_count=1,
        branch_horizon=3,
        temperature=0.1,
        action_candidate_count=3,
        branches=[
            LocalBranchOutcome(
                branch_index=0,
                actions_taken=["open mailbox", "take leaflet", "close mailbox"],
                total_reward=0.0,
                score_change=0,
                final_score=0,
                final_observation="The mailbox is closed.",
                terminated=False,
                termination_reason=BranchTerminationReason.HORIZON_REACHED,
                inventory_changed=True,
                persistent_inventory_gain_count=1,
                persistent_affordance_gain=1,
                affordance_gain=1,
                new_room_or_object_detected=True,
                novel_object_count=1,
                appears_stuck=False,
                durable_progress=True,
                metadata={
                    "movement_results": [],
                    "action_events": [
                        {
                            "action": "open mailbox",
                            "score_delta": 0,
                            "inventory_gained": False,
                            "persistent_affordance_gain": 1,
                            "persistent_exit_gain_count": 0,
                            "revealed_new_object": True,
                            "loop_detected": False,
                            "loop_penalty": 0.0,
                            "movement_penalty": 0.0,
                        },
                        {
                            "action": "take leaflet",
                            "score_delta": 0,
                            "inventory_gained": True,
                            "persistent_affordance_gain": 0,
                            "persistent_exit_gain_count": 0,
                            "revealed_new_object": False,
                            "loop_detected": False,
                            "loop_penalty": 0.0,
                            "movement_penalty": 0.0,
                        },
                        {
                            "action": "close mailbox",
                            "score_delta": 0,
                            "inventory_gained": False,
                            "persistent_affordance_gain": 1,
                            "persistent_exit_gain_count": 0,
                            "revealed_new_object": False,
                            "loop_detected": False,
                            "loop_penalty": 0.0,
                            "movement_penalty": 0.0,
                        },
                    ],
                },
            )
        ],
        best_branch_index=0,
        branch_commit_allowed=True,
    )

    result = engine.reflect_rollouts(exploration)

    assert "take leaflet" in result.guidance.try_actions
    assert "close mailbox" not in result.guidance.try_actions
    assert "close mailbox" in result.guidance.unsupported_items_removed


def test_reflection_engine_does_not_promote_movement_that_only_changed_affordances(tmp_path: Path) -> None:
    """Movement should not enter try-actions when it only exposed text/affordances without durable gain."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient(
        "try_actions: east; take lamp\n"
        "avoid_actions: repeated west\n"
        "salient_objects: lamp; window\n"
        "discovered_affordances: window -> open; lamp -> take\n"
        "short_guidance_text: go east to keep exploring"
    )
    engine = ReflectionEngine(config, prompt_manager, llm_client)
    steps = [
        TrajectoryStep(
            episode_id="episode-002",
            step_index=0,
            action="east",
            observation="The window is open and a brass lamp is visible.",
            reward=0.0,
            done=False,
            score=0,
            moves=1,
            world_state_hash="hash-east-house",
            metadata={
                "durable_progress": True,
                "persistent_affordance_gain": 1,
                "persistent_exit_gain_count": 0,
                "revealed_new_object": True,
            },
        ),
        TrajectoryStep(
            episode_id="episode-002",
            step_index=1,
            action="take lamp",
            observation="Taken.",
            reward=0.0,
            done=False,
            score=0,
            moves=2,
            world_state_hash="hash-lamp",
            inventory_text="lamp",
            metadata={
                "inventory_gained": True,
                "durable_progress": True,
            },
        ),
    ]

    result = engine.reflect_trajectory(steps)

    assert "take lamp" in result.guidance.try_actions
    assert "east" not in result.guidance.try_actions
    assert "east" in result.guidance.unsupported_items_removed


def test_reflection_engine_fallback_promotes_salient_objects_and_demotes_wandering(tmp_path: Path) -> None:
    """Fallback reflection should prefer productive object actions over repeated wandering."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    engine = ReflectionEngine(config, prompt_manager, llm_client=None)
    base_state = TextGameState(
        observation="West of House.",
        inventory_text="",
        valid_actions=["open mailbox", "go around trees", "look"],
        score=0,
        moves=0,
        world_state_hash="hash-west",
    )
    exploration = LocalExplorationResult(
        base_state=base_state,
        branch_count=2,
        branch_horizon=2,
        temperature=0.1,
        action_candidate_count=3,
        branches=[
            LocalBranchOutcome(
                branch_index=0,
                actions_taken=["open mailbox", "take leaflet"],
                total_reward=0.0,
                score_change=0,
                final_score=0,
                final_observation="You are now carrying the leaflet.",
                terminated=False,
                termination_reason=BranchTerminationReason.HORIZON_REACHED,
                inventory_changed=True,
                persistent_inventory_gain_count=1,
                new_affordance_count=1,
                persistent_affordance_gain=1,
                affordance_gain=1,
                new_room_or_object_detected=True,
                novel_object_count=1,
                appears_stuck=False,
                durable_progress=True,
                metadata={
                    "movement_results": [],
                    "action_events": [
                        {
                            "action": "open mailbox",
                            "score_delta": 0,
                            "inventory_gained": False,
                            "persistent_affordance_gain": 1,
                            "persistent_exit_gain_count": 0,
                            "revealed_new_object": True,
                            "loop_detected": False,
                            "loop_penalty": 0.0,
                            "movement_penalty": 0.0,
                        },
                        {
                            "action": "take leaflet",
                            "score_delta": 0,
                            "inventory_gained": True,
                            "persistent_affordance_gain": 0,
                            "persistent_exit_gain_count": 0,
                            "revealed_new_object": False,
                            "loop_detected": False,
                            "loop_penalty": 0.0,
                            "movement_penalty": 0.0,
                        },
                    ],
                },
            ),
            LocalBranchOutcome(
                branch_index=1,
                actions_taken=["go around trees", "go around trees"],
                total_reward=0.0,
                score_change=0,
                final_score=0,
                final_observation="Forest path among trees.",
                terminated=False,
                termination_reason=BranchTerminationReason.HORIZON_REACHED,
                appears_stuck=True,
                movement_penalty_total=2.0,
                metadata={
                    "movement_results": [
                        {"total_penalty": 1.0, "movement_repeat_count": 1},
                        {"total_penalty": 1.0, "movement_repeat_count": 2},
                    ]
                },
            ),
        ],
        best_branch_index=0,
        branch_commit_allowed=True,
    )

    result = engine.reflect_rollouts(exploration)

    assert result.mode is ReflectionMode.FALLBACK
    assert "take leaflet" in result.guidance.try_actions
    assert "leaflet" in result.guidance.salient_objects
    assert "go around trees" in result.guidance.avoid_actions
