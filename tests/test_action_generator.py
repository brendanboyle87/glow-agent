"""Tests for LLM-backed parser action generation.

TODO: add tests for longer recent-trajectory contexts if prompt shaping becomes more complex.
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
from zork_agent.policy.action_generator import ActionGenerator
from zork_agent.types import (
    ActionClusterHistory,
    ActionGenerationMode,
    LLMChatRequest,
    LLMResponse,
    LoopHeuristicResult,
)


class FakeLLMClient(BaseLLMClient):
    """Small fake LLM client for policy tests."""

    def __init__(self, text: str):
        super().__init__(default_model="fake-model")
        self.text = text

    def list_models(self) -> list[str]:
        return ["fake-model"]

    def complete_chat(self, request: LLMChatRequest) -> LLMResponse:
        return LLMResponse(
            text=self.text,
            model=request.model or "fake-model",
            latency_seconds=0.01,
            raw_payload={"messages": request.messages},
        )


def _build_config(tmp_path: Path) -> ProjectConfig:
    """Create a minimal config for action-generation tests."""

    rom_path = tmp_path / "games" / "zork1.z5"
    rom_path.parent.mkdir(parents=True, exist_ok=True)
    rom_path.write_text("fake rom", encoding="utf-8")
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "system.txt").write_text("System for {game_id}", encoding="utf-8")
    (prompt_dir / "action.txt").write_text(
        "Observation: {observation}\nMode: {generation_mode}\n{valid_actions_block}\nRecent: {recent_trajectory_context}\nTop {action_candidates}",
        encoding="utf-8",
    )
    (prompt_dir / "trajectory.txt").write_text("Trajectory: {trajectory_excerpt}", encoding="utf-8")
    (prompt_dir / "reflection.txt").write_text("Reflection: {state_summary}", encoding="utf-8")
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
        policy=PolicyConfig(action_candidates=3),
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


def test_action_generator_parses_constrained_candidates_against_valid_actions(tmp_path: Path) -> None:
    """Constrained mode should map noisy model lines back onto valid Jericho actions."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient(
            "1. open mailbox | confidence=0.82 | nearby interaction\n"
            "2. read leaflet | confidence=0.35 | if the leaflet is present\n"
            "3. dance wildly"
        ),
    )

    result = generator.generate(
        observation="You are west of the house near a small mailbox.",
        inventory_text="You are empty-handed.",
        recent_trajectory_context="look -> examine house",
        valid_actions=["look", "open mailbox", "read leaflet"],
        score=0,
        moves=1,
    )

    assert result.mode is ActionGenerationMode.CONSTRAINED
    assert result.raw_output.startswith("1. open mailbox")
    assert [candidate.action for candidate in result.candidates] == ["open mailbox", "read leaflet"]
    assert result.candidates[0].confidence == 0.82
    assert result.candidates[0].rationale == "nearby interaction"
    assert result.candidates[0].source == "llm_constrained"


def test_action_generator_parses_open_mode_rationales_and_confidence(tmp_path: Path) -> None:
    """Open mode should parse free-form actions without requiring strict formatting."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient(
            "1. open mailbox - nearby object\n"
            "2. examine house | confidence=0.55 | broader context\n"
            "3. inventory"
        ),
    )

    result = generator.generate(
        observation="You are standing west of a white house beside a mailbox.",
        inventory_text="You are empty-handed.",
        recent_trajectory_context="(none)",
        valid_actions=[],
        score=0,
        moves=0,
    )

    assert result.mode is ActionGenerationMode.OPEN
    assert [candidate.action for candidate in result.candidates] == ["open mailbox", "examine house", "inventory"]
    assert result.candidates[0].rationale == "nearby object"
    assert result.candidates[1].confidence == 0.55
    assert result.candidates[1].source == "llm_open"


def test_action_generator_falls_back_when_llm_output_is_unusable(tmp_path: Path) -> None:
    """Unusable model output should trigger deterministic fallback actions."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(config, prompt_manager, FakeLLMClient("I am not sure what to do here."))

    result = generator.generate(
        observation="You are standing west of a white house beside a mailbox.",
        inventory_text="You are empty-handed.",
        valid_actions=["look", "open mailbox", "read leaflet"],
        score=0,
        moves=0,
    )

    assert result.mode is ActionGenerationMode.FALLBACK
    assert result.fallback_reason == "LLM output was unusable."
    assert result.raw_output == "I am not sure what to do here."
    assert [candidate.action for candidate in result.candidates][:2] == ["open mailbox", "look"]
    assert result.candidates[0].source == "fallback_constrained"


def test_action_generator_penalizes_inverse_candidate_after_no_progress_loop(tmp_path: Path) -> None:
    """Immediate inverse candidates should be reranked downward after a no-progress loop signal."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient(
            "1. close mailbox | confidence=0.91 | reverse it\n"
            "2. look | confidence=0.30 | reset context"
        ),
    )

    result = generator.generate(
        observation="Opening the small mailbox reveals a leaflet.",
        inventory_text="You are empty-handed.",
        valid_actions=["close mailbox", "look"],
        score=0,
        moves=1,
        recent_actions=["open mailbox"],
        recent_loop_results=[
            LoopHeuristicResult(
                loop_detected=True,
                no_progress=True,
                total_penalty=1.75,
                inverse_of_previous=True,
                reason="inverse_of_previous, no_progress",
            )
        ],
    )

    assert result.reranked_by_loop_penalty is True
    assert [candidate.action for candidate in result.candidates] == ["look", "close mailbox"]
    assert result.candidates[1].loop_penalty > 0.0
    assert "inverse_of_previous" in result.candidates[1].loop_penalty_reason


def test_action_generator_does_not_penalize_inverse_candidate_after_real_gain(tmp_path: Path) -> None:
    """Inverse candidates should stay in place when the last reversal produced real gain."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient(
            "1. drop egg | confidence=0.65 | maybe place it back\n"
            "2. inventory | confidence=0.20 | inspect state"
        ),
    )

    result = generator.generate(
        observation="You are holding the jeweled egg.",
        inventory_text="jeweled egg",
        valid_actions=["drop egg", "inventory"],
        score=5,
        moves=4,
        recent_actions=["take egg"],
        recent_loop_results=[
            LoopHeuristicResult(
                loop_detected=False,
                no_progress=False,
                total_penalty=0.0,
                inverse_of_previous=True,
            )
        ],
    )

    assert result.reranked_by_loop_penalty is False
    assert [candidate.action for candidate in result.candidates] == ["drop egg", "inventory"]
    assert result.candidates[0].loop_penalty == 0.0


def test_action_generator_prefers_take_leaflet_over_repeated_close_mailbox_with_directions_present(
    tmp_path: Path,
) -> None:
    """A revealed object interaction should outrank a stale mailbox toggle even with movement options present."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient(
            "1. close mailbox | confidence=0.92 | tidy up first\n"
            "2. look | confidence=0.10 | reset"
        ),
    )
    history = ActionClusterHistory(cluster_label="mailbox-cluster")
    history.observe_state_nouns(
        observation="You are west of the house near a small mailbox.",
        valid_actions=["open mailbox", "look"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_attempt(
        action="close mailbox",
        score_changed=False,
        inventory_changed=False,
        observation_changed=False,
        valid_actions_changed=False,
    )
    history.record_attempt(
        action="close mailbox",
        score_changed=False,
        inventory_changed=False,
        observation_changed=False,
        valid_actions_changed=False,
    )

    result = generator.generate(
        observation="Opening the small mailbox reveals a leaflet.",
        inventory_text="You are empty-handed.",
        valid_actions=["close mailbox", "take leaflet", "north", "south", "look"],
        score=0,
        moves=2,
        candidate_count=5,
        recent_actions=["open mailbox"],
        recent_loop_results=[
            LoopHeuristicResult(
                loop_detected=True,
                no_progress=True,
                total_penalty=1.75,
                inverse_of_previous=True,
                reason="inverse_of_previous, no_progress",
            )
        ],
        state_action_history=history,
    )

    assert result.augmented_with_valid_actions is True
    assert result.reranked_by_affordance_heuristics is True
    ranked_actions = [candidate.action for candidate in result.candidates]

    assert result.candidates[0].action == "take leaflet"
    assert "close mailbox" in ranked_actions
    assert ranked_actions.index("take leaflet") < ranked_actions.index("close mailbox")
    assert result.candidates[0].heuristic_bonus > 0.0
    close_candidate = next(candidate for candidate in result.candidates if candidate.action == "close mailbox")
    assert close_candidate.heuristic_penalty > 0.0
    assert close_candidate.loop_penalty > 0.0


def test_action_generator_prefers_untried_object_interaction_over_stale_toggle(tmp_path: Path) -> None:
    """Untried salient object interactions should beat repeated no-gain toggle actions."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient(
            "1. close window | confidence=0.80 | secure things\n"
            "2. look | confidence=0.30 | inspect again"
        ),
    )
    history = ActionClusterHistory(cluster_label="window-cluster")
    history.observe_state_nouns(
        observation="A dusty room with a narrow window.",
        valid_actions=["open window", "look"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_attempt(
        action="close window",
        score_changed=False,
        inventory_changed=False,
        observation_changed=False,
        valid_actions_changed=False,
    )

    result = generator.generate(
        observation="The window is open and a brass lamp is visible on the sill.",
        inventory_text="You are empty-handed.",
        valid_actions=["close window", "examine lamp", "take lamp", "look"],
        score=0,
        moves=3,
        recent_actions=["open window"],
        recent_loop_results=[
            LoopHeuristicResult(
                loop_detected=True,
                no_progress=True,
                total_penalty=1.75,
                inverse_of_previous=True,
                reason="inverse_of_previous, no_progress",
            )
        ],
        state_action_history=history,
    )

    assert result.candidates[0].action in {"take lamp", "examine lamp"}
    assert result.candidates[-1].action == "close window"
    assert any("new_noun:lamp" in candidate.ranking_reason for candidate in result.candidates[:2])


def test_action_generator_penalizes_repeated_go_around_trees_and_prefers_take_leaflet(
    tmp_path: Path,
) -> None:
    """Repeated movement wandering should lose to an untried salient object interaction."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient(
            "1. go around trees | confidence=0.85 | keep exploring\n"
            "2. west | confidence=0.40 | maybe a new path"
        ),
    )
    history = ActionClusterHistory(cluster_label="forest-cluster")
    history.observe_state_nouns(
        observation="You are in a forest clearing.",
        valid_actions=["go around trees", "west"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_state_cluster(cluster_id="region:forest", observation="Forest path among trees.")
    history.record_state_cluster(cluster_id="region:forest", observation="Forest path among trees.")
    history.record_attempt(
        action="go around trees",
        score_changed=False,
        inventory_changed=False,
        observation_changed=False,
        valid_actions_changed=False,
    )
    history.record_attempt(
        action="go around trees",
        score_changed=False,
        inventory_changed=False,
        observation_changed=False,
        valid_actions_changed=False,
    )

    result = generator.generate(
        observation="You are in a forest clearing. A leaflet lies on the ground.",
        inventory_text="You are empty-handed.",
        valid_actions=["go around trees", "west", "take leaflet", "look"],
        score=0,
        moves=5,
        candidate_count=4,
        recent_actions=["go around trees", "go around trees"],
        state_action_history=history,
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert result.reranked_by_movement_heuristics is True
    assert ranked_actions[0] == "take leaflet"
    movement_candidate = next(candidate for candidate in result.candidates if candidate.action == "go around trees")
    assert movement_candidate.movement_only_action is True
    assert movement_candidate.movement_penalty > 0.0
    assert "same_region_repeat" in movement_candidate.movement_penalty_reason
    assert ranked_actions.index("take leaflet") < ranked_actions.index("go around trees")


def test_action_generator_prefers_untried_object_interaction_over_repeated_movement(
    tmp_path: Path,
) -> None:
    """A visible object affordance should outrank stale movement in the same region cluster."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient(
            "1. west | confidence=0.70 | keep moving\n"
            "2. go around forest | confidence=0.65 | scan more area"
        ),
    )
    history = ActionClusterHistory(cluster_label="forest-cluster")
    history.observe_state_nouns(
        observation="You are in a forest clearing.",
        valid_actions=["west", "go around forest"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_state_cluster(cluster_id="region:forest", observation="Forest path among trees.")
    history.record_state_cluster(cluster_id="region:forest", observation="Forest path among trees.")
    history.record_attempt(
        action="west",
        score_changed=False,
        inventory_changed=False,
        observation_changed=False,
        valid_actions_changed=False,
    )

    result = generator.generate(
        observation="You are in a forest clearing beside a white house window. A brass lamp is visible.",
        inventory_text="You are empty-handed.",
        valid_actions=["west", "go around forest", "take lamp", "examine lamp"],
        score=0,
        moves=6,
        recent_actions=["west"],
        state_action_history=history,
    )

    assert result.candidates[0].action in {"take lamp", "examine lamp"}
    assert result.candidates[-1].movement_only_action is True
