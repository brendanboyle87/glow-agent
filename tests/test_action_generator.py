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
    StrategicMode,
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
    assert [candidate.action for candidate in result.candidates][:2] == ["open mailbox", "read leaflet"]
    assert result.candidates[0].confidence == 0.82
    assert result.candidates[0].rationale == "nearby interaction"
    assert result.candidates[0].source == "llm_constrained"
    assert result.candidate_pool_before_rerank == ["open mailbox", "read leaflet", "look"]
    assert result.ranking_source == "hybrid_merge"
    assert result.candidates[0].features is not None
    assert result.candidates[0].features.verb == "open"
    assert result.candidates[0].features.noun_targets == ["mailbox"]


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
    assert set(candidate.action for candidate in result.candidates[:2]) == {"open mailbox", "examine house"}
    assert result.candidates[-1].action == "inventory"
    open_mailbox = next(candidate for candidate in result.candidates if candidate.action == "open mailbox")
    examine_house = next(candidate for candidate in result.candidates if candidate.action == "examine house")
    assert open_mailbox.rationale == "nearby object"
    assert examine_house.confidence == 0.55
    assert examine_house.source == "llm_open"


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
    assert result.ranking_source == "heuristic_rerank"
    assert set(result.candidate_pool_before_rerank) == {"open mailbox", "read leaflet", "look"}
    assert result.candidates[0].action == "open mailbox"
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
    drop_egg = next(candidate for candidate in result.candidates if candidate.action == "drop egg")
    assert drop_egg.loop_penalty == 0.0


def test_action_generator_prefers_structural_access_in_explore_mode(tmp_path: Path) -> None:
    """Explore mode should boost grounded structural actions over generic movement."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient(
            "1. west | generic exploration\n"
            "2. open window | likely access\n"
            "3. enter window | likely access"
        ),
    )

    result = generator.generate(
        observation="You are outside a white house. A window is slightly ajar.",
        inventory_text="You are empty-handed.",
        valid_actions=["west", "south", "open window", "enter window"],
        score=0,
        moves=3,
        strategic_mode=StrategicMode.EXPLORE,
        strategic_reason="Current region still has unexplored structural access.",
        strategic_try_actions=["open window", "enter window"],
        strategic_objects=["window", "house"],
    )

    assert result.strategic_mode == StrategicMode.EXPLORE.value
    assert result.candidates[0].action in {"open window", "enter window"}
    assert result.candidates[0].features is not None
    assert result.candidates[0].features.matches_strategic_try_action is True
    west = next(candidate for candidate in result.candidates if candidate.action == "west")
    assert west.selection_score < result.candidates[0].selection_score


def test_action_generator_prefers_object_followup_in_exploit_mode(tmp_path: Path) -> None:
    """Exploit mode should keep a grounded object follow-up ahead of unguided movement."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient(
            "1. west | maybe explore more\n"
            "2. take egg | collect the object\n"
            "3. examine egg | inspect it"
        ),
    )

    result = generator.generate(
        observation="A jeweled egg rests in a nest.",
        inventory_text="You are empty-handed.",
        valid_actions=["west", "take egg", "examine egg"],
        score=0,
        moves=8,
        strategic_mode=StrategicMode.EXPLOIT,
        strategic_reason="Current region still has grounded unresolved opportunities.",
        strategic_try_actions=["take egg", "examine egg"],
        strategic_objects=["egg", "nest"],
    )

    assert result.strategic_mode == StrategicMode.EXPLOIT.value
    assert result.candidates[0].action in {"take egg", "examine egg"}
    west = next(candidate for candidate in result.candidates if candidate.action == "west")
    assert west.selection_score < result.candidates[0].selection_score


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
    assert result.candidates[0].features is not None
    assert result.candidates[0].features.touches_newly_salient_object is True
    assert "close mailbox" in ranked_actions
    assert ranked_actions.index("take leaflet") < ranked_actions.index("close mailbox")
    assert result.candidates[0].heuristic_bonus > 0.0
    close_candidate = next(candidate for candidate in result.candidates if candidate.action == "close mailbox")
    assert close_candidate.heuristic_penalty > 0.0
    assert close_candidate.loop_penalty > 0.0


def test_action_generator_demotes_stale_toggle_even_with_reflection_support(tmp_path: Path) -> None:
    """Reflection support should not rescue a stale toggle with only affordance-deep history."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. close mailbox | confidence=0.95 | keep working the mailbox"),
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
        observation_changed=True,
        valid_actions_changed=True,
        valid_actions_improved=False,
        revealed_new_object=False,
        target_tokens={"mailbox"},
        inverse_pairs=config.policy.inverse_action_pairs,
    )

    result = generator.generate(
        observation="Opening the small mailbox reveals a leaflet.",
        inventory_text="You are empty-handed.",
        valid_actions=["close mailbox", "take leaflet", "north"],
        score=0,
        moves=2,
        recent_actions=["open mailbox"],
        state_action_history=history,
        supported_try_actions=["close mailbox"],
        supported_reflection_objects=["mailbox", "leaflet"],
    )

    assert result.candidates[0].action == "take leaflet"
    close_candidate = next(candidate for candidate in result.candidates if candidate.action == "close mailbox")
    assert close_candidate.features is not None
    assert close_candidate.features.matches_supported_try_action is True
    assert "stale_toggle_without_durable_gain" in close_candidate.ranking_reason
    assert "reflection_supported_try" not in close_candidate.ranking_reason


def test_action_generator_does_not_promote_movement_from_affordance_only_history(
    tmp_path: Path,
) -> None:
    """Movement should not inherit try/history bonuses when it only changed affordances before."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. east | confidence=0.90 | keep exploring"),
    )
    history = ActionClusterHistory(cluster_label="house-cluster")
    history.record_attempt(
        action="east",
        score_changed=False,
        inventory_changed=False,
        observation_changed=True,
        valid_actions_changed=True,
        valid_actions_improved=False,
        revealed_new_object=False,
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_state_cluster(cluster_id="region:house", observation="East of House.")
    history.record_state_cluster(cluster_id="region:house", observation="East of House.")

    result = generator.generate(
        observation="The window is open and the brass lamp is visible.",
        inventory_text="leaflet",
        valid_actions=["east", "examine lamp", "take lamp", "close window"],
        score=0,
        moves=7,
        candidate_count=4,
        state_action_history=history,
        supported_try_actions=["east"],
        supported_reflection_objects=["lamp", "window"],
    )

    assert result.candidates[0].action in {"take lamp", "examine lamp"}
    east_candidate = next(candidate for candidate in result.candidates if candidate.action == "east")
    assert "historical_gain" not in east_candidate.ranking_reason
    assert "reflection_supported_try" not in east_candidate.ranking_reason


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
        candidate_count=3,
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
    assert "close window" not in [candidate.action for candidate in result.candidates]
    assert any("new_object" in candidate.ranking_reason for candidate in result.candidates[:2])


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


def test_action_generator_prefers_leaflet_interaction_over_repeated_west_when_leaflet_is_visible(
    tmp_path: Path,
) -> None:
    """A newly visible leaflet interaction should beat repeated movement in constrained mode."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(config, prompt_manager, FakeLLMClient("1. west\n2. close mailbox"))
    history = ActionClusterHistory(cluster_label="mailbox-cluster")
    history.observe_state_nouns(
        observation="You are west of the house near a small mailbox.",
        valid_actions=["west", "south"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_state_cluster(cluster_id="region:house-field", observation="West of House.")
    history.record_state_cluster(cluster_id="region:house-field", observation="West of House.")
    history.record_attempt(
        action="west",
        score_changed=False,
        inventory_changed=False,
        observation_changed=False,
        valid_actions_changed=False,
    )

    result = generator.generate(
        observation="Opening the small mailbox reveals a leaflet.",
        inventory_text="You are empty-handed.",
        valid_actions=["west", "south", "read leaflet", "examine leaflet", "close mailbox"],
        score=0,
        moves=4,
        recent_actions=["west"],
        state_action_history=history,
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] in {"read leaflet", "examine leaflet"}
    assert "west" not in ranked_actions or ranked_actions.index("read leaflet") < ranked_actions.index("west")
    assert "west" not in ranked_actions or ranked_actions.index("examine leaflet") < ranked_actions.index("west")


def test_action_generator_does_not_demote_toggle_with_prior_affordance_gain(tmp_path: Path) -> None:
    """A reversible toggle should survive when it previously revealed a real affordance."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(config, prompt_manager, FakeLLMClient("1. open window\n2. west"))
    history = ActionClusterHistory(cluster_label="window-cluster")
    history.record_attempt(
        action="open window",
        score_changed=False,
        inventory_changed=False,
        observation_changed=True,
        valid_actions_changed=True,
        valid_actions_improved=True,
        revealed_new_object=True,
    )

    result = generator.generate(
        observation="The window is shut.",
        inventory_text="You are empty-handed.",
        valid_actions=["open window", "west"],
        score=0,
        moves=3,
        recent_actions=["close window"],
        recent_loop_results=[LoopHeuristicResult(no_progress=False)],
        state_action_history=history,
    )

    open_window = next(candidate for candidate in result.candidates if candidate.action == "open window")
    assert open_window.features is not None
    assert open_window.features.produced_affordance_gain_before is True
    assert open_window.heuristic_penalty < config.policy.reversible_toggle_penalty


def test_action_generator_demotes_exhausted_mailbox_family_actions_after_leaflet_is_handled(
    tmp_path: Path,
) -> None:
    """Exhausted mailbox-family toggles and put-back actions should lose to leaving the cul-de-sac."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(config, prompt_manager, FakeLLMClient("1. open mailbox\n2. put leaflet in mailbox"))
    history = ActionClusterHistory(cluster_label="mailbox-cluster")
    history.observe_state_nouns(
        observation="Opening the small mailbox reveals a leaflet.",
        valid_actions=["take leaflet", "close mailbox", "west"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_attempt(
        action="open mailbox",
        score_changed=False,
        inventory_changed=False,
        observation_changed=True,
        valid_actions_changed=True,
        valid_actions_improved=True,
        revealed_new_object=True,
        target_tokens={"mailbox"},
    )
    history.record_attempt(
        action="close mailbox",
        score_changed=False,
        inventory_changed=False,
        observation_changed=False,
        valid_actions_changed=False,
        target_tokens={"mailbox"},
    )
    history.record_attempt(
        action="close mailbox",
        score_changed=False,
        inventory_changed=False,
        observation_changed=False,
        valid_actions_changed=False,
        target_tokens={"mailbox"},
    )

    result = generator.generate(
        observation="The mailbox is closed. You are carrying a leaflet.",
        inventory_text="You are carrying a leaflet.",
        valid_actions=["open mailbox", "put leaflet in mailbox", "west", "north"],
        score=0,
        moves=5,
        recent_actions=["close mailbox"],
        state_action_history=history,
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] in {"west", "north"}
    assert ranked_actions.index("open mailbox") > 0
    assert ranked_actions.index("put leaflet in mailbox") > 0
    open_mailbox = next(candidate for candidate in result.candidates if candidate.action == "open mailbox")
    put_leaflet = next(candidate for candidate in result.candidates if candidate.action == "put leaflet in mailbox")
    assert "historical_gain" not in open_mailbox.ranking_reason
    assert "exhausted_family:mailbox" in open_mailbox.ranking_reason
    assert "discard_inventory" in put_leaflet.ranking_reason


def test_action_generator_prefers_exit_after_local_scene_already_paid_off(
    tmp_path: Path,
) -> None:
    """After a local object scene already yielded score, exits should beat stale same-scene churn."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. take nest\n2. throw egg at nest\n3. west"),
    )
    history = ActionClusterHistory(cluster_label="egg-cluster")
    history.observe_state_nouns(
        observation="A jeweled egg rests in a nest.",
        valid_actions=["take egg", "take nest", "west"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_attempt(
        action="take egg",
        score_changed=True,
        inventory_changed=True,
        inventory_gained=True,
        observation_changed=True,
        valid_actions_changed=False,
        target_tokens={"egg"},
        inverse_pairs=config.policy.inverse_action_pairs,
    )

    result = generator.generate(
        observation="You are in the tree. A nest is here.",
        inventory_text="You are carrying a jeweled egg and a leaflet.",
        valid_actions=["take nest", "throw egg at nest", "west", "down"],
        score=5,
        moves=8,
        recent_actions=["take egg"],
        state_action_history=history,
        supported_try_actions=["take nest", "throw egg at nest"],
        supported_reflection_objects=["egg", "nest"],
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] in {"west", "down"}
    take_nest = next(candidate for candidate in result.candidates if candidate.action == "take nest")
    throw_egg = next(candidate for candidate in result.candidates if candidate.action == "throw egg at nest")
    assert take_nest.features is not None
    assert take_nest.features.scene_has_score_harvested is True
    assert take_nest.features.local_scene_post_score_stale is True
    assert "post_score_scene_exhausted" in take_nest.ranking_reason
    assert "stale_reflection_try" in take_nest.ranking_reason
    assert "post_score_scene_exhausted" in throw_egg.ranking_reason


def test_action_generator_prefers_take_canary_when_canary_is_freshly_revealed(
    tmp_path: Path,
) -> None:
    """A freshly revealed canary should outrank stale egg/nest churn in the scored tree scene."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. close egg\n2. take nest\n3. down"),
    )
    history = ActionClusterHistory(cluster_label="tree-cluster")
    history.observe_state_nouns(
        observation="Up a Tree. A nest holds a jeweled egg.",
        inventory_text="jewel-encrusted egg",
        valid_actions=["down", "take nest", "take on egg", "close nest"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_attempt(
        action="take egg",
        score_changed=True,
        inventory_changed=True,
        inventory_gained=True,
        observation_changed=True,
        valid_actions_changed=False,
        target_tokens={"egg"},
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_attempt(
        action="take on egg",
        score_changed=False,
        inventory_changed=False,
        observation_changed=True,
        valid_actions_changed=True,
        valid_actions_improved=True,
        revealed_new_object=True,
        target_tokens={"egg"},
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.observe_state_nouns(
        observation=(
            "The egg is open and there is a golden clockwork canary nestled in the egg."
        ),
        inventory_text="broken jewel-encrusted egg",
        valid_actions=["down", "take canary", "close egg", "close nest", "take nest"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )

    result = generator.generate(
        observation="The egg is open and there is a golden clockwork canary nestled in the egg.",
        inventory_text="broken jewel-encrusted egg",
        valid_actions=["down", "take canary", "close egg", "close nest", "take nest"],
        score=5,
        moves=9,
        recent_actions=["take on egg"],
        state_action_history=history,
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] == "take canary"
    take_canary = result.candidates[0]
    assert take_canary.features is not None
    assert take_canary.features.target_is_new_salient_object is True
    close_egg = next(candidate for candidate in result.candidates if candidate.action == "close egg")
    assert "post_score_scene_exhausted" in close_egg.ranking_reason
    assert "take nest" not in ranked_actions or ranked_actions.index("take canary") < ranked_actions.index("take nest")


def test_action_generator_demotes_post_score_inventory_object_churn(
    tmp_path: Path,
) -> None:
    """After scoring with a carried object, exits should outrank destructive local follow-ups."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. take on egg\n2. close egg\n3. down"),
    )
    history = ActionClusterHistory(cluster_label="egg-cluster")
    history.observe_state_nouns(
        observation="Up a Tree. You are carrying a jewel-encrusted egg beside a nest.",
        inventory_text="jewel-encrusted egg",
        valid_actions=["down", "take on egg", "close egg", "take nest"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_attempt(
        action="take egg",
        score_changed=True,
        inventory_changed=True,
        inventory_gained=True,
        observation_changed=True,
        valid_actions_changed=False,
        target_tokens={"egg"},
        inverse_pairs=config.policy.inverse_action_pairs,
    )

    result = generator.generate(
        observation="Up a Tree. Beside you is a nest, and you are carrying a jewel-encrusted egg.",
        inventory_text="jewel-encrusted egg",
        valid_actions=["down", "take on egg", "close egg", "take nest"],
        score=5,
        moves=7,
        recent_actions=["take egg"],
        state_action_history=history,
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] == "down"
    take_on_egg = next(candidate for candidate in result.candidates if candidate.action == "take on egg")
    assert "post_score_inventory_object_preserve" in take_on_egg.ranking_reason
    assert ranked_actions.index("down") < ranked_actions.index("take on egg")
    assert "close egg" not in ranked_actions


def test_action_generator_does_not_let_reflection_try_rescue_stale_nonnew_acquire(
    tmp_path: Path,
) -> None:
    """A non-new acquire action should lose to exits when it only has soft reflection support."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. take nest\n2. down"),
    )
    history = ActionClusterHistory(cluster_label="nest-cluster")
    history.observe_state_nouns(
        observation="You are in the tree beside a nest and a canary.",
        valid_actions=["take nest", "down", "west"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )

    result = generator.generate(
        observation="You are in the tree beside a nest and a canary.",
        inventory_text="You are carrying a jeweled egg and a leaflet.",
        valid_actions=["take nest", "down", "west"],
        score=5,
        moves=12,
        candidate_count=3,
        state_action_history=history,
        supported_try_actions=["take nest", "down"],
        supported_reflection_objects=["nest", "egg"],
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] in {"down", "west"}
    take_nest = next(candidate for candidate in result.candidates if candidate.action == "take nest")
    assert take_nest.features is not None
    assert take_nest.features.matches_supported_try_action is True
    assert "reflection_supported_try" not in take_nest.ranking_reason
    assert "stale_nonnew_acquire" in take_nest.ranking_reason


def test_action_generator_does_not_transfer_object_family_success_to_aggressive_throw(
    tmp_path: Path,
) -> None:
    """Object-family success should not make aggressive throw actions rank like sensible follow-ups."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. throw nest at ground\n2. down"),
    )
    history = ActionClusterHistory(cluster_label="nest-cluster")
    history.observe_state_nouns(
        observation="You are in the tree holding a nest.",
        valid_actions=["throw nest at ground", "down", "close nest"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_attempt(
        action="take nest",
        score_changed=False,
        inventory_changed=True,
        inventory_gained=True,
        observation_changed=True,
        valid_actions_changed=False,
        target_tokens={"nest"},
        inverse_pairs=config.policy.inverse_action_pairs,
    )

    result = generator.generate(
        observation="You are in the tree holding a nest.",
        inventory_text="You are carrying a nest, a jeweled egg, and a leaflet.",
        valid_actions=["throw nest at ground", "down", "close nest"],
        score=5,
        moves=13,
        candidate_count=3,
        state_action_history=history,
        supported_reflection_objects=["nest"],
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] == "down"
    throw_nest = next(candidate for candidate in result.candidates if candidate.action == "throw nest at ground")
    assert throw_nest.features is not None
    assert throw_nest.features.prior_success_for_object_family is True
    assert "object_family_success" not in throw_nest.ranking_reason
    assert "speculative_tool_use" in throw_nest.ranking_reason


def test_action_generator_penalizes_jump_against_plain_exits(tmp_path: Path) -> None:
    """Ungrounded generic actions should lose to simple exits even without extra history support."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. jump\n2. west\n3. east"),
    )
    history = ActionClusterHistory(cluster_label="forest-cluster")
    history.record_attempt(
        action="go around trees",
        score_changed=False,
        inventory_changed=False,
        observation_changed=False,
        valid_actions_changed=False,
    )

    result = generator.generate(
        observation="This is a forest, with trees in all directions. To the east, there appears to be sunlight.",
        inventory_text="You are carrying a leaflet.",
        valid_actions=["west", "east", "jump", "put down leaflet", "northwest"],
        score=0,
        moves=5,
        candidate_count=4,
        recent_actions=["north", "go around trees", "go around trees"],
        state_action_history=history,
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] in {"west", "east", "northwest"}
    assert ranked_actions.index("jump") > ranked_actions.index("west")
    jump_candidate = next(candidate for candidate in result.candidates if candidate.action == "jump")
    assert jump_candidate.features is not None
    assert jump_candidate.features.verb_family == "other"
    assert "ungrounded_other_action" in jump_candidate.ranking_reason
    assert jump_candidate.selection_score < 0.0


def test_action_generator_prefers_up_in_tree_scene_over_generic_escape_movement(
    tmp_path: Path,
) -> None:
    """Climbable tree scenes should prefer `up` over arbitrary lateral wandering."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. east\n2. west\n3. south"),
    )
    history = ActionClusterHistory(cluster_label="forest-cluster")
    history.record_state_cluster(cluster_id="region:forest", observation="Forest path among trees.")
    history.record_state_cluster(cluster_id="region:forest", observation="Forest path among trees.")

    result = generator.generate(
        observation="You are at the base of a large tree. Branches and leaves are overhead.",
        inventory_text="You are carrying a leaflet.",
        valid_actions=["up", "east", "west", "south", "put down leaflet"],
        score=0,
        moves=6,
        candidate_count=4,
        recent_actions=["north", "south"],
        state_action_history=history,
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] == "up"
    up_candidate = next(candidate for candidate in result.candidates if candidate.action == "up")
    assert up_candidate.features is not None
    assert up_candidate.features.movement_targets_landmark is True
    assert "landmark_movement" in up_candidate.ranking_reason


def test_action_generator_demotes_shake_egg_after_the_scene_already_scored(tmp_path: Path) -> None:
    """Odd noun-targeted `other` verbs should not outrank exits after the scene already paid off."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. shake egg\n2. down\n3. take nest"),
    )
    history = ActionClusterHistory(cluster_label="egg-cluster")
    history.observe_state_nouns(
        observation="A jeweled egg rests in a nest.",
        valid_actions=["take egg", "take nest", "down"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_attempt(
        action="take egg",
        score_changed=True,
        inventory_changed=True,
        inventory_gained=True,
        observation_changed=True,
        valid_actions_changed=False,
        target_tokens={"egg"},
        inverse_pairs=config.policy.inverse_action_pairs,
    )

    result = generator.generate(
        observation="You are in the tree beside the nest.",
        inventory_text="You are carrying a jeweled egg and a leaflet.",
        valid_actions=["shake egg", "down", "take nest", "close nest"],
        score=5,
        moves=10,
        candidate_count=4,
        recent_actions=["take egg"],
        state_action_history=history,
        supported_try_actions=["shake egg", "take nest"],
        supported_reflection_objects=["egg", "nest"],
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions.index("down") < ranked_actions.index("shake egg")
    shake_candidate = next(candidate for candidate in result.candidates if candidate.action == "shake egg")
    assert shake_candidate.features is not None
    assert shake_candidate.features.verb_family == "other"
    assert "ungrounded_other_action" in shake_candidate.ranking_reason


def test_action_generator_prefers_take_egg_after_open_egg_revealed_affordances(
    tmp_path: Path,
) -> None:
    """Freshly opened/high-affordance objects should get an immediate follow-up bias."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. down\n2. close egg\n3. throw egg at ground"),
    )
    history = ActionClusterHistory(cluster_label="egg-cluster")
    history.observe_state_nouns(
        observation="There is a jeweled egg here.",
        valid_actions=["open egg with leaflet", "take egg", "down"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_attempt(
        action="open egg with leaflet",
        score_changed=False,
        inventory_changed=False,
        observation_changed=True,
        valid_actions_changed=True,
        valid_actions_improved=True,
        revealed_new_object=True,
        target_tokens={"egg"},
        inverse_pairs=config.policy.inverse_action_pairs,
    )

    result = generator.generate(
        observation="The egg is open. The jeweled egg is exposed beside the nest.",
        inventory_text="You are carrying a leaflet.",
        valid_actions=["take egg", "down", "close egg", "throw egg at ground"],
        score=0,
        moves=8,
        candidate_count=4,
        recent_actions=["open egg with leaflet"],
        state_action_history=history,
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] == "take egg"
    take_egg = next(candidate for candidate in result.candidates if candidate.action == "take egg")
    assert take_egg.features is not None
    assert take_egg.features.touches_recent_affordance_object is True
    assert "fresh_affordance_followup" in take_egg.ranking_reason
    assert ranked_actions.index("take egg") < ranked_actions.index("down")


def test_action_generator_top_k_keeps_object_centric_action_when_salient_objects_exist(
    tmp_path: Path,
) -> None:
    """Top-k diversity should still keep at least one object-centric action around salient objects."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. west\n2. south\n3. go around forest"),
    )
    history = ActionClusterHistory(cluster_label="forest-cluster")
    history.record_state_cluster(cluster_id="region:forest", observation="Forest path among trees.")
    history.record_state_cluster(cluster_id="region:forest", observation="Forest path among trees.")
    history.record_attempt(
        action="go around forest",
        score_changed=False,
        inventory_changed=False,
        observation_changed=False,
        valid_actions_changed=False,
    )

    result = generator.generate(
        observation="You are in a forest clearing. A brass lamp is here.",
        inventory_text="You are empty-handed.",
        valid_actions=["west", "south", "go around forest", "take lamp", "examine lamp"],
        score=0,
        moves=6,
        candidate_count=3,
        recent_actions=["go around forest"],
        state_action_history=history,
    )

    assert any(
        candidate.features is not None
        and not candidate.features.is_movement_action
        and "lamp" in candidate.features.noun_targets
        for candidate in result.candidates
    )


def test_action_generator_prefers_examine_egg_over_throw_leaflet_at_egg(tmp_path: Path) -> None:
    """Canonical inspect actions should beat speculative aggressive tool use on a new object."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. throw leaflet at egg\n2. examine egg"),
    )

    result = generator.generate(
        observation="A jeweled egg rests in a nest.",
        inventory_text="You are carrying a leaflet.",
        valid_actions=["throw leaflet at egg", "examine egg", "north"],
        score=0,
        moves=5,
        candidate_count=3,
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] == "examine egg"
    throw_candidate = next(candidate for candidate in result.candidates if candidate.action == "throw leaflet at egg")
    assert throw_candidate.features is not None
    assert throw_candidate.features.verb_family == "aggressive"
    assert throw_candidate.features.action_shape_is_complex_transitive is True
    assert "complex_transitive" in throw_candidate.ranking_reason
    assert "speculative_tool_use" in throw_candidate.ranking_reason


def test_action_generator_prefers_take_egg_over_throw_leaflet_at_nest(tmp_path: Path) -> None:
    """Acquire actions on a newly encountered object should outrank speculative throws."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. throw leaflet at nest\n2. take egg"),
    )

    result = generator.generate(
        observation="A jeweled egg is visible in a nest.",
        inventory_text="You are carrying a leaflet.",
        valid_actions=["throw leaflet at nest", "take egg", "west"],
        score=0,
        moves=5,
        candidate_count=3,
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] == "take egg"
    assert ranked_actions.index("take egg") < ranked_actions.index("throw leaflet at nest")


def test_action_generator_prefers_open_egg_over_close_egg_for_unexplored_object(tmp_path: Path) -> None:
    """New container-like objects should prefer `open` over `close` before they are explored."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. close egg\n2. open egg"),
    )

    result = generator.generate(
        observation="A jeweled egg lies here.",
        inventory_text="You are empty-handed.",
        valid_actions=["close egg", "open egg", "look"],
        score=0,
        moves=5,
        candidate_count=3,
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] == "open egg"
    close_candidate = next(candidate for candidate in result.candidates if candidate.action == "close egg")
    assert close_candidate.features is not None
    assert close_candidate.features.verb_family == "access"
    assert "canonical_new_object" in close_candidate.ranking_reason


def test_action_generator_prefers_read_leaflet_over_throw_leaflet_at_egg(tmp_path: Path) -> None:
    """Readable inventory objects should favor canonical read/inspect actions before odd tool use."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. throw leaflet at egg\n2. read leaflet"),
    )

    result = generator.generate(
        observation="A jeweled egg sits in a nest.",
        inventory_text="You are carrying a leaflet.",
        valid_actions=["throw leaflet at egg", "read leaflet", "north"],
        score=0,
        moves=5,
        candidate_count=3,
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] == "read leaflet"
    read_candidate = next(candidate for candidate in result.candidates if candidate.action == "read leaflet")
    assert read_candidate.features is not None
    assert read_candidate.features.target_is_readable_candidate is True
    assert "readable_candidate" in read_candidate.ranking_reason


def test_action_generator_demotes_bulk_inventory_commands_without_score_evidence(tmp_path: Path) -> None:
    """Bulk inventory commands should lose to specific object interactions when nothing supports them."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. take all\n2. take nest\n3. put down all"),
    )

    result = generator.generate(
        observation="A jeweled egg and a nest are here.",
        inventory_text="You are carrying a leaflet.",
        valid_actions=["take all", "take nest", "put down all", "north"],
        score=0,
        moves=5,
        candidate_count=4,
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] == "take nest"

    take_all = next(candidate for candidate in result.candidates if candidate.action == "take all")
    put_down_all = next(candidate for candidate in result.candidates if candidate.action == "put down all")
    assert take_all.features is not None
    assert take_all.features.is_bulk_inventory_action is True
    assert "bulk_inventory_action" in take_all.ranking_reason
    assert "bulk_inventory_action" in put_down_all.ranking_reason
    assert ranked_actions.index("take nest") < ranked_actions.index("take all")
    assert ranked_actions.index("take nest") < ranked_actions.index("put down all")


def test_action_generator_keeps_speculative_tool_use_available_but_low_ranked_without_evidence(
    tmp_path: Path,
) -> None:
    """Speculative tool-use actions should remain available but trail canonical alternatives."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. use leaflet on egg\n2. examine egg"),
    )

    result = generator.generate(
        observation="A jeweled egg is in a nest.",
        inventory_text="You are carrying a leaflet.",
        valid_actions=["use leaflet on egg", "examine egg", "open egg", "north"],
        score=0,
        moves=5,
        candidate_count=4,
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert "use leaflet on egg" in ranked_actions
    assert ranked_actions.index("use leaflet on egg") > ranked_actions.index("examine egg")
    speculative = next(candidate for candidate in result.candidates if candidate.action == "use leaflet on egg")
    assert speculative.features is not None
    assert speculative.features.verb_family == "use"
    assert speculative.features.action_shape_is_complex_transitive is True
    assert "speculative_tool_use" in speculative.ranking_reason


def test_action_generator_demotes_stale_leaflet_retake_after_local_inventory_churn(tmp_path: Path) -> None:
    """Reacquiring a recently dropped local object should not outrank an exit in a stale cluster."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. take leaflet\n2. east"),
    )
    history = ActionClusterHistory(cluster_label="window-cluster")
    history.observe_state_nouns(
        observation="Behind House. An open window is here.",
        inventory_text="You are empty-handed.",
        valid_actions=["open window", "take leaflet", "east"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_state_cluster(cluster_id="region:window", observation="Behind House.")
    history.record_state_cluster(cluster_id="region:window", observation="Behind House.")
    history.record_state_cluster(cluster_id="region:window", observation="Behind House.")
    history.record_attempt(
        action="take leaflet",
        score_changed=False,
        inventory_changed=True,
        inventory_gained=True,
        observation_changed=False,
        valid_actions_changed=False,
        target_tokens={"leaflet"},
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_attempt(
        action="put down leaflet",
        score_changed=False,
        inventory_changed=True,
        inventory_lost=True,
        observation_changed=False,
        valid_actions_changed=False,
        target_tokens={"leaflet"},
        inverse_pairs=config.policy.inverse_action_pairs,
        discard_like_action=True,
    )

    result = generator.generate(
        observation="Behind House. The window is open and the leaflet lies here.",
        inventory_text="You are empty-handed.",
        valid_actions=["take leaflet", "east", "west"],
        score=0,
        moves=12,
        candidate_count=3,
        state_action_history=history,
        strategic_mode=StrategicMode.EXPLORE,
        strategic_try_actions=["east", "west"],
        strategic_avoid_actions=["take leaflet"],
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] in {"east", "west"}
    retake = next(candidate for candidate in result.candidates if candidate.action == "take leaflet")
    assert retake.features is not None
    assert retake.features.prior_inventory_gain_for_object_family > 0
    assert retake.features.prior_inventory_loss_for_object_family > 0
    assert "stale_reacquire_churn" in retake.ranking_reason


def test_action_generator_prefers_exit_after_secure_leaflet_pickup(tmp_path: Path) -> None:
    """Once the leaflet is secured, exits should beat stale mailbox-scene churn."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. close mailbox\n2. north\n3. put down leaflet"),
    )
    history = ActionClusterHistory(cluster_label="mailbox-cluster")
    history.record_state_cluster(cluster_id="region:mailbox", observation="West of House.")
    history.record_state_cluster(cluster_id="region:mailbox", observation="West of House.")
    history.observe_state_nouns(
        observation="Opening the small mailbox reveals a leaflet.",
        inventory_text="You are empty-handed.",
        valid_actions=["take leaflet", "close mailbox", "north", "south"],
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_attempt(
        action="open mailbox",
        score_changed=False,
        inventory_changed=False,
        observation_changed=True,
        valid_actions_changed=True,
        valid_actions_improved=True,
        revealed_new_object=True,
        target_tokens={"mailbox"},
        revealed_object_tokens={"leaflet"},
        inverse_pairs=config.policy.inverse_action_pairs,
    )
    history.record_attempt(
        action="take leaflet",
        score_changed=False,
        inventory_changed=True,
        inventory_gained=True,
        observation_changed=False,
        valid_actions_changed=False,
        target_tokens={"leaflet"},
        inverse_pairs=config.policy.inverse_action_pairs,
    )

    result = generator.generate(
        observation="West of House. The small mailbox is open.",
        inventory_text="You are carrying a leaflet.",
        valid_actions=["close mailbox", "north", "put down leaflet", "south"],
        score=0,
        moves=2,
        candidate_count=4,
        state_action_history=history,
        strategic_mode=StrategicMode.EXPLOIT,
        strategic_try_actions=["north"],
    )

    ranked_actions = [candidate.action for candidate in result.candidates]
    assert ranked_actions[0] == "north"
    close_mailbox = next(candidate for candidate in result.candidates if candidate.action == "close mailbox")
    put_down_leaflet = next(candidate for candidate in result.candidates if candidate.action == "put down leaflet")
    assert ranked_actions.index("north") < ranked_actions.index("close mailbox")
    assert ranked_actions.index("north") < ranked_actions.index("put down leaflet")
    assert result.top_selection_reason
    assert "escape_mode_exit" in result.top_selection_reason
    assert "historical_gain" not in put_down_leaflet.ranking_reason


def test_action_generator_does_not_treat_discard_churn_as_exact_success(tmp_path: Path) -> None:
    """Discard-like local inventory churn should not bootstrap exact-action success."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    generator = ActionGenerator(
        config,
        prompt_manager,
        FakeLLMClient("1. put down leaflet\n2. north"),
    )
    history = ActionClusterHistory(cluster_label="mailbox-cluster")
    history.record_state_cluster(cluster_id="region:mailbox", observation="West of House.")
    history.record_state_cluster(cluster_id="region:mailbox", observation="West of House.")
    history.record_attempt(
        action="put down leaflet",
        score_changed=False,
        inventory_changed=True,
        inventory_lost=True,
        observation_changed=True,
        valid_actions_changed=True,
        valid_actions_improved=True,
        revealed_new_object=True,
        target_tokens={"leaflet"},
        revealed_object_tokens={"leaflet"},
        inverse_pairs=config.policy.inverse_action_pairs,
        discard_like_action=True,
    )

    result = generator.generate(
        observation="West of House. The leaflet lies here.",
        inventory_text="You are empty-handed.",
        valid_actions=["put down leaflet", "north", "south"],
        score=0,
        moves=4,
        candidate_count=3,
        state_action_history=history,
        strategic_mode=StrategicMode.EXPLORE,
        strategic_try_actions=["north", "south"],
    )

    put_down_leaflet = next(candidate for candidate in result.candidates if candidate.action == "put down leaflet")
    assert put_down_leaflet.features is not None
    assert put_down_leaflet.features.prior_success_for_exact_action is False
    assert "exact_action_success" not in put_down_leaflet.ranking_reason
