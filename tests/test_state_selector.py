"""Tests for archive-state selection and deterministic fallbacks.

TODO: add evaluation-facing tests once batch experiments compare selector modes directly.
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
from zork_agent.llm.base import BaseLLMClient
from zork_agent.llm.prompts import PromptManager
from zork_agent.memory.frontier import FrontierEntry, FrontierQueue
from zork_agent.policy.state_selector import StateSelector
from zork_agent.types import (
    ArchivedState,
    CriticalStateAnnotation,
    FrontierInsight,
    LLMChatRequest,
    LLMResponse,
    ReplayMetadata,
    StateSelectionMode,
)


class FakeLLMClient(BaseLLMClient):
    """Small fake LLM client for selector tests."""

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
    """Create a minimal config for selector tests."""

    rom_path = tmp_path / "games" / "zork1.z5"
    rom_path.parent.mkdir(parents=True, exist_ok=True)
    rom_path.write_text("fake rom", encoding="utf-8")
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "system.txt").write_text("System for {game_id}", encoding="utf-8")
    (prompt_dir / "action.txt").write_text("Action prompt", encoding="utf-8")
    (prompt_dir / "trajectory.txt").write_text("Trajectory prompt", encoding="utf-8")
    (prompt_dir / "frontier_analysis.txt").write_text("Frontier analysis prompt", encoding="utf-8")
    (prompt_dir / "reflection.txt").write_text("Reflection prompt", encoding="utf-8")
    (prompt_dir / "selection.txt").write_text(
        "Candidate archive states:\n{frontier_snapshot}\nPick one.",
        encoding="utf-8",
    )
    artifacts_dir = tmp_path / "artifacts"
    return ProjectConfig(
        runtime=RuntimeConfig(jericho_game_path=rom_path, use_live_jericho=False),
        llm=LLMConfig(model_name="fake-model"),
        prompts=PromptConfig(
            directory=prompt_dir,
            system_file="system.txt",
            action_proposal_file="action.txt",
            trajectory_analysis_file="trajectory.txt",
            frontier_analysis_file="frontier_analysis.txt",
            local_reflection_file="reflection.txt",
            state_selection_file="selection.txt",
        ),
        policy=PolicyConfig(frontier_max_size=8, state_selection_prompt_candidate_limit=4),
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


def _build_frontier() -> FrontierQueue:
    """Create a small frontier with deterministic ordering."""

    frontier = FrontierQueue(max_size=5)
    frontier.add(
        FrontierEntry(
            state_id="mailbox-state",
            score=1.0,
            depth=1,
            novelty=0.2,
            recent_gain=0.0,
            observation="You are beside the mailbox.",
            summary_text="West of House with a small mailbox nearby and a very long note about the surroundings.",
        )
    )
    frontier.add(
        FrontierEntry(
            state_id="leaflet-state",
            score=2.0,
            depth=2,
            novelty=0.5,
            recent_gain=1.0,
            observation="The mailbox is open and a leaflet is visible.",
            summary_text="Open mailbox branch with leaflet ready to inspect.",
        )
    )
    frontier.add(
        FrontierEntry(
            state_id="window-state",
            score=0.5,
            depth=1,
            novelty=0.1,
            recent_gain=0.0,
            observation="A window reflects the field.",
            summary_text="Window branch.",
        )
    )
    return frontier


def test_state_selector_heuristic_mode_uses_frontier_ordering() -> None:
    """Heuristic mode should pick the current best frontier entry."""

    frontier = _build_frontier()
    selector = StateSelector(summary_max_chars=48)

    result = selector.select_with_details(frontier)

    assert result.selection_mode is StateSelectionMode.HEURISTIC
    assert result.selected is not None
    assert result.selected.state_id == "leaflet-state"
    assert "priority=" in result.reason
    assert result.candidate_summaries[0].candidate_id == "C1"
    assert len(result.candidate_summaries[1].summary_text) <= 48


def test_state_selector_llm_mode_selects_from_compact_candidate_summaries(tmp_path: Path) -> None:
    """LLM-assisted mode should choose a candidate by compact prompt id."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient(
        '{'
        '"choice":"C2",'
        '"reason":"higher novelty with fresh score gain."'
        '}'
    )
    selector = StateSelector(
        config=config,
        prompt_manager=prompt_manager,
        llm_client=llm_client,
        selection_mode=StateSelectionMode.LLM_ASSISTED,
        prompt_candidate_limit=3,
        summary_max_chars=64,
    )
    frontier = _build_frontier()

    result = selector.select_with_details(frontier)

    assert result.selection_mode is StateSelectionMode.LLM_ASSISTED
    assert result.selected is not None
    assert result.selected.state_id == "mailbox-state"
    assert result.reason == "higher novelty with fresh score gain."
    assert "C1 | score=" in result.prompt_snapshot
    assert "C2 | score=" in result.prompt_snapshot
    assert llm_client.last_request is not None
    assert "Candidate archive states:" in llm_client.last_request.messages[1]["content"]
    assert llm_client.last_request.response_format is not None


def test_state_selector_falls_back_when_llm_output_is_malformed(tmp_path: Path) -> None:
    """Malformed LLM output should fall back to deterministic heuristic selection."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient(
        "Thinking Process:\n"
        "1. C2 appears interesting.\n"
        "2. reason: maybe the second branch is promising.\n"
        "No final answer yet."
    )
    selector = StateSelector(
        config=config,
        prompt_manager=prompt_manager,
        llm_client=llm_client,
        selection_mode=StateSelectionMode.LLM_ASSISTED,
    )
    frontier = _build_frontier()

    result = selector.select_with_details(frontier)

    assert result.selection_mode is StateSelectionMode.FALLBACK_HEURISTIC
    assert result.selected is not None
    assert result.selected.state_id == "leaflet-state"
    assert result.fallback_reason == "LLM output did not identify a valid candidate."
    assert "Thinking Process:" in result.raw_output


def _build_archive_states() -> list[ArchivedState]:
    """Create a small archived-state fixture with replay metadata."""

    return [
        ArchivedState(
            state_id="archive:high-achieved",
            provenance_trajectory_id="traj-a",
            provenance_timestep=4,
            achieved_value=8.0,
            first_seen_episode_id="episode-a",
            last_seen_episode_id="episode-a",
            replay_metadata=ReplayMetadata(
                restore_strategy="native_snapshot",
                replay_actions=["open mailbox", "take leaflet", "north"],
                world_state_hash="hash-high-achieved",
                native_snapshot_reference="native:high-achieved",
                native_snapshot=("native-high-achieved",),
            ),
            native_snapshot=("native-high-achieved",),
            metadata={"state_cluster_id": "house-exterior"},
        ),
        ArchivedState(
            state_id="archive:bottleneck-window",
            provenance_trajectory_id="traj-b",
            provenance_timestep=3,
            achieved_value=4.0,
            first_seen_episode_id="episode-b",
            last_seen_episode_id="episode-b",
            replay_metadata=ReplayMetadata(
                restore_strategy="action_replay_fallback",
                replay_actions=["west", "north", "east"],
                world_state_hash="hash-bottleneck-window",
            ),
            metadata={"state_cluster_id": "house-window"},
        ),
        ArchivedState(
            state_id="archive:low-value",
            provenance_trajectory_id="traj-c",
            provenance_timestep=2,
            achieved_value=1.0,
            first_seen_episode_id="episode-c",
            last_seen_episode_id="episode-c",
            replay_metadata=ReplayMetadata(
                restore_strategy="action_replay_fallback",
                replay_actions=["south"],
                world_state_hash="hash-low-value",
            ),
            metadata={"state_cluster_id": "forest"},
        ),
    ]


def test_archive_selector_prefers_high_achieved_state_when_potential_is_low(tmp_path: Path) -> None:
    """Balanced archive selection should exploit the best achieved state when potential is weak elsewhere."""

    config = _build_config(tmp_path)
    selector = StateSelector(config=config)

    result = selector.select_archive_state(
        archive_states=_build_archive_states(),
        frontier_insight=FrontierInsight(
            analysis_id="insight-001",
            frontier_trajectory_ids=["traj-a", "traj-b"],
            inferred_bottlenecks=["window remains unopened"],
        ),
        mode=StateSelectionMode.ARCHIVE_BALANCED,
        selection_id="archive-select-high-achieved",
    )

    assert result.selected_archive_state_id == "archive:high-achieved"
    assert result.achieved_contribution > result.potential_contribution
    assert result.chosen_replay_method == "native_snapshot"
    assert Path(result.artifact_directory, "selection_result.json").exists()


def test_archive_selector_prefers_bottleneck_adjacent_state_when_potential_is_high(tmp_path: Path) -> None:
    """Balanced archive selection should prefer bottleneck-adjacent states when potential dominates."""

    config = _build_config(tmp_path)
    selector = StateSelector(config=config)

    result = selector.select_archive_state(
        archive_states=_build_archive_states(),
        frontier_insight=FrontierInsight(
            analysis_id="insight-002",
            frontier_trajectory_ids=["traj-b", "traj-c"],
            inferred_bottlenecks=["house window is a repeated bottleneck"],
            candidate_critical_states=[
                CriticalStateAnnotation(
                    critical_state_id="archive:bottleneck-window",
                    achieved_value=4.0,
                    potential_value=10.0,
                    textual_rationale="The window state appears before every high-value continuation.",
                    source_frontier_trajectory_ids=["traj-b", "traj-c"],
                    confidence=0.8,
                    support_count=2,
                    supporting_state_ids=["archive:bottleneck-window"],
                )
            ],
        ),
        mode=StateSelectionMode.ARCHIVE_BALANCED,
        selection_id="archive-select-high-potential",
    )

    assert result.selected_archive_state_id == "archive:bottleneck-window"
    assert result.potential_contribution > result.achieved_contribution
    assert result.selected_critical_state_ids == ["archive:bottleneck-window"]
    assert "potential contribution" in result.rationale


def test_archive_selector_artifacts_include_provenance_and_restore_availability(tmp_path: Path) -> None:
    """Archive-selection artifacts should expose candidate provenance across runs."""

    config = _build_config(tmp_path)
    selector = StateSelector(config=config)

    result = selector.select_archive_state(
        archive_states=_build_archive_states(),
        frontier_insight=FrontierInsight(analysis_id="insight-artifacts"),
        mode=StateSelectionMode.ARCHIVE_BALANCED,
        selection_id="archive-select-artifacts",
    )

    candidate_scores = json.loads(
        Path(result.artifact_directory, "candidate_scores.json").read_text(encoding="utf-8")
    )
    high_achieved = next(item for item in candidate_scores if item["state_id"] == "archive:high-achieved")
    bottleneck = next(item for item in candidate_scores if item["state_id"] == "archive:bottleneck-window")

    assert high_achieved["first_seen_episode_id"] == "episode-a"
    assert high_achieved["replay_method"] == "native_snapshot"
    assert high_achieved["has_native_snapshot"] is True
    assert bottleneck["first_seen_episode_id"] == "episode-b"
    assert bottleneck["replay_method"] == "action_replay_fallback"
    assert bottleneck["recorded_restore_strategy"] == "action_replay_fallback"


def test_archive_selector_extracts_clean_reason_from_noisy_llm_output(tmp_path: Path) -> None:
    """Archive selection should ignore reasoning preambles and keep only the final rationale."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient(
        '{'
        '"choice":"C1",'
        '"reason":"bottleneck-adjacent state offers the highest potential."'
        '}'
    )
    selector = StateSelector(
        config=config,
        prompt_manager=prompt_manager,
        llm_client=llm_client,
        archive_selection_mode=StateSelectionMode.LLM_ASSISTED,
    )

    result = selector.select_archive_state(
        archive_states=_build_archive_states(),
        frontier_insight=FrontierInsight(
            analysis_id="insight-llm-001",
            frontier_trajectory_ids=["traj-b", "traj-c"],
            inferred_bottlenecks=["house window is a repeated bottleneck"],
            candidate_critical_states=[
                CriticalStateAnnotation(
                    critical_state_id="archive:bottleneck-window",
                    achieved_value=4.0,
                    potential_value=10.0,
                    textual_rationale="The window state appears before every high-value continuation.",
                    source_frontier_trajectory_ids=["traj-b", "traj-c"],
                    confidence=0.8,
                    support_count=2,
                    supporting_state_ids=["archive:bottleneck-window"],
                )
            ],
        ),
        mode=StateSelectionMode.LLM_ASSISTED,
        selection_id="archive-select-clean-reason",
    )

    assert result.selection_mode is StateSelectionMode.LLM_ASSISTED
    assert result.selected_archive_state_id == "archive:bottleneck-window"
    assert result.rationale == "bottleneck-adjacent state offers the highest potential."


def test_archive_selector_debug_mode_can_parse_terminal_structured_block(tmp_path: Path) -> None:
    """Debug-mode selection should ignore reasoning and parse the final structured block."""

    config = _build_config(tmp_path)
    config.llm.analysis_debug_mode = True
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient(
        "Thinking Process:\n"
        "1. Compare C1 and C2.\n"
        "2. C1 is bottleneck-adjacent.\n"
        "BEGIN_STRUCTURED_OUTPUT\n"
        '{"choice":"C1","reason":"bottleneck-adjacent state offers the highest potential."}\n'
        "END_STRUCTURED_OUTPUT"
    )
    selector = StateSelector(
        config=config,
        prompt_manager=prompt_manager,
        llm_client=llm_client,
        archive_selection_mode=StateSelectionMode.LLM_ASSISTED,
    )

    result = selector.select_archive_state(
        archive_states=_build_archive_states(),
        frontier_insight=FrontierInsight(
            analysis_id="insight-debug-001",
            frontier_trajectory_ids=["traj-b", "traj-c"],
            inferred_bottlenecks=["house window is a repeated bottleneck"],
            candidate_critical_states=[
                CriticalStateAnnotation(
                    critical_state_id="archive:bottleneck-window",
                    achieved_value=4.0,
                    potential_value=10.0,
                    textual_rationale="The window state appears before every high-value continuation.",
                    source_frontier_trajectory_ids=["traj-b", "traj-c"],
                    confidence=0.8,
                    support_count=2,
                    supporting_state_ids=["archive:bottleneck-window"],
                )
            ],
        ),
        mode=StateSelectionMode.LLM_ASSISTED,
        selection_id="archive-select-debug-block",
    )

    assert result.selection_mode is StateSelectionMode.LLM_ASSISTED
    assert result.selected_archive_state_id == "archive:bottleneck-window"
    assert result.rationale == "bottleneck-adjacent state offers the highest potential."


def test_archive_selector_penalizes_repeated_zero_achievement_revisits(tmp_path: Path) -> None:
    """Repeated no-achievement revisits should reduce a root's future selection priority."""

    config = _build_config(tmp_path)
    selector = StateSelector(config=config)
    fresh_state = ArchivedState(
        state_id="archive:fresh-window",
        provenance_trajectory_id="traj-fresh",
        provenance_timestep=3,
        achieved_value=0.0,
        projected_potential_value=4.0,
        first_seen_episode_id="episode-fresh",
        last_seen_episode_id="episode-fresh",
        replay_metadata=ReplayMetadata(
            restore_strategy="action_replay_fallback",
            replay_actions=["west", "north", "east"],
            world_state_hash="hash-fresh-window",
        ),
        metadata={"state_cluster_id": "house-window"},
    )
    stale_state = ArchivedState(
        state_id="archive:stale-window",
        provenance_trajectory_id="traj-stale",
        provenance_timestep=3,
        achieved_value=0.0,
        projected_potential_value=4.0,
        first_seen_episode_id="episode-stale",
        last_seen_episode_id="episode-stale",
        replay_metadata=ReplayMetadata(
            restore_strategy="action_replay_fallback",
            replay_actions=["west", "north", "east"],
            world_state_hash="hash-stale-window",
        ),
        metadata={
            "state_cluster_id": "house-window",
            "no_achievement_revisit_streak": 3,
            "nonproductive_revisit_streak": 1,
        },
    )

    result = selector.select_archive_state(
        archive_states=[stale_state, fresh_state],
        frontier_insight=FrontierInsight(analysis_id="insight-stale-revisit"),
        mode=StateSelectionMode.ARCHIVE_BALANCED,
        selection_id="archive-select-stale-revisit",
    )

    candidate_scores = json.loads(
        Path(result.artifact_directory, "candidate_scores.json").read_text(encoding="utf-8")
    )
    stale_record = next(item for item in candidate_scores if item["state_id"] == "archive:stale-window")
    fresh_record = next(item for item in candidate_scores if item["state_id"] == "archive:fresh-window")

    assert result.selected_archive_state_id == "archive:fresh-window"
    assert stale_record["stale_revisit_penalty"] > 0.0
    assert stale_record["total_score"] < fresh_record["total_score"]
    assert stale_record["no_achievement_revisit_streak"] == 3


def test_archive_selector_penalizes_replay_saturated_productive_root(tmp_path: Path) -> None:
    """Replay-saturated productive roots should lose priority to equally promising unsaturated roots."""

    config = _build_config(tmp_path)
    selector = StateSelector(config=config)
    saturated_state = ArchivedState(
        state_id="archive:saturated-root",
        provenance_trajectory_id="traj-saturated",
        provenance_timestep=5,
        achieved_value=5.0,
        projected_potential_value=6.0,
        first_seen_episode_id="episode-saturated",
        last_seen_episode_id="episode-saturated",
        replay_metadata=ReplayMetadata(
            restore_strategy="native_snapshot",
            replay_actions=["open mailbox"],
            world_state_hash="hash-saturated-root",
            native_snapshot_reference="native:saturated-root",
            native_snapshot=("native-saturated-root",),
        ),
        native_snapshot=("native-saturated-root",),
        metadata={
            "state_cluster_id": "house-window",
            "depth_progression": {
                "root_state_id": "archive:saturated-root",
                "productive_root": True,
                "productive_commit_count": 3,
                "materially_distinct_commit_count": 1,
                "repeated_winning_continuation_count": 2,
                "replay_saturation_level": 2,
                "unique_committed_prefix_count": 1,
                "committed_prefix_counts": {"open mailbox": 3},
                "saturated_try_actions": ["open mailbox"],
            },
        },
    )
    fresh_state = ArchivedState(
        state_id="archive:fresh-root",
        provenance_trajectory_id="traj-fresh",
        provenance_timestep=5,
        achieved_value=5.0,
        projected_potential_value=5.6,
        first_seen_episode_id="episode-fresh",
        last_seen_episode_id="episode-fresh",
        replay_metadata=ReplayMetadata(
            restore_strategy="action_replay_fallback",
            replay_actions=["north"],
            world_state_hash="hash-fresh-root",
        ),
        metadata={"state_cluster_id": "house-window"},
    )

    result = selector.select_archive_state(
        archive_states=[saturated_state, fresh_state],
        frontier_insight=FrontierInsight(analysis_id="insight-depth-progression"),
        mode=StateSelectionMode.ARCHIVE_BALANCED,
        selection_id="archive-select-depth-progression",
    )

    candidate_scores = json.loads(
        Path(result.artifact_directory, "candidate_scores.json").read_text(encoding="utf-8")
    )
    saturated_record = next(item for item in candidate_scores if item["state_id"] == "archive:saturated-root")
    fresh_record = next(item for item in candidate_scores if item["state_id"] == "archive:fresh-root")

    assert result.selected_archive_state_id == "archive:fresh-root"
    assert result.metadata["pre_depth_progression_top_state_id"] == "archive:saturated-root"
    assert result.metadata["post_depth_progression_top_state_id"] == "archive:fresh-root"
    assert saturated_record["depth_progression_penalty"] > 0.0
    assert saturated_record["total_score_before_depth_progression"] > fresh_record["total_score_before_depth_progression"]
    assert saturated_record["total_score"] < fresh_record["total_score"]


def test_archive_selector_no_potential_value_ablation_prefers_achieved_value(tmp_path: Path) -> None:
    """The no-potential-value ablation should suppress bottleneck-driven promotion."""

    config = _build_config(tmp_path)
    config.experiment.enable_potential_value_selection = False
    selector = StateSelector(config=config)

    result = selector.select_archive_state(
        archive_states=_build_archive_states(),
        frontier_insight=FrontierInsight(
            analysis_id="insight-004",
            frontier_trajectory_ids=["traj-b", "traj-c"],
            inferred_bottlenecks=["house window is a repeated bottleneck"],
            candidate_critical_states=[
                CriticalStateAnnotation(
                    critical_state_id="archive:bottleneck-window",
                    achieved_value=4.0,
                    potential_value=20.0,
                    textual_rationale="This state appears before the higher-value continuations.",
                    source_frontier_trajectory_ids=["traj-b", "traj-c"],
                    confidence=0.9,
                    support_count=2,
                    supporting_state_ids=["archive:bottleneck-window"],
                )
            ],
        ),
        mode=StateSelectionMode.ARCHIVE_BALANCED,
        selection_id="archive-select-no-potential",
    )

    assert result.selected_archive_state_id == "archive:high-achieved"
    assert result.potential_contribution == 0.0
    assert result.achieved_contribution > 0.0


def test_archive_selector_llm_adjudication_can_choose_among_top_candidates(tmp_path: Path) -> None:
    """Optional LLM adjudication should choose among the top archive candidates only."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient("choice: C2\nreason: closer to the repeated bottleneck with higher future upside.")
    selector = StateSelector(
        config=config,
        prompt_manager=prompt_manager,
        llm_client=llm_client,
        archive_selection_mode=StateSelectionMode.LLM_ASSISTED,
    )

    result = selector.select_archive_state(
        archive_states=_build_archive_states(),
        frontier_insight=FrontierInsight(
            analysis_id="insight-003",
            frontier_trajectory_ids=["traj-a", "traj-b"],
            candidate_critical_states=[
                CriticalStateAnnotation(
                    critical_state_id="archive:bottleneck-window",
                    achieved_value=4.0,
                    potential_value=6.0,
                    textual_rationale="The window state gates later value.",
                    source_frontier_trajectory_ids=["traj-b"],
                    confidence=0.75,
                    support_count=1,
                    supporting_state_ids=["archive:bottleneck-window"],
                )
            ],
        ),
        mode=StateSelectionMode.LLM_ASSISTED,
        selection_id="archive-select-llm",
    )

    assert result.selection_mode is StateSelectionMode.LLM_ASSISTED
    assert result.selected_archive_state_id == "archive:bottleneck-window"
    assert result.rationale == "closer to the repeated bottleneck with higher future upside."
    assert llm_client.last_request is not None
    assert "Candidate archive states:" in llm_client.last_request.messages[1]["content"]
