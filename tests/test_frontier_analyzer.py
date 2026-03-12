"""Tests for the global frontier-analysis stage.

TODO: add longer prompt-budget tests once the frontier carries more trajectories.
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
from zork_agent.memory.archive_updater import ArchiveUpdater
from zork_agent.memory.frontier import TrajectoryFrontier, TrajectoryFrontierConfig
from zork_agent.policy.frontier_analyzer import FrontierAnalyzer
from zork_agent.types import (
    EpisodeTrajectory,
    EpisodeTrajectorySummary,
    LLMChatRequest,
    LLMResponse,
    ReplayMetadata,
    TrajectoryStep,
)


class FakeLLMClient(BaseLLMClient):
    """Small fake LLM client for frontier-analysis tests."""

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
    """Create a minimal config for frontier-analysis tests."""

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
    (prompt_dir / "frontier_analysis.txt").write_text(
        "Analysis id:\n{analysis_id}\nFrontier trajectories:\n{frontier_trajectory_block}\n"
        "Achieved values:\n{achieved_value_block}\nBottlenecks:\n{bottleneck_candidate_block}\n"
        "Grounding constraints:\n{grounding_constraints}",
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
            frontier_analysis_file="frontier_analysis.txt",
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


def _episode_trajectory(episode_id: str, reward_prefix: list[float], actions: list[str]) -> EpisodeTrajectory:
    """Build a small typed episode trajectory fixture."""

    cumulative_reward = 0.0
    steps: list[TrajectoryStep] = []
    for index, (reward, action) in enumerate(zip(reward_prefix, actions, strict=True)):
        cumulative_reward += reward
        world_state_hash = f"{episode_id}-state-{index}"
        steps.append(
            TrajectoryStep(
                episode_id=episode_id,
                step_index=index,
                action=action,
                observation=f"Observation {index} for {episode_id}.",
                reward=reward,
                cumulative_reward=cumulative_reward,
                done=index == len(actions) - 1,
                score=int(cumulative_reward),
                moves=index + 1,
                world_state_hash=world_state_hash,
                inventory_text="leaflet" if index >= 1 else "",
                valid_actions=["north", "open window", "take leaflet"],
                state_cluster_id=f"cluster-{index // 2}",
                native_snapshot_reference=f"{episode_id}:native:{index}",
            )
        )
    trajectory = EpisodeTrajectory(
        episode_id=episode_id,
        root_state_id=f"{episode_id}:root",
        steps=steps,
        max_cumulative_reward_achieved=max(step.cumulative_reward for step in steps),
        final_score=steps[-1].score,
        final_done=steps[-1].done,
        replay_metadata=ReplayMetadata(
            restore_strategy="native_snapshot",
            replay_actions=list(actions),
            world_state_hash=steps[0].world_state_hash,
            native_snapshot_reference=steps[0].native_snapshot_reference,
        ),
        summary_fields=EpisodeTrajectorySummary(
            unique_world_state_count=len(steps),
            unique_cluster_count=len({step.state_cluster_id for step in steps}),
            action_count=len(steps),
            loop_event_count=0,
            max_score=max(step.score for step in steps),
            final_cluster_id=steps[-1].state_cluster_id,
            discovered_object_tokens=["leaflet", "window"],
            bottleneck_step_indices=[len(steps) - 1],
        ),
    )
    trajectory.validate_invariants()
    return trajectory


def _archive_states(*trajectories: EpisodeTrajectory):
    """Build archive states from complete trajectories for analyzer tests."""

    updater = ArchiveUpdater()
    archive_by_state_id = updater.ingest_episode_trajectories({}, list(trajectories))
    frontier = TrajectoryFrontier(TrajectoryFrontierConfig(max_size=max(1, len(trajectories))))
    for trajectory in trajectories:
        frontier.insert(trajectory)
    archive_by_state_id = updater.refresh_achieved_values_from_frontier(archive_by_state_id, frontier)
    return updater.sorted_states(archive_by_state_id)


def test_frontier_analyzer_parses_llm_output_and_writes_artifacts(tmp_path: Path) -> None:
    """The analyzer should parse structured LLM output and persist debug artifacts."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient(
        """
        {
          "analysis_id": "frontier-analysis-test",
          "bottlenecks": ["house window remains unopened", "forest cluster repeats"],
          "partial_solutions": ["traj-001 reaches the house exterior"],
          "missing_prerequisites": ["need safe entry beyond the window"],
          "critical_states": [
            {
              "critical_state_id": "traj-001:1:traj-001-state-1",
              "achieved_value": 1,
              "potential_value": 4,
              "confidence": 0.75,
              "support_count": 2,
              "source_frontier_trajectory_ids": ["traj-001", "traj-002"],
              "supporting_state_ids": ["traj-001:1:traj-001-state-1"],
              "textual_rationale": "both trajectories stall after reaching this state."
            }
          ]
        }
        """.strip()
    )
    analyzer = FrontierAnalyzer(config, prompt_manager, llm_client)
    trajectory_frontier = TrajectoryFrontier(TrajectoryFrontierConfig(max_size=4))
    traj_one = _episode_trajectory("traj-001", [0.0, 1.0, 0.0], ["open mailbox", "take leaflet", "east"])
    traj_two = _episode_trajectory("traj-002", [0.0, 1.0, 2.0], ["open mailbox", "take leaflet", "open window"])
    trajectory_frontier.insert(traj_one)
    trajectory_frontier.insert(traj_two)

    result = analyzer.analyze_frontier(
        trajectory_frontier,
        archive_states=_archive_states(traj_one, traj_two),
        analysis_id="frontier-analysis-test",
    )

    assert result.used_fallback is False
    assert result.insight.analysis_id == "frontier-analysis-test"
    assert result.insight.inferred_bottlenecks == ["house window remains unopened", "forest cluster repeats"]
    assert result.insight.partial_solutions == ["traj-001 reaches the house exterior"]
    assert result.insight.candidate_critical_states[0].critical_state_id == "traj-001:1:traj-001-state-1"
    assert result.insight.candidate_critical_states[0].potential_value == 4.0
    assert llm_client.last_request is not None
    assert "Analysis id:" in llm_client.last_request.messages[1]["content"]
    assert llm_client.last_request.response_format is not None
    assert llm_client.last_request.response_format["type"] == "json_schema"

    artifact_dir = Path(result.artifact_directory)
    assert artifact_dir.exists()
    assert (artifact_dir / "prompt.txt").exists()
    assert (artifact_dir / "raw_completion.txt").read_text(encoding="utf-8").lstrip().startswith("{")
    assert (artifact_dir / "parsed_insight.json").exists()
    assert (artifact_dir / "result.json").exists()


def test_frontier_analyzer_falls_back_on_malformed_output(tmp_path: Path) -> None:
    """Malformed output should preserve raw text and fall back to a heuristic insight."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient("not sure maybe analyze the trajectories more")
    analyzer = FrontierAnalyzer(config, prompt_manager, llm_client)
    trajectory_frontier = TrajectoryFrontier(TrajectoryFrontierConfig(max_size=3))
    traj_one = _episode_trajectory("traj-001", [0.0, 1.0, 0.0], ["open mailbox", "take leaflet", "east"])
    traj_two = _episode_trajectory("traj-002", [0.0, 1.0, 2.0], ["open mailbox", "take leaflet", "open window"])
    trajectory_frontier.insert(traj_one)
    trajectory_frontier.insert(traj_two)

    result = analyzer.analyze_frontier(
        trajectory_frontier,
        archive_states=_archive_states(traj_one, traj_two),
        analysis_id="frontier-analysis-fallback",
    )

    assert result.used_fallback is True
    assert result.raw_completion == "not sure maybe analyze the trajectories more"
    assert "parseable insight" in result.parse_error
    assert result.insight.metadata["analysis_mode"] == "heuristic_fallback"
    assert result.insight.frontier_trajectory_ids == ["traj-002", "traj-001"]
    assert result.insight.candidate_critical_states
    assert Path(result.artifact_directory, "result.json").exists()


def test_frontier_analyzer_strips_thinking_preamble_and_code_fences(tmp_path: Path) -> None:
    """Parser hardening should recover schema after a reasoning preamble."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient(
        "Thinking Process:\n"
        "1. Analyze the request.\n"
        "2. Compare the trajectories.\n\n"
        "BEGIN_STRUCTURED_OUTPUT\n"
        "{\n"
        "  \"analysis_id\": \"frontier-analysis-hardened\",\n"
        "  \"bottlenecks\": [\"repeated stall near the window\"],\n"
        "  \"partial_solutions\": [\"traj-001 reaches the house exterior\"],\n"
        "  \"missing_prerequisites\": [\"need a safer house entry path\"],\n"
        "  \"critical_states\": [\n"
        "    {\n"
        "      \"critical_state_id\": \"traj-001:1:traj-001-state-1\",\n"
        "      \"achieved_value\": 1,\n"
        "      \"potential_value\": 3,\n"
        "      \"confidence\": 0.7,\n"
        "      \"support_count\": 1,\n"
        "      \"source_frontier_trajectory_ids\": [\"traj-001\"],\n"
        "      \"supporting_state_ids\": [\"traj-001:1:traj-001-state-1\"],\n"
        "      \"textual_rationale\": \"this state precedes the best continuation.\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "END_STRUCTURED_OUTPUT"
    )
    analyzer = FrontierAnalyzer(config, prompt_manager, llm_client)
    trajectory_frontier = TrajectoryFrontier(TrajectoryFrontierConfig(max_size=3))
    traj_one = _episode_trajectory("traj-001", [0.0, 1.0, 0.0], ["open mailbox", "take leaflet", "east"])
    trajectory_frontier.insert(traj_one)

    result = analyzer.analyze_frontier(
        trajectory_frontier,
        archive_states=_archive_states(traj_one),
        analysis_id="frontier-analysis-hardened",
    )

    assert result.used_fallback is False
    assert result.insight.analysis_id == "frontier-analysis-hardened"
    assert result.insight.inferred_bottlenecks == ["repeated stall near the window"]
    assert result.insight.candidate_critical_states[0].critical_state_id == "traj-001:1:traj-001-state-1"
