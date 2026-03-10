"""Tests for heuristic and LLM-assisted frontier state selection.

TODO: add evaluation-facing tests once batch experiments compare selector modes directly.
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
from zork_agent.memory.frontier import FrontierEntry, FrontierQueue
from zork_agent.policy.state_selector import StateSelector
from zork_agent.types import LLMChatRequest, LLMResponse, StateSelectionMode


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
    (prompt_dir / "reflection.txt").write_text("Reflection prompt", encoding="utf-8")
    (prompt_dir / "selection.txt").write_text(
        "Frontier candidates:\n{frontier_snapshot}\nPick one.",
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
            local_reflection_file="reflection.txt",
            state_selection_file="selection.txt",
        ),
        policy=PolicyConfig(frontier_max_size=8),
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
    llm_client = FakeLLMClient("choice: C2\nreason: higher novelty with fresh score gain.")
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
    assert "Frontier candidates:" in llm_client.last_request.messages[1]["content"]


def test_state_selector_falls_back_when_llm_output_is_malformed(tmp_path: Path) -> None:
    """Malformed LLM output should fall back to deterministic heuristic selection."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient("the second branch seems interesting")
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
    assert result.raw_output == "the second branch seems interesting"
