"""Tests for prompt loading and prompt substitution helpers.

TODO: extend prompt tests if templating moves beyond simple Python format strings.
"""

from __future__ import annotations

from pathlib import Path

from zork_agent.config import PromptConfig
from zork_agent.llm.prompts import PromptManager
from zork_agent.types import (
    ActionBiasRecord,
    AdvantageHint,
    AffordanceRecord,
    LocalWorldModel,
    SubgoalRecord,
)


def test_prompt_manager_loads_and_renders_named_prompt(tmp_path: Path) -> None:
    """Prompt files should load from disk and substitute named variables."""

    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "system.txt").write_text("Game: {game_id}", encoding="utf-8")
    (prompt_dir / "action.txt").write_text("Obs: {observation} | Top {action_candidates}", encoding="utf-8")
    (prompt_dir / "trajectory.txt").write_text("Episode {episode_id} seed {seed}", encoding="utf-8")
    (prompt_dir / "frontier.txt").write_text("Frontier {frontier_trajectory_block} / {achieved_value_block}", encoding="utf-8")
    (prompt_dir / "reflection.txt").write_text("Summary: {state_summary}", encoding="utf-8")
    (prompt_dir / "selection.txt").write_text("Frontier: {frontier_snapshot}", encoding="utf-8")
    config = PromptConfig(
        directory=prompt_dir,
        system_file="system.txt",
        action_proposal_file="action.txt",
        trajectory_analysis_file="trajectory.txt",
        frontier_analysis_file="frontier.txt",
        local_reflection_file="reflection.txt",
        state_selection_file="selection.txt",
    )
    manager = PromptManager(config)

    assert manager.load_named_prompt("system.txt") == "Game: {game_id}"
    assert manager.render_system(game_id="zork1") == "Game: zork1"
    assert (
        manager.render_action_proposal(
            observation="A field.",
            inventory="Empty.",
            score=0,
            moves=0,
            action_candidates=3,
        )
        == "Obs: A field. | Top 3"
    )
    assert (
        manager.render_frontier_analysis(
            frontier_trajectory_block="T1",
            achieved_value_block="S1",
            bottleneck_candidate_block="B1",
        )
        == "Frontier T1 / S1"
    )


def test_prompt_manager_leaves_missing_placeholder_visible(tmp_path: Path) -> None:
    """Missing variables should remain visibly unresolved rather than crashing."""

    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "system.txt").write_text("Game: {game_id}", encoding="utf-8")
    (prompt_dir / "action.txt").write_text("Observation: {observation} / {missing_value}", encoding="utf-8")
    (prompt_dir / "trajectory.txt").write_text("Trajectory: {trajectory_excerpt}", encoding="utf-8")
    (prompt_dir / "frontier.txt").write_text("Frontier: {frontier_trajectory_block}", encoding="utf-8")
    (prompt_dir / "reflection.txt").write_text("Recent: {recent_actions}", encoding="utf-8")
    (prompt_dir / "selection.txt").write_text("Frontier: {frontier_snapshot}", encoding="utf-8")
    config = PromptConfig(
        directory=prompt_dir,
        system_file="system.txt",
        action_proposal_file="action.txt",
        trajectory_analysis_file="trajectory.txt",
        frontier_analysis_file="frontier.txt",
        local_reflection_file="reflection.txt",
        state_selection_file="selection.txt",
    )
    manager = PromptManager(config)

    rendered = manager.render("action.txt", observation="A field.")

    assert rendered == "Observation: A field. / {missing_value}"


def test_prompt_manager_summarizes_local_world_model_with_bounded_context(tmp_path: Path) -> None:
    """Local-world-model prompt summaries should stay compact and bounded."""

    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "system.txt").write_text("Game: {game_id}", encoding="utf-8")
    (prompt_dir / "action.txt").write_text("Action prompt", encoding="utf-8")
    (prompt_dir / "trajectory.txt").write_text("Trajectory: {trajectory_excerpt}", encoding="utf-8")
    (prompt_dir / "frontier.txt").write_text("Frontier: {frontier_trajectory_block}", encoding="utf-8")
    (prompt_dir / "reflection.txt").write_text("Recent: {recent_actions}", encoding="utf-8")
    (prompt_dir / "selection.txt").write_text("Frontier: {frontier_snapshot}", encoding="utf-8")
    config = PromptConfig(
        directory=prompt_dir,
        system_file="system.txt",
        action_proposal_file="action.txt",
        trajectory_analysis_file="trajectory.txt",
        frontier_analysis_file="frontier.txt",
        local_reflection_file="reflection.txt",
        state_selection_file="selection.txt",
    )
    manager = PromptManager(config)
    local_world_model = LocalWorldModel(
        root_state_id="root-window",
        accumulated_advantage_hints=[
            AdvantageHint(root_state_id="root-window", action_preferences=["look in window"], textual_reasoning="older"),
            AdvantageHint(root_state_id="root-window", action_preferences=["open window"], textual_reasoning="newer"),
            AdvantageHint(root_state_id="root-window", action_preferences=["enter window"], textual_reasoning="newest"),
        ],
        discovered_subgoals=[
            SubgoalRecord(subgoal_id="s1", description="Get inside the house."),
            SubgoalRecord(subgoal_id="s2", description="Open the window safely."),
            SubgoalRecord(subgoal_id="s3", description="Check the kitchen."),
        ],
        inferred_affordances=[
            AffordanceRecord(object_text="window", affordance="open"),
            AffordanceRecord(object_text="window", affordance="enter"),
            AffordanceRecord(object_text="house", affordance="enter"),
        ],
        action_priors=[
            ActionBiasRecord(action="open window", weight=1.0),
            ActionBiasRecord(action="enter window", weight=0.9),
            ActionBiasRecord(action="look in window", weight=0.8),
        ],
        action_antipriors=[
            ActionBiasRecord(action="west", weight=-0.5),
            ActionBiasRecord(action="south", weight=-0.4),
            ActionBiasRecord(action="wait", weight=-0.3),
        ],
    )

    summary = manager.summarize_local_world_model(
        local_world_model,
        root_state_id="root-window",
        max_advantage_hints=2,
        max_subgoals=2,
        max_affordances=2,
        max_action_biases=2,
        max_summary_chars=120,
    )

    assert summary.root_state_id == "root-window"
    assert len(summary.recent_advantage_hints) == 2
    assert len(summary.discovered_subgoals) == 2
    assert len(summary.inferred_affordances) == 2
    assert len(summary.action_priors) == 2
    assert len(summary.action_antipriors) == 2
    assert "older" not in " ".join(summary.recent_advantage_hints)
    assert len(summary.summary_text) <= 120
