"""Tests for prompt loading and prompt substitution helpers.

TODO: extend prompt tests if templating moves beyond simple Python format strings.
"""

from __future__ import annotations

from pathlib import Path

from zork_agent.config import PromptConfig
from zork_agent.llm.prompts import PromptManager


def test_prompt_manager_loads_and_renders_named_prompt(tmp_path: Path) -> None:
    """Prompt files should load from disk and substitute named variables."""

    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "system.txt").write_text("Game: {game_id}", encoding="utf-8")
    (prompt_dir / "action.txt").write_text("Obs: {observation} | Top {action_candidates}", encoding="utf-8")
    (prompt_dir / "trajectory.txt").write_text("Episode {episode_id} seed {seed}", encoding="utf-8")
    (prompt_dir / "reflection.txt").write_text("Summary: {state_summary}", encoding="utf-8")
    (prompt_dir / "selection.txt").write_text("Frontier: {frontier_snapshot}", encoding="utf-8")
    config = PromptConfig(
        directory=prompt_dir,
        system_file="system.txt",
        action_proposal_file="action.txt",
        trajectory_analysis_file="trajectory.txt",
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


def test_prompt_manager_leaves_missing_placeholder_visible(tmp_path: Path) -> None:
    """Missing variables should remain visibly unresolved rather than crashing."""

    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "system.txt").write_text("Game: {game_id}", encoding="utf-8")
    (prompt_dir / "action.txt").write_text("Observation: {observation} / {missing_value}", encoding="utf-8")
    (prompt_dir / "trajectory.txt").write_text("Trajectory: {trajectory_excerpt}", encoding="utf-8")
    (prompt_dir / "reflection.txt").write_text("Recent: {recent_actions}", encoding="utf-8")
    (prompt_dir / "selection.txt").write_text("Frontier: {frontier_snapshot}", encoding="utf-8")
    config = PromptConfig(
        directory=prompt_dir,
        system_file="system.txt",
        action_proposal_file="action.txt",
        trajectory_analysis_file="trajectory.txt",
        local_reflection_file="reflection.txt",
        state_selection_file="selection.txt",
    )
    manager = PromptManager(config)

    rendered = manager.render("action.txt", observation="A field.")

    assert rendered == "Observation: A field. / {missing_value}"

