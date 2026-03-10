"""Tests for scaffold config loading.

TODO: extend coverage when config validation grows beyond the current fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zork_agent.config import load_config


def test_load_config_resolves_extends_and_paths() -> None:
    """The debug config should inherit from base and resolve relative paths."""

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "configs" / "zork1_debug.yaml")

    assert config.project_name == "zork-agent"
    assert config.llm.offline_stub is True
    assert config.runtime.jericho_game_path.is_absolute()
    assert config.paths.trajectory_dir.is_absolute()
    assert config.runtime.use_live_jericho is False


def test_local_config_overrides_base_values() -> None:
    """The local config should override selected base settings."""

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "configs" / "zork1_local.yaml")

    assert config.llm.offline_stub is False
    assert config.llm.base_url == "http://host.docker.internal:1234/v1"
    assert config.policy.frontier_max_size == 48
    assert config.experiment.max_steps == 25


def test_load_config_rejects_invalid_budget_values(tmp_path: Path) -> None:
    """Config validation should fail early for invalid numeric budgets."""

    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
project_name: zork-agent
runtime:
  jericho_game_path: games/zork1.z5
llm:
  provider: lmstudio
  base_url: http://127.0.0.1:1234
  api_key: lm-studio
  model_name: local-model
prompts:
  directory: prompts
policy:
  rollout_count: 0
paths:
  artifacts_dir: artifacts
  trajectory_dir: artifacts/trajectories
  archive_dir: artifacts/archive
  log_dir: artifacts/logs
  summary_dir: artifacts/summaries
logging:
  level: INFO
experiment:
  max_steps: 2
  branch_commit_steps: 1
""".strip(),
        encoding="utf-8",
    )
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    for name in (
        "system.txt",
        "action_proposal.txt",
        "trajectory_analysis.txt",
        "local_reflection.txt",
        "state_selection.txt",
    ):
        (prompt_dir / name).write_text(name, encoding="utf-8")

    with pytest.raises(ValueError, match="policy.rollout_count"):
        load_config(config_path)


def test_load_config_rejects_missing_prompt_files(tmp_path: Path) -> None:
    """Config loading should fail clearly when prompt files are missing."""

    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "system.txt").write_text("system", encoding="utf-8")
    (prompt_dir / "action_proposal.txt").write_text("action", encoding="utf-8")
    (prompt_dir / "trajectory_analysis.txt").write_text("trajectory", encoding="utf-8")
    (prompt_dir / "local_reflection.txt").write_text("reflection", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project_name: zork-agent
runtime:
  jericho_game_path: games/zork1.z5
llm:
  provider: lmstudio
  base_url: http://127.0.0.1:1234/v1
  api_key: lm-studio
  model_name: local-model
prompts:
  directory: prompts
policy:
  rollout_count: 1
  rollout_depth: 1
  frontier_max_size: 1
  archive_top_k: 1
  action_candidates: 1
paths:
  artifacts_dir: artifacts
  trajectory_dir: artifacts/trajectories
  archive_dir: artifacts/archive
  log_dir: artifacts/logs
  summary_dir: artifacts/summaries
logging:
  level: INFO
experiment:
  max_steps: 2
  branch_commit_steps: 1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Configured prompt files are missing"):
        load_config(config_path)
