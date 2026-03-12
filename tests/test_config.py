"""Tests for GLoW config loading.

TODO: extend coverage when config validation grows beyond the current fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zork_agent.config import load_config


def test_load_config_resolves_extends_and_paths() -> None:
    """The smoke config should inherit from base and resolve relative paths."""

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "configs" / "glow_smoke.yaml")

    assert config.project_name == "zork-agent"
    assert config.llm.offline_stub is True
    assert config.runtime.jericho_game_path.is_absolute()
    assert config.paths.trajectory_dir.is_absolute()
    assert config.runtime.use_live_jericho is False


def test_core_config_overrides_base_values() -> None:
    """The core config should override selected base settings."""

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "configs" / "glow_core.yaml")

    assert config.llm.offline_stub is False
    assert config.llm.base_url == "http://host.docker.internal:1234/v1"
    assert config.experiment.runner_mode.value == "glow_faithful"
    assert config.experiment.max_steps == 100


def test_ablation_configs_toggle_expected_glow_features() -> None:
    """Named ablation configs should flip only the intended GLoW subsystem."""

    repo_root = Path(__file__).resolve().parents[1]

    no_global_analysis = load_config(repo_root / "configs" / "glow_ablation_no_global_analysis.yaml")
    assert no_global_analysis.experiment.enable_global_frontier_analysis is False
    assert no_global_analysis.experiment.enable_mar is True
    assert no_global_analysis.experiment.enable_potential_value_selection is True

    no_mar = load_config(repo_root / "configs" / "glow_ablation_no_mar.yaml")
    assert no_mar.experiment.enable_global_frontier_analysis is True
    assert no_mar.experiment.enable_mar is False
    assert no_mar.experiment.enable_potential_value_selection is True

    no_potential = load_config(repo_root / "configs" / "glow_ablation_no_potential_value_selection.yaml")
    assert no_potential.experiment.enable_global_frontier_analysis is True
    assert no_potential.experiment.enable_mar is True
    assert no_potential.experiment.enable_potential_value_selection is False


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
        "frontier_analysis.txt",
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
    (prompt_dir / "frontier_analysis.txt").write_text("frontier", encoding="utf-8")
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


def test_project_config_scoped_to_run_rewrites_artifact_paths_under_one_run_root() -> None:
    """Run-scoped configs should place all outputs under one drop-in run directory."""

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "configs" / "glow_smoke.yaml")

    scoped = config.scoped_to_run("diagnostic-run-001")

    assert scoped.paths.artifacts_dir == config.paths.artifacts_dir / "diagnostic-run-001"
    assert scoped.paths.trajectory_dir == scoped.paths.artifacts_dir / "trajectories"
    assert scoped.paths.archive_dir == scoped.paths.artifacts_dir / "archive"
    assert scoped.paths.log_dir == scoped.paths.artifacts_dir / "logs"
    assert scoped.paths.summary_dir == scoped.paths.artifacts_dir / "summaries"
    assert scoped.paths.metrics_dir == scoped.paths.artifacts_dir / "metrics"
