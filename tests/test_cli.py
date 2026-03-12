"""Tests for CLI run layout and run-manifest writing."""

from __future__ import annotations

import json
from pathlib import Path

from zork_agent import cli
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
from zork_agent.types import EpisodeResult


def _build_config(tmp_path: Path) -> ProjectConfig:
    """Create a small config fixture for CLI path-layout tests."""

    rom_path = tmp_path / "games" / "zork1.z5"
    rom_path.parent.mkdir(parents=True, exist_ok=True)
    rom_path.write_text("fake rom", encoding="utf-8")
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "system.txt",
        "action_proposal.txt",
        "trajectory_analysis.txt",
        "frontier_analysis.txt",
        "local_reflection.txt",
        "state_selection.txt",
    ):
        (prompt_dir / name).write_text(name, encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    config = ProjectConfig(
        runtime=RuntimeConfig(jericho_game_path=rom_path, use_live_jericho=False),
        llm=LLMConfig(offline_stub=True),
        prompts=PromptConfig(directory=prompt_dir),
        policy=PolicyConfig(),
        paths=PathsConfig(
            artifacts_dir=artifacts_dir,
            trajectory_dir=artifacts_dir / "trajectories",
            archive_dir=artifacts_dir / "archive",
            log_dir=artifacts_dir / "logs",
            summary_dir=artifacts_dir / "summaries",
        ),
        logging=LoggingConfig(run_log_filename="run.log"),
        experiment=ExperimentConfig(game_id="zork1", max_steps=2, batch_size=1, branch_commit_steps=1),
        source_config=tmp_path / "config.yaml",
    )
    config.source_config.write_text("project_name: zork-agent\n", encoding="utf-8")
    return config


def test_run_single_command_scopes_outputs_under_one_run_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """`run-single` should place logs, metrics, summaries, and trajectories under one run root."""

    base_config = _build_config(tmp_path)

    class _FakeRunner:
        def __init__(self, config: ProjectConfig, llm_client):
            self.config = config

        def run_episode(self, episode_index: int = 0, episode_id: str | None = None, episode_seed: int | None = None):
            assert episode_id == "per-run-test"
            trajectory_path = self.config.paths.trajectory_dir / f"{episode_id}.jsonl"
            summary_path = self.config.paths.summary_dir / f"{episode_id}.summary.json"
            metrics_path = self.config.paths.metrics_dir / "episodes" / f"{episode_id}.metrics.json"
            archive_path = self.config.paths.archive_dir / f"{episode_id}.archive.jsonl"
            selection_dir = self.config.paths.summary_dir / "state_selection" / "selection-001"
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            selection_dir.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text("{}\n", encoding="utf-8")
            summary_path.write_text("{}", encoding="utf-8")
            metrics_path.write_text("{}", encoding="utf-8")
            archive_path.write_text("{}\n", encoding="utf-8")
            return EpisodeResult(
                episode_id=episode_id,
                seed=13,
                total_reward=0.0,
                step_count=2,
                final_score=0,
                trajectory_path=trajectory_path,
                summary_path=summary_path,
                metrics_path=metrics_path,
                metadata={
                    "archive_snapshot_path": str(archive_path),
                    "selection_artifact_directories": [str(selection_dir)],
                },
            )

    monkeypatch.setattr(cli, "validate_config_command", lambda *args, **kwargs: base_config)
    monkeypatch.setattr(cli, "EpisodeRunner", _FakeRunner)
    monkeypatch.setattr(cli, "_build_llm_client", lambda config: None)

    payload = cli.run_single_command(tmp_path / "config.yaml", episode_id="per-run-test")

    run_root = base_config.paths.artifacts_dir / "per-run-test"
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))

    assert payload["trajectory_path"].endswith("per-run-test/trajectories/per-run-test.jsonl")
    assert (run_root / "source_config.yaml").exists()
    assert (run_root / "resolved_config.json").exists()
    assert (run_root / "logs" / "run.log").exists()
    assert manifest["run_id"] == "per-run-test"
    assert manifest["directories"]["metrics"] == "metrics"
    assert manifest["primary_files"]["trajectory"] == "trajectories/per-run-test.jsonl"
    assert manifest["primary_files"]["summary"] == "summaries/per-run-test.summary.json"
    assert manifest["primary_files"]["metrics"] == "metrics/episodes/per-run-test.metrics.json"
    assert manifest["result"]["metadata"]["archive_snapshot_path"] == "archive/per-run-test.archive.jsonl"
    assert manifest["result"]["metadata"]["selection_artifact_directories"] == [
        "summaries/state_selection/selection-001"
    ]
