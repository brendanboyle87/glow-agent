"""CLI entrypoints for running the GLoW implementation and ablations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import typer

from zork_agent.config import ProjectConfig, load_config
from zork_agent.env.replay import ReplaySession
from zork_agent.experiment.episode_runner import EpisodeRunner
from zork_agent.experiment.evaluator import Evaluator
from zork_agent.llm.base import BaseLLMClient
from zork_agent.llm.lmstudio_client import LMStudioClient
from zork_agent.utils.logging import configure_logging

app = typer.Typer(help="GLoW-style Jericho experiments with trajectory frontier, archive selection, and MAR.")


def main() -> None:
    """Run the Typer application."""

    app()


def validate_config_command(
    config_path: Path,
    *,
    ensure_output_directories: bool = False,
) -> ProjectConfig:
    """Load and validate a config file."""

    config = load_config(config_path)
    if ensure_output_directories:
        config.ensure_output_directories()
    return config


def run_single_command(config_path: Path, episode_id: str | None = None) -> dict[str, object]:
    """Run one GLoW episode and return a serializable summary."""

    base_config = validate_config_command(config_path)
    resolved_episode_id = episode_id or _default_episode_id(base_config, episode_index=0)
    config = _prepare_run_config(base_config, run_id=resolved_episode_id)
    logger = configure_logging(config.run_log_path, config.logging.level)
    runner = EpisodeRunner(config=config, llm_client=_build_llm_client(config))
    result = runner.run_episode(episode_index=0, episode_id=resolved_episode_id)
    logger.info("Completed episode %s with %s steps", result.episode_id, result.step_count)
    payload = result.to_record()
    _write_run_manifest(
        config,
        run_id=resolved_episode_id,
        run_mode="single",
        result_payload=payload,
    )
    return payload


def run_batch_command(config_path: Path, episodes: int | None = None) -> dict[str, object]:
    """Run a GLoW evaluation batch and return aggregate run metrics."""

    base_config = validate_config_command(config_path)
    seed_values = _default_batch_seed_values(base_config, episodes)
    run_id = _default_batch_run_id(base_config, seed_values)
    config = _prepare_run_config(base_config, run_id=run_id)
    logger = configure_logging(config.run_log_path, config.logging.level)
    evaluator = Evaluator(config=config, llm_client=_build_llm_client(config))
    results = evaluator.run_batch(seed_values=seed_values)
    summary = evaluator.summarize_results(results)
    logger.info("Completed batch of %s episodes", summary.episode_count)
    payload = summary.to_record()
    _write_run_manifest(
        config,
        run_id=run_id,
        run_mode="batch",
        result_payload=payload,
    )
    return payload


def inspect_trajectory_command(trajectory_path: Path) -> str:
    """Load a JSONL trajectory and render a brief human-readable summary."""

    replay = ReplaySession.from_path(trajectory_path)
    return replay.format_summary()


def _build_llm_client(config: ProjectConfig) -> BaseLLMClient | None:
    """Construct the configured LLM client if live calls are enabled."""

    # TODO: add backend registry support once a second provider lands.
    if config.llm.offline_stub:
        return None
    if config.llm.provider != "lmstudio":
        raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")
    return LMStudioClient(
        base_url=config.llm.base_url,
        default_model=config.llm.model_name,
        api_key=config.llm.api_key,
        timeout_seconds=config.llm.request_timeout_seconds,
    )


def _default_episode_id(config: ProjectConfig, *, episode_index: int) -> str:
    """Return the default episode id used by the runner for one single episode."""

    return f"{config.experiment.game_id}-episode-{episode_index:03d}"


def _default_batch_seed_values(config: ProjectConfig, episodes: int | None) -> list[int]:
    """Return the deterministic seed list for one batch CLI run."""

    count = episodes or config.experiment.batch_size
    return [config.experiment.seed + index for index in range(count)]


def _default_batch_run_id(config: ProjectConfig, seed_values: list[int]) -> str:
    """Build the deterministic run id used for one batch CLI run."""

    if not seed_values:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{config.experiment.game_id}-glow-run-empty-{timestamp}"
    return (
        f"{config.experiment.game_id}-glow-run-seeds-"
        f"{seed_values[0]}-{seed_values[-1]}-count-{len(seed_values)}"
    )


def _prepare_run_config(base_config: ProjectConfig, *, run_id: str) -> ProjectConfig:
    """Return a run-scoped config and persist the exact config files into the run root."""

    config = base_config.scoped_to_run(run_id)
    config.ensure_output_directories()
    _write_run_scaffold(config)
    return config


def _write_run_scaffold(config: ProjectConfig) -> None:
    """Write the source and resolved config into the run root for external tooling."""

    run_root = config.paths.artifacts_dir
    if config.source_config is not None and config.source_config.exists():
        (run_root / "source_config.yaml").write_text(
            config.source_config.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (run_root / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def _write_run_manifest(
    config: ProjectConfig,
    *,
    run_id: str,
    run_mode: str,
    result_payload: dict[str, object],
) -> None:
    """Write one manifest that points to all key files using run-root-relative paths."""

    run_root = config.paths.artifacts_dir
    manifest = {
        "run_id": run_id,
        "run_mode": run_mode,
        "source_config": "source_config.yaml" if (run_root / "source_config.yaml").exists() else "",
        "resolved_config": "resolved_config.json",
        "directories": {
            "logs": _relative_to_run_root(config.paths.log_dir, run_root),
            "metrics": _relative_to_run_root(config.paths.metrics_dir, run_root),
            "summaries": _relative_to_run_root(config.paths.summary_dir, run_root),
            "trajectories": _relative_to_run_root(config.paths.trajectory_dir, run_root),
            "archive": _relative_to_run_root(config.paths.archive_dir, run_root),
        },
        "primary_files": {
            "log": _relative_to_run_root(config.run_log_path, run_root),
            "trajectory": _relative_payload_path(result_payload.get("trajectory_path"), run_root),
            "summary": _relative_payload_path(result_payload.get("summary_path"), run_root),
            "metrics": _relative_payload_path(result_payload.get("metrics_path"), run_root),
        },
        "result": _relativize_payload_paths(result_payload, run_root),
    }
    (run_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _relative_payload_path(value: object, run_root: Path) -> str:
    """Return a run-root-relative artifact path from a serialized payload field."""

    if not isinstance(value, str) or not value.strip():
        return ""
    return _relative_to_run_root(Path(value), run_root)


def _relativize_payload_paths(payload: dict[str, object], run_root: Path) -> dict[str, object]:
    """Convert nested artifact paths in one CLI payload to run-root-relative strings."""

    converted = _relativize_payload_value(payload, run_root)
    assert isinstance(converted, dict)
    return converted


def _relativize_payload_value(value: object, run_root: Path) -> object:
    """Recursively convert any run-root-contained path strings to relative strings."""

    if isinstance(value, dict):
        return {key: _relativize_payload_value(item, run_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_relativize_payload_value(item, run_root) for item in value]
    if isinstance(value, str) and value.strip():
        return _relative_to_run_root(Path(value), run_root)
    return value


def _relative_to_run_root(path: Path, run_root: Path) -> str:
    """Return a relative path when the artifact lives under the run root."""

    try:
        return str(path.relative_to(run_root))
    except ValueError:
        return str(path)


@app.command("validate-config")
def validate_config_cli(config_path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Validate a YAML config and print the resolved result."""

    config = validate_config_command(config_path, ensure_output_directories=True)
    typer.echo(json.dumps(config.model_dump(mode="json"), indent=2))


@app.command("run-single")
def run_single_cli(
    config_path: Path = typer.Argument(..., exists=True, readable=True),
    episode_id: str | None = typer.Option(None, help="Optional explicit episode id."),
) -> None:
    """Run one GLoW episode."""

    typer.echo(json.dumps(run_single_command(config_path, episode_id=episode_id), indent=2))


@app.command("run-batch")
def run_batch_cli(
    config_path: Path = typer.Argument(..., exists=True, readable=True),
    episodes: int | None = typer.Option(None, min=1, help="Optional episode count override."),
) -> None:
    """Run a GLoW evaluation batch."""

    typer.echo(json.dumps(run_batch_command(config_path, episodes=episodes), indent=2))


@app.command("inspect-trajectory")
def inspect_trajectory_cli(trajectory_path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Inspect a saved trajectory JSONL file."""

    typer.echo(inspect_trajectory_command(trajectory_path))
