"""CLI entrypoints for the scaffold.

TODO: grow CLI coverage only when a workflow is exercised enough to justify it.
"""

from __future__ import annotations

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

app = typer.Typer(help="Research scaffold for Jericho + local LLM experiments.")


def main() -> None:
    """Run the Typer application."""

    app()


def validate_config_command(config_path: Path) -> ProjectConfig:
    """Load and validate a config file."""

    config = load_config(config_path)
    config.ensure_output_directories()
    return config


def run_single_command(config_path: Path, episode_id: str | None = None) -> dict[str, object]:
    """Run one placeholder episode and return a serializable summary."""

    config = validate_config_command(config_path)
    logger = configure_logging(config.run_log_path, config.logging.level)
    runner = EpisodeRunner(config=config, llm_client=_build_llm_client(config))
    result = runner.run_episode(episode_index=0, episode_id=episode_id)
    logger.info("Completed episode %s with %s steps", result.episode_id, result.step_count)
    return result.to_record()


def run_batch_command(config_path: Path, episodes: int | None = None) -> dict[str, object]:
    """Run a batch of placeholder episodes and return an aggregate summary."""

    config = validate_config_command(config_path)
    logger = configure_logging(config.run_log_path, config.logging.level)
    evaluator = Evaluator(config=config, llm_client=_build_llm_client(config))
    results = evaluator.run_batch(episode_count=episodes)
    summary = evaluator.summarize_results(results)
    logger.info("Completed batch of %s episodes", summary.episode_count)
    return summary.to_record()


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


@app.command("validate-config")
def validate_config_cli(config_path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Validate a YAML config and print the resolved result."""

    config = validate_config_command(config_path)
    typer.echo(json.dumps(config.model_dump(mode="json"), indent=2))


@app.command("run-single")
def run_single_cli(
    config_path: Path = typer.Argument(..., exists=True, readable=True),
    episode_id: str | None = typer.Option(None, help="Optional explicit episode id."),
) -> None:
    """Run one scaffolded episode."""

    typer.echo(json.dumps(run_single_command(config_path, episode_id=episode_id), indent=2))


@app.command("run-batch")
def run_batch_cli(
    config_path: Path = typer.Argument(..., exists=True, readable=True),
    episodes: int | None = typer.Option(None, min=1, help="Optional episode count override."),
) -> None:
    """Run a scaffolded episode batch."""

    typer.echo(json.dumps(run_batch_command(config_path, episodes=episodes), indent=2))


@app.command("inspect-trajectory")
def inspect_trajectory_cli(trajectory_path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Inspect a saved trajectory JSONL file."""

    typer.echo(inspect_trajectory_command(trajectory_path))
