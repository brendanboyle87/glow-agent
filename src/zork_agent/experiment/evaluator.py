"""Batch evaluation helpers for reproducible scaffold runs.

The evaluator keeps batch execution deliberately simple: run a list of seeds,
collect per-episode results, and persist a compact aggregate summary.

TODO: add resume/manifests only if multi-run workflows become long-lived.
"""

from __future__ import annotations

import json
from statistics import mean, pstdev

from zork_agent.config import ProjectConfig
from zork_agent.experiment.episode_runner import EpisodeRunner
from zork_agent.llm.base import BaseLLMClient
from zork_agent.types import BatchEvaluationSummary, EpisodeResult
from zork_agent.utils.serialization import to_jsonable


class Evaluator:
    """Run batches of episodes and summarize their outputs."""

    def __init__(self, config: ProjectConfig, llm_client: BaseLLMClient | None):
        self.config = config
        self.runner = EpisodeRunner(config=config, llm_client=llm_client)

    def run_batch(
        self,
        episode_count: int | None = None,
        *,
        seed_values: list[int] | None = None,
    ) -> list[EpisodeResult]:
        """Run multiple episodes, defaulting to a deterministic multi-seed sweep."""

        seeds = seed_values or self._default_seed_values(episode_count)
        return [
            self.runner.run_episode(episode_index=index, episode_seed=seed)
            for index, seed in enumerate(seeds)
        ]

    def summarize_results(self, results: list[EpisodeResult]) -> BatchEvaluationSummary:
        """Return and persist a compact aggregate summary for CLI output."""

        if not results:
            summary = BatchEvaluationSummary(
                episode_count=0,
                seed_values=[],
                mean_reward=0.0,
                std_reward=0.0,
                mean_steps=0.0,
                std_steps=0.0,
                mean_final_score=0.0,
                std_final_score=0.0,
                trajectory_paths=[],
            )
            summary.summary_path = self._write_summary(summary)
            return summary

        reward_values = [result.total_reward for result in results]
        step_values = [float(result.step_count) for result in results]
        final_score_values = [float(result.final_score) for result in results]
        summary = BatchEvaluationSummary(
            episode_count=len(results),
            seed_values=[result.seed for result in results],
            mean_reward=mean(reward_values),
            std_reward=self._std(reward_values),
            mean_steps=mean(step_values),
            std_steps=self._std(step_values),
            mean_final_score=mean(final_score_values),
            std_final_score=self._std(final_score_values),
            trajectory_paths=[result.trajectory_path for result in results],
            metadata={
                "episode_summary_paths": [
                    str(result.summary_path) for result in results if result.summary_path is not None
                ],
                "final_guidance": [result.notes for result in results],
            },
        )
        summary.summary_path = self._write_summary(summary)
        return summary

    def _default_seed_values(self, episode_count: int | None) -> list[int]:
        """Build the default deterministic seed list for the batch."""

        count = episode_count or self.config.experiment.batch_size
        return [self.config.experiment.seed + index for index in range(count)]

    def _std(self, values: list[float]) -> float:
        """Return a population standard deviation with a stable single-item fallback."""

        if len(values) <= 1:
            return 0.0
        return pstdev(values)

    def _write_summary(self, summary: BatchEvaluationSummary):
        """Persist a human-readable batch summary JSON file."""

        if summary.seed_values:
            seed_label = f"{summary.seed_values[0]}-{summary.seed_values[-1]}"
        else:
            seed_label = "empty"
        path = (
            self.config.paths.summary_dir
            / f"{self.config.experiment.game_id}-batch-seeds-{seed_label}-count-{summary.episode_count}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(to_jsonable(summary.to_record()), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path
