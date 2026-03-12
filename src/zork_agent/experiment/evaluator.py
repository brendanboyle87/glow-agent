"""Batch evaluation helpers for reproducible GLoW runs.

The evaluator keeps batch execution deliberately simple: run a list of seeds,
collect per-episode results, and persist a compact aggregate summary.

TODO: add resume/manifests only if multi-run workflows become long-lived.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

from zork_agent.config import ProjectConfig
from zork_agent.experiment.episode_runner import EpisodeRunner
from zork_agent.llm.base import BaseLLMClient
from zork_agent.types import EpisodeResult, GlowEpisodeMetrics, GlowRunMetrics
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

    def summarize_results(self, results: list[EpisodeResult]) -> GlowRunMetrics:
        """Return and persist aggregate GLoW run metrics for CLI output."""

        if not results:
            summary = GlowRunMetrics(
                run_id="empty-run",
                config_path=str(self.config.source_config) if self.config.source_config is not None else "",
                episode_count=0,
                seed_values=[],
                mean_final_score=0.0,
                std_final_score=0.0,
                mean_max_score=0.0,
                std_max_score=0.0,
                mean_environment_interactions=0.0,
                std_environment_interactions=0.0,
            )
            summary.summary_path = self._write_summary(summary)
            return summary

        episode_metrics = [self._load_episode_metrics(result) for result in results]
        final_score_values = [float(result.final_score) for result in results]
        max_score_values = [float(metrics.max_score) for metrics in episode_metrics]
        interaction_values = [float(metrics.environment_interactions) for metrics in episode_metrics]
        summary = GlowRunMetrics(
            run_id=self._default_run_id(results),
            config_path=str(self.config.source_config) if self.config.source_config is not None else "",
            episode_count=len(results),
            seed_values=[result.seed for result in results],
            mean_final_score=mean(final_score_values),
            std_final_score=self._std(final_score_values),
            mean_max_score=mean(max_score_values),
            std_max_score=self._std(max_score_values),
            mean_environment_interactions=mean(interaction_values),
            std_environment_interactions=self._std(interaction_values),
            total_restore_attempt_count=sum(metrics.restore_attempt_count for metrics in episode_metrics),
            total_restore_success_count=sum(metrics.restore_success_count for metrics in episode_metrics),
            total_frontier_analysis_count=sum(metrics.frontier_analysis_count for metrics in episode_metrics),
            total_mar_update_count=sum(metrics.mar_update_count for metrics in episode_metrics),
            episode_metric_paths=[
                str(result.metrics_path) for result in results if result.metrics_path is not None
            ],
            episode_summary_paths=[
                str(result.summary_path) for result in results if result.summary_path is not None
            ],
            trajectory_paths=[str(result.trajectory_path) for result in results],
            metadata={
                "frontier_analysis_ids": [
                    analysis_id
                    for metrics in episode_metrics
                    for analysis_id in metrics.frontier_analysis_ids
                ],
                "local_world_model_root_ids": sorted(
                    {
                        root_id
                        for metrics in episode_metrics
                        for root_id in metrics.local_world_model_root_ids
                    }
                ),
                "selection_counts_by_state": self._aggregate_selection_counts(episode_metrics),
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

    def _load_episode_metrics(self, result: EpisodeResult) -> GlowEpisodeMetrics:
        """Load one per-episode metrics artifact produced by the GLoW runner."""

        if result.metrics_path is None:
            return GlowEpisodeMetrics(
                episode_id=result.episode_id,
                seed=result.seed,
                environment_interactions=result.step_count,
                max_score=result.final_score,
                final_score=result.final_score,
            )
        payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
        return GlowEpisodeMetrics.from_record(payload)

    def _aggregate_selection_counts(
        self,
        episode_metrics: list[GlowEpisodeMetrics],
    ) -> dict[str, int]:
        """Aggregate archive-state selection counts across the run."""

        counts: dict[str, int] = {}
        for metrics in episode_metrics:
            for decision in metrics.selected_state_decisions:
                if decision.selected_state_id is None:
                    continue
                counts[decision.selected_state_id] = counts.get(decision.selected_state_id, 0) + 1
        return counts

    def _default_run_id(self, results: list[EpisodeResult]) -> str:
        """Build a deterministic run id for the current evaluation batch."""

        seed_label = f"{results[0].seed}-{results[-1].seed}" if results else "empty"
        return f"{self.config.experiment.game_id}-glow-run-seeds-{seed_label}-count-{len(results)}"

    def _write_summary(self, summary: GlowRunMetrics):
        """Persist a human-readable run metrics JSON file."""

        if summary.seed_values:
            seed_label = f"{summary.seed_values[0]}-{summary.seed_values[-1]}"
        else:
            seed_label = "empty"
        path = self.config.paths.metrics_dir / "runs" / (
            f"{self.config.experiment.game_id}-glow-run-seeds-{seed_label}-count-{summary.episode_count}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(to_jsonable(summary.to_record()), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return str(path)
