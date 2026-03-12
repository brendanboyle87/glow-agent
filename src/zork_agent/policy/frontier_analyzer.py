"""Global frontier-analysis stage for the paper-faithful world model.

This module sits on top of the complete-trajectory frontier. The primary path is
LLM-mediated analysis over top frontier trajectories. A deterministic fallback
remains available so the main episode loop can continue even when parsing fails
or no model is configured.

TODO: let downstream state selection consume cached analysis results directly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import logging
import re

from zork_agent.config import ProjectConfig
from zork_agent.llm.base import BaseLLMClient
from zork_agent.llm.prompts import PromptManager
from zork_agent.memory.archive_updater import archive_state_identity_for_step
from zork_agent.memory.frontier import TrajectoryFrontier
from zork_agent.memory.frontier_analysis_store import FrontierAnalysisStore
from zork_agent.memory.summaries import SummaryBuilder
from zork_agent.types import (
    ArchivedState,
    CriticalStateAnnotation,
    EpisodeTrajectory,
    FrontierAnalysisResult,
    FrontierInsight,
    FrontierTrajectoryEntry,
)

_LOGGER = logging.getLogger("zork_agent.frontier_analyzer")


class FrontierAnalyzer:
    """Analyze the complete-trajectory frontier into typed global insights."""

    def __init__(
        self,
        config: ProjectConfig,
        prompt_manager: PromptManager,
        llm_client: BaseLLMClient | None,
        analysis_store: FrontierAnalysisStore | None = None,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.prompt_manager = prompt_manager
        self.llm_client = llm_client
        self.summary_builder = SummaryBuilder()
        self.analysis_store = analysis_store or FrontierAnalysisStore(config.paths.summary_dir / "frontier_analysis")
        self.logger = logger or _LOGGER

    def analyze_frontier(
        self,
        trajectory_frontier: TrajectoryFrontier,
        *,
        archive_states: Sequence[ArchivedState] | None = None,
        top_k: int | None = None,
        analysis_id: str | None = None,
    ) -> FrontierAnalysisResult:
        """Analyze the retained frontier trajectories and persist a debug bundle."""

        effective_top_k = max(1, top_k or trajectory_frontier.config.max_size)
        entries = trajectory_frontier.top_k_entries(effective_top_k)
        trajectories = trajectory_frontier.top_k_trajectories(effective_top_k)
        available_archive_states = list(archive_states or [])
        resolved_analysis_id = analysis_id or self._default_analysis_id(entries)
        prompt_input = self._build_prompt_input(
            analysis_id=resolved_analysis_id,
            entries=entries,
            trajectories=trajectories,
            archived_states=available_archive_states,
        )

        raw_completion = ""
        parse_error = ""
        used_fallback = False
        if not entries or not trajectories:
            self.logger.warning(
                "Frontier analysis fallback engaged for %s: no retained trajectories were available.",
                resolved_analysis_id,
            )
            used_fallback = True
            parse_error = "No retained frontier trajectories were available."
            insight = self._heuristic_fallback_insight(
                analysis_id=resolved_analysis_id,
                entries=entries,
                trajectories=trajectories,
                archived_states=available_archive_states,
                fallback_reason=parse_error,
            )
        elif self.llm_client is None:
            self.logger.warning(
                "Frontier analysis fallback engaged for %s: no LLM client is configured.",
                resolved_analysis_id,
            )
            used_fallback = True
            parse_error = "No LLM client is configured for frontier analysis."
            insight = self._heuristic_fallback_insight(
                analysis_id=resolved_analysis_id,
                entries=entries,
                trajectories=trajectories,
                archived_states=available_archive_states,
                fallback_reason=parse_error,
            )
        else:
            try:
                response = self.llm_client.analyze_frontier(
                    prompt_manager=self.prompt_manager,
                    game_id=self.config.experiment.game_id,
                    analysis_id=resolved_analysis_id,
                    frontier_trajectory_block=self._build_frontier_trajectory_block(entries, trajectories),
                    achieved_value_block=self._build_achieved_value_block(available_archive_states),
                    bottleneck_candidate_block=self._build_bottleneck_candidate_block(entries, trajectories),
                    model=self.config.llm.model_name,
                    temperature=0.0,
                    max_tokens=(
                        self.config.llm.frontier_analysis_debug_max_tokens
                        if self.config.llm.analysis_debug_mode
                        else self.config.llm.frontier_analysis_max_tokens
                    ),
                    timeout_seconds=self.config.llm.request_timeout_seconds,
                    analysis_debug_mode=self.config.llm.analysis_debug_mode,
                    response_format=(
                        None
                        if self.config.llm.analysis_debug_mode
                        else _frontier_analysis_response_format()
                    ),
                )
                raw_completion = response.text
                insight = self._parse_frontier_output(
                    raw_output=raw_completion,
                    analysis_id=resolved_analysis_id,
                    considered_frontier_ids=[entry.trajectory_id for entry in entries],
                    known_state_ids={state.state_id for state in available_archive_states},
                )
                insight.metadata.setdefault("analysis_mode", "llm_frontier_analysis")
                insight.metadata.setdefault("llm_model", response.model)
            except Exception as exc:
                parse_error = str(exc)
                used_fallback = True
                self.logger.warning(
                    "Frontier analysis fallback engaged for %s after parse/request failure: %s",
                    resolved_analysis_id,
                    exc,
                )
                insight = self._heuristic_fallback_insight(
                    analysis_id=resolved_analysis_id,
                    entries=entries,
                    trajectories=trajectories,
                    archived_states=available_archive_states,
                    fallback_reason=parse_error,
                )

        result = FrontierAnalysisResult(
            analysis_id=resolved_analysis_id,
            insight=insight,
            prompt_input=prompt_input,
            raw_completion=raw_completion,
            parse_error=parse_error,
            used_fallback=used_fallback,
            metadata={
                "trajectory_count_considered": len(trajectories),
                "frontier_entry_count_considered": len(entries),
                "archived_state_count_considered": len(available_archive_states),
                "paper_faithful_intent": "llm_frontier_analysis_over_complete_trajectories",
                "fallback_policy": "deterministic_frontier_summary",
            },
        )
        self.analysis_store.write(result)
        return result

    def _build_prompt_input(
        self,
        *,
        analysis_id: str,
        entries: Sequence[FrontierTrajectoryEntry],
        trajectories: Sequence[EpisodeTrajectory],
        archived_states: Sequence[ArchivedState],
    ) -> str:
        """Render the exact frontier-analysis prompt fed into the model."""

        return self.prompt_manager.render_frontier_analysis(
            analysis_id=analysis_id,
            frontier_trajectory_block=self._build_frontier_trajectory_block(entries, trajectories),
            achieved_value_block=self._build_achieved_value_block(archived_states),
            bottleneck_candidate_block=self._build_bottleneck_candidate_block(entries, trajectories),
            grounding_constraints=(
                "Only cite states, bottlenecks, or partial solutions grounded in the supplied frontier "
                "trajectories or achieved-value state list. Potential value must be tied to later "
                "higher-value continuations or repeated stalls near the same state."
            ),
            analysis_debug_mode=self.config.llm.analysis_debug_mode,
        )

    def _build_frontier_trajectory_block(
        self,
        entries: Sequence[FrontierTrajectoryEntry],
        trajectories: Sequence[EpisodeTrajectory],
    ) -> str:
        """Build the compact frontier-trajectory block for analysis."""

        lines: list[str] = []
        entry_by_id = {entry.trajectory_id: entry for entry in entries}
        for trajectory in trajectories:
            entry = entry_by_id.get(trajectory.trajectory_id)
            if entry is None:
                continue
            final_step = trajectory.steps[-1] if trajectory.steps else None
            action_preview = " -> ".join(step.action for step in trajectory.steps[:6]) or "none"
            bottleneck_steps = ", ".join(str(index) for index in entry.bottleneck_step_indices[:5]) or "none"
            bottleneck_ids = ", ".join(entry.bottleneck_state_ids[:4]) or "none"
            objects = ", ".join(trajectory.summary_fields.discovered_object_tokens[:6]) or "none"
            lines.append(
                f"trajectory_id={trajectory.trajectory_id} value={entry.value:.2f} "
                f"max_reward={trajectory.max_cumulative_reward_achieved:.2f} final_score={trajectory.final_score} "
                f"final_done={trajectory.final_done} root_state_id={trajectory.root_state_id} "
                f"selected_from_archive={trajectory.selected_from_archive_state_id or 'none'} "
                f"final_cluster={trajectory.summary_fields.final_cluster_id or 'unknown'} "
                f"unique_clusters={trajectory.summary_fields.unique_cluster_count} "
                f"loops={trajectory.summary_fields.loop_event_count} discovered_objects={objects}"
            )
            if final_step is not None:
                lines.append(
                    f"  final_step={final_step.step_index} final_state_id={self._state_id_for_step(trajectory, final_step)} "
                    f"cum_reward={final_step.cumulative_reward:.2f} action={final_step.action} "
                    f"observation={_truncate(final_step.observation, 160)}"
                )
            lines.append(
                f"  bottleneck_steps={bottleneck_steps} bottleneck_state_ids={bottleneck_ids} "
                f"action_prefix={action_preview}"
            )
        return "\n".join(lines) or "(none)"

    def _build_achieved_value_block(self, archived_states: Sequence[ArchivedState]) -> str:
        """Build the achieved-value state block from the independent archive."""

        lines: list[str] = []
        for archived_state in list(archived_states)[:12]:
            lines.append(
                f"state_id={archived_state.state_id} achieved_value={archived_state.achieved_value:.2f} "
                f"trajectory_id={archived_state.provenance_trajectory_id} step={archived_state.provenance_timestep} "
                f"restore_strategy={archived_state.replay_metadata.restore_strategy} "
                f"world_state_hash={archived_state.replay_metadata.world_state_hash or 'unknown'}"
            )
        return "\n".join(lines) or "(none)"

    def _build_bottleneck_candidate_block(
        self,
        entries: Sequence[FrontierTrajectoryEntry],
        trajectories: Sequence[EpisodeTrajectory],
    ) -> str:
        """Build the bottleneck-candidate block from frontier metadata."""

        lines: list[str] = []
        trajectory_by_id = {trajectory.trajectory_id: trajectory for trajectory in trajectories}
        for entry in entries:
            trajectory = trajectory_by_id.get(entry.trajectory_id)
            final_cluster = trajectory.summary_fields.final_cluster_id if trajectory is not None else "unknown"
            lines.append(
                f"trajectory_id={entry.trajectory_id} bottleneck_state_ids={'; '.join(entry.bottleneck_state_ids[:4]) or 'none'} "
                f"bottleneck_step_indices={'; '.join(str(item) for item in entry.bottleneck_step_indices[:5]) or 'none'} "
                f"final_cluster={final_cluster or 'unknown'}"
            )
        return "\n".join(lines) or "(none)"

    def _parse_frontier_output(
        self,
        *,
        raw_output: str,
        analysis_id: str,
        considered_frontier_ids: Sequence[str],
        known_state_ids: set[str],
    ) -> FrontierInsight:
        """Parse a structured JSON frontier-analysis completion into typed artifacts."""

        payload = _extract_structured_json_object(raw_output)
        if not payload:
            raise ValueError("Frontier-analysis completion did not contain any parseable insight fields.")

        parsed_annotations = [
            self._annotation_from_block(
                block=block,
                considered_frontier_ids=considered_frontier_ids,
                known_state_ids=known_state_ids,
            )
            for block in _as_mapping_list(payload.get("critical_states"))
        ]
        annotations = [annotation for annotation in parsed_annotations if annotation is not None]
        bottlenecks = _as_string_list(payload.get("bottlenecks"))
        partial_solutions = _as_string_list(payload.get("partial_solutions"))
        missing_prerequisites = _as_string_list(payload.get("missing_prerequisites"))
        resolved_analysis_id = str(payload.get("analysis_id", "")).strip() or analysis_id

        if not any((bottlenecks, partial_solutions, missing_prerequisites, annotations)):
            raise ValueError("Frontier-analysis completion did not contain any parseable insight fields.")

        return FrontierInsight(
            analysis_id=resolved_analysis_id,
            frontier_trajectory_ids=list(considered_frontier_ids),
            inferred_bottlenecks=bottlenecks,
            partial_solutions=partial_solutions,
            missing_prerequisites=missing_prerequisites,
            candidate_critical_states=annotations,
            generated_at=_utc_timestamp(),
            metadata={"analysis_mode": "llm_frontier_analysis"},
        )

    def _annotation_from_block(
        self,
        *,
        block: Mapping[str, str],
        considered_frontier_ids: Sequence[str],
        known_state_ids: set[str],
    ) -> CriticalStateAnnotation | None:
        """Parse one critical-state object from the frontier-analysis completion."""

        state_id = str(block.get("critical_state_id", block.get("id", ""))).strip()
        if not state_id:
            return None
        source_frontier_ids = [
            item
            for item in _as_string_list(block.get("source_frontier_trajectory_ids"))
            if item in considered_frontier_ids
        ]
        supporting_state_ids = _as_string_list(block.get("supporting_state_ids"))
        if known_state_ids:
            supporting_state_ids = [item for item in supporting_state_ids if item in known_state_ids]
        achieved_value = _parse_float(block.get("achieved_value"), default=0.0)
        potential_value = max(_parse_float(block.get("potential_value"), default=achieved_value), achieved_value)
        confidence = min(max(_parse_float(block.get("confidence"), default=0.0), 0.0), 1.0)
        support_count = max(_parse_int(block.get("support_count"), default=len(source_frontier_ids)), 0)
        rationale = str(block.get("textual_rationale", "")).strip()
        return CriticalStateAnnotation(
            critical_state_id=state_id,
            achieved_value=achieved_value,
            potential_value=potential_value,
            textual_rationale=rationale,
            source_frontier_trajectory_ids=source_frontier_ids,
            confidence=confidence,
            support_count=support_count,
            supporting_state_ids=supporting_state_ids,
        )

    def _heuristic_fallback_insight(
        self,
        *,
        analysis_id: str,
        entries: Sequence[FrontierTrajectoryEntry],
        trajectories: Sequence[EpisodeTrajectory],
        archived_states: Sequence[ArchivedState],
        fallback_reason: str,
    ) -> FrontierInsight:
        """Build a deterministic fallback insight when the LLM path is unavailable."""

        trajectory_ids = [entry.trajectory_id for entry in entries]
        bottlenecks = self._heuristic_bottlenecks(entries, trajectories)
        partial_solutions = self._heuristic_partial_solutions(entries, trajectories)
        missing_prerequisites = self._heuristic_missing_prerequisites(entries, trajectories, bottlenecks)
        critical_states = self._heuristic_critical_states(entries, trajectories, archived_states)
        return FrontierInsight(
            analysis_id=analysis_id,
            frontier_trajectory_ids=trajectory_ids,
            inferred_bottlenecks=bottlenecks,
            partial_solutions=partial_solutions,
            missing_prerequisites=missing_prerequisites,
            candidate_critical_states=critical_states,
            generated_at=_utc_timestamp(),
            metadata={
                "analysis_mode": "heuristic_fallback",
                "fallback_reason": fallback_reason,
                "paper_faithful_intent": "global_world_model_frontier_analysis",
                "implementation_choice": "deterministic_summary_when_llm_unavailable",
            },
        )

    def _heuristic_bottlenecks(
        self,
        entries: Sequence[FrontierTrajectoryEntry],
        trajectories: Sequence[EpisodeTrajectory],
    ) -> list[str]:
        """Infer repeated bottlenecks from retained frontier metadata."""

        bottleneck_counter: Counter[str] = Counter()
        final_cluster_counter: Counter[str] = Counter()
        trajectory_by_id = {trajectory.trajectory_id: trajectory for trajectory in trajectories}
        for entry in entries:
            bottleneck_counter.update(item for item in entry.bottleneck_state_ids if item)
            trajectory = trajectory_by_id.get(entry.trajectory_id)
            if trajectory is None:
                continue
            if trajectory.summary_fields.final_cluster_id:
                final_cluster_counter.update([trajectory.summary_fields.final_cluster_id])

        inferred: list[str] = []
        for state_id, count in bottleneck_counter.most_common(3):
            inferred.append(f"{state_id} appears as a bottleneck on {count} frontier trajectories")
        for cluster_id, count in final_cluster_counter.most_common(2):
            if count > 1:
                inferred.append(f"{cluster_id} is a repeated stall cluster across {count} trajectories")
        return inferred

    def _heuristic_partial_solutions(
        self,
        entries: Sequence[FrontierTrajectoryEntry],
        trajectories: Sequence[EpisodeTrajectory],
    ) -> list[str]:
        """Summarize retained partial solutions from the highest-value trajectories."""

        partial_solutions: list[str] = []
        entry_by_id = {entry.trajectory_id: entry for entry in entries}
        for trajectory in list(trajectories)[:4]:
            entry = entry_by_id.get(trajectory.trajectory_id)
            if entry is None:
                continue
            if trajectory.steps:
                prefix = " -> ".join(step.action for step in trajectory.steps[:4]) or "none"
                partial_solutions.append(
                    f"{trajectory.trajectory_id} reaches value {entry.value:.2f} with prefix {prefix}"
                )
        return partial_solutions

    def _heuristic_missing_prerequisites(
        self,
        entries: Sequence[FrontierTrajectoryEntry],
        trajectories: Sequence[EpisodeTrajectory],
        bottlenecks: Sequence[str],
    ) -> list[str]:
        """Infer missing prerequisites from repeated stalls and retained frontier structure."""

        if bottlenecks:
            return [f"Need actions beyond: {bottlenecks[0]}"]
        if not trajectories:
            return []
        repeated_clusters = Counter(
            trajectory.summary_fields.final_cluster_id
            for trajectory in trajectories
            if trajectory.summary_fields.final_cluster_id
        )
        for cluster_id, count in repeated_clusters.most_common(1):
            if count > 1:
                return [f"Need progress beyond repeated cluster {cluster_id}"]
        return []

    def _heuristic_critical_states(
        self,
        entries: Sequence[FrontierTrajectoryEntry],
        trajectories: Sequence[EpisodeTrajectory],
        archived_states: Sequence[ArchivedState],
    ) -> list[CriticalStateAnnotation]:
        """Derive candidate critical states from achieved and potential value gaps."""

        state_to_potential_value = self._potential_value_by_state_id(trajectories)
        state_to_supporting_trajectories = self._supporting_trajectory_ids_by_state(trajectories)
        trajectory_ids = {entry.trajectory_id for entry in entries}
        annotations: list[CriticalStateAnnotation] = []
        for archived_state in archived_states[:10]:
            supporting_trajectory_ids = [
                trajectory_id
                for trajectory_id in state_to_supporting_trajectories.get(archived_state.state_id, [])
                if trajectory_id in trajectory_ids
            ]
            if not supporting_trajectory_ids:
                continue
            potential_value = max(
                state_to_potential_value.get(archived_state.state_id, archived_state.achieved_value),
                archived_state.achieved_value,
            )
            annotations.append(
                CriticalStateAnnotation(
                    critical_state_id=archived_state.state_id,
                    achieved_value=archived_state.achieved_value,
                    potential_value=potential_value,
                    textual_rationale=(
                        f"State appears on {len(supporting_trajectory_ids)} frontier trajectories and "
                        f"is followed by continuations reaching up to {potential_value:.2f}."
                    ),
                    source_frontier_trajectory_ids=supporting_trajectory_ids[:4],
                    confidence=min(1.0, 0.35 + 0.15 * len(supporting_trajectory_ids)),
                    support_count=len(supporting_trajectory_ids),
                    supporting_state_ids=[archived_state.state_id],
                )
            )
        annotations.sort(
            key=lambda item: (-(item.potential_value - item.achieved_value), -item.achieved_value, item.critical_state_id)
        )
        return annotations[:5]

    def _potential_value_by_state_id(self, trajectories: Sequence[EpisodeTrajectory]) -> dict[str, float]:
        """Return the best trajectory-level value observed downstream of each state."""

        potential_by_state_id: dict[str, float] = {}
        for trajectory in trajectories:
            trajectory_potential = float(trajectory.max_cumulative_reward_achieved)
            for step in trajectory.steps:
                state_id = self._state_id_for_step(trajectory, step)
                previous_value = potential_by_state_id.get(state_id, float("-inf"))
                if trajectory_potential > previous_value:
                    potential_by_state_id[state_id] = trajectory_potential
        return potential_by_state_id

    def _supporting_trajectory_ids_by_state(
        self,
        trajectories: Sequence[EpisodeTrajectory],
    ) -> dict[str, list[str]]:
        """Return the frontier trajectory ids that contain each state id."""

        supporting: dict[str, list[str]] = defaultdict(list)
        for trajectory in trajectories:
            seen_state_ids: set[str] = set()
            for step in trajectory.steps:
                state_id = self._state_id_for_step(trajectory, step)
                if state_id in seen_state_ids:
                    continue
                supporting[state_id].append(trajectory.trajectory_id)
                seen_state_ids.add(state_id)
        return supporting

    def _state_id_for_step(self, trajectory: EpisodeTrajectory, step) -> str:
        """Return the archive-compatible state id used across analysis artifacts."""

        state_id, _identity_strategy = archive_state_identity_for_step(
            step,
            trajectory_id=trajectory.trajectory_id,
        )
        return state_id

    def _default_analysis_id(self, entries: Sequence[FrontierTrajectoryEntry]) -> str:
        """Build a stable-ish analysis id when one is not supplied explicitly."""

        if not entries:
            return f"frontier-analysis-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        top_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", entries[0].trajectory_id).strip("-") or "frontier"
        return f"frontier-analysis-{top_id}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def _clean_analysis_line(value: str) -> str:
    """Strip bullet/number prefixes from a frontier-analysis line."""

    cleaned = value.strip()
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[\).:\-])\s*", "", cleaned)
    cleaned = re.sub(r"^\s*>+\s*", "", cleaned)
    cleaned = cleaned.strip()
    return cleaned


def _frontier_analysis_response_format() -> dict[str, object]:
    """Return the strict JSON schema used for normal-mode frontier analysis."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "frontier_analysis_result",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "analysis_id": {"type": "string"},
                    "bottlenecks": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "partial_solutions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "missing_prerequisites": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "critical_states": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "critical_state_id": {"type": "string"},
                                "achieved_value": {"type": "number"},
                                "potential_value": {"type": "number"},
                                "confidence": {"type": "number"},
                                "support_count": {"type": "integer"},
                                "source_frontier_trajectory_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "supporting_state_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "textual_rationale": {"type": "string"},
                            },
                            "required": [
                                "critical_state_id",
                                "achieved_value",
                                "potential_value",
                                "confidence",
                                "support_count",
                                "source_frontier_trajectory_ids",
                                "supporting_state_ids",
                                "textual_rationale",
                            ],
                        },
                    },
                },
                "required": [
                    "analysis_id",
                    "bottlenecks",
                    "partial_solutions",
                    "missing_prerequisites",
                    "critical_states",
                ],
            },
        },
    }


def _extract_structured_json_object(raw_output: str) -> dict[str, object]:
    """Extract the structured JSON payload from a completion.

    Normal mode expects only JSON. Debug mode may include reasoning before a final
    `BEGIN_STRUCTURED_OUTPUT` / `END_STRUCTURED_OUTPUT` block.
    """

    normalized = raw_output.strip()
    if not normalized:
        return {}

    marker_match = re.search(
        r"BEGIN_STRUCTURED_OUTPUT\s*(\{.*?\})\s*END_STRUCTURED_OUTPUT",
        normalized,
        flags=re.DOTALL,
    )
    if marker_match:
        return _parse_json_object(marker_match.group(1))

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", normalized, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        parsed = _parse_json_object(fence_match.group(1))
        if parsed:
            return parsed

    json_match = re.search(r"\{.*\}", normalized, flags=re.DOTALL)
    if json_match:
        return _parse_json_object(json_match.group(0))
    return {}


def _parse_json_object(payload: str) -> dict[str, object]:
    """Parse one JSON object conservatively."""

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_string_list(value: object) -> list[str]:
    """Normalize a JSON string/list field into a list of strings."""

    if isinstance(value, str):
        parts = re.split(r"\s*;\s*|\s*\|\s*", value)
        return [" ".join(part.split()).strip() for part in parts if " ".join(part.split()).strip()]
    if isinstance(value, list):
        return [" ".join(str(item).split()).strip() for item in value if str(item).strip()]
    return []


def _as_mapping_list(value: object) -> list[Mapping[str, object]]:
    """Normalize a JSON list field into mapping items only."""

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _split_field(line: str) -> tuple[str, str]:
    """Split a `field: value` line into its normalized parts."""

    if ":" not in line:
        return "", ""
    key, _, value = line.partition(":")
    normalized_key = re.sub(r"[*`]+", "", key).strip().lower().replace(" ", "_")
    return normalized_key, value.strip()


def _parse_float(value: object, *, default: float) -> float:
    """Parse a float field from a tolerant JSON/plain-text payload."""

    if value is None:
        return default
    try:
        return float(str(value).strip())
    except ValueError:
        return default


def _parse_int(value: object, *, default: int) -> int:
    """Parse an integer field from a tolerant JSON/plain-text payload."""

    if value is None:
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def _truncate(value: str, limit: int) -> str:
    """Truncate prompt/debug text conservatively."""

    normalized = " ".join(value.split()).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _utc_timestamp() -> str:
    """Return an ISO UTC timestamp used by typed analysis artifacts."""

    return datetime.now(timezone.utc).isoformat()
