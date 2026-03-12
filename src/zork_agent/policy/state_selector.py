"""State-selection logic for choosing which saved state to revisit next.

The primary path is the archive selector that balances achieved and potential
value using archived states plus global frontier analysis. A frontier-node path
still exists internally as a deterministic fallback for local heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from zork_agent.config import ProjectConfig
from zork_agent.llm.base import BaseLLMClient
from zork_agent.llm.prompts import PromptManager
from zork_agent.memory.frontier import FrontierEntry, FrontierQueue, TrajectoryFrontier
from zork_agent.memory.summaries import SummaryBuilder
from zork_agent.types import (
    ArchiveCandidateSummary,
    ArchiveStateSelectionResult,
    ArchivedState,
    CriticalStateAnnotation,
    FrontierCandidateSummary,
    FrontierInsight,
    RootDepthProgressionProfile,
    RestoreMode,
    StateSelectionMode,
    StateSelectionResult,
)
from zork_agent.utils.serialization import to_jsonable

_LOGGER = logging.getLogger("zork_agent.state_selector")


@dataclass(slots=True)
class _ArchiveSelectionCandidate:
    """Internal scored archive-selection candidate."""

    archived_state: ArchivedState
    achieved_value: float
    potential_value: float
    achieved_contribution: float
    potential_contribution: float
    total_score: float
    replay_method: str
    support_count: int = 0
    direct_match_count: int = 0
    supporting_match_count: int = 0
    source_trajectory_match_count: int = 0
    no_achievement_revisit_streak: int = 0
    nonproductive_revisit_streak: int = 0
    stale_revisit_penalty: float = 0.0
    depth_progression_productive_root: bool = False
    depth_progression_replay_saturation_level: int = 0
    depth_progression_penalty: float = 0.0
    total_score_before_depth_progression: float = 0.0
    selected_frontier_trajectory_ids: list[str] = field(default_factory=list)
    selected_critical_state_ids: list[str] = field(default_factory=list)
    rationale_parts: list[str] = field(default_factory=list)

    def to_summary(self, *, candidate_id: str, summary_text: str) -> ArchiveCandidateSummary:
        """Project the internal scored candidate into a compact summary object."""

        return ArchiveCandidateSummary(
            candidate_id=candidate_id,
            state_id=self.archived_state.state_id,
            achieved_value=self.achieved_value,
            potential_value=self.potential_value,
            achieved_contribution=self.achieved_contribution,
            potential_contribution=self.potential_contribution,
            replay_method=self.replay_method,
            visit_count=self.archived_state.visit_count,
            selection_count=self.archived_state.selection_count,
            support_count=self.support_count,
            summary_text=summary_text,
        )

    def to_record(self, *, candidate_id: str, summary_text: str) -> dict[str, Any]:
        """Convert the internal candidate into a JSON-serializable debug record."""

        return {
            "candidate_id": candidate_id,
            "state_id": self.archived_state.state_id,
            "first_seen_episode_id": self.archived_state.first_seen_episode_id,
            "last_seen_episode_id": self.archived_state.last_seen_episode_id,
            "provenance_trajectory_id": self.archived_state.provenance_trajectory_id,
            "provenance_timestep": self.archived_state.provenance_timestep,
            "achieved_value": self.achieved_value,
            "potential_value": self.potential_value,
            "achieved_contribution": self.achieved_contribution,
            "potential_contribution": self.potential_contribution,
            "total_score": self.total_score,
            "replay_method": self.replay_method,
            "recorded_restore_strategy": self.archived_state.replay_metadata.restore_strategy,
            "has_native_snapshot": bool(
                self.archived_state.native_snapshot is not None
                or self.archived_state.replay_metadata.native_snapshot is not None
            ),
            "visit_count": self.archived_state.visit_count,
            "selection_count": self.archived_state.selection_count,
            "support_count": self.support_count,
            "direct_match_count": self.direct_match_count,
            "supporting_match_count": self.supporting_match_count,
            "source_trajectory_match_count": self.source_trajectory_match_count,
            "no_achievement_revisit_streak": self.no_achievement_revisit_streak,
            "nonproductive_revisit_streak": self.nonproductive_revisit_streak,
            "stale_revisit_penalty": self.stale_revisit_penalty,
            "depth_progression_productive_root": self.depth_progression_productive_root,
            "depth_progression_replay_saturation_level": self.depth_progression_replay_saturation_level,
            "depth_progression_penalty": self.depth_progression_penalty,
            "total_score_before_depth_progression": self.total_score_before_depth_progression,
            "selected_frontier_trajectory_ids": list(self.selected_frontier_trajectory_ids),
            "selected_critical_state_ids": list(self.selected_critical_state_ids),
            "rationale_parts": list(self.rationale_parts),
            "summary_text": summary_text,
        }


class StateSelector:
    """Choose which previously discovered state should be revisited next."""

    def __init__(
        self,
        *,
        config: ProjectConfig | None = None,
        prompt_manager: PromptManager | None = None,
        llm_client: BaseLLMClient | None = None,
        selection_mode: StateSelectionMode | str = StateSelectionMode.HEURISTIC,
        archive_selection_mode: StateSelectionMode | str | None = None,
        prompt_candidate_limit: int = 6,
        summary_max_chars: int = 120,
    ):
        self.config = config
        self.prompt_manager = prompt_manager
        self.llm_client = llm_client
        self.selection_mode = StateSelectionMode(selection_mode)
        self.archive_selection_mode = StateSelectionMode(
            archive_selection_mode
            or (
                config.policy.archive_state_selection_mode
                if config is not None
                else StateSelectionMode.ARCHIVE_BALANCED
            )
        )
        configured_limit = (
            config.policy.state_selection_prompt_candidate_limit
            if config is not None
            else prompt_candidate_limit
        )
        self.prompt_candidate_limit = max(1, configured_limit)
        self.summary_max_chars = max(32, summary_max_chars)
        self.summary_builder = SummaryBuilder()
        self.last_result: StateSelectionResult | None = None
        self.last_archive_result: ArchiveStateSelectionResult | None = None
        self.artifact_root = (
            config.paths.summary_dir / "state_selection"
            if config is not None
            else Path("artifacts") / "summaries" / "state_selection"
        )

    def select(
        self,
        frontier: FrontierQueue,
        *,
        mode: StateSelectionMode | str | None = None,
    ) -> FrontierEntry | None:
        """Internal frontier-node selection path used for deterministic local heuristics."""

        result = self.select_with_details(frontier, mode=mode)
        self.last_result = result
        return result.selected

    def select_with_details(
        self,
        frontier: FrontierQueue,
        *,
        mode: StateSelectionMode | str | None = None,
    ) -> StateSelectionResult:
        """Legacy frontier-node selection path retained for the current episode loop."""

        candidate_entries = frontier.top_k(self.prompt_candidate_limit)
        candidate_summaries = self._build_candidate_summaries(candidate_entries)
        effective_mode = self._resolve_mode(mode, default_mode=self.selection_mode)

        if not candidate_entries:
            result = StateSelectionResult(
                selected=None,
                reason="No frontier candidates are available.",
                selection_mode=StateSelectionMode.HEURISTIC,
                candidate_summaries=[],
            )
            self.last_result = result
            return result

        if effective_mode is StateSelectionMode.LLM_ASSISTED:
            result = self._select_legacy_with_llm(candidate_entries, candidate_summaries)
        else:
            legacy_mode = (
                effective_mode
                if effective_mode in {StateSelectionMode.HEURISTIC, StateSelectionMode.LEGACY_HEURISTIC, StateSelectionMode.FALLBACK_HEURISTIC}
                else StateSelectionMode.HEURISTIC
            )
            result = self._select_heuristically(candidate_entries, candidate_summaries, legacy_mode)

        self.last_result = result
        return result

    def select_archive_state(
        self,
        *,
        archive_states: Sequence[ArchivedState],
        frontier_insight: FrontierInsight | None,
        trajectory_frontier: TrajectoryFrontier | None = None,
        mode: StateSelectionMode | str | None = None,
        selection_id: str | None = None,
    ) -> ArchiveStateSelectionResult:
        """Select an archived state using achieved plus potential value."""

        resolved_selection_id = selection_id or self._default_selection_id()
        effective_mode = self._resolve_mode(mode, default_mode=self.archive_selection_mode)
        candidates = self._generate_archive_candidates(
            archive_states=archive_states,
            frontier_insight=frontier_insight,
            trajectory_frontier=trajectory_frontier,
        )
        pre_depth_progression_top_candidate = min(
            candidates,
            key=lambda candidate: (
                -candidate.total_score_before_depth_progression,
                -candidate.potential_contribution,
                -candidate.achieved_contribution,
                candidate.archived_state.provenance_trajectory_id,
                candidate.archived_state.provenance_timestep,
                candidate.archived_state.state_id,
            ),
            default=None,
        )
        candidate_summaries = self._build_archive_candidate_summaries(candidates)
        prompt_snapshot = "\n".join(summary.to_prompt_line() for summary in candidate_summaries)

        if not candidates:
            result = ArchiveStateSelectionResult(
                selected_archived_state=None,
                chosen_replay_method="",
                achieved_contribution=0.0,
                potential_contribution=0.0,
                rationale="No archived states are available for selection.",
                selection_mode=StateSelectionMode.ARCHIVE_BALANCED,
                candidate_summaries=[],
                prompt_snapshot=prompt_snapshot,
                fallback_reason="No archived states are available.",
            )
            self.last_archive_result = result
            self._write_archive_selection_artifacts(
                selection_id=resolved_selection_id,
                prompt_snapshot=prompt_snapshot,
                raw_output="",
                result=result,
                candidates=[],
            )
            return result

        if effective_mode is StateSelectionMode.LLM_ASSISTED:
            result = self._select_archive_with_llm(
                candidates,
                candidate_summaries,
                frontier_insight=frontier_insight,
                prompt_snapshot=prompt_snapshot,
            )
        else:
            selected_candidate = candidates[0]
            result = self._archive_result_from_candidate(
                selected_candidate,
                candidate_summaries,
                selection_mode=(
                    effective_mode
                    if effective_mode in {StateSelectionMode.ARCHIVE_BALANCED, StateSelectionMode.FALLBACK_HEURISTIC}
                    else StateSelectionMode.ARCHIVE_BALANCED
                ),
            )

        self.last_archive_result = result
        result.metadata.update(
            {
                "pre_depth_progression_top_state_id": (
                    pre_depth_progression_top_candidate.archived_state.state_id
                    if pre_depth_progression_top_candidate is not None
                    else None
                ),
                "post_depth_progression_top_state_id": (
                    candidates[0].archived_state.state_id if candidates else None
                ),
                "pre_depth_progression_top_score": (
                    pre_depth_progression_top_candidate.total_score_before_depth_progression
                    if pre_depth_progression_top_candidate is not None
                    else 0.0
                ),
                "post_depth_progression_top_score": candidates[0].total_score if candidates else 0.0,
            }
        )
        self._write_archive_selection_artifacts(
            selection_id=resolved_selection_id,
            prompt_snapshot=prompt_snapshot,
            raw_output=result.raw_output,
            result=result,
            candidates=candidates,
        )
        return result

    def _generate_archive_candidates(
        self,
        *,
        archive_states: Sequence[ArchivedState],
        frontier_insight: FrontierInsight | None,
        trajectory_frontier: TrajectoryFrontier | None,
    ) -> list[_ArchiveSelectionCandidate]:
        """Generate achieved-plus-potential archive selection candidates."""

        critical_annotations = list(frontier_insight.candidate_critical_states) if frontier_insight is not None else []
        candidates = [
            self._score_archive_candidate(
                archived_state=archived_state,
                critical_annotations=critical_annotations,
                frontier_trajectory_ids=frontier_insight.frontier_trajectory_ids if frontier_insight is not None else [],
                trajectory_frontier=trajectory_frontier,
            )
            for archived_state in archive_states
        ]
        candidates.sort(
            key=lambda candidate: (
                -candidate.total_score,
                -candidate.potential_contribution,
                -candidate.achieved_contribution,
                candidate.archived_state.provenance_trajectory_id,
                candidate.archived_state.provenance_timestep,
                candidate.archived_state.state_id,
            )
        )
        return candidates

    def _score_archive_candidate(
        self,
        *,
        archived_state: ArchivedState,
        critical_annotations: Sequence[CriticalStateAnnotation],
        frontier_trajectory_ids: Sequence[str],
        trajectory_frontier: TrajectoryFrontier | None,
    ) -> _ArchiveSelectionCandidate:
        """Score one archived state by achieved and potential value."""

        config = self.config.policy if self.config is not None else None
        potential_value_enabled = (
            self.config.experiment.enable_potential_value_selection
            if self.config is not None
            else True
        )
        achieved_weight = config.state_selection_achieved_value_weight if config is not None else 1.0
        potential_weight = (
            (config.state_selection_potential_value_weight if config is not None else 1.5)
            if potential_value_enabled
            else 0.0
        )
        selection_penalty_weight = config.state_selection_selection_penalty if config is not None else 0.25
        visit_penalty_weight = config.state_selection_visit_penalty if config is not None else 0.1
        no_achievement_revisit_penalty_weight = (
            config.state_selection_no_achievement_revisit_penalty if config is not None else 1.25
        )
        nonproductive_revisit_penalty_weight = (
            config.state_selection_nonproductive_revisit_penalty if config is not None else 0.5
        )
        root_replay_saturation_penalty_weight = (
            config.state_selection_root_replay_saturation_penalty_weight if config is not None else 0.75
        )
        direct_bonus = config.state_selection_direct_match_bonus if config is not None else 1.0
        supporting_bonus = config.state_selection_supporting_match_bonus if config is not None else 0.75
        source_bonus = config.state_selection_source_trajectory_bonus if config is not None else 0.35
        native_snapshot_bonus = config.state_selection_native_snapshot_bonus if config is not None else 0.2
        no_achievement_revisit_streak = self._metadata_int(archived_state.metadata, "no_achievement_revisit_streak")
        nonproductive_revisit_streak = self._metadata_int(archived_state.metadata, "nonproductive_revisit_streak")
        depth_progression_profile = self._depth_progression_profile(archived_state.metadata)

        replay_method = self._available_archive_replay_method(archived_state)
        achieved_contribution = achieved_weight * float(archived_state.achieved_value)
        potential_contribution = 0.0
        potential_value = float(
            archived_state.projected_potential_value
            if archived_state.projected_potential_value is not None
            else archived_state.achieved_value
        )
        support_count = int(archived_state.frontier_support_count)
        direct_match_count = 0
        supporting_match_count = 0
        source_trajectory_match_count = 0
        selected_frontier_trajectory_ids: set[str] = set(archived_state.supporting_frontier_trajectory_ids)
        selected_critical_state_ids: set[str] = set()
        rationale_parts = [f"achieved={archived_state.achieved_value:.2f}"]
        if not potential_value_enabled:
            rationale_parts.append("potential_value_disabled")
        elif archived_state.projected_potential_value is not None:
            potential_gap = max(
                0.0,
                float(archived_state.projected_potential_value) - float(archived_state.achieved_value),
            )
            if potential_gap > 0.0:
                potential_contribution = max(
                    potential_contribution,
                    potential_weight * potential_gap,
                )
                rationale_parts.append(f"cached_potential={archived_state.projected_potential_value:.2f}")

        for annotation in critical_annotations:
            if not potential_value_enabled:
                continue
            alignment_bonus = 0.0
            alignment_scale = 0.0
            if archived_state.state_id == annotation.critical_state_id:
                alignment_bonus = direct_bonus
                alignment_scale = 1.0
                direct_match_count += 1
            elif archived_state.state_id in annotation.supporting_state_ids:
                alignment_bonus = supporting_bonus
                alignment_scale = 0.85
                supporting_match_count += 1
            elif archived_state.provenance_trajectory_id in annotation.source_frontier_trajectory_ids:
                alignment_bonus = source_bonus
                alignment_scale = 0.15
                source_trajectory_match_count += 1

            if alignment_bonus <= 0.0 and alignment_scale <= 0.0:
                continue

            potential_gap = max(0.0, float(annotation.potential_value) - float(archived_state.achieved_value))
            candidate_potential = potential_weight * potential_gap * alignment_scale + alignment_bonus
            if candidate_potential > potential_contribution:
                potential_contribution = candidate_potential
                potential_value = max(float(annotation.potential_value), potential_value)
            support_count = max(support_count, int(annotation.support_count or len(annotation.source_frontier_trajectory_ids)))
            selected_frontier_trajectory_ids.update(annotation.source_frontier_trajectory_ids)
            selected_critical_state_ids.add(annotation.critical_state_id)
        total_score = achieved_contribution + potential_contribution
        if archived_state.native_snapshot is not None or archived_state.native_snapshot_reference:
            total_score += native_snapshot_bonus
            rationale_parts.append("native_snapshot")

        stale_revisit_penalty = 0.0
        if no_achievement_revisit_streak > 0:
            stale_revisit_penalty += no_achievement_revisit_penalty_weight * float(no_achievement_revisit_streak)
            rationale_parts.append(f"no_achievement_streak={no_achievement_revisit_streak}")
        if nonproductive_revisit_streak > 0:
            stale_revisit_penalty += nonproductive_revisit_penalty_weight * float(nonproductive_revisit_streak)
            rationale_parts.append(f"nonproductive_streak={nonproductive_revisit_streak}")
        if stale_revisit_penalty > 0.0:
            total_score -= stale_revisit_penalty
            rationale_parts.append(f"stale_revisit_penalty={stale_revisit_penalty:.2f}")

        total_score_before_depth_progression = total_score
        depth_progression_penalty = 0.0
        if (
            depth_progression_profile.productive_root
            and depth_progression_profile.replay_saturation_level > 0
        ):
            depth_progression_penalty = (
                root_replay_saturation_penalty_weight
                * float(depth_progression_profile.replay_saturation_level)
            )
            total_score -= depth_progression_penalty
            rationale_parts.append(
                f"root_replay_saturation={depth_progression_profile.replay_saturation_level}"
            )
            rationale_parts.append(f"depth_progression_penalty={depth_progression_penalty:.2f}")

        if archived_state.selection_count > 0:
            total_score -= selection_penalty_weight * float(archived_state.selection_count)
            rationale_parts.append(f"selection_count={archived_state.selection_count}")
        if archived_state.visit_count > 0:
            total_score -= visit_penalty_weight * float(archived_state.visit_count)
            rationale_parts.append(f"visit_count={archived_state.visit_count}")

        if direct_match_count > 0:
            rationale_parts.append("direct_critical_match")
        if supporting_match_count > 0:
            rationale_parts.append("supporting_critical_match")
        if source_trajectory_match_count > 0:
            rationale_parts.append("source_frontier_alignment")
        if potential_contribution > 0.0:
            rationale_parts.append(f"potential={potential_value:.2f}")
        if (
            potential_contribution <= 0.0
            and trajectory_frontier is not None
            and archived_state.provenance_trajectory_id in {entry.trajectory_id for entry in trajectory_frontier.snapshot_entries()}
        ):
            rationale_parts.append("retained_frontier_provenance")
        if archived_state.provenance_trajectory_id in frontier_trajectory_ids:
            rationale_parts.append("frontier_member")

        return _ArchiveSelectionCandidate(
            archived_state=archived_state,
            achieved_value=float(archived_state.achieved_value),
            potential_value=potential_value,
            achieved_contribution=achieved_contribution,
            potential_contribution=potential_contribution,
            total_score=total_score,
            replay_method=replay_method,
            support_count=support_count,
            direct_match_count=direct_match_count,
            supporting_match_count=supporting_match_count,
            source_trajectory_match_count=source_trajectory_match_count,
            no_achievement_revisit_streak=no_achievement_revisit_streak,
            nonproductive_revisit_streak=nonproductive_revisit_streak,
            stale_revisit_penalty=stale_revisit_penalty,
            depth_progression_productive_root=depth_progression_profile.productive_root,
            depth_progression_replay_saturation_level=depth_progression_profile.replay_saturation_level,
            depth_progression_penalty=depth_progression_penalty,
            total_score_before_depth_progression=total_score_before_depth_progression,
            selected_frontier_trajectory_ids=sorted(selected_frontier_trajectory_ids),
            selected_critical_state_ids=sorted(selected_critical_state_ids),
            rationale_parts=rationale_parts,
        )

    @staticmethod
    def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
        """Read one metadata counter as a non-negative integer."""

        raw_value = metadata.get(key, 0)
        try:
            return max(0, int(raw_value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _depth_progression_profile(metadata: Mapping[str, Any]) -> RootDepthProgressionProfile:
        """Return the persisted depth-progression profile for one archived root."""

        raw_profile = metadata.get("depth_progression")
        if isinstance(raw_profile, Mapping):
            return RootDepthProgressionProfile.from_record(raw_profile)
        return RootDepthProgressionProfile()

    def _build_archive_candidate_summaries(
        self,
        candidates: Sequence[_ArchiveSelectionCandidate],
    ) -> list[ArchiveCandidateSummary]:
        """Build compact summaries for achieved-plus-potential archive selection."""

        summaries: list[ArchiveCandidateSummary] = []
        for index, candidate in enumerate(list(candidates)[: self.prompt_candidate_limit], start=1):
            summary_text = self._compact_text(
                self._archive_candidate_summary_text(candidate.archived_state, candidate)
            )
            summaries.append(candidate.to_summary(candidate_id=f"C{index}", summary_text=summary_text))
        return summaries

    def _archive_candidate_summary_text(
        self,
        archived_state: ArchivedState,
        candidate: _ArchiveSelectionCandidate,
    ) -> str:
        """Build a compact candidate summary from archive metadata."""

        metadata = archived_state.metadata
        state_cluster_id = archived_state.state_cluster_id or str(metadata.get("state_cluster_id", "")).strip() or "unknown"
        world_state_hash = archived_state.replay_metadata.world_state_hash or str(metadata.get("world_state_hash", "unknown"))
        observation_summary = archived_state.observation_summary or "unknown"
        return (
            f"first_seen={archived_state.first_seen_episode_id or 'unknown'} "
            f"traj={archived_state.provenance_trajectory_id} step={archived_state.provenance_timestep} "
            f"cluster={state_cluster_id} world={world_state_hash} "
            f"support={archived_state.frontier_support_count} "
            f"potential={archived_state.projected_potential_value if archived_state.projected_potential_value is not None else archived_state.achieved_value:.2f} "
            f"obs={self._compact_text(observation_summary)} "
            f"reason={'; '.join(candidate.rationale_parts[:4]) or 'none'}"
        )

    def _available_archive_replay_method(self, archived_state: ArchivedState) -> str:
        """Return the restore method that is actually available for this archive state."""

        if archived_state.native_snapshot is not None or archived_state.replay_metadata.native_snapshot is not None:
            return RestoreMode.NATIVE_SNAPSHOT.value
        if archived_state.replay_metadata.replay_actions:
            return RestoreMode.ACTION_REPLAY_FALLBACK.value
        return RestoreMode.RESET_ONLY.value

    def _select_archive_with_llm(
        self,
        candidates: Sequence[_ArchiveSelectionCandidate],
        candidate_summaries: Sequence[ArchiveCandidateSummary],
        *,
        frontier_insight: FrontierInsight | None,
        prompt_snapshot: str,
    ) -> ArchiveStateSelectionResult:
        """Ask the LLM to adjudicate among top achieved-plus-potential archive candidates."""

        if self.llm_client is None or self.prompt_manager is None or self.config is None:
            _LOGGER.warning(
                "Archive state selection fell back to deterministic mode: missing selector dependencies for LLM adjudication."
            )
            return self._archive_result_from_candidate(
                candidates[0],
                list(candidate_summaries),
                selection_mode=StateSelectionMode.FALLBACK_HEURISTIC,
                fallback_reason="LLM-assisted archive selection requested without full selector dependencies.",
                prompt_snapshot=prompt_snapshot,
            )

        try:
            response = self.llm_client.score_state_for_revisit(
                prompt_manager=self.prompt_manager,
                game_id=self.config.experiment.game_id,
                frontier_snapshot=prompt_snapshot,
                model=self.config.llm.model_name,
                temperature=min(self.config.llm.temperature, 0.2),
                max_tokens=(
                    self.config.llm.state_selection_debug_max_tokens
                    if self.config.llm.analysis_debug_mode
                    else self.config.llm.state_selection_max_tokens
                ),
                timeout_seconds=self.config.llm.request_timeout_seconds,
                analysis_debug_mode=self.config.llm.analysis_debug_mode,
                response_format=(
                    None
                    if self.config.llm.analysis_debug_mode
                    else _state_selection_response_format()
                ),
            )
        except Exception as exc:
            _LOGGER.warning("Archive state selection fell back after LLM request failure: %s", exc)
            return self._archive_result_from_candidate(
                candidates[0],
                list(candidate_summaries),
                selection_mode=StateSelectionMode.FALLBACK_HEURISTIC,
                fallback_reason=f"LLM request failed: {exc}",
                prompt_snapshot=prompt_snapshot,
            )

        selected_summary = self._parse_llm_selection(response.text, candidate_summaries)
        if selected_summary is None:
            _LOGGER.warning("Archive state selection fell back after malformed LLM output: %r", response.text)
            return self._archive_result_from_candidate(
                candidates[0],
                list(candidate_summaries),
                selection_mode=StateSelectionMode.FALLBACK_HEURISTIC,
                fallback_reason="LLM output did not identify a valid archive candidate.",
                raw_output=response.text,
                prompt_snapshot=prompt_snapshot,
                model_name=response.model,
            )

        selected_index = candidate_summaries.index(selected_summary)
        return self._archive_result_from_candidate(
            candidates[selected_index],
            list(candidate_summaries),
            selection_mode=StateSelectionMode.LLM_ASSISTED,
            rationale=self._extract_llm_reason(response.text, selected_summary),
            raw_output=response.text,
            prompt_snapshot=prompt_snapshot,
            model_name=response.model,
        )

    def _archive_result_from_candidate(
        self,
        candidate: _ArchiveSelectionCandidate,
        candidate_summaries: list[ArchiveCandidateSummary],
        *,
        selection_mode: StateSelectionMode,
        rationale: str | None = None,
        fallback_reason: str = "",
        raw_output: str = "",
        prompt_snapshot: str = "",
        model_name: str | None = None,
    ) -> ArchiveStateSelectionResult:
        """Project an internal archive candidate into the public selection result."""

        resolved_rationale = rationale or (
            f"Selected {candidate.archived_state.state_id} with achieved contribution "
            f"{candidate.achieved_contribution:.2f} and potential contribution {candidate.potential_contribution:.2f}."
        )
        if fallback_reason:
            resolved_rationale = f"Fell back to deterministic archive selection. {resolved_rationale}"
        return ArchiveStateSelectionResult(
            selected_archived_state=candidate.archived_state,
            chosen_replay_method=candidate.replay_method,
            achieved_contribution=candidate.achieved_contribution,
            potential_contribution=candidate.potential_contribution,
            rationale=resolved_rationale,
            selection_mode=selection_mode,
            candidate_summaries=candidate_summaries,
            selected_frontier_trajectory_ids=list(candidate.selected_frontier_trajectory_ids),
            selected_critical_state_ids=list(candidate.selected_critical_state_ids),
            raw_output=raw_output,
            prompt_snapshot=prompt_snapshot,
            model_name=model_name,
            fallback_reason=fallback_reason,
            metadata={
                "support_count": candidate.support_count,
                "total_score": candidate.total_score,
                "total_score_before_depth_progression": candidate.total_score_before_depth_progression,
                "depth_progression_productive_root": candidate.depth_progression_productive_root,
                "depth_progression_replay_saturation_level": (
                    candidate.depth_progression_replay_saturation_level
                ),
                "depth_progression_penalty": candidate.depth_progression_penalty,
            },
        )

    def _write_archive_selection_artifacts(
        self,
        *,
        selection_id: str,
        prompt_snapshot: str,
        raw_output: str,
        result: ArchiveStateSelectionResult,
        candidates: Sequence[_ArchiveSelectionCandidate],
    ) -> None:
        """Persist archive-selection prompt and scoring artifacts for offline inspection."""

        selection_dir = self.artifact_root / selection_id
        selection_dir.mkdir(parents=True, exist_ok=True)
        candidate_debug_records = []
        for index, candidate in enumerate(candidates[: self.prompt_candidate_limit], start=1):
            summary_text = self._compact_text(self._archive_candidate_summary_text(candidate.archived_state, candidate))
            candidate_debug_records.append(candidate.to_record(candidate_id=f"C{index}", summary_text=summary_text))

        (selection_dir / "prompt.txt").write_text(prompt_snapshot, encoding="utf-8")
        (selection_dir / "raw_completion.txt").write_text(raw_output, encoding="utf-8")
        (selection_dir / "candidate_scores.json").write_text(
            json.dumps(to_jsonable(candidate_debug_records), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        result.artifact_directory = str(selection_dir)
        (selection_dir / "selection_result.json").write_text(
            json.dumps(to_jsonable(result.to_record()), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _select_heuristically(
        self,
        candidate_entries: list[FrontierEntry],
        candidate_summaries: list[FrontierCandidateSummary],
        selection_mode: StateSelectionMode,
        *,
        fallback_reason: str = "",
        raw_output: str = "",
        prompt_snapshot: str = "",
        model_name: str | None = None,
    ) -> StateSelectionResult:
        """Pick the current best legacy frontier entry according to explicit ranking."""

        selected = candidate_entries[0]
        reason = (
            f"Selected {selected.state_id} by frontier priority={selected.priority:.2f} "
            f"(score={selected.score:.2f}, novelty={selected.novelty:.2f}, "
            f"recent_gain={selected.recent_gain:.2f}, depth={selected.depth})."
        )
        if fallback_reason:
            reason = f"Fell back to heuristic selection. {reason}"

        return StateSelectionResult(
            selected=selected,
            reason=reason,
            selection_mode=selection_mode,
            candidate_summaries=candidate_summaries,
            raw_output=raw_output,
            prompt_snapshot=prompt_snapshot,
            model_name=model_name,
            fallback_reason=fallback_reason,
        )

    def _select_legacy_with_llm(
        self,
        candidate_entries: list[FrontierEntry],
        candidate_summaries: list[FrontierCandidateSummary],
    ) -> StateSelectionResult:
        """Ask the LLM to choose among compact legacy frontier summaries."""

        if self.llm_client is None or self.prompt_manager is None or self.config is None:
            return self._select_heuristically(
                candidate_entries,
                candidate_summaries,
                StateSelectionMode.FALLBACK_HEURISTIC,
                fallback_reason="LLM-assisted legacy selection requested without full selector dependencies.",
            )

        prompt_snapshot = "\n".join(summary.to_prompt_line() for summary in candidate_summaries)
        try:
            response = self.llm_client.score_state_for_revisit(
                prompt_manager=self.prompt_manager,
                game_id=self.config.experiment.game_id,
                frontier_snapshot=prompt_snapshot,
                model=self.config.llm.model_name,
                temperature=min(self.config.llm.temperature, 0.2),
                max_tokens=(
                    self.config.llm.state_selection_debug_max_tokens
                    if self.config.llm.analysis_debug_mode
                    else self.config.llm.state_selection_max_tokens
                ),
                timeout_seconds=self.config.llm.request_timeout_seconds,
                analysis_debug_mode=self.config.llm.analysis_debug_mode,
                response_format=(
                    None
                    if self.config.llm.analysis_debug_mode
                    else _state_selection_response_format()
                ),
            )
        except Exception as exc:
            _LOGGER.warning("Legacy state selection fell back after LLM request failure: %s", exc)
            return self._select_heuristically(
                candidate_entries,
                candidate_summaries,
                StateSelectionMode.FALLBACK_HEURISTIC,
                fallback_reason=f"LLM request failed: {exc}",
                prompt_snapshot=prompt_snapshot,
            )

        selected_summary = self._parse_llm_selection(response.text, candidate_summaries)
        if selected_summary is None:
            _LOGGER.warning("Legacy state selection fell back after malformed LLM output: %r", response.text)
            return self._select_heuristically(
                candidate_entries,
                candidate_summaries,
                StateSelectionMode.FALLBACK_HEURISTIC,
                fallback_reason="LLM output did not identify a valid candidate.",
                raw_output=response.text,
                prompt_snapshot=prompt_snapshot,
                model_name=response.model,
            )

        selected = candidate_entries[candidate_summaries.index(selected_summary)]
        return StateSelectionResult(
            selected=selected,
            reason=self._extract_llm_reason(response.text, selected_summary),
            selection_mode=StateSelectionMode.LLM_ASSISTED,
            candidate_summaries=candidate_summaries,
            raw_output=response.text,
            prompt_snapshot=prompt_snapshot,
            model_name=response.model,
        )

    def _build_candidate_summaries(self, entries: list[FrontierEntry]) -> list[FrontierCandidateSummary]:
        """Build compact legacy frontier summaries for LLM-assisted comparison."""

        summaries: list[FrontierCandidateSummary] = []
        for index, entry in enumerate(entries, start=1):
            summary_text = entry.summary_text.strip() or self.summary_builder.summarize_state_candidate(
                observation=entry.observation,
                score=entry.score,
                depth=entry.depth,
                recent_gain=entry.recent_gain,
                inventory_text=entry.inventory_text,
                valid_actions=entry.valid_actions,
            )
            summaries.append(
                FrontierCandidateSummary(
                    candidate_id=f"C{index}",
                    state_id=entry.state_id,
                    score=float(entry.score),
                    depth=int(entry.depth),
                    novelty=float(entry.novelty),
                    recent_gain=float(entry.recent_gain),
                    summary_text=self._compact_text(summary_text),
                    has_native_snapshot=bool(entry.saved_node and entry.saved_node.has_native_state),
                )
            )
        return summaries

    def _compact_text(self, text: str) -> str:
        """Normalize and truncate candidate summaries to keep prompts bounded."""

        normalized = " ".join(text.split()).strip()
        if len(normalized) <= self.summary_max_chars:
            return normalized
        return normalized[: self.summary_max_chars - 3].rstrip() + "..."

    def _resolve_mode(
        self,
        mode: StateSelectionMode | str | None,
        *,
        default_mode: StateSelectionMode,
    ) -> StateSelectionMode:
        """Resolve the requested selection mode for this call."""

        if mode is None:
            return default_mode
        return StateSelectionMode(mode)

    def _parse_llm_selection(
        self,
        raw_output: str,
        candidate_summaries: Sequence[ArchiveCandidateSummary | FrontierCandidateSummary],
    ) -> ArchiveCandidateSummary | FrontierCandidateSummary | None:
        """Parse an LLM response and map it back onto one summarized candidate."""

        structured_payload = self._parse_structured_selection_output(raw_output)
        if structured_payload is not None:
            choice = str(structured_payload.get("choice", "")).strip().lower()
            if choice:
                summary_by_id = {summary.candidate_id.lower(): summary for summary in candidate_summaries}
                if choice in summary_by_id:
                    return summary_by_id[choice]
                if choice.startswith("c") and choice[1:].isdigit():
                    index = int(choice[1:]) - 1
                    if 0 <= index < len(candidate_summaries):
                        return candidate_summaries[index]

        answer_text = self._extract_terminal_selection_answer(raw_output)
        if not answer_text:
            return None

        summary_by_id = {summary.candidate_id.lower(): summary for summary in candidate_summaries}
        explicit_choice_match = re.search(
            r"\b(?:choice|candidate|pick|select(?:ed)?|option|final answer)\s*[:#=-]?\s*(c\d+|\d+)\b",
            answer_text,
            flags=re.IGNORECASE,
        )
        if explicit_choice_match:
            raw_choice = explicit_choice_match.group(1).lower()
            if raw_choice.startswith("c") and raw_choice in summary_by_id:
                return summary_by_id[raw_choice]
            if raw_choice.isdigit():
                index = int(raw_choice) - 1
                if 0 <= index < len(candidate_summaries):
                    return candidate_summaries[index]

        for summary in candidate_summaries:
            if summary.state_id and summary.state_id in answer_text:
                return summary

        standalone_candidate_match = re.match(r"^\s*(c\d+)\b", answer_text, flags=re.IGNORECASE)
        if standalone_candidate_match:
            candidate_id = standalone_candidate_match.group(1).lower()
            if candidate_id in summary_by_id:
                return summary_by_id[candidate_id]

        return None

    def _extract_llm_reason(
        self,
        raw_output: str,
        selected_summary: ArchiveCandidateSummary | FrontierCandidateSummary,
    ) -> str:
        """Extract a short human-readable reason from an LLM response."""

        structured_payload = self._parse_structured_selection_output(raw_output)
        if structured_payload is not None:
            structured_reason = str(structured_payload.get("reason", "")).strip()
            if structured_reason:
                return " ".join(structured_reason.split()).strip()

        answer_text = self._extract_terminal_selection_answer(raw_output)
        if not answer_text:
            return f"Selected {selected_summary.candidate_id}."

        reason_match = re.search(r"reason\s*[:=-]\s*(.+)", answer_text, flags=re.IGNORECASE | re.DOTALL)
        if reason_match:
            return " ".join(reason_match.group(1).split()).strip()

        inline_reason_match = re.search(
            r"\b(?:choice|candidate|pick|select(?:ed)?|option)\s*[:#=-]?\s*(?:c\d+|\d+|archive:\S+)\s*[-–—:]\s*(.+)",
            answer_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if inline_reason_match:
            return " ".join(inline_reason_match.group(1).split()).strip()

        lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
        for line in lines:
            if selected_summary.candidate_id.lower() in line.lower():
                return line

        if len(lines) >= 2:
            return lines[1]

        return " ".join(answer_text.split()).strip() or f"Selected {selected_summary.candidate_id}."

    def _extract_terminal_selection_answer(self, raw_output: str) -> str:
        """Extract the final answer block from an LLM selection response.

        The selector should not trust candidate ids or `reason:` text that appear
        inside a model's visible chain-of-thought. We therefore only parse the
        last explicit answer-shaped lines near the end of the completion.
        """

        normalized_output = raw_output.strip()
        if not normalized_output:
            return ""

        lines = [line.strip() for line in normalized_output.splitlines() if line.strip()]
        if not lines:
            return ""

        answer_line_indexes = [
            index
            for index, line in enumerate(lines)
            if re.match(
                r"^(?:choice|candidate|pick|select(?:ed)?|option|final answer)\s*[:#=-]",
                line,
                flags=re.IGNORECASE,
            )
            or re.match(r"^c\d+\b", line, flags=re.IGNORECASE)
        ]
        if not answer_line_indexes:
            return ""

        start_index = answer_line_indexes[-1]
        selected_lines = [lines[start_index]]
        for line in lines[start_index + 1 :]:
            if re.match(r"^reason\s*[:=-]", line, flags=re.IGNORECASE):
                selected_lines.append(line)
                continue
            if re.match(
                r"^(?:choice|candidate|pick|select(?:ed)?|option|final answer)\s*[:#=-]",
                line,
                flags=re.IGNORECASE,
            ):
                break
            if selected_lines and len(selected_lines) == 1:
                break
        return "\n".join(selected_lines).strip()

    def _parse_structured_selection_output(self, raw_output: str) -> dict[str, Any] | None:
        """Parse strict or debug-mode structured selection output if present."""

        normalized_output = raw_output.strip()
        if not normalized_output:
            return None

        candidates = [normalized_output]
        block_match = re.search(
            r"BEGIN_STRUCTURED_OUTPUT\s*(\{.*?\})\s*END_STRUCTURED_OUTPUT",
            normalized_output,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if block_match:
            candidates.insert(0, block_match.group(1))

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", normalized_output, flags=re.DOTALL | re.IGNORECASE)
        if fenced_match:
            candidates.insert(0, fenced_match.group(1))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(parsed, dict):
                continue
            if "choice" in parsed and "reason" in parsed:
                return parsed
        return None


def _state_selection_response_format() -> dict[str, object]:
    """Return a strict JSON schema for archive/state selection."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "state_selection",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "choice": {
                        "type": "string",
                        "pattern": "^C[0-9]+$",
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
                "required": ["choice", "reason"],
            },
        },
    }

    def _default_selection_id(self) -> str:
        """Return a timestamped archive-selection artifact id."""

        return f"state-selection-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
