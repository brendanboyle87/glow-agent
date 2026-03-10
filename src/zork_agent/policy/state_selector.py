"""State-selection logic for choosing which frontier node to revisit next.

The heuristic ranking remains the deterministic baseline. The optional LLM path
only chooses among a compact set of already-ranked frontier candidates, so it is
an assistive policy layer rather than a replacement for explicit frontier scoring.

TODO: revisit prompt/parse heuristics after collecting real Zork frontier traces.
"""

from __future__ import annotations

import logging
import re

from zork_agent.config import ProjectConfig
from zork_agent.llm.base import BaseLLMClient
from zork_agent.llm.prompts import PromptManager
from zork_agent.memory.frontier import FrontierEntry, FrontierQueue
from zork_agent.memory.summaries import SummaryBuilder
from zork_agent.types import (
    FrontierCandidateSummary,
    StateSelectionMode,
    StateSelectionResult,
)

_LOGGER = logging.getLogger("zork_agent.state_selector")


class StateSelector:
    """Choose which previously discovered frontier state should be revisited."""

    def __init__(
        self,
        *,
        config: ProjectConfig | None = None,
        prompt_manager: PromptManager | None = None,
        llm_client: BaseLLMClient | None = None,
        selection_mode: StateSelectionMode | str = StateSelectionMode.HEURISTIC,
        prompt_candidate_limit: int = 6,
        summary_max_chars: int = 120,
    ):
        # TODO: move selector-specific knobs into config only if experimentation demands it.
        self.config = config
        self.prompt_manager = prompt_manager
        self.llm_client = llm_client
        self.selection_mode = StateSelectionMode(selection_mode)
        self.prompt_candidate_limit = max(1, prompt_candidate_limit)
        self.summary_max_chars = max(32, summary_max_chars)
        self.summary_builder = SummaryBuilder()
        self.last_result: StateSelectionResult | None = None

    def select(
        self,
        frontier: FrontierQueue,
        *,
        mode: StateSelectionMode | str | None = None,
    ) -> FrontierEntry | None:
        """Return the selected frontier entry only."""

        result = self.select_with_details(frontier, mode=mode)
        self.last_result = result
        return result.selected

    def select_with_details(
        self,
        frontier: FrontierQueue,
        *,
        mode: StateSelectionMode | str | None = None,
    ) -> StateSelectionResult:
        """Return the selected frontier entry plus the selection rationale."""

        candidate_entries = frontier.top_k(self.prompt_candidate_limit)
        candidate_summaries = self._build_candidate_summaries(candidate_entries)
        effective_mode = self._resolve_mode(mode)

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
            result = self._select_with_llm(candidate_entries, candidate_summaries)
        else:
            result = self._select_heuristically(candidate_entries, candidate_summaries, effective_mode)

        self.last_result = result
        return result

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
        """Pick the current best frontier entry according to the explicit ranking."""

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

    def _select_with_llm(
        self,
        candidate_entries: list[FrontierEntry],
        candidate_summaries: list[FrontierCandidateSummary],
    ) -> StateSelectionResult:
        """Ask the LLM to choose among compact frontier summaries."""

        if self.llm_client is None or self.prompt_manager is None or self.config is None:
            return self._select_heuristically(
                candidate_entries,
                candidate_summaries,
                StateSelectionMode.FALLBACK_HEURISTIC,
                fallback_reason="LLM-assisted selection requested without full selector dependencies.",
            )

        prompt_snapshot = "\n".join(summary.to_prompt_line() for summary in candidate_summaries)
        try:
            response = self.llm_client.score_state_for_revisit(
                prompt_manager=self.prompt_manager,
                game_id=self.config.experiment.game_id,
                frontier_snapshot=prompt_snapshot,
                model=self.config.llm.model_name,
                temperature=min(self.config.llm.temperature, 0.2),
                max_tokens=min(self.config.llm.max_tokens, 128),
                timeout_seconds=self.config.llm.request_timeout_seconds,
            )
        except Exception as exc:
            _LOGGER.warning("State selection fell back after LLM request failure: %s", exc)
            return self._select_heuristically(
                candidate_entries,
                candidate_summaries,
                StateSelectionMode.FALLBACK_HEURISTIC,
                fallback_reason=f"LLM request failed: {exc}",
                prompt_snapshot=prompt_snapshot,
            )

        selected_summary = self._parse_llm_selection(response.text, candidate_summaries)
        if selected_summary is None:
            _LOGGER.warning("State selection fell back after malformed LLM output: %r", response.text)
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
        """Build compact frontier summaries for LLM-assisted comparison."""

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

    def _resolve_mode(self, mode: StateSelectionMode | str | None) -> StateSelectionMode:
        """Resolve the requested selection mode for this call."""

        if mode is None:
            return self.selection_mode
        return StateSelectionMode(mode)

    def _parse_llm_selection(
        self,
        raw_output: str,
        candidate_summaries: list[FrontierCandidateSummary],
    ) -> FrontierCandidateSummary | None:
        """Parse an LLM response and map it back onto one summarized candidate."""

        normalized_output = raw_output.strip()
        if not normalized_output:
            return None

        summary_by_id = {summary.candidate_id.lower(): summary for summary in candidate_summaries}
        for candidate_id, summary in summary_by_id.items():
            if re.search(rf"\b{re.escape(candidate_id)}\b", normalized_output, flags=re.IGNORECASE):
                return summary

        for summary in candidate_summaries:
            if summary.state_id and summary.state_id in normalized_output:
                return summary

        indexed_match = re.search(
            r"\b(?:choice|candidate|pick|select|selected|option)\s*[:#-]?\s*(\d+)\b",
            normalized_output,
            flags=re.IGNORECASE,
        )
        if indexed_match:
            index = int(indexed_match.group(1)) - 1
            if 0 <= index < len(candidate_summaries):
                return candidate_summaries[index]

        leading_match = re.match(r"^\s*(?:#\s*)?(\d+)\b", normalized_output)
        if leading_match:
            index = int(leading_match.group(1)) - 1
            if 0 <= index < len(candidate_summaries):
                return candidate_summaries[index]

        return None

    def _extract_llm_reason(self, raw_output: str, selected_summary: FrontierCandidateSummary) -> str:
        """Extract a short human-readable reason from an LLM response."""

        reason_match = re.search(r"reason\s*[:=-]\s*(.+)", raw_output, flags=re.IGNORECASE | re.DOTALL)
        if reason_match:
            return " ".join(reason_match.group(1).split()).strip()

        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        for line in lines:
            if selected_summary.candidate_id.lower() in line.lower():
                return line

        return " ".join(raw_output.split()).strip() or f"Selected {selected_summary.candidate_id}."
