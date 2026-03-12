"""Evidence-ranked parser action generation for Jericho-backed interactive fiction.

This module intentionally treats Jericho valid actions as the primary constrained-mode
candidate pool. The LLM provides a soft ordering hint, but the final ranking is driven
by explicit evidence features so the agent can stay object-centric without relying on
model quality alone.

TODO: revisit the weight defaults after collecting a few longer live traces.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Sequence

from zork_agent.config import ProjectConfig
from zork_agent.llm.base import BaseLLMClient
from zork_agent.llm.prompts import PromptManager
from zork_agent.types import (
    ActionCandidateFeatures,
    ActionClusterHistory,
    ActionGenerationMode,
    ActionGenerationResult,
    StrategicMode,
    ActionProposal,
    LocalWorldModel,
    LocalWorldModelPromptSummary,
    LoopHeuristicResult,
    TextGameState,
    action_target_tokens,
    action_object_count,
    action_shape_is_complex_transitive,
    action_shape_is_simple,
    action_uses_inventory_object,
    actions_are_inverse,
    canonical_if_verb_family,
    count_repeated_movement_actions,
    estimate_candidate_loop_penalty,
    extract_salient_nouns,
    inventory_item_tokens,
    is_bulk_inventory_action,
    is_discard_like_action,
    is_movement_action,
    normalize_parser_action,
    split_action_command,
    target_is_container_or_openable_candidate,
    target_is_readable_candidate,
)

_LOGGER = logging.getLogger("zork_agent.action_generator")

_OBJECT_PRIORITY_VERBS = {"take", "get", "read", "open", "examine", "enter", "look in", "look at"}
_OBJECT_PRIORITY_PREFIXES = ("look in ", "look inside ", "look at ")
_LOW_VALUE_GENERIC_ACTIONS = {
    "dance",
    "inventory",
    "jump",
    "look",
    "pray",
    "quit",
    "restart",
    "restore",
    "save",
    "shout",
    "sing",
    "sleep",
    "undo",
    "wait",
    "yell",
}
_CANONICAL_HIGH_VALUE_FAMILIES = {"inspect", "acquire", "access"}
_SPECULATIVE_TRANSITIVE_PREFIXES = ("throw ", "use ", "put ", "insert ", "attack ", "break ", "kick ")
_CLIMBABLE_LANDMARK_TOKENS = {"branch", "branches", "leaves", "tree", "trees"}
_LANDMARK_ENTRY_TOKENS = {"door", "gate", "house", "trapdoor", "window"}


class ActionGenerator:
    """Generate compact, evidence-ranked parser actions."""

    def __init__(self, config: ProjectConfig, prompt_manager: PromptManager, llm_client: BaseLLMClient | None):
        self.config = config
        self.prompt_manager = prompt_manager
        self.llm_client = llm_client
        self.last_result: ActionGenerationResult | None = None

    def propose_actions(
        self,
        state: TextGameState,
        *,
        recent_trajectory_context: str | None = None,
        candidate_count: int | None = None,
        temperature: float | None = None,
        recent_actions: list[str] | None = None,
        recent_loop_results: list[LoopHeuristicResult] | None = None,
        state_action_history: ActionClusterHistory | None = None,
        supported_try_actions: list[str] | None = None,
        supported_avoid_actions: list[str] | None = None,
        supported_reflection_objects: list[str] | None = None,
        root_state_id: str = "",
        local_world_model: LocalWorldModel | None = None,
        strategic_mode: StrategicMode | None = None,
        strategic_reason: str = "",
        strategic_try_actions: list[str] | None = None,
        strategic_avoid_actions: list[str] | None = None,
        strategic_objects: list[str] | None = None,
    ) -> list[ActionProposal]:
        """Return a ranked candidate list for the current state."""

        result = self.generate(
            observation=state.observation,
            inventory_text=state.inventory_text,
            recent_trajectory_context=recent_trajectory_context,
            valid_actions=state.valid_actions,
            score=state.score,
            moves=state.moves,
            candidate_count=candidate_count,
            temperature=temperature,
            recent_actions=recent_actions,
            recent_loop_results=recent_loop_results,
            state_action_history=state_action_history,
            supported_try_actions=supported_try_actions,
            supported_avoid_actions=supported_avoid_actions,
            supported_reflection_objects=supported_reflection_objects,
            root_state_id=root_state_id,
            local_world_model=local_world_model,
            strategic_mode=strategic_mode,
            strategic_reason=strategic_reason,
            strategic_try_actions=strategic_try_actions,
            strategic_avoid_actions=strategic_avoid_actions,
            strategic_objects=strategic_objects,
        )
        self.last_result = result
        return result.candidates

    def generate(
        self,
        *,
        observation: str,
        inventory_text: str | None = None,
        recent_trajectory_context: str | None = None,
        valid_actions: list[str] | None = None,
        score: int = 0,
        moves: int = 0,
        candidate_count: int | None = None,
        temperature: float | None = None,
        recent_actions: list[str] | None = None,
        recent_loop_results: list[LoopHeuristicResult] | None = None,
        state_action_history: ActionClusterHistory | None = None,
        supported_try_actions: list[str] | None = None,
        supported_avoid_actions: list[str] | None = None,
        supported_reflection_objects: list[str] | None = None,
        root_state_id: str = "",
        local_world_model: LocalWorldModel | None = None,
        strategic_mode: StrategicMode | None = None,
        strategic_reason: str = "",
        strategic_try_actions: list[str] | None = None,
        strategic_avoid_actions: list[str] | None = None,
        strategic_objects: list[str] | None = None,
    ) -> ActionGenerationResult:
        """Generate candidates in either constrained or open mode."""

        inventory_text = inventory_text or ""
        valid_actions = [action for action in (valid_actions or []) if normalize_parser_action(action)]
        candidate_count = candidate_count or self.config.policy.action_candidates
        recent_actions = list(recent_actions or [])
        recent_loop_results = list(recent_loop_results or [])
        external_supported_try_actions = self._normalize_action_list(supported_try_actions or [])
        external_supported_avoid_actions = self._normalize_action_list(supported_avoid_actions or [])
        external_supported_reflection_objects = sorted(self._normalize_token_list(supported_reflection_objects or []))
        local_world_model_summary, local_model_try_actions, local_model_avoid_actions, local_model_objects = (
            self._local_world_model_context(
                root_state_id=root_state_id,
                local_world_model=local_world_model,
            )
        )
        supported_try_actions = self._normalize_action_list(
            [*external_supported_try_actions, *local_model_try_actions]
        )
        supported_avoid_actions = self._normalize_action_list(
            [*external_supported_avoid_actions, *local_model_avoid_actions]
        )
        supported_reflection_objects = self._normalize_token_list(
            [*external_supported_reflection_objects, *local_model_objects]
        )
        strategic_try_actions = self._normalize_action_list(strategic_try_actions or [])
        strategic_avoid_actions = self._normalize_action_list(strategic_avoid_actions or [])
        strategic_objects = self._normalize_token_list(strategic_objects or [])
        strategic_mode = strategic_mode or StrategicMode.EXPLORE
        local_guidance_available = bool(
            local_world_model_summary.has_guidance()
            or local_model_try_actions
            or local_model_avoid_actions
            or local_model_objects
        )
        guidance_metadata = {
            "root_state_id": root_state_id,
            "local_world_model_present": local_world_model is not None,
            "local_world_model_no_achievement_revisit_streak": (
                int(local_world_model.metadata.get("no_achievement_revisit_streak", 0))
                if local_world_model is not None
                else 0
            ),
            "local_world_model_nonproductive_revisit_streak": (
                int(local_world_model.metadata.get("nonproductive_revisit_streak", 0))
                if local_world_model is not None
                else 0
            ),
            "local_world_model_summary_has_guidance": local_world_model_summary.has_guidance(),
            "local_world_model_hint_count": (
                len(local_world_model.accumulated_advantage_hints)
                if local_world_model is not None
                else 0
            ),
            "local_world_model_prior_count": (
                len(local_world_model.action_priors)
                if local_world_model is not None
                else 0
            ),
            "local_world_model_antiprior_count": (
                len(local_world_model.action_antipriors)
                if local_world_model is not None
                else 0
            ),
            "local_world_model_affordance_count": (
                len(local_world_model.inferred_affordances)
                if local_world_model is not None
                else 0
            ),
            "local_model_try_actions": list(local_model_try_actions),
            "local_model_avoid_actions": list(local_model_avoid_actions),
            "local_model_objects": list(local_model_objects),
            "local_model_try_action_count": len(local_model_try_actions),
            "local_model_avoid_action_count": len(local_model_avoid_actions),
            "local_model_object_count": len(local_model_objects),
            "external_supported_try_actions": list(external_supported_try_actions),
            "external_supported_avoid_actions": list(external_supported_avoid_actions),
            "external_supported_reflection_objects": list(external_supported_reflection_objects),
            "external_supported_try_action_count": len(external_supported_try_actions),
            "external_supported_avoid_action_count": len(external_supported_avoid_actions),
            "external_supported_object_count": len(external_supported_reflection_objects),
            "merged_supported_try_actions": list(supported_try_actions),
            "merged_supported_avoid_actions": list(supported_avoid_actions),
            "merged_supported_reflection_objects": sorted(supported_reflection_objects),
            "merged_supported_try_action_count": len(supported_try_actions),
            "merged_supported_avoid_action_count": len(supported_avoid_actions),
            "merged_supported_object_count": len(supported_reflection_objects),
            "local_guidance_available": local_guidance_available,
            "local_guidance_attached_to_policy_input": local_guidance_available,
            "local_guidance_attached_to_prompt": bool(self.llm_client is not None and local_guidance_available),
            "local_guidance_consumed_by_reranker": local_guidance_available,
            "local_world_model_try_guidance_suppressed": bool(
                local_world_model is not None
                and (
                    int(local_world_model.metadata.get("no_achievement_revisit_streak", 0))
                    >= self.config.policy.local_guidance_stale_revisit_threshold
                    or (
                        bool(local_world_model.metadata.get("depth_progression_productive_root", False))
                        and int(local_world_model.metadata.get("depth_progression_replay_saturation_level", 0)) > 0
                        and bool(local_world_model.metadata.get("depth_progression_saturated_try_actions", []))
                    )
                )
            ),
            "local_world_model_depth_progression_replay_saturation_level": (
                int(local_world_model.metadata.get("depth_progression_replay_saturation_level", 0))
                if local_world_model is not None
                else 0
            ),
        }

        if valid_actions:
            result = self._generate_constrained(
                observation=observation,
                inventory_text=inventory_text,
                valid_actions=valid_actions,
                score=score,
                moves=moves,
                candidate_count=candidate_count,
                temperature=temperature,
                recent_trajectory_context=recent_trajectory_context or "",
                recent_actions=recent_actions,
                recent_loop_results=recent_loop_results,
                state_action_history=state_action_history,
                supported_try_actions=supported_try_actions,
                supported_avoid_actions=supported_avoid_actions,
                supported_reflection_objects=supported_reflection_objects,
                root_state_id=root_state_id,
                local_world_model_summary=local_world_model_summary,
                strategic_mode=strategic_mode,
                strategic_reason=strategic_reason,
                strategic_try_actions=strategic_try_actions,
                strategic_avoid_actions=strategic_avoid_actions,
                strategic_objects=strategic_objects,
            )
        else:
            result = self._generate_open(
                observation=observation,
                inventory_text=inventory_text,
                score=score,
                moves=moves,
                candidate_count=candidate_count,
                temperature=temperature,
                recent_trajectory_context=recent_trajectory_context or "",
                recent_actions=recent_actions,
                recent_loop_results=recent_loop_results,
                state_action_history=state_action_history,
                supported_try_actions=supported_try_actions,
                supported_avoid_actions=supported_avoid_actions,
                supported_reflection_objects=supported_reflection_objects,
                root_state_id=root_state_id,
                local_world_model_summary=local_world_model_summary,
                strategic_mode=strategic_mode,
                strategic_reason=strategic_reason,
                strategic_try_actions=strategic_try_actions,
                strategic_avoid_actions=strategic_avoid_actions,
                strategic_objects=strategic_objects,
            )

        result.metadata.update(guidance_metadata)
        self.last_result = result
        return result

    def _generate_constrained(
        self,
        *,
        observation: str,
        inventory_text: str,
        valid_actions: list[str],
        score: int,
        moves: int,
        candidate_count: int,
        temperature: float | None,
        recent_trajectory_context: str,
        recent_actions: list[str],
        recent_loop_results: list[LoopHeuristicResult],
        state_action_history: ActionClusterHistory | None,
        supported_try_actions: list[str],
        supported_avoid_actions: list[str],
        supported_reflection_objects: list[str],
        root_state_id: str,
        local_world_model_summary: LocalWorldModelPromptSummary,
        strategic_mode: StrategicMode,
        strategic_reason: str,
        strategic_try_actions: list[str],
        strategic_avoid_actions: list[str],
        strategic_objects: list[str],
    ) -> ActionGenerationResult:
        """Rank the Jericho valid-action pool with LLM ordering hints plus evidence features."""

        llm_ranked_candidates: list[ActionProposal] = []
        raw_output = ""
        model_name: str | None = None
        result_mode = ActionGenerationMode.FALLBACK
        ranking_source = "heuristic_rerank"
        fallback_reason = ""

        if self.llm_client is not None:
            raw_output, model_name, llm_ranked_candidates = self._request_constrained_ranking(
                observation=observation,
                inventory_text=inventory_text,
                valid_actions=valid_actions,
                score=score,
                moves=moves,
                candidate_count=candidate_count,
                temperature=temperature,
                recent_trajectory_context=recent_trajectory_context,
                supported_try_actions=supported_try_actions,
                supported_avoid_actions=supported_avoid_actions,
                supported_reflection_objects=supported_reflection_objects,
                root_state_id=root_state_id,
                local_world_model_summary=local_world_model_summary,
                strategic_mode=strategic_mode,
                strategic_reason=strategic_reason,
                strategic_try_actions=strategic_try_actions,
                strategic_avoid_actions=strategic_avoid_actions,
                strategic_objects=strategic_objects,
            )
            if llm_ranked_candidates:
                result_mode = ActionGenerationMode.CONSTRAINED
                ranking_source = "hybrid_merge"
            elif raw_output:
                fallback_reason = "LLM output was unusable."
        else:
            fallback_reason = "No LLM client configured."

        llm_rank_map = {
            normalize_parser_action(candidate.action): rank
            for rank, candidate in enumerate(llm_ranked_candidates, start=1)
        }
        llm_metadata = {
            normalize_parser_action(candidate.action): candidate
            for candidate in llm_ranked_candidates
        }
        heuristic_seed_order = self._fallback_valid_actions(observation=observation, valid_actions=valid_actions)
        candidate_pool = self._merge_candidate_pool(
            valid_actions,
            heuristic_seed_order,
            [candidate.action for candidate in llm_ranked_candidates],
        )
        proposals = [
            ActionProposal(
                action=action,
                source=(
                    "llm_constrained"
                    if normalize_parser_action(action) in llm_rank_map
                    else "fallback_constrained"
                ),
                rank=llm_rank_map.get(normalize_parser_action(action)),
                confidence=(
                    llm_metadata[normalize_parser_action(action)].confidence
                    if normalize_parser_action(action) in llm_metadata
                    else None
                ),
                rationale=(
                    llm_metadata[normalize_parser_action(action)].rationale
                    if normalize_parser_action(action) in llm_metadata
                    else ""
                ),
                raw_line=(
                    llm_metadata[normalize_parser_action(action)].raw_line
                    if normalize_parser_action(action) in llm_metadata
                    else ""
                ),
            )
            for action in candidate_pool
        ]
        result = ActionGenerationResult(
            mode=result_mode,
            candidates=proposals,
            raw_output=raw_output,
            model_name=model_name,
            valid_actions=list(valid_actions),
            fallback_reason=fallback_reason,
            candidate_pool_before_rerank=list(candidate_pool),
            llm_ranked_actions=[candidate.action for candidate in llm_ranked_candidates],
            ranking_source=ranking_source,
            augmented_with_valid_actions=True,
        )
        self._rerank_candidates_evidence(
            result,
            observation=observation,
            inventory_text=inventory_text,
            recent_actions=recent_actions,
            recent_loop_results=recent_loop_results,
            state_action_history=state_action_history,
            supported_try_actions=supported_try_actions,
            supported_avoid_actions=supported_avoid_actions,
            supported_reflection_objects=supported_reflection_objects,
            strategic_mode=strategic_mode,
            strategic_reason=strategic_reason,
            strategic_try_actions=strategic_try_actions,
            strategic_avoid_actions=strategic_avoid_actions,
            strategic_objects=strategic_objects,
            candidate_count=candidate_count,
        )
        result.strategic_mode = strategic_mode.value
        result.strategic_reason = strategic_reason
        return result

    def _generate_open(
        self,
        *,
        observation: str,
        inventory_text: str,
        score: int,
        moves: int,
        candidate_count: int,
        temperature: float | None,
        recent_trajectory_context: str,
        recent_actions: list[str],
        recent_loop_results: list[LoopHeuristicResult],
        state_action_history: ActionClusterHistory | None,
        supported_try_actions: list[str],
        supported_avoid_actions: list[str],
        supported_reflection_objects: list[str],
        root_state_id: str,
        local_world_model_summary: LocalWorldModelPromptSummary,
        strategic_mode: StrategicMode,
        strategic_reason: str,
        strategic_try_actions: list[str],
        strategic_avoid_actions: list[str],
        strategic_objects: list[str],
    ) -> ActionGenerationResult:
        """Generate free-form actions, then rerank them with the same evidence scorer."""

        raw_output = ""
        model_name: str | None = None
        ranking_source = "heuristic_rerank"
        parsed_candidates: list[ActionProposal] = []

        if self.llm_client is not None:
            try:
                system_prompt = self.prompt_manager.render_system(game_id=self.config.experiment.game_id)
                user_prompt = self.prompt_manager.render_action_proposal(
                    observation=observation,
                    inventory=inventory_text or "Inventory unavailable.",
                    score=score,
                    moves=moves,
                    action_candidates=candidate_count,
                    generation_mode=ActionGenerationMode.OPEN.value,
                    recent_trajectory_context=recent_trajectory_context,
                    valid_actions=[],
                    salient_objects=sorted(
                        extract_salient_nouns(
                            observation=observation,
                            inventory_text=inventory_text,
                            valid_actions=[],
                            inverse_pairs=self.config.policy.inverse_action_pairs,
                        )
                    )[:6],
                    supported_try_actions=supported_try_actions,
                    supported_avoid_actions=supported_avoid_actions,
                    root_state_id=root_state_id,
                    local_world_model_summary=local_world_model_summary,
                    strategic_mode=strategic_mode.value,
                    strategic_reason=strategic_reason,
                    strategic_try_actions=strategic_try_actions,
                    strategic_avoid_actions=strategic_avoid_actions,
                    strategic_objects=strategic_objects,
                )
                response = self.llm_client.chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=self.config.llm.model_name,
                    temperature=self.config.llm.temperature if temperature is None else temperature,
                    max_tokens=self.config.llm.max_tokens,
                    timeout_seconds=self.config.llm.request_timeout_seconds,
                    metadata={"task": "action_generation_open"},
                )
                raw_output = response.text
                model_name = response.model
                parsed_candidates = self._parse_candidates(
                    raw_output=response.text,
                    mode=ActionGenerationMode.OPEN,
                    valid_actions=[],
                    candidate_count=max(candidate_count * 2, candidate_count),
                )
                if parsed_candidates:
                    ranking_source = "hybrid_merge"
            except Exception as exc:
                _LOGGER.warning("Action generation fell back after LLM request failure: %s", exc)
                raw_output = ""

        if not parsed_candidates:
            fallback_actions = self._fallback_open_actions(observation=observation, inventory_text=inventory_text)
            parsed_candidates = [
                ActionProposal(
                    action=action,
                    rationale="LLM output was unusable." if raw_output else "No LLM client configured.",
                    source="fallback_open",
                    rank=index + 1,
                )
                for index, action in enumerate(fallback_actions)
            ]
            result_mode = ActionGenerationMode.FALLBACK
        else:
            result_mode = ActionGenerationMode.OPEN

        result = ActionGenerationResult(
            mode=result_mode,
            candidates=parsed_candidates,
            raw_output=raw_output,
            model_name=model_name,
            fallback_reason="" if parsed_candidates else "LLM output was unusable.",
            candidate_pool_before_rerank=[candidate.action for candidate in parsed_candidates],
            llm_ranked_actions=[candidate.action for candidate in parsed_candidates if candidate.raw_line],
            ranking_source=ranking_source,
        )
        self._rerank_candidates_evidence(
            result,
            observation=observation,
            inventory_text=inventory_text,
            recent_actions=recent_actions,
            recent_loop_results=recent_loop_results,
            state_action_history=state_action_history,
            supported_try_actions=supported_try_actions,
            supported_avoid_actions=supported_avoid_actions,
            supported_reflection_objects=supported_reflection_objects,
            strategic_mode=strategic_mode,
            strategic_reason=strategic_reason,
            strategic_try_actions=strategic_try_actions,
            strategic_avoid_actions=strategic_avoid_actions,
            strategic_objects=strategic_objects,
            candidate_count=candidate_count,
        )
        result.strategic_mode = strategic_mode.value
        result.strategic_reason = strategic_reason
        return result

    def _request_constrained_ranking(
        self,
        *,
        observation: str,
        inventory_text: str,
        valid_actions: list[str],
        score: int,
        moves: int,
        candidate_count: int,
        temperature: float | None,
        recent_trajectory_context: str,
        supported_try_actions: list[str],
        supported_avoid_actions: list[str],
        supported_reflection_objects: list[str],
        root_state_id: str,
        local_world_model_summary: LocalWorldModelPromptSummary,
        strategic_mode: StrategicMode,
        strategic_reason: str,
        strategic_try_actions: list[str],
        strategic_avoid_actions: list[str],
        strategic_objects: list[str],
    ) -> tuple[str, str | None, list[ActionProposal]]:
        """Ask the model to rank the current valid-action list."""

        if self.llm_client is None:
            return "", None, []

        salient_objects = sorted(
            extract_salient_nouns(
                observation=observation,
                inventory_text=inventory_text,
                valid_actions=[],
                inverse_pairs=self.config.policy.inverse_action_pairs,
            )
            | set(supported_reflection_objects)
        )
        system_prompt = self.prompt_manager.render_system(game_id=self.config.experiment.game_id)
        user_prompt = self.prompt_manager.render_action_proposal(
            observation=observation,
            inventory=inventory_text or "Inventory unavailable.",
            score=score,
            moves=moves,
            action_candidates=min(max(candidate_count, 4), len(valid_actions)),
            generation_mode=ActionGenerationMode.CONSTRAINED.value,
            recent_trajectory_context=recent_trajectory_context,
            valid_actions=valid_actions,
            salient_objects=salient_objects[:8],
            supported_try_actions=supported_try_actions,
            supported_avoid_actions=supported_avoid_actions,
            root_state_id=root_state_id,
            local_world_model_summary=local_world_model_summary,
            strategic_mode=strategic_mode.value,
            strategic_reason=strategic_reason,
            strategic_try_actions=strategic_try_actions,
            strategic_avoid_actions=strategic_avoid_actions,
            strategic_objects=strategic_objects,
        )
        try:
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.config.llm.model_name,
                temperature=self.config.llm.temperature if temperature is None else temperature,
                max_tokens=self.config.llm.max_tokens,
                timeout_seconds=self.config.llm.request_timeout_seconds,
                metadata={"task": "action_generation_constrained"},
            )
        except Exception as exc:
            _LOGGER.warning("Action generation fell back after LLM request failure: %s", exc)
            return "", None, []

        parsed = self._parse_candidates(
            raw_output=response.text,
            mode=ActionGenerationMode.CONSTRAINED,
            valid_actions=valid_actions,
            candidate_count=len(valid_actions),
        )
        return response.text, response.model, parsed

    def _parse_candidates(
        self,
        *,
        raw_output: str,
        mode: ActionGenerationMode,
        valid_actions: list[str],
        candidate_count: int,
    ) -> list[ActionProposal]:
        """Parse action candidates from imperfect plain-text model output."""

        parsed: list[ActionProposal] = []
        seen_actions: set[str] = set()
        for rank, raw_line in enumerate(self._candidate_lines(raw_output), start=1):
            candidate = self._parse_candidate_line(raw_line=raw_line, mode=mode, valid_actions=valid_actions)
            if candidate is None:
                continue
            normalized = normalize_parser_action(candidate.action)
            if normalized in seen_actions:
                continue
            candidate.rank = rank
            seen_actions.add(normalized)
            parsed.append(candidate)
            if len(parsed) >= candidate_count:
                break
        return parsed

    def _candidate_lines(self, raw_output: str) -> list[str]:
        """Split model output into candidate-like lines."""

        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if len(lines) <= 1 and ";" in raw_output:
            lines = [line.strip() for line in raw_output.split(";") if line.strip()]
        return lines

    def _parse_candidate_line(
        self,
        *,
        raw_line: str,
        mode: ActionGenerationMode,
        valid_actions: list[str],
    ) -> ActionProposal | None:
        """Parse one candidate line into a structured action proposal."""

        cleaned = re.sub(r"^\s*(?:[-*]|\d+[\).:\-])\s*", "", raw_line).strip()
        if not cleaned:
            return None

        confidence = self._extract_confidence(cleaned)
        segments = [segment.strip() for segment in cleaned.split("|") if segment.strip()]
        primary_segment = segments[0] if segments else cleaned

        if mode is ActionGenerationMode.CONSTRAINED:
            action = self._match_valid_action(cleaned, valid_actions)
            if action is None:
                return None
        else:
            action = self._extract_open_action(primary_segment)
            if not action or not self._is_plausible_action(action):
                return None

        rationale = self._extract_rationale(cleaned, action)
        return ActionProposal(
            action=action,
            rationale=rationale,
            source=f"llm_{mode.value}",
            confidence=confidence,
            raw_line=raw_line,
        )

    def _extract_confidence(self, text: str) -> float | None:
        """Extract a confidence value if the model happened to provide one."""

        percent_match = re.search(r"(\d{1,3})\s*%", text)
        if percent_match:
            return round(min(int(percent_match.group(1)), 100) / 100.0, 3)

        decimal_match = re.search(r"(?:confidence\s*[:=]?\s*|\()([01](?:\.\d+)?)\)?", text, flags=re.IGNORECASE)
        if decimal_match:
            return float(decimal_match.group(1))
        return None

    def _match_valid_action(self, text: str, valid_actions: list[str]) -> str | None:
        """Map noisy model text back onto Jericho-provided valid actions."""

        normalized_text = normalize_parser_action(text)
        exact_map = {normalize_parser_action(action): action for action in valid_actions}
        if normalized_text in exact_map:
            return exact_map[normalized_text]

        for action in sorted(valid_actions, key=len, reverse=True):
            normalized_action = normalize_parser_action(action)
            if normalized_action and normalized_action in normalized_text:
                return action
        return None

    def _extract_open_action(self, text: str) -> str:
        """Extract a plausible parser command from a free-form candidate line."""

        stripped = re.sub(r"^(?:action|candidate)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
        for separator in (" | ", " - ", " — ", " because ", " since "):
            if separator in stripped:
                stripped = stripped.split(separator, 1)[0].strip()
                break
        stripped = re.sub(r"\((?:confidence\s*[:=]?\s*)?[01](?:\.\d+)?\)$", "", stripped, flags=re.IGNORECASE)
        return stripped.strip(" .").lower()

    def _extract_rationale(self, text: str, action: str) -> str:
        """Extract any trailing explanation after the action text."""

        lowered = text.lower()
        action_lower = action.lower()
        if "|" in text:
            extras = [segment.strip() for segment in text.split("|")[1:] if segment.strip()]
            filtered = [segment for segment in extras if "confidence" not in segment.lower()]
            return " | ".join(filtered)

        if action_lower in lowered:
            suffix = text[lowered.find(action_lower) + len(action_lower) :].strip(" -:|")
            if suffix and "confidence" not in suffix.lower():
                return suffix
        return ""

    def _fallback_valid_actions(self, *, observation: str, valid_actions: list[str]) -> list[str]:
        """Prefer locally relevant valid actions when constrained fallback is needed."""

        ranked: list[str] = []
        observation_text = observation.lower()
        preferred_patterns = (
            "examine",
            "look at",
            "read",
            "take",
            "get",
            "open",
            "look in",
            "enter",
            "look",
            "inventory",
            "go ",
        )

        for keyword in ("mailbox", "leaflet", "door", "window", "house", "rope", "lamp"):
            if keyword in observation_text:
                for action in valid_actions:
                    if keyword in action.lower() and action not in ranked:
                        ranked.append(action)
        for action in valid_actions:
            lowered = action.lower()
            if any(pattern in lowered for pattern in preferred_patterns) and action not in ranked:
                ranked.append(action)
        for action in valid_actions:
            if action not in ranked:
                ranked.append(action)
        return ranked

    def _fallback_open_actions(self, *, observation: str, inventory_text: str) -> list[str]:
        """Build deterministic free-form fallback actions from obvious local cues."""

        actions: list[str] = ["look"]
        observation_text = observation.lower()
        inventory_lower = inventory_text.lower()
        if "mailbox" in observation_text:
            actions.extend(["open mailbox", "take leaflet", "read leaflet"])
        if "leaflet" in observation_text:
            actions.extend(["take leaflet", "read leaflet", "examine leaflet"])
        if "door" in observation_text:
            actions.append("open door")
        if "window" in observation_text:
            actions.append("examine window")
        if "house" in observation_text:
            actions.append("examine house")
        if inventory_text and "empty" not in inventory_lower and "unavailable" not in inventory_lower:
            actions.append("inventory")
        else:
            actions.append("inventory")
        deduped: list[str] = []
        seen: set[str] = set()
        for action in actions:
            normalized = normalize_parser_action(action)
            if normalized not in seen:
                deduped.append(action)
                seen.add(normalized)
        return deduped

    def _rerank_candidates_evidence(
        self,
        result: ActionGenerationResult,
        *,
        observation: str,
        inventory_text: str,
        recent_actions: list[str],
        recent_loop_results: list[LoopHeuristicResult],
        state_action_history: ActionClusterHistory | None,
        supported_try_actions: list[str],
        supported_avoid_actions: list[str],
        supported_reflection_objects: list[str],
        strategic_mode: StrategicMode,
        strategic_reason: str,
        strategic_try_actions: list[str],
        strategic_avoid_actions: list[str],
        strategic_objects: list[str],
        candidate_count: int,
    ) -> None:
        """Rerank candidates with explicit evidence features and diversity constraints."""

        if not result.candidates:
            return

        normalized_try_actions = {normalize_parser_action(action) for action in supported_try_actions}
        normalized_avoid_actions = {normalize_parser_action(action) for action in supported_avoid_actions}
        normalized_strategic_try_actions = {normalize_parser_action(action) for action in strategic_try_actions}
        normalized_strategic_avoid_actions = {
            normalize_parser_action(action) for action in strategic_avoid_actions
        }
        supported_object_tokens = self._supported_object_tokens(
            supported_reflection_objects=supported_reflection_objects,
            supported_try_actions=supported_try_actions,
        )
        strategic_object_tokens = self._normalize_token_list(strategic_objects)
        visible_nouns = extract_salient_nouns(
            observation=observation,
            inventory_text=inventory_text,
            valid_actions=[],
            inverse_pairs=self.config.policy.inverse_action_pairs,
        )
        inventory_tokens = inventory_item_tokens(
            inventory_text,
            inverse_pairs=self.config.policy.inverse_action_pairs,
        )
        known_nouns = set(state_action_history.seen_nouns) if state_action_history is not None else set()
        new_visible_nouns = (
            set(state_action_history.fresh_visible_nouns)
            if (
                state_action_history is not None
                and state_action_history.current_visible_nouns == visible_nouns
                and state_action_history.fresh_visible_nouns
            )
            else visible_nouns - known_nouns
        )
        exhausted_families = (
            state_action_history.exhausted_families(self.config.policy.object_family_no_progress_threshold)
            if state_action_history is not None
            else set()
        )
        cluster_repeat_count = state_action_history.cluster_visit_count() if state_action_history is not None else 0
        scene_score_success_count = (
            state_action_history.prior_score_success_for_object_nouns(visible_nouns)
            if state_action_history is not None
            else 0
        )

        llm_rank_map = {
            normalize_parser_action(action): rank
            for rank, action in enumerate(result.llm_ranked_actions, start=1)
        }
        base_scored: list[ActionProposal] = []
        object_centric_candidates_exist = False
        for index, candidate in enumerate(result.candidates):
            features = self._build_candidate_features(
                candidate=candidate,
                state_action_history=state_action_history,
                recent_actions=recent_actions,
                observation=observation,
                inventory_text=inventory_text,
                valid_actions=result.valid_actions,
                inventory_tokens=inventory_tokens,
                visible_nouns=visible_nouns,
                new_visible_nouns=new_visible_nouns,
                supported_object_tokens=supported_object_tokens,
                strategic_object_tokens=set(strategic_object_tokens),
                normalized_try_actions=normalized_try_actions,
                normalized_avoid_actions=normalized_avoid_actions,
                normalized_strategic_try_actions=normalized_strategic_try_actions,
                normalized_strategic_avoid_actions=normalized_strategic_avoid_actions,
                cluster_repeat_count=cluster_repeat_count,
                scene_score_success_count=scene_score_success_count,
                strategic_mode=strategic_mode,
            )
            candidate.features = features
            candidate.movement_only_action = features.is_movement_action
            candidate.loop_penalty, candidate.loop_penalty_reason = estimate_candidate_loop_penalty(
                candidate_action=candidate.action,
                recent_actions=recent_actions,
                recent_loop_results=recent_loop_results,
                inverse_pairs=self.config.policy.inverse_action_pairs,
                immediate_inverse_penalty=self.config.policy.immediate_inverse_penalty,
                repeated_pair_penalty=self.config.policy.repeated_pair_penalty,
                reversible_no_progress_penalty=self.config.policy.reversible_no_progress_penalty,
            )
            positive_score, negative_score, reasons = self._candidate_evidence_score(
                candidate=candidate,
                features=features,
                exhausted_families=exhausted_families,
                inventory_tokens=inventory_tokens,
                strategic_mode=strategic_mode,
            )
            candidate.heuristic_bonus = positive_score
            candidate.heuristic_penalty = negative_score
            candidate.selection_score = positive_score - negative_score - candidate.loop_penalty
            llm_rank = llm_rank_map.get(normalize_parser_action(candidate.action), index + 1)
            if llm_rank_map and normalize_parser_action(candidate.action) in llm_rank_map:
                candidate.rank = llm_rank
            candidate.ranking_reason = ", ".join(reasons)
            if self._is_object_centric_priority_action(candidate.action, features) and not (
                features.local_scene_post_score_stale
                and not (
                    features.touches_recent_affordance_object
                    or features.touches_strategic_object
                    or features.matches_strategic_try_action
                )
            ):
                object_centric_candidates_exist = True
            base_scored.append(candidate)

        if self._should_prefer_exit_escape(base_scored, inventory_tokens):
            for candidate in base_scored:
                if candidate.features is None:
                    continue
                if candidate.features.is_movement_action and not self._movement_has_prior_evidence(candidate.features):
                    candidate.heuristic_bonus += self.config.policy.escape_mode_exit_bonus
                    candidate.selection_score += self.config.policy.escape_mode_exit_bonus
                    candidate.ranking_reason = self._append_reason(
                        candidate.ranking_reason,
                        "escape_mode_exit",
                    )
                elif self._is_ungrounded_other_action(candidate.features):
                    candidate.heuristic_penalty += self.config.policy.escape_mode_other_action_penalty
                    candidate.selection_score -= self.config.policy.escape_mode_other_action_penalty
                    candidate.ranking_reason = self._append_reason(
                        candidate.ranking_reason,
                        "escape_mode_other_action",
                    )

        if object_centric_candidates_exist:
            for candidate in base_scored:
                if candidate.features is None:
                    continue
                if candidate.features.is_movement_action and not self._movement_has_prior_evidence(candidate.features):
                    candidate.movement_penalty += self.config.policy.same_cluster_movement_penalty
                    candidate.heuristic_penalty += self.config.policy.same_cluster_movement_penalty
                    candidate.selection_score -= self.config.policy.same_cluster_movement_penalty
                    candidate.ranking_reason = self._append_reason(
                        candidate.ranking_reason,
                        "object_centric_action_available",
                    )

        candidate_feature_records = [
            {
                **candidate.features.to_record(),
                "loop_penalty": candidate.loop_penalty,
                "movement_penalty": candidate.movement_penalty,
                "heuristic_bonus": candidate.heuristic_bonus,
                "heuristic_penalty": candidate.heuristic_penalty,
                "selection_score": candidate.selection_score,
                "ranking_reason": candidate.ranking_reason,
            }
            for candidate in base_scored
            if candidate.features is not None
        ]

        base_scored.sort(
            key=lambda candidate: (
                -candidate.selection_score,
                candidate.loop_penalty + candidate.movement_penalty,
                llm_rank_map.get(normalize_parser_action(candidate.action), 9999),
                normalize_parser_action(candidate.action),
            )
        )
        reranked_candidates = self._select_diverse_top_k(
            base_scored,
            candidate_count=candidate_count,
        )
        for rank, candidate in enumerate(reranked_candidates, start=1):
            candidate.rank = rank

        result.candidates = reranked_candidates
        result.candidate_feature_records = candidate_feature_records
        result.reranked_by_loop_penalty = any(candidate.loop_penalty > 0.0 for candidate in reranked_candidates)
        result.reranked_by_affordance_heuristics = any(
            candidate.heuristic_bonus > 0.0 or candidate.heuristic_penalty > 0.0
            for candidate in reranked_candidates
        )
        result.reranked_by_movement_heuristics = any(candidate.movement_penalty > 0.0 for candidate in reranked_candidates)
        if result.llm_ranked_actions:
            result.ranking_source = "hybrid_merge"
        elif result.ranking_source != "heuristic_rerank":
            result.ranking_source = "heuristic_rerank"
        if result.candidates:
            result.top_selection_reason = result.candidates[0].ranking_reason or "highest evidence score"
        result.strategic_mode = strategic_mode.value
        result.strategic_reason = strategic_reason

    def _build_candidate_features(
        self,
        *,
        candidate: ActionProposal,
        state_action_history: ActionClusterHistory | None,
        recent_actions: list[str],
        observation: str,
        inventory_text: str,
        valid_actions: list[str],
        inventory_tokens: set[str],
        visible_nouns: set[str],
        new_visible_nouns: set[str],
        supported_object_tokens: set[str],
        strategic_object_tokens: set[str],
        normalized_try_actions: set[str],
        normalized_avoid_actions: set[str],
        normalized_strategic_try_actions: set[str],
        normalized_strategic_avoid_actions: set[str],
        cluster_repeat_count: int,
        scene_score_success_count: int,
        strategic_mode: StrategicMode,
    ) -> ActionCandidateFeatures:
        """Build the explicit feature vector for one candidate action."""

        normalized_action = normalize_parser_action(candidate.action)
        verb, _obj = split_action_command(candidate.action, self.config.policy.inverse_action_pairs)
        verb_family = canonical_if_verb_family(candidate.action, self.config.policy.inverse_action_pairs)
        noun_targets = sorted(action_target_tokens(candidate.action, self.config.policy.inverse_action_pairs))
        stats = state_action_history.action_stats.get(normalized_action) if state_action_history is not None else None
        previous_attempt_count = stats.attempts if stats is not None else 0
        produced_affordance_gain_before = bool(
            stats is not None and (stats.valid_actions_improvement_count > 0 or stats.revealed_object_count > 0)
        )
        prior_success_for_verb_family = (
            state_action_history.prior_success_for_verb_family(verb_family) > 0
            if state_action_history is not None
            else False
        )
        prior_score_success_for_verb_family = (
            state_action_history.prior_score_success_for_verb_family(verb_family) > 0
            if state_action_history is not None
            else False
        )
        prior_score_success_for_object_family = (
            state_action_history.prior_score_success_for_object_nouns(set(noun_targets)) > 0
            if state_action_history is not None
            else False
        )
        prior_inventory_gain_for_object_family = (
            state_action_history.prior_inventory_gain_for_object_nouns(set(noun_targets))
            if state_action_history is not None
            else 0
        )
        prior_inventory_loss_for_object_family = (
            state_action_history.prior_inventory_loss_for_object_nouns(set(noun_targets))
            if state_action_history is not None
            else 0
        )
        prior_affordance_gain_for_object_family = (
            state_action_history.prior_affordance_gain_for_object_nouns(set(noun_targets))
            if state_action_history is not None
            else 0
        )
        prior_success_for_object_family = (
            state_action_history.prior_success_for_object_nouns(set(noun_targets)) > 0
            if state_action_history is not None
            else False
        )
        prior_failure_for_object_family = (
            state_action_history.prior_failure_for_object_nouns(set(noun_targets)) > 0
            if state_action_history is not None
            else False
        )
        features = ActionCandidateFeatures(
            action_text=candidate.action,
            source_mode=candidate.source,
            verb=verb,
            verb_family=verb_family,
            noun_targets=noun_targets,
            object_count=action_object_count(candidate.action, self.config.policy.inverse_action_pairs),
            is_movement_action=is_movement_action(candidate.action, self.config.policy.inverse_action_pairs),
            is_reversible_toggle=verb in self.config.policy.inverse_action_pairs,
            is_bulk_inventory_action=is_bulk_inventory_action(candidate.action),
            is_discard_like_action=is_discard_like_action(candidate.action),
            uses_inventory_object=action_uses_inventory_object(
                candidate.action,
                inventory_tokens,
                self.config.policy.inverse_action_pairs,
            ),
            target_is_new_salient_object=bool(set(noun_targets) & new_visible_nouns),
            target_is_container_or_openable=target_is_container_or_openable_candidate(
                candidate.action,
                valid_actions,
                self.config.policy.inverse_action_pairs,
            ),
            target_is_readable_candidate=target_is_readable_candidate(
                candidate.action,
                observation=observation,
                inventory_text=inventory_text,
                valid_actions=valid_actions,
                inverse_pairs=self.config.policy.inverse_action_pairs,
            ),
            action_shape_is_simple=action_shape_is_simple(
                candidate.action,
                self.config.policy.inverse_action_pairs,
            ),
            action_shape_is_complex_transitive=action_shape_is_complex_transitive(
                candidate.action,
                self.config.policy.inverse_action_pairs,
            ),
            was_tried_before_in_local_cluster=previous_attempt_count > 0,
            previous_attempt_count=previous_attempt_count,
            produced_score_gain_before=bool(stats is not None and stats.score_change_count > 0),
            produced_inventory_gain_before=bool(stats is not None and stats.inventory_gain_count > 0),
            produced_affordance_gain_before=produced_affordance_gain_before,
            prior_score_success_for_verb_family=prior_score_success_for_verb_family,
            prior_success_for_verb_family=prior_success_for_verb_family,
            prior_success_for_exact_action=bool(stats is not None and stats.durable_gain_count > 0),
            prior_failure_for_exact_action=bool(stats is not None and stats.no_durable_gain_count > 0),
            prior_score_success_for_object_family=prior_score_success_for_object_family,
            prior_success_for_object_family=prior_success_for_object_family,
            prior_failure_for_object_family=prior_failure_for_object_family,
            prior_inventory_gain_for_object_family=prior_inventory_gain_for_object_family,
            prior_inventory_loss_for_object_family=prior_inventory_loss_for_object_family,
            prior_affordance_gain_for_object_family=prior_affordance_gain_for_object_family,
            touches_newly_salient_object=bool(set(noun_targets) & new_visible_nouns),
            touches_recent_affordance_object=(
                state_action_history.touches_recent_affordance_targets(set(noun_targets))
                if state_action_history is not None
                else False
            ),
            touches_supported_reflection_object=bool(set(noun_targets) & supported_object_tokens),
            touches_strategic_object=bool(set(noun_targets) & strategic_object_tokens),
            inverse_of_previous_action=bool(recent_actions)
            and actions_are_inverse(candidate.action, recent_actions[-1], self.config.policy.inverse_action_pairs),
            movement_repeat_count=count_repeated_movement_actions(
                [*recent_actions, candidate.action],
                self.config.policy.inverse_action_pairs,
            )
            if recent_actions
            else 0,
            cluster_repeat_count=cluster_repeat_count,
            movement_targets_landmark=self._movement_targets_landmark(
                candidate.action,
                visible_nouns=visible_nouns,
            ),
            is_freshly_unlocked_action=bool(
                state_action_history is not None
                and normalized_action in state_action_history.fresh_valid_actions
            ),
            is_freshly_unlocked_exit=bool(
                state_action_history is not None
                and normalized_action in state_action_history.fresh_exit_actions
            ),
            follows_recent_affordance_unlock=bool(
                state_action_history is not None
                and normalized_action in state_action_history.fresh_exit_actions
                and state_action_history.recent_affordance_targets
            ),
            touches_exhausted_family=bool(
                state_action_history is not None
                and state_action_history.exhausted_families(self.config.policy.object_family_no_progress_threshold)
                & set(noun_targets)
            ),
            exhausted_family_count=len(
                (
                    state_action_history.exhausted_families(self.config.policy.object_family_no_progress_threshold)
                    & set(noun_targets)
                )
                if state_action_history is not None
                else set()
            ),
            max_family_no_progress_count=(
                state_action_history.max_family_no_progress_count(set(noun_targets))
                if state_action_history is not None
                else 0
            ),
            scene_has_score_harvested=scene_score_success_count > 0,
            scene_score_success_count=scene_score_success_count,
            matches_supported_try_action=normalized_action in normalized_try_actions,
            matches_supported_avoid_action=normalized_action in normalized_avoid_actions,
            matches_strategic_try_action=normalized_action in normalized_strategic_try_actions,
            matches_strategic_avoid_action=normalized_action in normalized_strategic_avoid_actions,
            current_mode=strategic_mode.value,
        )
        if features.target_is_new_salient_object and (
            features.prior_score_success_for_object_family
            or features.prior_inventory_gain_for_object_family > 0
            or (set(noun_targets) & inventory_tokens)
        ):
            features.target_is_new_salient_object = False
        if features.touches_newly_salient_object and not features.target_is_new_salient_object:
            features.touches_newly_salient_object = False
        if (
            not features.target_is_new_salient_object
            and not (
                features.prior_score_success_for_object_family
                or features.prior_inventory_gain_for_object_family > 0
                or (set(noun_targets) & inventory_tokens)
            )
        ):
            features.target_is_new_salient_object = bool(set(noun_targets) & visible_nouns and set(noun_targets) & new_visible_nouns)
        features.local_scene_post_score_stale = self._is_local_scene_post_score_stale(
            features,
            strong_prior_gain=self._has_strong_prior_gain(features),
        )
        return features

    def _candidate_evidence_score(
        self,
        *,
        candidate: ActionProposal,
        features: ActionCandidateFeatures,
        exhausted_families: set[str],
        inventory_tokens: set[str],
        strategic_mode: StrategicMode,
    ) -> tuple[float, float, list[str]]:
        """Compute the transparent evidence-based ranking formula for one candidate."""

        positive = 0.0
        negative = 0.0
        reasons: list[str] = []
        target_tokens = set(features.noun_targets)
        normalized_action = normalize_parser_action(candidate.action)
        is_put_action = normalized_action.startswith("put ")
        is_drop_action = normalized_action.startswith("drop ")
        is_discard_action = is_put_action or is_drop_action or features.is_discard_like_action
        strong_prior_gain = self._has_strong_prior_gain(features)
        stale_movement_without_durable_gain = self._is_stale_movement_without_durable_gain(
            features,
            strong_prior_gain=strong_prior_gain,
        )
        stale_toggle_without_durable_gain = self._is_stale_toggle_without_durable_gain(
            features,
            strong_prior_gain=strong_prior_gain,
        )
        stale_reacquire_inventory_churn = self._is_stale_reacquire_inventory_churn(
            features,
            inventory_tokens=inventory_tokens,
            strong_prior_gain=strong_prior_gain,
        )
        stale_reexpose_container = self._is_stale_reexpose_container_without_durable_gain(
            features,
            inventory_tokens=inventory_tokens,
            strong_prior_gain=strong_prior_gain,
        )
        trustworthy_soft_support = self._soft_support_is_trustworthy(
            features,
            strong_prior_gain=strong_prior_gain,
            stale_reexpose_container=stale_reexpose_container,
        )
        scene_stale_after_score = self._is_local_scene_post_score_stale(
            features,
            strong_prior_gain=strong_prior_gain,
        )
        post_score_low_value_new_object_acquire = self._is_post_score_low_value_new_object_acquire(
            features,
            inventory_tokens=inventory_tokens,
            strong_prior_gain=strong_prior_gain,
        )
        canonical_prior_contribution = 0.0
        object_family_evidence_contribution = 0.0
        plausibility_score = 0.0
        exhausted_family_level = 0
        if features.touches_exhausted_family:
            exhausted_family_level = max(
                features.max_family_no_progress_count - self.config.policy.object_family_no_progress_threshold + 1,
                1,
            )

        family_prior = self.config.policy.verb_family_prior_scores.get(features.verb_family, 0.0)
        canonical_prior_contribution += family_prior
        if family_prior >= 0:
            positive += family_prior
        else:
            negative += abs(family_prior)
        plausibility_score += family_prior
        reasons.append(f"verb_family={features.verb_family}")

        if (
            not features.was_tried_before_in_local_cluster
            and not features.is_movement_action
            and not features.is_bulk_inventory_action
            and (features.noun_targets or normalized_action not in _LOW_VALUE_GENERIC_ACTIONS)
            and not self._is_ungrounded_other_action(features)
            and not self._is_stale_nonnew_acquire_in_inventory_scene(
                features,
                inventory_tokens=inventory_tokens,
                strong_prior_gain=strong_prior_gain,
            )
            and not stale_reacquire_inventory_churn
            and not (
                features.touches_exhausted_family
                and (features.is_reversible_toggle or is_discard_action or not strong_prior_gain)
            )
            and not (
                is_put_action
                and target_tokens & inventory_tokens
                and not strong_prior_gain
            )
            and not scene_stale_after_score
            and not post_score_low_value_new_object_acquire
        ):
            positive += self.config.policy.untried_action_bonus
            plausibility_score += self.config.policy.untried_action_bonus
            reasons.append("untried_action")
        elif (
            features.was_tried_before_in_local_cluster
            and not (
                features.produced_score_gain_before
                or features.produced_inventory_gain_before
                or features.produced_affordance_gain_before
            )
        ):
            repeated_penalty = self.config.policy.repeated_no_gain_penalty * float(
                max(features.previous_attempt_count, 1)
            )
            negative += repeated_penalty
            plausibility_score -= repeated_penalty
            reasons.append(f"repeated_no_gain={features.previous_attempt_count}")

        if (
            features.touches_newly_salient_object
            and self._is_object_centric_priority_action(candidate.action, features)
            and not post_score_low_value_new_object_acquire
        ):
            positive += self.config.policy.new_object_bonus * float(max(len(features.noun_targets), 1))
            plausibility_score += self.config.policy.new_object_bonus * float(max(len(features.noun_targets), 1))
            reasons.append("new_object")

        durable_history_signal = (
            features.produced_score_gain_before
            or (
                features.produced_inventory_gain_before
                and not features.is_bulk_inventory_action
                and not features.is_discard_like_action
            )
        )
        transferable_affordance_history = (
            features.produced_affordance_gain_before
            and not features.is_discard_like_action
            and not stale_toggle_without_durable_gain
            and not stale_reacquire_inventory_churn
            and not stale_reexpose_container
            and not scene_stale_after_score
            and not self._is_stale_nonnew_acquire_in_inventory_scene(
                features,
                inventory_tokens=inventory_tokens,
                strong_prior_gain=strong_prior_gain,
            )
        )

        if (
            durable_history_signal
            or transferable_affordance_history
        ) and (
            strong_prior_gain
            or (
                not stale_movement_without_durable_gain
                and
                not features.touches_exhausted_family
                and not stale_toggle_without_durable_gain
                and not stale_reacquire_inventory_churn
                and not stale_reexpose_container
                and not scene_stale_after_score
            )
        ):
            positive += self.config.policy.inventory_affordance_bonus
            plausibility_score += self.config.policy.inventory_affordance_bonus
            reasons.append("historical_gain")
        elif features.produced_affordance_gain_before and (
            features.touches_exhausted_family
            or stale_toggle_without_durable_gain
            or stale_reacquire_inventory_churn
            or stale_reexpose_container
        ):
            negative += self.config.policy.object_family_exhaustion_penalty
            plausibility_score -= self.config.policy.object_family_exhaustion_penalty
            reasons.append("stale_affordance_history")

        if (
            self._is_object_centric_priority_action(candidate.action, features)
            and not post_score_low_value_new_object_acquire
            and (
                features.target_is_new_salient_object
                or features.touches_recent_affordance_object
                or strong_prior_gain
                or features.prior_score_success_for_object_family
                or (
                    features.produced_affordance_gain_before
                    and not features.touches_exhausted_family
                    and not stale_toggle_without_durable_gain
                    and not stale_reacquire_inventory_churn
                    and not stale_reexpose_container
                )
                or (
                    trustworthy_soft_support
                    and features.matches_supported_try_action
                    and self._reflection_try_has_durable_support(
                        features,
                        strong_prior_gain=strong_prior_gain,
                    )
                )
            )
        ):
            positive += self.config.policy.examine_read_take_bonus
            plausibility_score += self.config.policy.examine_read_take_bonus
            reasons.append("object_centric_priority")

        if (
            features.touches_recent_affordance_object
            and self._is_object_centric_priority_action(candidate.action, features)
            and not features.is_discard_like_action
            and not self._is_speculative_tool_use(features)
        ):
            positive += self.config.policy.fresh_affordance_followup_bonus
            canonical_prior_contribution += self.config.policy.fresh_affordance_followup_bonus
            plausibility_score += self.config.policy.fresh_affordance_followup_bonus
            reasons.append("fresh_affordance_followup")

        if features.is_freshly_unlocked_exit:
            unlocked_exit_bonus = self.config.policy.fresh_unlocked_exit_bonus
            if features.follows_recent_affordance_unlock:
                unlocked_exit_bonus += self.config.policy.fresh_affordance_followup_bonus
            positive += unlocked_exit_bonus
            plausibility_score += unlocked_exit_bonus
            reasons.append(
                "fresh_affordance_unlocked_exit"
                if features.follows_recent_affordance_unlock
                else "fresh_unlocked_exit"
            )

        if trustworthy_soft_support and (
            features.matches_supported_try_action or features.touches_supported_reflection_object
        ):
            positive += self.config.policy.reflection_supported_try_bonus
            plausibility_score += self.config.policy.reflection_supported_try_bonus
            reasons.append("reflection_supported_try")
        elif (
            (scene_stale_after_score or stale_reacquire_inventory_churn)
            and (features.matches_supported_try_action or features.touches_supported_reflection_object)
            and not strong_prior_gain
        ):
            negative += self.config.policy.stale_reflection_try_penalty
            plausibility_score -= self.config.policy.stale_reflection_try_penalty
            reasons.append("stale_reflection_try")

        if features.matches_supported_avoid_action:
            negative += self.config.policy.reflection_supported_avoid_penalty
            plausibility_score -= self.config.policy.reflection_supported_avoid_penalty
            reasons.append("reflection_supported_avoid")

        strategic_contribution = 0.0
        if features.matches_strategic_try_action:
            positive += self.config.policy.strategic_try_bonus
            plausibility_score += self.config.policy.strategic_try_bonus
            strategic_contribution += self.config.policy.strategic_try_bonus
            reasons.append("strategic_try")
        if (
            features.touches_strategic_object
            and self._is_object_centric_priority_action(candidate.action, features)
        ):
            positive += self.config.policy.strategic_object_bonus
            plausibility_score += self.config.policy.strategic_object_bonus
            strategic_contribution += self.config.policy.strategic_object_bonus
            reasons.append("strategic_object")
        if features.matches_strategic_avoid_action:
            negative += self.config.policy.strategic_avoid_penalty
            plausibility_score -= self.config.policy.strategic_avoid_penalty
            strategic_contribution -= self.config.policy.strategic_avoid_penalty
            reasons.append("strategic_avoid")
        if (
            strategic_mode is StrategicMode.EXPLORE
            and features.is_movement_action
            and features.matches_strategic_try_action
            and not self._movement_has_prior_evidence(features)
        ):
            positive += self.config.policy.explore_mode_exit_bonus
            plausibility_score += self.config.policy.explore_mode_exit_bonus
            strategic_contribution += self.config.policy.explore_mode_exit_bonus
            reasons.append("explore_mode_exit")
        if (
            strategic_mode is StrategicMode.EXPLOIT
            and not features.is_movement_action
            and (features.matches_strategic_try_action or features.touches_strategic_object)
            and not features.is_discard_like_action
            and not stale_toggle_without_durable_gain
            and not stale_reacquire_inventory_churn
            and not scene_stale_after_score
        ):
            positive += 0.5 * self.config.policy.strategic_try_bonus
            plausibility_score += 0.5 * self.config.policy.strategic_try_bonus
            strategic_contribution += 0.5 * self.config.policy.strategic_try_bonus
            reasons.append("exploit_mode_focus")

        if self._is_ungrounded_other_action(features):
            negative += self.config.policy.other_action_penalty
            plausibility_score -= self.config.policy.other_action_penalty
            reasons.append("ungrounded_other_action")

        if features.is_bulk_inventory_action and not strong_prior_gain:
            negative += self.config.policy.bulk_inventory_action_penalty
            plausibility_score -= self.config.policy.bulk_inventory_action_penalty
            reasons.append("bulk_inventory_action")

        if features.inverse_of_previous_action and not self._has_prior_gain(features):
            negative += self.config.policy.inverse_action_penalty
            plausibility_score -= self.config.policy.inverse_action_penalty
            reasons.append("inverse_previous_action")

        if (
            features.is_reversible_toggle
            and features.was_tried_before_in_local_cluster
            and not self._has_prior_gain(features)
        ):
            negative += self.config.policy.reversible_toggle_penalty
            plausibility_score -= self.config.policy.reversible_toggle_penalty
            reasons.append("reversible_toggle_no_gain")

        if (
            features.is_reversible_toggle
            and inventory_tokens
            and not (target_tokens & inventory_tokens)
            and not strong_prior_gain
        ):
            negative += self.config.policy.reversible_toggle_penalty + (
                0.5 * self.config.policy.object_family_exhaustion_penalty
            )
            plausibility_score -= self.config.policy.reversible_toggle_penalty + (
                0.5 * self.config.policy.object_family_exhaustion_penalty
            )
            reasons.append("post_inventory_toggle")

        if stale_toggle_without_durable_gain:
            stale_toggle_penalty = self.config.policy.reversible_toggle_penalty + (
                0.5 * self.config.policy.object_family_exhaustion_penalty
            )
            if features.verb == "close":
                stale_toggle_penalty += 0.5 * self.config.policy.object_family_exhaustion_penalty
            negative += stale_toggle_penalty
            plausibility_score -= stale_toggle_penalty
            reasons.append("stale_toggle_without_durable_gain")

        if features.is_movement_action:
            if (
                features.movement_targets_landmark
                and not stale_movement_without_durable_gain
            ):
                positive += self.config.policy.landmark_movement_bonus
                canonical_prior_contribution += self.config.policy.landmark_movement_bonus
                plausibility_score += self.config.policy.landmark_movement_bonus
                reasons.append("landmark_movement")
            if features.movement_repeat_count > 0:
                movement_penalty = self.config.policy.movement_repeat_penalty * float(features.movement_repeat_count)
                negative += movement_penalty
                plausibility_score -= movement_penalty
                candidate.movement_penalty += movement_penalty
                candidate.movement_penalty_reason = self._append_reason(
                    candidate.movement_penalty_reason,
                    f"movement_repeat_count={features.movement_repeat_count}",
                )
                reasons.append(f"movement_repeat_count={features.movement_repeat_count}")
            if features.cluster_repeat_count > 1 and not self._movement_has_prior_evidence(features):
                cluster_penalty = self.config.policy.same_cluster_movement_penalty * float(
                    features.cluster_repeat_count - 1
                )
                negative += cluster_penalty
                plausibility_score -= cluster_penalty
                candidate.movement_penalty += cluster_penalty
                candidate.movement_penalty_reason = self._append_reason(
                    candidate.movement_penalty_reason,
                    f"same_region_repeat={features.cluster_repeat_count}",
                )
                reasons.append(f"same_cluster_movement={features.cluster_repeat_count}")

        if exhausted_families and target_tokens & exhausted_families:
            exhaustion_penalty = self.config.policy.object_family_exhaustion_penalty * float(
                max(len(target_tokens & exhausted_families), 1) * max(exhausted_family_level, 1)
            )
            if features.is_reversible_toggle:
                exhaustion_penalty *= 1.5
            if is_discard_action:
                exhaustion_penalty *= 1.5
            negative += exhaustion_penalty
            plausibility_score -= exhaustion_penalty
            reasons.append("exhausted_family:" + ",".join(sorted(target_tokens & exhausted_families)))

        if scene_stale_after_score:
            scene_penalty = self.config.policy.post_score_scene_exhaustion_penalty
            if features.is_discard_like_action or features.action_shape_is_complex_transitive:
                scene_penalty += 0.5 * self.config.policy.post_score_scene_exhaustion_penalty
            if features.verb_family == "aggressive":
                scene_penalty += 0.5 * self.config.policy.post_score_scene_exhaustion_penalty
            negative += scene_penalty
            plausibility_score -= scene_penalty
            reasons.append("post_score_scene_exhausted")

        if (
            scene_stale_after_score
            and not features.is_movement_action
            and not features.target_is_new_salient_object
            and not features.touches_recent_affordance_object
            and not strong_prior_gain
        ):
            local_churn_penalty = self.config.policy.post_score_local_churn_penalty
            if features.is_reversible_toggle:
                local_churn_penalty += 0.5 * self.config.policy.post_score_local_churn_penalty
            if features.verb_family in {"use", "aggressive"} or is_discard_action:
                local_churn_penalty += 0.5 * self.config.policy.post_score_local_churn_penalty
            negative += local_churn_penalty
            plausibility_score -= local_churn_penalty
            reasons.append("post_score_local_churn")

        if post_score_low_value_new_object_acquire:
            negative += self.config.policy.post_score_local_churn_penalty
            plausibility_score -= self.config.policy.post_score_local_churn_penalty
            reasons.append("post_score_low_value_new_object")

        if self._is_post_score_inventory_object_disruption(
            features,
            inventory_tokens=inventory_tokens,
            strong_prior_gain=strong_prior_gain,
        ):
            scored_object_penalty = self.config.policy.post_score_inventory_object_penalty
            if features.action_shape_is_complex_transitive or features.verb_family in {"use", "aggressive"}:
                scored_object_penalty += 0.5 * self.config.policy.post_score_inventory_object_penalty
            negative += scored_object_penalty
            plausibility_score -= scored_object_penalty
            reasons.append("post_score_inventory_object_preserve")

        if stale_reexpose_container:
            negative += self.config.policy.object_family_exhaustion_penalty
            plausibility_score -= self.config.policy.object_family_exhaustion_penalty
            reasons.append("stale_reexpose_container")

        if (
            is_put_action
            and target_tokens & inventory_tokens
            and not strong_prior_gain
        ):
            negative += self.config.policy.discard_inventory_penalty
            plausibility_score -= self.config.policy.discard_inventory_penalty
            reasons.append("discard_inventory")

        if self._is_stale_nonnew_acquire_in_inventory_scene(
            features,
            inventory_tokens=inventory_tokens,
            strong_prior_gain=strong_prior_gain,
        ):
            stale_acquire_penalty = (
                self.config.policy.object_family_exhaustion_penalty
                + 0.5 * self.config.policy.stale_reflection_try_penalty
            )
            negative += stale_acquire_penalty
            plausibility_score -= stale_acquire_penalty
            reasons.append("stale_nonnew_acquire")

        if stale_reacquire_inventory_churn:
            reacquire_penalty = (
                self.config.policy.object_family_exhaustion_penalty
                + 0.5 * self.config.policy.discard_inventory_penalty
            )
            negative += reacquire_penalty
            plausibility_score -= reacquire_penalty
            reasons.append("stale_reacquire_churn")

        if (
            is_drop_action
            and target_tokens & inventory_tokens
            and (
                features.touches_exhausted_family
                or (
                    features.was_tried_before_in_local_cluster
                    and not strong_prior_gain
                )
            )
        ):
            negative += self.config.policy.discard_inventory_penalty
            plausibility_score -= self.config.policy.discard_inventory_penalty
            reasons.append("discard_inventory")

        if features.target_is_new_salient_object and not post_score_low_value_new_object_acquire:
            canonical_bonus = self._canonical_new_object_bonus(features)
            if canonical_bonus != 0.0:
                positive += max(canonical_bonus, 0.0)
                negative += max(-canonical_bonus, 0.0)
                canonical_prior_contribution += canonical_bonus
                plausibility_score += canonical_bonus
                reasons.append(f"canonical_new_object={canonical_bonus:.2f}")

        if (
            features.target_is_readable_candidate
            and features.verb_family == "inspect"
            and (
                features.uses_inventory_object
                or features.target_is_new_salient_object
                or features.touches_supported_reflection_object
                or features.prior_success_for_object_family
            )
        ):
            positive += self.config.policy.readable_action_bonus
            canonical_prior_contribution += self.config.policy.readable_action_bonus
            plausibility_score += self.config.policy.readable_action_bonus
            reasons.append("readable_candidate")

        if (
            features.target_is_container_or_openable
            and features.verb in {"open", "look in", "look inside"}
            and not stale_reexpose_container
        ):
            positive += self.config.policy.openable_action_bonus
            canonical_prior_contribution += self.config.policy.openable_action_bonus
            plausibility_score += self.config.policy.openable_action_bonus
            reasons.append("openable_candidate")

        if features.action_shape_is_simple and self._is_object_centric_priority_action(candidate.action, features):
            positive += self.config.policy.simple_action_shape_bonus
            canonical_prior_contribution += self.config.policy.simple_action_shape_bonus
            plausibility_score += self.config.policy.simple_action_shape_bonus
            reasons.append("simple_action_shape")

        if (
            features.prior_success_for_verb_family
            and not features.is_discard_like_action
            and not post_score_low_value_new_object_acquire
        ) and (
            strong_prior_gain
            or (
                not stale_movement_without_durable_gain
                and not stale_toggle_without_durable_gain
                and not stale_reacquire_inventory_churn
                and not stale_reexpose_container
                and not scene_stale_after_score
            )
        ):
            positive += self.config.policy.verb_family_success_bonus
            plausibility_score += self.config.policy.verb_family_success_bonus
            reasons.append("verb_family_success")

        if (
            features.prior_success_for_exact_action
            and not features.is_discard_like_action
        ) and (
            strong_prior_gain
            or (
                not stale_movement_without_durable_gain
                and not stale_toggle_without_durable_gain
                and not stale_reacquire_inventory_churn
                and not stale_reexpose_container
                and not scene_stale_after_score
            )
        ):
            positive += self.config.policy.exact_action_success_bonus
            plausibility_score += self.config.policy.exact_action_success_bonus
            reasons.append("exact_action_success")

        if features.prior_failure_for_exact_action and not strong_prior_gain:
            negative += self.config.policy.exact_action_failure_penalty
            plausibility_score -= self.config.policy.exact_action_failure_penalty
            reasons.append("exact_action_failure")

        if (
            features.prior_success_for_object_family
            and self._object_family_evidence_can_transfer(
                features,
                strong_prior_gain=strong_prior_gain,
                scene_stale_after_score=scene_stale_after_score,
                stale_movement_without_durable_gain=stale_movement_without_durable_gain,
                stale_toggle_without_durable_gain=stale_toggle_without_durable_gain,
                stale_reexpose_container=stale_reexpose_container,
                stale_reacquire_inventory_churn=stale_reacquire_inventory_churn,
            )
        ):
            positive += self.config.policy.object_family_success_bonus
            object_family_evidence_contribution += self.config.policy.object_family_success_bonus
            plausibility_score += self.config.policy.object_family_success_bonus
            reasons.append("object_family_success")

        if features.prior_failure_for_object_family and not strong_prior_gain:
            object_failure_penalty = 0.5 * self.config.policy.exact_action_failure_penalty
            negative += object_failure_penalty
            object_family_evidence_contribution -= object_failure_penalty
            plausibility_score -= object_failure_penalty
            reasons.append("object_family_failure")

        if features.action_shape_is_complex_transitive and not self._is_grounded_tool_use(features):
            complex_penalty = self.config.policy.complex_transitive_penalty
            negative += complex_penalty
            plausibility_score -= complex_penalty
            reasons.append("complex_transitive")

        if self._is_speculative_tool_use(features) and not self._is_grounded_tool_use(features):
            negative += self.config.policy.speculative_tool_use_penalty
            plausibility_score -= self.config.policy.speculative_tool_use_penalty
            reasons.append("speculative_tool_use")

        features.plausibility_score = plausibility_score
        features.canonical_prior_contribution = canonical_prior_contribution
        features.object_family_evidence_contribution = object_family_evidence_contribution
        features.strategic_contribution = strategic_contribution

        return positive, negative, reasons

    def _select_diverse_top_k(
        self,
        candidates: Sequence[ActionProposal],
        *,
        candidate_count: int,
    ) -> list[ActionProposal]:
        """Greedily select a diverse top-k candidate set."""

        selected: list[ActionProposal] = []
        remaining = list(candidates)
        selected_families: set[str] = set()
        selected_objects: set[str] = set()
        movement_selected = 0
        speculative_selected = 0

        while remaining and len(selected) < candidate_count:
            best_index = 0
            best_score = float("-inf")
            for index, candidate in enumerate(remaining):
                features = candidate.features
                if features is None:
                    continue
                if (
                    features.is_movement_action
                    and movement_selected >= 1
                    and not self._movement_candidate_can_repeat(features)
                ):
                    continue
                if (
                    self._is_speculative_tool_use(features)
                    and speculative_selected >= 1
                    and not self._is_grounded_tool_use(features)
                ):
                    continue
                if self._is_duplicate_low_value_candidate(candidate, selected):
                    continue
                diversity_bonus = 0.0
                if not features.is_movement_action:
                    if features.verb_family and features.verb_family not in selected_families:
                        diversity_bonus += self.config.policy.candidate_diversity_bonus
                    if set(features.noun_targets) and set(features.noun_targets).isdisjoint(selected_objects):
                        diversity_bonus += self.config.policy.candidate_diversity_bonus
                candidate.diversity_bonus = diversity_bonus
                dynamic_score = candidate.selection_score + diversity_bonus
                if dynamic_score > best_score:
                    best_score = dynamic_score
                    best_index = index
            chosen = remaining.pop(best_index)
            chosen.selection_score += chosen.diversity_bonus
            chosen.ranking_reason = self._append_reason(
                chosen.ranking_reason,
                f"diversity_bonus={chosen.diversity_bonus:.2f}" if chosen.diversity_bonus > 0 else "",
            )
            selected.append(chosen)
            if chosen.features is not None:
                selected_families.add(chosen.features.verb_family)
                selected_objects.update(chosen.features.noun_targets)
                if chosen.features.is_movement_action:
                    movement_selected += 1
                if self._is_speculative_tool_use(chosen.features) and not self._is_grounded_tool_use(chosen.features):
                    speculative_selected += 1

        return selected

    def _is_duplicate_low_value_candidate(
        self,
        candidate: ActionProposal,
        selected: Sequence[ActionProposal],
    ) -> bool:
        """Reject near-duplicate low-value candidates from the final top-k."""

        if candidate.features is None:
            return False
        candidate_targets = set(candidate.features.noun_targets)
        for existing in selected:
            if existing.features is None:
                continue
            if (
                candidate.features.is_movement_action
                and existing.features.is_movement_action
                and candidate.selection_score <= existing.selection_score
            ):
                return True
            if (
                candidate.features.verb_family == existing.features.verb_family
                and candidate_targets
                and candidate_targets == set(existing.features.noun_targets)
                and candidate.selection_score <= existing.selection_score
            ):
                return True
        return False

    def _movement_candidate_can_repeat(self, features: ActionCandidateFeatures) -> bool:
        """Return whether more than one movement candidate should survive top-k selection."""

        return (
            features.produced_score_gain_before
            or (
                features.produced_inventory_gain_before
                and not features.is_bulk_inventory_action
                and not features.is_discard_like_action
            )
            or features.produced_affordance_gain_before
            or features.touches_supported_reflection_object
        )

    def _movement_has_prior_evidence(self, features: ActionCandidateFeatures) -> bool:
        """Return whether movement has prior durable evidence of usefulness."""

        return (
            features.is_freshly_unlocked_exit
            or
            features.produced_score_gain_before
            or (
                features.produced_inventory_gain_before
                and not features.is_bulk_inventory_action
                and not features.is_discard_like_action
            )
        )

    def _has_prior_gain(self, features: ActionCandidateFeatures) -> bool:
        """Return whether an action already demonstrated durable value."""

        return (
            features.produced_score_gain_before
            or (
                features.produced_inventory_gain_before
                and not features.is_bulk_inventory_action
                and not features.is_discard_like_action
            )
            or features.produced_affordance_gain_before
        )

    def _has_strong_prior_gain(self, features: ActionCandidateFeatures) -> bool:
        """Return whether an action demonstrated durable value worth revisiting in an exhausted family."""

        if features.produced_score_gain_before:
            return True
        if (
            features.produced_inventory_gain_before
            and not features.is_bulk_inventory_action
            and not features.is_discard_like_action
        ):
            if (
                features.prior_inventory_loss_for_object_family > 0
                and not features.prior_score_success_for_object_family
                and features.prior_affordance_gain_for_object_family <= 0
            ):
                return False
            return True
        return False

    def _soft_support_is_trustworthy(
        self,
        features: ActionCandidateFeatures,
        *,
        strong_prior_gain: bool,
        stale_reexpose_container: bool,
    ) -> bool:
        """Return whether reflection/history soft support should influence reranking."""

        if (
            features.matches_supported_try_action
            or features.touches_supported_reflection_object
        ) and not self._reflection_try_has_durable_support(
            features,
            strong_prior_gain=strong_prior_gain,
        ):
            return False

        return not (
            self._is_stale_toggle_without_durable_gain(
                features,
                strong_prior_gain=strong_prior_gain,
            )
            or self._is_stale_reacquire_inventory_churn(
                features,
                inventory_tokens=set(),
                strong_prior_gain=strong_prior_gain,
            )
            or self._is_stale_movement_without_durable_gain(
                features,
                strong_prior_gain=strong_prior_gain,
            )
            or self._is_local_scene_post_score_stale(
                features,
                strong_prior_gain=strong_prior_gain,
            )
            or stale_reexpose_container
            or (features.is_bulk_inventory_action and not strong_prior_gain)
            or (features.is_discard_like_action and not strong_prior_gain)
        )

    def _reflection_try_has_durable_support(
        self,
        features: ActionCandidateFeatures,
        *,
        strong_prior_gain: bool,
    ) -> bool:
        """Return whether reflection support is backed by concrete durable evidence."""

        if strong_prior_gain:
            return True
        if features.target_is_new_salient_object:
            return True
        if features.produced_score_gain_before or features.prior_score_success_for_verb_family:
            return True
        if (
            features.prior_score_success_for_object_family
            and self._object_family_evidence_can_transfer(
                features,
                strong_prior_gain=strong_prior_gain,
                scene_stale_after_score=features.local_scene_post_score_stale,
                stale_movement_without_durable_gain=False,
                stale_toggle_without_durable_gain=False,
                stale_reexpose_container=False,
                stale_reacquire_inventory_churn=self._is_stale_reacquire_inventory_churn(
                    features,
                    inventory_tokens=set(),
                    strong_prior_gain=strong_prior_gain,
                ),
            )
        ):
            return True
        if (
            features.produced_inventory_gain_before
            and not features.is_discard_like_action
            and not features.is_bulk_inventory_action
        ):
            return True
        return False

    def _is_stale_movement_without_durable_gain(
        self,
        features: ActionCandidateFeatures,
        *,
        strong_prior_gain: bool,
    ) -> bool:
        """Return whether movement soft support is only affordance-deep and should be ignored."""

        if not features.is_movement_action or strong_prior_gain:
            return False
        return (
            features.produced_affordance_gain_before
            or features.matches_supported_try_action
            or features.prior_success_for_verb_family
            or features.prior_success_for_exact_action
        )

    def _is_stale_toggle_without_durable_gain(
        self,
        features: ActionCandidateFeatures,
        *,
        strong_prior_gain: bool,
    ) -> bool:
        """Return whether a reversible toggle should be treated as stale local churn."""

        if not features.is_reversible_toggle or strong_prior_gain:
            return False
        if features.verb == "close":
            return True
        if features.target_is_new_salient_object and features.verb == "open":
            return False
        return (
            features.touches_exhausted_family
            or features.was_tried_before_in_local_cluster
            or features.produced_affordance_gain_before
            or (
                features.max_family_no_progress_count
                >= self.config.policy.object_family_no_progress_threshold
            )
        )

    def _is_stale_nonnew_acquire_in_inventory_scene(
        self,
        features: ActionCandidateFeatures,
        *,
        inventory_tokens: set[str],
        strong_prior_gain: bool,
    ) -> bool:
        """Return whether a non-new acquire action is likely stale post-loot churn."""

        if features.verb_family != "acquire" or strong_prior_gain:
            return False
        if features.target_is_new_salient_object:
            return False
        if not inventory_tokens:
            return False
        if features.touches_recent_affordance_object:
            return False
        if features.prior_score_success_for_object_family or features.prior_success_for_exact_action:
            return False
        return (
            features.prior_inventory_gain_for_object_family > 0
            or features.prior_inventory_loss_for_object_family > 0
            or features.touches_exhausted_family
            or features.max_family_no_progress_count
            >= self.config.policy.object_family_no_progress_threshold
            or features.target_is_container_or_openable
            or features.touches_supported_reflection_object
            or features.matches_supported_try_action
        )

    def _is_stale_reacquire_inventory_churn(
        self,
        features: ActionCandidateFeatures,
        *,
        inventory_tokens: set[str],
        strong_prior_gain: bool,
    ) -> bool:
        """Return whether reacquiring a previously dropped local object is stale churn.

        This specifically targets local object-family loops like `take leaflet` after the
        object was already acquired, dropped, and failed to produce any durable score or
        affordance improvement in the current cluster.
        """

        if features.verb_family != "acquire" or strong_prior_gain:
            return False
        if features.target_is_new_salient_object or features.touches_recent_affordance_object:
            return False
        if not features.noun_targets:
            return False
        if features.prior_score_success_for_object_family:
            return False
        if features.prior_affordance_gain_for_object_family > 0 and features.target_is_container_or_openable:
            return False
        if inventory_tokens & set(features.noun_targets):
            return False
        reacquire_seen_before = features.prior_inventory_gain_for_object_family > 0
        churn_signal = (
            features.prior_inventory_loss_for_object_family > 0
            or features.cluster_repeat_count >= 3
            or features.max_family_no_progress_count >= self.config.policy.object_family_no_progress_threshold
            or features.local_scene_post_score_stale
            or features.matches_supported_try_action
        )
        return reacquire_seen_before and churn_signal

    def _is_stale_reexpose_container_without_durable_gain(
        self,
        features: ActionCandidateFeatures,
        *,
        inventory_tokens: set[str],
        strong_prior_gain: bool,
    ) -> bool:
        """Return whether reopening a local container is stale after its contents already paid off."""

        if features.verb != "open" or strong_prior_gain:
            return False
        if not features.target_is_container_or_openable:
            return False
        if features.target_is_new_salient_object or features.touches_recent_affordance_object:
            return False
        if not inventory_tokens:
            return False
        if features.prior_score_success_for_object_family or features.prior_success_for_exact_action:
            return False
        return (
            features.produced_affordance_gain_before
            or features.prior_success_for_object_family
            or features.matches_supported_try_action
            or features.touches_supported_reflection_object
        )

    def _should_prefer_exit_escape(
        self,
        candidates: Sequence[ActionProposal],
        inventory_tokens: set[str],
    ) -> bool:
        """Return whether the local state looks exhausted enough to prefer leaving."""

        if not inventory_tokens:
            return False
        has_new_object_priority = False
        has_recent_affordance_followup = False
        has_stale_local_churn = False
        has_exit = False
        for candidate in candidates:
            if candidate.features is None:
                continue
            if candidate.features.is_movement_action:
                has_exit = True
                continue
            if candidate.features.touches_newly_salient_object and not (
                candidate.features.local_scene_post_score_stale
                and not (
                    candidate.features.touches_recent_affordance_object
                    or candidate.features.touches_strategic_object
                    or candidate.features.matches_strategic_try_action
                )
            ) and not self._is_post_score_low_value_new_object_acquire(
                candidate.features,
                inventory_tokens=inventory_tokens,
                strong_prior_gain=self._has_strong_prior_gain(candidate.features),
            ):
                has_new_object_priority = True
            if (
                candidate.features.touches_recent_affordance_object
                and not candidate.features.is_discard_like_action
                and not self._is_stale_toggle_without_durable_gain(
                    candidate.features,
                    strong_prior_gain=self._has_strong_prior_gain(candidate.features),
                )
                and not self._is_stale_reacquire_inventory_churn(
                    candidate.features,
                    inventory_tokens=inventory_tokens,
                    strong_prior_gain=self._has_strong_prior_gain(candidate.features),
                )
                and not self._is_stale_nonnew_acquire_in_inventory_scene(
                    candidate.features,
                    inventory_tokens=inventory_tokens,
                    strong_prior_gain=self._has_strong_prior_gain(candidate.features),
                )
            ):
                has_recent_affordance_followup = True
            normalized_action = normalize_parser_action(candidate.action)
            if (
                candidate.features.is_reversible_toggle
                or candidate.features.is_bulk_inventory_action
                or normalized_action.startswith("put ")
                or candidate.features.touches_exhausted_family
                or candidate.features.local_scene_post_score_stale
            ):
                has_stale_local_churn = True
        return (
            has_exit
            and has_stale_local_churn
            and not has_new_object_priority
            and not has_recent_affordance_followup
        )

    def _is_local_scene_post_score_stale(
        self,
        features: ActionCandidateFeatures,
        *,
        strong_prior_gain: bool,
    ) -> bool:
        """Return whether the current visible object scene already paid off and should be de-emphasized."""

        if strong_prior_gain or features.is_movement_action:
            return False
        if not features.scene_has_score_harvested:
            return False
        if features.target_is_new_salient_object:
            return False
        if not features.noun_targets:
            return False
        return True

    def _is_post_score_inventory_object_disruption(
        self,
        features: ActionCandidateFeatures,
        *,
        inventory_tokens: set[str],
        strong_prior_gain: bool,
    ) -> bool:
        """Return whether an action risks churning a newly valuable carried object after scoring.

        This is intentionally narrow: once a scene already yielded score through an inventory
        object family, follow-on manipulation of that same carried object should lose to exits
        or broader mapping unless there is specific evidence that the exact action is valuable.
        """

        if strong_prior_gain or features.is_movement_action:
            return False
        if not features.scene_has_score_harvested or not features.prior_score_success_for_object_family:
            return False
        if not features.noun_targets or not inventory_tokens:
            return False
        if not (set(features.noun_targets) & inventory_tokens):
            return False
        if features.target_is_new_salient_object or features.touches_recent_affordance_object:
            return False
        if features.prior_success_for_exact_action and features.produced_score_gain_before:
            return False
        if features.verb_family == "inspect" and not features.action_shape_is_complex_transitive:
            return False
        return (
            features.verb_family in {"acquire", "use", "aggressive"}
            or features.is_reversible_toggle
            or features.action_shape_is_complex_transitive
            or features.uses_inventory_object
        )

    def _is_post_score_low_value_new_object_acquire(
        self,
        features: ActionCandidateFeatures,
        *,
        inventory_tokens: set[str],
        strong_prior_gain: bool,
    ) -> bool:
        """Return whether a new-looking local acquire should yield to leaving a scored scene.

        This targets cases like grabbing nearby container clutter after already securing a
        higher-value inventory state. Truly fresh payoff objects still bypass this rule.
        """

        if strong_prior_gain or not features.scene_has_score_harvested:
            return False
        if not features.target_is_new_salient_object or features.verb_family != "acquire":
            return False
        if not inventory_tokens or not features.target_is_container_or_openable:
            return False
        if features.touches_recent_affordance_object or features.touches_strategic_object:
            return False
        if features.matches_strategic_try_action or features.prior_score_success_for_object_family:
            return False
        return True

    def _object_family_evidence_can_transfer(
        self,
        features: ActionCandidateFeatures,
        *,
        strong_prior_gain: bool,
        scene_stale_after_score: bool,
        stale_movement_without_durable_gain: bool,
        stale_toggle_without_durable_gain: bool,
        stale_reexpose_container: bool,
        stale_reacquire_inventory_churn: bool,
    ) -> bool:
        """Return whether object-family success should transfer to this action shape."""

        if not (
            strong_prior_gain
            or (
                not stale_movement_without_durable_gain
                and not stale_toggle_without_durable_gain
                and not stale_reacquire_inventory_churn
                and not stale_reexpose_container
                and not scene_stale_after_score
            )
        ):
            return False
        if features.is_discard_like_action or features.is_bulk_inventory_action:
            return False
        if features.verb_family in {"aggressive", "use"}:
            return features.prior_success_for_exact_action or features.prior_success_for_verb_family
        if features.action_shape_is_complex_transitive and features.verb_family not in {"inspect", "access"}:
            return features.prior_success_for_exact_action or features.prior_success_for_verb_family
        if features.verb_family == "movement":
            return features.produced_score_gain_before or features.prior_score_success_for_verb_family
        if features.verb_family == "acquire" and not features.target_is_new_salient_object:
            return (
                features.prior_score_success_for_object_family
                or features.prior_success_for_exact_action
                or features.produced_inventory_gain_before
            )
        return True

    def _is_ungrounded_other_action(self, features: ActionCandidateFeatures) -> bool:
        """Return whether a generic non-object action lacks enough grounding to rank highly."""

        if features.verb_family != "other":
            return False
        if self._has_strong_prior_gain(features):
            return False
        if (
            features.prior_success_for_exact_action
            or features.prior_success_for_verb_family
        ):
            return False
        return True

    def _is_object_centric_priority_action(
        self,
        action: str,
        features: ActionCandidateFeatures,
    ) -> bool:
        """Return whether an action should receive strong object-centric priority."""

        normalized = normalize_parser_action(action)
        if normalized.startswith(_OBJECT_PRIORITY_PREFIXES):
            return bool(features.noun_targets)
        if features.is_bulk_inventory_action or features.is_discard_like_action:
            return False
        if features.verb == "close":
            return False
        return (
            features.verb in _OBJECT_PRIORITY_VERBS
            or features.verb_family in {"inspect", "acquire"}
            or (features.verb_family == "access" and features.verb == "open")
        ) and bool(features.noun_targets)

    def _canonical_new_object_bonus(self, features: ActionCandidateFeatures) -> float:
        """Return a canonical IF preference bonus for newly revealed objects."""

        if not features.target_is_new_salient_object:
            return 0.0
        if features.verb_family == "inspect":
            return self.config.policy.inspect_new_object_bonus
        if features.verb_family == "acquire":
            return self.config.policy.acquire_new_object_bonus
        if features.verb_family == "access" and features.verb == "open":
            return (
                self.config.policy.access_new_object_bonus
                if features.target_is_container_or_openable
                else 0.5 * self.config.policy.access_new_object_bonus
            )
        if features.verb_family == "access" and features.verb == "close":
            return -self.config.policy.access_new_object_bonus
        if features.verb_family == "movement":
            return -0.5 * self.config.policy.inspect_new_object_bonus
        if features.verb_family in {"use", "aggressive"}:
            return -self.config.policy.inspect_new_object_bonus
        return 0.0

    def _is_speculative_tool_use(self, features: ActionCandidateFeatures) -> bool:
        """Return whether an action is a likely speculative multi-object/tool command."""

        return (
            features.verb_family in {"use", "aggressive"}
            or (
                features.action_shape_is_complex_transitive
                and features.verb_family not in {"inspect", "access"}
            )
            or normalize_parser_action(features.action_text).startswith(_SPECULATIVE_TRANSITIVE_PREFIXES)
        )

    def _is_grounded_tool_use(self, features: ActionCandidateFeatures) -> bool:
        """Return whether a speculative tool-use action has enough prior evidence."""

        return (
            features.produced_score_gain_before
            or (
                features.produced_inventory_gain_before
                and not features.is_bulk_inventory_action
                and not features.is_discard_like_action
            )
            or features.prior_success_for_exact_action
            or features.prior_success_for_verb_family
            or features.prior_score_success_for_object_family
            or (features.target_is_container_or_openable and features.verb_family == "access")
            or (features.target_is_readable_candidate and features.verb == "read")
        )

    def _movement_targets_landmark(
        self,
        action: str,
        *,
        visible_nouns: set[str],
    ) -> bool:
        """Return whether a movement action aligns with a salient climbable or enterable landmark."""

        normalized_action = normalize_parser_action(action)
        if not normalized_action:
            return False
        if normalized_action in {"up", "climb", "climb tree", "climb up"}:
            return bool(visible_nouns & _CLIMBABLE_LANDMARK_TOKENS)
        if normalized_action.startswith(("enter", "in ", "go in", "go through", "go to")):
            return bool(visible_nouns & _LANDMARK_ENTRY_TOKENS)
        return False

    def _merge_candidate_pool(
        self,
        valid_actions: list[str],
        heuristic_seed_order: list[str],
        llm_ranked_actions: list[str],
    ) -> list[str]:
        """Merge the valid-action pool with any LLM-ranked subset while preserving full coverage."""

        ordered: list[str] = []
        seen: set[str] = set()
        for action in [*llm_ranked_actions, *heuristic_seed_order, *valid_actions]:
            normalized = normalize_parser_action(action)
            if not normalized or normalized in seen:
                continue
            if normalized not in {normalize_parser_action(candidate) for candidate in valid_actions}:
                continue
            ordered.append(action)
            seen.add(normalized)
        return ordered

    def _supported_object_tokens(
        self,
        *,
        supported_reflection_objects: Iterable[str],
        supported_try_actions: Iterable[str],
    ) -> set[str]:
        """Build a small reflection-grounded object token set."""

        tokens = self._normalize_token_list(supported_reflection_objects)
        for action in supported_try_actions:
            tokens.update(action_target_tokens(action, self.config.policy.inverse_action_pairs))
        return tokens

    def _local_world_model_context(
        self,
        *,
        root_state_id: str,
        local_world_model: LocalWorldModel | None,
    ) -> tuple[LocalWorldModelPromptSummary, list[str], list[str], list[str]]:
        """Build bounded prompt context and action guidance from a local world model."""

        summary = self.prompt_manager.summarize_local_world_model(
            local_world_model,
            root_state_id=root_state_id,
        )
        if local_world_model is None:
            return summary, [], [], []
        no_achievement_revisit_streak = int(local_world_model.metadata.get("no_achievement_revisit_streak", 0))
        replay_saturation_level = int(local_world_model.metadata.get("depth_progression_replay_saturation_level", 0))
        productive_root = bool(local_world_model.metadata.get("depth_progression_productive_root", False))
        saturated_try_actions = {
            normalize_parser_action(action)
            for action in local_world_model.metadata.get("depth_progression_saturated_try_actions", [])
            if isinstance(action, str) and normalize_parser_action(action)
        }
        suppress_try_guidance = (
            no_achievement_revisit_streak >= self.config.policy.local_guidance_stale_revisit_threshold
        )
        try_actions = [
            item.action.strip()
            for item in local_world_model.action_priors
            if item.action.strip()
        ]
        avoid_actions = [
            item.action.strip()
            for item in local_world_model.action_antipriors
            if item.action.strip()
        ]
        salient_objects = [
            item.object_text.strip()
            for item in local_world_model.inferred_affordances
            if item.object_text.strip()
        ]
        for hint in local_world_model.accumulated_advantage_hints[-2:]:
            try_actions.extend(item for item in hint.action_preferences if item.strip())
            avoid_actions.extend(item for item in hint.action_avoidances if item.strip())
        if suppress_try_guidance:
            summary.summary_text = ""
            summary.recent_advantage_hints = []
            summary.action_priors = []
            try_actions = []
        elif productive_root and replay_saturation_level > 0 and saturated_try_actions:
            try_actions = [
                action
                for action in try_actions
                if normalize_parser_action(action) not in saturated_try_actions
            ]
            summary.action_priors = [
                action
                for action in summary.action_priors
                if normalize_parser_action(action) not in saturated_try_actions
            ]
        return (
            summary,
            self._normalize_action_list(try_actions),
            self._normalize_action_list(avoid_actions),
            sorted(self._normalize_token_list(salient_objects)),
        )

    def _normalize_action_list(self, actions: Iterable[str]) -> list[str]:
        """Normalize and deduplicate a list of parser actions."""

        normalized: list[str] = []
        seen: set[str] = set()
        for action in actions:
            canonical = normalize_parser_action(action)
            if canonical and canonical not in seen:
                normalized.append(canonical)
                seen.add(canonical)
        return normalized

    def _normalize_token_list(self, values: Iterable[str]) -> set[str]:
        """Normalize and deduplicate noun-like tokens."""

        tokens: set[str] = set()
        for value in values:
            tokens.update(
                {
                    token
                    for token in re.findall(r"[a-z]+", normalize_parser_action(value))
                    if len(token) > 2
                }
            )
        return tokens

    def _append_reason(self, current: str, new_reason: str) -> str:
        """Append one ranking reason if it is non-empty and not already present."""

        if not new_reason:
            return current
        if not current:
            return new_reason
        if new_reason in current:
            return current
        return f"{current}, {new_reason}"

    def _is_plausible_action(self, action: str) -> bool:
        """Return whether parsed free-form text looks like a parser command."""

        normalized = normalize_parser_action(action)
        if not normalized:
            return False
        if len(normalized.split()) > 6:
            return False
        disallowed_prefixes = (
            "i ",
            "i'm ",
            "im ",
            "maybe ",
            "perhaps ",
            "try ",
            "not sure",
            "unclear",
        )
        return not normalized.startswith(disallowed_prefixes)
