"""Multi-path Advantage Reflection (MAR) over local rollouts from one root state.

This module keeps the paper-faithful intent explicit: compare multiple rollouts
from the same restored root, infer intermediate advantages as typed hints, and
update a persistent local world model keyed by that root state.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
import re
from typing import Iterable, Mapping, Sequence

from zork_agent.config import ProjectConfig
from zork_agent.llm.base import BaseLLMClient
from zork_agent.llm.prompts import PromptManager
from zork_agent.types import (
    ActionBiasRecord,
    AdvantageHint,
    AdvantagePoint,
    AffordanceRecord,
    LocalBranchOutcome,
    LocalWorldModel,
    MARInferenceResult,
    SubgoalRecord,
    action_target_tokens,
    canonical_if_verb_family,
    normalize_parser_action,
)

_LOGGER = logging.getLogger("zork_agent.mar")


class MultiPathAdvantageReflector:
    """Infer typed local advantages by comparing multiple local rollouts."""

    def __init__(self, config: ProjectConfig, prompt_manager: PromptManager, llm_client: BaseLLMClient | None):
        self.config = config
        self.prompt_manager = prompt_manager
        self.llm_client = llm_client

    def infer_advantage_hint(
        self,
        *,
        root_state_id: str,
        branches: Sequence[LocalBranchOutcome],
        existing_local_world_model: LocalWorldModel | None = None,
        analysis_id: str | None = None,
    ) -> MARInferenceResult:
        """Infer one advantage hint from multiple branches of the same root state."""

        comparable_branches = [
            branch
            for branch in branches
            if branch.actions_taken
        ]
        branch_ids = [self._branch_id(branch) for branch in comparable_branches]
        prompt_input = self._build_prompt_input(
            root_state_id=root_state_id,
            branches=comparable_branches,
            existing_local_world_model=existing_local_world_model,
        )
        fallback = self._heuristic_inference(
            root_state_id=root_state_id,
            branches=comparable_branches,
            analysis_id=analysis_id,
            prompt_input=prompt_input,
        )

        if len(comparable_branches) < 2 or self.llm_client is None:
            return fallback

        try:
            system_prompt = self.prompt_manager.render_system(game_id=self.config.experiment.game_id)
            user_prompt = self.prompt_manager.render_mar_advantage(
                branch_comparison_block=self._branch_comparison_block(comparable_branches),
                existing_local_model_block=self._existing_model_block(existing_local_world_model),
                grounding_constraints=self._grounding_constraints(comparable_branches),
                analysis_debug_mode=self.config.llm.analysis_debug_mode,
            )
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.config.llm.model_name,
                temperature=0.0,
                max_tokens=(
                    self.config.llm.mar_debug_max_tokens
                    if self.config.llm.analysis_debug_mode
                    else self.config.llm.mar_max_tokens
                ),
                timeout_seconds=self.config.llm.request_timeout_seconds,
                response_format=(
                    None
                    if self.config.llm.analysis_debug_mode
                    else _mar_response_format()
                ),
                metadata={"task": "mar_advantage"},
            )
        except Exception as exc:
            _LOGGER.warning("MAR fell back after LLM request failure: %s", exc)
            fallback.raw_completion = ""
            fallback.parse_error = f"llm_request_failed: {exc}"
            fallback.used_fallback = True
            return fallback

        parsed = self._parse_mar_output(
            raw_output=response.text,
            root_state_id=root_state_id,
            compared_branch_ids=branch_ids,
            analysis_id=analysis_id,
        )
        if parsed.advantage_hint is None:
            _LOGGER.warning("MAR fell back after malformed LLM output.")
            fallback.raw_completion = response.text
            fallback.parse_error = parsed.parse_error or "mar_parse_failed"
            fallback.used_fallback = True
            return fallback
        parsed.raw_completion = response.text
        return parsed

    def update_local_world_model(
        self,
        *,
        root_state_id: str,
        existing_local_world_model: LocalWorldModel | None,
        inference_result: MARInferenceResult,
    ) -> LocalWorldModel:
        """Merge one MAR inference result into the persistent local world model."""

        model = existing_local_world_model or LocalWorldModel(root_state_id=root_state_id)
        if inference_result.advantage_hint is not None:
            if not self._contains_equivalent_hint(model, inference_result.advantage_hint):
                model.accumulated_advantage_hints.append(inference_result.advantage_hint)

        model.discovered_subgoals = self._merge_subgoals(
            model.discovered_subgoals,
            inference_result.discovered_subgoals,
        )
        model.inferred_affordances = self._merge_affordances(
            model.inferred_affordances,
            inference_result.inferred_affordances,
        )
        if inference_result.advantage_hint is not None:
            model.action_priors = self._merge_action_biases(
                model.action_priors,
                inference_result.advantage_hint.action_preferences,
                rationale=inference_result.advantage_hint.textual_reasoning,
                weight=max(0.5, inference_result.advantage_hint.confidence),
            )
            model.action_antipriors = self._merge_action_biases(
                model.action_antipriors,
                inference_result.advantage_hint.action_avoidances,
                rationale=inference_result.advantage_hint.textual_reasoning,
                weight=max(0.5, inference_result.advantage_hint.confidence),
            )

        model.last_updated_timestamp = datetime.now(timezone.utc).isoformat()
        model.metadata["last_analysis_id"] = str(inference_result.metadata.get("analysis_id", ""))
        model.metadata["mar_used_fallback"] = inference_result.used_fallback
        model.validate_invariants()
        return model

    def guidance_from_local_world_model(
        self,
        local_world_model: LocalWorldModel | None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Project a local world model into action/object guidance for reranking."""

        if local_world_model is None:
            return [], [], []

        try_actions = [bias.action for bias in sorted(local_world_model.action_priors, key=lambda item: (-item.weight, item.action))]
        avoid_actions = [bias.action for bias in sorted(local_world_model.action_antipriors, key=lambda item: (-item.weight, item.action))]
        salient_objects = [record.object_text for record in local_world_model.inferred_affordances]
        return _dedupe_preserve_order(try_actions), _dedupe_preserve_order(avoid_actions), _dedupe_preserve_order(salient_objects)

    def _build_prompt_input(
        self,
        *,
        root_state_id: str,
        branches: Sequence[LocalBranchOutcome],
        existing_local_world_model: LocalWorldModel | None,
    ) -> str:
        """Build a researcher-inspectable prompt snapshot for MAR."""

        return self.prompt_manager.render_mar_advantage(
            branch_comparison_block=self._branch_comparison_block(branches),
            existing_local_model_block=self._existing_model_block(existing_local_world_model),
            grounding_constraints=self._grounding_constraints(branches),
            analysis_debug_mode=self.config.llm.analysis_debug_mode,
        )

    def _branch_comparison_block(self, branches: Sequence[LocalBranchOutcome]) -> str:
        """Render compact branch summaries for MAR prompting."""

        lines: list[str] = []
        for branch in branches[:6]:
            lines.append(
                f"{self._branch_id(branch)} | actions={' -> '.join(branch.actions_taken) or 'none'} "
                f"| progress={branch.branch_progress_score:.2f} | score_delta={branch.score_change} "
                f"| inventory_gain={branch.persistent_inventory_gain_count} "
                f"| affordance_gain={branch.persistent_affordance_gain} "
                f"| exit_gain={branch.persistent_exit_gain_count} "
                f"| loop_events={branch.loop_event_count} | final={branch.final_observation}"
            )
        return "\n".join(lines) or "(none)"

    def _existing_model_block(self, local_world_model: LocalWorldModel | None) -> str:
        """Render a compact summary of the existing local world model."""

        if local_world_model is None:
            return "(none)"
        try_actions, avoid_actions, objects = self.guidance_from_local_world_model(local_world_model)
        return (
            f"prior_try_actions={', '.join(try_actions[:6]) or '(none)'}\n"
            f"prior_avoid_actions={', '.join(avoid_actions[:6]) or '(none)'}\n"
            f"known_objects={', '.join(objects[:6]) or '(none)'}\n"
            f"hint_count={len(local_world_model.accumulated_advantage_hints)}"
        )

    def _grounding_constraints(self, branches: Sequence[LocalBranchOutcome]) -> str:
        """Render compact grounding constraints for MAR prompting."""

        productive_actions: list[str] = []
        low_value_actions: list[str] = []
        for branch in branches:
            for event in branch.metadata.get("action_events", []):
                if not isinstance(event, dict):
                    continue
                action = str(event.get("action", "")).strip()
                if not action:
                    continue
                if (
                    float(event.get("score_delta", 0.0)) > 0
                    or bool(event.get("inventory_gained", False))
                    or int(event.get("persistent_affordance_gain", 0)) > 0
                    or int(event.get("persistent_exit_gain_count", 0)) > 0
                ):
                    productive_actions.append(action)
                elif bool(event.get("loop_detected", False)) or float(event.get("movement_penalty", 0.0)) > 0:
                    low_value_actions.append(action)
        return (
            f"Supported productive actions: {', '.join(_dedupe_preserve_order(productive_actions)) or '(none)'}\n"
            f"Supported low-value actions: {', '.join(_dedupe_preserve_order(low_value_actions)) or '(none)'}"
        )

    def _heuristic_inference(
        self,
        *,
        root_state_id: str,
        branches: Sequence[LocalBranchOutcome],
        analysis_id: str | None,
        prompt_input: str,
    ) -> MARInferenceResult:
        """Build a deterministic advantage hint from observed branch differences."""

        branch_ids = [self._branch_id(branch) for branch in branches]
        if not branches:
            return MARInferenceResult(
                root_state_id=root_state_id,
                compared_branch_ids=branch_ids,
                prompt_input=prompt_input,
                parse_error="no_branches_to_compare",
                used_fallback=True,
                metadata={"analysis_id": analysis_id or self._default_analysis_id(root_state_id)},
            )

        sorted_branches = sorted(branches, key=self._branch_sort_key, reverse=True)
        best_branch = sorted_branches[0]
        weaker_branches = sorted_branches[1:] or [sorted_branches[0]]
        key_points = self._heuristic_advantage_points(root_state_id, best_branch)
        action_preferences = _dedupe_preserve_order(
            [point.action for point in key_points if point.action]
            + ([best_branch.first_durable_gain_action] if best_branch.first_durable_gain_action else [])
            + ([best_branch.last_durable_progress_action] if best_branch.last_durable_progress_action else [])
        )[:4]
        action_avoidances = _dedupe_preserve_order(self._low_value_actions(weaker_branches))[:4]
        subgoals = self._heuristic_subgoals(best_branch, key_points)
        affordances = self._heuristic_affordances(best_branch, key_points)
        reasoning = (
            f"Compared {len(branches)} rollouts from {root_state_id}. "
            f"Best branch {self._branch_id(best_branch)} outperformed alternatives through "
            f"{', '.join(action_preferences) or 'its key actions'}."
        )
        confidence = min(
            1.0,
            max(
                0.2,
                0.45
                + 0.1 * len(key_points)
                + 0.05 * max(0.0, best_branch.branch_progress_score - weaker_branches[0].branch_progress_score),
            ),
        )
        hint = AdvantageHint(
            root_state_id=root_state_id,
            compared_trajectory_ids=branch_ids,
            key_state_action_points=key_points,
            intermediate_advantage_summaries=[
                point.outcome_delta for point in key_points if point.outcome_delta
            ][:4],
            action_preferences=action_preferences,
            action_avoidances=action_avoidances,
            textual_reasoning=reasoning,
            confidence=confidence,
            metadata={"analysis_mode": "heuristic_fallback"},
        )
        return MARInferenceResult(
            root_state_id=root_state_id,
            compared_branch_ids=branch_ids,
            advantage_hint=hint,
            discovered_subgoals=subgoals,
            inferred_affordances=affordances,
            prompt_input=prompt_input,
            used_fallback=True,
            metadata={"analysis_id": analysis_id or self._default_analysis_id(root_state_id)},
        )

    def _parse_mar_output(
        self,
        *,
        raw_output: str,
        root_state_id: str,
        compared_branch_ids: Sequence[str],
        analysis_id: str | None,
    ) -> MARInferenceResult:
        """Parse structured JSON MAR output into typed structures."""

        payload = _extract_structured_json_object(raw_output)
        key_points: list[AdvantagePoint] = []
        action_preferences = _as_string_list(payload.get("prefer"))
        action_avoidances = _as_string_list(payload.get("avoid"))
        subgoals: list[SubgoalRecord] = []
        affordances: list[AffordanceRecord] = []
        reasoning_text = " ".join(str(payload.get("reasoning", "")).split()).strip()

        for key_point_payload in _as_mapping_list(payload.get("key_points")):
            point = self._parse_key_point(root_state_id, key_point_payload)
            if point is not None:
                key_points.append(point)
        for description in _as_string_list(payload.get("subgoals")):
            subgoals.append(
                SubgoalRecord(
                    subgoal_id=f"{root_state_id}:subgoal:{len(subgoals)}",
                    description=description,
                    confidence=0.6,
                )
            )
        for affordance_payload in _as_affordance_payloads(payload.get("affordances")):
            affordance = self._parse_affordance(affordance_payload)
            if affordance is not None:
                affordances.append(affordance)

        if not key_points and not action_preferences:
            return MARInferenceResult(
                root_state_id=root_state_id,
                compared_branch_ids=list(compared_branch_ids),
                prompt_input="",
                raw_completion=raw_output,
                parse_error="mar_output_missing_key_points_and_preferences",
                used_fallback=False,
                metadata={"analysis_id": analysis_id or self._default_analysis_id(root_state_id)},
            )

        hint = AdvantageHint(
            root_state_id=root_state_id,
            compared_trajectory_ids=list(compared_branch_ids),
            key_state_action_points=key_points,
            intermediate_advantage_summaries=[point.outcome_delta for point in key_points if point.outcome_delta][:4],
            action_preferences=_dedupe_preserve_order(action_preferences)[:4],
            action_avoidances=_dedupe_preserve_order(action_avoidances)[:4],
            textual_reasoning=reasoning_text or "Inferred MAR advantage hint.",
            confidence=max([point.confidence for point in key_points], default=0.55),
            metadata={"analysis_mode": "llm"},
        )
        return MARInferenceResult(
            root_state_id=root_state_id,
            compared_branch_ids=list(compared_branch_ids),
            advantage_hint=hint,
            discovered_subgoals=subgoals,
            inferred_affordances=affordances,
            prompt_input="",
            raw_completion=raw_output,
            metadata={"analysis_id": analysis_id or self._default_analysis_id(root_state_id)},
        )

    def _parse_key_point(self, root_state_id: str, payload: Mapping[str, object]) -> AdvantagePoint | None:
        """Parse one JSON key-point object."""

        action = str(payload.get("action", "")).strip()
        if not action:
            return None
        support_ids = _as_string_list(payload.get("support"))
        return AdvantagePoint(
            state_id=root_state_id,
            action=action,
            inferred_advantage=float(payload.get("advantage", 0.0) or 0.0),
            outcome_delta=str(payload.get("outcome", "")),
            textual_rationale=str(payload.get("rationale", "")),
            support_trajectory_ids=support_ids,
            confidence=float(payload.get("confidence", 0.0) or 0.0),
        )

    def _parse_affordance(self, payload: Mapping[str, object]) -> AffordanceRecord | None:
        """Parse one JSON affordance object."""

        object_text = " ".join(str(payload.get("object", "")).split()).strip()
        affordance = " ".join(str(payload.get("verb", "")).split()).strip()
        if not object_text or not affordance:
            return None
        return AffordanceRecord(
            object_text=object_text,
            affordance=affordance,
            supporting_action="",
            confidence=0.6,
        )

    def _heuristic_advantage_points(
        self,
        root_state_id: str,
        best_branch: LocalBranchOutcome,
    ) -> list[AdvantagePoint]:
        """Infer key advantage points from the best branch's observed action events."""

        points: list[AdvantagePoint] = []
        branch_id = self._branch_id(best_branch)
        for event in best_branch.metadata.get("action_events", []):
            if not isinstance(event, dict):
                continue
            action = str(event.get("action", "")).strip()
            if not action:
                continue
            score_delta = float(event.get("score_delta", 0.0))
            inventory_gained = bool(event.get("inventory_gained", False))
            affordance_gain = int(event.get("persistent_affordance_gain", 0))
            exit_gain = int(event.get("persistent_exit_gain_count", 0))
            if score_delta <= 0.0 and not inventory_gained and affordance_gain <= 0 and exit_gain <= 0:
                continue
            advantage = score_delta + (1.0 if inventory_gained else 0.0) + 0.5 * affordance_gain + 0.5 * exit_gain
            points.append(
                AdvantagePoint(
                    state_id=root_state_id,
                    action=action,
                    inferred_advantage=advantage,
                    outcome_delta=self._event_outcome_delta(event),
                    textual_rationale=self._event_rationale(event),
                    support_trajectory_ids=[branch_id],
                    confidence=min(1.0, 0.55 + 0.1 * advantage),
                )
            )
        return points[:4]

    def _event_outcome_delta(self, event: dict[str, object]) -> str:
        """Render a compact observed outcome delta for one action event."""

        parts: list[str] = []
        if float(event.get("score_delta", 0.0)) > 0:
            parts.append(f"score +{int(float(event.get('score_delta', 0.0)))}")
        if bool(event.get("inventory_gained", False)):
            parts.append("inventory gain")
        if int(event.get("persistent_affordance_gain", 0)) > 0:
            parts.append(f"affordance +{int(event.get('persistent_affordance_gain', 0))}")
        if int(event.get("persistent_exit_gain_count", 0)) > 0:
            parts.append(f"exit +{int(event.get('persistent_exit_gain_count', 0))}")
        return ", ".join(parts) or "local progress"

    def _event_rationale(self, event: dict[str, object]) -> str:
        """Render a compact rationale for a productive branch event."""

        action = str(event.get("action", "")).strip()
        outcome = self._event_outcome_delta(event)
        return f"{action} led to {outcome}".strip()

    def _low_value_actions(self, branches: Iterable[LocalBranchOutcome]) -> list[str]:
        """Collect repeated low-value actions from weaker branches."""

        low_value: list[str] = []
        for branch in branches:
            for event in branch.metadata.get("action_events", []):
                if not isinstance(event, dict):
                    continue
                action = str(event.get("action", "")).strip()
                if not action:
                    continue
                if (
                    bool(event.get("loop_detected", False))
                    or float(event.get("movement_penalty", 0.0)) > 0
                    or (
                        float(event.get("score_delta", 0.0)) <= 0
                        and not bool(event.get("inventory_gained", False))
                        and int(event.get("persistent_affordance_gain", 0)) <= 0
                        and int(event.get("persistent_exit_gain_count", 0)) <= 0
                    )
                ):
                    low_value.append(action)
        counts = Counter(normalize_parser_action(action) for action in low_value if action)
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [action for action, _ in ordered]

    def _heuristic_subgoals(
        self,
        best_branch: LocalBranchOutcome,
        key_points: Sequence[AdvantagePoint],
    ) -> list[SubgoalRecord]:
        """Derive lightweight subgoals from productive branch actions."""

        subgoals: list[SubgoalRecord] = []
        for index, point in enumerate(key_points[:3]):
            description = f"Reach a state where `{point.action}` is available and worthwhile."
            subgoals.append(
                SubgoalRecord(
                    subgoal_id=f"{self._branch_id(best_branch)}:subgoal:{index}",
                    description=description,
                    supporting_state_ids=[point.state_id],
                    confidence=point.confidence,
                )
            )
        return subgoals

    def _heuristic_affordances(
        self,
        best_branch: LocalBranchOutcome,
        key_points: Sequence[AdvantagePoint],
    ) -> list[AffordanceRecord]:
        """Derive lightweight affordances from productive actions."""

        affordances: list[AffordanceRecord] = []
        for point in key_points:
            family = canonical_if_verb_family(point.action)
            for noun in action_target_tokens(point.action):
                affordances.append(
                    AffordanceRecord(
                        object_text=noun,
                        affordance=family,
                        supporting_action=point.action,
                        confidence=point.confidence,
                    )
                )
        return self._merge_affordances([], affordances)

    def _contains_equivalent_hint(self, model: LocalWorldModel, hint: AdvantageHint) -> bool:
        """Return whether an equivalent advantage hint is already stored."""

        signature = (
            tuple(sorted(normalize_parser_action(action) for action in hint.action_preferences)),
            tuple(sorted(normalize_parser_action(action) for action in hint.action_avoidances)),
            tuple(sorted(normalize_parser_action(point.action) for point in hint.key_state_action_points)),
        )
        for existing in model.accumulated_advantage_hints:
            existing_signature = (
                tuple(sorted(normalize_parser_action(action) for action in existing.action_preferences)),
                tuple(sorted(normalize_parser_action(action) for action in existing.action_avoidances)),
                tuple(sorted(normalize_parser_action(point.action) for point in existing.key_state_action_points)),
            )
            if existing_signature == signature:
                return True
        return False

    def _merge_subgoals(
        self,
        existing: Sequence[SubgoalRecord],
        new_items: Sequence[SubgoalRecord],
    ) -> list[SubgoalRecord]:
        """Merge discovered subgoals by description."""

        merged: dict[str, SubgoalRecord] = {item.description.lower(): item for item in existing}
        for item in new_items:
            key = item.description.lower()
            if key not in merged or item.confidence > merged[key].confidence:
                merged[key] = item
        return list(merged.values())

    def _merge_affordances(
        self,
        existing: Sequence[AffordanceRecord],
        new_items: Sequence[AffordanceRecord],
    ) -> list[AffordanceRecord]:
        """Merge affordances by object/affordance pair."""

        merged: dict[tuple[str, str], AffordanceRecord] = {
            (item.object_text.lower(), item.affordance.lower()): item for item in existing
        }
        for item in new_items:
            key = (item.object_text.lower(), item.affordance.lower())
            if key not in merged or item.confidence > merged[key].confidence:
                merged[key] = item
        return list(merged.values())

    def _merge_action_biases(
        self,
        existing: Sequence[ActionBiasRecord],
        actions: Sequence[str],
        *,
        rationale: str,
        weight: float,
    ) -> list[ActionBiasRecord]:
        """Merge action priors or anti-priors by normalized action text."""

        merged: dict[str, ActionBiasRecord] = {
            normalize_parser_action(item.action): item for item in existing
        }
        for action in actions:
            normalized = normalize_parser_action(action)
            if not normalized:
                continue
            if normalized in merged:
                merged[normalized].support_count += 1
                merged[normalized].weight = max(merged[normalized].weight, weight)
                if rationale and not merged[normalized].rationale:
                    merged[normalized].rationale = rationale
            else:
                merged[normalized] = ActionBiasRecord(
                    action=normalized,
                    weight=weight,
                    rationale=rationale,
                    support_count=1,
                )
        return list(merged.values())

    def _branch_sort_key(self, branch: LocalBranchOutcome) -> tuple[float, float, int, int]:
        """Return the same explicit signals used elsewhere to compare branches."""

        return (
            float(branch.branch_progress_score),
            float(branch.total_reward),
            int(branch.score_change),
            int(branch.persistent_affordance_gain),
        )

    def _branch_id(self, branch: LocalBranchOutcome) -> str:
        """Return a stable branch identifier string."""

        return f"branch-{branch.branch_index}"

    def _default_analysis_id(self, root_state_id: str) -> str:
        """Return a timestamped MAR analysis id."""

        safe_root = re.sub(r"[^A-Za-z0-9._-]+", "_", root_state_id.strip()) or "root"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"mar-{safe_root}-{timestamp}"


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    """Deduplicate normalized strings while preserving order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = normalize_parser_action(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _split_semicolon_values(payload: str) -> list[str]:
    """Split a semicolon-delimited action/value list."""

    raw_parts = re.split(r"\s*;\s*|\s*\|\s*", payload)
    return [" ".join(part.split()).strip() for part in raw_parts if " ".join(part.split()).strip()]


def _split_comma_values(payload: str) -> list[str]:
    """Split a comma-delimited list."""

    return [" ".join(part.split()).strip() for part in payload.split(",") if " ".join(part.split()).strip()]


def _extract_structured_json_object(raw_output: str) -> dict[str, object]:
    """Extract the structured JSON payload from a completion."""

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


def _mar_response_format() -> dict[str, object]:
    """Return the strict JSON schema used for normal-mode MAR calls."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "mar_advantage",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "key_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "action": {"type": "string"},
                                "advantage": {"type": "number"},
                                "outcome": {"type": "string"},
                                "rationale": {"type": "string"},
                                "support": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "confidence": {"type": "number"},
                            },
                            "required": [
                                "action",
                                "advantage",
                                "outcome",
                                "rationale",
                                "support",
                                "confidence",
                            ],
                        },
                    },
                    "prefer": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "avoid": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "subgoals": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "affordances": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "object": {"type": "string"},
                                "verb": {"type": "string"},
                            },
                            "required": ["object", "verb"],
                        },
                    },
                    "reasoning": {"type": "string"},
                },
                "required": [
                    "key_points",
                    "prefer",
                    "avoid",
                    "subgoals",
                    "affordances",
                    "reasoning",
                ],
            },
        },
    }


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
        return _split_semicolon_values(value)
    if isinstance(value, list):
        return [" ".join(str(item).split()).strip() for item in value if str(item).strip()]
    return []


def _as_mapping_list(value: object) -> list[Mapping[str, object]]:
    """Normalize a JSON list field into mapping items only."""

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _as_affordance_payloads(value: object) -> list[Mapping[str, object]]:
    """Normalize affordance payloads into `{object, verb}` mappings."""

    payloads: list[Mapping[str, object]] = []
    for item in _as_mapping_list(value):
        if "object" in item and "verb" in item:
            payloads.append(item)
    return payloads
