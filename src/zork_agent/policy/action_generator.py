"""Action proposal logic for parser commands.

This policy keeps parsing intentionally resilient and plain-text oriented. It does
not require JSON mode from the model, and it treats Jericho valid-action lists as
an optional constraint rather than a hard dependency.

TODO: revisit parsing heuristics after collecting real LM Studio traces.
"""

from __future__ import annotations

import logging
import re

from zork_agent.config import ProjectConfig
from zork_agent.llm.base import BaseLLMClient
from zork_agent.llm.prompts import PromptManager
from zork_agent.types import (
    ActionGenerationMode,
    ActionGenerationResult,
    ActionClusterHistory,
    LoopHeuristicResult,
    ActionProposal,
    TextGameState,
    action_target_tokens,
    count_movement_cycle_repetitions,
    count_repeated_movement_actions,
    estimate_candidate_loop_penalty,
    extract_salient_nouns,
    inventory_item_tokens,
    is_movement_action,
    movement_actions_are_cycle,
    normalize_parser_action,
    split_action_command,
)

_LOGGER = logging.getLogger("zork_agent.action_generator")


class ActionGenerator:
    """Generate parser actions from the current state."""

    def __init__(self, config: ProjectConfig, prompt_manager: PromptManager, llm_client: BaseLLMClient | None):
        # TODO: split prompt construction from response parsing if either grows.
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
    ) -> ActionGenerationResult:
        """Generate candidate parser actions in constrained or open mode."""

        inventory_text = inventory_text or ""
        valid_actions = [action for action in (valid_actions or []) if action.strip()]
        candidate_count = candidate_count or self.config.policy.action_candidates
        mode = ActionGenerationMode.CONSTRAINED if valid_actions else ActionGenerationMode.OPEN

        if self.llm_client is None:
            result = self._fallback_result(
                mode=mode,
                observation=observation,
                inventory_text=inventory_text,
                valid_actions=valid_actions,
                candidate_count=candidate_count,
                reason="No LLM client configured.",
            )
            self._rerank_candidates_for_loops(
                result,
                observation=observation,
                inventory_text=inventory_text,
                valid_actions=valid_actions,
                recent_actions=recent_actions or [],
                recent_loop_results=recent_loop_results or [],
                state_action_history=state_action_history,
                candidate_count=candidate_count,
            )
            self.last_result = result
            return result

        system_prompt = self.prompt_manager.render_system(game_id=self.config.experiment.game_id)
        user_prompt = self.prompt_manager.render_action_proposal(
            observation=observation,
            inventory=inventory_text or "Inventory unavailable.",
            score=score,
            moves=moves,
            action_candidates=candidate_count,
            generation_mode=mode.value,
            recent_trajectory_context=recent_trajectory_context or "",
            valid_actions=valid_actions,
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
            )
        except Exception as exc:
            _LOGGER.warning("Action generation fell back after LLM request failure: %s", exc)
            result = self._fallback_result(
                mode=mode,
                observation=observation,
                inventory_text=inventory_text,
                valid_actions=valid_actions,
                candidate_count=candidate_count,
                reason=f"LLM request failed: {exc}",
            )
            self._rerank_candidates_for_loops(
                result,
                observation=observation,
                inventory_text=inventory_text,
                valid_actions=valid_actions,
                recent_actions=recent_actions or [],
                recent_loop_results=recent_loop_results or [],
                state_action_history=state_action_history,
                candidate_count=candidate_count,
            )
            self.last_result = result
            return result

        candidates = self._parse_candidates(
            raw_output=response.text,
            mode=mode,
            valid_actions=valid_actions,
            candidate_count=candidate_count,
        )
        if not candidates:
            result = self._fallback_result(
                mode=mode,
                observation=observation,
                inventory_text=inventory_text,
                valid_actions=valid_actions,
                candidate_count=candidate_count,
                reason="LLM output was unusable.",
                raw_output=response.text,
                model_name=response.model,
            )
            self._rerank_candidates_for_loops(
                result,
                observation=observation,
                inventory_text=inventory_text,
                valid_actions=valid_actions,
                recent_actions=recent_actions or [],
                recent_loop_results=recent_loop_results or [],
                state_action_history=state_action_history,
                candidate_count=candidate_count,
            )
            self.last_result = result
            return result

        result = ActionGenerationResult(
            mode=mode,
            candidates=candidates,
            raw_output=response.text,
            model_name=response.model,
            valid_actions=list(valid_actions),
        )
        self._rerank_candidates_for_loops(
            result,
            observation=observation,
            inventory_text=inventory_text,
            valid_actions=valid_actions,
            recent_actions=recent_actions or [],
            recent_loop_results=recent_loop_results or [],
            state_action_history=state_action_history,
            candidate_count=candidate_count,
        )
        self.last_result = result
        return result

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
            normalized = self._normalize_action(candidate.action)
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

        normalized_text = self._normalize_action(text)
        exact_map = {self._normalize_action(action): action for action in valid_actions}
        if normalized_text in exact_map:
            return exact_map[normalized_text]

        for action in sorted(valid_actions, key=len, reverse=True):
            normalized_action = self._normalize_action(action)
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
        stripped = stripped.strip(" .")
        return stripped.lower()

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

    def _fallback_result(
        self,
        *,
        mode: ActionGenerationMode,
        observation: str,
        inventory_text: str,
        valid_actions: list[str],
        candidate_count: int,
        reason: str,
        raw_output: str = "",
        model_name: str | None = None,
    ) -> ActionGenerationResult:
        """Return deterministic fallback actions when the LLM path is unusable."""

        fallback_mode = ActionGenerationMode.FALLBACK
        if valid_actions:
            actions = self._fallback_valid_actions(observation=observation, valid_actions=valid_actions)
            source = f"{fallback_mode.value}_constrained"
        else:
            actions = self._fallback_open_actions(observation=observation, inventory_text=inventory_text)
            source = f"{fallback_mode.value}_{mode.value}"

        candidates = [
            ActionProposal(
                action=action,
                rationale=reason,
                source=source,
                rank=index + 1,
                raw_line="",
            )
            for index, action in enumerate(actions[:candidate_count])
        ]
        return ActionGenerationResult(
            mode=fallback_mode,
            candidates=candidates,
            raw_output=raw_output,
            model_name=model_name,
            valid_actions=list(valid_actions),
            fallback_reason=reason,
        )

    def _fallback_valid_actions(self, *, observation: str, valid_actions: list[str]) -> list[str]:
        """Prefer locally relevant valid actions when constrained fallback is needed."""

        ranked: list[str] = []
        observation_text = observation.lower()
        preferred_patterns = (
            "take",
            "get",
            "read",
            "open",
            "examine",
            "look in",
            "enter",
            "look",
            "inventory",
            "go ",
        )

        for keyword in ("mailbox", "leaflet", "door", "window", "house"):
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
            actions.append("open mailbox")
        if "leaflet" in observation_text:
            actions.append("read leaflet")
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
            normalized = self._normalize_action(action)
            if normalized not in seen:
                deduped.append(action)
                seen.add(normalized)
        return deduped

    def _normalize_action(self, action: str) -> str:
        """Normalize parser commands for matching and deduplication."""

        return " ".join(action.split()).strip().lower()

    def _rerank_candidates_for_loops(
        self,
        result: ActionGenerationResult,
        *,
        observation: str,
        inventory_text: str,
        valid_actions: list[str],
        recent_actions: list[str],
        recent_loop_results: list[LoopHeuristicResult],
        state_action_history: ActionClusterHistory | None,
        candidate_count: int,
    ) -> None:
        """Rerank candidates with loop, movement, and salient-affordance bonuses."""

        if not result.candidates:
            return

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
        new_visible_nouns = visible_nouns - known_nouns
        apply_affordance_heuristics = bool(valid_actions and state_action_history is not None)
        exhausted_families = (
            state_action_history.exhausted_families(self.config.policy.object_family_no_progress_threshold)
            if state_action_history is not None
            else set()
        )
        if apply_affordance_heuristics:
            self._augment_candidates_with_valid_actions(
                result,
                valid_actions,
                candidate_count=candidate_count,
                visible_nouns=visible_nouns,
                new_visible_nouns=new_visible_nouns,
                inventory_tokens=inventory_tokens,
                exhausted_families=exhausted_families,
                state_action_history=state_action_history,
            )

        reranked = False
        affordance_reranked = False
        movement_reranked = False
        for index, candidate in enumerate(result.candidates):
            candidate.movement_only_action = is_movement_action(
                candidate.action,
                self.config.policy.inverse_action_pairs,
            )
            loop_penalty, loop_reason = estimate_candidate_loop_penalty(
                candidate_action=candidate.action,
                recent_actions=recent_actions,
                recent_loop_results=recent_loop_results,
                inverse_pairs=self.config.policy.inverse_action_pairs,
                immediate_inverse_penalty=self.config.policy.immediate_inverse_penalty,
                repeated_pair_penalty=self.config.policy.repeated_pair_penalty,
                reversible_no_progress_penalty=self.config.policy.reversible_no_progress_penalty,
            )
            if apply_affordance_heuristics:
                heuristic_bonus, heuristic_penalty, ranking_reason = self._action_affordance_score(
                    candidate=candidate,
                    state_action_history=state_action_history,
                    visible_nouns=visible_nouns,
                    new_visible_nouns=new_visible_nouns,
                    inventory_tokens=inventory_tokens,
                    exhausted_families=exhausted_families,
                )
            else:
                heuristic_bonus = 0.0
                heuristic_penalty = 0.0
                ranking_reason = ""
            candidate.loop_penalty = loop_penalty
            candidate.loop_penalty_reason = loop_reason
            candidate.heuristic_bonus = heuristic_bonus
            candidate.heuristic_penalty = heuristic_penalty
            candidate.ranking_reason = ranking_reason
            reranked = reranked or loop_penalty > 0.0
            affordance_reranked = affordance_reranked or heuristic_bonus > 0.0 or heuristic_penalty > 0.0

        preferred_object_actions_exist = any(
            not candidate.movement_only_action
            and (
                candidate.heuristic_bonus > 0.0
                or bool(action_target_tokens(candidate.action, self.config.policy.inverse_action_pairs) & visible_nouns)
            )
            for candidate in result.candidates
        )

        scored_candidates: list[tuple[float, float, int, int, ActionProposal]] = []
        for index, candidate in enumerate(result.candidates):
            movement_penalty, movement_reason = self._movement_action_penalty(
                candidate=candidate,
                state_action_history=state_action_history,
                recent_actions=recent_actions,
                visible_nouns=visible_nouns,
                new_visible_nouns=new_visible_nouns,
                preferred_object_actions_exist=preferred_object_actions_exist,
                exhausted_families=exhausted_families,
            )
            candidate.movement_penalty = movement_penalty
            candidate.movement_penalty_reason = movement_reason
            candidate.heuristic_penalty += movement_penalty
            candidate.selection_score = candidate.heuristic_bonus - candidate.heuristic_penalty - candidate.loop_penalty
            movement_reranked = movement_reranked or movement_penalty > 0.0
            if movement_penalty > 0.0 and movement_reason:
                candidate.ranking_reason = (
                    f"{candidate.ranking_reason}, {movement_reason}".strip(", ")
                    if candidate.ranking_reason
                    else movement_reason
                )
            original_rank = candidate.rank if candidate.rank is not None else index + 1
            scored_candidates.append(
                (
                    -candidate.selection_score,
                    candidate.loop_penalty + candidate.movement_penalty,
                    original_rank,
                    index,
                    candidate,
                )
            )

        if reranked or affordance_reranked or movement_reranked or result.augmented_with_valid_actions:
            scored_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            result.candidates = [item[4] for item in scored_candidates][:candidate_count]
            for rank, candidate in enumerate(result.candidates, start=1):
                candidate.rank = rank
        result.reranked_by_loop_penalty = reranked
        result.reranked_by_affordance_heuristics = affordance_reranked
        result.reranked_by_movement_heuristics = movement_reranked

    def _augment_candidates_with_valid_actions(
        self,
        result: ActionGenerationResult,
        valid_actions: list[str],
        *,
        candidate_count: int,
        visible_nouns: set[str],
        new_visible_nouns: set[str],
        inventory_tokens: set[str],
        exhausted_families: set[str],
        state_action_history: ActionClusterHistory | None,
    ) -> None:
        """Ensure reranking can consider available Jericho valid actions."""

        if not valid_actions or state_action_history is None:
            return

        seen_actions = {normalize_parser_action(candidate.action) for candidate in result.candidates}
        missing_actions = [
            action
            for action in valid_actions
            if (normalized := normalize_parser_action(action)) and normalized not in seen_actions
        ]
        if not missing_actions:
            return

        candidate_slots = max(1, candidate_count - len(result.candidates))
        scored_missing_actions: list[tuple[float, str]] = []
        for action in missing_actions:
            preview_candidate = ActionProposal(
                action=action,
                source="valid_action_fallback",
                movement_only_action=is_movement_action(action, self.config.policy.inverse_action_pairs),
            )
            bonus, penalty, _reason = self._action_affordance_score(
                candidate=preview_candidate,
                state_action_history=state_action_history,
                visible_nouns=visible_nouns,
                new_visible_nouns=new_visible_nouns,
                inventory_tokens=inventory_tokens,
                exhausted_families=exhausted_families,
            )
            scored_missing_actions.append((bonus - penalty, action))

        scored_missing_actions.sort(key=lambda item: (-item[0], item[1]))
        for score, action in scored_missing_actions[:candidate_slots]:
            if score <= 0.0:
                continue
            result.candidates.append(
                ActionProposal(
                    action=action,
                    source="valid_action_fallback",
                    rank=len(result.candidates) + 1,
                )
            )
            result.augmented_with_valid_actions = True

    def _action_affordance_score(
        self,
        *,
        candidate: ActionProposal,
        state_action_history: ActionClusterHistory | None,
        visible_nouns: set[str],
        new_visible_nouns: set[str],
        inventory_tokens: set[str],
        exhausted_families: set[str],
    ) -> tuple[float, float, str]:
        """Score one candidate for salient affordances and stale-action penalties."""

        bonus = 0.0
        penalty = 0.0
        reasons: list[str] = []
        normalized_action = normalize_parser_action(candidate.action)
        verb, _obj = split_action_command(candidate.action, self.config.policy.inverse_action_pairs)
        attempt_stats = (
            state_action_history.action_stats.get(normalized_action)
            if state_action_history is not None
            else None
        )
        attempts = attempt_stats.attempts if attempt_stats is not None else 0
        target_tokens = action_target_tokens(
            candidate.action,
            inverse_pairs=self.config.policy.inverse_action_pairs,
        )
        is_generic_action = not target_tokens

        if attempts == 0 and not is_generic_action:
            bonus += self.config.policy.action_untried_bonus
            reasons.append("untried_action")
        elif attempt_stats is not None and not attempt_stats.had_any_gain and attempt_stats.no_durable_gain_count > 0:
            penalty += self.config.policy.action_repeated_no_gain_penalty * float(attempt_stats.no_durable_gain_count)
            reasons.append(f"repeated_no_gain={attempt_stats.no_durable_gain_count}")

        touched_new_nouns = target_tokens & new_visible_nouns
        if touched_new_nouns:
            bonus += self.config.policy.action_new_noun_bonus * float(len(touched_new_nouns))
            reasons.append("new_noun:" + ",".join(sorted(touched_new_nouns)))
            if verb in {"take", "get", "examine", "read", "open", "enter"} or normalized_action.startswith(("look in ", "look inside ")):
                bonus += self.config.policy.action_new_noun_interaction_bonus
                reasons.append(f"salient_interaction:{verb}")
        elif target_tokens and visible_nouns and target_tokens.isdisjoint(visible_nouns):
            penalty += 0.25
            reasons.append("nonvisible_target")

        if exhausted_families and target_tokens & exhausted_families:
            penalty += self.config.policy.object_family_exhaustion_penalty * float(len(target_tokens & exhausted_families))
            reasons.append("exhausted_family:" + ",".join(sorted(target_tokens & exhausted_families)))

        if verb in {"drop", "put"} and target_tokens & inventory_tokens:
            penalty += self.config.policy.discard_inventory_penalty
            reasons.append("discard_inventory")

        if candidate.movement_only_action and exhausted_families and attempts == 0:
            bonus += self.config.policy.escape_mode_exit_bonus
            reasons.append("escape_exhausted_family")

        return bonus, penalty, ", ".join(reasons)

    def _movement_action_penalty(
        self,
        *,
        candidate: ActionProposal,
        state_action_history: ActionClusterHistory | None,
        recent_actions: list[str],
        visible_nouns: set[str],
        new_visible_nouns: set[str],
        preferred_object_actions_exist: bool,
        exhausted_families: set[str],
    ) -> tuple[float, str]:
        """Penalize stale movement wandering when object affordances are available."""

        if state_action_history is None or not candidate.movement_only_action:
            return 0.0, ""

        penalty = 0.0
        reasons: list[str] = []
        normalized_action = normalize_parser_action(candidate.action)
        attempt_stats = state_action_history.action_stats.get(normalized_action)
        no_gain_attempts = attempt_stats.no_durable_gain_count if attempt_stats is not None else 0
        cluster_visits = state_action_history.cluster_visit_count()
        repeated_count = count_repeated_movement_actions(
            [*recent_actions, candidate.action],
            self.config.policy.inverse_action_pairs,
        )

        if no_gain_attempts > 0:
            penalty += self.config.policy.repeated_movement_penalty * float(no_gain_attempts)
            reasons.append(f"movement_no_gain={no_gain_attempts}")
        if repeated_count > 0:
            penalty += self.config.policy.repeated_movement_penalty * float(repeated_count)
            reasons.append(f"movement_repeat_count={repeated_count}")

        if movement_actions_are_cycle(candidate.action, recent_actions[-1] if recent_actions else None):
            penalty += self.config.policy.movement_cycle_penalty
            reasons.append("movement_cycle")
        cycle_repetitions = count_movement_cycle_repetitions(
            [*recent_actions, candidate.action],
            self.config.policy.inverse_action_pairs,
        )
        if cycle_repetitions > 0:
            penalty += self.config.policy.movement_cycle_penalty * float(cycle_repetitions)
            reasons.append(f"movement_cycle_count={cycle_repetitions}")

        if cluster_visits > 1:
            penalty += self.config.policy.same_region_repeat_penalty * float(cluster_visits - 1)
            reasons.append(f"same_region_repeat={cluster_visits}")

        if preferred_object_actions_exist and not exhausted_families:
            penalty += self.config.policy.same_region_repeat_penalty
            reasons.append("object_affordance_available")

        if new_visible_nouns and not exhausted_families:
            penalty += self.config.policy.same_region_repeat_penalty
            reasons.append("new_object_available")
        elif not visible_nouns:
            penalty += 0.25
            reasons.append("no_salient_nouns")

        return penalty, ", ".join(reasons)

    def _is_plausible_action(self, action: str) -> bool:
        """Return whether parsed free-form text looks like a parser command."""

        normalized = self._normalize_action(action)
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
        if normalized.startswith(disallowed_prefixes):
            return False
        return True
