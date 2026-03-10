"""Compact, game-grounded reflection over trajectories and local branch rollouts.

The reflection layer is intentionally conservative: it keeps only concrete actions,
objects, and affordances that are supported by observed rollout or trajectory
outcomes. Meta-template leakage and prompt language are stripped before anything
enters reusable policy memory.

TODO: revisit the grounding heuristics once longer live-agent traces are available.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import logging
import re

from zork_agent.config import ProjectConfig
from zork_agent.llm.base import BaseLLMClient
from zork_agent.llm.prompts import PromptManager
from zork_agent.memory.summaries import SummaryBuilder
from zork_agent.types import (
    LocalBranchOutcome,
    LocalExplorationResult,
    ReflectionGuidance,
    ReflectionMode,
    ReflectionResult,
    TrajectoryStep,
    action_target_tokens,
    extract_salient_nouns,
    is_bulk_inventory_action,
    is_movement_action,
    normalize_parser_action,
    split_action_command,
)

_LOGGER = logging.getLogger("zork_agent.reflection")

_META_PREFIXES = (
    "objective:",
    "task:",
    "format:",
    "preferred format:",
    "preferred plain-text format:",
    "state summary:",
    "recent actions:",
    "recent action patterns:",
    "local rollout summaries:",
    "recent trajectory:",
    "episode id:",
    "seed:",
)

_META_SUBSTRINGS = (
    "as an ai",
    "you are",
    "the agent should",
    "return up to",
    "one per line",
    "plain-text format",
    "parser decisions",
    "write compact",
    "write concise",
    "keep the response",
    "focus on",
    "emphasize",
    "follow the format",
    "follow format",
    "prompt instructions",
    "template text",
    "self-referential",
    "respond with",
    "the response should",
    "instruction:",
)

_ACTION_PREFIX_STRIP = (
    "try ",
    "avoid ",
    "do not ",
    "don't ",
    "dont ",
    "repeated ",
    "repeat ",
)

_NON_OBJECT_REFLECTION_TOKENS = {
    "all",
    "broken",
    "certainly",
    "close",
    "closed",
    "closing",
    "look",
    "open",
    "opened",
    "opening",
    "shut",
}

_LOW_VALUE_REFLECTION_ACTION_PREFIXES = (
    "close ",
    "drop ",
    "get all",
    "put ",
    "put down all",
    "insert ",
    "take all",
    "throw ",
    "attack ",
    "break ",
    "kick ",
)


@dataclass(slots=True)
class _ReflectionEvidence:
    """Grounding evidence extracted from observed branches or trajectory steps."""

    try_actions: list[str] = field(default_factory=list)
    avoid_actions: list[str] = field(default_factory=list)
    salient_objects: list[str] = field(default_factory=list)
    discovered_affordances: list[str] = field(default_factory=list)
    grounded_hypotheses: list[str] = field(default_factory=list)

    def render_constraints(self) -> str:
        """Render a compact grounding block for the reflection prompt."""

        return (
            "Only keep items supported by observed outcomes.\n"
            f"Supported try_actions: {', '.join(self.try_actions) or '(none)'}\n"
            f"Supported avoid_actions: {', '.join(self.avoid_actions) or '(none)'}\n"
            f"Supported salient_objects: {', '.join(self.salient_objects) or '(none)'}\n"
            f"Supported discovered_affordances: {', '.join(self.discovered_affordances) or '(none)'}"
        )


class ReflectionEngine:
    """Generate compact game-grounded reflection guidance from branches or trajectories."""

    def __init__(self, config: ProjectConfig, prompt_manager: PromptManager, llm_client: BaseLLMClient | None):
        # TODO: split trajectory-vs-rollout prompts only if they diverge more sharply.
        self.config = config
        self.prompt_manager = prompt_manager
        self.llm_client = llm_client
        self.summary_builder = SummaryBuilder()

    def reflect_rollouts(self, local_exploration: LocalExplorationResult) -> ReflectionResult:
        """Reflect on local branch outcomes and extract grounded operational guidance."""

        evidence = self._build_rollout_evidence(local_exploration)
        branch_summary = self._summarize_branches(local_exploration)
        recent_actions = self._rollout_action_digest(local_exploration.branches)
        fallback_result = self._heuristic_rollout_reflection(local_exploration, evidence)

        if self.llm_client is None:
            return fallback_result

        try:
            system_prompt = self.prompt_manager.render_system(game_id=self.config.experiment.game_id)
            user_prompt = self.prompt_manager.render_local_reflection(
                state_summary=branch_summary,
                recent_actions=recent_actions,
                grounding_constraints=evidence.render_constraints(),
            )
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.config.llm.model_name,
                temperature=min(self.config.llm.temperature, 0.2),
                max_tokens=min(self.config.llm.max_tokens, 160),
                timeout_seconds=self.config.llm.request_timeout_seconds,
            )
        except Exception as exc:
            _LOGGER.warning("Rollout reflection fell back after LLM request failure: %s", exc)
            return fallback_result

        guidance = self._parse_reflection_output(response.text, evidence)
        if not self._has_guidance(guidance):
            fallback_result.raw_output = response.text
            fallback_result.model_name = response.model
            return fallback_result

        return ReflectionResult(
            mode=ReflectionMode.LLM,
            guidance=guidance,
            raw_output=response.text,
            model_name=response.model,
        )

    def reflect_trajectory(self, steps: list[TrajectoryStep]) -> ReflectionResult:
        """Reflect on an episode trajectory and return grounded operational guidance."""

        evidence = self._build_trajectory_evidence(steps)
        fallback_result = self._heuristic_trajectory_reflection(steps, evidence)
        if not steps or self.llm_client is None:
            return fallback_result

        trajectory_excerpt = self._trajectory_excerpt(steps)
        try:
            system_prompt = self.prompt_manager.render_system(game_id=self.config.experiment.game_id)
            user_prompt = self.prompt_manager.render_trajectory_analysis(
                episode_id=steps[0].episode_id,
                seed=int(steps[0].metadata.get("seed", self.config.experiment.seed)),
                trajectory_excerpt=trajectory_excerpt,
                grounding_constraints=evidence.render_constraints(),
            )
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.config.llm.model_name,
                temperature=min(self.config.llm.temperature, 0.2),
                max_tokens=min(self.config.llm.max_tokens, 160),
                timeout_seconds=self.config.llm.request_timeout_seconds,
            )
        except Exception as exc:
            _LOGGER.warning("Trajectory reflection fell back after LLM request failure: %s", exc)
            return fallback_result

        guidance = self._parse_reflection_output(response.text, evidence)
        if not self._has_guidance(guidance):
            fallback_result.raw_output = response.text
            fallback_result.model_name = response.model
            return fallback_result

        return ReflectionResult(
            mode=ReflectionMode.LLM,
            guidance=guidance,
            raw_output=response.text,
            model_name=response.model,
        )

    def reflect(self, steps: list[TrajectoryStep]) -> str:
        """Backward-compatible entrypoint used by the episode runner."""

        return self.reflect_trajectory(steps).normalized_guidance_text

    def _summarize_branches(self, local_exploration: LocalExplorationResult) -> str:
        """Build a compact branch comparison block for the reflection prompt."""

        lines: list[str] = []
        for branch in local_exploration.branches[:6]:
            notes = "; ".join(branch.notable_observation_changes[:3]) or "none"
            lines.append(
                f"B{branch.branch_index} | actions={' -> '.join(branch.actions_taken) or 'none'} "
                f"| score_delta={branch.score_change} | reward={branch.total_reward:.2f} "
                f"| inventory_changed={'yes' if branch.inventory_changed else 'no'} "
                f"| new_affordances={branch.new_affordance_count} "
                f"| new_room={'yes' if branch.new_room_location_signal else 'no'} "
                f"| novel_objects={branch.novel_object_count} "
                f"| stuck={'yes' if branch.appears_stuck else 'no'} "
                f"| loop_events={branch.loop_event_count} "
                f"| final={self._truncate(branch.final_observation, 80)} | notes={notes}"
            )
        if not lines:
            lines.append("No local branches were recorded.")
        return "\n".join(lines)

    def _rollout_action_digest(self, branches: list[LocalBranchOutcome]) -> str:
        """Build a small action digest to accompany branch summaries."""

        digest: list[str] = []
        for branch in branches[:6]:
            digest.append(f"B{branch.branch_index}:{' -> '.join(branch.actions_taken[:4]) or 'none'}")
        return " | ".join(digest) or "none"

    def _trajectory_excerpt(self, steps: list[TrajectoryStep]) -> str:
        """Build a compact episode excerpt for trajectory reflection."""

        excerpt_lines: list[str] = []
        previous_score = 0
        for step in steps[-8:]:
            score_delta = step.score - previous_score
            excerpt_lines.append(
                f"step={step.step_index} action={step.action} reward={step.reward:.2f} "
                f"score_delta={score_delta} loop_penalty={step.loop_penalty:.2f} "
                f"obs={self._truncate(step.observation, 80)}"
            )
            previous_score = step.score
        return "\n".join(excerpt_lines) or "No recorded steps."

    def _heuristic_rollout_reflection(
        self,
        local_exploration: LocalExplorationResult,
        evidence: _ReflectionEvidence,
    ) -> ReflectionResult:
        """Derive grounded guidance directly from branch outcomes."""

        guidance = ReflectionGuidance(
            try_actions=evidence.try_actions[:4],
            avoid_actions=evidence.avoid_actions[:4],
            salient_objects=evidence.salient_objects[:4],
            discovered_affordances=evidence.discovered_affordances[:4],
            supported_try_actions=evidence.try_actions[:4],
            supported_avoid_actions=evidence.avoid_actions[:4],
        )
        guidance.short_guidance_text = self._compact_guidance_text(
            guidance,
            grounded_hypotheses=evidence.grounded_hypotheses,
        )
        return ReflectionResult(
            mode=ReflectionMode.FALLBACK,
            guidance=guidance,
        )

    def _heuristic_trajectory_reflection(
        self,
        steps: list[TrajectoryStep],
        evidence: _ReflectionEvidence,
    ) -> ReflectionResult:
        """Derive grounded guidance from raw episode steps."""

        if not steps:
            return ReflectionResult(
                mode=ReflectionMode.FALLBACK,
                guidance=ReflectionGuidance(short_guidance_text="No reliable guidance yet."),
            )

        guidance = ReflectionGuidance(
            try_actions=evidence.try_actions[:4],
            avoid_actions=evidence.avoid_actions[:4],
            salient_objects=evidence.salient_objects[:4],
            discovered_affordances=evidence.discovered_affordances[:4],
            supported_try_actions=evidence.try_actions[:4],
            supported_avoid_actions=evidence.avoid_actions[:4],
        )
        guidance.short_guidance_text = self._compact_guidance_text(
            guidance,
            grounded_hypotheses=evidence.grounded_hypotheses,
        )
        if not guidance.short_guidance_text:
            guidance.short_guidance_text = self.summary_builder.summarize_steps(steps)
        return ReflectionResult(
            mode=ReflectionMode.FALLBACK,
            guidance=guidance,
        )

    def _parse_reflection_output(self, raw_output: str, evidence: _ReflectionEvidence) -> ReflectionGuidance:
        """Parse and ground a tolerant plain-text reflection response."""

        try_action_items: list[str] = []
        avoid_action_items: list[str] = []
        salient_object_items: list[str] = []
        affordance_items: list[str] = []
        freeform_items: list[str] = []

        for raw_line in raw_output.splitlines():
            line = self._clean_line(raw_line)
            if not line:
                continue

            label_match = re.match(
                r"^(try_actions?|try|actions?|verbs?|useful verbs?|avoid_actions?|avoid|dead ends?|"
                r"salient_objects?|objects?|promising nouns?|discovered_affordances?|affordances?|"
                r"short_guidance_text|short guidance text|guidance|hypothesis|hypotheses)\s*[:\-]\s*(.+)$",
                line,
                flags=re.IGNORECASE,
            )
            if label_match:
                label = label_match.group(1).lower()
                payload = label_match.group(2)
                if label.startswith(("try", "action", "verb", "useful verb")):
                    try_action_items.extend(self._split_items(payload))
                elif label.startswith(("avoid", "dead")):
                    avoid_action_items.extend(self._split_items(payload))
                elif label.startswith(("salient_object", "object", "promising noun")):
                    salient_object_items.extend(self._split_items(payload))
                elif label.startswith(("discovered_affordance", "affordance")):
                    affordance_items.extend(self._split_items(payload))
                else:
                    freeform_items.extend(self._split_items(payload))
                continue

            if self._is_meta_instruction(line):
                continue
            freeform_items.append(line)

        grounded_try_actions = self._ground_actions(try_action_items, evidence.try_actions)
        grounded_avoid_actions = self._ground_actions(avoid_action_items, evidence.avoid_actions)
        unsupported_avoid_items = [
            item
            for item in avoid_action_items
            if self._normalize_phrase(item)
            and self._normalize_phrase(item)
            not in {self._normalize_phrase(action) for action in grounded_avoid_actions}
        ]

        guidance = ReflectionGuidance(
            try_actions=grounded_try_actions,
            avoid_actions=grounded_avoid_actions,
            salient_objects=self._ground_objects(salient_object_items, evidence.salient_objects),
            discovered_affordances=self._ground_affordances(
                affordance_items,
                evidence.discovered_affordances,
            ),
            supported_try_actions=evidence.try_actions[:4],
            supported_avoid_actions=evidence.avoid_actions[:4],
        )
        grounded_hypotheses = self._ground_hypotheses(freeform_items, evidence)
        guidance = self._finalize_guidance(
            guidance,
            evidence=evidence,
            explicit_removed_items=unsupported_avoid_items,
            raw_items=[
                *try_action_items,
                *avoid_action_items,
                *salient_object_items,
                *affordance_items,
                *freeform_items,
            ],
        )
        guidance.short_guidance_text = self._compact_guidance_text(
            guidance,
            grounded_hypotheses=grounded_hypotheses,
        )
        return guidance

    def _build_rollout_evidence(self, local_exploration: LocalExplorationResult) -> _ReflectionEvidence:
        """Extract supported reflection evidence from local branch outcomes."""

        productive_branches = [
            branch for branch in local_exploration.branches if self._branch_has_progress(branch)
        ]
        avoid_branches = [
            branch
            for branch in local_exploration.branches
            if (
                branch.appears_stuck
                or branch.loop_event_count > 0
                or branch.oscillation_penalty_total > 0.0
                or branch.movement_penalty_total > 0.0
                or (branch.exhausted_family_count > 0 and not branch.durable_progress)
            )
        ]

        productive_events = [
            event
            for branch in sorted(productive_branches, key=self._branch_sort_key, reverse=True)
            for event in self._productive_branch_events(branch)
        ]
        try_actions = self._dedupe_preserve_order(
            str(event["action"])
            for event in productive_events
            if event.get("action")
        )[:4]
        productive_action_keys = {normalize_parser_action(action) for action in try_actions}
        avoid_action_counts: Counter[str] = Counter()
        avoid_action_labels: dict[str, str] = {}
        for branch in avoid_branches:
            for event in self._avoid_branch_events(branch):
                action = str(event.get("action", ""))
                normalized = normalize_parser_action(action)
                if not normalized or normalized in productive_action_keys:
                    continue
                avoid_action_counts[normalized] += 1
                avoid_action_labels.setdefault(normalized, action)
        avoid_actions = [
            avoid_action_labels[key]
            for key, count in avoid_action_counts.items()
            if count > 0
        ][:4]

        salient_objects = self._collect_branch_objects(productive_branches)[:4]
        discovered_affordances = self._dedupe_preserve_order(
            self._action_affordance(action)
            for branch in productive_branches
            for action in branch.actions_taken
            if self._action_affordance(action)
        )[:4]
        hypotheses = self._dedupe_preserve_order(
            hypothesis
            for branch in productive_branches
            for hypothesis in self._branch_hypotheses(branch)
        )[:2]
        return _ReflectionEvidence(
            try_actions=try_actions,
            avoid_actions=avoid_actions,
            salient_objects=salient_objects,
            discovered_affordances=discovered_affordances,
            grounded_hypotheses=hypotheses,
        )

    def _build_trajectory_evidence(self, steps: list[TrajectoryStep]) -> _ReflectionEvidence:
        """Extract supported reflection evidence from a trajectory."""

        if not steps:
            return _ReflectionEvidence()

        productive_steps: list[TrajectoryStep] = []
        avoid_steps: list[TrajectoryStep] = []
        previous_score = 0
        previous_inventory = ""
        action_counts: Counter[str] = Counter()
        for step in steps:
            score_delta = step.score - previous_score
            inventory_changed = normalize_parser_action(step.inventory_text) != normalize_parser_action(previous_inventory)
            had_progress = self._trajectory_step_has_progress(step, score_delta=score_delta)
            if had_progress:
                productive_steps.append(step)
            if step.loop_penalty > 0.0 or step.movement_penalty > 0.0:
                avoid_steps.append(step)
            elif not had_progress:
                action_counts[normalize_parser_action(step.action)] += 1
                if action_counts[normalize_parser_action(step.action)] > 1 or (
                    step.movement_only_action and step.movement_repeat_count > 0
                ):
                    avoid_steps.append(step)
            previous_score = step.score
            previous_inventory = step.inventory_text

        try_actions = self._dedupe_preserve_order(step.action for step in productive_steps if step.action)[:4]
        productive_action_keys = {normalize_parser_action(step.action) for step in productive_steps if step.action}
        avoid_actions = self._dedupe_preserve_order(
            step.action
            for step in avoid_steps
            if step.action and normalize_parser_action(step.action) not in productive_action_keys
        )[:4]
        salient_objects = self._dedupe_preserve_order(
            token
            for step in productive_steps
            for token in self._objects_from_step(step)
        )[:4]
        discovered_affordances = self._dedupe_preserve_order(
            self._action_affordance(step.action)
            for step in productive_steps
            if self._action_affordance(step.action)
        )[:4]
        hypotheses = self._dedupe_preserve_order(
            self._trajectory_hypothesis(step, previous_step=productive_steps[index - 1] if index > 0 else None)
            for index, step in enumerate(productive_steps)
        )[:2]
        return _ReflectionEvidence(
            try_actions=try_actions,
            avoid_actions=avoid_actions,
            salient_objects=salient_objects,
            discovered_affordances=discovered_affordances,
            grounded_hypotheses=[item for item in hypotheses if item],
        )

    def _branch_has_progress(self, branch: LocalBranchOutcome) -> bool:
        """Return whether a branch exposed meaningful forward progress."""

        return any(
            (
                branch.score_change > 0,
                branch.total_reward > 0.0,
                (
                    branch.persistent_inventory_gain_count > 0
                    and branch.persistent_inventory_loss_count <= 0
                    and branch.bulk_inventory_action_count <= 0
                ),
                branch.persistent_affordance_gain > 0 and branch.novel_object_count > 0,
                branch.persistent_exit_gain_count > 0,
            )
        )

    def _productive_branch_events(self, branch: LocalBranchOutcome) -> list[dict[str, object]]:
        """Return branch action events that reflect durable forward progress."""

        events = branch.metadata.get("action_events", [])
        if not isinstance(events, list):
            return []
        productive_events: list[dict[str, object]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            if self._event_supports_try_action(event):
                productive_events.append(event)
        return productive_events

    def _avoid_branch_events(self, branch: LocalBranchOutcome) -> list[dict[str, object]]:
        """Return branch action events that clearly look low-value or loop-like."""

        events = branch.metadata.get("action_events", [])
        if not isinstance(events, list) or not events:
            return self._synthetic_avoid_branch_events(branch)
        avoid_events: list[dict[str, object]] = []
        action_counts = Counter(
            normalize_parser_action(str(event.get("action", "")))
            for event in events
            if isinstance(event, dict) and str(event.get("action", "")).strip()
        )
        for event in events:
            if not isinstance(event, dict):
                continue
            action = str(event.get("action", ""))
            normalized = normalize_parser_action(action)
            if not normalized:
                continue
            is_productive = self._event_supports_try_action(event)
            if is_productive:
                continue
            if (
                bool(event.get("loop_detected", False))
                or float(event.get("loop_penalty", 0.0)) > 0.0
                or float(event.get("movement_penalty", 0.0)) > 0.0
                or action_counts[normalized] > 1
            ):
                avoid_events.append(event)
        return avoid_events

    def _synthetic_avoid_branch_events(self, branch: LocalBranchOutcome) -> list[dict[str, object]]:
        """Fallback avoid-event synthesis for fixtures or older branches without per-step events."""

        loop_results = branch.metadata.get("loop_results", [])
        movement_results = branch.metadata.get("movement_results", [])
        branch_low_value = (
            branch.score_change <= 0
            and branch.persistent_inventory_gain_count <= 0
            and branch.persistent_affordance_gain <= 0
            and branch.persistent_exit_gain_count <= 0
        )
        events: list[dict[str, object]] = []
        for index, action in enumerate(branch.actions_taken):
            loop_flag = (
                bool(loop_results[index].get("loop_detected", False))
                if index < len(loop_results) and isinstance(loop_results[index], dict)
                else False
            )
            movement_flag = False
            if index < len(movement_results) and isinstance(movement_results[index], dict):
                movement_flag = bool(
                    float(movement_results[index].get("total_penalty", 0.0)) > 0.0
                    or int(movement_results[index].get("movement_repeat_count", 0)) > 0
                )
            repeated_low_value = branch.actions_taken.count(action) > 1 and branch_low_value
            if loop_flag or movement_flag or repeated_low_value:
                events.append(
                    {
                        "action": action,
                        "loop_detected": loop_flag,
                        "movement_penalty": 1.0 if movement_flag else 0.0,
                    }
                )
        return events

    def _collect_branch_objects(self, branches: list[LocalBranchOutcome]) -> list[str]:
        """Collect salient object tokens from productive branches."""

        objects = self._dedupe_preserve_order(
            token
            for branch in branches
            for token in (
                *self._objects_from_actions(branch.actions_taken),
                *self._objects_from_text(branch.final_observation),
                *[
                    item
                    for change in branch.notable_observation_changes
                    for item in self._objects_from_text(change)
                ],
            )
        )
        return objects

    def _objects_from_actions(self, actions: list[str]) -> list[str]:
        """Extract object tokens from concrete parser actions."""

        return self._dedupe_preserve_order(
            token
            for action in actions
            for token in sorted(action_target_tokens(action, self.config.policy.inverse_action_pairs))
            if token not in _NON_OBJECT_REFLECTION_TOKENS
        )

    def _objects_from_step(self, step: TrajectoryStep) -> list[str]:
        """Extract grounded object tokens from one trajectory step."""

        tokens = [
            *sorted(action_target_tokens(step.action, self.config.policy.inverse_action_pairs)),
            *sorted(
                extract_salient_nouns(
                    observation=step.observation,
                    inventory_text=step.inventory_text,
                    valid_actions=[],
                    inverse_pairs=self.config.policy.inverse_action_pairs,
                )
            ),
        ]
        return self._dedupe_preserve_order(
            token for token in tokens if token not in _NON_OBJECT_REFLECTION_TOKENS
        )

    def _objects_from_text(self, text: str) -> list[str]:
        """Extract grounded object tokens from observation text."""

        return [
            token
            for token in sorted(
                extract_salient_nouns(
                    observation=text,
                    inventory_text="",
                    valid_actions=[],
                    inverse_pairs=self.config.policy.inverse_action_pairs,
                )
            )
            if token not in _NON_OBJECT_REFLECTION_TOKENS
        ]

    def _branch_hypotheses(self, branch: LocalBranchOutcome) -> list[str]:
        """Infer short grounded hypotheses from productive branch outcomes."""

        hypotheses: list[str] = []
        objects = self._objects_from_actions(branch.actions_taken)
        if branch.score_change > 0 and objects:
            hypotheses.append(f"{objects[0]} interactions changed score")
        if branch.inventory_changed and objects:
            hypotheses.append(f"{objects[0]} can likely be taken or carried")
        if branch.new_room_location_signal:
            direction_action = next((action for action in branch.actions_taken if self._is_direction_action(action)), "")
            if direction_action:
                hypotheses.append(f"{direction_action} may open a new area")
        return hypotheses

    def _trajectory_hypothesis(self, step: TrajectoryStep, previous_step: TrajectoryStep | None) -> str:
        """Infer a short grounded hypothesis from a productive trajectory step."""

        objects = sorted(action_target_tokens(step.action, self.config.policy.inverse_action_pairs))
        if step.loop_penalty > 0.0:
            return ""
        if objects and step.reward > 0.0:
            return f"{objects[0]} interaction produced reward"
        if previous_step is not None and step.score > previous_step.score and objects:
            return f"{objects[0]} interaction increased score"
        if self._is_direction_action(step.action):
            return f"{step.action} changed location"
        return ""

    def _ground_actions(self, items: list[str], supported_actions: list[str]) -> list[str]:
        """Keep only grounded actions supported by observed outcomes."""

        normalized_support = {normalize_parser_action(action): action for action in supported_actions}
        grounded: list[str] = []
        for item in items:
            cleaned = self._normalize_phrase(item)
            if not cleaned or self._is_meta_instruction(cleaned):
                continue
            for prefix in _ACTION_PREFIX_STRIP:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix) :].strip()
            if cleaned in normalized_support:
                grounded.append(normalized_support[cleaned])
                continue

            verb, obj = split_action_command(cleaned, self.config.policy.inverse_action_pairs)
            if verb and not obj:
                for action in supported_actions:
                    action_verb, _action_obj = split_action_command(action, self.config.policy.inverse_action_pairs)
                    if action_verb == verb:
                        grounded.append(action)
                        break
        return self._dedupe_preserve_order(grounded)[:4]

    def _ground_objects(self, items: list[str], supported_objects: list[str]) -> list[str]:
        """Keep only object tokens supported by observed outcomes."""

        supported = {self._normalize_phrase(item) for item in supported_objects}
        grounded: list[str] = []
        for item in items:
            cleaned = self._normalize_phrase(item)
            if not cleaned or self._is_meta_instruction(cleaned):
                continue
            tokens = set(re.findall(r"[a-z]+", cleaned))
            for supported_object in supported:
                if supported_object in tokens or supported_object == cleaned:
                    grounded.append(supported_object)
        return self._dedupe_preserve_order(grounded)[:4]

    def _ground_affordances(self, items: list[str], supported_affordances: list[str]) -> list[str]:
        """Keep only affordances supported by observed outcomes."""

        supported = {self._normalize_phrase(item): item for item in supported_affordances}
        grounded: list[str] = []
        for item in items:
            cleaned = self._normalize_phrase(item)
            if not cleaned or self._is_meta_instruction(cleaned):
                continue
            if cleaned in supported:
                grounded.append(supported[cleaned])
        return self._dedupe_preserve_order(grounded)[:4]

    def _ground_hypotheses(self, items: list[str], evidence: _ReflectionEvidence) -> list[str]:
        """Keep only short hypotheses tied to observed actions or objects."""

        supported_actions = {normalize_parser_action(action) for action in evidence.try_actions}
        supported_objects = {self._normalize_phrase(item) for item in evidence.salient_objects}
        grounded: list[str] = []
        for item in items:
            cleaned = self._normalize_phrase(item)
            if not cleaned or self._is_meta_instruction(cleaned):
                continue
            if "->" in cleaned:
                continue
            action_supported = any(action in cleaned for action in supported_actions if action)
            object_supported = any(obj in cleaned for obj in supported_objects if obj)
            if action_supported or object_supported:
                grounded.append(cleaned)
        return self._dedupe_preserve_order(grounded)[:2]

    def _finalize_guidance(
        self,
        guidance: ReflectionGuidance,
        *,
        evidence: _ReflectionEvidence,
        explicit_removed_items: list[str] | None = None,
        raw_items: list[str],
    ) -> ReflectionGuidance:
        """Remove contradictory or unsupported guidance before it enters policy memory."""

        removed_items: list[str] = list(explicit_removed_items or [])
        productive_actions = {normalize_parser_action(action) for action in evidence.try_actions}
        protected_affordance_actions = {
            normalize_parser_action(action)
            for action in evidence.try_actions
            if self._action_affordance(action)
        }
        cleaned_avoid: list[str] = []
        try_action_keys = {normalize_parser_action(action) for action in guidance.try_actions}
        for action in guidance.avoid_actions:
            normalized = normalize_parser_action(action)
            if normalized in try_action_keys:
                removed_items.append(action)
                continue
            if normalized in productive_actions or normalized in protected_affordance_actions:
                removed_items.append(action)
                continue
            cleaned_avoid.append(action)
        guidance.avoid_actions = self._dedupe_preserve_order(cleaned_avoid)[:4]
        grounded_items = {
            *[self._normalize_phrase(item) for item in guidance.try_actions],
            *[self._normalize_phrase(item) for item in guidance.avoid_actions],
            *[self._normalize_phrase(item) for item in guidance.salient_objects],
            *[self._normalize_phrase(item) for item in guidance.discovered_affordances],
        }
        for raw_item in raw_items:
            normalized = self._normalize_phrase(raw_item)
            if not normalized or normalized in grounded_items or self._is_meta_instruction(normalized):
                continue
            removed_items.append(raw_item)
        guidance.unsupported_items_removed = self._dedupe_preserve_order(removed_items)[:8]
        return guidance

    def _compact_guidance_text(
        self,
        guidance: ReflectionGuidance,
        *,
        grounded_hypotheses: list[str] | None = None,
    ) -> str:
        """Build a compact prompt-friendly guidance string."""

        parts: list[str] = []
        if guidance.try_actions:
            parts.append("Try: " + "; ".join(guidance.try_actions[:3]))
        if guidance.avoid_actions:
            parts.append("Avoid: " + "; ".join(guidance.avoid_actions[:3]))
        if guidance.salient_objects:
            parts.append("Objects: " + ", ".join(guidance.salient_objects[:3]))
        if guidance.discovered_affordances:
            parts.append("Affordances: " + "; ".join(guidance.discovered_affordances[:3]))
        if grounded_hypotheses:
            parts.append("Observed: " + "; ".join(grounded_hypotheses[:2]))
        return ". ".join(part.strip().rstrip(".") for part in parts if part).strip() + ("." if parts else "")

    def _has_guidance(self, guidance: ReflectionGuidance) -> bool:
        """Return whether the parsed guidance contains actionable content."""

        return any(
            (
                guidance.try_actions,
                guidance.avoid_actions,
                guidance.salient_objects,
                guidance.discovered_affordances,
                guidance.short_guidance_text,
            )
        )

    def _split_items(self, payload: str) -> list[str]:
        """Split one labeled reflection payload into candidate items."""

        items = re.split(r"[;,|]", payload)
        normalized = [self._normalize_phrase(item) for item in items]
        return [item for item in normalized if item and not self._is_meta_instruction(item)]

    def _clean_line(self, line: str) -> str:
        """Strip numbering and bullet prefixes from a reflection line."""

        return re.sub(r"^\s*(?:[-*]|\d+[\).:\-])\s*", "", line).strip()

    def _normalize_phrase(self, value: str) -> str:
        """Normalize a reflection phrase into compact prompt-safe text."""

        return " ".join(value.split()).strip().lower().strip(".")

    def _is_meta_instruction(self, text: str) -> bool:
        """Return whether a line or item looks like prompt/template leakage."""

        normalized = self._normalize_phrase(text)
        if not normalized:
            return True
        if "{" in normalized or "}" in normalized:
            return True
        if normalized.startswith(_META_PREFIXES):
            return True
        return any(fragment in normalized for fragment in _META_SUBSTRINGS)

    def _truncate(self, text: str, limit: int) -> str:
        """Truncate text conservatively for prompt reuse."""

        normalized = " ".join(text.split()).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _dedupe_preserve_order(self, values) -> list[str]:
        """Deduplicate a stream of candidate guidance items while preserving order."""

        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if not value:
                continue
            normalized = self._normalize_phrase(str(value))
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(str(value).strip())
        return ordered

    def _action_affordance(self, action: str) -> str:
        """Express a parser action as an object-affordance hint when possible."""

        verb, obj = split_action_command(action, self.config.policy.inverse_action_pairs)
        if not verb or not obj:
            return ""
        return f"{obj} -> {verb}"

    def _branch_sort_key(self, branch: LocalBranchOutcome) -> tuple[float, float, int, int, float]:
        """Return the same explicit comparison signals used for branch comparison."""

        return (
            float(branch.branch_progress_score),
            float(branch.score_change),
            int(branch.new_room_or_object_detected),
            int(not branch.appears_stuck),
            float(branch.total_reward),
        )

    def _is_direction_action(self, action: str) -> bool:
        """Return whether an action looks like a room-transition command."""

        return is_movement_action(action, self.config.policy.inverse_action_pairs)

    def _event_supports_try_action(self, event: dict[str, object]) -> bool:
        """Return whether an action event has enough grounded evidence to seed try-actions."""

        action = str(event.get("action", ""))
        if not action.strip():
            return False
        if is_bulk_inventory_action(action):
            return int(event.get("score_delta", 0)) > 0 or int(event.get("persistent_exit_gain_count", 0)) > 0
        is_movement = self._is_direction_action(action)
        if int(event.get("score_delta", 0)) > 0:
            return True
        if bool(event.get("inventory_gained", False)):
            return True
        if int(event.get("persistent_exit_gain_count", 0)) > 0:
            return True
        if is_movement:
            return False
        if bool(event.get("revealed_new_object", False)) and not self._is_low_value_toggle_or_use(action):
            return True
        return (
            int(event.get("persistent_affordance_gain", 0)) > 0
            and bool(event.get("revealed_new_object", False))
            and not self._is_low_value_toggle_or_use(action)
        )

    def _trajectory_step_has_progress(self, step: TrajectoryStep, *, score_delta: int) -> bool:
        """Return whether a trajectory step exposed grounded forward progress."""

        is_movement = self._is_direction_action(step.action)
        if is_bulk_inventory_action(step.action):
            return step.reward > 0.0 or score_delta > 0 or int(step.metadata.get("persistent_exit_gain_count", 0)) > 0
        if bool(step.metadata.get("durable_progress", False)) and not is_movement:
            return True
        if step.reward > 0.0 or score_delta > 0:
            return True
        if bool(step.metadata.get("inventory_gained", False)):
            return True
        if int(step.metadata.get("persistent_exit_gain_count", 0)) > 0:
            return True
        if is_movement:
            return False
        revealed_new_object = bool(step.metadata.get("revealed_new_object", False))
        if revealed_new_object and not self._is_low_value_toggle_or_use(step.action):
            return True
        return (
            int(step.metadata.get("persistent_affordance_gain", 0)) > 0
            and revealed_new_object
            and not self._is_low_value_toggle_or_use(step.action)
        )

    def _is_low_value_toggle_or_use(self, action: str) -> bool:
        """Return whether an action looks like low-value local churn for reflection memory."""

        normalized = normalize_parser_action(action)
        if not normalized:
            return False
        if is_bulk_inventory_action(action):
            return True
        if normalized.startswith(_LOW_VALUE_REFLECTION_ACTION_PREFIXES):
            return True
        verb, _obj = split_action_command(action, self.config.policy.inverse_action_pairs)
        return verb in {"close", "drop", "put", "insert", "throw", "attack", "break", "kick"}


ReflectionPolicy = ReflectionEngine
