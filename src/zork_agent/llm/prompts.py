"""Prompt template loading and light rendering helpers.

TODO: add richer template validation if prompt complexity grows past plain format strings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zork_agent.config import PromptConfig
from zork_agent.types import (
    AdvantageHint,
    AffordanceRecord,
    LocalWorldModel,
    LocalWorldModelPromptSummary,
)


class _SafeFormatDict(dict[str, Any]):
    """Dictionary that leaves missing keys visibly unresolved."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class PromptManager:
    """Load prompt templates from the configured prompt directory."""

    def __init__(self, config: PromptConfig):
        # TODO: cache invalidation is unnecessary for now because files are tiny.
        self.config = config
        self.directory = config.directory

    def load_named_prompt(self, name: str) -> str:
        """Load an arbitrary prompt file by name."""

        path = self.directory / name
        return path.read_text(encoding="utf-8")

    def render_template(self, template: str, **values: Any) -> str:
        """Render an in-memory prompt template with Python format placeholders."""

        return template.format_map(_SafeFormatDict(values))

    def load_system_prompt(self) -> str:
        """Load the shared system prompt."""

        return self.load_named_prompt(self.config.system_file)

    def render(self, name: str, **values: Any) -> str:
        """Render a prompt file using Python format syntax."""

        template = self.load_named_prompt(name)
        return self.render_template(template, **values)

    def render_system(self, **values: Any) -> str:
        """Render the shared system prompt."""

        return self.render(self.config.system_file, **values)

    def render_action_proposal(
        self,
        *,
        observation: str,
        inventory: str,
        score: int,
        moves: int,
        action_candidates: int,
        generation_mode: str = "open",
        recent_trajectory_context: str = "",
        valid_actions: list[str] | None = None,
        salient_objects: list[str] | None = None,
        supported_try_actions: list[str] | None = None,
        supported_avoid_actions: list[str] | None = None,
        root_state_id: str = "",
        local_world_model_summary: LocalWorldModelPromptSummary | None = None,
        strategic_mode: str = "",
        strategic_reason: str = "",
        strategic_try_actions: list[str] | None = None,
        strategic_avoid_actions: list[str] | None = None,
        strategic_objects: list[str] | None = None,
    ) -> str:
        """Render the action proposal prompt."""

        valid_actions = valid_actions or []
        salient_objects = salient_objects or []
        supported_try_actions = supported_try_actions or []
        supported_avoid_actions = supported_avoid_actions or []
        strategic_try_actions = strategic_try_actions or []
        strategic_avoid_actions = strategic_avoid_actions or []
        strategic_objects = strategic_objects or []
        local_world_model_summary = local_world_model_summary or LocalWorldModelPromptSummary(
            root_state_id=root_state_id.strip()
        )
        valid_actions_block = (
            "Valid actions:\n" + "\n".join(f"- {action}" for action in valid_actions)
            if valid_actions
            else "Valid actions:\n(unavailable)"
        )
        recent_trajectory_block = recent_trajectory_context.strip() or "(none)"
        salient_objects_block = ", ".join(salient_objects) if salient_objects else "(none)"
        evidence_block = (
            f"Prefer if supported: {', '.join(supported_try_actions) or '(none)'}\n"
            f"Avoid if repeated low-value: {', '.join(supported_avoid_actions) or '(none)'}"
        )
        strategic_guidance_block = (
            f"Strategic mode: {strategic_mode or 'unknown'}\n"
            f"Strategic focus: {strategic_reason.strip() or '(none)'}\n"
            f"Strategic try actions: {', '.join(strategic_try_actions) or '(none)'}\n"
            f"Strategic avoid actions: {', '.join(strategic_avoid_actions) or '(none)'}\n"
            f"Strategic objects: {', '.join(strategic_objects) or '(none)'}"
        )
        local_world_model_block = self._local_world_model_block(local_world_model_summary)
        canonical_guidance_block = (
            "Canonical preference:\n"
            "1. examine/look at/read a newly revealed object\n"
            "2. take/get a newly revealed portable object\n"
            "3. open/look in a container-like object\n"
            "4. move only after direct object actions are exhausted\n"
            "5. speculative tool use or aggressive actions only if grounded by evidence"
        )
        return self.render(
            self.config.action_proposal_file,
            observation=observation,
            inventory=inventory,
            score=score,
            moves=moves,
            action_candidates=action_candidates,
            generation_mode=generation_mode,
            recent_trajectory_context=recent_trajectory_block,
            valid_actions_block=valid_actions_block,
            salient_objects_block=salient_objects_block,
            evidence_block=evidence_block,
            root_state_id=root_state_id.strip() or local_world_model_summary.root_state_id or "(unknown)",
            local_world_model_block=local_world_model_block,
            strategic_guidance_block=strategic_guidance_block,
            canonical_guidance_block=canonical_guidance_block,
        )

    def render_trajectory_analysis(
        self,
        *,
        episode_id: str,
        seed: int,
        trajectory_excerpt: str,
        grounding_constraints: str = "",
    ) -> str:
        """Render the trajectory analysis prompt."""

        return self.render(
            self.config.trajectory_analysis_file,
            episode_id=episode_id,
            seed=seed,
            trajectory_excerpt=trajectory_excerpt,
            grounding_constraints=grounding_constraints.strip() or "(none)",
        )

    def render_frontier_analysis(
        self,
        *,
        analysis_id: str = "",
        frontier_trajectory_block: str,
        achieved_value_block: str,
        bottleneck_candidate_block: str,
        grounding_constraints: str = "",
        analysis_debug_mode: bool = False,
    ) -> str:
        """Render the frontier-analysis prompt."""

        return self.render(
            self.config.frontier_analysis_file,
            analysis_id=analysis_id.strip() or "frontier-analysis",
            frontier_trajectory_block=frontier_trajectory_block.strip() or "(none)",
            achieved_value_block=achieved_value_block.strip() or "(none)",
            bottleneck_candidate_block=bottleneck_candidate_block.strip() or "(none)",
            grounding_constraints=grounding_constraints.strip() or "(none)",
            thinking_mode_block=(
                "Debug mode is ON. You may include brief reasoning before the final structured output."
                if analysis_debug_mode
                else "Debug mode is OFF. Do not include any visible reasoning or preamble."
            ),
            structured_output_mode_block=(
                "Emit any optional reasoning first, then emit the final JSON object between "
                "`BEGIN_STRUCTURED_OUTPUT` and `END_STRUCTURED_OUTPUT`."
                if analysis_debug_mode
                else "Emit only the final JSON object and nothing else."
            ),
        )

    def render_local_reflection(
        self,
        *,
        state_summary: str,
        recent_actions: str,
        grounding_constraints: str = "",
    ) -> str:
        """Render the local reflection prompt."""

        return self.render(
            self.config.local_reflection_file,
            state_summary=state_summary,
            recent_actions=recent_actions,
            grounding_constraints=grounding_constraints.strip() or "(none)",
        )

    def render_mar_advantage(
        self,
        *,
        branch_comparison_block: str,
        existing_local_model_block: str = "",
        grounding_constraints: str = "",
        analysis_debug_mode: bool = False,
    ) -> str:
        """Render the MAR prompt from the default `mar_advantage.txt` template."""

        return self.render(
            "mar_advantage.txt",
            branch_comparison_block=branch_comparison_block.strip() or "(none)",
            existing_local_model_block=existing_local_model_block.strip() or "(none)",
            grounding_constraints=grounding_constraints.strip() or "(none)",
            thinking_mode_block=(
                "Debug mode is ON. You may include brief reasoning before the final structured output."
                if analysis_debug_mode
                else "Debug mode is OFF. Do not include any visible reasoning or preamble."
            ),
            structured_output_mode_block=(
                "Emit any optional reasoning first, then emit the final JSON object between "
                "`BEGIN_STRUCTURED_OUTPUT` and `END_STRUCTURED_OUTPUT`."
                if analysis_debug_mode
                else "A strict JSON schema is enforced by the caller. Return only the final JSON object."
            ),
            schema_contract_block=(
                (
                    "Final structured output contract:\n"
                    "{\n"
                    '  "key_points": [\n'
                    "    {\n"
                    '      "action": "parser action",\n'
                    '      "advantage": 0.0,\n'
                    '      "outcome": "short observed delta",\n'
                    '      "rationale": "short why",\n'
                    '      "support": ["branch-id"],\n'
                    '      "confidence": 0.0\n'
                    "    }\n"
                    "  ],\n"
                    '  "prefer": ["parser action"],\n'
                    '  "avoid": ["parser action"],\n'
                    '  "subgoals": ["short grounded subgoal"],\n'
                    '  "affordances": [\n'
                    "    {\n"
                    '      "object": "noun",\n'
                    '      "verb": "verb"\n'
                    "    }\n"
                    "  ],\n"
                    '  "reasoning": "one short operational explanation"\n'
                    "}"
                )
                if analysis_debug_mode
                else "Populate only grounded values for key_points, prefer, avoid, subgoals, affordances, and reasoning."
            ),
        )

    def render_state_selection(self, *, frontier_snapshot: str, analysis_debug_mode: bool = False) -> str:
        """Render the state-selection prompt."""

        return self.render(
            self.config.state_selection_file,
            frontier_snapshot=frontier_snapshot,
            thinking_mode_block=(
                "Debug mode is ON. You may include brief reasoning before the final structured output."
                if analysis_debug_mode
                else "Debug mode is OFF. Do not include any visible reasoning or preamble."
            ),
            structured_output_mode_block=(
                "Emit any optional reasoning first, then emit the final JSON object between "
                "`BEGIN_STRUCTURED_OUTPUT` and `END_STRUCTURED_OUTPUT`."
                if analysis_debug_mode
                else "A strict JSON schema is enforced by the caller. Return only the final JSON object."
            ),
        )

    def available_prompt_paths(self) -> list[Path]:
        """Return the expected prompt files for quick inspection."""

        return [
            self.directory / self.config.system_file,
            self.directory / self.config.action_proposal_file,
            self.directory / self.config.trajectory_analysis_file,
            self.directory / self.config.frontier_analysis_file,
            self.directory / self.config.local_reflection_file,
            self.directory / self.config.state_selection_file,
        ] + ([self.directory / "mar_advantage.txt"] if (self.directory / "mar_advantage.txt").exists() else [])

    def summarize_local_world_model(
        self,
        local_world_model: LocalWorldModel | None,
        *,
        root_state_id: str = "",
        max_advantage_hints: int = 2,
        max_subgoals: int = 3,
        max_affordances: int = 4,
        max_action_biases: int = 4,
        max_summary_chars: int = 320,
    ) -> LocalWorldModelPromptSummary:
        """Build a compact prompt payload from a typed local world model."""

        effective_root_state_id = root_state_id.strip() or (
            local_world_model.root_state_id if local_world_model is not None else ""
        )
        if local_world_model is None:
            return LocalWorldModelPromptSummary(root_state_id=effective_root_state_id)

        recent_hints = [
            self._compact_advantage_hint_text(hint)
            for hint in local_world_model.accumulated_advantage_hints[-max_advantage_hints:]
        ]
        summary_parts: list[str] = []
        if recent_hints:
            summary_parts.append("Recent MAR hints: " + " | ".join(item for item in recent_hints if item))
        if local_world_model.discovered_subgoals:
            summary_parts.append(
                "Subgoals: "
                + ", ".join(
                    item.description.strip()
                    for item in local_world_model.discovered_subgoals[:max_subgoals]
                    if item.description.strip()
                )
            )
        if local_world_model.inferred_affordances:
            summary_parts.append(
                "Affordances: "
                + ", ".join(
                    self._compact_affordance_text(item)
                    for item in local_world_model.inferred_affordances[:max_affordances]
                    if self._compact_affordance_text(item)
                )
            )
        summary_text = self._truncate(" ".join(part for part in summary_parts if part), max_summary_chars)
        return LocalWorldModelPromptSummary(
            root_state_id=effective_root_state_id,
            summary_text=summary_text,
            recent_advantage_hints=[item for item in recent_hints if item],
            discovered_subgoals=[
                item.description.strip()
                for item in local_world_model.discovered_subgoals[:max_subgoals]
                if item.description.strip()
            ],
            inferred_affordances=[
                self._compact_affordance_text(item)
                for item in local_world_model.inferred_affordances[:max_affordances]
                if self._compact_affordance_text(item)
            ],
            action_priors=[
                item.action.strip()
                for item in local_world_model.action_priors[:max_action_biases]
                if item.action.strip()
            ],
            action_antipriors=[
                item.action.strip()
                for item in local_world_model.action_antipriors[:max_action_biases]
                if item.action.strip()
            ],
        )

    def _local_world_model_block(self, summary: LocalWorldModelPromptSummary) -> str:
        """Render the bounded local-world-model prompt block."""

        if not summary.has_guidance():
            return (
                f"Root state id: {summary.root_state_id or '(unknown)'}\n"
                "Local world model summary: (none yet)\n"
                "Recent advantage hints: (none)\n"
                "Discovered subgoals: (none)\n"
                "Discovered affordances: (none)\n"
                "Local try actions: (none)\n"
                "Local avoid actions: (none)"
            )
        return (
            f"Root state id: {summary.root_state_id or '(unknown)'}\n"
            f"Local world model summary: {summary.summary_text or '(none)'}\n"
            f"Recent advantage hints: {', '.join(summary.recent_advantage_hints) or '(none)'}\n"
            f"Discovered subgoals: {', '.join(summary.discovered_subgoals) or '(none)'}\n"
            f"Discovered affordances: {', '.join(summary.inferred_affordances) or '(none)'}\n"
            f"Local try actions: {', '.join(summary.action_priors) or '(none)'}\n"
            f"Local avoid actions: {', '.join(summary.action_antipriors) or '(none)'}"
        )

    def _compact_advantage_hint_text(self, hint: AdvantageHint) -> str:
        """Render one recent advantage hint into a short line."""

        prefer = ", ".join(item for item in hint.action_preferences[:2] if item)
        avoid = ", ".join(item for item in hint.action_avoidances[:2] if item)
        reasoning = self._truncate(" ".join(hint.textual_reasoning.split()), 90)
        parts = []
        if prefer:
            parts.append(f"prefer {prefer}")
        if avoid:
            parts.append(f"avoid {avoid}")
        if reasoning:
            parts.append(reasoning)
        return "; ".join(parts)

    def _compact_affordance_text(self, affordance: AffordanceRecord) -> str:
        """Render one affordance into a compact object->verb string."""

        object_text = str(getattr(affordance, "object_text", "")).strip()
        affordance_text = str(getattr(affordance, "affordance", "")).strip()
        if not object_text and not affordance_text:
            return ""
        return f"{object_text} -> {affordance_text}".strip(" ->")

    def _truncate(self, text: str, limit: int) -> str:
        """Truncate prompt support text conservatively."""

        normalized = " ".join(text.split()).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."
