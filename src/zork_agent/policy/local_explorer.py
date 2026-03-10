"""Shallow local branching from a replay-restored frontier state.

This module intentionally avoids complex tree search. It restores the same base
state before each branch rollout, tries a small number of short horizons, and
compares branch outcomes using explicit heuristics.

TODO: revisit branch diversification once real Jericho runs expose better signals.
"""

from __future__ import annotations

import logging
import re

from zork_agent.env.jericho_env import JerichoEnv
from zork_agent.env.replay import restore_saved_node
from zork_agent.memory.frontier import FrontierEntry, FrontierQueue
from zork_agent.policy.action_generator import ActionGenerator
from zork_agent.policy.state_selector import StateSelector
from zork_agent.types import (
    ActionClusterHistory,
    ActionProposal,
    BranchTerminationReason,
    LocalBranchOutcome,
    LocalExplorationResult,
    LoopHeuristicResult,
    MovementHeuristicResult,
    ReplayResult,
    RestoreMode,
    SavedNode,
    TextGameState,
    action_target_tokens,
    action_shape_is_complex_transitive,
    canonical_if_verb_family,
    compute_region_novelty_score,
    derive_state_cluster_id,
    evaluate_reversible_action_loop,
    evaluate_movement_action,
    extract_salient_nouns,
    inventory_item_tokens,
    is_bulk_inventory_action,
    is_discard_like_action,
    is_movement_action,
    valid_actions_materially_different,
)

_LOGGER = logging.getLogger("zork_agent.local_explorer")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "you",
    "your",
}

_LOCATION_HINT_TOKENS = {
    "attic",
    "canyon",
    "cellar",
    "east",
    "field",
    "forest",
    "house",
    "inside",
    "kitchen",
    "north",
    "outside",
    "passage",
    "room",
    "south",
    "valley",
    "west",
}


class LocalExplorer:
    """Run shallow multi-branch local exploration from one restored state."""

    def __init__(
        self,
        action_generator: ActionGenerator,
        state_selector: StateSelector | None = None,
        *,
        env: JerichoEnv | None = None,
        logger: logging.Logger | None = None,
        default_branch_count: int | None = None,
        default_branch_horizon: int | None = None,
        default_temperature: float | None = None,
        default_action_candidate_count: int | None = None,
    ):
        # TODO: move explorer-specific config into a dedicated model only if it grows further.
        self.action_generator = action_generator
        self.state_selector = state_selector
        self.env = env
        self.logger = logger or _LOGGER
        self.default_branch_count = (
            default_branch_count
            if default_branch_count is not None
            else action_generator.config.policy.rollout_count
        )
        self.default_branch_horizon = (
            default_branch_horizon
            if default_branch_horizon is not None
            else action_generator.config.policy.rollout_depth
        )
        self.default_temperature = (
            default_temperature
            if default_temperature is not None
            else action_generator.config.llm.temperature
        )
        self.default_action_candidate_count = (
            default_action_candidate_count
            if default_action_candidate_count is not None
            else action_generator.config.policy.action_candidates
        )

    def expand(self, state: TextGameState, frontier: FrontierQueue) -> tuple[FrontierEntry | None, list[ActionProposal]]:
        """Backward-compatible helper that selects a frontier node and proposes actions."""

        selected = self.state_selector.select(frontier) if self.state_selector is not None else None
        actions = self.action_generator.propose_actions(
            state,
            state_action_history=ActionClusterHistory(cluster_label=state.world_state_hash),
        )
        return selected, actions

    def explore_from_state(
        self,
        restored_state: TextGameState,
        *,
        env: JerichoEnv | None = None,
        branch_count: int | None = None,
        branch_horizon: int | None = None,
        temperature: float | None = None,
        action_candidate_count: int | None = None,
        recent_trajectory_context: str | None = None,
    ) -> LocalExplorationResult:
        """Restore the same base state before each branch and compare shallow rollouts."""

        active_env = env or self.env
        if active_env is None:
            raise ValueError("LocalExplorer requires a JerichoEnv instance to run branch rollouts.")

        effective_branch_count = max(1, branch_count or self.default_branch_count)
        effective_branch_horizon = max(1, branch_horizon or self.default_branch_horizon)
        effective_temperature = self.default_temperature if temperature is None else temperature
        effective_action_candidate_count = max(
            1,
            action_candidate_count or self.default_action_candidate_count,
        )

        base_saved_node = self._saved_node_from_state(restored_state)
        comparison_notes = ""
        if base_saved_node is None and effective_branch_count > 1:
            comparison_notes = (
                "Base state had no replayable snapshot; local exploration was reduced to one branch."
            )
            self.logger.warning(comparison_notes)
            effective_branch_count = 1

        branches: list[LocalBranchOutcome] = []
        post_score_stale_branch_count = 0
        for branch_index in range(effective_branch_count):
            outcome = self._run_branch(
                branch_index=branch_index,
                base_state=restored_state,
                base_saved_node=base_saved_node,
                env=active_env,
                branch_horizon=effective_branch_horizon,
                temperature=effective_temperature,
                action_candidate_count=effective_action_candidate_count,
                recent_trajectory_context=recent_trajectory_context,
            )
            branches.append(outcome)
            if self._is_post_score_stale_branch(base_state=restored_state, branch=outcome):
                post_score_stale_branch_count += 1
            else:
                post_score_stale_branch_count = 0

            if (
                restored_state.score > 0
                and post_score_stale_branch_count
                >= self.action_generator.config.policy.post_score_scene_branch_patience
            ):
                comparison_notes = (
                    "Local exploration stopped early after repeated no-progress branches in a "
                    "post-score scene."
                )
                self.logger.info(
                    "Stopping local exploration early for post-score scene after %s stale branches.",
                    post_score_stale_branch_count,
                )
                break

        best_branch_index = None
        branch_commit_allowed = False
        commit_rejection_reason = ""
        comparable_branches = [branch for branch in branches if branch.termination_reason is not BranchTerminationReason.RESTORE_FAILED]
        if comparable_branches:
            best_branch = max(comparable_branches, key=self._branch_sort_key)
            best_branch_index = best_branch.branch_index
            branch_commit_allowed, commit_rejection_reason = self._branch_clears_commit_gate(best_branch)
            best_branch.branch_commit_allowed = branch_commit_allowed
            best_branch.commit_rejection_reason = commit_rejection_reason
            if not branch_commit_allowed:
                pass
        else:
            commit_rejection_reason = "No comparable branches were available for commit."

        if not comparison_notes:
            comparison_notes = (
                "Branches are compared by an explicit progress score over score gain, inventory change, "
                "new affordances, room/location signals, novel object mentions, and loop reduction."
            )

        if best_branch_index is not None:
            best_branch = next(branch for branch in branches if branch.branch_index == best_branch_index)
            self.logger.info(
                "Local exploration best branch=%s progress_score=%.2f commit_allowed=%s rejection=%s",
                best_branch.branch_index,
                best_branch.branch_progress_score,
                branch_commit_allowed,
                commit_rejection_reason or "none",
            )

        return LocalExplorationResult(
            base_state=restored_state,
            branch_count=len(branches),
            branch_horizon=effective_branch_horizon,
            temperature=effective_temperature,
            action_candidate_count=effective_action_candidate_count,
            branches=branches,
            best_branch_index=best_branch_index,
            branch_commit_allowed=branch_commit_allowed,
            commit_rejection_reason=commit_rejection_reason,
            comparison_notes=comparison_notes,
        )

    def run_local_rollouts(
        self,
        restored_state: TextGameState,
        *,
        env: JerichoEnv | None = None,
        branch_count: int | None = None,
        branch_horizon: int | None = None,
        temperature: float | None = None,
        action_candidate_count: int | None = None,
        recent_trajectory_context: str | None = None,
    ) -> LocalExplorationResult:
        """Backward-compatible alias for `explore_from_state`."""

        return self.explore_from_state(
            restored_state,
            env=env,
            branch_count=branch_count,
            branch_horizon=branch_horizon,
            temperature=temperature,
            action_candidate_count=action_candidate_count,
            recent_trajectory_context=recent_trajectory_context,
        )

    def _run_branch(
        self,
        *,
        branch_index: int,
        base_state: TextGameState,
        base_saved_node: SavedNode | None,
        env: JerichoEnv,
        branch_horizon: int,
        temperature: float,
        action_candidate_count: int,
        recent_trajectory_context: str | None,
    ) -> LocalBranchOutcome:
        """Restore the base state and run one shallow local rollout."""

        restore_result = self._restore_branch_start(
            env=env,
            base_state=base_state,
            base_saved_node=base_saved_node,
            branch_index=branch_index,
        )
        if not restore_result.success:
            self.logger.warning(
                "Local branch %s could not restore the base state: %s",
                branch_index,
                restore_result.message,
            )
            return LocalBranchOutcome(
                branch_index=branch_index,
                actions_taken=[],
                total_reward=0.0,
                score_change=0,
                final_score=restore_result.final_score,
                final_observation=restore_result.final_observation,
                terminated=False,
                termination_reason=BranchTerminationReason.RESTORE_FAILED,
                final_state=restore_result.final_state,
                restore_result=restore_result,
                metadata={"restore_mode": restore_result.restore_mode.value},
            )

        current_state = restore_result.final_state
        current_state = self._annotate_branch_state(
            current_state,
            history=None,
            previous_state=None,
            score_gain=0,
            inventory_changed=False,
            affordance_gain=0,
            novel_object_count=0,
            movement_only_action=False,
            materially_new_actions=False,
        )
        visited_states = [current_state]
        action_cluster_history = ActionClusterHistory(cluster_label=current_state.world_state_hash)
        action_cluster_history.record_state_cluster(
            cluster_id=current_state.state_cluster_id or current_state.world_state_hash,
            observation=current_state.observation,
        )
        action_cluster_history.observe_state_nouns(
            observation=current_state.observation,
            inventory_text=current_state.inventory_text,
            valid_actions=current_state.valid_actions,
            inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
        )
        actions_taken: list[str] = []
        action_events: list[dict[str, object]] = []
        loop_results: list[LoopHeuristicResult] = []
        movement_results: list[MovementHeuristicResult] = []
        oscillation_penalty_total = 0.0
        movement_penalty_total = 0.0
        bulk_inventory_action_count = 0
        discard_like_action_count = 0
        aggressive_action_count = 0
        speculative_tool_use_action_count = 0
        post_gain_churn_action_count = 0
        total_reward = 0.0
        first_durable_gain_action_index: int | None = None
        first_durable_gain_action = ""
        last_durable_progress_action_index: int | None = None
        last_durable_progress_action = ""
        last_meaningful_progress_action_index: int | None = None
        last_meaningful_progress_action = ""
        max_persistent_inventory_gain_count = 0
        max_persistent_affordance_gain = 0
        max_persistent_exit_gain_count = 0
        termination_reason = BranchTerminationReason.HORIZON_REACHED
        terminated = False

        for step_offset in range(branch_horizon):
            context = self._branch_context(
                base_context=recent_trajectory_context,
                actions_taken=actions_taken,
            )
            proposals = self.action_generator.propose_actions(
                current_state,
                recent_trajectory_context=context,
                candidate_count=action_candidate_count,
                temperature=temperature,
                recent_actions=actions_taken,
                recent_loop_results=loop_results,
                state_action_history=action_cluster_history,
            )
            if not proposals:
                termination_reason = BranchTerminationReason.NO_ACTIONS
                break

            action = self._choose_branch_action(
                proposals=proposals,
                branch_index=branch_index,
                step_offset=step_offset,
            )
            pre_step_actions = list(actions_taken)
            pre_step_states = list(visited_states)
            gain_already_established = first_durable_gain_action_index is not None
            verb_family = canonical_if_verb_family(
                action,
                self.action_generator.config.policy.inverse_action_pairs,
            )
            aggressive_action = verb_family == "aggressive"
            transition = env.step(action)
            total_reward += transition.reward
            raw_state = transition.to_state()
            inventory_gain_count = self._count_inventory_gain(
                base_state=pre_step_states[-1],
                final_state=raw_state,
            )
            inventory_loss_count = self._count_inventory_loss(
                base_state=pre_step_states[-1],
                final_state=raw_state,
            )
            persistent_inventory_gain_count = self._count_inventory_gain(
                base_state=base_state,
                final_state=raw_state,
            )
            persistent_inventory_loss_count = self._count_inventory_loss(
                base_state=base_state,
                final_state=raw_state,
            )
            inventory_gained = inventory_gain_count > 0
            inventory_lost = inventory_loss_count > 0
            persistent_inventory_gained_vs_base = persistent_inventory_gain_count > 0
            persistent_inventory_lost_vs_base = persistent_inventory_loss_count > 0
            persistent_affordance_gain = self._count_persistent_affordances(
                base_state=base_state,
                final_state=raw_state,
            )
            step_affordance_gain = self._count_persistent_affordances(
                base_state=pre_step_states[-1],
                final_state=raw_state,
            )
            persistent_exit_gain = self._count_new_exit_actions(
                base_state=base_state,
                final_state=raw_state,
            )
            step_exit_gain = self._count_new_exit_actions(
                base_state=pre_step_states[-1],
                final_state=raw_state,
            )
            affordance_gain = step_affordance_gain + step_exit_gain
            movement_only_action = self._is_movement_action(action)
            bulk_inventory_action = is_bulk_inventory_action(action)
            discard_like_action = is_discard_like_action(action)
            speculative_tool_use_action = aggressive_action or action_shape_is_complex_transitive(
                action,
                self.action_generator.config.policy.inverse_action_pairs,
            )
            novel_object_tokens = (
                extract_salient_nouns(
                    observation=raw_state.observation,
                    inventory_text=raw_state.inventory_text,
                    valid_actions=[],
                    inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
                )
                - extract_salient_nouns(
                    observation=pre_step_states[-1].observation,
                    inventory_text=pre_step_states[-1].inventory_text,
                    valid_actions=[],
                    inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
                )
            )
            persistent_novel_object_tokens = (
                extract_salient_nouns(
                    observation=raw_state.observation,
                    inventory_text=raw_state.inventory_text,
                    valid_actions=[],
                    inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
                )
                - extract_salient_nouns(
                    observation=base_state.observation,
                    inventory_text=base_state.inventory_text,
                    valid_actions=[],
                    inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
                )
            )
            materially_new_actions = valid_actions_materially_different(
                pre_step_states[-1].valid_actions,
                raw_state.valid_actions,
            )
            revealed_new_object = bool(novel_object_tokens) and (
                not movement_only_action or affordance_gain > 0 or materially_new_actions
            )
            persistent_revealed_new_object = bool(persistent_novel_object_tokens) and (
                not movement_only_action or persistent_affordance_gain > 0 or persistent_exit_gain > 0
            )
            current_state = self._annotate_branch_state(
                raw_state,
                history=action_cluster_history,
                previous_state=pre_step_states[-1],
                score_gain=raw_state.score - pre_step_states[-1].score,
                inventory_changed=self._normalize_text(raw_state.inventory_text)
                != self._normalize_text(pre_step_states[-1].inventory_text),
                affordance_gain=affordance_gain,
                novel_object_count=len(novel_object_tokens) if revealed_new_object else 0,
                movement_only_action=movement_only_action,
                materially_new_actions=materially_new_actions,
            )
            action_cluster_history.record_attempt(
                action=action,
                score_changed=current_state.score != pre_step_states[-1].score,
                inventory_changed=self._normalize_text(current_state.inventory_text)
                != self._normalize_text(pre_step_states[-1].inventory_text),
                inventory_gained=persistent_inventory_gained_vs_base,
                inventory_lost=persistent_inventory_lost_vs_base,
                observation_changed=self._normalize_text(current_state.observation)
                != self._normalize_text(pre_step_states[-1].observation),
                valid_actions_changed={
                    self._normalize_text(candidate) for candidate in current_state.valid_actions
                }
                != {
                    self._normalize_text(candidate) for candidate in pre_step_states[-1].valid_actions
                },
                valid_actions_improved=persistent_affordance_gain > 0 or persistent_exit_gain > 0,
                revealed_new_object=persistent_revealed_new_object,
                target_tokens=extract_salient_nouns(
                    observation="",
                    inventory_text="",
                    valid_actions=[action],
                    inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
                )
                or action_target_tokens(action, self.action_generator.config.policy.inverse_action_pairs),
                movement_only_action=movement_only_action,
                inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
                bulk_inventory_action=bulk_inventory_action,
                discard_like_action=discard_like_action,
                revealed_object_tokens=persistent_novel_object_tokens if persistent_revealed_new_object else set(),
            )
            action_cluster_history.observe_state_nouns(
                observation=current_state.observation,
                inventory_text=current_state.inventory_text,
                valid_actions=current_state.valid_actions,
                inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
            )
            loop_result = evaluate_reversible_action_loop(
                action=action,
                recent_actions=pre_step_actions,
                state_history=pre_step_states,
                current_state=current_state,
                inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
                immediate_inverse_penalty=self.action_generator.config.policy.immediate_inverse_penalty,
                repeated_pair_penalty=self.action_generator.config.policy.repeated_pair_penalty,
                reversible_no_progress_penalty=self.action_generator.config.policy.reversible_no_progress_penalty,
            )
            actions_taken.append(action)
            visited_states.append(current_state)
            loop_results.append(loop_result)
            oscillation_penalty_total += loop_result.total_penalty
            if bulk_inventory_action:
                bulk_inventory_action_count += 1
            if discard_like_action:
                discard_like_action_count += 1
            if aggressive_action:
                aggressive_action_count += 1
            if speculative_tool_use_action:
                speculative_tool_use_action_count += 1
            if gain_already_established and (
                discard_like_action
                or aggressive_action
                or speculative_tool_use_action
            ):
                post_gain_churn_action_count += 1
            movement_result = evaluate_movement_action(
                action=action,
                recent_actions=pre_step_actions,
                previous_state=pre_step_states[-1],
                current_state=current_state,
                inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
                repeated_movement_penalty=self.action_generator.config.policy.repeated_movement_penalty,
                movement_cycle_penalty=self.action_generator.config.policy.movement_cycle_penalty,
                same_region_repeat_penalty=self.action_generator.config.policy.same_region_repeat_penalty,
            )
            movement_results.append(movement_result)
            movement_penalty_total += movement_result.total_penalty
            meaningful_progress_this_step = self._action_is_meaningful_branch_progress(
                action=action,
                movement_only_action=movement_only_action,
                discard_like_action=discard_like_action,
                aggressive_action=aggressive_action,
                speculative_tool_use_action=speculative_tool_use_action,
                score_delta=current_state.score - pre_step_states[-1].score,
                persistent_inventory_gain_count=persistent_inventory_gain_count,
                persistent_affordance_gain=persistent_affordance_gain,
                persistent_exit_gain_count=persistent_exit_gain,
                persistent_revealed_new_object=persistent_revealed_new_object,
                max_persistent_inventory_gain_count=max_persistent_inventory_gain_count,
                max_persistent_affordance_gain=max_persistent_affordance_gain,
                max_persistent_exit_gain_count=max_persistent_exit_gain_count,
            )
            durable_progress_this_step = self._action_is_durable_branch_progress(
                action=action,
                movement_only_action=movement_only_action,
                discard_like_action=discard_like_action,
                aggressive_action=aggressive_action,
                speculative_tool_use_action=speculative_tool_use_action,
                score_delta=current_state.score - pre_step_states[-1].score,
                persistent_inventory_gain_count=persistent_inventory_gain_count,
                persistent_affordance_gain=persistent_affordance_gain,
                persistent_exit_gain_count=persistent_exit_gain,
                persistent_revealed_new_object=persistent_revealed_new_object,
                max_persistent_inventory_gain_count=max_persistent_inventory_gain_count,
                max_persistent_affordance_gain=max_persistent_affordance_gain,
                max_persistent_exit_gain_count=max_persistent_exit_gain_count,
                score_progress_established=base_state.score > 0 or current_state.score > base_state.score,
            )
            if durable_progress_this_step:
                last_durable_progress_action_index = len(actions_taken) - 1
                last_durable_progress_action = action
            if meaningful_progress_this_step:
                last_meaningful_progress_action_index = len(actions_taken) - 1
                last_meaningful_progress_action = action
            max_persistent_inventory_gain_count = max(
                max_persistent_inventory_gain_count,
                persistent_inventory_gain_count,
            )
            max_persistent_affordance_gain = max(
                max_persistent_affordance_gain,
                persistent_affordance_gain,
            )
            max_persistent_exit_gain_count = max(
                max_persistent_exit_gain_count,
                persistent_exit_gain,
            )
            action_events.append(
                {
                    "action": action,
                    "score_delta": current_state.score - pre_step_states[-1].score,
                    "inventory_gained": persistent_inventory_gained_vs_base,
                    "inventory_lost": persistent_inventory_lost_vs_base,
                    "revealed_new_object": persistent_revealed_new_object,
                    "persistent_affordance_gain": persistent_affordance_gain,
                    "persistent_exit_gain_count": persistent_exit_gain,
                    "bulk_inventory_action": bulk_inventory_action,
                    "discard_like_action": discard_like_action,
                    "loop_detected": loop_result.loop_detected,
                    "loop_penalty": loop_result.total_penalty,
                    "movement_penalty": movement_result.total_penalty,
                    "state_cluster_id": current_state.state_cluster_id,
                    "durable_progress": durable_progress_this_step,
                    "meaningful_progress": meaningful_progress_this_step,
                }
            )
            if (
                first_durable_gain_action_index is None
                and (
                    current_state.score > base_state.score
                    or (
                        persistent_inventory_gained_vs_base
                        and not persistent_inventory_lost_vs_base
                    )
                )
            ):
                first_durable_gain_action_index = len(actions_taken) - 1
                first_durable_gain_action = action
            if loop_result.loop_detected:
                self.logger.info(
                    "Local branch %s detected reversible loop for action=%s penalty=%.2f reason=%s",
                    branch_index,
                    action,
                    loop_result.total_penalty,
                    loop_result.reason,
                )
            if movement_result.total_penalty > 0.0:
                self.logger.info(
                    "Local branch %s detected low-value movement for action=%s penalty=%.2f reason=%s",
                    branch_index,
                    action,
                    movement_result.total_penalty,
                    movement_result.reason,
                )
            if self._should_abort_branch_early(
                action_cluster_history=action_cluster_history,
                oscillation_penalty_total=oscillation_penalty_total,
                movement_penalty_total=movement_penalty_total,
                base_state=base_state,
                current_state=current_state,
                persistent_inventory_gained=persistent_inventory_gained_vs_base,
                persistent_affordance_gain=persistent_affordance_gain,
                persistent_exit_gain_count=persistent_exit_gain,
                post_gain_churn_action_count=post_gain_churn_action_count,
            ):
                termination_reason = BranchTerminationReason.LOOP_ABORTED
                self.logger.info(
                    "Local branch %s aborted early after low-value loop accumulation. "
                    "loop_penalty=%.2f movement_penalty=%.2f bulk_inventory=%s post_gain_churn=%s exhausted_families=%s",
                    branch_index,
                    oscillation_penalty_total,
                    movement_penalty_total,
                    bulk_inventory_action_count,
                    post_gain_churn_action_count,
                    sorted(
                        action_cluster_history.exhausted_families(
                            self.action_generator.config.policy.object_family_no_progress_threshold
                        )
                    ),
                )
                break
            if transition.done:
                terminated = True
                termination_reason = BranchTerminationReason.TERMINATED
                break

        score_change = current_state.score - base_state.score
        notable_changes, new_room_or_object_detected, new_room_location_signal, novel_tokens = self._describe_observation_changes(
            base_state=base_state,
            visited_states=visited_states,
            score_change=score_change,
        )
        inventory_changed = self._normalize_text(current_state.inventory_text) != self._normalize_text(base_state.inventory_text)
        persistent_inventory_gain_count = self._count_inventory_gain(base_state=base_state, final_state=current_state)
        persistent_inventory_loss_count = self._count_inventory_loss(base_state=base_state, final_state=current_state)
        new_affordance_count = self._count_persistent_affordances(base_state=base_state, final_state=current_state)
        persistent_affordance_gain = new_affordance_count
        affordance_gain = new_affordance_count
        persistent_exit_gain_count = self._count_new_exit_actions(base_state=base_state, final_state=current_state)
        ended_in_same_cluster = (
            bool(base_state.state_cluster_id)
            and bool(current_state.state_cluster_id)
            and current_state.state_cluster_id == base_state.state_cluster_id
        )
        movement_action_ratio = (
            sum(1 for result in movement_results if result.movement_only_action) / len(actions_taken)
            if actions_taken
            else 0.0
        )
        exhausted_family_count = len(
            action_cluster_history.exhausted_families(
                self.action_generator.config.policy.object_family_no_progress_threshold
            )
        )
        durable_progress = self._branch_has_durable_progress(
            score_change=score_change,
            persistent_inventory_gain_count=persistent_inventory_gain_count,
            persistent_inventory_loss_count=persistent_inventory_loss_count,
            persistent_affordance_gain=persistent_affordance_gain,
            persistent_exit_gain_count=persistent_exit_gain_count,
            bulk_inventory_action_count=bulk_inventory_action_count,
            discard_like_action_count=discard_like_action_count,
            aggressive_action_count=aggressive_action_count,
            speculative_tool_use_action_count=speculative_tool_use_action_count,
            post_gain_churn_action_count=post_gain_churn_action_count,
        )
        novel_object_count = min(len(novel_tokens), 4)
        landmark_gain = 1 if (
            new_room_location_signal
            and (
                novel_object_count > 0
                or new_affordance_count > 0
                or persistent_exit_gain_count > 0
                or persistent_inventory_gain_count > 0
                or score_change > 0
            )
        ) else 0
        room_text_only_gain = self._room_text_only_gain(
            base_state=base_state,
            final_state=current_state,
            score_change=score_change,
            inventory_changed=inventory_changed,
            affordance_gain=new_affordance_count,
            novel_object_count=novel_object_count,
            landmark_gain=landmark_gain,
        )
        loop_penalty_reduction = self._loop_penalty_reduction(loop_results)
        branch_progress_score = self._branch_progress_score(
            score_change=score_change,
            persistent_inventory_gain_count=persistent_inventory_gain_count,
            persistent_inventory_loss_count=persistent_inventory_loss_count,
            persistent_affordance_gain=new_affordance_count,
            persistent_exit_gain_count=persistent_exit_gain_count,
            landmark_gain=landmark_gain,
            novel_object_count=novel_object_count,
            room_text_only_gain=room_text_only_gain,
            loop_penalty_reduction=loop_penalty_reduction,
            oscillation_penalty_total=oscillation_penalty_total,
            movement_penalty_total=movement_penalty_total,
        )
        commit_anchor = self._branch_has_commit_anchor(
            score_change=score_change,
            persistent_inventory_gain_count=persistent_inventory_gain_count,
            persistent_inventory_loss_count=persistent_inventory_loss_count,
        )
        low_value_movement_branch = self._is_low_value_movement_branch(
            score_change=score_change,
            persistent_inventory_gain_count=persistent_inventory_gain_count,
            persistent_affordance_gain=persistent_affordance_gain,
            movement_action_ratio=movement_action_ratio,
        )
        low_value_object_churn_branch = self._is_low_value_object_churn_branch(
            score_change=score_change,
            persistent_inventory_gain_count=persistent_inventory_gain_count,
            persistent_exit_gain_count=persistent_exit_gain_count,
            persistent_affordance_gain=persistent_affordance_gain,
            novel_object_count=novel_object_count,
        )
        inventory_churn_penalty = self._inventory_churn_penalty(
            score_change=score_change,
            persistent_inventory_gain_count=persistent_inventory_gain_count,
            persistent_inventory_loss_count=persistent_inventory_loss_count,
            persistent_exit_gain_count=persistent_exit_gain_count,
            bulk_inventory_action_count=bulk_inventory_action_count,
            discard_like_action_count=discard_like_action_count,
            aggressive_action_count=aggressive_action_count,
            speculative_tool_use_action_count=speculative_tool_use_action_count,
            post_gain_churn_action_count=post_gain_churn_action_count,
        )
        branch_progress_score -= inventory_churn_penalty
        if (
            exhausted_family_count > 0
            and not durable_progress
            and ended_in_same_cluster
        ):
            branch_progress_score = min(
                branch_progress_score,
                self.action_generator.config.policy.movement_progress_cap,
            )
        if low_value_movement_branch:
            branch_progress_score = min(branch_progress_score, 0.0)
        if low_value_object_churn_branch:
            branch_progress_score = min(branch_progress_score, 0.0)
        if inventory_churn_penalty > 0.0:
            branch_progress_score = min(branch_progress_score, 0.0)
        if (
            score_change <= 0
            and persistent_inventory_gain_count > 0
            and not self._branch_has_simple_inventory_progress(
                score_change=score_change,
                persistent_inventory_gain_count=persistent_inventory_gain_count,
                persistent_inventory_loss_count=persistent_inventory_loss_count,
                persistent_affordance_gain=persistent_affordance_gain,
                persistent_exit_gain_count=persistent_exit_gain_count,
                discard_like_action_count=discard_like_action_count,
                aggressive_action_count=aggressive_action_count,
                speculative_tool_use_action_count=speculative_tool_use_action_count,
                post_gain_churn_action_count=post_gain_churn_action_count,
            )
        ):
            branch_progress_score = min(branch_progress_score, 0.0)
        if not commit_anchor:
            branch_progress_score = min(branch_progress_score, 0.0)
        if (
            score_change <= 0
            and persistent_inventory_gain_count <= 0
            and persistent_exit_gain_count <= 0
            and new_affordance_count <= 0
        ):
            branch_progress_score = min(
                branch_progress_score,
                self.action_generator.config.policy.movement_progress_cap,
            )
        appears_stuck = self._appears_stuck(
            base_state=base_state,
            visited_states=visited_states,
            total_reward=total_reward,
            score_change=score_change,
            new_room_or_object_detected=new_room_or_object_detected,
            oscillation_penalty_total=oscillation_penalty_total,
            movement_penalty_total=movement_penalty_total,
        )
        if self._branch_is_zero_score_inventory_scene_churn(
            score_change=score_change,
            persistent_inventory_gain_count=persistent_inventory_gain_count,
            persistent_exit_gain_count=persistent_exit_gain_count,
            oscillation_penalty_total=oscillation_penalty_total,
            appears_stuck=appears_stuck,
            inventory_churn_penalty=inventory_churn_penalty,
            post_gain_churn_action_count=post_gain_churn_action_count,
            exhausted_family_count=exhausted_family_count,
        ):
            branch_progress_score = min(branch_progress_score, 0.0)
        if appears_stuck:
            notable_changes.append("branch appears stuck")
        if terminated:
            notable_changes.append("branch terminated")
        if oscillation_penalty_total > 0.0:
            notable_changes.append(f"oscillation penalty {oscillation_penalty_total:.2f}")
        if movement_penalty_total > 0.0:
            notable_changes.append(f"movement penalty {movement_penalty_total:.2f}")
        if low_value_object_churn_branch:
            notable_changes.append("local object churn without durable gain")
        if inventory_churn_penalty > 0.0:
            notable_changes.append(f"inventory churn penalty {inventory_churn_penalty:.2f}")
        if inventory_changed:
            notable_changes.append("inventory changed")
        if new_affordance_count > 0:
            notable_changes.append(f"new affordances {new_affordance_count}")
        if new_room_location_signal:
            notable_changes.append("room/location signal")
        if room_text_only_gain > 0.0:
            notable_changes.append(f"room text only gain {room_text_only_gain:.2f}")
        if branch_progress_score > 0.0:
            notable_changes.append(f"progress score {branch_progress_score:.2f}")

        self.logger.info(
            "Local branch %s finished: actions=%s score_change=%s reward=%.2f terminated=%s stuck=%s "
            "loop_penalty=%.2f movement_penalty=%.2f bulk_inventory=%s discard_like=%s aggressive=%s "
            "post_gain_churn=%s inventory_churn_penalty=%.2f progress_score=%.2f",
            branch_index,
            actions_taken,
            score_change,
            total_reward,
            terminated,
            appears_stuck,
            oscillation_penalty_total,
            movement_penalty_total,
            bulk_inventory_action_count,
            discard_like_action_count,
            aggressive_action_count,
            post_gain_churn_action_count,
            inventory_churn_penalty,
            branch_progress_score,
        )
        return LocalBranchOutcome(
            branch_index=branch_index,
            actions_taken=actions_taken,
            total_reward=total_reward,
            score_change=score_change,
            final_score=current_state.score,
            final_observation=current_state.observation,
            terminated=terminated,
            termination_reason=termination_reason,
            new_room_or_object_detected=new_room_or_object_detected,
            appears_stuck=appears_stuck,
            inventory_changed=inventory_changed,
            persistent_inventory_gain_count=persistent_inventory_gain_count,
            persistent_inventory_loss_count=persistent_inventory_loss_count,
            new_affordance_count=new_affordance_count,
            persistent_affordance_gain=persistent_affordance_gain,
            persistent_exit_gain_count=persistent_exit_gain_count,
            ended_in_same_cluster=ended_in_same_cluster,
            durable_progress=durable_progress,
            exhausted_family_count=exhausted_family_count,
            affordance_gain=affordance_gain,
            new_room_location_signal=new_room_location_signal,
            landmark_gain=landmark_gain,
            novel_object_count=novel_object_count,
            room_text_only_gain=room_text_only_gain,
            loop_penalty_reduction=loop_penalty_reduction,
            branch_progress_score=branch_progress_score,
            oscillation_penalty_total=oscillation_penalty_total,
            movement_penalty_total=movement_penalty_total,
            movement_only_action_count=sum(1 for result in movement_results if result.movement_only_action),
            movement_action_ratio=movement_action_ratio,
            movement_repeat_count=max((result.movement_repeat_count for result in movement_results), default=0),
            bulk_inventory_action_count=bulk_inventory_action_count,
            inventory_churn_penalty=inventory_churn_penalty,
            discard_like_action_count=discard_like_action_count,
            aggressive_action_count=aggressive_action_count,
            speculative_tool_use_action_count=speculative_tool_use_action_count,
            post_gain_churn_action_count=post_gain_churn_action_count,
            loop_event_count=sum(1 for loop_result in loop_results if loop_result.loop_detected),
            first_durable_gain_action_index=first_durable_gain_action_index,
            first_durable_gain_action=first_durable_gain_action,
            last_durable_progress_action_index=last_durable_progress_action_index,
            last_durable_progress_action=last_durable_progress_action,
            last_meaningful_progress_action_index=last_meaningful_progress_action_index,
            last_meaningful_progress_action=last_meaningful_progress_action,
            notable_observation_changes=notable_changes,
            final_state=current_state,
            restore_result=restore_result,
            metadata={
                "restore_mode": restore_result.restore_mode.value,
                "action_events": action_events,
                "loop_results": [loop_result.to_record() for loop_result in loop_results],
                "movement_results": [movement_result.to_record() for movement_result in movement_results],
                "movement_action_ratio": movement_action_ratio,
                "low_value_movement_branch": low_value_movement_branch,
                "inventory_churn_penalty": inventory_churn_penalty,
                "bulk_inventory_action_count": bulk_inventory_action_count,
                "discard_like_action_count": discard_like_action_count,
                "aggressive_action_count": aggressive_action_count,
                "speculative_tool_use_action_count": speculative_tool_use_action_count,
                "post_gain_churn_action_count": post_gain_churn_action_count,
            },
        )

    def _restore_branch_start(
        self,
        *,
        env: JerichoEnv,
        base_state: TextGameState,
        base_saved_node: SavedNode | None,
        branch_index: int,
    ) -> ReplayResult:
        """Restore the same base state before every branch rollout."""

        if base_saved_node is None:
            self.logger.info(
                "Local branch %s is starting from the current env state because no saved snapshot is available.",
                branch_index,
            )
            return ReplayResult(
                success=True,
                restore_mode=RestoreMode.RESET_ONLY,
                divergence_reason=env.validate_restored_state(
                    expected_observation=base_state.observation,
                    expected_score=base_state.score,
                    expected_inventory_text=base_state.inventory_text,
                ).divergence_reason,
                final_observation=base_state.observation,
                final_score=base_state.score,
                replayed_action_count=0,
                target_step_index=None,
                final_state=base_state,
                message="No saved node was available; using the current env state as the branch start.",
            )

        return restore_saved_node(
            env,
            base_saved_node,
            target_step_index=base_saved_node.step_index,
            logger=self.logger,
        )

    def _saved_node_from_state(self, state: TextGameState) -> SavedNode | None:
        """Build a replayable saved node from a restored state snapshot."""

        snapshot = state.world_state_snapshot
        if snapshot is None:
            return None
        step_index = len(snapshot.replay_actions) - 1
        return SavedNode(
            state_id=state.world_state_hash,
            episode_id=str(state.metadata.get("episode_id", "local-explorer")),
            step_index=step_index,
            native_state=snapshot.native_state,
            action_prefix=list(snapshot.replay_actions),
            score=state.score,
            observation=state.observation,
            inventory_text=state.inventory_text,
            world_state_hash=state.world_state_hash,
            valid_actions=list(state.valid_actions),
            summary_text=str(state.metadata.get("summary_text", "")),
            metadata=dict(state.metadata),
        )

    def _choose_branch_action(
        self,
        *,
        proposals: list[ActionProposal],
        branch_index: int,
        step_offset: int,
    ) -> str:
        """Diversify shallow branches mainly through the first action."""

        if step_offset == 0:
            proposal_index = min(branch_index, len(proposals) - 1)
        else:
            proposal_index = 0
        return proposals[proposal_index].action

    def _branch_context(self, *, base_context: str | None, actions_taken: list[str]) -> str:
        """Build a short trajectory context string for the current branch."""

        context_parts: list[str] = []
        if base_context:
            context_parts.append(base_context.strip())
        if actions_taken:
            context_parts.append("branch_actions=" + " -> ".join(actions_taken[-3:]))
        return " | ".join(part for part in context_parts if part)

    def _describe_observation_changes(
        self,
        *,
        base_state: TextGameState,
        visited_states: list[TextGameState],
        score_change: int,
    ) -> tuple[list[str], bool, bool, set[str]]:
        """Describe notable branch changes and whether anything novel appeared."""

        notes: list[str] = []
        base_tokens = self._significant_tokens(f"{base_state.observation} {base_state.inventory_text}")
        state_hash_changed = False

        for state in visited_states[1:]:
            if state.world_state_hash != base_state.world_state_hash:
                state_hash_changed = True

        if score_change > 0:
            notes.append(f"score increased by {score_change}")
        elif score_change < 0:
            notes.append(f"score decreased by {-score_change}")

        if state_hash_changed:
            notes.append("world state changed")

        final_state = visited_states[-1]
        novel_tokens = self._significant_tokens(f"{final_state.observation} {final_state.inventory_text}") - base_tokens
        new_room_location_signal = (
            final_state.world_state_hash != base_state.world_state_hash
            and self._normalize_text(final_state.observation) != self._normalize_text(base_state.observation)
            and self._looks_like_location_signal(final_state.observation)
        )
        if self._normalize_text(final_state.observation) != self._normalize_text(base_state.observation):
            notes.append("final observation changed")
        if novel_tokens:
            preview = ", ".join(sorted(novel_tokens)[:5])
            notes.append(f"new tokens: {preview}")

        new_room_or_object_detected = bool(novel_tokens) or new_room_location_signal
        return notes, new_room_or_object_detected, new_room_location_signal, novel_tokens

    def _appears_stuck(
        self,
        *,
        base_state: TextGameState,
        visited_states: list[TextGameState],
        total_reward: float,
        score_change: int,
        new_room_or_object_detected: bool,
        oscillation_penalty_total: float,
        movement_penalty_total: float,
    ) -> bool:
        """Return whether a branch looks unproductive by simple local heuristics."""

        normalized_observations = {self._normalize_text(state.observation) for state in visited_states}
        unique_hashes = {state.world_state_hash for state in visited_states}
        return (
            total_reward <= 0.0
            and score_change == 0
            and not new_room_or_object_detected
            and len(normalized_observations) <= 1
            and len(unique_hashes) <= 1
            and oscillation_penalty_total <= 0.0
            and movement_penalty_total <= 0.0
        ) or (
            total_reward <= 0.0
            and score_change == 0
            and (oscillation_penalty_total > 0.0 or movement_penalty_total > 0.0)
        )

    def _significant_tokens(self, text: str) -> set[str]:
        """Extract a small set of informative tokens for novelty heuristics."""

        return {
            token
            for token in re.findall(r"[a-z]+", text.lower())
            if len(token) > 2 and token not in _STOPWORDS
        }

    def _normalize_text(self, text: str) -> str:
        """Normalize observation text for simple equality checks."""

        return " ".join(text.split()).strip().lower()

    def _branch_sort_key(
        self,
        branch: LocalBranchOutcome,
    ) -> tuple[float, float, int, int, int, float, float, int, int, int]:
        """Return an explicit comparison key for branch outcomes."""

        return (
            float(branch.branch_progress_score),
            float(branch.score_change),
            int(branch.persistent_inventory_gain_count),
            int(branch.persistent_affordance_gain),
            int(branch.persistent_exit_gain_count),
            -int(branch.persistent_inventory_loss_count),
            float(branch.total_reward - branch.oscillation_penalty_total - branch.movement_penalty_total),
            -float(branch.movement_penalty_total),
            int(not branch.appears_stuck),
            -len(branch.actions_taken),
        )

    def _branch_has_commit_anchor(
        self,
        *,
        score_change: int,
        persistent_inventory_gain_count: int,
        persistent_inventory_loss_count: int,
    ) -> bool:
        """Return whether a branch ended with a durable outcome strong enough to score."""

        return any(
            (
                score_change > 0,
                persistent_inventory_gain_count > 0 and persistent_inventory_loss_count <= 0,
            )
        )

    def _branch_has_durable_progress(
        self,
        *,
        score_change: int,
        persistent_inventory_gain_count: int,
        persistent_inventory_loss_count: int,
        persistent_affordance_gain: int,
        persistent_exit_gain_count: int,
        bulk_inventory_action_count: int,
        discard_like_action_count: int,
        aggressive_action_count: int,
        speculative_tool_use_action_count: int,
        post_gain_churn_action_count: int,
    ) -> bool:
        """Return whether a branch ended with durable, reusable progress."""

        return any(
            (
                score_change > 0,
                self._branch_has_simple_inventory_progress(
                    score_change=score_change,
                    persistent_inventory_gain_count=persistent_inventory_gain_count,
                    persistent_inventory_loss_count=persistent_inventory_loss_count,
                    persistent_affordance_gain=persistent_affordance_gain,
                    persistent_exit_gain_count=persistent_exit_gain_count,
                    discard_like_action_count=discard_like_action_count,
                    aggressive_action_count=aggressive_action_count,
                    speculative_tool_use_action_count=speculative_tool_use_action_count,
                    post_gain_churn_action_count=post_gain_churn_action_count,
                )
                and bulk_inventory_action_count <= 0,
            )
        )

    def _branch_clears_commit_gate(self, branch: LocalBranchOutcome) -> tuple[bool, str]:
        """Return whether a branch is strong enough to commit into the main episode."""

        policy = self.action_generator.config.policy
        threshold = policy.branch_commit_min_progress_score
        if branch.branch_progress_score <= threshold:
            return (
                False,
                f"best branch progress_score {branch.branch_progress_score:.2f} did not clear threshold {threshold:.2f}",
            )

        if self._branch_is_movement_commit_reject(branch):
            return (
                False,
                "best branch was movement-dominated with no durable score, inventory, or affordance gain",
            )

        if self._branch_is_local_object_churn_reject(branch):
            return (
                False,
                "best branch only changed local affordances without score, inventory, exit, or new-object gain",
            )

        if self._branch_is_inventory_churn_reject(branch):
            return (
                False,
                "best branch relied on bulk or transient inventory churn without score or exit gain",
            )

        if self._branch_is_zero_score_inventory_scene_churn_reject(branch):
            return (
                False,
                "best branch gained inventory only through a stuck or oscillatory local scene",
            )

        if branch.score_change > 0:
            return True, ""

        if branch.persistent_inventory_gain_count > 0 and branch.persistent_inventory_loss_count <= 0:
            if not self._branch_has_simple_inventory_progress(
                score_change=branch.score_change,
                persistent_inventory_gain_count=branch.persistent_inventory_gain_count,
                persistent_inventory_loss_count=branch.persistent_inventory_loss_count,
                persistent_affordance_gain=branch.persistent_affordance_gain,
                persistent_exit_gain_count=branch.persistent_exit_gain_count,
                discard_like_action_count=branch.discard_like_action_count,
                aggressive_action_count=branch.aggressive_action_count,
                speculative_tool_use_action_count=branch.speculative_tool_use_action_count,
                post_gain_churn_action_count=branch.post_gain_churn_action_count,
            ):
                return (
                    False,
                    "best branch gained inventory only through low-value local object churn",
                )
            return True, ""

        if branch.persistent_exit_gain_count > 0:
            return False, "best branch only changed reachable exits without score or inventory gain"

        if branch.score_change <= 0 and branch.persistent_inventory_gain_count <= 0:
            return (
                False,
                "best branch had no score gain or durable inventory gain",
            )

        if branch.ended_in_same_cluster and branch.persistent_exit_gain_count <= 0 and branch.score_change <= 0:
            return False, "best branch stayed in the same local cluster without durable progress"

        if branch.exhausted_family_count > 0 and not branch.durable_progress:
            return False, "best branch remained inside an exhausted local object family"

        return False, "best branch did not demonstrate durable end-state progress"

    def _count_persistent_affordances(self, *, base_state: TextGameState, final_state: TextGameState) -> int:
        """Count end-state affordances that expose genuinely new targets or interactions."""

        base_targets = {
            token
            for action in base_state.valid_actions
            if not self._is_movement_action(action)
            for token in action_target_tokens(action, self.action_generator.config.policy.inverse_action_pairs)
        }
        discovered_actions = 0
        for action in final_state.valid_actions:
            if self._is_movement_action(action):
                continue
            targets = action_target_tokens(action, self.action_generator.config.policy.inverse_action_pairs)
            if targets - base_targets:
                discovered_actions += 1
        return min(discovered_actions, 4)

    def _count_new_exit_actions(self, *, base_state: TextGameState, final_state: TextGameState) -> int:
        """Count newly available movement exits at the branch end state."""

        base_exits = {
            self._normalize_text(action)
            for action in base_state.valid_actions
            if self._is_movement_action(action)
        }
        final_exits = {
            self._normalize_text(action)
            for action in final_state.valid_actions
            if self._is_movement_action(action)
        }
        return max(0, len(final_exits - base_exits))

    def _count_inventory_gain(self, *, base_state: TextGameState, final_state: TextGameState) -> int:
        """Count durable inventory items gained by the branch end state."""

        base_items = inventory_item_tokens(
            base_state.inventory_text,
            inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
        )
        final_items = inventory_item_tokens(
            final_state.inventory_text,
            inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
        )
        return max(0, len(final_items - base_items))

    def _count_inventory_loss(self, *, base_state: TextGameState, final_state: TextGameState) -> int:
        """Count inventory items lost by the branch end state."""

        base_items = inventory_item_tokens(
            base_state.inventory_text,
            inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
        )
        final_items = inventory_item_tokens(
            final_state.inventory_text,
            inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
        )
        return max(0, len(base_items - final_items))

    def _loop_penalty_reduction(self, loop_results: list[LoopHeuristicResult]) -> float:
        """Measure whether the branch moved away from an early oscillation pattern."""

        if not loop_results:
            return 0.0
        penalties = [loop_result.total_penalty for loop_result in loop_results]
        return max(0.0, max(penalties) - penalties[-1])

    def _branch_progress_score(
        self,
        *,
        score_change: int,
        persistent_inventory_gain_count: int,
        persistent_inventory_loss_count: int,
        persistent_affordance_gain: int,
        persistent_exit_gain_count: int,
        landmark_gain: int,
        novel_object_count: int,
        room_text_only_gain: float,
        loop_penalty_reduction: float,
        oscillation_penalty_total: float,
        movement_penalty_total: float,
    ) -> float:
        """Score one branch using explicit weighted progress signals."""

        policy = self.action_generator.config.policy
        return (
            max(float(score_change), 0.0) * policy.branch_progress_score_weight
            + float(persistent_inventory_gain_count) * policy.branch_progress_inventory_weight
            + float(persistent_affordance_gain) * policy.branch_progress_affordance_weight
            + float(persistent_exit_gain_count) * policy.branch_progress_exit_weight
            + float(landmark_gain) * policy.branch_progress_location_weight
            + float(novel_object_count) * policy.branch_progress_object_weight
            + float(room_text_only_gain) * policy.room_text_only_weight
            + float(loop_penalty_reduction) * policy.branch_progress_loop_reduction_weight
            - float(persistent_inventory_loss_count) * policy.branch_progress_inventory_loss_penalty
            - float(oscillation_penalty_total)
            - float(movement_penalty_total)
        )

    def _action_is_meaningful_branch_progress(
        self,
        *,
        action: str,
        movement_only_action: bool,
        discard_like_action: bool,
        aggressive_action: bool,
        speculative_tool_use_action: bool,
        score_delta: int,
        persistent_inventory_gain_count: int,
        persistent_affordance_gain: int,
        persistent_exit_gain_count: int,
        persistent_revealed_new_object: bool,
        max_persistent_inventory_gain_count: int,
        max_persistent_affordance_gain: int,
        max_persistent_exit_gain_count: int,
    ) -> bool:
        """Return whether a branch step added durable progress worth committing through."""

        if score_delta > 0:
            return True
        if persistent_inventory_gain_count > max_persistent_inventory_gain_count and not discard_like_action:
            return True
        if persistent_revealed_new_object and not movement_only_action:
            return True
        if persistent_affordance_gain > max_persistent_affordance_gain:
            if not movement_only_action:
                return True
            return persistent_inventory_gain_count > max_persistent_inventory_gain_count
        if persistent_exit_gain_count > max_persistent_exit_gain_count:
            return not movement_only_action
        if aggressive_action or discard_like_action or speculative_tool_use_action:
            return False
        action_text = self._normalize_text(action)
        if action_text.startswith(("take ", "get ", "read ", "examine ", "look at ", "open ")):
            return persistent_inventory_gain_count > max_persistent_inventory_gain_count
        return False

    def _action_is_durable_branch_progress(
        self,
        *,
        action: str,
        movement_only_action: bool,
        discard_like_action: bool,
        aggressive_action: bool,
        speculative_tool_use_action: bool,
        score_delta: int,
        persistent_inventory_gain_count: int,
        persistent_affordance_gain: int,
        persistent_exit_gain_count: int,
        persistent_revealed_new_object: bool,
        max_persistent_inventory_gain_count: int,
        max_persistent_affordance_gain: int,
        max_persistent_exit_gain_count: int,
        score_progress_established: bool,
    ) -> bool:
        """Return whether a step adds durable progress worth committing through.

        This is stricter than `meaningful_progress`: once a branch has already secured
        score progress, later local object churn does not extend the commit boundary
        unless it adds more score or a materially new route/structure.
        """

        if score_delta > 0:
            return True

        if score_progress_established:
            if persistent_exit_gain_count > max_persistent_exit_gain_count and not movement_only_action:
                return True
            return False

        if persistent_inventory_gain_count > max_persistent_inventory_gain_count and not (
            discard_like_action or aggressive_action
        ):
            return True
        if persistent_revealed_new_object and not (
            movement_only_action or discard_like_action or aggressive_action or speculative_tool_use_action
        ):
            return True
        if persistent_affordance_gain > max_persistent_affordance_gain:
            return not (movement_only_action or speculative_tool_use_action)
        if persistent_exit_gain_count > max_persistent_exit_gain_count:
            return True

        action_text = self._normalize_text(action)
        if action_text.startswith(("take ", "get ", "read ", "examine ", "look at ", "open ")):
            return persistent_inventory_gain_count > max_persistent_inventory_gain_count
        return False

    def _should_abort_branch_early(
        self,
        *,
        action_cluster_history: ActionClusterHistory,
        oscillation_penalty_total: float,
        movement_penalty_total: float,
        base_state: TextGameState,
        current_state: TextGameState,
        persistent_inventory_gained: bool,
        persistent_affordance_gain: int,
        persistent_exit_gain_count: int,
        post_gain_churn_action_count: int,
    ) -> bool:
        """Return whether a local branch should fail fast due to low-value looping."""

        total_penalty = oscillation_penalty_total + movement_penalty_total
        if total_penalty >= self.action_generator.config.policy.branch_fail_fast_penalty_threshold:
            return True

        exhausted_families = action_cluster_history.exhausted_families(
            self.action_generator.config.policy.object_family_no_progress_threshold
        )
        if not exhausted_families:
            return False

        durable_progress = (
            current_state.score > base_state.score
            or self._count_inventory_gain(base_state=base_state, final_state=current_state) > 0
            or self._count_new_exit_actions(base_state=base_state, final_state=current_state) > 0
            or self._count_persistent_affordances(base_state=base_state, final_state=current_state) > 0
        )
        if not durable_progress and action_cluster_history.no_progress_steps >= (
            self.action_generator.config.policy.object_family_no_progress_threshold + 1
        ):
            return True

        if (
            action_cluster_history.movement_no_progress_steps
            >= self.action_generator.config.policy.branch_fail_fast_min_movement_actions
        ):
            recent_clusters = [
                cluster
                for cluster in action_cluster_history.recent_cluster_sequence[
                    -action_cluster_history.movement_no_progress_steps :
                ]
                if cluster
            ]
            if recent_clusters and len(set(recent_clusters)) <= 2:
                return True

        if (
            action_cluster_history.bulk_inventory_no_progress_steps
            >= self.action_generator.config.policy.branch_fail_fast_bulk_no_progress_steps
            and not durable_progress
        ):
            return True

        if (
            post_gain_churn_action_count
            >= self.action_generator.config.policy.branch_fail_fast_post_gain_churn_actions
            and persistent_inventory_gained
            and current_state.score <= base_state.score
            and persistent_affordance_gain <= 0
            and persistent_exit_gain_count <= 0
        ):
            return True

        if (
            base_state.score > 0
            and not durable_progress
            and action_cluster_history.no_progress_steps
            >= self.action_generator.config.policy.branch_fail_fast_post_score_no_progress_steps
            and (
                post_gain_churn_action_count > 0
                or current_state.state_cluster_id == base_state.state_cluster_id
            )
        ):
            return True

        return (
            not durable_progress
            and current_state.state_cluster_id == base_state.state_cluster_id
            and action_cluster_history.movement_no_progress_steps >= self.action_generator.config.policy.object_family_no_progress_threshold
        )

    def _is_post_score_stale_branch(
        self,
        *,
        base_state: TextGameState,
        branch: LocalBranchOutcome,
    ) -> bool:
        """Return whether a post-score branch is just re-proving a stale local scene."""

        if base_state.score <= 0:
            return False
        return (
            branch.score_change <= 0
            and branch.persistent_inventory_gain_count <= 0
            and branch.persistent_exit_gain_count <= 0
            and branch.persistent_affordance_gain <= 0
            and (
                branch.post_gain_churn_action_count > 0
                or branch.ended_in_same_cluster
                or branch.termination_reason is BranchTerminationReason.LOOP_ABORTED
                or branch.appears_stuck
            )
        )

    def _is_low_value_movement_branch(
        self,
        *,
        score_change: int,
        persistent_inventory_gain_count: int,
        persistent_affordance_gain: int,
        movement_action_ratio: float,
    ) -> bool:
        """Return whether a branch was mostly movement without durable gains."""

        policy = self.action_generator.config.policy
        return (
            movement_action_ratio >= policy.branch_commit_movement_ratio_threshold
            and score_change <= 0
            and persistent_inventory_gain_count <= 0
            and persistent_affordance_gain <= 0
        )

    def _branch_is_movement_commit_reject(self, branch: LocalBranchOutcome) -> bool:
        """Return whether a branch should be rejected as low-value wandering."""

        return self._is_low_value_movement_branch(
            score_change=branch.score_change,
            persistent_inventory_gain_count=branch.persistent_inventory_gain_count,
            persistent_affordance_gain=branch.persistent_affordance_gain,
            movement_action_ratio=branch.movement_action_ratio,
        )

    def _is_low_value_object_churn_branch(
        self,
        *,
        score_change: int,
        persistent_inventory_gain_count: int,
        persistent_exit_gain_count: int,
        persistent_affordance_gain: int,
        novel_object_count: int,
    ) -> bool:
        """Return whether a branch only churned local affordances without durable gain."""

        return (
            score_change <= 0
            and persistent_inventory_gain_count <= 0
            and persistent_exit_gain_count <= 0
            and persistent_affordance_gain > 0
            and novel_object_count <= 0
        )

    def _branch_is_local_object_churn_reject(self, branch: LocalBranchOutcome) -> bool:
        """Return whether a branch should be rejected as non-movement local object churn."""

        return self._is_low_value_object_churn_branch(
            score_change=branch.score_change,
            persistent_inventory_gain_count=branch.persistent_inventory_gain_count,
            persistent_exit_gain_count=branch.persistent_exit_gain_count,
            persistent_affordance_gain=branch.persistent_affordance_gain,
            novel_object_count=branch.novel_object_count,
        )

    def _inventory_churn_penalty(
        self,
        *,
        score_change: int,
        persistent_inventory_gain_count: int,
        persistent_inventory_loss_count: int,
        persistent_exit_gain_count: int,
        bulk_inventory_action_count: int,
        discard_like_action_count: int,
        aggressive_action_count: int,
        speculative_tool_use_action_count: int,
        post_gain_churn_action_count: int,
    ) -> float:
        """Penalize branches whose apparent value mostly comes from bulk inventory churn."""

        if score_change > 0 or persistent_exit_gain_count > 0:
            return 0.0
        policy = self.action_generator.config.policy
        penalty = 0.0
        if bulk_inventory_action_count > 0:
            penalty += float(bulk_inventory_action_count) * policy.branch_inventory_churn_penalty
        if persistent_inventory_gain_count > 0 and persistent_inventory_loss_count > 0:
            penalty += policy.branch_inventory_churn_penalty
        if persistent_inventory_gain_count > 1:
            penalty += float(persistent_inventory_gain_count - 1) * policy.branch_inventory_churn_penalty
        penalty += float(discard_like_action_count) * policy.branch_post_gain_churn_penalty
        penalty += float(aggressive_action_count) * policy.branch_aggressive_action_penalty
        penalty += float(speculative_tool_use_action_count) * policy.branch_speculative_action_penalty
        penalty += float(post_gain_churn_action_count) * policy.branch_post_gain_churn_penalty
        return penalty

    def _branch_is_inventory_churn_reject(self, branch: LocalBranchOutcome) -> bool:
        """Return whether a branch should be rejected as bulk/transient inventory churn."""

        return (
            branch.score_change <= 0
            and branch.persistent_exit_gain_count <= 0
            and (
                branch.bulk_inventory_action_count > 0
                or branch.discard_like_action_count > 0
                or branch.aggressive_action_count > 0
                or branch.post_gain_churn_action_count > 0
            )
            and branch.inventory_churn_penalty > 0.0
        )

    def _branch_is_zero_score_inventory_scene_churn(
        self,
        *,
        score_change: int,
        persistent_inventory_gain_count: int,
        persistent_exit_gain_count: int,
        oscillation_penalty_total: float,
        appears_stuck: bool,
        inventory_churn_penalty: float,
        post_gain_churn_action_count: int,
        exhausted_family_count: int,
    ) -> bool:
        """Return whether a zero-score inventory branch only churned a local scene.

        This keeps clean zero-score inventory pickups possible, but blocks branches that
        only look good because they grabbed a local item while also looping or stalling.
        """

        return (
            score_change <= 0
            and persistent_inventory_gain_count > 0
            and persistent_exit_gain_count <= 0
            and (
                oscillation_penalty_total > 0.0
                or appears_stuck
                or inventory_churn_penalty > 0.0
                or post_gain_churn_action_count > 0
                or exhausted_family_count > 0
            )
        )

    def _branch_is_zero_score_inventory_scene_churn_reject(self, branch: LocalBranchOutcome) -> bool:
        """Return whether a best branch should be rejected as local inventory scene churn."""

        return self._branch_is_zero_score_inventory_scene_churn(
            score_change=branch.score_change,
            persistent_inventory_gain_count=branch.persistent_inventory_gain_count,
            persistent_exit_gain_count=branch.persistent_exit_gain_count,
            oscillation_penalty_total=branch.oscillation_penalty_total,
            appears_stuck=branch.appears_stuck,
            inventory_churn_penalty=branch.inventory_churn_penalty,
            post_gain_churn_action_count=branch.post_gain_churn_action_count,
            exhausted_family_count=branch.exhausted_family_count,
        )

    def _branch_has_simple_inventory_progress(
        self,
        *,
        score_change: int,
        persistent_inventory_gain_count: int,
        persistent_inventory_loss_count: int,
        persistent_affordance_gain: int,
        persistent_exit_gain_count: int,
        discard_like_action_count: int,
        aggressive_action_count: int,
        speculative_tool_use_action_count: int,
        post_gain_churn_action_count: int,
    ) -> bool:
        """Return whether a zero-score inventory branch ended in a clean, durable state.

        This is intentionally conservative. Zero-score inventory gain only counts when the
        branch keeps the new item(s) without immediately degenerating into local object churn.
        """

        if score_change > 0:
            return True
        if persistent_inventory_gain_count <= 0 or persistent_inventory_loss_count > 0:
            return False
        if discard_like_action_count > 0 or aggressive_action_count > 0:
            return False
        if speculative_tool_use_action_count > 0 or post_gain_churn_action_count > 0:
            return False
        if persistent_inventory_gain_count > 1 and persistent_affordance_gain <= 0 and persistent_exit_gain_count <= 0:
            return False
        return True

    def _looks_like_location_signal(self, observation: str) -> bool:
        """Return whether an observation looks like a room/location transition."""

        tokens = self._significant_tokens(observation)
        return bool(tokens & _LOCATION_HINT_TOKENS)

    def _room_text_only_gain(
        self,
        *,
        base_state: TextGameState,
        final_state: TextGameState,
        score_change: int,
        inventory_changed: bool,
        affordance_gain: int,
        novel_object_count: int,
        landmark_gain: int,
    ) -> float:
        """Return a small gain only when branch novelty is mostly room-text variation."""

        if self._normalize_text(base_state.observation) == self._normalize_text(final_state.observation):
            return 0.0
        if score_change != 0 or inventory_changed or affordance_gain > 0 or novel_object_count > 0:
            return 0.0
        if landmark_gain > 0:
            return 0.0
        return 1.0

    def _annotate_branch_state(
        self,
        state: TextGameState,
        *,
        history: ActionClusterHistory | None,
        previous_state: TextGameState | None,
        score_gain: int,
        inventory_changed: bool,
        affordance_gain: int,
        novel_object_count: int,
        movement_only_action: bool,
        materially_new_actions: bool,
    ) -> TextGameState:
        """Attach region-cluster metadata used by movement suppression heuristics."""

        state.state_cluster_id = derive_state_cluster_id(
            observation=state.observation,
            inventory_text=state.inventory_text,
            valid_actions=state.valid_actions,
            summary_text=str(state.metadata.get("summary_text", "")),
            inverse_pairs=self.action_generator.config.policy.inverse_action_pairs,
        )
        if history is None:
            observation_seen_before = False
            state.cluster_visit_count = max(state.cluster_visit_count, 1)
        else:
            state.cluster_visit_count, observation_seen_before = history.record_state_cluster(
                cluster_id=state.state_cluster_id,
                observation=state.observation,
            )
        state.region_novelty_score = compute_region_novelty_score(
            cluster_visit_count=state.cluster_visit_count,
            observation_seen_before=observation_seen_before,
            score_gain=score_gain,
            inventory_changed=inventory_changed,
            affordance_gain=affordance_gain,
            novel_object_count=novel_object_count,
            materially_new_actions=materially_new_actions,
            movement_only_action=movement_only_action,
        )
        state.metadata = dict(state.metadata)
        state.metadata["state_cluster_id"] = state.state_cluster_id
        state.metadata["cluster_visit_count"] = state.cluster_visit_count
        state.metadata["region_novelty_score"] = state.region_novelty_score
        if previous_state is not None:
            state.metadata["previous_state_cluster_id"] = previous_state.state_cluster_id
        return state

    def _is_movement_action(self, action: str) -> bool:
        """Return whether one action is primarily navigation."""

        return is_movement_action(action, self.action_generator.config.policy.inverse_action_pairs)
