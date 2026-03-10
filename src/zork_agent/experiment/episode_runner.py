"""Single-episode orchestration for the scaffold.

This runner is intentionally explicit. It mixes a deterministic heuristic frontier
with optional LLM-mediated action generation, state selection, and reflection, but
keeps the control flow easy to inspect.

TODO: split diagnostics into dedicated result objects if the episode metadata grows further.
"""

from __future__ import annotations

from collections import Counter
import json
import logging

from zork_agent.config import ProjectConfig
from zork_agent.env.jericho_env import JerichoEnv
from zork_agent.env.replay import restore_saved_node
from zork_agent.llm.base import BaseLLMClient
from zork_agent.llm.prompts import PromptManager
from zork_agent.memory.frontier import FrontierEntry, FrontierQueue, FrontierScoringConfig
from zork_agent.memory.trajectory_store import TrajectoryStore
from zork_agent.policy.action_generator import ActionGenerator
from zork_agent.policy.local_explorer import LocalExplorer
from zork_agent.policy.reflection import ReflectionPolicy
from zork_agent.policy.state_selector import StateSelector
from zork_agent.types import (
    ActionClusterHistory,
    EpisodeResult,
    LoopHeuristicResult,
    MovementHeuristicResult,
    StateSelectionResult,
    TextGameState,
    Trajectory,
    TrajectoryStep,
    derive_state_family_key,
    compute_region_novelty_score,
    derive_state_cluster_id,
    evaluate_reversible_action_loop,
    evaluate_movement_action,
    extract_salient_nouns,
    is_movement_action,
    valid_actions_materially_different,
)
from zork_agent.utils.seeds import set_global_seed
from zork_agent.utils.serialization import to_jsonable


class EpisodeRunner:
    """Run a single research episode with revisit, local branching, and reflection."""

    def __init__(self, config: ProjectConfig, llm_client: BaseLLMClient | None):
        # TODO: split orchestration into smaller services only if the baseline loop becomes hard to follow.
        self.config = config
        self.logger = logging.getLogger("zork_agent.episode_runner")
        self.prompt_manager = PromptManager(config.prompts)
        self.llm_client = llm_client
        self.env = JerichoEnv(config)
        self.trajectory_store = TrajectoryStore(config.paths.trajectory_dir)
        self.summary_builder = self.trajectory_store.summary_builder
        self.frontier = self._build_frontier()
        self.action_generator = ActionGenerator(config, self.prompt_manager, llm_client)
        self.state_selector = StateSelector(
            config=config,
            prompt_manager=self.prompt_manager,
            llm_client=llm_client,
            selection_mode=config.policy.state_selection_mode,
        )
        self.local_explorer = LocalExplorer(
            self.action_generator,
            self.state_selector,
            env=self.env,
            logger=self.logger,
        )
        self.reflection = ReflectionPolicy(config, self.prompt_manager, llm_client)

    def run_episode(
        self,
        episode_index: int = 0,
        episode_id: str | None = None,
        episode_seed: int | None = None,
    ) -> EpisodeResult:
        """Run one episode and persist trajectory plus summary artifacts."""

        resolved_seed = episode_seed if episode_seed is not None else self.config.experiment.seed + episode_index
        set_global_seed(resolved_seed)
        generated_episode_id = episode_id or f"{self.config.experiment.game_id}-episode-{episode_index:03d}"
        self.frontier = self._build_frontier()

        seen_world_hashes: set[str] = set()
        seen_observation_signatures: set[str] = set()
        steps: list[TrajectoryStep] = []
        total_reward = 0.0
        reflection_context = ""
        replay_attempt_count = 0
        replay_success_count = 0
        local_exploration_count = 0
        reflection_update_count = 0
        selection_mode_counts: Counter[str] = Counter()
        action_generation_mode_counts: Counter[str] = Counter()
        restore_mode_counts: Counter[str] = Counter()
        action_source_counts: Counter[str] = Counter()
        pending_branch_actions: list[str] = []
        episode_action_history: list[str] = []
        episode_state_history: list[TextGameState] = []
        episode_loop_results: list[LoopHeuristicResult] = []
        episode_movement_results: list[MovementHeuristicResult] = []
        action_cluster_history = ActionClusterHistory(cluster_label="episode")
        loop_detected_count = 0
        oscillation_penalty_total = 0.0
        movement_loop_detected_count = 0
        movement_penalty_total = 0.0
        branch_commit_rejection_count = 0
        last_selection_result: StateSelectionResult | None = None
        last_local_exploration = None

        try:
            current_state = self._annotate_state(
                self.env.reset(seed=resolved_seed),
                episode_id=generated_episode_id,
                recent_gain=0.0,
                action_cluster_history=action_cluster_history,
                previous_state=None,
                inventory_changed=False,
                affordance_gain=0,
                novel_object_count=0,
                movement_only_action=False,
                materially_new_actions=False,
            )
            action_cluster_history.observe_state_nouns(
                observation=current_state.observation,
                inventory_text=current_state.inventory_text,
                valid_actions=current_state.valid_actions,
                inverse_pairs=self.config.policy.inverse_action_pairs,
            )
            episode_state_history = [current_state]
            seen_world_hashes.add(current_state.world_state_hash)
            seen_observation_signatures.add(self._normalize_text(current_state.observation))
            self._record_frontier_state(
                current_state,
                episode_id=generated_episode_id,
                recent_gain=0.0,
                loop_penalty=0.0,
                movement_penalty=0.0,
                affordance_gain=0,
                room_text_only_gain=0.0,
                seen_world_hashes=seen_world_hashes,
                seen_observation_signatures=seen_observation_signatures,
                notes="initial-state",
                force=True,
                metadata={"seed": resolved_seed, "episode_step_index": -1},
            )

            while len(steps) < self.config.experiment.max_steps and not current_state.done:
                episode_step_index = len(steps)
                selected_entry = None
                restore_result = None
                reflection_result = None
                revisit_made_progress = False
                chosen_action: str | None = None
                action_source = "action_generator"
                loaded_branch_plan_this_step = False
                branch_progress_score = 0.0
                branch_commit_allowed = False
                commit_rejection_reason = ""

                # Revisit a saved frontier node periodically, then probe it locally before committing.
                if (
                    not pending_branch_actions
                    and self._should_run_local_exploration(episode_step_index, replay_attempt_count)
                ):
                    last_selection_result = self.state_selector.select_with_details(
                        self.frontier,
                        mode=self.config.policy.state_selection_mode,
                    )
                    selected_entry = last_selection_result.selected
                    selection_mode_counts[last_selection_result.selection_mode.value] += 1

                    if selected_entry is not None and selected_entry.saved_node is not None:
                        replay_attempt_count += 1
                        restore_result = restore_saved_node(
                            self.env,
                            selected_entry.saved_node,
                            target_step_index=selected_entry.step_index,
                            logger=self.logger,
                        )
                        restore_mode_counts[restore_result.restore_mode.value] += 1
                        if restore_result.success:
                            replay_success_count += 1
                            action_cluster_history = ActionClusterHistory(cluster_label="revisit")
                            current_state = self._annotate_state(
                                restore_result.final_state,
                                episode_id=generated_episode_id,
                                recent_gain=0.0,
                                action_cluster_history=action_cluster_history,
                                previous_state=None,
                                inventory_changed=False,
                                affordance_gain=0,
                                novel_object_count=0,
                                movement_only_action=False,
                                materially_new_actions=False,
                            )
                            action_cluster_history.observe_state_nouns(
                                observation=current_state.observation,
                                inventory_text=current_state.inventory_text,
                                valid_actions=current_state.valid_actions,
                                inverse_pairs=self.config.policy.inverse_action_pairs,
                            )
                            episode_action_history = []
                            episode_state_history = [current_state]
                            episode_loop_results = []
                            episode_movement_results = []
                            last_local_exploration = self.local_explorer.explore_from_state(
                                current_state,
                                env=self.env,
                                branch_count=self.config.policy.rollout_count,
                                branch_horizon=self.config.policy.rollout_depth,
                                temperature=self.config.llm.temperature,
                                action_candidate_count=self.config.policy.action_candidates,
                                recent_trajectory_context=self._prompt_context(steps, reflection_context),
                            )
                            local_exploration_count += 1
                            reflection_result = self.reflection.reflect_rollouts(last_local_exploration)
                            reflection_update_count += 1
                            if reflection_result.prompt_context:
                                reflection_context = self._merge_guidance(
                                    reflection_context,
                                    reflection_result.prompt_context,
                                )

                            best_branch = last_local_exploration.best_branch
                            branch_progress_score = last_local_exploration.best_branch_progress_score
                            branch_commit_allowed = last_local_exploration.branch_commit_allowed
                            commit_rejection_reason = last_local_exploration.commit_rejection_reason
                            if not branch_commit_allowed:
                                branch_commit_rejection_count += 1
                                self.logger.info(
                                    "Skipping local branch commit for %s because no branch cleared the progress "
                                    "threshold. progress_score=%.2f rejection=%s",
                                    selected_entry.state_id,
                                    branch_progress_score,
                                    commit_rejection_reason or "none",
                                )
                            if (
                                best_branch is not None
                                and best_branch.actions_taken
                                and selected_entry.saved_node is not None
                                and branch_commit_allowed
                            ):
                                branch_restore = restore_saved_node(
                                    self.env,
                                    selected_entry.saved_node,
                                    target_step_index=selected_entry.step_index,
                                    logger=self.logger,
                                )
                                if branch_restore.success:
                                    action_cluster_history = ActionClusterHistory(cluster_label="branch-commit")
                                    current_state = self._annotate_state(
                                        branch_restore.final_state,
                                        episode_id=generated_episode_id,
                                        recent_gain=0.0,
                                        action_cluster_history=action_cluster_history,
                                        previous_state=None,
                                        inventory_changed=False,
                                        affordance_gain=0,
                                        novel_object_count=0,
                                        movement_only_action=False,
                                        materially_new_actions=False,
                                    )
                                    action_cluster_history.observe_state_nouns(
                                        observation=current_state.observation,
                                        inventory_text=current_state.inventory_text,
                                        valid_actions=current_state.valid_actions,
                                        inverse_pairs=self.config.policy.inverse_action_pairs,
                                    )
                                    episode_action_history = []
                                    episode_state_history = [current_state]
                                    episode_loop_results = []
                                    episode_movement_results = []
                                    pending_branch_actions = list(
                                        best_branch.actions_taken[: self.config.experiment.branch_commit_steps]
                                    )
                                    loaded_branch_plan_this_step = bool(pending_branch_actions)
                                else:
                                    self.logger.warning(
                                        "Could not restore selected node %s after local exploration; "
                                        "falling back to action generation.",
                                        selected_entry.state_id,
                                    )
                                    action_cluster_history = ActionClusterHistory(cluster_label="branch-fallback")
                                    current_state = self._annotate_state(
                                        branch_restore.final_state,
                                        episode_id=generated_episode_id,
                                        recent_gain=0.0,
                                        action_cluster_history=action_cluster_history,
                                        previous_state=None,
                                        inventory_changed=False,
                                        affordance_gain=0,
                                        novel_object_count=0,
                                        movement_only_action=False,
                                        materially_new_actions=False,
                                    )
                                    action_cluster_history.observe_state_nouns(
                                        observation=current_state.observation,
                                        inventory_text=current_state.inventory_text,
                                        valid_actions=current_state.valid_actions,
                                        inverse_pairs=self.config.policy.inverse_action_pairs,
                                    )
                                    episode_action_history = []
                                    episode_state_history = [current_state]
                                    episode_loop_results = []
                                    episode_movement_results = []
                                    pending_branch_actions = []
                            revisit_made_progress = bool(branch_commit_allowed)
                        else:
                            revisit_made_progress = False

                    if selected_entry is not None:
                        self.frontier.note_revisit_outcome(
                            selected_entry,
                            made_progress=revisit_made_progress,
                        )

                # Take exactly one real environment step for the episode trace.
                if chosen_action is None:
                    if pending_branch_actions:
                        chosen_action = pending_branch_actions.pop(0)
                        action_source = "best_local_branch" if loaded_branch_plan_this_step else "best_local_branch_plan"
                    else:
                        proposals = self.action_generator.propose_actions(
                            current_state,
                            recent_trajectory_context=self._prompt_context(steps, reflection_context),
                            recent_actions=episode_action_history,
                            recent_loop_results=episode_loop_results,
                            state_action_history=action_cluster_history,
                        )
                        if self.action_generator.last_result is not None:
                            action_generation_mode_counts[self.action_generator.last_result.mode.value] += 1
                        chosen_action = proposals[0].action if proposals else "look"

                action_source_counts[action_source] += 1
                pre_step_state = current_state
                transition = self.env.step(chosen_action)
                total_reward += transition.reward
                raw_state = transition.to_state()
                affordance_gain = self._count_new_affordances(
                    base_state=pre_step_state,
                    final_state=raw_state,
                )
                novel_object_tokens = (
                    extract_salient_nouns(
                        observation=raw_state.observation,
                        inventory_text=raw_state.inventory_text,
                        valid_actions=[],
                        inverse_pairs=self.config.policy.inverse_action_pairs,
                    )
                    - extract_salient_nouns(
                        observation=pre_step_state.observation,
                        inventory_text=pre_step_state.inventory_text,
                        valid_actions=[],
                        inverse_pairs=self.config.policy.inverse_action_pairs,
                    )
                )
                inventory_changed = self._normalize_text(raw_state.inventory_text) != self._normalize_text(
                    pre_step_state.inventory_text
                )
                materially_new_actions = valid_actions_materially_different(
                    pre_step_state.valid_actions,
                    raw_state.valid_actions,
                )
                next_state = self._annotate_state(
                    raw_state,
                    episode_id=generated_episode_id,
                    recent_gain=float(transition.score - pre_step_state.score),
                    action_cluster_history=action_cluster_history,
                    previous_state=pre_step_state,
                    inventory_changed=inventory_changed,
                    affordance_gain=affordance_gain,
                    novel_object_count=len(novel_object_tokens),
                    movement_only_action=is_movement_action(
                        chosen_action,
                        self.config.policy.inverse_action_pairs,
                    ),
                    materially_new_actions=materially_new_actions,
                )
                action_cluster_history.record_attempt(
                    action=chosen_action,
                    score_changed=next_state.score != pre_step_state.score,
                    inventory_changed=inventory_changed,
                    observation_changed=self._normalize_text(next_state.observation)
                    != self._normalize_text(pre_step_state.observation),
                    valid_actions_changed={
                        self._normalize_text(candidate) for candidate in next_state.valid_actions
                    }
                    != {
                        self._normalize_text(candidate) for candidate in pre_step_state.valid_actions
                    },
                )
                loop_result = evaluate_reversible_action_loop(
                    action=chosen_action,
                    recent_actions=episode_action_history,
                    state_history=episode_state_history,
                    current_state=next_state,
                    inverse_pairs=self.config.policy.inverse_action_pairs,
                    immediate_inverse_penalty=self.config.policy.immediate_inverse_penalty,
                    repeated_pair_penalty=self.config.policy.repeated_pair_penalty,
                    reversible_no_progress_penalty=self.config.policy.reversible_no_progress_penalty,
                )
                if loop_result.loop_detected:
                    loop_detected_count += 1
                    oscillation_penalty_total += loop_result.total_penalty
                movement_result = evaluate_movement_action(
                    action=chosen_action,
                    recent_actions=episode_action_history,
                    previous_state=pre_step_state,
                    current_state=next_state,
                    inverse_pairs=self.config.policy.inverse_action_pairs,
                    repeated_movement_penalty=self.config.policy.repeated_movement_penalty,
                    movement_cycle_penalty=self.config.policy.movement_cycle_penalty,
                    same_region_repeat_penalty=self.config.policy.same_region_repeat_penalty,
                )
                episode_movement_results.append(movement_result)
                movement_penalty_total += movement_result.total_penalty
                if movement_result.total_penalty > 0.0:
                    movement_loop_detected_count += 1
                room_text_only_gain = self._room_text_only_gain(
                    previous_state=pre_step_state,
                    current_state=next_state,
                    score_gain=next_state.score - pre_step_state.score,
                    inventory_changed=inventory_changed,
                    affordance_gain=affordance_gain,
                    novel_object_count=len(novel_object_tokens),
                )
                step = transition.to_trajectory_step(
                    episode_id=generated_episode_id,
                    step_index_override=episode_step_index,
                    loop_result=loop_result,
                    metadata={
                        "seed": resolved_seed,
                        "mode": transition.metadata.get("mode", "unknown"),
                        "action_source": action_source,
                        "loop_penalty": loop_result.total_penalty,
                        "loop_no_progress": loop_result.no_progress,
                        "loop_reason": loop_result.reason,
                        "movement_only_action": movement_result.movement_only_action,
                        "movement_repeat_count": movement_result.movement_repeat_count,
                        "movement_penalty": movement_result.total_penalty,
                        "movement_penalty_reason": movement_result.reason,
                        "state_cluster_id": next_state.state_cluster_id,
                        "cluster_visit_count": next_state.cluster_visit_count,
                        "region_novelty_score": next_state.region_novelty_score,
                        "room_text_only_gain": room_text_only_gain,
                        "affordance_gain": affordance_gain,
                        "selected_frontier_state": selected_entry.state_id if selected_entry is not None else None,
                        "selected_frontier_reason": last_selection_result.reason if last_selection_result is not None else "",
                        "selection_mode": (
                            last_selection_result.selection_mode.value if last_selection_result is not None else ""
                        ),
                        "restore_mode": restore_result.restore_mode.value if restore_result is not None else "",
                        "reflection_mode": (
                            reflection_result.mode.value if reflection_result is not None else ""
                        ),
                        "reflection_removed_items": (
                            list(reflection_result.guidance.unsupported_items_removed)
                            if reflection_result is not None
                            else []
                        ),
                        "reflection_supported_try_actions": (
                            list(reflection_result.guidance.supported_try_actions)
                            if reflection_result is not None
                            else []
                        ),
                        "reflection_supported_avoid_actions": (
                            list(reflection_result.guidance.supported_avoid_actions)
                            if reflection_result is not None
                            else []
                        ),
                        "branch_progress_score": branch_progress_score,
                        "branch_commit_allowed": branch_commit_allowed,
                        "commit_rejection_reason": commit_rejection_reason,
                        "episode_step_index": episode_step_index,
                        "restore_step_index": (
                            next_state.world_state_snapshot.step_index
                            if next_state.world_state_snapshot is not None
                            else transition.step_index
                        ),
                    },
                )
                steps.append(step)

                self.logger.info(
                    "Episode %s step %s action=%s source=%s score=%s reward=%.2f frontier=%s loop_penalty=%.2f "
                    "movement_penalty=%.2f cluster=%s region_novelty=%.2f affordance_gain=%s room_text_only_gain=%.2f "
                    "branch_progress_score=%.2f branch_commit_allowed=%s commit_rejection_reason=%s",
                    generated_episode_id,
                    episode_step_index,
                    chosen_action,
                    action_source,
                    next_state.score,
                    transition.reward,
                    len(self.frontier),
                    loop_result.total_penalty,
                    movement_result.total_penalty,
                    next_state.state_cluster_id or "unknown",
                    next_state.region_novelty_score,
                    affordance_gain,
                    room_text_only_gain,
                    branch_progress_score,
                    branch_commit_allowed,
                    commit_rejection_reason or "none",
                )

                if self._should_refresh_frontier(
                    step_count=len(steps),
                    previous_state=pre_step_state,
                    current_state=next_state,
                    reward=transition.reward,
                ):
                    self._record_frontier_state(
                        next_state,
                        episode_id=generated_episode_id,
                        recent_gain=float(next_state.score - pre_step_state.score),
                        loop_penalty=loop_result.total_penalty,
                        movement_penalty=movement_result.total_penalty,
                        affordance_gain=affordance_gain,
                        room_text_only_gain=room_text_only_gain,
                        seen_world_hashes=seen_world_hashes,
                        seen_observation_signatures=seen_observation_signatures,
                        notes=f"after:{chosen_action}",
                        force=False,
                        metadata={
                            "seed": resolved_seed,
                            "episode_step_index": episode_step_index,
                            "action_source": action_source,
                            "loop_no_progress": loop_result.no_progress,
                        },
                    )

                episode_action_history.append(chosen_action)
                episode_state_history.append(next_state)
                episode_loop_results.append(loop_result)
                action_cluster_history.observe_state_nouns(
                    observation=next_state.observation,
                    inventory_text=next_state.inventory_text,
                    valid_actions=next_state.valid_actions,
                    inverse_pairs=self.config.policy.inverse_action_pairs,
                )
                current_state = next_state
        finally:
            self.env.close()

        final_reflection = self.reflection.reflect_trajectory(steps)
        if final_reflection.prompt_context:
            reflection_context = self._merge_guidance(reflection_context, final_reflection.prompt_context)
        trajectory = Trajectory(episode_id=generated_episode_id, steps=steps)
        trajectory_path = self.trajectory_store.write_trajectory(trajectory)
        episode_metadata = {
            "done": current_state.done if steps else False,
            "frontier_size": len(self.frontier),
            "replay_attempt_count": replay_attempt_count,
            "replay_success_count": replay_success_count,
            "local_exploration_count": local_exploration_count,
            "reflection_update_count": reflection_update_count,
            "selection_mode_counts": dict(selection_mode_counts),
            "action_generation_mode_counts": dict(action_generation_mode_counts),
            "restore_mode_counts": dict(restore_mode_counts),
            "action_source_counts": dict(action_source_counts),
            "loop_detected_count": loop_detected_count,
            "oscillation_penalty_total": oscillation_penalty_total,
            "movement_loop_detected_count": movement_loop_detected_count,
            "movement_penalty_total": movement_penalty_total,
            "branch_commit_rejection_count": branch_commit_rejection_count,
            "final_guidance": reflection_context,
            "final_reflection_mode": final_reflection.mode.value,
            "reflection_removed_items": list(final_reflection.guidance.unsupported_items_removed),
            "reflection_supported_try_actions": list(final_reflection.guidance.supported_try_actions),
            "reflection_supported_avoid_actions": list(final_reflection.guidance.supported_avoid_actions),
            "state_selection_mode": self.config.policy.state_selection_mode,
            "max_replay_attempts": self.config.experiment.max_replay_attempts,
            "frontier_refresh_cadence": self.config.experiment.frontier_refresh_cadence,
            "local_exploration_cadence": self.config.experiment.local_exploration_cadence,
            "branch_commit_steps": self.config.experiment.branch_commit_steps,
            "rollout_count": self.config.policy.rollout_count,
            "rollout_depth": self.config.policy.rollout_depth,
            "action_candidates": self.config.policy.action_candidates,
        }
        if last_selection_result is not None:
            episode_metadata["last_selection_reason"] = last_selection_result.reason
            episode_metadata["last_selection_mode"] = last_selection_result.selection_mode.value
        if last_local_exploration is not None and last_local_exploration.best_branch is not None:
            episode_metadata["best_branch_actions"] = list(last_local_exploration.best_branch.actions_taken)
            episode_metadata["best_branch_score_change"] = last_local_exploration.best_branch.score_change
            episode_metadata["best_branch_progress_score"] = last_local_exploration.best_branch.branch_progress_score
            episode_metadata["branch_commit_allowed"] = last_local_exploration.branch_commit_allowed
            episode_metadata["commit_rejection_reason"] = last_local_exploration.commit_rejection_reason

        summary_path = self._write_summary_json(
            generated_episode_id,
            {
                "episode_id": generated_episode_id,
                "seed": resolved_seed,
                "total_reward": total_reward,
                "step_count": len(steps),
                "final_score": current_state.score,
                "trajectory_path": str(trajectory_path),
                "final_state_summary": self.summary_builder.summarize_state_candidate(
                    observation=current_state.observation,
                    score=current_state.score,
                    depth=(
                        current_state.world_state_snapshot.step_index + 1
                        if current_state.world_state_snapshot is not None
                        else len(steps)
                    ),
                    recent_gain=0.0,
                    inventory_text=current_state.inventory_text,
                    valid_actions=current_state.valid_actions,
                ),
                "notes": reflection_context,
                "metadata": episode_metadata,
            },
        )
        return EpisodeResult(
            episode_id=generated_episode_id,
            seed=resolved_seed,
            total_reward=total_reward,
            step_count=len(steps),
            final_score=current_state.score,
            trajectory_path=trajectory_path,
            summary_path=summary_path,
            notes=reflection_context,
            metadata=episode_metadata,
        )

    def _build_frontier(self) -> FrontierQueue:
        """Construct a new frontier queue from config."""

        return FrontierQueue(
            max_size=self.config.policy.frontier_max_size,
            scoring=FrontierScoringConfig(
                snapshot_retention_limit=self.config.policy.snapshot_retention_limit,
                loop_penalty_weight=self.config.policy.frontier_loop_penalty_weight,
                movement_penalty_weight=self.config.policy.frontier_movement_penalty_weight,
                room_text_only_penalty_weight=self.config.policy.frontier_room_text_only_penalty_weight,
                reversible_state_penalty_weight=self.config.policy.frontier_reversible_state_penalty_weight,
                revisit_saturation_penalty_weight=self.config.policy.frontier_revisit_saturation_penalty_weight,
                cluster_revisit_saturation_weight=self.config.policy.frontier_cluster_revisit_saturation_weight,
                reversible_toggle_novelty_scale=self.config.policy.frontier_reversible_toggle_novelty_scale,
                trivial_observation_novelty_scale=self.config.policy.frontier_trivial_observation_novelty_scale,
                oscillating_pair_novelty_scale=self.config.policy.frontier_oscillating_pair_novelty_scale,
                family_repeat_novelty_decay=self.config.policy.frontier_family_repeat_novelty_decay,
                family_repeat_penalty=self.config.policy.frontier_family_repeat_penalty,
                family_revisit_saturation_weight=self.config.policy.frontier_family_revisit_saturation_weight,
                base_reversible_state_penalty=self.config.policy.frontier_base_reversible_state_penalty,
                oscillating_pair_penalty=self.config.policy.frontier_oscillating_pair_penalty,
                trivial_reversible_penalty=self.config.policy.frontier_trivial_reversible_penalty,
            ),
        )

    def _annotate_state(
        self,
        state: TextGameState,
        *,
        episode_id: str,
        recent_gain: float,
        action_cluster_history: ActionClusterHistory,
        previous_state: TextGameState | None,
        inventory_changed: bool,
        affordance_gain: int,
        novel_object_count: int,
        movement_only_action: bool,
        materially_new_actions: bool,
    ) -> TextGameState:
        """Attach episode-scoped metadata to a live state object."""

        state.state_cluster_id = derive_state_cluster_id(
            observation=state.observation,
            inventory_text=state.inventory_text,
            valid_actions=state.valid_actions,
            summary_text=str(state.metadata.get("summary_text", "")),
            inverse_pairs=self.config.policy.inverse_action_pairs,
        )
        cluster_visit_count, observation_seen_before = action_cluster_history.record_state_cluster(
            cluster_id=state.state_cluster_id,
            observation=state.observation,
        )
        state.cluster_visit_count = cluster_visit_count
        state.region_novelty_score = compute_region_novelty_score(
            cluster_visit_count=cluster_visit_count,
            observation_seen_before=observation_seen_before,
            score_gain=int(recent_gain),
            inventory_changed=inventory_changed,
            affordance_gain=affordance_gain,
            novel_object_count=novel_object_count,
            materially_new_actions=materially_new_actions,
            movement_only_action=movement_only_action,
        )
        summary_text = self.summary_builder.summarize_state_candidate(
            observation=state.observation,
            score=state.score,
            depth=(
                state.world_state_snapshot.step_index + 1
                if state.world_state_snapshot is not None
                else state.moves
            ),
            recent_gain=recent_gain,
            inventory_text=state.inventory_text,
            valid_actions=state.valid_actions,
        )
        state.metadata = dict(state.metadata)
        state.metadata["episode_id"] = episode_id
        state.metadata["summary_text"] = summary_text
        state.metadata["state_cluster_id"] = state.state_cluster_id
        state.metadata["cluster_visit_count"] = state.cluster_visit_count
        state.metadata["region_novelty_score"] = state.region_novelty_score
        if previous_state is not None:
            state.metadata["previous_state_cluster_id"] = previous_state.state_cluster_id
        return state

    def _record_frontier_state(
        self,
        state: TextGameState,
        *,
        episode_id: str,
        recent_gain: float,
        seen_world_hashes: set[str],
        seen_observation_signatures: set[str],
        notes: str,
        force: bool,
        loop_penalty: float = 0.0,
        movement_penalty: float = 0.0,
        affordance_gain: int = 0,
        room_text_only_gain: float = 0.0,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Convert the current state into a replayable frontier entry."""

        summary_text = str(state.metadata.get("summary_text", "")) or self.summary_builder.summarize_state_candidate(
            observation=state.observation,
            score=state.score,
            depth=(
                state.world_state_snapshot.step_index + 1
                if state.world_state_snapshot is not None
                else state.moves
            ),
            recent_gain=recent_gain,
            inventory_text=state.inventory_text,
            valid_actions=state.valid_actions,
        )
        observation_signature = self._normalize_text(state.observation)
        is_new_hash = state.world_state_hash not in seen_world_hashes
        is_new_observation = observation_signature not in seen_observation_signatures
        state_family_key = derive_state_family_key(
            observation=state.observation,
            inventory_text=state.inventory_text,
            valid_actions=state.valid_actions,
            summary_text=summary_text,
            inverse_pairs=self.config.policy.inverse_action_pairs,
        )
        novelty = self._estimate_novelty(
            is_new_hash=is_new_hash,
            is_new_observation=is_new_observation,
            state=state,
            recent_gain=recent_gain,
            affordance_gain=affordance_gain,
            movement_penalty=movement_penalty,
            room_text_only_gain=room_text_only_gain,
        )

        if not force and novelty <= 0.0 and recent_gain <= 0.0:
            return

        snapshot = state.world_state_snapshot
        restore_step_index = snapshot.step_index if snapshot is not None else state.moves - 1
        replay_actions = list(snapshot.replay_actions) if snapshot is not None else []
        saved_node = self.trajectory_store.saved_node_from_snapshot(
            episode_id=episode_id,
            step_index=restore_step_index,
            observation=state.observation,
            score=state.score,
            inventory_text=state.inventory_text,
            world_state_hash=state.world_state_hash,
            action_prefix=replay_actions,
            snapshot=snapshot,
            valid_actions=state.valid_actions,
            summary_text=summary_text,
            metadata={
                **dict(metadata or {}),
                "state_family_key": state_family_key,
                "state_cluster_id": state.state_cluster_id,
                "region_novelty_score": state.region_novelty_score,
                "movement_penalty": movement_penalty,
                "affordance_gain": affordance_gain,
                "room_text_only_gain": room_text_only_gain,
            },
        )
        self.frontier.add(
            FrontierEntry(
                state_id=saved_node.state_id,
                score=float(state.score),
                depth=max(restore_step_index + 1, 0),
                novelty=novelty,
                effective_novelty=novelty,
                recent_gain=recent_gain,
                loop_penalty=loop_penalty,
                movement_penalty=movement_penalty,
                affordance_gain=affordance_gain,
                room_text_only_gain=room_text_only_gain,
                state_family_key=state_family_key,
                state_cluster_id=state.state_cluster_id,
                cluster_visit_count=state.cluster_visit_count,
                region_novelty_score=state.region_novelty_score,
                oscillating_pair_member=loop_penalty > 0.0 or movement_penalty > 0.0,
                trivial_reversible_change=bool(metadata and metadata.get("loop_no_progress", False))
                or (movement_penalty > 0.0 and affordance_gain <= 0 and recent_gain <= 0.0),
                episode_id=episode_id,
                step_index=restore_step_index,
                world_state_hash=state.world_state_hash,
                observation=state.observation,
                inventory_text=state.inventory_text,
                valid_actions=list(state.valid_actions),
                replay_actions=replay_actions,
                summary_text=summary_text,
                saved_node=saved_node,
                notes=notes,
                metadata=dict(metadata or {}),
            )
        )
        seen_world_hashes.add(state.world_state_hash)
        seen_observation_signatures.add(observation_signature)

    def _should_run_local_exploration(self, episode_step_index: int, replay_attempt_count: int) -> bool:
        """Return whether the loop should revisit a saved node before the next real step."""

        cadence = max(1, self.config.experiment.local_exploration_cadence)
        if self.config.experiment.max_replay_attempts <= replay_attempt_count:
            return False
        if episode_step_index <= 0:
            return False
        return episode_step_index % cadence == 0 and len(self.frontier) > 0

    def _should_refresh_frontier(
        self,
        *,
        step_count: int,
        previous_state: TextGameState,
        current_state: TextGameState,
        reward: float,
    ) -> bool:
        """Return whether the current state should be offered to the frontier."""

        cadence = max(1, self.config.experiment.frontier_refresh_cadence)
        return (
            step_count % cadence == 0
            or reward != 0.0
            or current_state.score != previous_state.score
            or current_state.done
            or current_state.world_state_hash != previous_state.world_state_hash
        )

    def _prompt_context(self, steps: list[TrajectoryStep], reflection_context: str) -> str:
        """Build compact prompt context from recent actions plus reflection guidance."""

        parts: list[str] = []
        if steps:
            parts.append("recent_actions=" + " -> ".join(step.action for step in steps[-4:]))
        if reflection_context:
            parts.append("guidance=" + reflection_context)
        return " | ".join(parts)

    def _merge_guidance(self, existing: str, new_guidance: str) -> str:
        """Keep reflection guidance compact enough for reuse in later prompts."""

        if not new_guidance:
            return existing
        merged = " | ".join(part.strip() for part in (existing, new_guidance) if part.strip())
        normalized = " ".join(merged.split()).strip()
        if len(normalized) <= 240:
            return normalized
        return normalized[:237].rstrip() + "..."

    def _estimate_novelty(
        self,
        *,
        is_new_hash: bool,
        is_new_observation: bool,
        state: TextGameState,
        recent_gain: float,
        affordance_gain: int,
        movement_penalty: float,
        room_text_only_gain: float,
    ) -> float:
        """Return an explicit novelty heuristic that discounts movement-only variation."""

        novelty = 0.0
        if is_new_hash:
            novelty += 1.0
        if is_new_observation:
            novelty += 0.25
        if recent_gain > 0.0:
            novelty += min(recent_gain, 2.0)
        if affordance_gain > 0:
            novelty += min(float(affordance_gain) * 0.35, 1.0)
        novelty *= max(state.region_novelty_score, 0.05)
        if room_text_only_gain > 0.0 and recent_gain <= 0.0 and affordance_gain <= 0:
            novelty *= 0.2
        if movement_penalty > 0.0 and recent_gain <= 0.0 and affordance_gain <= 0:
            novelty *= 0.2
        return novelty

    def _count_new_affordances(self, *, base_state: TextGameState, final_state: TextGameState) -> int:
        """Count newly surfaced valid actions between two neighboring states."""

        base_actions = {
            self._normalize_text(action)
            for action in base_state.valid_actions
            if self._normalize_text(action)
        }
        discovered_actions = {
            self._normalize_text(action)
            for action in final_state.valid_actions
            if self._normalize_text(action) and self._normalize_text(action) not in base_actions
        }
        return min(len(discovered_actions), 4)

    def _room_text_only_gain(
        self,
        *,
        previous_state: TextGameState,
        current_state: TextGameState,
        score_gain: int,
        inventory_changed: bool,
        affordance_gain: int,
        novel_object_count: int,
    ) -> float:
        """Return a small gain only when the state changed mostly in room text."""

        if self._normalize_text(previous_state.observation) == self._normalize_text(current_state.observation):
            return 0.0
        if score_gain != 0 or inventory_changed or affordance_gain > 0 or novel_object_count > 0:
            return 0.0
        if current_state.state_cluster_id != previous_state.state_cluster_id:
            return 1.0
        return 0.5

    def _normalize_text(self, text: str) -> str:
        """Normalize text for novelty bookkeeping."""

        return " ".join(text.split()).strip().lower()

    def _write_summary_json(self, episode_id: str, payload: dict[str, object]):
        """Persist a human-readable episode summary JSON sidecar."""

        summary_path = self.config.paths.summary_dir / f"{episode_id}.summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(to_jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary_path
