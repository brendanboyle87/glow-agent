"""Single-episode orchestration for the GLoW implementation.

This runner is intentionally explicit. It mixes a deterministic heuristic frontier
with optional LLM-mediated action generation, state selection, and reflection, but
keeps the control flow easy to inspect.

TODO: split diagnostics into dedicated result objects if the episode metadata grows further.
"""

from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path

from zork_agent.config import ProjectConfig
from zork_agent.env.jericho_env import JerichoEnv
from zork_agent.env.replay import restore_saved_node
from zork_agent.llm.base import BaseLLMClient
from zork_agent.llm.prompts import PromptManager
from zork_agent.memory.archive_store import ArchiveStore
from zork_agent.memory.archive_updater import ArchiveUpdater
from zork_agent.memory.frontier import (
    FrontierEntry,
    FrontierQueue,
    FrontierScoringConfig,
    TrajectoryFrontier,
    TrajectoryFrontierConfig,
)
from zork_agent.memory.trajectory_store import TrajectoryStore
from zork_agent.policy.action_generator import ActionGenerator
from zork_agent.policy.frontier_analyzer import FrontierAnalyzer
from zork_agent.policy.local_explorer import LocalExplorer
from zork_agent.policy.reflection import ReflectionPolicy
from zork_agent.policy.state_selector import StateSelector
from zork_agent.types import (
    ActionClusterHistory,
    ArchiveStateSelectionResult,
    ArchivedState,
    EpisodeMapMemory,
    GlowEpisodeMetrics,
    GlowSelectionDecisionMetric,
    EpisodeResult,
    EpisodeTrajectory,
    FrontierAnalysisResult,
    LocalBranchOutcome,
    LocalExplorationResult,
    LoopHeuristicResult,
    MovementHeuristicResult,
    ReplayMetadata,
    ReflectionGuidance,
    RunnerMode,
    SavedNode,
    StrategicGuidance,
    StrategicMode,
    StateSelectionResult,
    TextGameState,
    Trajectory,
    TrajectoryStep,
    derive_state_family_key,
    compute_region_novelty_score,
    derive_state_cluster_id,
    action_target_tokens,
    evaluate_reversible_action_loop,
    evaluate_movement_action,
    extract_salient_nouns,
    inventory_item_tokens,
    is_bulk_inventory_action,
    is_discard_like_action,
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
        self.archive_store = ArchiveStore(config.paths.archive_dir)
        self.archive_updater = ArchiveUpdater(self.trajectory_store.summary_builder)
        self.summary_builder = self.trajectory_store.summary_builder
        self.frontier = self._build_frontier()
        self.trajectory_frontier = self._build_trajectory_frontier()
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
        self.frontier_analyzer = FrontierAnalyzer(
            config=config,
            prompt_manager=self.prompt_manager,
            llm_client=llm_client,
            logger=self.logger,
        )
        self.reflection = ReflectionPolicy(config, self.prompt_manager, llm_client)

    def run_episode(
        self,
        episode_index: int = 0,
        episode_id: str | None = None,
        episode_seed: int | None = None,
    ) -> EpisodeResult:
        """Run one episode using the configured legacy or GLoW-style control loop."""

        if self.config.experiment.runner_mode is RunnerMode.GLOW_FAITHFUL:
            return self._run_glow_episode(
                episode_index=episode_index,
                episode_id=episode_id,
                episode_seed=episode_seed,
            )
        return self._run_legacy_episode(
            episode_index=episode_index,
            episode_id=episode_id,
            episode_seed=episode_seed,
        )

    def _run_legacy_episode(
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
        latest_reflection_guidance = ReflectionGuidance()
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
        episode_map_memory = EpisodeMapMemory()
        loop_detected_count = 0
        oscillation_penalty_total = 0.0
        movement_loop_detected_count = 0
        movement_penalty_total = 0.0
        branch_commit_rejection_count = 0
        consecutive_no_durable_gain_steps = 0
        consecutive_same_cluster_movement_steps = 0
        recent_stalled_movement_clusters: list[str] = []
        episode_fail_fast_reason = ""
        local_exploration_cooldown_steps = 0
        last_selection_result: StateSelectionResult | None = None
        last_local_exploration = None
        latest_strategic_guidance = StrategicGuidance(mode=StrategicMode.EXPLORE, reason="initial mapping")
        episode_mode_counts: Counter[str] = Counter()

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
            self._annotate_strategic_state(
                current_state,
                episode_map_memory=episode_map_memory,
                step_index=0,
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
                action_generation_result = None
                loaded_branch_plan_this_step = False
                branch_progress_score = 0.0
                branch_commit_allowed = False
                commit_rejection_reason = ""
                latest_strategic_guidance = episode_map_memory.recommend_guidance(
                    current_state=current_state,
                    frontier_entries=self.frontier.top_k(6),
                    cluster_visit_exhaustion_threshold=self.config.policy.region_visit_exhaustion_threshold,
                    min_opportunity_priority=self.config.policy.min_strategic_opportunity_priority,
                )
                latest_strategic_guidance = self._apply_stall_recovery_guidance_override(
                    strategic_guidance=latest_strategic_guidance,
                    current_state=current_state,
                    consecutive_no_durable_gain_steps=consecutive_no_durable_gain_steps,
                )
                episode_mode_counts[latest_strategic_guidance.mode.value] += 1

                # Revisit a saved frontier node periodically, then probe it locally before committing.
                if (
                    not pending_branch_actions
                    and self._should_run_local_exploration(
                        episode_step_index,
                        replay_attempt_count,
                        local_exploration_cooldown_steps,
                        strategic_guidance=latest_strategic_guidance,
                        current_cluster_id=current_state.state_cluster_id,
                    )
                ):
                    last_selection_result = self.state_selector.select_with_details(
                        self.frontier,
                        mode=self.config.policy.state_selection_mode,
                    )
                    last_selection_result = self._apply_strategic_selection_override(
                        last_selection_result,
                        strategic_guidance=latest_strategic_guidance,
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
                            planning_action_cluster_history = ActionClusterHistory(cluster_label="revisit")
                            planning_state = self._annotate_state(
                                restore_result.final_state,
                                episode_id=generated_episode_id,
                                recent_gain=0.0,
                                action_cluster_history=planning_action_cluster_history,
                                previous_state=None,
                                inventory_changed=False,
                                affordance_gain=0,
                                novel_object_count=0,
                                movement_only_action=False,
                                materially_new_actions=False,
                            )
                            planning_action_cluster_history.observe_state_nouns(
                                observation=planning_state.observation,
                                inventory_text=planning_state.inventory_text,
                                valid_actions=planning_state.valid_actions,
                                inverse_pairs=self.config.policy.inverse_action_pairs,
                            )
                            last_local_exploration = self.local_explorer.explore_from_state(
                                planning_state,
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
                            latest_reflection_guidance = reflection_result.guidance

                            best_branch = last_local_exploration.best_branch
                            branch_progress_score = last_local_exploration.best_branch_progress_score
                            branch_commit_allowed = last_local_exploration.branch_commit_allowed
                            commit_rejection_reason = last_local_exploration.commit_rejection_reason
                            timeline_commit_allowed, timeline_commit_reason = self._validate_branch_commit_origin(
                                origin_state=current_state,
                                selected_entry=selected_entry,
                            )
                            if branch_commit_allowed and not timeline_commit_allowed:
                                branch_commit_allowed = False
                                commit_rejection_reason = timeline_commit_reason
                                last_local_exploration.branch_commit_allowed = False
                                last_local_exploration.commit_rejection_reason = timeline_commit_reason
                            if not branch_commit_allowed:
                                branch_commit_rejection_count += 1
                                self.logger.info(
                                    "Skipping local branch commit for %s because no branch cleared the progress "
                                    "threshold. progress_score=%.2f rejection=%s",
                                    selected_entry.state_id,
                                    branch_progress_score,
                                    commit_rejection_reason or "none",
                                )
                            self._restore_episode_state(current_state)
                            if (
                                best_branch is not None
                                and best_branch.actions_taken
                                and selected_entry.saved_node is not None
                                and branch_commit_allowed
                            ):
                                pending_branch_actions = self._branch_commit_actions(best_branch)
                                self.logger.info(
                                    "Committing branch %s with actions=%s first_durable_gain_index=%s "
                                    "first_durable_gain_action=%s last_durable_progress_index=%s "
                                    "last_durable_progress_action=%s last_meaningful_progress_index=%s "
                                    "last_meaningful_progress_action=%s",
                                    best_branch.branch_index,
                                    pending_branch_actions,
                                    best_branch.first_durable_gain_action_index,
                                    best_branch.first_durable_gain_action or "none",
                                    best_branch.last_durable_progress_action_index,
                                    best_branch.last_durable_progress_action or "none",
                                    best_branch.last_meaningful_progress_action_index,
                                    best_branch.last_meaningful_progress_action or "none",
                                )
                                loaded_branch_plan_this_step = bool(pending_branch_actions)
                            revisit_made_progress = bool(
                                best_branch is not None and best_branch.branch_progress_score > 0.0
                            )
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
                            supported_try_actions=self._merge_action_lists(
                                latest_reflection_guidance.supported_try_actions,
                                latest_strategic_guidance.try_actions,
                            ),
                            supported_avoid_actions=self._merge_action_lists(
                                latest_reflection_guidance.supported_avoid_actions,
                                latest_strategic_guidance.avoid_actions,
                            ),
                            supported_reflection_objects=self._merge_action_lists(
                                latest_reflection_guidance.salient_objects,
                                latest_strategic_guidance.salient_objects,
                            ),
                            strategic_mode=latest_strategic_guidance.mode,
                            strategic_reason=latest_strategic_guidance.reason,
                            strategic_try_actions=latest_strategic_guidance.try_actions,
                            strategic_avoid_actions=latest_strategic_guidance.avoid_actions,
                            strategic_objects=latest_strategic_guidance.salient_objects,
                        )
                        action_generation_result = self.action_generator.last_result
                        if action_generation_result is not None:
                            action_generation_mode_counts[action_generation_result.mode.value] += 1
                            self.logger.info(
                                "Action ranking before=%s final=%s source=%s top_reason=%s features=%s",
                                action_generation_result.candidate_pool_before_rerank,
                                [candidate.action for candidate in action_generation_result.candidates],
                                action_generation_result.ranking_source or action_generation_result.mode.value,
                                action_generation_result.top_selection_reason or "none",
                                to_jsonable(action_generation_result.feature_records()),
                            )
                        chosen_action = proposals[0].action if proposals else "look"

                action_source_counts[action_source] += 1
                pre_step_state = current_state
                transition = self.env.step(chosen_action)
                total_reward += transition.reward
                raw_state = transition.to_state()
                persistent_affordance_gain = self._count_persistent_affordances(
                    base_state=pre_step_state,
                    final_state=raw_state,
                )
                persistent_exit_gain_count = self._count_new_exit_actions(
                    base_state=pre_step_state,
                    final_state=raw_state,
                )
                affordance_gain = persistent_affordance_gain + persistent_exit_gain_count
                movement_only_action = is_movement_action(
                    chosen_action,
                    self.config.policy.inverse_action_pairs,
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
                inventory_gained = self._count_inventory_gain(
                    base_state=pre_step_state,
                    final_state=raw_state,
                ) > 0
                inventory_lost = self._count_inventory_loss(
                    base_state=pre_step_state,
                    final_state=raw_state,
                ) > 0
                materially_new_actions = valid_actions_materially_different(
                    pre_step_state.valid_actions,
                    raw_state.valid_actions,
                )
                revealed_new_object = bool(novel_object_tokens) and (
                    not movement_only_action or affordance_gain > 0 or materially_new_actions
                )
                next_state = self._annotate_state(
                    raw_state,
                    episode_id=generated_episode_id,
                    recent_gain=float(transition.score - pre_step_state.score),
                    action_cluster_history=action_cluster_history,
                    previous_state=pre_step_state,
                    inventory_changed=inventory_changed,
                    affordance_gain=affordance_gain,
                    novel_object_count=len(novel_object_tokens) if revealed_new_object else 0,
                    movement_only_action=movement_only_action,
                    materially_new_actions=materially_new_actions,
                )
                episode_map_memory.record_transition(
                    previous_state=pre_step_state,
                    action=chosen_action,
                    current_state=next_state,
                    step_index=episode_step_index,
                    score_gain=next_state.score - pre_step_state.score,
                    inventory_gain_count=self._count_inventory_gain(
                        base_state=pre_step_state,
                        final_state=next_state,
                    ),
                    affordance_gain=affordance_gain,
                    novel_object_count=len(novel_object_tokens) if revealed_new_object else 0,
                    materially_new_actions=materially_new_actions,
                )
                self._annotate_strategic_state(
                    next_state,
                    episode_map_memory=episode_map_memory,
                    step_index=episode_step_index + 1,
                )
                action_cluster_history.record_attempt(
                    action=chosen_action,
                    score_changed=next_state.score != pre_step_state.score,
                    inventory_changed=inventory_changed,
                    inventory_gained=inventory_gained,
                    inventory_lost=inventory_lost,
                    observation_changed=self._normalize_text(next_state.observation)
                    != self._normalize_text(pre_step_state.observation),
                    valid_actions_changed={
                        self._normalize_text(candidate) for candidate in next_state.valid_actions
                    }
                    != {
                        self._normalize_text(candidate) for candidate in pre_step_state.valid_actions
                    },
                    valid_actions_improved=affordance_gain > 0,
                    revealed_new_object=revealed_new_object,
                    target_tokens=action_target_tokens(
                        chosen_action,
                        self.config.policy.inverse_action_pairs,
                    ),
                    movement_only_action=movement_only_action,
                    inverse_pairs=self.config.policy.inverse_action_pairs,
                    bulk_inventory_action=is_bulk_inventory_action(chosen_action),
                    discard_like_action=is_discard_like_action(chosen_action),
                    revealed_object_tokens=novel_object_tokens if revealed_new_object else set(),
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
                        "persistent_affordance_gain": persistent_affordance_gain,
                        "persistent_exit_gain_count": persistent_exit_gain_count,
                        "inventory_gained": inventory_gained,
                        "inventory_lost": inventory_lost,
                        "revealed_new_object": revealed_new_object,
                        "durable_progress": False,
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
                        "candidate_pool_before_rerank": (
                            list(action_generation_result.candidate_pool_before_rerank)
                            if action_source == "action_generator" and action_generation_result is not None
                            else []
                        ),
                        "candidate_feature_vectors": (
                            to_jsonable(action_generation_result.feature_records())
                            if action_source == "action_generator" and action_generation_result is not None
                            else []
                        ),
                        "final_ranked_candidates": (
                            [candidate.action for candidate in action_generation_result.candidates]
                            if action_source == "action_generator" and action_generation_result is not None
                            else []
                        ),
                        "candidate_ranking_source": (
                            action_generation_result.ranking_source
                            if action_source == "action_generator" and action_generation_result is not None
                            else ""
                        ),
                        "top_selection_reason": (
                            action_generation_result.top_selection_reason
                            if action_source == "action_generator" and action_generation_result is not None
                            else ""
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

                durable_progress = self._step_has_durable_progress(
                    score_gain=next_state.score - pre_step_state.score,
                    inventory_gained=inventory_gained,
                    affordance_gain=affordance_gain,
                    novel_object_count=len(novel_object_tokens) if revealed_new_object else 0,
                    movement_only_action=movement_only_action,
                )
                step.metadata["durable_progress"] = durable_progress

                score_gain = next_state.score - pre_step_state.score
                if durable_progress and score_gain > 0:
                    local_exploration_cooldown_steps = max(
                        local_exploration_cooldown_steps,
                        self.config.experiment.local_exploration_post_gain_cooldown_steps,
                    )
                elif local_exploration_cooldown_steps > 0:
                    local_exploration_cooldown_steps -= 1

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
                            "persistent_affordance_gain": persistent_affordance_gain,
                            "persistent_exit_gain_count": persistent_exit_gain_count,
                            "inventory_gained": inventory_gained,
                            "inventory_lost": inventory_lost,
                            "revealed_new_object": bool(novel_object_tokens),
                        },
                    )

                episode_action_history.append(chosen_action)
                episode_state_history.append(next_state)
                episode_loop_results.append(loop_result)
                if pending_branch_actions and action_source.startswith("best_local_branch"):
                    exhausted_families = action_cluster_history.exhausted_families(
                        self.config.policy.object_family_no_progress_threshold
                    )
                    if (
                        exhausted_families
                        and next_state.score <= pre_step_state.score
                        and not inventory_gained
                        and affordance_gain <= 0
                        and movement_result.total_penalty + loop_result.total_penalty > 0.0
                    ):
                        self.logger.info(
                            "Discarding remaining local branch plan after exhausted-family no-progress step. "
                            "action=%s exhausted_families=%s",
                            chosen_action,
                            sorted(exhausted_families),
                        )
                        pending_branch_actions = []
                action_cluster_history.observe_state_nouns(
                    observation=next_state.observation,
                    inventory_text=next_state.inventory_text,
                    valid_actions=next_state.valid_actions,
                    inverse_pairs=self.config.policy.inverse_action_pairs,
                )
                current_state = next_state

                (
                    consecutive_no_durable_gain_steps,
                    consecutive_same_cluster_movement_steps,
                    recent_stalled_movement_clusters,
                    episode_fail_fast_reason,
                ) = self._update_episode_fail_fast_state(
                    consecutive_no_durable_gain_steps=consecutive_no_durable_gain_steps,
                    consecutive_same_cluster_movement_steps=consecutive_same_cluster_movement_steps,
                    recent_stalled_movement_clusters=recent_stalled_movement_clusters,
                    durable_progress=durable_progress,
                    movement_only_action=movement_result.movement_only_action,
                    current_cluster_id=current_state.state_cluster_id,
                )
                if episode_fail_fast_reason:
                    self.logger.info(
                        "Fail-fast ending episode %s at step %s: %s",
                        generated_episode_id,
                        episode_step_index,
                        episode_fail_fast_reason,
                    )
                    break
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
            "episode_mode_counts": dict(episode_mode_counts),
            "loop_detected_count": loop_detected_count,
            "oscillation_penalty_total": oscillation_penalty_total,
            "movement_loop_detected_count": movement_loop_detected_count,
            "movement_penalty_total": movement_penalty_total,
            "branch_commit_rejection_count": branch_commit_rejection_count,
            "consecutive_no_durable_gain_steps": consecutive_no_durable_gain_steps,
            "consecutive_same_cluster_movement_steps": consecutive_same_cluster_movement_steps,
            "episode_fail_fast_reason": episode_fail_fast_reason,
            "final_guidance": reflection_context,
            "final_reflection_mode": final_reflection.mode.value,
            "reflection_removed_items": list(final_reflection.guidance.unsupported_items_removed),
            "reflection_supported_try_actions": list(final_reflection.guidance.supported_try_actions),
            "reflection_supported_avoid_actions": list(final_reflection.guidance.supported_avoid_actions),
            "state_selection_mode": self.config.policy.state_selection_mode,
            "max_replay_attempts": self.config.experiment.max_replay_attempts,
            "frontier_refresh_cadence": self.config.experiment.frontier_refresh_cadence,
            "local_exploration_cadence": self.config.experiment.local_exploration_cadence,
            "local_exploration_post_gain_cooldown_steps": self.config.experiment.local_exploration_post_gain_cooldown_steps,
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

    def _run_glow_episode(
        self,
        episode_index: int = 0,
        episode_id: str | None = None,
        episode_seed: int | None = None,
    ) -> EpisodeResult:
        """Run one episode using an explicit GLoW-style global/local control loop."""

        resolved_seed = episode_seed if episode_seed is not None else self.config.experiment.seed + episode_index
        set_global_seed(resolved_seed)
        generated_episode_id = episode_id or f"{self.config.experiment.game_id}-episode-{episode_index:03d}"
        self.trajectory_frontier = self._build_trajectory_frontier()

        cycle_count = 0
        total_rollout_steps = 0
        archive_by_state_id: dict[str, ArchivedState] = {}
        loaded_archive_snapshot_path = ""
        loaded_archive_state_count = 0
        loaded_archive_prior_run_state_count = 0
        archive_snapshot_path = ""
        frontier_analysis_result: FrontierAnalysisResult | None = None
        selection_artifact_directories: list[str] = []
        frontier_analysis_artifact_directories: list[str] = []
        mar_artifact_directories: list[str] = []
        local_world_model_snapshot_paths: list[str] = []
        frontier_size_over_time: list[int] = []
        frontier_root_diversity_over_time: list[int] = []
        archive_state_count_over_time: list[int] = []
        max_score_over_time: list[int] = []
        score_milestones: list[dict[str, object]] = []
        frontier_analysis_diagnostics: list[dict[str, object]] = []
        frontier_analysis_count = 0
        frontier_analysis_success_count = 0
        frontier_analysis_fallback_count = 0
        frontier_analysis_ids: list[str] = []
        restore_attempt_count = 0
        restore_success_count = 0
        mar_update_count = 0
        local_world_model_write_count = 0
        local_world_model_read_count = 0
        local_world_model_read_hit_count = 0
        selected_state_decisions: list[GlowSelectionDecisionMetric] = []
        branch_counts_per_root_state: Counter[str] = Counter()
        root_exploration_counts: Counter[str] = Counter()
        local_world_model_root_ids: set[str] = set()
        reused_local_world_model_root_ids: set[str] = set()
        local_world_model_reuse_count = 0
        roots_with_prior_mar_update: set[str] = set()
        revisited_roots_after_mar_update: set[str] = set()
        same_root_revisit_after_mar_update_count = 0
        local_guidance_attached_branch_count = 0
        local_guidance_prompt_attachment_count = 0
        local_guidance_action_generation_count = 0
        local_guidance_action_generation_prompt_count = 0
        local_guidance_selected_action_match_count = 0
        mar_reuse_status_reason_counts: Counter[str] = Counter()
        selected_root_ids_over_time: list[str] = []
        selected_root_observation_summaries: dict[str, str] = {}
        selected_root_details: dict[str, dict[str, object]] = {}
        selected_region_counts: Counter[str] = Counter()
        branch_action_counts: Counter[str] = Counter()
        committed_action_counts: Counter[str] = Counter()
        committed_root_counts: Counter[str] = Counter()
        action_target_token_counts: Counter[str] = Counter()
        winning_branch_signature_counts: Counter[str] = Counter()
        unique_first_actions_per_root: dict[str, set[str]] = {}
        commit_allowed_count = 0
        committed_branch_positive_progress_count = 0
        committed_branch_score_gain_count = 0
        shadow_commit_threshold = 0.5
        shadow_threshold_would_commit_count = 0
        threshold_blocked_exploratory_best_branch_count = 0
        branch_cycle_diagnostics: list[dict[str, object]] = []
        last_score_improvement_cycle = 0
        last_new_root_cycle = 0
        best_trajectory: EpisodeTrajectory | None = None
        initial_archived_state: ArchivedState | None = None

        try:
            (
                archive_by_state_id,
                loaded_archive_snapshot_path,
                loaded_archive_state_count,
                loaded_archive_prior_run_state_count,
            ) = self._load_persisted_archive(generated_episode_id)
            if loaded_archive_state_count > 0:
                self.logger.info(
                    "Loaded %s archived states from %s (%s first seen in prior runs).",
                    loaded_archive_state_count,
                    loaded_archive_snapshot_path,
                    loaded_archive_prior_run_state_count,
                )
            initial_state = self.env.reset(seed=resolved_seed)
            initial_archived_state = self.archive_updater.archived_state_for_live_state(
                state=initial_state,
                episode_id=generated_episode_id,
                provenance_trajectory_id=f"{generated_episode_id}-root",
                provenance_timestep=0,
            )
            archive_by_state_id = self.archive_updater.upsert_archived_state(
                archive_by_state_id,
                initial_archived_state,
            )
            archive_snapshot_path = str(
                self.archive_store.write_states(
                    generated_episode_id,
                    self.archive_updater.sorted_states(archive_by_state_id),
                )
            )

            while total_rollout_steps < self.config.experiment.max_steps:
                cycle_count += 1
                frontier_size_before = len(self.trajectory_frontier)
                frontier_size_over_time.append(frontier_size_before)
                archive_states = self.archive_updater.sorted_states(archive_by_state_id)
                selection_result = self.state_selector.select_archive_state(
                    archive_states=archive_states,
                    frontier_insight=frontier_analysis_result.insight if frontier_analysis_result is not None else None,
                    trajectory_frontier=self.trajectory_frontier,
                    mode=self.config.policy.archive_state_selection_mode,
                    selection_id=f"{generated_episode_id}-selection-{cycle_count:03d}",
                )
                if selection_result.artifact_directory:
                    selection_artifact_directories.append(selection_result.artifact_directory)
                selected_archived_state = selection_result.selected_archived_state
                if selected_archived_state is None:
                    self.logger.info(
                        "GLoW runner ended for %s because no archived state was available for selection.",
                        generated_episode_id,
                    )
                    break

                tracked_selected_state = archive_by_state_id.setdefault(
                    selected_archived_state.state_id,
                    selected_archived_state,
                )
                self.archive_updater.note_selection(tracked_selected_state)
                restore_attempt_count += 1
                restore_result = self._restore_archived_state(selected_archived_state)
                if not restore_result.success:
                    self.logger.warning(
                        "GLoW runner could not restore archived state %s: %s",
                        selected_archived_state.state_id,
                        restore_result.message,
                    )
                    selected_state_decisions.append(
                        GlowSelectionDecisionMetric(
                            cycle_index=cycle_count,
                            selected_state_id=selected_archived_state.state_id,
                            root_state_id=selected_archived_state.state_id,
                            replay_method=selection_result.chosen_replay_method,
                            achieved_contribution=selection_result.achieved_contribution,
                            potential_contribution=selection_result.potential_contribution,
                            rationale=selection_result.rationale,
                            frontier_size_before=frontier_size_before,
                            frontier_size_after=len(self.trajectory_frontier),
                            branch_count=0,
                            restore_succeeded=False,
                            used_llm_adjudication=selection_result.selection_mode.value == "llm_assisted",
                            selected_frontier_trajectory_ids=list(selection_result.selected_frontier_trajectory_ids),
                            selected_critical_state_ids=list(selection_result.selected_critical_state_ids),
                            metadata={"selection_artifact_directory": selection_result.artifact_directory},
                        )
                    )
                    self.archive_updater.note_restore_result(tracked_selected_state, success=False)
                    if cycle_count >= self.config.experiment.max_replay_attempts + 1:
                        break
                    continue
                restore_success_count += 1
                self.archive_updater.note_restore_result(tracked_selected_state, success=True)
                self.logger.info(
                    "Restored archived state %s first_seen=%s via %s.",
                    selected_archived_state.state_id,
                    selected_archived_state.first_seen_episode_id or "unknown",
                    restore_result.restore_mode.value,
                )

                remaining_steps = max(1, self.config.experiment.max_steps - total_rollout_steps)
                effective_branch_count = max(
                    1,
                    min(self.config.experiment.glow_local_branches_per_root, remaining_steps),
                )
                effective_branch_horizon = max(
                    1,
                    min(
                        self.config.policy.rollout_depth,
                        max(1, remaining_steps // effective_branch_count),
                    ),
                )
                best_known_score_before_cycle = best_trajectory.final_score if best_trajectory is not None else 0
                local_result = self.local_explorer.explore_from_state(
                    restore_result.final_state,
                    env=self.env,
                    branch_count=effective_branch_count,
                    branch_horizon=effective_branch_horizon,
                    temperature=self.config.llm.temperature,
                    action_candidate_count=self.config.policy.action_candidates,
                    recent_trajectory_context=selection_result.rationale,
                    best_known_score=best_known_score_before_cycle,
                )
                local_result_metadata = dict(local_result.metadata)
                root_exploration_counts[local_result.root_state_id] += 1
                selected_root_ids_over_time.append(local_result.root_state_id)
                root_summary = self._archived_state_summary(selected_archived_state)
                root_region_label = self._coarse_region_label(
                    summary=root_summary,
                    state_cluster_id=selected_archived_state.state_cluster_id,
                )
                if root_summary:
                    selected_root_observation_summaries.setdefault(local_result.root_state_id, root_summary)
                selected_region_counts[root_region_label] += 1
                selected_root_details.setdefault(
                    local_result.root_state_id,
                    {
                        "state_id": local_result.root_state_id,
                        "summary": root_summary,
                        "region_label": root_region_label,
                        "first_seen_episode_id": selected_archived_state.first_seen_episode_id or "",
                        "provenance_trajectory_id": selected_archived_state.provenance_trajectory_id,
                    },
                )
                selected_root_details[local_result.root_state_id]["selection_count"] = root_exploration_counts[
                    local_result.root_state_id
                ]
                if root_exploration_counts[local_result.root_state_id] == 1:
                    last_new_root_cycle = cycle_count
                if local_result.root_state_id in roots_with_prior_mar_update:
                    same_root_revisit_after_mar_update_count += 1
                    revisited_roots_after_mar_update.add(local_result.root_state_id)
                    self.logger.info(
                        "Revisited root %s after prior MAR update; reuse_status=%s",
                        local_result.root_state_id,
                        local_result_metadata.get("reuse_status_reason", "unknown"),
                    )
                if bool(local_result_metadata.get("local_world_model_read_attempted", False)):
                    local_world_model_read_count += 1
                if bool(local_result_metadata.get("local_world_model_read_hit", False)):
                    local_world_model_read_hit_count += 1
                if local_result.used_local_world_model is not None:
                    local_world_model_reuse_count += 1
                    reused_local_world_model_root_ids.add(local_result.root_state_id)
                local_guidance_attached_branch_count += int(
                    local_result_metadata.get("branches_with_local_guidance_count", 0)
                )
                local_guidance_prompt_attachment_count += int(
                    local_result_metadata.get("branches_with_prompt_guidance_count", 0)
                )
                local_guidance_action_generation_count += int(
                    local_result_metadata.get("action_generation_calls_with_local_guidance_count", 0)
                )
                local_guidance_action_generation_prompt_count += int(
                    local_result_metadata.get("action_generation_calls_with_prompt_guidance_count", 0)
                )
                local_guidance_selected_action_match_count += int(
                    local_result_metadata.get("selected_actions_matching_local_try_guidance_count", 0)
                )
                reuse_status_reason = str(local_result_metadata.get("reuse_status_reason", "")).strip()
                if reuse_status_reason:
                    mar_reuse_status_reason_counts[reuse_status_reason] += 1
                branch_counts_per_root_state[local_result.root_state_id] += len(local_result.branches)
                for branch in local_result.branches:
                    branch_action_counts.update(action for action in branch.actions_taken if action.strip())
                    action_target_token_counts.update(
                        token
                        for action in branch.actions_taken
                        for token in action_target_tokens(
                            action,
                            self.config.policy.inverse_action_pairs,
                        )
                    )
                    if branch.actions_taken:
                        unique_first_actions_per_root.setdefault(local_result.root_state_id, set()).add(
                            branch.actions_taken[0]
                        )
                best_branch = local_result.best_branch
                if local_result.branch_commit_allowed and best_branch is not None:
                    commit_allowed_count += 1
                    committed_root_counts[local_result.root_state_id] += 1
                    committed_action_counts.update(action for action in best_branch.actions_taken if action.strip())
                    winning_branch_signature_counts[" -> ".join(best_branch.actions_taken)] += 1
                    if best_branch.branch_progress_score > 0.0:
                        committed_branch_positive_progress_count += 1
                    if best_branch.score_change > 0:
                        committed_branch_score_gain_count += 1
                threshold_blocker = (
                    best_branch is not None
                    and not local_result.branch_commit_allowed
                    and "threshold" in local_result.commit_rejection_reason.lower()
                )
                plausibly_exploratory_best_branch = (
                    best_branch is not None
                    and best_branch.score_change == 0
                    and best_branch.branch_progress_score > 0.0
                    and not best_branch.appears_stuck
                )
                if (
                    best_branch is not None
                    and not local_result.branch_commit_allowed
                    and best_branch.branch_progress_score > shadow_commit_threshold
                ):
                    shadow_threshold_would_commit_count += 1
                if threshold_blocker and plausibly_exploratory_best_branch:
                    threshold_blocked_exploratory_best_branch_count += 1
                branch_cycle_diagnostics.append(
                    {
                        "cycle_index": cycle_count,
                        "root_state_id": local_result.root_state_id,
                        "root_region_label": root_region_label,
                        "root_summary": root_summary,
                        "selected_archive_state_id": selected_archived_state.state_id,
                        "first_seen_episode_id": selected_archived_state.first_seen_episode_id or "",
                        "branch_commit_allowed": local_result.branch_commit_allowed,
                        "commit_rejection_reason": local_result.commit_rejection_reason,
                        "commit_threshold_blocker": threshold_blocker,
                        "plausibly_exploratory_best_branch": plausibly_exploratory_best_branch,
                        "best_branch_index": local_result.best_branch_index,
                        "best_branch_progress_score": (
                            best_branch.branch_progress_score if best_branch is not None else 0.0
                        ),
                        "best_branch_score_change": best_branch.score_change if best_branch is not None else 0,
                        "best_branch_actions": list(best_branch.actions_taken) if best_branch is not None else [],
                        "best_branch_region_label": (
                            self._coarse_region_label(
                                summary=best_branch.final_observation,
                                state_cluster_id=(
                                    best_branch.final_state.state_cluster_id
                                    if best_branch.final_state is not None
                                    else ""
                                ),
                            )
                            if best_branch is not None
                            else ""
                        ),
                        "branches": [
                            {
                                "branch_index": branch.branch_index,
                                "actions": list(branch.actions_taken),
                                "first_action": branch.actions_taken[0] if branch.actions_taken else "",
                                "progress_score": branch.branch_progress_score,
                                "score_change": branch.score_change,
                                "final_score": branch.final_score,
                                "appears_stuck": branch.appears_stuck,
                                "terminated": branch.terminated,
                                "termination_reason": branch.termination_reason.value,
                                "region_label": self._coarse_region_label(
                                    summary=branch.final_observation,
                                    state_cluster_id=(
                                        branch.final_state.state_cluster_id
                                        if branch.final_state is not None
                                        else ""
                                    ),
                                ),
                                "exploratory_location_progress": bool(
                                    branch.metadata.get("exploratory_location_progress", False)
                                ),
                                "repeated_score_replay": bool(
                                    branch.metadata.get("repeated_score_replay", False)
                                ),
                                "repeated_known_local_affordance": bool(
                                    branch.metadata.get("repeated_known_local_affordance", False)
                                ),
                                "action_events": list(branch.metadata.get("action_events", [])),
                            }
                            for branch in local_result.branches
                        ],
                    }
                )
                revisit_progress = self._classify_revisit_progress(local_result.branches)
                self.archive_updater.note_revisit_progress(
                    tracked_selected_state,
                    achieved_progress=bool(revisit_progress["achieved_progress"]),
                    exploratory_progress=bool(revisit_progress["exploratory_progress"]),
                    branch_commit_allowed=local_result.branch_commit_allowed,
                    best_branch_actions=best_branch.actions_taken if best_branch is not None else [],
                )
                local_world_model_for_decay = (
                    local_result.updated_local_world_model or local_result.used_local_world_model
                )
                if local_world_model_for_decay is not None:
                    local_world_model_for_decay.metadata.update(
                        {
                            "no_achievement_revisit_count": int(
                                tracked_selected_state.metadata.get("no_achievement_revisit_count", 0)
                            ),
                            "no_achievement_revisit_streak": int(
                                tracked_selected_state.metadata.get("no_achievement_revisit_streak", 0)
                            ),
                            "nonproductive_revisit_count": int(
                                tracked_selected_state.metadata.get("nonproductive_revisit_count", 0)
                            ),
                            "nonproductive_revisit_streak": int(
                                tracked_selected_state.metadata.get("nonproductive_revisit_streak", 0)
                            ),
                            "last_revisit_achieved_progress": bool(revisit_progress["achieved_progress"]),
                            "last_revisit_score_progress": bool(revisit_progress["score_progress"]),
                            "last_revisit_fresh_durable_progress": bool(
                                revisit_progress["fresh_durable_progress"]
                            ),
                            "last_revisit_repeated_local_affordance_only_progress": bool(
                                revisit_progress["repeated_local_affordance_only_progress"]
                            ),
                            "last_revisit_exploratory_progress": bool(revisit_progress["exploratory_progress"]),
                            "last_revisit_branch_commit_allowed": local_result.branch_commit_allowed,
                            "last_revisit_best_branch_actions": (
                                list(best_branch.actions_taken) if best_branch is not None else []
                            ),
                        }
                    )
                    self.local_explorer.local_world_model_store.write(local_world_model_for_decay)
                if local_result.mar_inference is not None and local_result.mar_inference.artifact_directory:
                    mar_artifact_directories.append(local_result.mar_inference.artifact_directory)
                    mar_update_count += 1
                if local_result.updated_local_world_model is not None:
                    local_world_model_write_count += 1
                    local_world_model_root_ids.add(local_result.updated_local_world_model.root_state_id)
                    roots_with_prior_mar_update.add(local_result.updated_local_world_model.root_state_id)
                    local_world_model_snapshot_paths.append(
                        str(
                            self.local_explorer.local_world_model_store.model_path(
                                local_result.updated_local_world_model.root_state_id
                            )
                        )
                    )

                branch_trajectories = self._build_glow_branch_trajectories(
                    episode_id=generated_episode_id,
                    cycle_index=cycle_count,
                    selected_archived_state=selected_archived_state,
                    local_result=local_result,
                )
                total_rollout_steps += sum(len(branch.actions_taken) for branch in local_result.branches)
                if not branch_trajectories:
                    self.logger.info(
                        "GLoW runner ended for %s because local exploration produced no replayable branch trajectories.",
                        generated_episode_id,
                    )
                    break

                for trajectory in branch_trajectories:
                    self.trajectory_frontier.insert(trajectory)
                    best_trajectory = self._prefer_higher_value_trajectory(best_trajectory, trajectory)
                archive_by_state_id = self.archive_updater.ingest_episode_trajectories(
                    archive_by_state_id,
                    branch_trajectories,
                )
                archive_by_state_id = self.archive_updater.refresh_achieved_values_from_frontier(
                    archive_by_state_id,
                    self.trajectory_frontier,
                )
                if self._should_run_glow_frontier_analysis(cycle_index=cycle_count):
                    frontier_analysis_result = self.frontier_analyzer.analyze_frontier(
                        self.trajectory_frontier,
                        archive_states=self.archive_updater.sorted_states(archive_by_state_id),
                        top_k=min(
                            self.config.experiment.glow_frontier_size,
                            max(1, len(self.trajectory_frontier)),
                        ),
                        analysis_id=f"{generated_episode_id}-analysis-{cycle_count:03d}",
                    )
                    if frontier_analysis_result.artifact_directory:
                        frontier_analysis_artifact_directories.append(frontier_analysis_result.artifact_directory)
                    frontier_analysis_count += 1
                    if frontier_analysis_result.used_fallback:
                        frontier_analysis_fallback_count += 1
                    else:
                        frontier_analysis_success_count += 1
                    frontier_analysis_ids.append(frontier_analysis_result.analysis_id)
                    frontier_analysis_diagnostics.append(
                        self._frontier_analysis_diagnostic_record(
                            frontier_analysis_result=frontier_analysis_result,
                            archive_by_state_id=archive_by_state_id,
                        )
                    )
                archive_by_state_id = self.archive_updater.apply_projected_potential_from_frontier_analysis(
                    archive_by_state_id,
                    frontier_analysis_result.insight if frontier_analysis_result is not None else None,
                )
                current_cycle_max_score = max(
                    (
                        trajectory.summary_fields.max_score
                        for trajectory in branch_trajectories
                    ),
                    default=max_score_over_time[-1] if max_score_over_time else 0,
                )
                previous_cycle_max_score = max_score_over_time[-1] if max_score_over_time else 0
                if current_cycle_max_score > previous_cycle_max_score:
                    milestone_trajectory = max(
                        branch_trajectories,
                        key=lambda trajectory: trajectory.summary_fields.max_score,
                    )
                    score_milestones.append(
                        {
                            "cycle_index": cycle_count,
                            "score": current_cycle_max_score,
                            "trajectory_id": milestone_trajectory.trajectory_id,
                            "root_state_id": milestone_trajectory.root_state_id,
                        }
                    )
                    last_score_improvement_cycle = cycle_count
                max_score_over_time.append(max(current_cycle_max_score, previous_cycle_max_score))
                archive_state_count_over_time.append(len(archive_by_state_id))
                frontier_root_diversity_over_time.append(self._frontier_root_diversity())
                archive_snapshot_path = str(
                    self.archive_store.write_states(
                        generated_episode_id,
                        self.archive_updater.sorted_states(archive_by_state_id),
                    )
                )

                selected_state_decisions.append(
                    GlowSelectionDecisionMetric(
                        cycle_index=cycle_count,
                        selected_state_id=selected_archived_state.state_id,
                        root_state_id=local_result.root_state_id,
                        replay_method=selection_result.chosen_replay_method,
                        achieved_contribution=selection_result.achieved_contribution,
                        potential_contribution=selection_result.potential_contribution,
                        rationale=selection_result.rationale,
                        frontier_size_before=frontier_size_before,
                        frontier_size_after=len(self.trajectory_frontier),
                        branch_count=len(local_result.branches),
                        restore_succeeded=True,
                        used_llm_adjudication=selection_result.selection_mode.value == "llm_assisted",
                        selected_frontier_trajectory_ids=list(selection_result.selected_frontier_trajectory_ids),
                        selected_critical_state_ids=list(selection_result.selected_critical_state_ids),
                        metadata={
                            "selection_artifact_directory": selection_result.artifact_directory,
                            "mar_artifact_directory": (
                                local_result.mar_inference.artifact_directory
                                if local_result.mar_inference is not None
                                else ""
                            ),
                            "frontier_analysis_id": (
                                frontier_analysis_result.analysis_id
                                if frontier_analysis_result is not None
                                else ""
                            ),
                            "local_world_model_read_hit": bool(
                                local_result_metadata.get("local_world_model_read_hit", False)
                            ),
                            "local_world_model_guidance_available": bool(
                                local_result_metadata.get("local_world_model_guidance_available", False)
                            ),
                            "branches_with_local_guidance_count": int(
                                local_result_metadata.get("branches_with_local_guidance_count", 0)
                            ),
                            "branches_with_prompt_guidance_count": int(
                                local_result_metadata.get("branches_with_prompt_guidance_count", 0)
                            ),
                            "revisit_achieved_progress": bool(revisit_progress["achieved_progress"]),
                            "revisit_score_progress": bool(revisit_progress["score_progress"]),
                            "revisit_fresh_durable_progress": bool(
                                revisit_progress["fresh_durable_progress"]
                            ),
                            "revisit_repeated_local_affordance_only_progress": bool(
                                revisit_progress["repeated_local_affordance_only_progress"]
                            ),
                            "revisit_exploratory_progress": bool(revisit_progress["exploratory_progress"]),
                            "root_no_achievement_revisit_streak": int(
                                tracked_selected_state.metadata.get("no_achievement_revisit_streak", 0)
                            ),
                            "root_nonproductive_revisit_streak": int(
                                tracked_selected_state.metadata.get("nonproductive_revisit_streak", 0)
                            ),
                            "action_generation_calls_with_local_guidance_count": int(
                                local_result_metadata.get(
                                    "action_generation_calls_with_local_guidance_count",
                                    0,
                                )
                            ),
                            "action_generation_calls_with_prompt_guidance_count": int(
                                local_result_metadata.get(
                                    "action_generation_calls_with_prompt_guidance_count",
                                    0,
                                )
                            ),
                            "selected_actions_matching_local_try_guidance_count": int(
                                local_result_metadata.get(
                                    "selected_actions_matching_local_try_guidance_count",
                                    0,
                                )
                            ),
                            "reuse_status_reason": reuse_status_reason,
                            "branch_commit_allowed": local_result.branch_commit_allowed,
                            "commit_rejection_reason": local_result.commit_rejection_reason,
                            "best_branch_progress_score": local_result.best_branch_progress_score,
                            "best_branch_actions": list(best_branch.actions_taken) if best_branch is not None else [],
                        },
                    )
                )

                committed_terminal_branch = (
                    local_result.branch_commit_allowed
                    and best_branch is not None
                    and best_branch.terminated
                )
                if committed_terminal_branch:
                    self.logger.info(
                        "GLoW runner terminated early for %s after a completed branch trajectory.",
                        generated_episode_id,
                    )
                    break

        finally:
            self.env.close()

        if best_trajectory is None:
            best_trajectory = EpisodeTrajectory(
                episode_id=generated_episode_id,
                root_state_id=(
                    initial_archived_state.state_id
                    if initial_archived_state is not None
                    else f"{generated_episode_id}:root"
                ),
                selected_from_archive_state_id=None,
                steps=[],
                max_cumulative_reward_achieved=0.0,
                final_score=0,
                final_done=False,
                replay_metadata=(
                    initial_archived_state.replay_metadata
                    if initial_archived_state is not None
                    else ReplayMetadata()
                ),
                metadata={"runner_mode": RunnerMode.GLOW_FAITHFUL.value, "seed": resolved_seed},
            )

        stored_trajectory = self._trajectory_from_episode_trajectory(
            episode_id=generated_episode_id,
            episode_trajectory=best_trajectory,
        )
        trajectory_path = self.trajectory_store.write_trajectory(stored_trajectory)
        notes = self._glow_notes(frontier_analysis_result)
        llm_call_counts = self.llm_client.stage_call_counts() if self.llm_client is not None else {}
        llm_success_counts = self.llm_client.stage_success_counts() if self.llm_client is not None else {}
        llm_error_counts = self.llm_client.stage_error_counts() if self.llm_client is not None else {}
        unique_selected_root_count = len(root_exploration_counts)
        cycles_since_last_new_root = cycle_count - last_new_root_cycle if cycle_count > 0 else 0
        cycles_since_last_new_score = cycle_count - last_score_improvement_cycle if cycle_count > 0 else 0
        top_selected_region_concentration = 0.0
        if selected_region_counts:
            top_selected_region_concentration = selected_region_counts.most_common(1)[0][1] / max(
                1,
                sum(selected_region_counts.values()),
            )
        top_selected_root_concentration = 0.0
        if root_exploration_counts:
            top_selected_root_concentration = root_exploration_counts.most_common(1)[0][1] / max(
                1,
                sum(root_exploration_counts.values()),
            )
        top_branch_action_concentration = 0.0
        if branch_action_counts:
            top_branch_action_concentration = branch_action_counts.most_common(1)[0][1] / max(
                1,
                sum(branch_action_counts.values()),
            )
        progress_stuckness_summary = {
            "top_selected_root_concentration": top_selected_root_concentration,
            "top_branch_action_concentration": top_branch_action_concentration,
            "cycles_since_last_new_root": cycles_since_last_new_root,
            "cycles_since_last_new_score": cycles_since_last_new_score,
            "score_exceeded_zero": bool(max_score_over_time and max(max_score_over_time) > 0),
        }
        repeated_root_local_world_model_summaries = self._repeated_root_local_world_model_summaries(
            root_exploration_counts=root_exploration_counts,
            selected_root_details=selected_root_details,
        )
        episode_metrics = GlowEpisodeMetrics(
            episode_id=generated_episode_id,
            seed=resolved_seed,
            environment_interactions=total_rollout_steps,
            max_score=best_trajectory.summary_fields.max_score if best_trajectory.steps else 0,
            final_score=best_trajectory.final_score,
            frontier_size_over_time=frontier_size_over_time,
            frontier_analysis_count=frontier_analysis_count,
            frontier_analysis_ids=frontier_analysis_ids,
            mar_update_count=mar_update_count,
            restore_attempt_count=restore_attempt_count,
            restore_success_count=restore_success_count,
            selected_state_decisions=selected_state_decisions,
            branch_counts_per_root_state=dict(branch_counts_per_root_state),
            local_world_model_root_ids=sorted(local_world_model_root_ids),
            frontier_analysis_artifact_directories=frontier_analysis_artifact_directories,
            selection_artifact_directories=selection_artifact_directories,
            mar_artifact_directories=mar_artifact_directories,
            local_world_model_snapshot_paths=local_world_model_snapshot_paths,
            metadata={
                "frontier_retained_trajectory_count": len(self.trajectory_frontier),
                "archive_state_count": len(archive_by_state_id),
                "llm_client_initialized": self.llm_client is not None,
                "llm_client_class": type(self.llm_client).__name__ if self.llm_client is not None else "",
                "llm_default_model": (
                    self.llm_client.default_model
                    if self.llm_client is not None and self.llm_client.default_model is not None
                    else ""
                ),
                "llm_stage_call_counts": llm_call_counts,
                "llm_stage_success_counts": llm_success_counts,
                "llm_stage_error_counts": llm_error_counts,
                "frontier_analysis_success_count": frontier_analysis_success_count,
                "frontier_analysis_fallback_count": frontier_analysis_fallback_count,
                "frontier_analysis_diagnostics": frontier_analysis_diagnostics,
                "loaded_archive_snapshot_path": loaded_archive_snapshot_path,
                "loaded_archive_state_count": loaded_archive_state_count,
                "loaded_archive_prior_run_state_count": loaded_archive_prior_run_state_count,
                "local_world_model_reuse_count": local_world_model_reuse_count,
                "local_world_model_write_count": local_world_model_write_count,
                "local_world_model_read_count": local_world_model_read_count,
                "local_world_model_read_hit_count": local_world_model_read_hit_count,
                "reused_local_world_model_root_ids": sorted(reused_local_world_model_root_ids),
                "root_exploration_counts": dict(root_exploration_counts),
                "same_root_revisit_after_mar_update_count": same_root_revisit_after_mar_update_count,
                "revisited_roots_after_mar_update": sorted(revisited_roots_after_mar_update),
                "local_guidance_attached_branch_count": local_guidance_attached_branch_count,
                "local_guidance_prompt_attachment_count": local_guidance_prompt_attachment_count,
                "local_guidance_action_generation_count": local_guidance_action_generation_count,
                "local_guidance_action_generation_prompt_count": (
                    local_guidance_action_generation_prompt_count
                ),
                "local_guidance_selected_action_match_count": (
                    local_guidance_selected_action_match_count
                ),
                "mar_reuse_status_reason_counts": dict(mar_reuse_status_reason_counts),
                "archive_snapshot_path": archive_snapshot_path,
                "archive_state_count_over_time": archive_state_count_over_time,
                "frontier_root_diversity_over_time": frontier_root_diversity_over_time,
                "max_score_over_time": max_score_over_time,
                "score_milestones": score_milestones,
                "selected_root_ids_over_time": selected_root_ids_over_time,
                "selected_root_observation_summaries": selected_root_observation_summaries,
                "selected_root_details": list(selected_root_details.values()),
                "selected_region_counts": dict(selected_region_counts),
                "top_selected_regions": self._counter_top_entries(
                    selected_region_counts,
                    key_name="region_label",
                ),
                "top_selected_region_concentration": top_selected_region_concentration,
                "unique_selected_root_count": unique_selected_root_count,
                "top_selected_roots": self._counter_top_entries(
                    root_exploration_counts,
                    key_name="state_id",
                    summaries=selected_root_observation_summaries,
                ),
                "top_selected_root_concentration": top_selected_root_concentration,
                "commit_allowed_count": commit_allowed_count,
                "committed_branch_positive_progress_count": committed_branch_positive_progress_count,
                "committed_branch_score_gain_count": committed_branch_score_gain_count,
                "shadow_commit_threshold": shadow_commit_threshold,
                "shadow_threshold_would_commit_count": shadow_threshold_would_commit_count,
                "threshold_blocked_exploratory_best_branch_count": (
                    threshold_blocked_exploratory_best_branch_count
                ),
                "branch_cycle_diagnostics": branch_cycle_diagnostics,
                "unique_branch_action_count": len(branch_action_counts),
                "unique_committed_action_count": len(committed_action_counts),
                "unique_first_actions_per_root": {
                    root_state_id: sorted(actions)
                    for root_state_id, actions in unique_first_actions_per_root.items()
                },
                "top_branch_actions": self._counter_top_entries(branch_action_counts, key_name="action"),
                "top_committed_actions": self._counter_top_entries(committed_action_counts, key_name="action"),
                "top_action_target_tokens": self._counter_top_entries(
                    action_target_token_counts,
                    key_name="token",
                ),
                "top_winning_action_sequences": self._counter_top_entries(
                    winning_branch_signature_counts,
                    key_name="action_sequence",
                ),
                "top_committed_roots": self._counter_top_entries(
                    committed_root_counts,
                    key_name="state_id",
                    summaries=selected_root_observation_summaries,
                ),
                "cycles_since_last_new_root": cycles_since_last_new_root,
                "cycles_since_last_new_score": cycles_since_last_new_score,
                "repeated_root_local_world_model_summaries": repeated_root_local_world_model_summaries,
                "progress_stuckness_summary": progress_stuckness_summary,
                "selected_state_statistics": self._selection_statistics(selected_state_decisions),
            },
        )
        metrics_path = self._write_metrics_json(generated_episode_id, episode_metrics.to_record())
        episode_metadata = {
            "runner_mode": RunnerMode.GLOW_FAITHFUL.value,
            "seed": resolved_seed,
            "glow_cycle_count": cycle_count,
            "glow_total_rollout_steps": total_rollout_steps,
            "glow_frontier_size": len(self.trajectory_frontier),
            "glow_archive_state_count": len(archive_by_state_id),
            "llm_client_initialized": self.llm_client is not None,
            "llm_client_class": type(self.llm_client).__name__ if self.llm_client is not None else "",
            "llm_default_model": (
                self.llm_client.default_model
                if self.llm_client is not None and self.llm_client.default_model is not None
                else ""
            ),
            "llm_stage_call_counts": llm_call_counts,
            "llm_stage_success_counts": llm_success_counts,
            "llm_stage_error_counts": llm_error_counts,
            "frontier_analysis_success_count": frontier_analysis_success_count,
            "frontier_analysis_fallback_count": frontier_analysis_fallback_count,
            "frontier_analysis_diagnostics": frontier_analysis_diagnostics,
            "loaded_archive_snapshot_path": loaded_archive_snapshot_path,
            "loaded_archive_state_count": loaded_archive_state_count,
            "loaded_archive_prior_run_state_count": loaded_archive_prior_run_state_count,
            "local_world_model_reuse_count": local_world_model_reuse_count,
            "local_world_model_write_count": local_world_model_write_count,
            "local_world_model_read_count": local_world_model_read_count,
            "local_world_model_read_hit_count": local_world_model_read_hit_count,
            "reused_local_world_model_root_ids": sorted(reused_local_world_model_root_ids),
            "root_exploration_counts": dict(root_exploration_counts),
            "same_root_revisit_after_mar_update_count": same_root_revisit_after_mar_update_count,
            "revisited_roots_after_mar_update": sorted(revisited_roots_after_mar_update),
            "local_guidance_attached_branch_count": local_guidance_attached_branch_count,
            "local_guidance_prompt_attachment_count": local_guidance_prompt_attachment_count,
            "local_guidance_action_generation_count": local_guidance_action_generation_count,
            "local_guidance_action_generation_prompt_count": local_guidance_action_generation_prompt_count,
            "local_guidance_selected_action_match_count": local_guidance_selected_action_match_count,
            "mar_reuse_status_reason_counts": dict(mar_reuse_status_reason_counts),
            "archive_snapshot_path": archive_snapshot_path,
            "archive_state_count_over_time": archive_state_count_over_time,
            "frontier_root_diversity_over_time": frontier_root_diversity_over_time,
            "max_score_over_time": max_score_over_time,
            "score_milestones": score_milestones,
            "selected_root_ids_over_time": selected_root_ids_over_time,
            "selected_root_observation_summaries": selected_root_observation_summaries,
            "selected_root_details": list(selected_root_details.values()),
            "selected_region_counts": dict(selected_region_counts),
            "top_selected_regions": self._counter_top_entries(
                selected_region_counts,
                key_name="region_label",
            ),
            "top_selected_region_concentration": top_selected_region_concentration,
            "unique_selected_root_count": unique_selected_root_count,
            "top_selected_roots": self._counter_top_entries(
                root_exploration_counts,
                key_name="state_id",
                summaries=selected_root_observation_summaries,
            ),
            "top_selected_root_concentration": top_selected_root_concentration,
            "commit_allowed_count": commit_allowed_count,
            "committed_branch_positive_progress_count": committed_branch_positive_progress_count,
            "committed_branch_score_gain_count": committed_branch_score_gain_count,
            "shadow_commit_threshold": shadow_commit_threshold,
            "shadow_threshold_would_commit_count": shadow_threshold_would_commit_count,
            "threshold_blocked_exploratory_best_branch_count": (
                threshold_blocked_exploratory_best_branch_count
            ),
            "branch_cycle_diagnostics": branch_cycle_diagnostics,
            "unique_branch_action_count": len(branch_action_counts),
            "unique_committed_action_count": len(committed_action_counts),
            "unique_first_actions_per_root": {
                root_state_id: sorted(actions)
                for root_state_id, actions in unique_first_actions_per_root.items()
            },
            "top_branch_actions": self._counter_top_entries(branch_action_counts, key_name="action"),
            "top_committed_actions": self._counter_top_entries(committed_action_counts, key_name="action"),
            "top_action_target_tokens": self._counter_top_entries(
                action_target_token_counts,
                key_name="token",
            ),
            "top_winning_action_sequences": self._counter_top_entries(
                winning_branch_signature_counts,
                key_name="action_sequence",
            ),
            "top_committed_roots": self._counter_top_entries(
                committed_root_counts,
                key_name="state_id",
                summaries=selected_root_observation_summaries,
            ),
            "cycles_since_last_new_root": cycles_since_last_new_root,
            "cycles_since_last_new_score": cycles_since_last_new_score,
            "repeated_root_local_world_model_summaries": repeated_root_local_world_model_summaries,
            "progress_stuckness_summary": progress_stuckness_summary,
            "glow_frontier_analysis_frequency": self.config.experiment.glow_frontier_analysis_frequency,
            "glow_local_branches_per_root": self.config.experiment.glow_local_branches_per_root,
            "archive_selection_mode": self.config.policy.archive_state_selection_mode.value,
            "selection_artifact_directories": selection_artifact_directories,
            "frontier_analysis_artifact_directories": frontier_analysis_artifact_directories,
            "mar_artifact_directories": mar_artifact_directories,
            "local_world_model_snapshot_paths": local_world_model_snapshot_paths,
            "best_frontier_trajectory_id": best_trajectory.trajectory_id,
            "best_frontier_root_state_id": best_trajectory.root_state_id,
            "frontier_analysis_id": frontier_analysis_result.analysis_id if frontier_analysis_result is not None else "",
            "frontier_inferred_bottlenecks": (
                list(frontier_analysis_result.insight.inferred_bottlenecks[:6])
                if frontier_analysis_result is not None
                else []
            ),
            "frontier_partial_solutions": (
                list(frontier_analysis_result.insight.partial_solutions[:6])
                if frontier_analysis_result is not None
                else []
            ),
            "metrics_path": str(metrics_path),
        }
        summary_path = self._write_summary_json(
            generated_episode_id,
            {
                "episode_id": generated_episode_id,
                "seed": resolved_seed,
                "total_reward": best_trajectory.max_cumulative_reward_achieved,
                "step_count": len(best_trajectory.steps),
                "final_score": best_trajectory.final_score,
                "trajectory_path": str(trajectory_path),
                "final_state_summary": (
                    self.summary_builder.summarize_state_candidate(
                        observation=best_trajectory.steps[-1].observation,
                        score=best_trajectory.final_score,
                        depth=best_trajectory.steps[-1].step_index + 1,
                        recent_gain=0.0,
                        inventory_text=best_trajectory.steps[-1].inventory_text,
                        valid_actions=best_trajectory.steps[-1].valid_actions,
                    )
                    if best_trajectory.steps
                    else "No completed branch trajectory."
                ),
                "notes": notes,
                "metadata": episode_metadata,
            },
        )
        return EpisodeResult(
            episode_id=generated_episode_id,
            seed=resolved_seed,
            total_reward=best_trajectory.max_cumulative_reward_achieved,
            step_count=len(best_trajectory.steps),
            final_score=best_trajectory.final_score,
            trajectory_path=trajectory_path,
            summary_path=summary_path,
            metrics_path=metrics_path,
            notes=notes,
            metadata=episode_metadata,
        )

    def _restore_archived_state(self, archived_state: ArchivedState):
        """Restore an archived state using its provenance trajectory when available."""

        saved_node = self._saved_node_from_archived_state(archived_state)
        if saved_node is None:
            return restore_saved_node(self.env, None, logger=self.logger)
        return restore_saved_node(
            self.env,
            saved_node,
            target_step_index=saved_node.step_index,
            logger=self.logger,
        )

    def _load_persisted_archive(
        self,
        episode_id: str,
    ) -> tuple[dict[str, ArchivedState], str, int, int]:
        """Load the newest persisted archive snapshot for cross-run reuse."""

        latest_snapshot = self.archive_store.read_latest_snapshot()
        if latest_snapshot is None:
            return {}, "", 0, 0

        snapshot_path, loaded_states = latest_snapshot
        archive_by_state_id: dict[str, ArchivedState] = {}
        for archived_state in loaded_states:
            archive_by_state_id = self.archive_updater.upsert_archived_state(
                archive_by_state_id,
                archived_state,
            )
        prior_run_state_count = sum(
            1
            for archived_state in archive_by_state_id.values()
            if archived_state.first_seen_episode_id != episode_id
        )
        return (
            archive_by_state_id,
            str(snapshot_path),
            len(archive_by_state_id),
            prior_run_state_count,
        )

    def _saved_node_from_archived_state(self, archived_state: ArchivedState) -> SavedNode | None:
        """Build a replayable saved node from an archived state plus provenance data."""

        metadata = archived_state.metadata
        observation = archived_state.observation_summary or str(metadata.get("observation", "")).strip()
        if not observation:
            return None
        return SavedNode(
            state_id=archived_state.state_id,
            episode_id=archived_state.provenance_trajectory_id,
            step_index=archived_state.provenance_timestep,
            native_state=archived_state.native_snapshot or archived_state.replay_metadata.native_snapshot,
            action_prefix=list(archived_state.replay_metadata.replay_actions),
            score=archived_state.score_at_state or int(metadata.get("score", 0)),
            observation=observation,
            inventory_text=archived_state.inventory_summary or str(metadata.get("inventory_text", "")),
            world_state_hash=archived_state.replay_metadata.world_state_hash or str(metadata.get("world_state_hash", "unknown")),
            valid_actions=(
                [item.strip() for item in archived_state.valid_action_summary.split(",") if item.strip()]
                or [str(item) for item in (metadata.get("valid_actions") or [])]
            ),
            summary_text=self.summary_builder.summarize_state_candidate(
                observation=observation,
                score=archived_state.score_at_state or int(metadata.get("score", 0)),
                depth=archived_state.provenance_timestep + 1,
                recent_gain=0.0,
                inventory_text=archived_state.inventory_summary or str(metadata.get("inventory_text", "")),
                valid_actions=(
                    [item.strip() for item in archived_state.valid_action_summary.split(",") if item.strip()]
                    or [str(item) for item in (metadata.get("valid_actions") or [])]
                ),
            ),
            metadata=dict(metadata),
        )

    def _build_glow_branch_trajectories(
        self,
        *,
        episode_id: str,
        cycle_index: int,
        selected_archived_state: ArchivedState,
        local_result: LocalExplorationResult,
    ) -> list[EpisodeTrajectory]:
        """Convert local branch rollouts into complete trajectories for the global frontier."""

        prefix_steps = self._prefix_steps_for_archived_state(selected_archived_state)
        branch_trajectories: list[EpisodeTrajectory] = []
        for branch in local_result.branches:
            if not branch.trajectory_steps:
                continue
            trajectory_id = f"{episode_id}-cycle-{cycle_index:03d}-branch-{branch.branch_index:02d}"
            combined_steps = self._combined_branch_steps(
                trajectory_id=trajectory_id,
                prefix_steps=prefix_steps,
                branch_steps=branch.trajectory_steps,
            )
            legacy_trajectory = Trajectory(
                episode_id=trajectory_id,
                steps=combined_steps,
                metadata={
                    "runner_mode": RunnerMode.GLOW_FAITHFUL.value,
                    "source_episode_id": episode_id,
                    "cycle_index": cycle_index,
                    "root_state_id": local_result.root_state_id,
                    "selected_from_archive_state_id": selected_archived_state.state_id,
                    "branch_index": branch.branch_index,
                    "branch_progress_score": branch.branch_progress_score,
                    "branch_commit_allowed": branch.branch_commit_allowed,
                    "commit_rejection_reason": branch.commit_rejection_reason,
                },
            )
            branch_trajectories.append(
                self.trajectory_store.episode_trajectory_for_trajectory(
                    legacy_trajectory,
                    root_state_id=local_result.root_state_id,
                    selected_from_archive_state_id=selected_archived_state.state_id,
                    replay_metadata=selected_archived_state.replay_metadata,
                    metadata=legacy_trajectory.metadata,
                )
            )
        return branch_trajectories

    def _prefix_steps_for_archived_state(self, archived_state: ArchivedState) -> list[TrajectoryStep]:
        """Return a retained provenance prefix for an archived state when available.

        The archive is now independent from the bounded trajectory frontier, so an
        archived state may outlive the complete provenance trajectory that first
        produced it. In that case we keep restore/replay exact through
        `archived_state.replay_metadata`, but the frontier-facing branch trajectory
        starts at the restored root rather than reconstructing a synthetic prefix.
        """

        provenance_trajectory = self.trajectory_frontier.get_trajectory(archived_state.provenance_trajectory_id)
        if provenance_trajectory is None:
            return []
        return [
            step
            for step in provenance_trajectory.steps
            if step.step_index <= archived_state.provenance_timestep
        ]

    def _combined_branch_steps(
        self,
        *,
        trajectory_id: str,
        prefix_steps: list[TrajectoryStep],
        branch_steps: list[TrajectoryStep],
    ) -> list[TrajectoryStep]:
        """Combine a retained provenance prefix with a newly explored local branch."""

        combined: list[TrajectoryStep] = []
        for step in [*prefix_steps, *branch_steps]:
            record = step.to_record()
            record["episode_id"] = trajectory_id
            record["step_index"] = len(combined)
            combined.append(TrajectoryStep.from_record(record))
        return combined

    def _should_run_glow_frontier_analysis(self, *, cycle_index: int) -> bool:
        """Return whether the global frontier analysis should run this cycle."""

        if not self.config.experiment.enable_global_frontier_analysis:
            return False
        frequency = max(1, self.config.experiment.glow_frontier_analysis_frequency)
        return cycle_index % frequency == 0 and len(self.trajectory_frontier) > 0

    def _prefer_higher_value_trajectory(
        self,
        current_best: EpisodeTrajectory | None,
        candidate: EpisodeTrajectory,
    ) -> EpisodeTrajectory:
        """Keep the highest-value trajectory discovered so far."""

        if current_best is None:
            return candidate
        if candidate.max_cumulative_reward_achieved > current_best.max_cumulative_reward_achieved:
            return candidate
        if candidate.max_cumulative_reward_achieved < current_best.max_cumulative_reward_achieved:
            return current_best
        if candidate.final_score > current_best.final_score:
            return candidate
        return current_best

    def _trajectory_from_episode_trajectory(
        self,
        *,
        episode_id: str,
        episode_trajectory: EpisodeTrajectory,
    ) -> Trajectory:
        """Project a typed episode trajectory back into the legacy JSONL artifact format."""

        copied_steps: list[TrajectoryStep] = []
        for index, step in enumerate(episode_trajectory.steps):
            record = step.to_record()
            record["episode_id"] = episode_id
            record["step_index"] = index
            copied_steps.append(TrajectoryStep.from_record(record))
        return Trajectory(
            episode_id=episode_id,
            steps=copied_steps,
            metadata={
                "runner_mode": RunnerMode.GLOW_FAITHFUL.value,
                "root_state_id": episode_trajectory.root_state_id,
                "selected_from_archive_state_id": episode_trajectory.selected_from_archive_state_id,
                "trajectory_id": episode_trajectory.trajectory_id,
                **dict(episode_trajectory.metadata),
            },
        )

    def _glow_notes(self, analysis_result: FrontierAnalysisResult | None) -> str:
        """Build a compact notes string for the paper-faithful runner summary."""

        if analysis_result is None:
            return "No frontier analysis was available."
        insight = analysis_result.insight
        fragments = [
            *insight.partial_solutions[:2],
            *insight.inferred_bottlenecks[:2],
            *insight.missing_prerequisites[:1],
        ]
        return " | ".join(fragment for fragment in fragments if fragment.strip()) or "Frontier analysis completed."

    def _branch_commit_actions(self, branch: LocalBranchOutcome) -> list[str]:
        """Return the branch prefix that should be committed into the main episode.

        The baseline behavior used a fixed prefix length, which can truncate the
        actual gain event from an otherwise strong branch. We now commit through
        the last meaningful progress action, while trimming any churn after that.
        """

        if not branch.actions_taken:
            return []

        commit_length = min(len(branch.actions_taken), self.config.experiment.branch_commit_steps)
        if branch.last_durable_progress_action_index is not None:
            commit_length = max(commit_length, branch.last_durable_progress_action_index + 1)
            commit_length = min(commit_length, branch.last_durable_progress_action_index + 1)
        elif branch.last_meaningful_progress_action_index is not None:
            commit_length = max(commit_length, branch.last_meaningful_progress_action_index + 1)
            commit_length = min(commit_length, branch.last_meaningful_progress_action_index + 1)
        elif branch.first_durable_gain_action_index is not None:
            commit_length = max(commit_length, branch.first_durable_gain_action_index + 1)
            commit_length = min(commit_length, branch.first_durable_gain_action_index + 1)
        return list(branch.actions_taken[:commit_length])

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
                strategic_score_weight=self.config.policy.frontier_strategic_score_weight,
            ),
        )

    def _build_trajectory_frontier(self) -> TrajectoryFrontier:
        """Construct the paper-faithful complete-trajectory frontier."""

        return TrajectoryFrontier(
            TrajectoryFrontierConfig(
                max_size=self.config.experiment.glow_frontier_size,
            )
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

    def _annotate_strategic_state(
        self,
        state: TextGameState,
        *,
        episode_map_memory: EpisodeMapMemory,
        step_index: int,
    ) -> TextGameState:
        """Attach region/opportunity memory signals to one state."""

        episode_map_memory.record_state(state, step_index=step_index)
        state.strategic_cluster_score = episode_map_memory.cluster_strategic_score(state.state_cluster_id)
        state.unresolved_opportunity_count = len(
            episode_map_memory.opportunities_for_cluster(
                state.state_cluster_id,
                min_priority=self.config.policy.min_strategic_opportunity_priority,
            )
        )
        state.metadata["strategic_cluster_score"] = state.strategic_cluster_score
        state.metadata["unresolved_opportunity_count"] = state.unresolved_opportunity_count
        state.metadata["strategic_objects"] = episode_map_memory.suggested_objects_for_cluster(
            state.state_cluster_id,
            limit=6,
            min_priority=self.config.policy.min_strategic_opportunity_priority,
        )
        state.metadata["strategic_actions"] = episode_map_memory.suggested_actions_for_cluster(
            state.state_cluster_id,
            limit=6,
            min_priority=self.config.policy.min_strategic_opportunity_priority,
        )
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
                "strategic_value": state.strategic_cluster_score,
                "unresolved_opportunity_count": state.unresolved_opportunity_count,
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
                strategic_value=state.strategic_cluster_score,
                unresolved_opportunity_count=state.unresolved_opportunity_count,
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

    def _should_run_local_exploration(
        self,
        episode_step_index: int,
        replay_attempt_count: int,
        local_exploration_cooldown_steps: int,
        *,
        strategic_guidance: StrategicGuidance,
        current_cluster_id: str,
    ) -> bool:
        """Return whether the loop should revisit a saved node before the next real step."""

        cadence = max(1, self.config.experiment.local_exploration_cadence)
        if self.config.experiment.max_replay_attempts <= replay_attempt_count:
            return False
        if episode_step_index <= 0:
            return False
        if local_exploration_cooldown_steps > 0:
            return False
        if strategic_guidance.mode is StrategicMode.EXPLORE and strategic_guidance.try_actions:
            return False
        if (
            strategic_guidance.mode is StrategicMode.EXPLOIT
            and strategic_guidance.preferred_cluster_id
            and self._normalize_text(strategic_guidance.preferred_cluster_id)
            != self._normalize_text(current_cluster_id)
            and len(self.frontier) > 0
        ):
            return True
        return episode_step_index % cadence == 0 and len(self.frontier) > 0

    def _apply_strategic_selection_override(
        self,
        selection_result: StateSelectionResult,
        *,
        strategic_guidance: StrategicGuidance,
    ) -> StateSelectionResult:
        """Prefer a frontier entry from the strategically preferred cluster when available."""

        preferred_cluster_id = self._normalize_text(strategic_guidance.preferred_cluster_id)
        if (
            not preferred_cluster_id
            or strategic_guidance.mode is not StrategicMode.EXPLOIT
            or selection_result.selected is None
        ):
            return selection_result

        for entry in self.frontier.top_k(8):
            entry_cluster_id = self._normalize_text(entry.state_cluster_id or entry.world_state_hash)
            if entry_cluster_id != preferred_cluster_id:
                continue
            if selection_result.selected.state_id == entry.state_id:
                return selection_result
            return StateSelectionResult(
                selected=entry,
                reason=(
                    f"Strategic override selected {entry.state_id} to follow preferred cluster "
                    f"{preferred_cluster_id}. Baseline selector reason: {selection_result.reason}"
                ),
                selection_mode=selection_result.selection_mode,
                candidate_summaries=selection_result.candidate_summaries,
                raw_output=selection_result.raw_output,
                prompt_snapshot=selection_result.prompt_snapshot,
                model_name=selection_result.model_name,
                fallback_reason=selection_result.fallback_reason,
            )
        return selection_result

    def _apply_stall_recovery_guidance_override(
        self,
        *,
        strategic_guidance: StrategicGuidance,
        current_state: TextGameState,
        consecutive_no_durable_gain_steps: int,
    ) -> StrategicGuidance:
        """Force a temporary explore bias when a local cluster has gone stale.

        This is deliberately simple: after several no-progress steps in the same live
        episode cluster, prefer exits over more local object churn even if the broader
        strategic layer still thinks the region is exploitable.
        """

        recovery_threshold = max(4, self.config.experiment.fail_fast_no_durable_gain_steps // 2)
        if consecutive_no_durable_gain_steps < recovery_threshold:
            return strategic_guidance

        exit_actions = [
            action
            for action in current_state.valid_actions
            if is_movement_action(action, self.config.policy.inverse_action_pairs)
        ]
        if not exit_actions:
            return strategic_guidance

        avoid_actions = [
            action
            for action in current_state.valid_actions
            if not is_movement_action(action, self.config.policy.inverse_action_pairs)
        ][:6]
        return StrategicGuidance(
            mode=StrategicMode.EXPLORE,
            reason=(
                f"Stall recovery after {consecutive_no_durable_gain_steps} no-progress steps; "
                "prefer exits over local object churn."
            ),
            preferred_cluster_id=current_state.state_cluster_id,
            try_actions=exit_actions[:4],
            avoid_actions=avoid_actions,
            salient_objects=list(strategic_guidance.salient_objects),
            opportunity_labels=list(strategic_guidance.opportunity_labels),
        )

    def _validate_branch_commit_origin(
        self,
        *,
        origin_state: TextGameState,
        selected_entry: FrontierEntry,
    ) -> tuple[bool, str]:
        """Return whether a replay-planned branch may be committed into the real episode.

        Replay restores are used as a planning tool. To keep the episode trajectory
        comparable to a normal playthrough, we only commit a replay-derived branch when
        it is local to the current episode state rather than a teleport back to a
        different cluster or a lower-value inventory/score basin.
        """

        origin_cluster = self._normalize_text(origin_state.state_cluster_id or origin_state.world_state_hash)
        target_cluster = self._normalize_text(selected_entry.state_cluster_id or selected_entry.world_state_hash)
        if origin_cluster and target_cluster and origin_cluster != target_cluster:
            return (
                False,
                (
                    "planning-only off-cluster restore: local branch came from "
                    f"{selected_entry.state_cluster_id or selected_entry.world_state_hash}, "
                    f"current cluster is {origin_state.state_cluster_id or origin_state.world_state_hash}"
                ),
            )

        if selected_entry.score < float(origin_state.score):
            return (
                False,
                (
                    "planning-only lower-score restore: selected node score "
                    f"{selected_entry.score:.2f} is below current score {origin_state.score:.2f}"
                ),
            )

        selected_inventory = self._normalize_text(selected_entry.inventory_text)
        current_inventory = self._normalize_text(origin_state.inventory_text)
        if (
            selected_inventory
            and current_inventory
            and selected_inventory != current_inventory
            and selected_entry.score <= float(origin_state.score)
        ):
            return (
                False,
                "planning-only inventory mismatch: replay node would discard current durable inventory context",
            )

        return True, ""

    def _restore_episode_state(self, state: TextGameState) -> None:
        """Restore the live environment to the current committed episode state after planning."""

        snapshot = state.world_state_snapshot
        if snapshot is None:
            return
        restored_state = self.env.restore_snapshot(snapshot)
        self.logger.debug(
            "Restored committed episode state after planning: cluster=%s score=%s inventory=%s",
            state.state_cluster_id or restored_state.state_cluster_id,
            restored_state.score,
            restored_state.inventory_text,
        )

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

    def _merge_action_lists(self, *lists: list[str]) -> list[str]:
        """Merge action/object guidance lists while preserving order."""

        merged: list[str] = []
        seen: set[str] = set()
        for values in lists:
            for value in values:
                normalized = self._normalize_text(value)
                if not normalized or normalized in seen:
                    continue
                merged.append(value)
                seen.add(normalized)
        return merged

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

    def _count_persistent_affordances(self, *, base_state: TextGameState, final_state: TextGameState) -> int:
        """Count end-state affordances that expose genuinely new targets or interactions."""

        base_targets = {
            token
            for action in base_state.valid_actions
            if not is_movement_action(action, self.config.policy.inverse_action_pairs)
            for token in action_target_tokens(action, self.config.policy.inverse_action_pairs)
        }
        discovered_actions = 0
        for action in final_state.valid_actions:
            if is_movement_action(action, self.config.policy.inverse_action_pairs):
                continue
            targets = action_target_tokens(action, self.config.policy.inverse_action_pairs)
            if targets - base_targets:
                discovered_actions += 1
        return min(discovered_actions, 4)

    def _count_new_exit_actions(self, *, base_state: TextGameState, final_state: TextGameState) -> int:
        """Count newly available movement exits at the end state."""

        base_exits = {
            self._normalize_text(action)
            for action in base_state.valid_actions
            if is_movement_action(action, self.config.policy.inverse_action_pairs)
        }
        final_exits = {
            self._normalize_text(action)
            for action in final_state.valid_actions
            if is_movement_action(action, self.config.policy.inverse_action_pairs)
        }
        return max(0, len(final_exits - base_exits))

    def _count_inventory_gain(self, *, base_state: TextGameState, final_state: TextGameState) -> int:
        """Count durable inventory items gained by the end state."""

        base_items = inventory_item_tokens(
            base_state.inventory_text,
            inverse_pairs=self.config.policy.inverse_action_pairs,
        )
        final_items = inventory_item_tokens(
            final_state.inventory_text,
            inverse_pairs=self.config.policy.inverse_action_pairs,
        )
        return max(0, len(final_items - base_items))

    def _count_inventory_loss(self, *, base_state: TextGameState, final_state: TextGameState) -> int:
        """Count inventory items lost by the end state."""

        base_items = inventory_item_tokens(
            base_state.inventory_text,
            inverse_pairs=self.config.policy.inverse_action_pairs,
        )
        final_items = inventory_item_tokens(
            final_state.inventory_text,
            inverse_pairs=self.config.policy.inverse_action_pairs,
        )
        return max(0, len(base_items - final_items))

    def _step_has_durable_progress(
        self,
        *,
        score_gain: int,
        inventory_gained: bool,
        affordance_gain: int,
        novel_object_count: int,
        movement_only_action: bool,
    ) -> bool:
        """Return whether the current real episode step made durable progress."""

        return any(
            (
                score_gain > 0,
                inventory_gained,
                affordance_gain > 0,
                novel_object_count > 0 and not movement_only_action,
            )
        )

    def _update_episode_fail_fast_state(
        self,
        *,
        consecutive_no_durable_gain_steps: int,
        consecutive_same_cluster_movement_steps: int,
        recent_stalled_movement_clusters: list[str],
        durable_progress: bool,
        movement_only_action: bool,
        current_cluster_id: str,
    ) -> tuple[int, int, list[str], str]:
        """Update episode-level stall counters and return an optional fail-fast reason."""

        if durable_progress:
            return 0, 0, [], ""

        no_durable_gain_steps = consecutive_no_durable_gain_steps + 1
        movement_steps = 0
        stalled_clusters: list[str] = []
        if movement_only_action:
            movement_steps = consecutive_same_cluster_movement_steps + 1
            stalled_clusters = [*recent_stalled_movement_clusters, current_cluster_id][-32:]

        if no_durable_gain_steps >= self.config.experiment.fail_fast_no_durable_gain_steps:
            return (
                no_durable_gain_steps,
                movement_steps,
                stalled_clusters,
                (
                    f"no durable progress for {no_durable_gain_steps} consecutive steps "
                    f"(threshold {self.config.experiment.fail_fast_no_durable_gain_steps})"
                ),
            )

        if (
            movement_steps >= self.config.experiment.fail_fast_same_cluster_movement_steps
            and stalled_clusters
            and len(set(stalled_clusters[-movement_steps:])) <= 2
        ):
            return (
                no_durable_gain_steps,
                movement_steps,
                stalled_clusters,
                (
                    f"movement wandering persisted for {movement_steps} steps across "
                    f"{len(set(stalled_clusters[-movement_steps:]))} cluster(s)"
                ),
            )

        return no_durable_gain_steps, movement_steps, stalled_clusters, ""

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

    def _write_metrics_json(self, episode_id: str, payload: dict[str, object]) -> Path:
        """Persist a structured per-episode metrics artifact."""

        metrics_path = self.config.paths.metrics_dir / "episodes" / f"{episode_id}.metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(to_jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return metrics_path

    def _selection_statistics(
        self,
        selection_decisions: list[GlowSelectionDecisionMetric],
    ) -> dict[str, dict[str, float | int]]:
        """Aggregate archive-selection behavior for episode-level metrics."""

        stats: dict[str, dict[str, float | int]] = {}
        for decision in selection_decisions:
            if decision.selected_state_id is None:
                continue
            record = stats.setdefault(
                decision.selected_state_id,
                {
                    "selection_count": 0,
                    "restore_success_count": 0,
                    "achieved_contribution_total": 0.0,
                    "potential_contribution_total": 0.0,
                    "branch_count_total": 0,
                },
            )
            record["selection_count"] += 1
            record["restore_success_count"] += int(decision.restore_succeeded)
            record["achieved_contribution_total"] += float(decision.achieved_contribution)
            record["potential_contribution_total"] += float(decision.potential_contribution)
            record["branch_count_total"] += int(decision.branch_count)
        return stats

    @staticmethod
    def _classify_revisit_progress(branches: list[LocalBranchOutcome]) -> dict[str, bool]:
        """Classify whether one revisit produced fresh gains or stale local replay only."""

        score_progress = any(branch.score_change > 0 for branch in branches)
        fresh_durable_progress = any(
            branch.durable_progress and not bool(branch.metadata.get("repeated_known_local_affordance", False))
            for branch in branches
        )
        repeated_local_affordance_only_progress = (
            not score_progress
            and not fresh_durable_progress
            and any(
                branch.durable_progress and bool(branch.metadata.get("repeated_known_local_affordance", False))
                for branch in branches
            )
        )
        exploratory_progress = any(
            bool(branch.metadata.get("exploratory_location_progress", False))
            for branch in branches
        )
        return {
            "achieved_progress": score_progress or fresh_durable_progress,
            "score_progress": score_progress,
            "fresh_durable_progress": fresh_durable_progress,
            "repeated_local_affordance_only_progress": repeated_local_affordance_only_progress,
            "exploratory_progress": exploratory_progress,
        }

    def _frontier_root_diversity(self) -> int:
        """Return the number of unique root states currently retained in the frontier."""

        return len(
            {
                trajectory.root_state_id
                for trajectory in self.trajectory_frontier.trajectories_for_analysis()
                if trajectory.root_state_id.strip()
            }
        )

    def _archived_state_summary(self, archived_state: ArchivedState) -> str:
        """Return one compact human-readable summary for a selected archived state."""

        summary = archived_state.observation_summary.strip()
        if summary:
            return summary
        observation = str(archived_state.metadata.get("observation", "")).strip()
        return observation

    def _coarse_region_label(self, *, summary: str, state_cluster_id: str) -> str:
        """Infer one compact semantic region label for diagnostics and stall summaries."""

        cluster_text = state_cluster_id.strip().lower()
        summary_text = summary.strip().lower()
        combined = f"{cluster_text} {summary_text}"

        if "behind house" in combined or "open window" in combined or "region:window" in cluster_text:
            return "behind-house-window"
        if "window" in combined:
            return "window"
        if "mailbox" in combined or "west of house" in combined:
            return "west-house-mailbox"
        if "north of house" in combined or "south of house" in combined or "east of house" in combined:
            return "house-perimeter"
        if "forest path" in combined or "region:title:forest-path" in cluster_text:
            return "forest-path"
        if "up a tree" in combined or "region:title:up-a-tree" in cluster_text:
            return "tree"
        if "canary" in combined:
            return "canary"
        if "egg" in combined or "region:egg" in cluster_text:
            return "egg"
        if "nest" in combined:
            return "nest"
        if "leaves" in combined:
            return "leaves"
        if "forest" in combined:
            return "forest"
        if cluster_text.startswith("region:"):
            return cluster_text.split("region:", 1)[1]
        return "other"

    def _frontier_analysis_diagnostic_record(
        self,
        *,
        frontier_analysis_result: FrontierAnalysisResult,
        archive_by_state_id: dict[str, ArchivedState],
    ) -> dict[str, object]:
        """Build one compact per-analysis diagnostic record for bottleneck inspection."""

        output_text = " ".join(
            [
                frontier_analysis_result.raw_completion,
                *frontier_analysis_result.insight.inferred_bottlenecks,
                *frontier_analysis_result.insight.partial_solutions,
                *frontier_analysis_result.insight.missing_prerequisites,
                *[
                    annotation.textual_rationale
                    for annotation in frontier_analysis_result.insight.candidate_critical_states
                ],
            ]
        ).lower()
        tracked_terms = [
            "house",
            "window",
            "behind house",
            "open window",
            "forest",
            "tree",
            "egg",
            "canary",
            "nest",
            "leaves",
            "mailbox",
        ]
        mentioned_terms = [term for term in tracked_terms if term in output_text]
        critical_state_regions: list[str] = []
        for annotation in frontier_analysis_result.insight.candidate_critical_states:
            archived_state = archive_by_state_id.get(annotation.critical_state_id)
            if archived_state is not None:
                critical_state_regions.append(
                    self._coarse_region_label(
                        summary=self._archived_state_summary(archived_state),
                        state_cluster_id=archived_state.state_cluster_id,
                    )
                )
            else:
                critical_state_regions.append("unknown")
        non_egg_regions = {
            region
            for region in critical_state_regions
            if region not in {"egg", "nest", "canary", "tree", "unknown"}
        }
        return {
            "analysis_id": frontier_analysis_result.analysis_id,
            "used_fallback": frontier_analysis_result.used_fallback,
            "parse_error": frontier_analysis_result.parse_error,
            "mentioned_terms": mentioned_terms,
            "critical_state_count": len(frontier_analysis_result.insight.candidate_critical_states),
            "critical_state_regions": critical_state_regions,
            "identified_non_egg_critical_state": bool(non_egg_regions),
            "non_egg_critical_regions": sorted(non_egg_regions),
        }

    def _repeated_root_local_world_model_summaries(
        self,
        *,
        root_exploration_counts: Counter[str],
        selected_root_details: dict[str, dict[str, object]],
    ) -> list[dict[str, object]]:
        """Summarize persisted local-world-model contents for repeatedly selected roots."""

        summaries: list[dict[str, object]] = []
        for root_state_id, selection_count in root_exploration_counts.items():
            if selection_count < 2:
                continue
            local_world_model = self.local_explorer.local_world_model_store.read(root_state_id)
            if local_world_model is None:
                continue
            root_detail = selected_root_details.get(root_state_id, {})
            summaries.append(
                {
                    "root_state_id": root_state_id,
                    "selection_count": selection_count,
                    "region_label": str(root_detail.get("region_label", "")),
                    "summary": str(root_detail.get("summary", "")),
                    "hint_count": len(local_world_model.accumulated_advantage_hints),
                    "try_actions": [
                        bias.action
                        for bias in sorted(
                            local_world_model.action_priors,
                            key=lambda item: (-item.weight, item.action),
                        )[:5]
                    ],
                    "avoid_actions": [
                        bias.action
                        for bias in sorted(
                            local_world_model.action_antipriors,
                            key=lambda item: (-item.weight, item.action),
                        )[:5]
                    ],
                    "object_hints": [
                        affordance.object_text
                        for affordance in local_world_model.inferred_affordances[:5]
                    ],
                }
            )
        return summaries

    def _counter_top_entries(
        self,
        counts: Counter[str],
        *,
        key_name: str,
        summaries: dict[str, str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, object]]:
        """Serialize the top entries of a counter for JSON metrics payloads."""

        entries: list[dict[str, object]] = []
        for item, count in counts.most_common(limit):
            entry: dict[str, object] = {key_name: item, "count": int(count)}
            if summaries is not None:
                summary = summaries.get(item, "").strip()
                if summary:
                    entry["summary"] = summary
            entries.append(entry)
        return entries
