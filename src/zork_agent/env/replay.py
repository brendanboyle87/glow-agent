"""Restoration helpers for saved trajectories and Jericho-first frontier nodes.

The primary restore path is Jericho native `get_state()` / `set_state()` snapshots.
Reset-plus-replay remains available as a conservative fallback when native restore
is unavailable or validation fails.

TODO: add branch-level replay metrics once local rollouts start branching heavily.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from zork_agent.env.jericho_env import JerichoEnv
from zork_agent.types import (
    ReplayDivergenceReason,
    ReplayResult,
    RestoreMode,
    SavedNode,
    TextGameState,
    TrajectoryStep,
)
from zork_agent.utils.serialization import read_jsonl

_LOGGER = logging.getLogger("zork_agent.replay")


@dataclass(slots=True)
class ReplaySession:
    """A loaded trajectory that can be inspected or restored."""

    steps: list[TrajectoryStep]
    path: Path | None = None

    @classmethod
    def from_path(cls, path: Path) -> "ReplaySession":
        """Load a replay session from a JSONL file."""

        steps = [TrajectoryStep.from_record(record) for record in read_jsonl(path)]
        _validate_replay_steps(steps, source=str(path))
        return cls(steps=steps, path=path)

    @classmethod
    def from_steps(cls, steps: list[TrajectoryStep], path: Path | None = None) -> "ReplaySession":
        """Build a replay session directly from in-memory trajectory steps."""

        copied_steps = list(steps)
        _validate_replay_steps(copied_steps, source=str(path) if path is not None else "<in-memory>")
        return cls(steps=copied_steps, path=path)

    @property
    def episode_id(self) -> str:
        """Return the episode id from the first step if available."""

        return self.steps[0].episode_id if self.steps else "unknown"

    def total_reward(self) -> float:
        """Return the total reward over the replay."""

        return sum(step.reward for step in self.steps)

    def final_score(self) -> int:
        """Return the final score from the last step."""

        return self.steps[-1].score if self.steps else 0

    def step_count(self) -> int:
        """Return the number of stored steps."""

        return len(self.steps)

    def steps_up_to(self, step_index: int | None = None) -> list[TrajectoryStep]:
        """Return stored steps up to and including the chosen trajectory node."""

        if step_index is None:
            return list(self.steps)
        if step_index < 0:
            raise IndexError(f"Replay step index out of range: {step_index}")
        selected_steps = [step for step in self.steps if step.step_index <= step_index]
        if not selected_steps or selected_steps[-1].step_index != step_index:
            raise IndexError(f"Replay step index out of range: {step_index}")
        return selected_steps

    def actions_up_to(self, step_index: int | None = None) -> list[str]:
        """Return actions up to and including the chosen trajectory node."""

        return [step.action for step in self.steps_up_to(step_index)]

    def saved_node_at(self, step_index: int | None = None) -> SavedNode | None:
        """Build a saved-node restore target for the chosen trajectory node."""

        target_steps = self.steps_up_to(step_index)
        if not target_steps:
            return None
        step = target_steps[-1]
        snapshot = step.world_state_snapshot or {}
        native_state = snapshot.get("native_state")
        if isinstance(native_state, list):
            native_state = tuple(native_state)
        elif not isinstance(native_state, tuple):
            native_state = None
        action_prefix = (
            list(snapshot.get("replay_actions", []))
            if isinstance(snapshot.get("replay_actions"), list)
            else [candidate.action for candidate in target_steps]
        )
        return SavedNode(
            state_id=f"{step.episode_id}:{step.step_index}:{step.world_state_hash}",
            episode_id=step.episode_id,
            step_index=step.step_index,
            native_state=native_state,
            action_prefix=action_prefix,
            score=step.score,
            observation=step.observation,
            inventory_text=step.inventory_text,
            world_state_hash=step.world_state_hash,
            valid_actions=list(step.valid_actions),
            summary_text=str(step.metadata.get("summary_text", "")),
            metadata=dict(step.metadata),
        )

    def restore(
        self,
        env: JerichoEnv,
        *,
        step_index: int | None = None,
        saved_node: SavedNode | None = None,
        logger: logging.Logger | None = None,
    ) -> ReplayResult:
        """Restore a trajectory node using native snapshots first and replay second."""

        target_steps = self.steps_up_to(step_index)
        target_step_index = target_steps[-1].step_index if target_steps else step_index
        node = saved_node or self.saved_node_at(step_index)
        return restore_saved_node(
            env,
            node,
            expected_steps=target_steps,
            target_step_index=target_step_index,
            logger=logger,
        )

    def replay(
        self,
        env: JerichoEnv,
        *,
        step_index: int | None = None,
        logger: logging.Logger | None = None,
    ) -> ReplayResult:
        """Backward-compatible alias for restore-based replay."""

        return self.restore(env, step_index=step_index, logger=logger)

    def restore_to_step(
        self,
        env: JerichoEnv,
        step_index: int,
        *,
        logger: logging.Logger | None = None,
    ) -> ReplayResult:
        """Convenience wrapper to restore up to one specific trajectory node."""

        return self.restore(env, step_index=step_index, logger=logger)

    def replay_to_step(
        self,
        env: JerichoEnv,
        step_index: int,
        *,
        logger: logging.Logger | None = None,
    ) -> ReplayResult:
        """Backward-compatible alias for restore-to-step."""

        return self.restore_to_step(env, step_index=step_index, logger=logger)

    def format_summary(self) -> str:
        """Render a short text summary for CLI inspection."""

        if not self.steps:
            label = str(self.path) if self.path is not None else "<in-memory>"
            return f"{label}: empty trajectory"

        preview = ", ".join(step.action for step in self.steps[:3])
        return (
            f"episode_id={self.episode_id} "
            f"steps={len(self.steps)} "
            f"total_reward={self.total_reward():.2f} "
            f"final_score={self.final_score()} "
            f"preview_actions=[{preview}]"
        )


def _validate_replay_steps(steps: list[TrajectoryStep], *, source: str) -> None:
    """Validate replay steps so restore logic can rely on stable ordering."""

    if not steps:
        return

    episode_ids = {step.episode_id for step in steps}
    if len(episode_ids) > 1:
        raise ValueError(f"Replay session {source} contains multiple episode ids: {sorted(episode_ids)}")

    step_indices = [step.step_index for step in steps]
    if step_indices != sorted(step_indices):
        raise ValueError(f"Replay session {source} must be sorted by step_index.")
    if len(step_indices) != len(set(step_indices)):
        raise ValueError(f"Replay session {source} contains duplicate step_index values.")


def restore_saved_node(
    env: JerichoEnv,
    saved_node: SavedNode | None,
    *,
    expected_steps: list[TrajectoryStep] | None = None,
    target_step_index: int | None = None,
    logger: logging.Logger | None = None,
) -> ReplayResult:
    """Restore a saved node using native snapshots first, replay fallback second."""

    active_logger = logger or _LOGGER
    if saved_node is None:
        state = env.reset()
        return ReplayResult(
            success=True,
            restore_mode=RestoreMode.RESET_ONLY,
            divergence_reason=ReplayDivergenceReason.NONE,
            final_observation=state.observation,
            final_score=state.score,
            replayed_action_count=0,
            target_step_index=target_step_index,
            final_state=state,
            restored_node_id=None,
            message="No saved node was provided; returning the reset state.",
        )

    native_restore_attempted = False
    if saved_node.native_state is not None:
        native_restore_attempted = True
        native_set = env.set_native_state(
            saved_node.native_state,
            action_prefix=saved_node.action_prefix,
            step_index=saved_node.step_index,
        )
        if native_set:
            validation = env.validate_restored_state(
                expected_observation=saved_node.observation,
                expected_score=saved_node.score,
                expected_inventory_text=saved_node.inventory_text,
            )
            if validation.is_valid:
                state = env.capture_state(expected_observation=saved_node.observation)
                return ReplayResult(
                    success=True,
                    restore_mode=RestoreMode.NATIVE_SNAPSHOT,
                    divergence_reason=ReplayDivergenceReason.NONE,
                    final_observation=state.observation,
                    final_score=state.score,
                    replayed_action_count=0,
                    target_step_index=target_step_index if target_step_index is not None else saved_node.step_index,
                    final_state=state,
                    validation=validation,
                    native_restore_attempted=True,
                    expected_observation=saved_node.observation,
                    expected_score=saved_node.score,
                    restored_node_id=saved_node.state_id,
                    message="Native snapshot restore succeeded.",
                )
            active_logger.warning(
                "Native snapshot restore failed validation for %s; falling back to action replay. %s",
                saved_node.state_id,
                validation.message,
            )
        else:
            active_logger.warning(
                "Native snapshot restore could not call set_state() for %s; falling back to action replay.",
                saved_node.state_id,
            )

    return _restore_by_action_replay(
        env,
        saved_node,
        expected_steps=expected_steps,
        target_step_index=target_step_index,
        logger=active_logger,
        native_restore_attempted=native_restore_attempted,
    )


def _restore_by_action_replay(
    env: JerichoEnv,
    saved_node: SavedNode,
    *,
    expected_steps: list[TrajectoryStep] | None,
    target_step_index: int | None,
    logger: logging.Logger,
    native_restore_attempted: bool,
) -> ReplayResult:
    """Restore a saved node by reset-plus-replay and validate conservatively."""

    state = env.reset()
    if not saved_node.action_prefix:
        validation = env.validate_restored_state(
            expected_observation=saved_node.observation,
            expected_score=saved_node.score,
            expected_inventory_text=saved_node.inventory_text,
        )
        final_state = env.capture_state(expected_observation=saved_node.observation)
        return ReplayResult(
            success=validation.is_valid,
            restore_mode=RestoreMode.RESET_ONLY if validation.is_valid else RestoreMode.FAILED,
            divergence_reason=validation.divergence_reason,
            final_observation=final_state.observation,
            final_score=final_state.score,
            replayed_action_count=0,
            target_step_index=target_step_index if target_step_index is not None else saved_node.step_index,
            final_state=final_state,
            validation=validation,
            native_restore_attempted=native_restore_attempted,
            expected_observation=saved_node.observation,
            expected_score=saved_node.score,
            restored_node_id=saved_node.state_id,
            message=validation.message,
        )

    replayed_action_count = 0
    final_state = state
    if expected_steps:
        for expected_step in expected_steps:
            try:
                transition = env.step(expected_step.action)
            except Exception as exc:
                message = (
                    f"Replay fallback failed at step {expected_step.step_index} while executing "
                    f"action `{expected_step.action}`: {exc}"
                )
                logger.warning(message)
                return ReplayResult(
                    success=False,
                    restore_mode=RestoreMode.FAILED,
                    divergence_reason=ReplayDivergenceReason.STEP_EXCEPTION,
                    final_observation=final_state.observation,
                    final_score=final_state.score,
                    replayed_action_count=replayed_action_count,
                    target_step_index=target_step_index if target_step_index is not None else saved_node.step_index,
                    final_state=final_state,
                    native_restore_attempted=native_restore_attempted,
                    expected_observation=expected_step.observation,
                    expected_score=expected_step.score,
                    diverged_at_step_index=expected_step.step_index,
                    restored_node_id=saved_node.state_id,
                    message=message,
                )

            replayed_action_count += 1
            final_state = transition.to_state()
            validation = env.validate_restored_state(
                expected_observation=expected_step.observation,
                expected_score=expected_step.score,
                expected_inventory_text=expected_step.inventory_text,
            )
            if not validation.is_valid:
                logger.warning(
                    "Replay fallback diverged at step %s for %s. %s",
                    expected_step.step_index,
                    saved_node.state_id,
                    validation.message,
                )
                return ReplayResult(
                    success=False,
                    restore_mode=RestoreMode.FAILED,
                    divergence_reason=validation.divergence_reason,
                    final_observation=validation.actual_observation,
                    final_score=validation.actual_score,
                    replayed_action_count=replayed_action_count,
                    target_step_index=target_step_index if target_step_index is not None else saved_node.step_index,
                    final_state=env.capture_state(expected_observation=expected_step.observation),
                    validation=validation,
                    native_restore_attempted=native_restore_attempted,
                    expected_observation=expected_step.observation,
                    expected_score=expected_step.score,
                    diverged_at_step_index=expected_step.step_index,
                    restored_node_id=saved_node.state_id,
                    message=validation.message,
                )
    else:
        for action in saved_node.action_prefix:
            try:
                final_state = env.step(action).to_state()
            except Exception as exc:
                message = f"Replay fallback failed while executing action `{action}`: {exc}"
                logger.warning(message)
                return ReplayResult(
                    success=False,
                    restore_mode=RestoreMode.FAILED,
                    divergence_reason=ReplayDivergenceReason.STEP_EXCEPTION,
                    final_observation=final_state.observation,
                    final_score=final_state.score,
                    replayed_action_count=replayed_action_count,
                    target_step_index=target_step_index if target_step_index is not None else saved_node.step_index,
                    final_state=final_state,
                    native_restore_attempted=native_restore_attempted,
                    expected_observation=saved_node.observation,
                    expected_score=saved_node.score,
                    restored_node_id=saved_node.state_id,
                    message=message,
                )
            replayed_action_count += 1

    validation = env.validate_restored_state(
        expected_observation=saved_node.observation,
        expected_score=saved_node.score,
        expected_inventory_text=saved_node.inventory_text,
    )
    final_state = env.capture_state(expected_observation=saved_node.observation)
    restore_mode = RestoreMode.ACTION_REPLAY_FALLBACK if validation.is_valid else RestoreMode.FAILED
    return ReplayResult(
        success=validation.is_valid,
        restore_mode=restore_mode,
        divergence_reason=validation.divergence_reason,
        final_observation=final_state.observation,
        final_score=final_state.score,
        replayed_action_count=replayed_action_count,
        target_step_index=target_step_index if target_step_index is not None else saved_node.step_index,
        final_state=final_state,
        validation=validation,
        native_restore_attempted=native_restore_attempted,
        expected_observation=saved_node.observation,
        expected_score=saved_node.score,
        restored_node_id=saved_node.state_id,
        message=(
            "Replay fallback restore succeeded."
            if validation.is_valid
            else validation.message
        ),
    )
