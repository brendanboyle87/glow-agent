"""Jericho environment wrapper with native snapshot support and a stub fallback.

This module is intentionally Jericho-first. Native `FrotzEnv.get_state()` and
`FrotzEnv.set_state()` snapshots are treated as the primary restoration mechanism.
Reset-plus-replay remains available as a robustness fallback when native restore
is unavailable or validation fails.

TODO: replace the stub transition logic with real branching and state snapshots.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import logging
from typing import Any

from zork_agent.config import ProjectConfig
from zork_agent.types import (
    ReplayDivergenceReason,
    RestoreValidation,
    TextGameState,
    TextGameTransition,
    WorldStateSnapshot,
)

try:
    from jericho import FrotzEnv as NativeFrotzEnv  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised indirectly through stub mode
    NativeFrotzEnv = None

_LOGGER = logging.getLogger("zork_agent.env.jericho")


class JerichoEnv:
    """Small Jericho wrapper with typed outputs and native snapshot support.

    Jericho-specific internals:
    - `get_state()` / `set_state()` are first-class APIs here and the preferred
      restore path for revisit/branching.
    - `get_valid_actions()` is optional and may be expensive or unsupported.
    - `get_inventory()` returns Jericho object instances, which this wrapper renders
      into a readable string.
    - Jericho does not expose a clean "current observation" getter after `set_state()`
      in the public docs, so restored observation validation uses the best available
      live value and otherwise falls back to the cached saved-node observation.
    """

    def __init__(
        self,
        config: ProjectConfig,
        env_factory: Callable[..., Any] | None = None,
    ):
        # TODO: split live and stub implementations if either gains complexity.
        self.config = config
        self._env_factory = env_factory or NativeFrotzEnv
        self._env: Any | None = None
        self._live_mode = False
        self._step_index = 0
        self._action_history: list[str] = []
        self._last_state: TextGameState | None = None

    @property
    def live_backend_available(self) -> bool:
        """Return whether a live Jericho backend looks usable."""

        return self._env_factory is not None and self.config.runtime.jericho_game_path.exists()

    def reset(self, seed: int | None = None) -> TextGameState:
        """Reset the environment and return the initial state."""

        self.close()
        self._step_index = 0
        self._action_history = []
        self._last_state = None
        if self.config.runtime.use_live_jericho and self.live_backend_available:
            self._live_mode = True
            self._env = self._env_factory(str(self.config.runtime.jericho_game_path), seed=seed)
            observation, info = self._env.reset()
            state = self._build_state(observation=observation, done=False, info=info, mode="live")
            self._last_state = state
            return state

        self._live_mode = False
        state = TextGameState(
            observation=(
                "Stub reset: you are standing in an open field west of a white house, "
                "with a boarded front door."
            ),
            inventory_text="You are empty-handed.",
            valid_actions=self._stub_valid_actions(),
            score=0,
            moves=0,
            done=False,
            world_state_hash="stub-west-of-house",
            metadata={"mode": "stub", "seed": seed, "runtime": "action_replay_fallback"},
        )
        state.world_state_snapshot = self.get_world_state_snapshot(state=state)
        self._last_state = state
        return state

    def step(self, action: str) -> TextGameTransition:
        """Apply an action and return a normalized transition result."""

        if self._live_mode and self._env is not None:
            step_index = self._step_index
            observation, reward, done, info = self._env.step(action)
            self._action_history.append(action)
            self._step_index += 1
            state = self._build_state(observation=observation, done=done, info=info, mode="live")
            self._last_state = state
            return TextGameTransition(
                action=action,
                step_index=step_index,
                observation=state.observation,
                reward=float(reward),
                done=state.done,
                score=state.score,
                moves=state.moves,
                inventory_text=state.inventory_text,
                valid_actions=list(state.valid_actions),
                world_state_hash=state.world_state_hash,
                world_state_snapshot=state.world_state_snapshot,
                metadata=dict(state.metadata),
            )

        return self._step_stub(action)

    def get_native_state(self) -> tuple[Any, ...] | None:
        """Return Jericho's native emulator snapshot when available.

        This is Jericho-specific and is the preferred restore mechanism for frontier
        nodes because it avoids expensive and drift-prone action replay.
        """

        if not (self._live_mode and self._env is not None and hasattr(self._env, "get_state")):
            return None
        try:
            raw_state = self._env.get_state()
        except Exception:
            return None
        if isinstance(raw_state, tuple):
            return raw_state
        if isinstance(raw_state, list):
            return tuple(raw_state)
        return (raw_state,)

    def set_native_state(
        self,
        state: tuple[Any, ...] | list[Any] | Any,
        *,
        action_prefix: list[str] | None = None,
        step_index: int | None = None,
    ) -> bool:
        """Restore a Jericho native snapshot directly via `set_state()`.

        The optional `action_prefix` and `step_index` keep the wrapper's own replay
        metadata aligned with the restored branch.
        """

        if not (self._live_mode and self._env is not None and hasattr(self._env, "set_state")):
            return False
        try:
            payload = tuple(state) if isinstance(state, list) else state
            self._env.set_state(payload)
        except Exception:
            return False
        self._action_history = list(action_prefix or [])
        self._step_index = (step_index + 1) if step_index is not None else len(self._action_history)
        return True

    def validate_restored_state(
        self,
        *,
        expected_observation: str,
        expected_score: int,
        expected_inventory_text: str | None = None,
        observation_exact_match: bool = False,
    ) -> RestoreValidation:
        """Validate a restored state conservatively using score, observation, and inventory."""

        actual_state = self._capture_current_state(expected_observation=expected_observation)
        score_matches = actual_state.score == expected_score
        observation_matches = self._texts_match(
            expected_observation,
            actual_state.observation,
            exact=observation_exact_match,
        )

        inventory_matches: bool | None = None
        if self._inventory_comparable(expected_inventory_text, actual_state.inventory_text):
            inventory_matches = self._texts_match(
                expected_inventory_text or "",
                actual_state.inventory_text,
                exact=False,
            )

        if not score_matches:
            reason = ReplayDivergenceReason.SCORE_MISMATCH
        elif not observation_matches:
            reason = ReplayDivergenceReason.OBSERVATION_MISMATCH
        elif inventory_matches is False:
            reason = ReplayDivergenceReason.INVENTORY_MISMATCH
        else:
            reason = ReplayDivergenceReason.NONE

        is_valid = reason is ReplayDivergenceReason.NONE
        message = (
            "Restore validation succeeded."
            if is_valid
            else (
                f"Restore validation failed: reason={reason.value} "
                f"expected_score={expected_score} actual_score={actual_state.score} "
                f"expected_observation={expected_observation!r} actual_observation={actual_state.observation!r}"
            )
        )
        return RestoreValidation(
            is_valid=is_valid,
            divergence_reason=reason,
            score_matches=score_matches,
            observation_matches=observation_matches,
            inventory_matches=inventory_matches,
            actual_observation=actual_state.observation,
            actual_score=actual_state.score,
            actual_inventory_text=actual_state.inventory_text,
            expected_observation=expected_observation,
            expected_score=expected_score,
            expected_inventory_text=expected_inventory_text,
            message=message,
        )

    def capture_state(self, *, expected_observation: str | None = None) -> TextGameState:
        """Capture the current state after native restoration or replay."""

        return self._capture_current_state(expected_observation=expected_observation)

    def get_valid_actions(self) -> list[str]:
        """Return Jericho's valid actions when available, else an empty list."""

        if self._live_mode and self._env is not None and hasattr(self._env, "get_valid_actions"):
            try:
                actions = self._env.get_valid_actions()
                if isinstance(actions, str):
                    return [actions]
                if actions is None:
                    return []
                deduped: list[str] = []
                seen: set[str] = set()
                for action in actions:
                    normalized = str(action)
                    if normalized not in seen:
                        deduped.append(normalized)
                        seen.add(normalized)
                return deduped
            except Exception:
                return []

        if self._last_state is not None:
            return list(self._last_state.valid_actions)
        return self._stub_valid_actions() if not self._live_mode else []

    def get_score(self, info: Mapping[str, Any] | None = None) -> int:
        """Return the current score with a fallback to Jericho info dicts."""

        if self._live_mode and self._env is not None and hasattr(self._env, "get_score"):
            try:
                return int(self._env.get_score())
            except Exception:
                pass
        if info and info.get("score") is not None:
            return int(info["score"])
        return self._last_state.score if self._last_state is not None else 0

    def get_inventory_text(self) -> str:
        """Return a readable inventory string when Jericho exposes inventory objects."""

        if self._live_mode and self._env is not None:
            if hasattr(self._env, "get_inventory"):
                try:
                    items = self._env.get_inventory()
                    if not items:
                        return "Inventory empty."
                    return ", ".join(self._inventory_item_name(item) for item in items)
                except Exception:
                    return "Inventory unavailable."
            return "Inventory unavailable."

        if self._last_state is not None:
            return self._last_state.inventory_text
        return "You are empty-handed."

    def get_world_state_snapshot(self, state: TextGameState | None = None) -> WorldStateSnapshot:
        """Return a native Jericho snapshot when supported, else a replay fallback snapshot."""

        current_state = state or self._last_state or TextGameState(observation="", inventory_text="")
        native_state = self.get_native_state()
        restore_strategy = "native_snapshot" if native_state is not None else "action_replay_fallback"
        return WorldStateSnapshot(
            step_index=self._step_index,
            observation=current_state.observation,
            done=current_state.done,
            score=current_state.score,
            moves=current_state.moves,
            inventory_text=current_state.inventory_text,
            valid_actions=list(current_state.valid_actions),
            world_state_hash=current_state.world_state_hash,
            restore_strategy=restore_strategy,
            replay_actions=list(self._action_history),
            native_state=native_state,
        )

    def restore_snapshot(self, snapshot: WorldStateSnapshot) -> TextGameState:
        """Restore a snapshot using native Jericho state when possible, else replay actions."""

        if snapshot.native_state is not None and self.set_native_state(
            snapshot.native_state,
            action_prefix=snapshot.replay_actions,
            step_index=snapshot.step_index,
        ):
            validation = self.validate_restored_state(
                expected_observation=snapshot.observation,
                expected_score=snapshot.score,
                expected_inventory_text=snapshot.inventory_text,
            )
            if validation.is_valid:
                state = self._capture_current_state(expected_observation=snapshot.observation)
                self._last_state = state
                return state
            _LOGGER.warning("Native snapshot restore failed validation; falling back to action replay.")

        replay_state = self.reset()
        for action in snapshot.replay_actions:
            replay_state = self.step(action).to_state()
        return replay_state

    def close(self) -> None:
        """Close the underlying environment if needed."""

        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
        self._env = None
        self._live_mode = False

    def _build_state(self, observation: str, done: bool, info: Mapping[str, Any], mode: str) -> TextGameState:
        """Translate a live or replayed observation into the shared state type."""

        score = self.get_score(info)
        moves = self._get_moves(info)
        inventory_text = self.get_inventory_text()
        valid_actions = self.get_valid_actions()
        world_state_hash = self._get_world_state_hash(observation=observation, inventory_text=inventory_text)
        state = TextGameState(
            observation=observation,
            inventory_text=inventory_text,
            valid_actions=valid_actions,
            score=score,
            moves=moves,
            done=done,
            world_state_hash=world_state_hash,
            metadata={
                "mode": mode,
                "runtime": "jericho" if mode == "live" else "action_replay_fallback",
            },
        )
        state.world_state_snapshot = self.get_world_state_snapshot(state=state)
        return state

    def _capture_current_state(self, *, expected_observation: str | None = None) -> TextGameState:
        """Capture the best current state view available after native restore."""

        if not self._live_mode:
            return self._last_state or self.reset()

        observation = self._read_current_observation() or expected_observation or (
            self._last_state.observation if self._last_state is not None else ""
        )
        state = TextGameState(
            observation=observation,
            inventory_text=self.get_inventory_text(),
            valid_actions=self.get_valid_actions(),
            score=self.get_score(),
            moves=self._get_moves(info=None),
            done=self._last_state.done if self._last_state is not None else False,
            world_state_hash=self._get_world_state_hash(
                observation=observation,
                inventory_text=self.get_inventory_text(),
            ),
            metadata={"mode": "live", "runtime": "jericho", "restored": True},
        )
        state.world_state_snapshot = self.get_world_state_snapshot(state=state)
        self._last_state = state
        return state

    def _read_current_observation(self) -> str | None:
        """Read the best available current observation from the live backend."""

        if self._env is None:
            return None
        for attr_name in ("observation", "last_observation"):
            value = getattr(self._env, attr_name, None)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _step_stub(self, action: str) -> TextGameTransition:
        """Execute a deterministic placeholder transition."""

        step_index = self._step_index
        self._step_index += 1
        self._action_history.append(action)
        normalized = action.strip().lower()

        observation = f"Stub transition for action: {normalized or 'look'}."
        reward = 0.0
        world_state_hash = f"stub-step-{self._step_index}"

        if normalized in {"look", ""}:
            observation = "You are standing in an open field west of a white house, near a small mailbox."
            world_state_hash = "stub-west-of-house"
        elif normalized == "open mailbox":
            observation = "Opening the small mailbox reveals a leaflet."
            world_state_hash = "stub-open-mailbox"
        elif normalized == "read leaflet":
            observation = "Welcome to the scaffold. The leaflet mostly confirms this is a placeholder."
            world_state_hash = "stub-read-leaflet"

        done = self._step_index >= self.config.experiment.stub_episode_length or normalized == "quit"
        state = TextGameState(
            observation=observation,
            inventory_text="You are empty-handed.",
            valid_actions=self._stub_valid_actions(),
            score=0,
            moves=self._step_index,
            done=done,
            world_state_hash=world_state_hash,
            metadata={"mode": "stub", "runtime": "action_replay_fallback"},
        )
        state.world_state_snapshot = self.get_world_state_snapshot(state=state)
        self._last_state = state
        return TextGameTransition(
            action=action,
            step_index=step_index,
            observation=state.observation,
            reward=reward,
            done=state.done,
            score=state.score,
            moves=state.moves,
            inventory_text=state.inventory_text,
            valid_actions=list(state.valid_actions),
            world_state_hash=state.world_state_hash,
            world_state_snapshot=state.world_state_snapshot,
            metadata=dict(state.metadata),
        )

    def _get_moves(self, info: Mapping[str, Any] | None) -> int:
        """Return the current move count with graceful fallbacks."""

        if self._live_mode and self._env is not None and hasattr(self._env, "get_moves"):
            try:
                return int(self._env.get_moves())
            except Exception:
                pass
        if info and info.get("moves") is not None:
            return int(info["moves"])
        return self._last_state.moves if self._last_state is not None else self._step_index

    def _get_world_state_hash(self, observation: str, inventory_text: str) -> str:
        """Return Jericho's world-state hash when available, else a deterministic approximation."""

        if self._live_mode and self._env is not None and hasattr(self._env, "get_world_state_hash"):
            try:
                return str(self._env.get_world_state_hash())
            except Exception:
                pass
        digest = hashlib.sha1(
            "||".join([observation, inventory_text, *self._action_history]).encode("utf-8")
        ).hexdigest()
        return f"approx::{digest}"

    def _stub_valid_actions(self) -> list[str]:
        """Return a fixed set of parser actions for the deterministic stub."""

        return ["look", "open mailbox", "read leaflet", "quit"]

    def _inventory_item_name(self, item: Any) -> str:
        """Render one Jericho inventory object into readable text."""

        return str(getattr(item, "name", item))

    def _inventory_comparable(self, expected: str | None, actual: str | None) -> bool:
        """Return whether inventory strings are meaningful enough to compare."""

        if not expected or not actual:
            return False
        lowered = {expected.lower(), actual.lower()}
        return "inventory unavailable." not in lowered

    def _texts_match(self, expected: str, actual: str, *, exact: bool) -> bool:
        """Compare text conservatively, with optional exact matching."""

        normalized_expected = self._normalize_text(expected)
        normalized_actual = self._normalize_text(actual)
        if exact:
            return normalized_expected == normalized_actual
        return (
            normalized_expected == normalized_actual
            or normalized_expected in normalized_actual
            or normalized_actual in normalized_expected
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for conservative validation."""

        return " ".join(text.split()).strip().lower()
