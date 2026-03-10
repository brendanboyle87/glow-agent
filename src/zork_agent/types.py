"""Shared typed data containers used across the scaffold.

TODO: introduce stricter event schemas once the agent loop stabilizes.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import re
from typing import Any, Literal, Mapping, TypedDict


class ChatMessage(TypedDict):
    """OpenAI-style chat message payload."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(slots=True)
class TokenUsage:
    """Token usage reported by an LLM backend, when available."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> "TokenUsage | None":
        """Parse token usage from a backend payload."""

        # TODO: extend parsing if other backends use different usage field names.
        if not isinstance(payload, MappingABC):
            return None

        def _to_optional_int(value: Any) -> int | None:
            return int(value) if value is not None else None

        usage = cls(
            prompt_tokens=_to_optional_int(payload.get("prompt_tokens")),
            completion_tokens=_to_optional_int(payload.get("completion_tokens")),
            total_tokens=_to_optional_int(payload.get("total_tokens")),
        )
        if usage.prompt_tokens is None and usage.completion_tokens is None and usage.total_tokens is None:
            return None
        return usage


@dataclass(slots=True)
class LLMChatRequest:
    """Normalized chat-completion request for local backends."""

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 256
    timeout_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_prompts(
        cls,
        *,
        system_prompt: str | None,
        user_prompt: str,
        model: str | None,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "LLMChatRequest":
        """Build a chat request from simple system and user prompts."""

        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return cls(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            metadata=dict(metadata or {}),
        )


@dataclass(slots=True)
class LLMResponse:
    """Structured response returned by a local LLM backend."""

    text: str
    model: str
    latency_seconds: float
    usage: TokenUsage | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def raw(self) -> dict[str, Any]:
        """Backward-compatible alias for the raw payload."""

        return self.raw_payload


@dataclass(slots=True)
class WorldStateSnapshot:
    """Best-effort snapshot of the current text-game world state.

    Jericho exposes `get_state()` / `set_state()` for exact restoration on supported
    environments. When those internals are unavailable, the scaffold falls back to an
    action-replay snapshot built from the action history.
    """

    step_index: int
    observation: str
    done: bool
    score: int
    moves: int
    inventory_text: str = ""
    valid_actions: list[str] = field(default_factory=list)
    world_state_hash: str = "unknown"
    restore_strategy: Literal["native_snapshot", "action_replay_fallback"] = "action_replay_fallback"
    replay_actions: list[str] = field(default_factory=list)
    native_state: tuple[Any, ...] | None = None

    def to_record(self, include_internal_state: bool = False) -> dict[str, Any]:
        """Convert the snapshot into a JSON-serializable dictionary."""

        record = {
            "step_index": self.step_index,
            "observation": self.observation,
            "done": self.done,
            "score": self.score,
            "moves": self.moves,
            "inventory_text": self.inventory_text,
            "valid_actions": list(self.valid_actions),
            "world_state_hash": self.world_state_hash,
            "restore_strategy": self.restore_strategy,
            "replay_actions": list(self.replay_actions),
        }
        if include_internal_state and self.native_state is not None:
            record["native_state"] = list(self.native_state)
        return record

    @property
    def jericho_state(self) -> tuple[Any, ...] | None:
        """Backward-compatible alias for older code paths."""

        return self.native_state


def normalize_world_state_snapshot_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a serialized snapshot payload into the canonical in-memory shape.

    This keeps persisted JSONL data backward-compatible with older restore-strategy
    labels while ensuring native snapshot payloads are rehydrated into the tuple form
    expected by the Jericho-first restore path.
    """

    normalized = dict(payload)
    restore_strategy = normalized.get("restore_strategy")
    if isinstance(restore_strategy, str):
        normalized["restore_strategy"] = _normalize_restore_strategy_name(restore_strategy)

    for key in ("step_index", "score", "moves"):
        if key in normalized and normalized[key] is not None:
            normalized[key] = int(normalized[key])

    for key in ("observation", "inventory_text", "world_state_hash"):
        if key in normalized and normalized[key] is not None:
            normalized[key] = str(normalized[key])

    if "done" in normalized and normalized["done"] is not None:
        normalized["done"] = bool(normalized["done"])

    replay_actions = normalized.get("replay_actions")
    normalized["replay_actions"] = _normalize_string_list(replay_actions)

    if "valid_actions" in normalized:
        normalized["valid_actions"] = _normalize_string_list(normalized.get("valid_actions"))

    native_state = normalized.get("native_state")
    if isinstance(native_state, list):
        normalized["native_state"] = tuple(native_state)
    elif isinstance(native_state, tuple):
        normalized["native_state"] = tuple(native_state)

    return normalized


def _normalize_restore_strategy_name(value: str) -> str:
    """Normalize known restore-strategy aliases to the current schema."""

    normalized = value.strip()
    if normalized == "replay_only":
        return "action_replay_fallback"
    return normalized


def _normalize_string_list(value: Any) -> list[str]:
    """Normalize one-or-many string payloads into a list of strings."""

    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


_ACTION_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

_GENERIC_ACTION_TOKENS = {
    "around",
    "climb",
    "close",
    "drop",
    "enter",
    "examine",
    "exit",
    "get",
    "go",
    "inventory",
    "listen",
    "look",
    "move",
    "open",
    "push",
    "pull",
    "put",
    "read",
    "take",
    "touch",
    "turn",
    "use",
    "wait",
}

_DIRECTION_TOKENS = {
    "east",
    "north",
    "south",
    "west",
    "up",
    "down",
    "inside",
    "outside",
}

_SALIENT_INTERACTION_VERBS = {"take", "get", "examine", "read"}
_STATE_TOGGLE_FAMILY_VERBS = {"open", "close", "enter", "exit", "turn on", "turn off"}
_REVERSIBLE_STATE_TOKENS = {
    "close",
    "closed",
    "closing",
    "exit",
    "inside",
    "off",
    "on",
    "open",
    "opened",
    "opening",
    "outside",
    "shut",
    "toggle",
    "turn",
    "unlit",
    "lit",
}
_EXTENDED_DIRECTION_TOKENS = _DIRECTION_TOKENS | {
    "northeast",
    "northwest",
    "southeast",
    "southwest",
    "in",
    "out",
}
_MOVEMENT_VERBS = {
    "climb",
    "enter",
    "exit",
    "go",
    "move",
    "travel",
    "walk",
} | _EXTENDED_DIRECTION_TOKENS
_MOVEMENT_MACRO_PREFIXES = (
    "go around ",
    "go through ",
    "go to ",
    "go into ",
    "go in ",
    "go out ",
    "go down ",
    "go up ",
)
_MOVEMENT_DIRECTION_OPPOSITES = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
    "northeast": "southwest",
    "southwest": "northeast",
    "northwest": "southeast",
    "southeast": "northwest",
    "up": "down",
    "down": "up",
    "inside": "outside",
    "outside": "inside",
    "in": "out",
    "out": "in",
    "enter": "exit",
    "exit": "enter",
}
_ROOM_TEXT_ONLY_TOKENS = {
    "around",
    "area",
    "forest",
    "path",
    "region",
    "room",
    "trail",
    "trees",
}
_STATE_CLUSTER_PRIORITY_TOKENS = (
    "mailbox",
    "leaflet",
    "forest",
    "field",
    "house",
    "window",
    "door",
    "path",
    "stream",
    "room",
    "kitchen",
    "living",
    "attic",
    "cellar",
    "canyon",
    "valley",
)


def default_inverse_action_pairs() -> dict[str, str]:
    """Return the default inverse-verb mapping used for loop heuristics."""

    return {
        "open": "close",
        "close": "open",
        "take": "drop",
        "drop": "take",
        "enter": "exit",
        "exit": "enter",
        "turn on": "turn off",
        "turn off": "turn on",
    }


def normalize_inverse_action_pairs(pairs: Mapping[str, str] | None = None) -> dict[str, str]:
    """Normalize an inverse-action mapping and make it symmetric."""

    normalized: dict[str, str] = {}
    for left, right in dict(pairs or default_inverse_action_pairs()).items():
        left_key = normalize_parser_action(left)
        right_key = normalize_parser_action(right)
        if not left_key or not right_key:
            continue
        normalized[left_key] = right_key
        normalized[right_key] = left_key
    return normalized


def normalize_parser_action(action: str) -> str:
    """Normalize a parser command for loop detection."""

    return " ".join(action.split()).strip().lower()


def split_action_command(action: str, inverse_pairs: Mapping[str, str] | None = None) -> tuple[str, str]:
    """Split a parser command into its verb phrase and object phrase."""

    normalized = normalize_parser_action(action)
    if not normalized:
        return "", ""

    known_verbs = sorted(normalize_inverse_action_pairs(inverse_pairs).keys(), key=len, reverse=True)
    for verb in known_verbs:
        if normalized == verb:
            return verb, ""
        if normalized.startswith(verb + " "):
            return verb, _normalize_action_object(normalized[len(verb) :].strip())

    parts = normalized.split(" ", 1)
    verb = parts[0]
    obj = _normalize_action_object(parts[1]) if len(parts) > 1 else ""
    return verb, obj


def actions_are_inverse(
    action: str,
    previous_action: str | None,
    inverse_pairs: Mapping[str, str] | None = None,
) -> bool:
    """Return whether two parser commands are inverse actions on the same target."""

    if not previous_action:
        return False
    normalized_pairs = normalize_inverse_action_pairs(inverse_pairs)
    current_verb, current_object = split_action_command(action, normalized_pairs)
    previous_verb, previous_object = split_action_command(previous_action, normalized_pairs)
    if not current_verb or not previous_verb:
        return False
    if normalized_pairs.get(previous_verb) != current_verb:
        return False
    if not current_object and not previous_object:
        return True
    return current_object == previous_object


def count_repeated_inverse_pairs(
    actions: list[str],
    inverse_pairs: Mapping[str, str] | None = None,
) -> int:
    """Count alternating inverse-pair repetitions in the action suffix.

    Examples:
    - `open, close` => `0`
    - `open, close, open, close` => `1`
    - `open, close, open, close, open, close` => `2`
    """

    if len(actions) < 4:
        return 0
    pair_first = normalize_parser_action(actions[-2])
    pair_second = normalize_parser_action(actions[-1])
    if not actions_are_inverse(pair_second, pair_first, inverse_pairs):
        return 0

    repetitions = 0
    cursor = len(actions) - 4
    while cursor >= 0:
        if (
            normalize_parser_action(actions[cursor]) == pair_first
            and normalize_parser_action(actions[cursor + 1]) == pair_second
        ):
            repetitions += 1
            cursor -= 2
            continue
        break
    return repetitions


def observation_change_is_trivial(
    reference_observation: str,
    current_observation: str,
    inverse_pairs: Mapping[str, str] | None = None,
) -> bool:
    """Return whether an observation delta looks like a reversible surface flip."""

    normalized_reference = normalize_parser_action(reference_observation)
    normalized_current = normalize_parser_action(current_observation)
    if normalized_reference == normalized_current:
        return True

    ignored_tokens = {
        token
        for phrase in normalize_inverse_action_pairs(inverse_pairs).keys()
        for token in phrase.split()
    }
    reference_tokens = _significant_text_tokens(reference_observation, ignored_tokens)
    current_tokens = _significant_text_tokens(current_observation, ignored_tokens)
    if not reference_tokens or not current_tokens:
        return False
    overlap = len(reference_tokens & current_tokens)
    union = len(reference_tokens | current_tokens)
    return union > 0 and (overlap / union) >= 0.8


def valid_actions_equivalent(left: list[str], right: list[str]) -> bool:
    """Return whether two valid-action lists are effectively unchanged."""

    return {normalize_parser_action(action) for action in left if normalize_parser_action(action)} == {
        normalize_parser_action(action) for action in right if normalize_parser_action(action)
    }


@dataclass(slots=True)
class LoopHeuristicResult:
    """Heuristic signal describing a reversible no-progress loop."""

    loop_detected: bool = False
    inverse_of_previous: bool = False
    repeated_pair_count: int = 0
    no_progress: bool = False
    inventory_unchanged: bool = False
    valid_actions_unchanged: bool = False
    observation_delta_trivial: bool = False
    immediate_inverse_penalty: float = 0.0
    repeated_pair_penalty: float = 0.0
    reversible_no_progress_penalty: float = 0.0
    total_penalty: float = 0.0
    reference_step_distance: int = 0
    reason: str = ""

    def to_record(self) -> dict[str, Any]:
        """Convert the loop signal into a JSON-serializable record."""

        return asdict(self)


@dataclass(slots=True)
class MovementHeuristicResult:
    """Heuristic signal describing low-value movement wandering."""

    movement_only_action: bool = False
    movement_repeat_count: int = 0
    movement_cycle_count: int = 0
    same_region_repeat: bool = False
    no_progress: bool = False
    materially_new_actions: bool = False
    revealed_new_salient_object: bool = False
    observation_repeat: bool = False
    repeated_movement_penalty: float = 0.0
    movement_cycle_penalty: float = 0.0
    same_region_repeat_penalty: float = 0.0
    total_penalty: float = 0.0
    state_cluster_id: str = ""
    cluster_visit_count: int = 0
    region_novelty_score: float = 0.0
    reason: str = ""

    def to_record(self) -> dict[str, Any]:
        """Convert the movement signal into a JSON-serializable record."""

        return asdict(self)


def evaluate_reversible_action_loop(
    *,
    action: str,
    recent_actions: list[str],
    state_history: list["TextGameState"],
    current_state: "TextGameState",
    inverse_pairs: Mapping[str, str] | None = None,
    immediate_inverse_penalty: float,
    repeated_pair_penalty: float,
    reversible_no_progress_penalty: float,
) -> LoopHeuristicResult:
    """Evaluate whether the latest action completed a reversible no-progress loop."""

    normalized_pairs = normalize_inverse_action_pairs(inverse_pairs)
    inverse_of_previous = bool(recent_actions) and actions_are_inverse(
        action,
        recent_actions[-1],
        normalized_pairs,
    )
    repeated_pair_count = count_repeated_inverse_pairs([*recent_actions, action], normalized_pairs)

    reference_state: TextGameState | None = None
    reference_step_distance = 0
    if repeated_pair_count > 0:
        alternating_span = 2 * (repeated_pair_count + 1)
        if len(state_history) >= alternating_span:
            reference_state = state_history[-alternating_span]
            reference_step_distance = alternating_span
    if reference_state is None and inverse_of_previous and len(state_history) >= 2:
        reference_state = state_history[-2]
        reference_step_distance = 2
    if reference_state is None and state_history:
        reference_state = state_history[-1]
        reference_step_distance = 1

    if reference_state is None:
        return LoopHeuristicResult()

    score_delta = current_state.score - reference_state.score
    inventory_unchanged = normalize_parser_action(current_state.inventory_text) == normalize_parser_action(
        reference_state.inventory_text
    )
    valid_actions_unchanged = valid_actions_equivalent(current_state.valid_actions, reference_state.valid_actions)
    observation_delta_trivial = observation_change_is_trivial(
        reference_state.observation,
        current_state.observation,
        normalized_pairs,
    )
    no_progress = (
        score_delta == 0
        and inventory_unchanged
        and valid_actions_unchanged
        and observation_delta_trivial
    )

    immediate_penalty = immediate_inverse_penalty if inverse_of_previous and no_progress else 0.0
    repeated_penalty = repeated_pair_penalty * repeated_pair_count if repeated_pair_count > 0 and no_progress else 0.0
    reversible_penalty = (
        reversible_no_progress_penalty
        if (inverse_of_previous or repeated_pair_count > 0) and no_progress
        else 0.0
    )
    total_penalty = immediate_penalty + repeated_penalty + reversible_penalty

    reason_parts: list[str] = []
    if inverse_of_previous:
        reason_parts.append("inverse_of_previous")
    if repeated_pair_count > 0:
        reason_parts.append(f"repeated_pair_count={repeated_pair_count}")
    if no_progress:
        reason_parts.append("no_progress")

    return LoopHeuristicResult(
        loop_detected=total_penalty > 0.0,
        inverse_of_previous=inverse_of_previous,
        repeated_pair_count=repeated_pair_count,
        no_progress=no_progress,
        inventory_unchanged=inventory_unchanged,
        valid_actions_unchanged=valid_actions_unchanged,
        observation_delta_trivial=observation_delta_trivial,
        immediate_inverse_penalty=immediate_penalty,
        repeated_pair_penalty=repeated_penalty,
        reversible_no_progress_penalty=reversible_penalty,
        total_penalty=total_penalty,
        reference_step_distance=reference_step_distance,
        reason=", ".join(reason_parts),
    )


def estimate_candidate_loop_penalty(
    *,
    candidate_action: str,
    recent_actions: list[str],
    recent_loop_results: list[LoopHeuristicResult],
    inverse_pairs: Mapping[str, str] | None = None,
    immediate_inverse_penalty: float,
    repeated_pair_penalty: float,
    reversible_no_progress_penalty: float,
) -> tuple[float, str]:
    """Estimate whether a candidate action likely continues a recent reversible loop."""

    if not recent_actions or not recent_loop_results:
        return 0.0, ""
    last_loop = recent_loop_results[-1]
    if not last_loop.no_progress:
        return 0.0, ""

    normalized_pairs = normalize_inverse_action_pairs(inverse_pairs)
    inverse_of_previous = actions_are_inverse(candidate_action, recent_actions[-1], normalized_pairs)
    repeated_pair_count = count_repeated_inverse_pairs([*recent_actions, candidate_action], normalized_pairs)
    if not inverse_of_previous and repeated_pair_count <= 0:
        return 0.0, ""

    total_penalty = 0.0
    reasons: list[str] = []
    if inverse_of_previous:
        total_penalty += immediate_inverse_penalty
        reasons.append("inverse_of_previous")
    if repeated_pair_count > 0:
        total_penalty += repeated_pair_penalty * repeated_pair_count
        reasons.append(f"repeated_pair_count={repeated_pair_count}")
    total_penalty += reversible_no_progress_penalty
    reasons.append("recent_no_progress")
    return total_penalty, ", ".join(reasons)


def evaluate_movement_action(
    *,
    action: str,
    recent_actions: list[str],
    previous_state: "TextGameState",
    current_state: "TextGameState",
    inverse_pairs: Mapping[str, str] | None = None,
    repeated_movement_penalty: float,
    movement_cycle_penalty: float,
    same_region_repeat_penalty: float,
) -> MovementHeuristicResult:
    """Evaluate whether a movement action is low-value wandering."""

    movement_only_action = is_movement_action(action, inverse_pairs)
    current_cluster_id = current_state.state_cluster_id or derive_state_cluster_id(
        observation=current_state.observation,
        inventory_text=current_state.inventory_text,
        valid_actions=current_state.valid_actions,
        inverse_pairs=inverse_pairs,
    )
    cluster_visit_count = max(current_state.cluster_visit_count, 0)
    if not movement_only_action:
        return MovementHeuristicResult(
            movement_only_action=False,
            state_cluster_id=current_cluster_id,
            cluster_visit_count=cluster_visit_count,
            region_novelty_score=current_state.region_novelty_score,
        )

    previous_cluster_id = previous_state.state_cluster_id or derive_state_cluster_id(
        observation=previous_state.observation,
        inventory_text=previous_state.inventory_text,
        valid_actions=previous_state.valid_actions,
        inverse_pairs=inverse_pairs,
    )
    score_gain = current_state.score - previous_state.score
    inventory_changed = normalize_parser_action(current_state.inventory_text) != normalize_parser_action(
        previous_state.inventory_text
    )
    materially_new_actions = valid_actions_materially_different(previous_state.valid_actions, current_state.valid_actions)
    previous_nouns = extract_salient_nouns(
        observation=previous_state.observation,
        inventory_text=previous_state.inventory_text,
        valid_actions=[],
        inverse_pairs=inverse_pairs,
    )
    current_nouns = extract_salient_nouns(
        observation=current_state.observation,
        inventory_text=current_state.inventory_text,
        valid_actions=[],
        inverse_pairs=inverse_pairs,
    )
    revealed_new_salient_object = bool(current_nouns - previous_nouns)
    observation_repeat = (
        current_cluster_id == previous_cluster_id
        and region_observation_similarity(previous_state.observation, current_state.observation) >= 0.7
    )
    no_progress = (
        score_gain == 0
        and not inventory_changed
        and not materially_new_actions
        and not revealed_new_salient_object
    )

    movement_actions = [*recent_actions, action]
    movement_repeat_count = count_repeated_movement_actions(movement_actions, inverse_pairs)
    movement_cycle_count = count_movement_cycle_repetitions(movement_actions, inverse_pairs)
    if movement_cycle_count <= 0 and movement_actions_are_cycle(action, recent_actions[-1] if recent_actions else None):
        movement_cycle_count = 1

    same_region_repeat = (
        current_cluster_id == previous_cluster_id
        and cluster_visit_count > 1
        and observation_repeat
    )

    repeated_penalty = 0.0
    cycle_penalty_value = 0.0
    same_region_penalty_value = 0.0
    reasons: list[str] = []
    if no_progress:
        if movement_repeat_count > 0:
            repeated_penalty = repeated_movement_penalty * float(movement_repeat_count)
            reasons.append(f"movement_repeat_count={movement_repeat_count}")
        if movement_cycle_count > 0:
            cycle_penalty_value = movement_cycle_penalty * float(movement_cycle_count)
            reasons.append(f"movement_cycle_count={movement_cycle_count}")
        if same_region_repeat:
            same_region_penalty_value = same_region_repeat_penalty * float(max(cluster_visit_count - 1, 1))
            reasons.append(f"same_region_repeat={current_cluster_id}")
    total_penalty = repeated_penalty + cycle_penalty_value + same_region_penalty_value
    if no_progress:
        reasons.append("no_progress")

    return MovementHeuristicResult(
        movement_only_action=movement_only_action,
        movement_repeat_count=movement_repeat_count,
        movement_cycle_count=movement_cycle_count,
        same_region_repeat=same_region_repeat,
        no_progress=no_progress,
        materially_new_actions=materially_new_actions,
        revealed_new_salient_object=revealed_new_salient_object,
        observation_repeat=observation_repeat,
        repeated_movement_penalty=repeated_penalty,
        movement_cycle_penalty=cycle_penalty_value,
        same_region_repeat_penalty=same_region_penalty_value,
        total_penalty=total_penalty,
        state_cluster_id=current_cluster_id,
        cluster_visit_count=cluster_visit_count,
        region_novelty_score=current_state.region_novelty_score,
        reason=", ".join(reasons),
    )


def _normalize_action_object(value: str) -> str:
    """Normalize the object portion of a parser command."""

    tokens = [token for token in value.split() if token not in _ACTION_TOKEN_STOPWORDS]
    return " ".join(tokens)


def _significant_text_tokens(text: str, ignored_tokens: set[str]) -> set[str]:
    """Extract informative tokens for trivial-observation comparisons."""

    tokens = {
        token
        for token in re.findall(r"[a-z]+", text.lower())
        if len(token) > 2 and token not in _ACTION_TOKEN_STOPWORDS and token not in ignored_tokens
    }
    return tokens


def extract_salient_nouns(
    *,
    observation: str,
    inventory_text: str = "",
    valid_actions: list[str] | None = None,
    inverse_pairs: Mapping[str, str] | None = None,
) -> set[str]:
    """Extract lightweight noun-like tokens from the current local state.

    This is intentionally heuristic. It works from parser observations and valid
    actions without relying on any NLP library.
    """

    ignored_tokens = {
        token
        for phrase in normalize_inverse_action_pairs(inverse_pairs).keys()
        for token in phrase.split()
    } | _GENERIC_ACTION_TOKENS | _DIRECTION_TOKENS
    text_blocks = [observation, inventory_text, *(valid_actions or [])]
    nouns: set[str] = set()
    for block in text_blocks:
        nouns.update(_significant_text_tokens(block, ignored_tokens))
    return nouns


def action_target_tokens(
    action: str,
    inverse_pairs: Mapping[str, str] | None = None,
) -> set[str]:
    """Extract noun-like target tokens from a parser action."""

    _verb, obj = split_action_command(action, inverse_pairs)
    if not obj:
        return set()
    return {
        token
        for token in re.findall(r"[a-z]+", obj.lower())
        if len(token) > 2 and token not in _ACTION_TOKEN_STOPWORDS and token not in _DIRECTION_TOKENS
    }


def is_movement_action(
    action: str,
    inverse_pairs: Mapping[str, str] | None = None,
) -> bool:
    """Return whether a parser command is primarily a movement/navigation action."""

    normalized = normalize_parser_action(action)
    if not normalized:
        return False
    if normalized in _EXTENDED_DIRECTION_TOKENS:
        return True
    if any(normalized.startswith(prefix) for prefix in _MOVEMENT_MACRO_PREFIXES):
        return True
    verb, _obj = split_action_command(normalized, inverse_pairs)
    return verb in _MOVEMENT_VERBS


def movement_action_signature(
    action: str,
    inverse_pairs: Mapping[str, str] | None = None,
) -> str:
    """Return a normalized signature for movement-loop detection."""

    normalized = normalize_parser_action(action)
    if not is_movement_action(normalized, inverse_pairs):
        return ""
    if normalized in _EXTENDED_DIRECTION_TOKENS:
        return normalized
    for prefix in _MOVEMENT_MACRO_PREFIXES:
        if normalized.startswith(prefix):
            return normalized
    verb, obj = split_action_command(normalized, inverse_pairs)
    if verb == "go" and obj:
        return f"go:{obj}"
    return normalized


def movement_actions_are_cycle(
    action: str,
    previous_action: str | None,
    inverse_pairs: Mapping[str, str] | None = None,
) -> bool:
    """Return whether two movement actions are obvious opposites."""

    if not previous_action:
        return False
    current = movement_action_signature(action, inverse_pairs)
    previous = movement_action_signature(previous_action, inverse_pairs)
    if not current or not previous:
        return False
    return _MOVEMENT_DIRECTION_OPPOSITES.get(previous) == current


def count_repeated_movement_actions(
    actions: list[str],
    inverse_pairs: Mapping[str, str] | None = None,
) -> int:
    """Count consecutive repeats of the latest movement action."""

    if not actions:
        return 0
    latest = movement_action_signature(actions[-1], inverse_pairs)
    if not latest:
        return 0
    repeats = 0
    cursor = len(actions) - 2
    while cursor >= 0 and movement_action_signature(actions[cursor], inverse_pairs) == latest:
        repeats += 1
        cursor -= 1
    return repeats


def count_movement_cycle_repetitions(
    actions: list[str],
    inverse_pairs: Mapping[str, str] | None = None,
) -> int:
    """Count short alternating movement cycles in the action suffix."""

    if len(actions) < 4:
        return 0
    signatures = [movement_action_signature(action, inverse_pairs) for action in actions]
    if not all(signatures[-4:]):
        return 0
    if signatures[-1] != signatures[-3] or signatures[-2] != signatures[-4]:
        return 0
    if signatures[-1] == signatures[-2]:
        return 0
    repetitions = 1
    cursor = len(signatures) - 6
    while cursor >= 0:
        if (
            signatures[cursor] == signatures[-2]
            and signatures[cursor + 1] == signatures[-1]
        ):
            repetitions += 1
            cursor -= 2
            continue
        break
    return repetitions


def valid_actions_materially_different(left: list[str], right: list[str]) -> bool:
    """Return whether two valid-action lists differ in materially useful ways."""

    def material_actions(actions: list[str]) -> set[str]:
        return {
            normalize_parser_action(action)
            for action in actions
            if normalize_parser_action(action)
            and not is_movement_action(action)
            and normalize_parser_action(action) not in {"inventory", "look", "wait"}
        }

    return material_actions(left) != material_actions(right)


def derive_state_cluster_id(
    *,
    observation: str,
    inventory_text: str = "",
    valid_actions: list[str] | None = None,
    summary_text: str = "",
    inverse_pairs: Mapping[str, str] | None = None,
) -> str:
    """Build a lightweight region/state-cluster identifier for movement novelty decay."""

    normalized_observation = normalize_parser_action(observation)
    if "west of a white house" in normalized_observation or "west of house" in normalized_observation:
        return "region:house-field"

    salient_tokens = extract_salient_nouns(
        observation=observation,
        inventory_text=inventory_text,
        valid_actions=[],
        inverse_pairs=inverse_pairs,
    )
    for token in _STATE_CLUSTER_PRIORITY_TOKENS:
        if token in salient_tokens:
            return f"region:{token}"

    valid_action_targets = {
        token
        for action in (valid_actions or [])
        if not is_movement_action(action, inverse_pairs)
        for token in action_target_tokens(action, inverse_pairs)
    }
    for token in _STATE_CLUSTER_PRIORITY_TOKENS:
        if token in valid_action_targets:
            return f"region:{token}"

    stable_tokens = sorted((salient_tokens | valid_action_targets) - _ROOM_TEXT_ONLY_TOKENS)
    if stable_tokens:
        return "region:" + "|".join(stable_tokens[:3])
    if summary_text.strip():
        return "region:summary:" + normalize_parser_action(summary_text)[:48]
    return "region:observation:" + normalize_parser_action(observation)[:48]


def region_observation_similarity(left: str, right: str) -> float:
    """Compute a simple token-overlap similarity for room-description clustering."""

    left_tokens = _significant_text_tokens(left, set())
    right_tokens = _significant_text_tokens(right, set())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def compute_region_novelty_score(
    *,
    cluster_visit_count: int,
    observation_seen_before: bool,
    score_gain: int,
    inventory_changed: bool,
    affordance_gain: int,
    novel_object_count: int,
    materially_new_actions: bool,
    movement_only_action: bool,
) -> float:
    """Return a lightweight novelty score for a state cluster revisit."""

    if score_gain > 0 or inventory_changed or affordance_gain > 0:
        return 1.0
    if novel_object_count > 0 or materially_new_actions:
        return 0.7
    if cluster_visit_count <= 1 and not observation_seen_before:
        return 0.35
    if movement_only_action:
        if cluster_visit_count <= 1 and not observation_seen_before:
            return 0.2
        if cluster_visit_count == 2:
            return 0.1
        return 0.05
    return 0.15 if not observation_seen_before else 0.05


def derive_state_family_key(
    *,
    observation: str,
    inventory_text: str = "",
    valid_actions: list[str] | None = None,
    last_action: str = "",
    summary_text: str = "",
    inverse_pairs: Mapping[str, str] | None = None,
) -> str:
    """Build a lightweight family key for near-identical reversible states.

    Preference order:
    1. Toggle-target nouns from inverse valid actions or the last action.
    2. Stable salient nouns from the observation/inventory after stripping reversible words.
    3. A normalized observation fallback.
    """

    normalized_pairs = normalize_inverse_action_pairs(inverse_pairs)
    toggle_targets: set[str] = set()
    for action in [*(valid_actions or []), last_action]:
        if not action:
            continue
        verb, _obj = split_action_command(action, normalized_pairs)
        if verb in _STATE_TOGGLE_FAMILY_VERBS:
            toggle_targets.update(action_target_tokens(action, normalized_pairs))
    if toggle_targets:
        return "toggle:" + "|".join(sorted(toggle_targets))

    salient_tokens = extract_salient_nouns(
        observation=observation,
        inventory_text=inventory_text,
        valid_actions=[],
        inverse_pairs=normalized_pairs,
    )
    stable_tokens = sorted(token for token in salient_tokens if token not in _REVERSIBLE_STATE_TOKENS)
    if stable_tokens:
        return "state:" + "|".join(stable_tokens[:4])

    if summary_text.strip():
        return "summary:" + normalize_parser_action(summary_text)[:64]
    return "observation:" + normalize_parser_action(observation)[:64]


@dataclass(slots=True)
class ActionAttemptStats:
    """Outcome statistics for one action within a local state cluster."""

    action: str
    attempts: int = 0
    score_change_count: int = 0
    inventory_change_count: int = 0
    observation_change_count: int = 0
    valid_actions_change_count: int = 0
    no_gain_count: int = 0

    @property
    def had_any_gain(self) -> bool:
        """Return whether the action ever produced an observable gain signal."""

        return any(
            (
                self.score_change_count,
                self.inventory_change_count,
                self.observation_change_count,
                self.valid_actions_change_count,
            )
        )

    def record_attempt(
        self,
        *,
        score_changed: bool,
        inventory_changed: bool,
        observation_changed: bool,
        valid_actions_changed: bool,
    ) -> None:
        """Accumulate one observed action outcome."""

        self.attempts += 1
        if score_changed:
            self.score_change_count += 1
        if inventory_changed:
            self.inventory_change_count += 1
        if observation_changed:
            self.observation_change_count += 1
        if valid_actions_changed:
            self.valid_actions_change_count += 1
        if not any((score_changed, inventory_changed, observation_changed, valid_actions_changed)):
            self.no_gain_count += 1


@dataclass(slots=True)
class ActionClusterHistory:
    """Recent action memory for the current local state cluster.

    This is a pragmatic approximation. A cluster is the current local region of
    play between resets/restores, not a formal game-state equivalence class.
    """

    cluster_label: str = ""
    seen_nouns: set[str] = field(default_factory=set)
    action_stats: dict[str, ActionAttemptStats] = field(default_factory=dict)
    current_state_cluster_id: str = ""
    cluster_visit_counts: dict[str, int] = field(default_factory=dict)
    cluster_observation_signatures: dict[str, set[str]] = field(default_factory=dict)
    recent_cluster_sequence: list[str] = field(default_factory=list)

    def stats_for(self, action: str) -> ActionAttemptStats:
        """Return mutable stats for an action within this cluster."""

        normalized = normalize_parser_action(action)
        if normalized not in self.action_stats:
            self.action_stats[normalized] = ActionAttemptStats(action=normalized)
        return self.action_stats[normalized]

    def record_attempt(
        self,
        *,
        action: str,
        score_changed: bool,
        inventory_changed: bool,
        observation_changed: bool,
        valid_actions_changed: bool,
    ) -> None:
        """Record one attempted action and its observed effect."""

        self.stats_for(action).record_attempt(
            score_changed=score_changed,
            inventory_changed=inventory_changed,
            observation_changed=observation_changed,
            valid_actions_changed=valid_actions_changed,
        )

    def observe_state_nouns(
        self,
        *,
        observation: str,
        inventory_text: str = "",
        valid_actions: list[str] | None = None,
        inverse_pairs: Mapping[str, str] | None = None,
    ) -> None:
        """Remember nouns that have already been surfaced in this local cluster."""

        self.seen_nouns.update(
            extract_salient_nouns(
                observation=observation,
                inventory_text=inventory_text,
                valid_actions=valid_actions,
                inverse_pairs=inverse_pairs,
            )
        )

    def record_state_cluster(self, *, cluster_id: str, observation: str) -> tuple[int, bool]:
        """Record a visit to the current state cluster and return visit metadata."""

        normalized_cluster = normalize_parser_action(cluster_id)
        if not normalized_cluster:
            normalized_cluster = "region:unknown"
        observation_signature = normalize_parser_action(observation)
        seen_signatures = self.cluster_observation_signatures.setdefault(normalized_cluster, set())
        observation_seen_before = observation_signature in seen_signatures
        seen_signatures.add(observation_signature)
        self.cluster_visit_counts[normalized_cluster] = self.cluster_visit_counts.get(normalized_cluster, 0) + 1
        self.current_state_cluster_id = normalized_cluster
        self.recent_cluster_sequence.append(normalized_cluster)
        self.recent_cluster_sequence = self.recent_cluster_sequence[-16:]
        return self.cluster_visit_counts[normalized_cluster], observation_seen_before

    def cluster_visit_count(self, cluster_id: str | None = None) -> int:
        """Return the visit count for the current or chosen state cluster."""

        target_cluster = normalize_parser_action(cluster_id or self.current_state_cluster_id)
        if not target_cluster:
            return 0
        return self.cluster_visit_counts.get(target_cluster, 0)


class ReplayDivergenceReason(str, Enum):
    """Conservative reasons why restoration-via-replay may have diverged."""

    NONE = "none"
    OBSERVATION_MISMATCH = "observation_mismatch"
    SCORE_MISMATCH = "score_mismatch"
    INVENTORY_MISMATCH = "inventory_mismatch"
    DONE_MISMATCH = "done_mismatch"
    STEP_EXCEPTION = "step_exception"


class RestoreMode(str, Enum):
    """Restoration mode used to revisit a previously saved node."""

    NATIVE_SNAPSHOT = "native_snapshot"
    ACTION_REPLAY_FALLBACK = "action_replay_fallback"
    RESET_ONLY = "reset_only"
    FAILED = "failed"


@dataclass(slots=True)
class TextGameState:
    """Minimal text-game state description."""

    observation: str
    inventory_text: str = ""
    valid_actions: list[str] = field(default_factory=list)
    score: int = 0
    moves: int = 0
    done: bool = False
    world_state_hash: str = "unknown"
    state_cluster_id: str = ""
    cluster_visit_count: int = 0
    region_novelty_score: float = 0.0
    world_state_snapshot: WorldStateSnapshot | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def inventory(self) -> str:
        """Backward-compatible alias for inventory text."""

        return self.inventory_text


@dataclass(slots=True)
class SavedNode:
    """Jericho-first saved frontier node used for fast restoration.

    Native `get_state()` / `set_state()` snapshots are the preferred restore path.
    The action prefix is retained as a fallback when native restore is unavailable
    or validation fails.
    """

    state_id: str
    episode_id: str
    step_index: int
    native_state: tuple[Any, ...] | None
    action_prefix: list[str]
    score: int
    observation: str
    inventory_text: str
    world_state_hash: str = "unknown"
    valid_actions: list[str] = field(default_factory=list)
    summary_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_native_state(self) -> bool:
        """Return whether this node still retains a native Jericho snapshot."""

        return self.native_state is not None

    def without_native_state(self) -> "SavedNode":
        """Return a copy without the retained native snapshot payload."""

        return SavedNode(
            state_id=self.state_id,
            episode_id=self.episode_id,
            step_index=self.step_index,
            native_state=None,
            action_prefix=list(self.action_prefix),
            score=self.score,
            observation=self.observation,
            inventory_text=self.inventory_text,
            world_state_hash=self.world_state_hash,
            valid_actions=list(self.valid_actions),
            summary_text=self.summary_text,
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class RestoreValidation:
    """Validation result for a restored Jericho state."""

    is_valid: bool
    divergence_reason: ReplayDivergenceReason
    score_matches: bool
    observation_matches: bool
    inventory_matches: bool | None
    actual_observation: str
    actual_score: int
    actual_inventory_text: str
    expected_observation: str | None = None
    expected_score: int | None = None
    expected_inventory_text: str | None = None
    message: str = ""


@dataclass(slots=True)
class ReplayResult:
    """Typed result for replay-based restoration attempts."""

    success: bool
    restore_mode: RestoreMode
    divergence_reason: ReplayDivergenceReason
    final_observation: str
    final_score: int
    replayed_action_count: int
    target_step_index: int | None
    final_state: TextGameState
    validation: RestoreValidation | None = None
    native_restore_attempted: bool = False
    expected_observation: str | None = None
    expected_score: int | None = None
    diverged_at_step_index: int | None = None
    restored_node_id: str | None = None
    message: str = ""

    @property
    def clean_replay(self) -> bool:
        """Return whether the replay finished without detected divergence."""

        return self.success and self.divergence_reason is ReplayDivergenceReason.NONE


@dataclass(slots=True)
class TextGameTransition:
    """Normalized result of taking one parser action."""

    action: str
    step_index: int
    observation: str
    reward: float
    done: bool
    score: int
    moves: int
    inventory_text: str = ""
    valid_actions: list[str] = field(default_factory=list)
    world_state_hash: str = "unknown"
    world_state_snapshot: WorldStateSnapshot | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_state(self) -> TextGameState:
        """Convert a transition into the resulting state."""

        return TextGameState(
            observation=self.observation,
            inventory_text=self.inventory_text,
            valid_actions=list(self.valid_actions),
            score=self.score,
            moves=self.moves,
            done=self.done,
            world_state_hash=self.world_state_hash,
            state_cluster_id=str(self.metadata.get("state_cluster_id", "")),
            cluster_visit_count=int(self.metadata.get("cluster_visit_count", 0)),
            region_novelty_score=float(self.metadata.get("region_novelty_score", 0.0)),
            world_state_snapshot=self.world_state_snapshot,
            metadata=dict(self.metadata),
        )

    def to_trajectory_step(
        self,
        *,
        episode_id: str,
        metadata: Mapping[str, Any] | None = None,
        step_index_override: int | None = None,
        loop_result: LoopHeuristicResult | None = None,
    ) -> "TrajectoryStep":
        """Convert a transition into a JSONL-friendly trajectory step."""

        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return TrajectoryStep(
            episode_id=episode_id,
            step_index=self.step_index if step_index_override is None else step_index_override,
            action=self.action,
            observation=self.observation,
            reward=self.reward,
            done=self.done,
            score=self.score,
            moves=self.moves,
            inventory_text=self.inventory_text,
            valid_actions=list(self.valid_actions),
            world_state_hash=self.world_state_hash,
            movement_only_action=bool(merged_metadata.get("movement_only_action", False)),
            movement_repeat_count=int(merged_metadata.get("movement_repeat_count", 0)),
            loop_detected=loop_result.loop_detected if loop_result is not None else False,
            inverse_of_previous=loop_result.inverse_of_previous if loop_result is not None else False,
            repeated_pair_count=loop_result.repeated_pair_count if loop_result is not None else 0,
            loop_penalty=loop_result.total_penalty if loop_result is not None else 0.0,
            movement_penalty=float(merged_metadata.get("movement_penalty", 0.0)),
            state_cluster_id=str(merged_metadata.get("state_cluster_id", "")),
            cluster_visit_count=int(merged_metadata.get("cluster_visit_count", 0)),
            region_novelty_score=float(merged_metadata.get("region_novelty_score", 0.0)),
            room_text_only_gain=float(merged_metadata.get("room_text_only_gain", 0.0)),
            affordance_gain=int(merged_metadata.get("affordance_gain", 0)),
            world_state_snapshot=(
                self.world_state_snapshot.to_record(include_internal_state=False)
                if self.world_state_snapshot is not None
                else None
            ),
            metadata=merged_metadata,
        )


@dataclass(slots=True)
class ActionProposal:
    """A proposed parser action plus optional rationale and ranking data."""

    action: str
    rationale: str = ""
    source: str = "stub"
    rank: int | None = None
    confidence: float | None = None
    heuristic_bonus: float = 0.0
    heuristic_penalty: float = 0.0
    selection_score: float = 0.0
    ranking_reason: str = ""
    loop_penalty: float = 0.0
    loop_penalty_reason: str = ""
    movement_penalty: float = 0.0
    movement_penalty_reason: str = ""
    movement_only_action: bool = False
    raw_line: str = ""


class ActionGenerationMode(str, Enum):
    """Mode used to generate candidate parser actions."""

    CONSTRAINED = "constrained"
    OPEN = "open"
    FALLBACK = "fallback"


@dataclass(slots=True)
class ActionGenerationResult:
    """Structured result from the action-generation policy."""

    mode: ActionGenerationMode
    candidates: list[ActionProposal]
    raw_output: str = ""
    model_name: str | None = None
    valid_actions: list[str] = field(default_factory=list)
    fallback_reason: str = ""
    reranked_by_loop_penalty: bool = False
    reranked_by_affordance_heuristics: bool = False
    reranked_by_movement_heuristics: bool = False
    augmented_with_valid_actions: bool = False

    def top_actions(self) -> list[str]:
        """Return the candidate action texts in ranked order."""

        return [candidate.action for candidate in self.candidates]


class StateSelectionMode(str, Enum):
    """Mode used to choose the next frontier state for revisit."""

    HEURISTIC = "heuristic"
    LLM_ASSISTED = "llm_assisted"
    FALLBACK_HEURISTIC = "fallback_heuristic"


@dataclass(slots=True)
class FrontierCandidateSummary:
    """Compact frontier-candidate summary used for prompt construction and analysis."""

    candidate_id: str
    state_id: str
    score: float
    depth: int
    novelty: float
    recent_gain: float
    summary_text: str
    has_native_snapshot: bool = False

    def to_prompt_line(self) -> str:
        """Render a bounded one-line summary for LLM-assisted selection."""

        native_snapshot = "yes" if self.has_native_snapshot else "no"
        return (
            f"{self.candidate_id} | score={self.score:.2f} | depth={self.depth} "
            f"| novelty={self.novelty:.2f} | gain={self.recent_gain:.2f} "
            f"| native_snapshot={native_snapshot} | state={self.summary_text}"
        )


@dataclass(slots=True)
class StateSelectionResult:
    """Structured result from heuristic or LLM-assisted state selection."""

    selected: FrontierEntry | None
    reason: str
    selection_mode: StateSelectionMode
    candidate_summaries: list[FrontierCandidateSummary] = field(default_factory=list)
    raw_output: str = ""
    prompt_snapshot: str = ""
    model_name: str | None = None
    fallback_reason: str = ""

    @property
    def selected_state_id(self) -> str | None:
        """Return the selected frontier state id if available."""

        return self.selected.state_id if self.selected is not None else None


class BranchTerminationReason(str, Enum):
    """Reason why a shallow local rollout ended."""

    HORIZON_REACHED = "horizon_reached"
    TERMINATED = "terminated"
    NO_ACTIONS = "no_actions"
    RESTORE_FAILED = "restore_failed"


@dataclass(slots=True)
class LocalBranchOutcome:
    """Outcome of one shallow local rollout from a restored frontier state."""

    branch_index: int
    actions_taken: list[str]
    total_reward: float
    score_change: int
    final_score: int
    final_observation: str
    terminated: bool
    termination_reason: BranchTerminationReason
    new_room_or_object_detected: bool = False
    appears_stuck: bool = False
    inventory_changed: bool = False
    new_affordance_count: int = 0
    affordance_gain: int = 0
    new_room_location_signal: bool = False
    landmark_gain: int = 0
    novel_object_count: int = 0
    room_text_only_gain: float = 0.0
    loop_penalty_reduction: float = 0.0
    branch_progress_score: float = 0.0
    branch_commit_allowed: bool = False
    commit_rejection_reason: str = ""
    oscillation_penalty_total: float = 0.0
    movement_penalty_total: float = 0.0
    movement_only_action_count: int = 0
    movement_repeat_count: int = 0
    loop_event_count: int = 0
    notable_observation_changes: list[str] = field(default_factory=list)
    final_state: TextGameState | None = None
    restore_result: ReplayResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LocalExplorationResult:
    """Structured result for shallow multi-branch local exploration."""

    base_state: TextGameState
    branch_count: int
    branch_horizon: int
    temperature: float
    action_candidate_count: int
    branches: list[LocalBranchOutcome]
    best_branch_index: int | None = None
    branch_commit_allowed: bool = False
    commit_rejection_reason: str = ""
    comparison_notes: str = ""

    @property
    def best_branch(self) -> LocalBranchOutcome | None:
        """Return the best branch outcome if one was identified."""

        if self.best_branch_index is None:
            return None
        for branch in self.branches:
            if branch.branch_index == self.best_branch_index:
                return branch
        return None

    @property
    def best_branch_progress_score(self) -> float:
        """Return the best branch progress score if available."""

        branch = self.best_branch
        return branch.branch_progress_score if branch is not None else 0.0


class ReflectionMode(str, Enum):
    """Mode used to generate compact operational reflection guidance."""

    LLM = "llm"
    FALLBACK = "fallback"


@dataclass(slots=True)
class ReflectionGuidance:
    """Normalized operational guidance extracted from local branch comparison."""

    try_actions: list[str] = field(default_factory=list)
    avoid_actions: list[str] = field(default_factory=list)
    salient_objects: list[str] = field(default_factory=list)
    discovered_affordances: list[str] = field(default_factory=list)
    supported_try_actions: list[str] = field(default_factory=list)
    supported_avoid_actions: list[str] = field(default_factory=list)
    unsupported_items_removed: list[str] = field(default_factory=list)
    short_guidance_text: str = ""

    @property
    def useful_verbs(self) -> list[str]:
        """Backward-compatible alias derived from `try_actions`."""

        verbs: list[str] = []
        for action in self.try_actions:
            verb, _obj = split_action_command(action)
            if verb and verb not in verbs:
                verbs.append(verb)
        return verbs

    @property
    def promising_directions(self) -> list[str]:
        """Backward-compatible alias for object-centric prompt consumers."""

        return list(self.salient_objects)

    @property
    def dead_ends_to_avoid(self) -> list[str]:
        """Backward-compatible alias for avoid-list consumers."""

        return list(self.avoid_actions)

    @property
    def object_affordances(self) -> list[str]:
        """Backward-compatible alias for discovered affordances."""

        return list(self.discovered_affordances)

    @property
    def compact_text(self) -> str:
        """Backward-compatible alias for compact reflection text."""

        return self.short_guidance_text


@dataclass(slots=True)
class ReflectionResult:
    """Structured reflection result suitable for reuse in later prompts."""

    mode: ReflectionMode
    guidance: ReflectionGuidance
    raw_output: str = ""
    model_name: str | None = None

    @property
    def prompt_context(self) -> str:
        """Return the compact guidance text to inject into later prompts."""

        return self.guidance.short_guidance_text

    @property
    def normalized_guidance_text(self) -> str:
        """Backward-compatible alias for compact reflection text."""

        return self.guidance.short_guidance_text


@dataclass(slots=True)
class StateCandidate:
    """One replayable state that may be worth revisiting.

    This is an engineering-oriented approximation for ranking revisit targets. It
    intentionally exposes heuristic features such as novelty and recent gain so the
    frontier policy can be tuned without obscuring the score formula.
    """

    state_id: str
    episode_id: str
    step_index: int
    observation: str
    score: float
    depth: int
    world_state_hash: str = "unknown"
    state_cluster_id: str = ""
    cluster_visit_count: int = 0
    region_novelty_score: float = 0.0
    inventory_text: str = ""
    valid_actions: list[str] = field(default_factory=list)
    replay_actions: list[str] = field(default_factory=list)
    state_family_key: str = ""
    novelty: float = 0.0
    effective_novelty: float = 0.0
    recent_gain: float = 0.0
    loop_penalty: float = 0.0
    movement_penalty: float = 0.0
    affordance_gain: int = 0
    room_text_only_gain: float = 0.0
    reversible_state_penalty: float = 0.0
    revisit_saturation_penalty: float = 0.0
    oscillating_pair_member: bool = False
    trivial_reversible_change: bool = False
    no_progress_revisit_count: int = 0
    family_no_progress_revisit_count: int = 0
    summary_text: str = ""
    saved_node: SavedNode | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dedupe_text(self) -> str:
        """Return normalized text used for near-duplicate detection."""

        return " ".join(
            part.strip().lower()
            for part in (self.summary_text or self.observation, self.inventory_text)
            if part.strip()
        )

    @classmethod
    def from_trajectory_step(
        cls,
        step: "TrajectoryStep",
        *,
        replay_actions: list[str],
        novelty: float = 0.0,
        recent_gain: float = 0.0,
        summary_text: str = "",
        saved_node: SavedNode | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StateCandidate":
        """Create a replayable state candidate from a stored trajectory step."""

        return cls(
            state_id=f"{step.episode_id}:{step.step_index}:{step.world_state_hash}",
            episode_id=step.episode_id,
            step_index=step.step_index,
            observation=step.observation,
            score=float(step.score),
            depth=step.step_index + 1,
            world_state_hash=step.world_state_hash,
            state_cluster_id=step.state_cluster_id,
            cluster_visit_count=step.cluster_visit_count,
            region_novelty_score=step.region_novelty_score,
            inventory_text=step.inventory_text,
            valid_actions=list(step.valid_actions),
            replay_actions=list(replay_actions),
            state_family_key=str(metadata.get("state_family_key", "")) if metadata else "",
            novelty=novelty,
            effective_novelty=novelty,
            recent_gain=recent_gain,
            loop_penalty=float(step.loop_penalty),
            movement_penalty=float(step.movement_penalty),
            affordance_gain=int(step.affordance_gain),
            room_text_only_gain=float(step.room_text_only_gain),
            reversible_state_penalty=float(metadata.get("reversible_state_penalty", 0.0))
            if metadata
            else 0.0,
            revisit_saturation_penalty=float(metadata.get("revisit_saturation_penalty", 0.0))
            if metadata
            else 0.0,
            oscillating_pair_member=bool(
                metadata.get("oscillating_pair_member", False)
                if metadata
                else (step.loop_detected or step.inverse_of_previous or step.repeated_pair_count > 0)
            ),
            trivial_reversible_change=bool(metadata.get("trivial_reversible_change", False))
            if metadata
            else False,
            summary_text=summary_text,
            saved_node=saved_node,
            metadata=dict(metadata or step.metadata),
        )


@dataclass(slots=True)
class FrontierEntry:
    """One ranked frontier item retained for possible replay and branching."""

    state_id: str
    score: float
    depth: int
    novelty: float = 0.0
    effective_novelty: float = 0.0
    recent_gain: float = 0.0
    loop_penalty: float = 0.0
    reversible_state_penalty: float = 0.0
    revisit_saturation_penalty: float = 0.0
    episode_id: str = "unknown"
    step_index: int = 0
    world_state_hash: str = "unknown"
    state_family_key: str = ""
    state_cluster_id: str = ""
    cluster_visit_count: int = 0
    region_novelty_score: float = 0.0
    oscillating_pair_member: bool = False
    trivial_reversible_change: bool = False
    no_progress_revisit_count: int = 0
    family_no_progress_revisit_count: int = 0
    cluster_no_progress_revisit_count: int = 0
    observation: str = ""
    inventory_text: str = ""
    valid_actions: list[str] = field(default_factory=list)
    replay_actions: list[str] = field(default_factory=list)
    movement_penalty: float = 0.0
    affordance_gain: int = 0
    room_text_only_gain: float = 0.0
    summary_text: str = ""
    saved_node: SavedNode | None = None
    priority: float = 0.0
    dedupe_key: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_candidate(
        cls,
        candidate: StateCandidate,
        *,
        priority: float = 0.0,
        dedupe_key: str = "",
        notes: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "FrontierEntry":
        """Build a frontier entry from a replayable state candidate."""

        merged_metadata = dict(candidate.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return cls(
            state_id=candidate.state_id,
            score=float(candidate.score),
            depth=int(candidate.depth),
            novelty=float(candidate.novelty),
            effective_novelty=float(candidate.effective_novelty or candidate.novelty),
            recent_gain=float(candidate.recent_gain),
            loop_penalty=float(candidate.loop_penalty),
            reversible_state_penalty=float(candidate.reversible_state_penalty),
            revisit_saturation_penalty=float(candidate.revisit_saturation_penalty),
            episode_id=candidate.episode_id,
            step_index=candidate.step_index,
            world_state_hash=candidate.world_state_hash,
            state_family_key=candidate.state_family_key,
            state_cluster_id=candidate.state_cluster_id,
            cluster_visit_count=int(candidate.cluster_visit_count),
            region_novelty_score=float(candidate.region_novelty_score),
            oscillating_pair_member=bool(candidate.oscillating_pair_member),
            trivial_reversible_change=bool(candidate.trivial_reversible_change),
            no_progress_revisit_count=int(candidate.no_progress_revisit_count),
            family_no_progress_revisit_count=int(candidate.family_no_progress_revisit_count),
            cluster_no_progress_revisit_count=int(candidate.metadata.get("cluster_no_progress_revisit_count", 0)),
            observation=candidate.observation,
            inventory_text=candidate.inventory_text,
            valid_actions=list(candidate.valid_actions),
            replay_actions=list(candidate.replay_actions),
            movement_penalty=float(candidate.movement_penalty),
            affordance_gain=int(candidate.affordance_gain),
            room_text_only_gain=float(candidate.room_text_only_gain),
            summary_text=candidate.summary_text,
            saved_node=candidate.saved_node,
            priority=float(priority),
            dedupe_key=dedupe_key,
            notes=notes,
            metadata=merged_metadata,
        )

    def to_state_candidate(self) -> StateCandidate:
        """Convert the frontier entry back into a replayable state candidate."""

        return StateCandidate(
            state_id=self.state_id,
            episode_id=self.episode_id,
            step_index=self.step_index,
            observation=self.observation,
            score=float(self.score),
            depth=int(self.depth),
            world_state_hash=self.world_state_hash,
            state_family_key=self.state_family_key,
            state_cluster_id=self.state_cluster_id,
            cluster_visit_count=int(self.cluster_visit_count),
            region_novelty_score=float(self.region_novelty_score),
            inventory_text=self.inventory_text,
            valid_actions=list(self.valid_actions),
            replay_actions=list(self.replay_actions),
            novelty=float(self.novelty),
            effective_novelty=float(self.effective_novelty or self.novelty),
            recent_gain=float(self.recent_gain),
            loop_penalty=float(self.loop_penalty),
            movement_penalty=float(self.movement_penalty),
            affordance_gain=int(self.affordance_gain),
            room_text_only_gain=float(self.room_text_only_gain),
            reversible_state_penalty=float(self.reversible_state_penalty),
            revisit_saturation_penalty=float(self.revisit_saturation_penalty),
            oscillating_pair_member=bool(self.oscillating_pair_member),
            trivial_reversible_change=bool(self.trivial_reversible_change),
            no_progress_revisit_count=int(self.no_progress_revisit_count),
            family_no_progress_revisit_count=int(self.family_no_progress_revisit_count),
            summary_text=self.summary_text,
            saved_node=self.saved_node,
            metadata={**dict(self.metadata), "cluster_no_progress_revisit_count": int(self.cluster_no_progress_revisit_count)},
        )


@dataclass(slots=True)
class TrajectoryStep:
    """One step in an episode trajectory."""

    episode_id: str
    step_index: int
    action: str
    observation: str
    reward: float
    done: bool
    score: int
    moves: int
    world_state_hash: str
    inventory_text: str = ""
    valid_actions: list[str] = field(default_factory=list)
    movement_only_action: bool = False
    movement_repeat_count: int = 0
    loop_detected: bool = False
    inverse_of_previous: bool = False
    repeated_pair_count: int = 0
    loop_penalty: float = 0.0
    movement_penalty: float = 0.0
    state_cluster_id: str = ""
    cluster_visit_count: int = 0
    region_novelty_score: float = 0.0
    room_text_only_gain: float = 0.0
    affordance_gain: int = 0
    world_state_snapshot: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert the step into a JSONL-friendly dictionary."""

        return asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "TrajectoryStep":
        """Construct a step from a stored JSON record."""

        return cls(
            episode_id=str(record["episode_id"]),
            step_index=int(record["step_index"]),
            action=str(record["action"]),
            observation=str(record["observation"]),
            reward=float(record["reward"]),
            done=bool(record["done"]),
            score=int(record["score"]),
            moves=int(record["moves"]),
            world_state_hash=str(record["world_state_hash"]),
            inventory_text=str(record.get("inventory_text", "")),
            valid_actions=_normalize_string_list(record.get("valid_actions")),
            movement_only_action=bool(record.get("movement_only_action", False)),
            movement_repeat_count=int(record.get("movement_repeat_count", 0)),
            loop_detected=bool(record.get("loop_detected", False)),
            inverse_of_previous=bool(record.get("inverse_of_previous", False)),
            repeated_pair_count=int(record.get("repeated_pair_count", 0)),
            loop_penalty=float(record.get("loop_penalty", 0.0)),
            movement_penalty=float(record.get("movement_penalty", 0.0)),
            state_cluster_id=str(record.get("state_cluster_id", "")),
            cluster_visit_count=int(record.get("cluster_visit_count", 0)),
            region_novelty_score=float(record.get("region_novelty_score", 0.0)),
            room_text_only_gain=float(record.get("room_text_only_gain", 0.0)),
            affordance_gain=int(record.get("affordance_gain", 0)),
            world_state_snapshot=normalize_world_state_snapshot_record(record["world_state_snapshot"])
            if isinstance(record.get("world_state_snapshot"), MappingABC)
            else None,
            metadata=dict(record["metadata"]) if isinstance(record.get("metadata"), MappingABC) else {},
        )


@dataclass(slots=True)
class Trajectory:
    """A full episode trajectory or a replayable sub-trajectory prefix."""

    episode_id: str
    steps: list[TrajectoryStep]
    source_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_records(self) -> list[dict[str, Any]]:
        """Convert the full trajectory into JSONL-friendly records."""

        return [step.to_record() for step in self.steps]

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]],
        *,
        source_path: Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Trajectory":
        """Construct a trajectory from stored step records."""

        steps = [TrajectoryStep.from_record(record) for record in records]
        episode_id = steps[0].episode_id if steps else "unknown"
        trajectory = cls(
            episode_id=episode_id,
            steps=steps,
            source_path=source_path,
            metadata=dict(metadata or {}),
        )
        trajectory.validate_integrity()
        return trajectory

    def validate_integrity(self) -> None:
        """Validate core trajectory invariants used by replay and persistence."""

        if not self.steps:
            return

        episode_ids = {step.episode_id for step in self.steps}
        if len(episode_ids) != 1:
            raise ValueError(f"Trajectory contains multiple episode ids: {sorted(episode_ids)}")
        only_episode_id = next(iter(episode_ids))
        if self.episode_id not in {"unknown", only_episode_id}:
            raise ValueError(
                f"Trajectory episode_id {self.episode_id!r} does not match step episode_id {only_episode_id!r}."
            )

        step_indices = [step.step_index for step in self.steps]
        if step_indices != sorted(step_indices):
            raise ValueError("Trajectory steps must be sorted by step_index.")
        if len(step_indices) != len(set(step_indices)):
            raise ValueError("Trajectory contains duplicate step_index values.")

    def prefix(self, up_to_step_index: int) -> "Trajectory":
        """Return a replayable prefix ending at the requested step index."""

        if up_to_step_index < 0:
            raise IndexError(f"Trajectory step index out of range: {up_to_step_index}")
        return Trajectory(
            episode_id=self.episode_id,
            steps=[step for step in self.steps if step.step_index <= up_to_step_index],
            source_path=self.source_path,
            metadata=dict(self.metadata),
        )

    def replay_actions(self, up_to_step_index: int | None = None) -> list[str]:
        """Return replay actions for the full trajectory or a prefix."""

        relevant_steps = self.steps if up_to_step_index is None else self.prefix(up_to_step_index).steps
        return [step.action for step in relevant_steps]

    def total_reward(self) -> float:
        """Return the trajectory's total reward."""

        return sum(step.reward for step in self.steps)

    def final_score(self) -> int:
        """Return the final score from the last recorded step."""

        return self.steps[-1].score if self.steps else 0

    def final_step(self) -> TrajectoryStep | None:
        """Return the last trajectory step if available."""

        return self.steps[-1] if self.steps else None


@dataclass(slots=True)
class EpisodeResult:
    """Summary of a completed episode."""

    episode_id: str
    seed: int
    total_reward: float
    step_count: int
    final_score: int
    trajectory_path: Path
    summary_path: Path | None = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert the episode summary into a serializable dictionary."""

        payload = asdict(self)
        payload["trajectory_path"] = str(self.trajectory_path)
        payload["summary_path"] = str(self.summary_path) if self.summary_path is not None else None
        return payload


@dataclass(slots=True)
class BatchEvaluationSummary:
    """Aggregate summary for a multi-seed evaluation run."""

    episode_count: int
    seed_values: list[int]
    mean_reward: float
    std_reward: float
    mean_steps: float
    std_steps: float
    mean_final_score: float
    std_final_score: float
    trajectory_paths: list[Path] = field(default_factory=list)
    summary_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert the batch summary into a serializable dictionary."""

        payload = asdict(self)
        payload["trajectory_paths"] = [str(path) for path in self.trajectory_paths]
        payload["summary_path"] = str(self.summary_path) if self.summary_path is not None else None
        return payload


EpisodeSummary = EpisodeResult
FrontierState = SavedNode
