"""Shared typed data containers for the GLoW implementation.

TODO: introduce stricter event schemas once the agent loop stabilizes.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence, TypedDict


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
    response_format: dict[str, Any] | None = None
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
        response_format: Mapping[str, Any] | None = None,
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
            response_format=dict(response_format) if response_format is not None else None,
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
    environments. When those internals are unavailable, the runtime falls back to an
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
_INSPECT_VERBS = {"examine", "inspect", "look", "look at", "read"}
_ACQUIRE_VERBS = {"take", "get"}
_ACCESS_VERBS = {"open", "close", "unlock", "look in", "look inside"}
_USE_VERBS = {"put", "insert", "use", "light", "extinguish", "unlock with", "open with"}
_AGGRESSIVE_VERBS = {"throw", "attack", "break", "kick", "hit"}
_READABLE_OBJECT_TOKENS = {"book", "card", "code", "diary", "document", "inscription", "label", "leaflet", "letter", "manual", "map", "note", "page", "paper", "plaque", "runes", "scroll", "sign", "tablet"}
_OPENABLE_OBJECT_TOKENS = {"bag", "box", "cabinet", "case", "chest", "cupboard", "door", "drawer", "egg", "gate", "hatch", "mailbox", "nest", "package", "safe", "trapdoor", "trunk", "window"}
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
_BULK_INVENTORY_ACTION_PREFIXES = (
    "take all",
    "take everything",
    "get all",
    "get everything",
    "drop all",
    "drop everything",
    "put down all",
    "put down everything",
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
    "house",
    "field",
    "forest",
    "tree",
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

_SECONDARY_STATE_CLUSTER_TOKENS = (
    "window",
    "door",
    "mailbox",
    "leaflet",
    "egg",
    "nest",
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


def default_verb_family_prior_scores() -> dict[str, float]:
    """Return the default IF-oriented prior score per verb family."""

    return {
        "inspect": 1.2,
        "acquire": 1.0,
        "access": 0.7,
        "movement": 0.0,
        "use": -0.2,
        "aggressive": -1.25,
        "other": 0.0,
    }


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


def canonical_if_verb_family(
    action: str,
    inverse_pairs: Mapping[str, str] | None = None,
) -> str:
    """Return a lightweight IF verb-family label for one parser action."""

    verb, _obj = split_action_command(action, inverse_pairs)
    if verb in _INSPECT_VERBS:
        return "inspect"
    if verb in _ACQUIRE_VERBS:
        return "acquire"
    if verb in _ACCESS_VERBS:
        return "access"
    if is_movement_action(action, inverse_pairs):
        return "movement"
    if verb in _USE_VERBS or any(token in normalize_parser_action(action) for token in (" with ", " on ", " into ", " in ")):
        return "use"
    if verb in _AGGRESSIVE_VERBS:
        return "aggressive"
    return "other"


def action_object_count(
    action: str,
    inverse_pairs: Mapping[str, str] | None = None,
) -> int:
    """Return a lightweight count of object-like arguments in one action."""

    normalized = normalize_parser_action(action)
    if not normalized:
        return 0
    if canonical_if_verb_family(action, inverse_pairs) == "movement":
        return 0
    object_tokens = action_target_tokens(normalized, inverse_pairs)
    if not object_tokens:
        return 0
    if any(separator in normalized for separator in (" at ", " with ", " into ", " on ")):
        return 2
    if normalized.startswith("put ") and " in " in normalized:
        return 2
    return 1


def action_uses_inventory_object(
    action: str,
    inventory_tokens: set[str],
    inverse_pairs: Mapping[str, str] | None = None,
) -> bool:
    """Return whether an action appears to use a currently held object."""

    if not inventory_tokens:
        return False
    return bool(action_target_tokens(action, inverse_pairs) & inventory_tokens)


def is_bulk_inventory_action(action: str) -> bool:
    """Return whether an action is a generic bulk inventory command."""

    normalized = normalize_parser_action(action)
    if not normalized:
        return False
    if normalized in _BULK_INVENTORY_ACTION_PREFIXES:
        return True
    if any(normalized.startswith(prefix + " ") for prefix in _BULK_INVENTORY_ACTION_PREFIXES):
        return True
    return normalized.endswith(" all") or normalized.endswith(" everything")


def is_discard_like_action(action: str) -> bool:
    """Return whether an action primarily discards or sheds inventory/state."""

    normalized = normalize_parser_action(action)
    return (
        normalized.startswith("drop ")
        or normalized.startswith("put ")
        or normalized.startswith("insert ")
        or normalized.startswith("throw ")
        or normalized.startswith("put down ")
    )


def target_is_container_or_openable_candidate(
    action: str,
    valid_actions: list[str] | None = None,
    inverse_pairs: Mapping[str, str] | None = None,
) -> bool:
    """Return whether an action appears to target an openable/container-like object."""

    target_tokens = action_target_tokens(action, inverse_pairs)
    if target_tokens & _OPENABLE_OBJECT_TOKENS:
        return True
    normalized_targets = {normalize_parser_action(target) for target in target_tokens}
    for valid_action in valid_actions or []:
        verb, _obj = split_action_command(valid_action, inverse_pairs)
        if verb not in {"open", "close", "look in", "look inside", "unlock"}:
            continue
        if action_target_tokens(valid_action, inverse_pairs) & normalized_targets:
            return True
    return False


def target_is_readable_candidate(
    action: str,
    *,
    observation: str,
    inventory_text: str = "",
    valid_actions: list[str] | None = None,
    inverse_pairs: Mapping[str, str] | None = None,
) -> bool:
    """Return whether an action appears to target a readable object."""

    target_tokens = action_target_tokens(action, inverse_pairs)
    if target_tokens & _READABLE_OBJECT_TOKENS:
        return True
    salient_tokens = extract_salient_nouns(
        observation=observation,
        inventory_text=inventory_text,
        valid_actions=valid_actions or [],
        inverse_pairs=inverse_pairs,
    )
    readable_salient_tokens = salient_tokens & _READABLE_OBJECT_TOKENS
    if target_tokens & readable_salient_tokens:
        return True
    for valid_action in valid_actions or []:
        verb, _obj = split_action_command(valid_action, inverse_pairs)
        if verb == "read" and action_target_tokens(valid_action, inverse_pairs) & target_tokens:
            return True
    return False


def action_shape_is_simple(
    action: str,
    inverse_pairs: Mapping[str, str] | None = None,
) -> bool:
    """Return whether an action has a short canonical IF shape."""

    normalized = normalize_parser_action(action)
    if not normalized:
        return False
    family = canonical_if_verb_family(normalized, inverse_pairs)
    if family == "movement":
        return True
    if family == "inspect" and (normalized.startswith("look at ") or normalized.startswith("look in ")):
        return True
    return action_object_count(normalized, inverse_pairs) <= 1 and not action_shape_is_complex_transitive(
        normalized,
        inverse_pairs,
    )


def action_shape_is_complex_transitive(
    action: str,
    inverse_pairs: Mapping[str, str] | None = None,
) -> bool:
    """Return whether an action is a speculative multi-argument transitive command."""

    normalized = normalize_parser_action(action)
    if not normalized:
        return False
    family = canonical_if_verb_family(normalized, inverse_pairs)
    if family == "inspect" and any(normalized.startswith(prefix) for prefix in ("look at ", "look in ", "look inside ")):
        return False
    if action_object_count(normalized, inverse_pairs) >= 2:
        return True
    return any(separator in normalized for separator in (" with ", " at ", " on ", " into ")) and family in {
        "use",
        "aggressive",
        "other",
    }


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


def inventory_item_tokens(
    inventory_text: str,
    inverse_pairs: Mapping[str, str] | None = None,
) -> set[str]:
    """Extract lightweight inventory item tokens, ignoring empty-inventory boilerplate."""

    normalized_inventory = normalize_parser_action(inventory_text)
    if not normalized_inventory:
        return set()
    if any(
        marker in normalized_inventory
        for marker in ("inventory empty", "empty-handed", "empty handed", "inventory unavailable")
    ):
        return set()
    return extract_salient_nouns(
        observation="",
        inventory_text=inventory_text,
        valid_actions=[],
        inverse_pairs=inverse_pairs,
    )


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

    title_cluster = _cluster_id_from_room_title(observation)
    if title_cluster:
        return title_cluster

    observation_tokens = _observation_cluster_tokens(observation)
    for token in _STATE_CLUSTER_PRIORITY_TOKENS:
        if token in observation_tokens:
            return f"region:{token}"

    summary_tokens = _observation_cluster_tokens(summary_text)
    for token in _STATE_CLUSTER_PRIORITY_TOKENS:
        if token in summary_tokens:
            return f"region:{token}"

    valid_action_targets = {
        token
        for action in (valid_actions or [])
        if not is_movement_action(action, inverse_pairs)
        for token in action_target_tokens(action, inverse_pairs)
    }
    for token in _SECONDARY_STATE_CLUSTER_TOKENS:
        if token in valid_action_targets:
            return f"region:{token}"

    inventory_tokens = inventory_item_tokens(inventory_text, inverse_pairs)
    for token in _SECONDARY_STATE_CLUSTER_TOKENS:
        if token in inventory_tokens and token not in {"leaflet", "lamp"}:
            return f"region:{token}"

    stable_tokens = sorted((observation_tokens | summary_tokens) - _ROOM_TEXT_ONLY_TOKENS)
    if stable_tokens:
        return "region:" + "|".join(stable_tokens[:3])
    if summary_text.strip():
        return "region:summary:" + normalize_parser_action(summary_text)[:48]
    return "region:observation:" + normalize_parser_action(observation)[:48]


def _cluster_id_from_room_title(observation: str) -> str:
    """Extract a stable cluster id from a likely room-title line when present."""

    lines = [line.strip() for line in observation.splitlines() if line.strip()]
    if not lines:
        return ""
    raw_title = lines[0]
    title = normalize_parser_action(raw_title)
    if not title:
        return ""
    title_words = title.split()
    if len(title_words) > 4:
        return ""
    if raw_title.endswith((".", "!", "?")):
        return ""
    if any(
        title.startswith(prefix)
        for prefix in (
            "you are",
            "there is",
            "this is",
            "it is",
            "from here",
            "a path",
            "the path",
        )
    ):
        return ""
    if any(
        token in title_words
        for token in ("prevents", "blocked", "blocks", "cannot", "can't", "impassable", "undergrowth")
    ):
        return ""
    if "|" in title or ":" in title:
        return ""
    return "region:title:" + title.replace(" ", "-")[:48]


def _observation_cluster_tokens(text: str) -> set[str]:
    """Extract location-oriented tokens for region clustering.

    This intentionally biases toward room/region landmarks rather than local object nouns,
    so map memory stays anchored to places instead of transient affordances.
    """

    normalized = normalize_parser_action(text)
    if not normalized:
        return set()

    tokens: set[str] = set()
    if "white house" in normalized or "house" in normalized:
        tokens.add("house")
    if "forest" in normalized:
        tokens.add("forest")
    if "field" in normalized:
        tokens.add("field")
    if "path" in normalized or "trail" in normalized:
        tokens.add("path")
    if "stream" in normalized or "river" in normalized:
        tokens.add("stream")
    if "tree" in normalized:
        tokens.add("tree")
    if "kitchen" in normalized:
        tokens.add("kitchen")
    if "living room" in normalized or "living-room" in normalized:
        tokens.add("living")
    if "attic" in normalized:
        tokens.add("attic")
    if "cellar" in normalized or "basement" in normalized:
        tokens.add("cellar")
    if "canyon" in normalized:
        tokens.add("canyon")
    if "valley" in normalized:
        tokens.add("valley")
    if "grating" in normalized:
        tokens.add("grating")
    if "clearing" in normalized:
        tokens.add("clearing")
    if "gallery" in normalized:
        tokens.add("gallery")
    if "studio" in normalized:
        tokens.add("studio")

    if tokens:
        return tokens

    return {
        token
        for token in extract_salient_nouns(
            observation=text,
            inventory_text="",
            valid_actions=[],
        )
        if token in _STATE_CLUSTER_PRIORITY_TOKENS
    }


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

    if score_gain > 0:
        return 1.0
    if inventory_changed:
        return 0.85
    if affordance_gain > 0 and (novel_object_count > 0 or materially_new_actions):
        return 0.7
    if affordance_gain > 0 and cluster_visit_count <= 1 and not observation_seen_before:
        return 0.45
    if novel_object_count > 0 or materially_new_actions:
        return 0.45
    if cluster_visit_count <= 1 and not observation_seen_before:
        return 0.25
    if movement_only_action:
        if cluster_visit_count <= 1 and not observation_seen_before:
            return 0.15
        if cluster_visit_count == 2:
            return 0.05
        return 0.02
    if cluster_visit_count == 2:
        return 0.08 if not observation_seen_before else 0.04
    return 0.05 if not observation_seen_before else 0.02


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
    inventory_gain_count: int = 0
    inventory_loss_count: int = 0
    observation_change_count: int = 0
    valid_actions_change_count: int = 0
    valid_actions_improvement_count: int = 0
    revealed_object_count: int = 0
    bulk_inventory_count: int = 0
    discard_like_count: int = 0
    durable_gain_count: int = 0
    no_gain_count: int = 0
    no_durable_gain_count: int = 0

    @property
    def had_any_gain(self) -> bool:
        """Return whether the action ever produced an observable gain signal."""

        return self.durable_gain_count > 0

    def record_attempt(
        self,
        *,
        score_changed: bool,
        inventory_changed: bool,
        inventory_gained: bool = False,
        inventory_lost: bool = False,
        observation_changed: bool,
        valid_actions_changed: bool,
        valid_actions_improved: bool = False,
        revealed_new_object: bool = False,
        bulk_inventory_action: bool = False,
        discard_like_action: bool = False,
        durable_progress_override: bool | None = None,
    ) -> None:
        """Accumulate one observed action outcome."""

        self.attempts += 1
        if score_changed:
            self.score_change_count += 1
        if inventory_changed:
            self.inventory_change_count += 1
        if inventory_gained:
            self.inventory_gain_count += 1
        if inventory_lost:
            self.inventory_loss_count += 1
        if observation_changed:
            self.observation_change_count += 1
        if valid_actions_changed:
            self.valid_actions_change_count += 1
        if valid_actions_improved:
            self.valid_actions_improvement_count += 1
        if revealed_new_object:
            self.revealed_object_count += 1
        if bulk_inventory_action:
            self.bulk_inventory_count += 1
        if discard_like_action:
            self.discard_like_count += 1
        if not any((score_changed, inventory_changed, observation_changed, valid_actions_changed)):
            self.no_gain_count += 1
        durable_progress = (
            durable_progress_override
            if durable_progress_override is not None
            else any(
                (
                    score_changed,
                    inventory_gained and not bulk_inventory_action and not discard_like_action,
                )
            )
        )
        if durable_progress:
            self.durable_gain_count += 1
        elif not (valid_actions_improved or revealed_new_object):
            self.no_durable_gain_count += 1


@dataclass(slots=True)
class ActionClusterHistory:
    """Recent action memory for the current local state cluster.

    This is a pragmatic approximation. A cluster is the current local region of
    play between resets/restores, not a formal game-state equivalence class.
    """

    cluster_label: str = ""
    seen_nouns: set[str] = field(default_factory=set)
    action_stats: dict[str, ActionAttemptStats] = field(default_factory=dict)
    verb_family_stats: dict[str, ActionAttemptStats] = field(default_factory=dict)
    object_noun_stats: dict[str, ActionAttemptStats] = field(default_factory=dict)
    current_state_cluster_id: str = ""
    cluster_visit_counts: dict[str, int] = field(default_factory=dict)
    cluster_observation_signatures: dict[str, set[str]] = field(default_factory=dict)
    object_family_no_progress_counts: dict[str, int] = field(default_factory=dict)
    recent_affordance_targets: set[str] = field(default_factory=set)
    current_visible_nouns: set[str] = field(default_factory=set)
    fresh_visible_nouns: set[str] = field(default_factory=set)
    current_valid_actions: set[str] = field(default_factory=set)
    fresh_valid_actions: set[str] = field(default_factory=set)
    fresh_exit_actions: set[str] = field(default_factory=set)
    recent_cluster_sequence: list[str] = field(default_factory=list)
    no_progress_steps: int = 0
    movement_no_progress_steps: int = 0
    bulk_inventory_no_progress_steps: int = 0

    def stats_for(self, action: str) -> ActionAttemptStats:
        """Return mutable stats for an action within this cluster."""

        normalized = normalize_parser_action(action)
        if normalized not in self.action_stats:
            self.action_stats[normalized] = ActionAttemptStats(action=normalized)
        return self.action_stats[normalized]

    def stats_for_verb_family(self, family: str) -> ActionAttemptStats:
        """Return mutable stats for a verb family within this cluster."""

        normalized = normalize_parser_action(family) or "other"
        if normalized not in self.verb_family_stats:
            self.verb_family_stats[normalized] = ActionAttemptStats(action=normalized)
        return self.verb_family_stats[normalized]

    def stats_for_object_noun(self, noun: str) -> ActionAttemptStats:
        """Return mutable stats for a local object noun within this cluster."""

        normalized = normalize_parser_action(noun)
        if not normalized:
            normalized = "unknown"
        if normalized not in self.object_noun_stats:
            self.object_noun_stats[normalized] = ActionAttemptStats(action=normalized)
        return self.object_noun_stats[normalized]

    def record_attempt(
        self,
        *,
        action: str,
        score_changed: bool,
        inventory_changed: bool,
        inventory_gained: bool = False,
        inventory_lost: bool = False,
        observation_changed: bool,
        valid_actions_changed: bool,
        valid_actions_improved: bool = False,
        revealed_new_object: bool = False,
        target_tokens: set[str] | None = None,
        movement_only_action: bool = False,
        inverse_pairs: Mapping[str, str] | None = None,
        bulk_inventory_action: bool = False,
        discard_like_action: bool = False,
        revealed_object_tokens: set[str] | None = None,
    ) -> None:
        """Record one attempted action and its observed effect."""

        verb_family = canonical_if_verb_family(action, inverse_pairs)
        durable_progress = any(
            (
                score_changed,
                inventory_gained and not bulk_inventory_action and not discard_like_action,
            )
        )
        exact_stats = self.stats_for(action)
        exact_stats.record_attempt(
            score_changed=score_changed,
            inventory_changed=inventory_changed,
            inventory_gained=inventory_gained,
            inventory_lost=inventory_lost,
            observation_changed=observation_changed,
            valid_actions_changed=valid_actions_changed,
            valid_actions_improved=valid_actions_improved,
            revealed_new_object=revealed_new_object,
            bulk_inventory_action=bulk_inventory_action,
            discard_like_action=discard_like_action,
            durable_progress_override=durable_progress,
        )
        self.stats_for_verb_family(verb_family).record_attempt(
            score_changed=score_changed,
            inventory_changed=inventory_changed,
            inventory_gained=inventory_gained,
            inventory_lost=inventory_lost,
            observation_changed=observation_changed,
            valid_actions_changed=valid_actions_changed,
            valid_actions_improved=valid_actions_improved,
            revealed_new_object=revealed_new_object,
            bulk_inventory_action=bulk_inventory_action,
            discard_like_action=discard_like_action,
            durable_progress_override=durable_progress,
        )
        normalized_targets = {normalize_parser_action(token) for token in (target_tokens or set()) if normalize_parser_action(token)}
        normalized_revealed_targets = {
            normalize_parser_action(token)
            for token in (revealed_object_tokens or set())
            if normalize_parser_action(token)
        }
        for token in normalized_targets:
            self.stats_for_object_noun(token).record_attempt(
                score_changed=score_changed,
                inventory_changed=inventory_changed,
                inventory_gained=inventory_gained,
                inventory_lost=inventory_lost,
                observation_changed=observation_changed,
                valid_actions_changed=valid_actions_changed,
                valid_actions_improved=valid_actions_improved,
                revealed_new_object=revealed_new_object,
                bulk_inventory_action=bulk_inventory_action,
                discard_like_action=discard_like_action,
                durable_progress_override=durable_progress,
            )
        affordance_targets = set(normalized_targets) | set(normalized_revealed_targets)
        if valid_actions_improved or revealed_new_object:
            self.recent_affordance_targets = set(affordance_targets)
        elif affordance_targets & self.recent_affordance_targets:
            # Once we have attempted a follow-up on the freshly revealed object family,
            # stop treating it as "just revealed" for subsequent ranking steps.
            self.recent_affordance_targets.difference_update(affordance_targets)
        if inventory_gained and affordance_targets:
            # Carrying the object means the immediate local reveal has already been cashed in.
            self.recent_affordance_targets.difference_update(affordance_targets)
        if durable_progress:
            self.no_progress_steps = 0
            if movement_only_action:
                self.movement_no_progress_steps = 0
            self.bulk_inventory_no_progress_steps = 0
            for token in normalized_targets:
                self.object_family_no_progress_counts.pop(token, None)
        else:
            self.no_progress_steps += 1
            if movement_only_action:
                self.movement_no_progress_steps += 1
            if bulk_inventory_action:
                self.bulk_inventory_no_progress_steps += 1
            for token in normalized_targets:
                self.object_family_no_progress_counts[token] = self.object_family_no_progress_counts.get(token, 0) + 1

    def observe_state_nouns(
        self,
        *,
        observation: str,
        inventory_text: str = "",
        valid_actions: list[str] | None = None,
        inverse_pairs: Mapping[str, str] | None = None,
    ) -> None:
        """Remember nouns that have already been surfaced in this local cluster."""

        visible_nouns = extract_salient_nouns(
            observation=observation,
            inventory_text=inventory_text,
            valid_actions=[],
            inverse_pairs=inverse_pairs,
        )
        memory_nouns = extract_salient_nouns(
            observation=observation,
            inventory_text=inventory_text,
            valid_actions=valid_actions,
            inverse_pairs=inverse_pairs,
        )
        self.current_visible_nouns = set(visible_nouns)
        self.fresh_visible_nouns = visible_nouns - self.seen_nouns
        self.seen_nouns.update(memory_nouns)
        normalized_valid_actions = {
            normalize_parser_action(action)
            for action in (valid_actions or [])
            if normalize_parser_action(action)
        }
        if self.current_valid_actions:
            self.fresh_valid_actions = normalized_valid_actions - self.current_valid_actions
        else:
            self.fresh_valid_actions = set()
        self.fresh_exit_actions = {
            action
            for action in self.fresh_valid_actions
            if is_movement_action(action, inverse_pairs)
        }
        self.current_valid_actions = normalized_valid_actions

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

    def family_no_progress_count(self, token: str) -> int:
        """Return the no-progress count for a local object/action family token."""

        normalized = normalize_parser_action(token)
        if not normalized:
            return 0
        return self.object_family_no_progress_counts.get(normalized, 0)

    def exhausted_families(self, threshold: int) -> set[str]:
        """Return local object families that have exceeded the no-progress threshold."""

        if threshold <= 0:
            return set()
        return {
            family
            for family, count in self.object_family_no_progress_counts.items()
            if count >= threshold
        }

    def touches_recent_affordance_targets(self, targets: set[str]) -> bool:
        """Return whether a candidate touches an object family with freshly improved affordances."""

        normalized_targets = {
            normalize_parser_action(token)
            for token in targets
            if normalize_parser_action(token)
        }
        return bool(normalized_targets & self.recent_affordance_targets)

    def max_family_no_progress_count(self, tokens: set[str]) -> int:
        """Return the maximum no-progress count among the provided family tokens."""

        if not tokens:
            return 0
        return max(self.family_no_progress_count(token) for token in tokens)

    def prior_success_for_verb_family(self, family: str) -> int:
        """Return durable-success count for one verb family."""

        return self.verb_family_stats.get(normalize_parser_action(family), ActionAttemptStats(action="")).durable_gain_count

    def prior_score_success_for_verb_family(self, family: str) -> int:
        """Return score-gain count for one verb family."""

        return self.verb_family_stats.get(normalize_parser_action(family), ActionAttemptStats(action="")).score_change_count

    def prior_failure_for_verb_family(self, family: str) -> int:
        """Return durable-failure count for one verb family."""

        return self.verb_family_stats.get(normalize_parser_action(family), ActionAttemptStats(action="")).no_durable_gain_count

    def prior_success_for_object_nouns(self, nouns: set[str]) -> int:
        """Return the highest durable-success count among the provided object nouns."""

        if not nouns:
            return 0
        return max(
            self.object_noun_stats.get(normalize_parser_action(noun), ActionAttemptStats(action="")).durable_gain_count
            for noun in nouns
        )

    def prior_score_success_for_object_nouns(self, nouns: set[str]) -> int:
        """Return the highest score-gain count among the provided object nouns."""

        if not nouns:
            return 0
        return max(
            self.object_noun_stats.get(normalize_parser_action(noun), ActionAttemptStats(action="")).score_change_count
            for noun in nouns
        )

    def prior_failure_for_object_nouns(self, nouns: set[str]) -> int:
        """Return the highest durable-failure count among the provided object nouns."""

        if not nouns:
            return 0
        return max(
            self.object_noun_stats.get(normalize_parser_action(noun), ActionAttemptStats(action="")).no_durable_gain_count
            for noun in nouns
        )

    def prior_inventory_gain_for_object_nouns(self, nouns: set[str]) -> int:
        """Return the highest inventory-gain count among the provided object nouns."""

        if not nouns:
            return 0
        return max(
            self.object_noun_stats.get(normalize_parser_action(noun), ActionAttemptStats(action="")).inventory_gain_count
            for noun in nouns
        )

    def prior_inventory_loss_for_object_nouns(self, nouns: set[str]) -> int:
        """Return the highest inventory-loss count among the provided object nouns."""

        if not nouns:
            return 0
        return max(
            self.object_noun_stats.get(normalize_parser_action(noun), ActionAttemptStats(action="")).inventory_loss_count
            for noun in nouns
        )

    def prior_affordance_gain_for_object_nouns(self, nouns: set[str]) -> int:
        """Return the highest affordance-gain count among the provided object nouns."""

        if not nouns:
            return 0
        return max(
            (
                self.object_noun_stats.get(normalize_parser_action(noun), ActionAttemptStats(action="")).valid_actions_improvement_count
                + self.object_noun_stats.get(normalize_parser_action(noun), ActionAttemptStats(action="")).revealed_object_count
            )
            for noun in nouns
        )


class StrategicMode(str, Enum):
    """High-level episode mode for balancing mapping and known opportunity exploitation."""

    EXPLORE = "explore"
    EXPLOIT = "exploit"


class RegionOpportunityKind(str, Enum):
    """Lightweight unresolved opportunity types attached to one region cluster."""

    STRUCTURAL_ACCESS = "structural_access"
    OBJECT_FOLLOWUP = "object_followup"
    UNEXPLORED_EXIT = "unexplored_exit"


@dataclass(slots=True)
class RegionOpportunity:
    """One unresolved region-level opportunity that can bias replay and action ranking."""

    opportunity_id: str
    cluster_id: str
    kind: RegionOpportunityKind
    token: str
    action_hints: list[str] = field(default_factory=list)
    discovered_step: int = 0
    last_seen_step: int = 0
    base_priority: float = 0.0
    attempt_count: int = 0
    no_progress_attempt_count: int = 0
    durable_progress_count: int = 0
    resolved: bool = False
    resolution_reason: str = ""

    def current_priority(self) -> float:
        """Return the current strategic value of this opportunity."""

        if self.resolved:
            return 0.0
        return max(
            0.0,
            self.base_priority
            + 0.4 * float(self.durable_progress_count)
            - 0.75 * float(self.no_progress_attempt_count),
        )


@dataclass(slots=True)
class RegionNodeMemory:
    """Aggregated map memory for one coarse region/state cluster."""

    cluster_id: str
    summary_text: str = ""
    first_seen_step: int = 0
    last_seen_step: int = 0
    visit_count: int = 0
    durable_progress_count: int = 0
    seen_objects: set[str] = field(default_factory=set)
    exit_actions: set[str] = field(default_factory=set)
    explored_exit_actions: set[str] = field(default_factory=set)
    opportunity_ids: set[str] = field(default_factory=set)

    def unresolved_opportunity_ids(
        self,
        opportunities: Mapping[str, "RegionOpportunity"],
    ) -> list[str]:
        """Return unresolved opportunity ids linked to this region."""

        return [
            opportunity_id
            for opportunity_id in self.opportunity_ids
            if opportunity_id in opportunities and not opportunities[opportunity_id].resolved
        ]


@dataclass(slots=True)
class StrategicGuidance:
    """Compact high-level guidance for the current episode step."""

    mode: StrategicMode
    reason: str
    preferred_cluster_id: str = ""
    try_actions: list[str] = field(default_factory=list)
    avoid_actions: list[str] = field(default_factory=list)
    salient_objects: list[str] = field(default_factory=list)
    opportunity_labels: list[str] = field(default_factory=list)


_STRUCTURAL_ACCESS_TOKENS = {
    "door",
    "gate",
    "grating",
    "house",
    "ladder",
    "stairs",
    "trapdoor",
    "window",
}

_OBJECT_FOLLOWUP_VERBS = ("examine", "look at", "look in", "open", "read", "take", "get", "enter")


@dataclass(slots=True)
class EpisodeMapMemory:
    """Lightweight region graph and unresolved-opportunity memory for one episode."""

    nodes: dict[str, RegionNodeMemory] = field(default_factory=dict)
    opportunities: dict[str, RegionOpportunity] = field(default_factory=dict)
    recent_cluster_sequence: list[str] = field(default_factory=list)

    def record_state(self, state: "TextGameState", *, step_index: int) -> RegionNodeMemory:
        """Record one visited state and infer/update region-level opportunities."""

        cluster_id = normalize_parser_action(state.state_cluster_id or state.world_state_hash) or "region:unknown"
        node = self.nodes.get(cluster_id)
        if node is None:
            node = RegionNodeMemory(
                cluster_id=cluster_id,
                summary_text=str(state.metadata.get("summary_text", "")),
                first_seen_step=step_index,
            )
            self.nodes[cluster_id] = node
        node.visit_count += 1
        node.last_seen_step = step_index
        if state.metadata.get("summary_text"):
            node.summary_text = str(state.metadata["summary_text"])
        node.seen_objects.update(
            extract_salient_nouns(
                observation=state.observation,
                inventory_text=state.inventory_text,
                valid_actions=[],
            )
        )
        node.exit_actions.update(
            normalize_parser_action(action)
            for action in state.valid_actions
            if is_movement_action(action)
        )
        self.recent_cluster_sequence.append(cluster_id)
        self.recent_cluster_sequence = self.recent_cluster_sequence[-32:]
        self._infer_opportunities(node=node, state=state, step_index=step_index)
        return node

    def record_transition(
        self,
        *,
        previous_state: "TextGameState",
        action: str,
        current_state: "TextGameState",
        step_index: int,
        score_gain: int,
        inventory_gain_count: int,
        affordance_gain: int,
        novel_object_count: int,
        materially_new_actions: bool,
    ) -> None:
        """Update opportunity success/failure counts after one real episode transition."""

        previous_cluster_id = normalize_parser_action(previous_state.state_cluster_id or previous_state.world_state_hash)
        if not previous_cluster_id:
            return
        node = self.nodes.setdefault(previous_cluster_id, RegionNodeMemory(cluster_id=previous_cluster_id))
        normalized_action = normalize_parser_action(action)
        # Region "harvest" should only advance on durable payoffs, not on every local
        # affordance reveal. For example, opening the mailbox should not immediately
        # mark the mailbox region harvested before the agent has actually taken the leaflet.
        durable_progress = any(
            (
                score_gain > 0,
                inventory_gain_count > 0,
            )
        )
        if durable_progress:
            node.durable_progress_count += 1

        if is_movement_action(action):
            node.explored_exit_actions.add(normalized_action)

        action_targets = action_target_tokens(action)
        verb, _obj = split_action_command(action)
        current_cluster_id = normalize_parser_action(current_state.state_cluster_id or current_state.world_state_hash)

        for opportunity_id in node.unresolved_opportunity_ids(self.opportunities):
            opportunity = self.opportunities[opportunity_id]
            if not self._opportunity_matches_action(
                opportunity=opportunity,
                action=normalized_action,
                verb=verb,
                action_targets=action_targets,
            ):
                continue
            opportunity.attempt_count += 1
            opportunity.last_seen_step = step_index
            if durable_progress:
                opportunity.durable_progress_count += 1
                if self._opportunity_is_resolved(
                    opportunity=opportunity,
                    previous_state=previous_state,
                    current_state=current_state,
                    action=normalized_action,
                    score_gain=score_gain,
                    inventory_gain_count=inventory_gain_count,
                    affordance_gain=affordance_gain,
                ):
                    opportunity.resolved = True
                    opportunity.resolution_reason = "durable progress observed"
            else:
                opportunity.no_progress_attempt_count += 1
                if (
                    opportunity.kind is RegionOpportunityKind.OBJECT_FOLLOWUP
                    and (
                        opportunity.no_progress_attempt_count >= 3
                        or (
                            node.durable_progress_count > 0
                            and opportunity.no_progress_attempt_count >= 1
                        )
                    )
                ):
                    opportunity.resolved = True
                    opportunity.resolution_reason = "repeated no-progress follow-up"

        if is_movement_action(action) and current_cluster_id and current_cluster_id != previous_cluster_id:
            opportunity_id = f"exit:{previous_cluster_id}:{normalized_action}"
            if opportunity_id in self.opportunities:
                self.opportunities[opportunity_id].resolved = True
                self.opportunities[opportunity_id].resolution_reason = "exit traversed"

    def recommend_guidance(
        self,
        *,
        current_state: "TextGameState",
        frontier_entries: Sequence["FrontierEntry"] | None = None,
        cluster_visit_exhaustion_threshold: int = 4,
        min_opportunity_priority: float = 0.0,
    ) -> StrategicGuidance:
        """Return high-level explore/exploit guidance for the current episode step."""

        current_cluster_id = normalize_parser_action(current_state.state_cluster_id or current_state.world_state_hash)
        current_node = self.nodes.get(current_cluster_id)
        current_opportunities = self.opportunities_for_cluster(
            current_cluster_id,
            active_only=True,
            min_priority=min_opportunity_priority,
        )
        current_structural_opportunities = [
            opportunity
            for opportunity in current_opportunities
            if opportunity.kind is RegionOpportunityKind.STRUCTURAL_ACCESS
        ]
        current_object_opportunities = [
            opportunity
            for opportunity in current_opportunities
            if opportunity.kind is RegionOpportunityKind.OBJECT_FOLLOWUP
        ]
        current_exit_actions = self._unexplored_exit_actions(current_cluster_id)
        current_cluster_score = self.cluster_strategic_score(current_cluster_id)
        current_region_harvested = bool(current_node is not None and current_node.durable_progress_count > 0)
        current_object_actions = self._cluster_object_action_hints(
            current_cluster_id,
            min_priority=min_opportunity_priority,
        )
        current_region_exhausted = bool(
            current_region_harvested
            and not current_structural_opportunities
            and (
                not current_object_opportunities
                or all(opportunity.no_progress_attempt_count > 0 for opportunity in current_object_opportunities)
                or (current_node is not None and current_node.visit_count > cluster_visit_exhaustion_threshold)
            )
        )

        best_frontier_cluster = ""
        best_frontier_score = float("-inf")
        best_frontier_has_structural_access = False
        for entry in frontier_entries or []:
            cluster_id = normalize_parser_action(entry.state_cluster_id or entry.world_state_hash)
            score = self.cluster_strategic_score(cluster_id)
            if score > best_frontier_score:
                best_frontier_score = score
                best_frontier_cluster = cluster_id
                best_frontier_has_structural_access = any(
                    opportunity.kind is RegionOpportunityKind.STRUCTURAL_ACCESS
                    for opportunity in self.opportunities_for_cluster(
                        cluster_id,
                        active_only=True,
                        min_priority=min_opportunity_priority,
                    )
                )

        if current_structural_opportunities and current_cluster_score > 0.0 and (
            current_node is None or current_node.visit_count <= cluster_visit_exhaustion_threshold
        ):
            return StrategicGuidance(
                mode=StrategicMode.EXPLOIT,
                reason="Current region still has grounded structural access opportunities.",
                preferred_cluster_id=current_cluster_id,
                try_actions=self.suggested_actions_for_cluster(
                    current_cluster_id,
                    min_priority=min_opportunity_priority,
                ),
                salient_objects=self.suggested_objects_for_cluster(
                    current_cluster_id,
                    min_priority=min_opportunity_priority,
                ),
                opportunity_labels=[opportunity.token for opportunity in current_structural_opportunities[:4]],
            )

        if (
            best_frontier_cluster
            and best_frontier_cluster != current_cluster_id
            and (
                best_frontier_has_structural_access
                or current_region_exhausted
                or best_frontier_score >= max(current_cluster_score + 0.25, 0.8)
            )
        ):
            return StrategicGuidance(
                mode=StrategicMode.EXPLOIT,
                reason=(
                    "Current region looks harvested; shift to the strongest known structural or route opportunity."
                    if current_region_exhausted
                    else "A known region now offers a stronger structural or unresolved route opportunity."
                ),
                preferred_cluster_id=best_frontier_cluster,
                try_actions=[],
                avoid_actions=current_object_actions[:6],
                salient_objects=self.suggested_objects_for_cluster(
                    best_frontier_cluster,
                    min_priority=min_opportunity_priority,
                ),
                opportunity_labels=[
                    opportunity.token
                    for opportunity in self.opportunities_for_cluster(
                        best_frontier_cluster,
                        min_priority=min_opportunity_priority,
                    )[:4]
                ],
            )

        if (
            current_object_opportunities
            and not current_region_exhausted
            and not current_region_harvested
            and current_cluster_score > 0.8
            and (current_node is None or current_node.visit_count <= cluster_visit_exhaustion_threshold)
        ):
            return StrategicGuidance(
                mode=StrategicMode.EXPLOIT,
                reason="Current region still has unresolved object interactions worth probing.",
                preferred_cluster_id=current_cluster_id,
                try_actions=self.suggested_actions_for_cluster(
                    current_cluster_id,
                    min_priority=min_opportunity_priority,
                ),
                salient_objects=self.suggested_objects_for_cluster(
                    current_cluster_id,
                    min_priority=min_opportunity_priority,
                ),
                opportunity_labels=[opportunity.token for opportunity in current_object_opportunities[:4]],
            )

        if current_exit_actions:
            return StrategicGuidance(
                mode=StrategicMode.EXPLORE,
                reason=(
                    "Current region looks harvested; prefer unexplored exits over more local object churn."
                    if current_region_exhausted
                    else "Current region still has unexplored exits or routes worth mapping."
                ),
                preferred_cluster_id=current_cluster_id,
                try_actions=current_exit_actions[:4],
                avoid_actions=current_object_actions[:6] if current_region_exhausted else [],
                salient_objects=self.suggested_objects_for_cluster(
                    current_cluster_id,
                    min_priority=min_opportunity_priority,
                ),
                opportunity_labels=["route:" + action for action in current_exit_actions[:4]],
            )

        if best_frontier_cluster and best_frontier_cluster != current_cluster_id and best_frontier_score > 0.75:
            return StrategicGuidance(
                mode=StrategicMode.EXPLOIT,
                reason="Known frontier route has a stronger unresolved opportunity than the current region.",
                preferred_cluster_id=best_frontier_cluster,
                try_actions=[],
                avoid_actions=current_object_actions[:6] if current_region_exhausted else [],
                salient_objects=self.suggested_objects_for_cluster(
                    best_frontier_cluster,
                    min_priority=min_opportunity_priority,
                ),
                opportunity_labels=[
                    opportunity.token
                    for opportunity in self.opportunities_for_cluster(
                        best_frontier_cluster,
                        min_priority=min_opportunity_priority,
                    )[:4]
                ],
            )

        return StrategicGuidance(
            mode=StrategicMode.EXPLORE,
            reason="No strong current exploit target; continue broad mapping.",
            preferred_cluster_id=current_cluster_id,
            try_actions=current_exit_actions[:4],
            salient_objects=self.suggested_objects_for_cluster(current_cluster_id),
            opportunity_labels=[],
        )

    def cluster_strategic_score(self, cluster_id: str | None) -> float:
        """Return the current strategic value of a region cluster."""

        normalized_cluster = normalize_parser_action(cluster_id or "")
        if not normalized_cluster or normalized_cluster not in self.nodes:
            return 0.0
        node = self.nodes[normalized_cluster]
        active_opportunities = self.opportunities_for_cluster(normalized_cluster, active_only=True)
        structural = sum(1 for opportunity in active_opportunities if opportunity.kind is RegionOpportunityKind.STRUCTURAL_ACCESS)
        object_followups = sum(1 for opportunity in active_opportunities if opportunity.kind is RegionOpportunityKind.OBJECT_FOLLOWUP)
        unexplored_exits = sum(1 for opportunity in active_opportunities if opportunity.kind is RegionOpportunityKind.UNEXPLORED_EXIT)
        harvested_region = node.durable_progress_count > 0
        object_weight = 0.1 if harvested_region and structural == 0 else 0.9
        exit_weight = 1.05 if harvested_region else 0.6
        revisit_penalty = max(node.visit_count - 1, 0) * (0.9 if harvested_region else 0.45)
        no_progress_penalty = sum(opportunity.no_progress_attempt_count for opportunity in active_opportunities) * 0.25
        exhausted_object_penalty = (
            sum(
                1
                for opportunity in active_opportunities
                if opportunity.kind is RegionOpportunityKind.OBJECT_FOLLOWUP and opportunity.no_progress_attempt_count > 0
            )
            * (0.45 if harvested_region else 0.15)
        )
        return max(
            0.0,
            1.9 * structural
            + object_weight * object_followups
            + exit_weight * unexplored_exits
            + 0.15 * float(node.durable_progress_count)
            - revisit_penalty
            - no_progress_penalty,
            - exhausted_object_penalty,
        )

    def unresolved_opportunity_count(self, cluster_id: str | None) -> int:
        """Return the number of unresolved opportunities in a region cluster."""

        return len(self.opportunities_for_cluster(cluster_id, active_only=True))

    def suggested_actions_for_cluster(
        self,
        cluster_id: str | None,
        *,
        limit: int = 6,
        min_priority: float = 0.0,
    ) -> list[str]:
        """Return compact action hints for the strongest unresolved opportunities in one cluster."""

        suggestions: list[str] = []
        seen: set[str] = set()
        for opportunity in self.opportunities_for_cluster(
            cluster_id,
            active_only=True,
            min_priority=min_priority,
        ):
            for action in opportunity.action_hints:
                normalized = normalize_parser_action(action)
                if not normalized or normalized in seen:
                    continue
                suggestions.append(action)
                seen.add(normalized)
                if len(suggestions) >= limit:
                    return suggestions
        return suggestions

    def _cluster_object_action_hints(
        self,
        cluster_id: str | None,
        *,
        limit: int = 6,
        min_priority: float = 0.0,
    ) -> list[str]:
        """Return object-followup action hints for one cluster."""

        suggestions: list[str] = []
        seen: set[str] = set()
        for opportunity in self.opportunities_for_cluster(
            cluster_id,
            active_only=True,
            min_priority=min_priority,
        ):
            if opportunity.kind is not RegionOpportunityKind.OBJECT_FOLLOWUP:
                continue
            for action in opportunity.action_hints:
                normalized = normalize_parser_action(action)
                if not normalized or normalized in seen:
                    continue
                suggestions.append(action)
                seen.add(normalized)
                if len(suggestions) >= limit:
                    return suggestions
        return suggestions

    def suggested_objects_for_cluster(
        self,
        cluster_id: str | None,
        *,
        limit: int = 6,
        min_priority: float = 0.0,
    ) -> list[str]:
        """Return the top unresolved object tokens for one cluster."""

        tokens: list[str] = []
        seen: set[str] = set()
        for opportunity in self.opportunities_for_cluster(
            cluster_id,
            active_only=True,
            min_priority=min_priority,
        ):
            token = normalize_parser_action(opportunity.token)
            if not token or token in seen:
                continue
            tokens.append(token)
            seen.add(token)
            if len(tokens) >= limit:
                break
        return tokens

    def opportunities_for_cluster(
        self,
        cluster_id: str | None,
        *,
        active_only: bool = True,
        min_priority: float = 0.0,
    ) -> list[RegionOpportunity]:
        """Return opportunities linked to a cluster, ordered by current priority."""

        normalized_cluster = normalize_parser_action(cluster_id or "")
        if not normalized_cluster or normalized_cluster not in self.nodes:
            return []
        node = self.nodes[normalized_cluster]
        opportunities = [
            self.opportunities[opportunity_id]
            for opportunity_id in node.opportunity_ids
            if opportunity_id in self.opportunities and (not active_only or not self.opportunities[opportunity_id].resolved)
        ]
        if min_priority > 0.0:
            opportunities = [
                opportunity
                for opportunity in opportunities
                if opportunity.current_priority() >= min_priority
            ]
        opportunities.sort(
            key=lambda opportunity: (
                -opportunity.current_priority(),
                opportunity.kind.value,
                opportunity.token,
            )
        )
        return opportunities

    def _infer_opportunities(
        self,
        *,
        node: RegionNodeMemory,
        state: "TextGameState",
        step_index: int,
    ) -> None:
        """Infer unresolved opportunities from the current state description and valid actions."""

        visible_tokens = extract_salient_nouns(
            observation=state.observation,
            inventory_text=state.inventory_text,
            valid_actions=[],
        )
        inventory_tokens = inventory_item_tokens(state.inventory_text)
        valid_actions = [normalize_parser_action(action) for action in state.valid_actions if normalize_parser_action(action)]

        for token in sorted(visible_tokens):
            if token in _STRUCTURAL_ACCESS_TOKENS:
                action_hints = self._structural_action_hints(token=token, valid_actions=valid_actions)
                self._upsert_opportunity(
                    node=node,
                    opportunity_id=f"structural:{node.cluster_id}:{token}",
                    kind=RegionOpportunityKind.STRUCTURAL_ACCESS,
                    token=token,
                    action_hints=action_hints,
                    step_index=step_index,
                    base_priority=2.0,
                )

        for action in valid_actions:
            if not is_movement_action(action):
                continue
            self._upsert_opportunity(
                node=node,
                opportunity_id=f"exit:{node.cluster_id}:{action}",
                kind=RegionOpportunityKind.UNEXPLORED_EXIT,
                token=action,
                action_hints=[action],
                step_index=step_index,
                base_priority=1.0,
            )

        for action in valid_actions:
            if is_movement_action(action):
                continue
            verb, _obj = split_action_command(action)
            if verb not in _OBJECT_FOLLOWUP_VERBS:
                continue
            targets = action_target_tokens(action)
            if not targets:
                continue
            token = sorted(targets)[0]
            if token in inventory_tokens and verb in {"take", "get"}:
                continue
            self._upsert_opportunity(
                node=node,
                opportunity_id=f"object:{node.cluster_id}:{token}",
                kind=RegionOpportunityKind.OBJECT_FOLLOWUP,
                token=token,
                action_hints=self._canonical_object_action_hints(
                    token=token,
                    valid_actions=valid_actions,
                    inventory_tokens=inventory_tokens,
                ),
                step_index=step_index,
                base_priority=1.35 if token not in inventory_tokens else 0.9,
            )

    def _upsert_opportunity(
        self,
        *,
        node: RegionNodeMemory,
        opportunity_id: str,
        kind: RegionOpportunityKind,
        token: str,
        action_hints: list[str],
        step_index: int,
        base_priority: float,
    ) -> None:
        """Create or refresh one unresolved opportunity record."""

        opportunity = self.opportunities.get(opportunity_id)
        if opportunity is None:
            opportunity = RegionOpportunity(
                opportunity_id=opportunity_id,
                cluster_id=node.cluster_id,
                kind=kind,
                token=token,
                action_hints=list(action_hints),
                discovered_step=step_index,
                last_seen_step=step_index,
                base_priority=base_priority,
            )
            self.opportunities[opportunity_id] = opportunity
        else:
            opportunity.last_seen_step = step_index
            opportunity.base_priority = max(opportunity.base_priority, base_priority)
            if action_hints:
                existing = {normalize_parser_action(action) for action in opportunity.action_hints}
                for action in action_hints:
                    normalized = normalize_parser_action(action)
                    if normalized and normalized not in existing:
                        opportunity.action_hints.append(action)
                        existing.add(normalized)
        node.opportunity_ids.add(opportunity_id)

    def _canonical_object_action_hints(
        self,
        *,
        token: str,
        valid_actions: list[str],
        inventory_tokens: set[str],
    ) -> list[str]:
        """Return canonical object-centric action hints for one token."""

        canonical_prefixes = (
            f"examine {token}",
            f"look at {token}",
            f"read {token}",
            f"take {token}",
            f"get {token}",
            f"open {token}",
            f"look in {token}",
            f"enter {token}",
        )
        ranked: list[str] = []
        for candidate in canonical_prefixes:
            normalized_candidate = normalize_parser_action(candidate)
            if normalized_candidate in valid_actions:
                if token in inventory_tokens and normalized_candidate.startswith(("take ", "get ")):
                    continue
                ranked.append(normalized_candidate)
        if ranked:
            return ranked[:4]
        return [f"examine {token}", f"take {token}", f"open {token}"]

    def _structural_action_hints(self, *, token: str, valid_actions: list[str]) -> list[str]:
        """Return canonical structure/entry action hints for one structural token."""

        canonical_candidates = [
            f"open {token}",
            f"look in {token}",
            f"enter {token}",
            f"examine {token}",
        ]
        if token == "house":
            canonical_candidates = [
                "open window",
                "look in window",
                "enter window",
                "open door",
                "enter house",
                "examine house",
            ] + canonical_candidates
        ranked = [
            normalize_parser_action(candidate)
            for candidate in canonical_candidates
            if normalize_parser_action(candidate) in valid_actions
        ]
        if ranked:
            return ranked[:5]
        return canonical_candidates[:5]

    def _unexplored_exit_actions(self, cluster_id: str | None) -> list[str]:
        """Return currently known but not yet traversed exit actions for a cluster."""

        normalized_cluster = normalize_parser_action(cluster_id or "")
        if not normalized_cluster or normalized_cluster not in self.nodes:
            return []
        node = self.nodes[normalized_cluster]
        return [
            exit_action
            for exit_action in sorted(node.exit_actions)
            if exit_action not in node.explored_exit_actions
        ]

    def _opportunity_matches_action(
        self,
        *,
        opportunity: RegionOpportunity,
        action: str,
        verb: str,
        action_targets: set[str],
    ) -> bool:
        """Return whether an action plausibly touched an unresolved opportunity."""

        if opportunity.kind is RegionOpportunityKind.UNEXPLORED_EXIT:
            return action == normalize_parser_action(opportunity.token)
        if normalize_parser_action(opportunity.token) in action_targets:
            return True
        if opportunity.kind is RegionOpportunityKind.STRUCTURAL_ACCESS and verb in {"open", "enter", "look in", "examine"}:
            return normalize_parser_action(opportunity.token) in action or opportunity.token == "house"
        return False

    def _opportunity_is_resolved(
        self,
        *,
        opportunity: RegionOpportunity,
        previous_state: "TextGameState",
        current_state: "TextGameState",
        action: str,
        score_gain: int,
        inventory_gain_count: int,
        affordance_gain: int,
    ) -> bool:
        """Return whether an opportunity should be treated as resolved by this outcome."""

        if opportunity.kind is RegionOpportunityKind.UNEXPLORED_EXIT:
            return normalize_parser_action(current_state.state_cluster_id) != normalize_parser_action(previous_state.state_cluster_id)
        if opportunity.kind is RegionOpportunityKind.STRUCTURAL_ACCESS:
            return (
                normalize_parser_action(current_state.state_cluster_id)
                != normalize_parser_action(previous_state.state_cluster_id)
                or action.startswith(("enter ", "climb ", "go in", "go into "))
            )
        if opportunity.kind is RegionOpportunityKind.OBJECT_FOLLOWUP:
            return any((score_gain > 0, inventory_gain_count > 0, affordance_gain > 0))
        return False


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
    strategic_cluster_score: float = 0.0
    unresolved_opportunity_count: int = 0
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
            strategic_cluster_score=float(self.metadata.get("strategic_cluster_score", 0.0)),
            unresolved_opportunity_count=int(self.metadata.get("unresolved_opportunity_count", 0)),
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
        cumulative_reward: float | None = None,
        native_snapshot_reference: str | None = None,
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
            cumulative_reward=self.reward if cumulative_reward is None else float(cumulative_reward),
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
            strategic_cluster_score=float(merged_metadata.get("strategic_cluster_score", 0.0)),
            unresolved_opportunity_count=int(merged_metadata.get("unresolved_opportunity_count", 0)),
            room_text_only_gain=float(merged_metadata.get("room_text_only_gain", 0.0)),
            affordance_gain=int(merged_metadata.get("affordance_gain", 0)),
            native_snapshot_reference=(
                native_snapshot_reference
                or str(merged_metadata.get("native_snapshot_reference", "")).strip()
                or None
            ),
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
    diversity_bonus: float = 0.0
    features: ActionCandidateFeatures | None = None
    raw_line: str = ""


@dataclass(slots=True)
class ActionCandidateFeatures:
    """Explicit evidence features used to rank parser-action candidates."""

    action_text: str
    source_mode: str
    verb: str = ""
    verb_family: str = "other"
    noun_targets: list[str] = field(default_factory=list)
    object_count: int = 0
    is_movement_action: bool = False
    is_reversible_toggle: bool = False
    is_bulk_inventory_action: bool = False
    is_discard_like_action: bool = False
    uses_inventory_object: bool = False
    target_is_new_salient_object: bool = False
    target_is_container_or_openable: bool = False
    target_is_readable_candidate: bool = False
    action_shape_is_simple: bool = False
    action_shape_is_complex_transitive: bool = False
    was_tried_before_in_local_cluster: bool = False
    previous_attempt_count: int = 0
    produced_score_gain_before: bool = False
    produced_inventory_gain_before: bool = False
    produced_affordance_gain_before: bool = False
    prior_score_success_for_verb_family: bool = False
    prior_success_for_verb_family: bool = False
    prior_success_for_exact_action: bool = False
    prior_failure_for_exact_action: bool = False
    prior_score_success_for_object_family: bool = False
    prior_success_for_object_family: bool = False
    prior_failure_for_object_family: bool = False
    prior_inventory_gain_for_object_family: int = 0
    prior_inventory_loss_for_object_family: int = 0
    prior_affordance_gain_for_object_family: int = 0
    touches_newly_salient_object: bool = False
    touches_recent_affordance_object: bool = False
    touches_supported_reflection_object: bool = False
    touches_strategic_object: bool = False
    inverse_of_previous_action: bool = False
    movement_repeat_count: int = 0
    cluster_repeat_count: int = 0
    movement_targets_landmark: bool = False
    is_freshly_unlocked_action: bool = False
    is_freshly_unlocked_exit: bool = False
    follows_recent_affordance_unlock: bool = False
    touches_exhausted_family: bool = False
    exhausted_family_count: int = 0
    max_family_no_progress_count: int = 0
    scene_has_score_harvested: bool = False
    scene_score_success_count: int = 0
    local_scene_post_score_stale: bool = False
    matches_supported_try_action: bool = False
    matches_supported_avoid_action: bool = False
    matches_strategic_try_action: bool = False
    matches_strategic_avoid_action: bool = False
    current_mode: str = ""
    plausibility_score: float = 0.0
    canonical_prior_contribution: float = 0.0
    object_family_evidence_contribution: float = 0.0
    strategic_contribution: float = 0.0

    def to_record(self) -> dict[str, Any]:
        """Convert the feature vector into a JSON-serializable dictionary."""

        return asdict(self)


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
    candidate_pool_before_rerank: list[str] = field(default_factory=list)
    llm_ranked_actions: list[str] = field(default_factory=list)
    candidate_feature_records: list[dict[str, Any]] = field(default_factory=list)
    ranking_source: str = ""
    top_selection_reason: str = ""
    strategic_mode: str = ""
    strategic_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def top_actions(self) -> list[str]:
        """Return the candidate action texts in ranked order."""

        return [candidate.action for candidate in self.candidates]

    def feature_records(self) -> list[dict[str, Any]]:
        """Return feature vectors for logging and debug summaries."""

        if self.candidate_feature_records:
            return list(self.candidate_feature_records)
        return [
            candidate.features.to_record()
            for candidate in self.candidates
            if candidate.features is not None
        ]


class StateSelectionMode(str, Enum):
    """Mode used to choose the next frontier state for revisit."""

    ARCHIVE_BALANCED = "archive_balanced"
    HEURISTIC = "heuristic"
    LEGACY_HEURISTIC = "legacy_heuristic"
    LLM_ASSISTED = "llm_assisted"
    FALLBACK_HEURISTIC = "fallback_heuristic"


class RunnerMode(str, Enum):
    """Episode-runner mode used to orchestrate global/local exploration."""

    LEGACY = "legacy"
    GLOW_FAITHFUL = "glow_faithful"


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


@dataclass(slots=True)
class ArchiveCandidateSummary:
    """Compact archive-state summary used for state-selection prompts and logs."""

    candidate_id: str
    state_id: str
    achieved_value: float
    potential_value: float
    achieved_contribution: float
    potential_contribution: float
    replay_method: str
    visit_count: int
    selection_count: int
    support_count: int = 0
    summary_text: str = ""

    def to_prompt_line(self) -> str:
        """Render a bounded one-line summary for LLM-assisted state selection."""

        return (
            f"{self.candidate_id} | state_id={self.state_id} "
            f"| achieved={self.achieved_value:.2f} potential={self.potential_value:.2f} "
            f"| achieved_contrib={self.achieved_contribution:.2f} potential_contrib={self.potential_contribution:.2f} "
            f"| replay={self.replay_method} | visits={self.visit_count} selects={self.selection_count} "
            f"| support={self.support_count} | state={self.summary_text}"
        )


@dataclass(slots=True)
class ArchiveStateSelectionResult:
    """Structured result from achieved-plus-potential archive state selection."""

    selected_archived_state: ArchivedState | None
    chosen_replay_method: str
    achieved_contribution: float
    potential_contribution: float
    rationale: str
    selection_mode: StateSelectionMode
    candidate_summaries: list[ArchiveCandidateSummary] = field(default_factory=list)
    selected_frontier_trajectory_ids: list[str] = field(default_factory=list)
    selected_critical_state_ids: list[str] = field(default_factory=list)
    raw_output: str = ""
    prompt_snapshot: str = ""
    model_name: str | None = None
    fallback_reason: str = ""
    artifact_directory: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_archive_state_id(self) -> str | None:
        """Return the selected archived state id if available."""

        return self.selected_archived_state.state_id if self.selected_archived_state is not None else None

    def to_record(self) -> dict[str, Any]:
        """Convert the selection result into a JSON-serializable record."""

        return {
            "selected_archived_state": (
                self.selected_archived_state.to_record(include_native_snapshot=False)
                if self.selected_archived_state is not None
                else None
            ),
            "chosen_replay_method": self.chosen_replay_method,
            "achieved_contribution": self.achieved_contribution,
            "potential_contribution": self.potential_contribution,
            "rationale": self.rationale,
            "selection_mode": self.selection_mode.value,
            "candidate_summaries": [candidate.to_prompt_line() for candidate in self.candidate_summaries],
            "selected_frontier_trajectory_ids": list(self.selected_frontier_trajectory_ids),
            "selected_critical_state_ids": list(self.selected_critical_state_ids),
            "raw_output": self.raw_output,
            "prompt_snapshot": self.prompt_snapshot,
            "model_name": self.model_name,
            "fallback_reason": self.fallback_reason,
            "artifact_directory": self.artifact_directory,
            "metadata": dict(self.metadata),
        }


class BranchTerminationReason(str, Enum):
    """Reason why a shallow local rollout ended."""

    HORIZON_REACHED = "horizon_reached"
    TERMINATED = "terminated"
    NO_ACTIONS = "no_actions"
    RESTORE_FAILED = "restore_failed"
    LOOP_ABORTED = "loop_aborted"


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
    persistent_inventory_gain_count: int = 0
    persistent_inventory_loss_count: int = 0
    new_affordance_count: int = 0
    persistent_affordance_gain: int = 0
    persistent_exit_gain_count: int = 0
    ended_in_same_cluster: bool = False
    durable_progress: bool = False
    exhausted_family_count: int = 0
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
    movement_action_ratio: float = 0.0
    movement_repeat_count: int = 0
    bulk_inventory_action_count: int = 0
    inventory_churn_penalty: float = 0.0
    discard_like_action_count: int = 0
    aggressive_action_count: int = 0
    speculative_tool_use_action_count: int = 0
    post_gain_churn_action_count: int = 0
    loop_event_count: int = 0
    first_durable_gain_action_index: int | None = None
    first_durable_gain_action: str = ""
    last_durable_progress_action_index: int | None = None
    last_durable_progress_action: str = ""
    last_meaningful_progress_action_index: int | None = None
    last_meaningful_progress_action: str = ""
    notable_observation_changes: list[str] = field(default_factory=list)
    trajectory_steps: list[TrajectoryStep] = field(default_factory=list)
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
    root_state_id: str = ""
    best_branch_index: int | None = None
    branch_commit_allowed: bool = False
    commit_rejection_reason: str = ""
    comparison_notes: str = ""
    used_local_world_model: LocalWorldModel | None = None
    mar_inference: MARInferenceResult | None = None
    updated_local_world_model: LocalWorldModel | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
    strategic_value: float = 0.0
    unresolved_opportunity_count: int = 0
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
            strategic_value=float(metadata.get("strategic_value", 0.0)) if metadata else 0.0,
            unresolved_opportunity_count=int(metadata.get("unresolved_opportunity_count", 0)) if metadata else 0,
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
    strategic_value: float = 0.0
    unresolved_opportunity_count: int = 0
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
            strategic_value=float(candidate.strategic_value),
            unresolved_opportunity_count=int(candidate.unresolved_opportunity_count),
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
            strategic_value=float(self.strategic_value),
            unresolved_opportunity_count=int(self.unresolved_opportunity_count),
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
    cumulative_reward: float = 0.0
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
    strategic_cluster_score: float = 0.0
    unresolved_opportunity_count: int = 0
    room_text_only_gain: float = 0.0
    affordance_gain: int = 0
    native_snapshot_reference: str | None = None
    world_state_snapshot: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_invariants(self) -> None:
        """Validate basic trajectory-step invariants used by typed artifacts."""

        if not self.episode_id.strip():
            raise ValueError("TrajectoryStep.episode_id must not be empty.")
        if self.step_index < 0:
            raise ValueError("TrajectoryStep.step_index must be >= 0.")
        if self.moves < 0:
            raise ValueError("TrajectoryStep.moves must be >= 0.")

    def to_record(self) -> dict[str, Any]:
        """Convert the step into a JSONL-friendly dictionary."""

        self.validate_invariants()
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "TrajectoryStep":
        """Construct a step from a stored JSON record."""

        step = cls(
            episode_id=str(record["episode_id"]),
            step_index=int(record["step_index"]),
            action=str(record["action"]),
            observation=str(record["observation"]),
            reward=float(record["reward"]),
            cumulative_reward=float(record.get("cumulative_reward", record["reward"])),
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
            strategic_cluster_score=float(record.get("strategic_cluster_score", 0.0)),
            unresolved_opportunity_count=int(record.get("unresolved_opportunity_count", 0)),
            room_text_only_gain=float(record.get("room_text_only_gain", 0.0)),
            affordance_gain=int(record.get("affordance_gain", 0)),
            native_snapshot_reference=(
                str(record["native_snapshot_reference"]).strip()
                if record.get("native_snapshot_reference") not in (None, "")
                else None
            ),
            world_state_snapshot=normalize_world_state_snapshot_record(record["world_state_snapshot"])
            if isinstance(record.get("world_state_snapshot"), MappingABC)
            else None,
            metadata=dict(record["metadata"]) if isinstance(record.get("metadata"), MappingABC) else {},
        )
        step.validate_invariants()
        return step


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

        for step in self.steps:
            step.validate_invariants()

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
class ReplayMetadata:
    """Replay and restore metadata shared by archived states and trajectories."""

    restore_strategy: str = "action_replay_fallback"
    replay_actions: list[str] = field(default_factory=list)
    world_state_hash: str = ""
    native_snapshot_reference: str | None = None
    native_snapshot: tuple[Any, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self, include_native_snapshot: bool = False) -> dict[str, Any]:
        """Convert replay metadata into a JSON-serializable record."""

        record = {
            "restore_strategy": self.restore_strategy,
            "replay_actions": list(self.replay_actions),
            "world_state_hash": self.world_state_hash,
            "native_snapshot_reference": self.native_snapshot_reference,
            "metadata": dict(self.metadata),
        }
        if include_native_snapshot and self.native_snapshot is not None:
            record["native_snapshot"] = list(self.native_snapshot)
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ReplayMetadata":
        """Construct replay metadata from a serialized record."""

        native_snapshot = record.get("native_snapshot")
        normalized_snapshot: tuple[Any, ...] | None = None
        if isinstance(native_snapshot, list):
            normalized_snapshot = tuple(native_snapshot)
        elif isinstance(native_snapshot, tuple):
            normalized_snapshot = tuple(native_snapshot)

        return cls(
            restore_strategy=_normalize_restore_strategy_name(str(record.get("restore_strategy", "action_replay_fallback"))),
            replay_actions=_normalize_string_list(record.get("replay_actions")),
            world_state_hash=str(record.get("world_state_hash", "")),
            native_snapshot_reference=(
                str(record["native_snapshot_reference"]).strip()
                if record.get("native_snapshot_reference") not in (None, "")
                else None
            ),
            native_snapshot=normalized_snapshot,
            metadata=dict(record["metadata"]) if isinstance(record.get("metadata"), MappingABC) else {},
        )


@dataclass(slots=True)
class EpisodeTrajectorySummary:
    """Summary fields derived from an episode trajectory for frontier analysis."""

    unique_world_state_count: int = 0
    unique_cluster_count: int = 0
    action_count: int = 0
    loop_event_count: int = 0
    max_score: int = 0
    final_cluster_id: str = ""
    discovered_object_tokens: list[str] = field(default_factory=list)
    bottleneck_step_indices: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert the trajectory summary into a JSON-serializable record."""

        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "EpisodeTrajectorySummary":
        """Construct a trajectory summary from a serialized record."""

        return cls(
            unique_world_state_count=int(record.get("unique_world_state_count", 0)),
            unique_cluster_count=int(record.get("unique_cluster_count", 0)),
            action_count=int(record.get("action_count", 0)),
            loop_event_count=int(record.get("loop_event_count", 0)),
            max_score=int(record.get("max_score", 0)),
            final_cluster_id=str(record.get("final_cluster_id", "")),
            discovered_object_tokens=_normalize_string_list(record.get("discovered_object_tokens")),
            bottleneck_step_indices=[int(value) for value in (record.get("bottleneck_step_indices") or [])],
            metadata=dict(record["metadata"]) if isinstance(record.get("metadata"), MappingABC) else {},
        )


@dataclass(slots=True)
class EpisodeTrajectory:
    """Typed full-episode trajectory for a more faithful GLoW-style frontier."""

    episode_id: str
    root_state_id: str
    selected_from_archive_state_id: str | None = None
    steps: list[TrajectoryStep] = field(default_factory=list)
    max_cumulative_reward_achieved: float = 0.0
    final_score: int = 0
    final_done: bool = False
    replay_metadata: ReplayMetadata = field(default_factory=ReplayMetadata)
    summary_fields: EpisodeTrajectorySummary = field(default_factory=EpisodeTrajectorySummary)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def trajectory_id(self) -> str:
        """Return the stable trajectory id used by global frontier artifacts."""

        return self.episode_id

    def validate_invariants(self) -> None:
        """Validate basic ordering and aggregate invariants."""

        if not self.root_state_id.strip():
            raise ValueError("EpisodeTrajectory.root_state_id must not be empty.")
        if not self.steps:
            return

        episode_ids = {step.episode_id for step in self.steps}
        if episode_ids != {self.episode_id}:
            raise ValueError(
                f"EpisodeTrajectory {self.episode_id!r} contains mismatched step episode ids: {sorted(episode_ids)}"
            )

        step_indices = [step.step_index for step in self.steps]
        if step_indices != sorted(step_indices):
            raise ValueError("EpisodeTrajectory steps must be sorted by step_index.")
        if len(step_indices) != len(set(step_indices)):
            raise ValueError("EpisodeTrajectory contains duplicate step_index values.")

        running_reward = 0.0
        max_cumulative_reward = float("-inf")
        for step in self.steps:
            running_reward += float(step.reward)
            if abs(float(step.cumulative_reward) - running_reward) > 1e-6:
                raise ValueError(
                    f"TrajectoryStep cumulative_reward mismatch at step {step.step_index}: "
                    f"expected {running_reward}, got {step.cumulative_reward}"
                )
            max_cumulative_reward = max(max_cumulative_reward, running_reward)

        final_step = self.steps[-1]
        if self.final_score != final_step.score:
            raise ValueError(
                f"EpisodeTrajectory.final_score {self.final_score} does not match last step score {final_step.score}."
            )
        if self.final_done != final_step.done:
            raise ValueError(
                f"EpisodeTrajectory.final_done {self.final_done} does not match last step done {final_step.done}."
            )
        if abs(self.max_cumulative_reward_achieved - max_cumulative_reward) > 1e-6:
            raise ValueError(
                "EpisodeTrajectory.max_cumulative_reward_achieved does not match the step sequence."
            )

    def to_record(self) -> dict[str, Any]:
        """Convert the full episode trajectory into a JSON-serializable record."""

        return {
            "episode_id": self.episode_id,
            "root_state_id": self.root_state_id,
            "selected_from_archive_state_id": self.selected_from_archive_state_id,
            "steps": [step.to_record() for step in self.steps],
            "max_cumulative_reward_achieved": self.max_cumulative_reward_achieved,
            "final_score": self.final_score,
            "final_done": self.final_done,
            "replay_metadata": self.replay_metadata.to_record(),
            "summary_fields": self.summary_fields.to_record(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "EpisodeTrajectory":
        """Construct an episode trajectory from a serialized record."""

        trajectory = cls(
            episode_id=str(record["episode_id"]),
            root_state_id=str(record.get("root_state_id", "")),
            selected_from_archive_state_id=(
                str(record["selected_from_archive_state_id"])
                if record.get("selected_from_archive_state_id") not in (None, "")
                else None
            ),
            steps=[
                TrajectoryStep.from_record(step_record)
                for step_record in (record.get("steps") or [])
                if isinstance(step_record, MappingABC)
            ],
            max_cumulative_reward_achieved=float(record.get("max_cumulative_reward_achieved", 0.0)),
            final_score=int(record.get("final_score", 0)),
            final_done=bool(record.get("final_done", False)),
            replay_metadata=ReplayMetadata.from_record(record["replay_metadata"])
            if isinstance(record.get("replay_metadata"), MappingABC)
            else ReplayMetadata(),
            summary_fields=EpisodeTrajectorySummary.from_record(record["summary_fields"])
            if isinstance(record.get("summary_fields"), MappingABC)
            else EpisodeTrajectorySummary(),
            metadata=dict(record["metadata"]) if isinstance(record.get("metadata"), MappingABC) else {},
        )
        trajectory.validate_invariants()
        return trajectory

    @classmethod
    def from_legacy_trajectory(
        cls,
        trajectory: Trajectory,
        *,
        root_state_id: str | None = None,
        selected_from_archive_state_id: str | None = None,
        replay_metadata: ReplayMetadata | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "EpisodeTrajectory":
        """Adapt the existing JSONL `Trajectory` into the GLoW-style representation."""

        cumulative_reward = 0.0
        adapted_steps: list[TrajectoryStep] = []
        discovered_objects: set[str] = set()
        for step in trajectory.steps:
            cumulative_reward += float(step.reward)
            discovered_objects.update(action_target_tokens(step.action))
            if step.inventory_text:
                discovered_objects.update(inventory_item_tokens(step.inventory_text))
            step_record = step.to_record()
            step_record["cumulative_reward"] = cumulative_reward
            adapted_steps.append(TrajectoryStep.from_record(step_record))

        if adapted_steps:
            final_step = adapted_steps[-1]
            max_cumulative_reward = max(step.cumulative_reward for step in adapted_steps)
            derived_root_state_id = root_state_id or f"{trajectory.episode_id}:root:{adapted_steps[0].world_state_hash}"
            derived_replay_metadata = replay_metadata or ReplayMetadata(
                restore_strategy=(
                    str(adapted_steps[0].world_state_snapshot.get("restore_strategy", "action_replay_fallback"))
                    if adapted_steps[0].world_state_snapshot
                    else "action_replay_fallback"
                ),
                replay_actions=[],
                world_state_hash=adapted_steps[0].world_state_hash,
                native_snapshot_reference=adapted_steps[0].native_snapshot_reference,
                metadata={"source_path": str(trajectory.source_path) if trajectory.source_path is not None else ""},
            )
            summary_fields = EpisodeTrajectorySummary(
                unique_world_state_count=len({step.world_state_hash for step in adapted_steps}),
                unique_cluster_count=len({step.state_cluster_id for step in adapted_steps if step.state_cluster_id}),
                action_count=len(adapted_steps),
                loop_event_count=sum(1 for step in adapted_steps if step.loop_detected),
                max_score=max(step.score for step in adapted_steps),
                final_cluster_id=final_step.state_cluster_id,
                discovered_object_tokens=sorted(discovered_objects),
                bottleneck_step_indices=[
                    step.step_index for step in adapted_steps if step.done or step.loop_detected
                ],
            )
            episode_trajectory = cls(
                episode_id=trajectory.episode_id,
                root_state_id=derived_root_state_id,
                selected_from_archive_state_id=selected_from_archive_state_id,
                steps=adapted_steps,
                max_cumulative_reward_achieved=max_cumulative_reward,
                final_score=final_step.score,
                final_done=final_step.done,
                replay_metadata=derived_replay_metadata,
                summary_fields=summary_fields,
                metadata=dict(metadata or trajectory.metadata),
            )
            episode_trajectory.validate_invariants()
            return episode_trajectory

        return cls(
            episode_id=trajectory.episode_id,
            root_state_id=root_state_id or f"{trajectory.episode_id}:root:empty",
            selected_from_archive_state_id=selected_from_archive_state_id,
            steps=[],
            replay_metadata=replay_metadata or ReplayMetadata(),
            metadata=dict(metadata or trajectory.metadata),
        )

    def to_legacy_trajectory(self) -> Trajectory:
        """Project the GLoW-style episode trajectory back to the JSONL trajectory type."""

        legacy_trajectory = Trajectory(
            episode_id=self.episode_id,
            steps=[TrajectoryStep.from_record(step.to_record()) for step in self.steps],
            metadata={
                **dict(self.metadata),
                "root_state_id": self.root_state_id,
                "selected_from_archive_state_id": self.selected_from_archive_state_id,
                "max_cumulative_reward_achieved": self.max_cumulative_reward_achieved,
            },
        )
        legacy_trajectory.validate_integrity()
        return legacy_trajectory


@dataclass(slots=True)
class FrontierTrajectoryEntry:
    """One value-ranked trajectory retained in the global frontier."""

    trajectory_id: str
    value: float
    novelty_score: float = 0.0
    diversity_score: float = 0.0
    state_cluster_ids: list[str] = field(default_factory=list)
    state_family_keys: list[str] = field(default_factory=list)
    bottleneck_state_ids: list[str] = field(default_factory=list)
    bottleneck_step_indices: list[int] = field(default_factory=list)
    inserted_at: str = ""
    insertion_order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert the frontier trajectory entry into a JSON-serializable record."""

        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "FrontierTrajectoryEntry":
        """Construct a frontier trajectory entry from a serialized record."""

        return cls(
            trajectory_id=str(record["trajectory_id"]),
            value=float(record.get("value", 0.0)),
            novelty_score=float(record.get("novelty_score", 0.0)),
            diversity_score=float(record.get("diversity_score", 0.0)),
            state_cluster_ids=_normalize_string_list(record.get("state_cluster_ids")),
            state_family_keys=_normalize_string_list(record.get("state_family_keys")),
            bottleneck_state_ids=_normalize_string_list(record.get("bottleneck_state_ids")),
            bottleneck_step_indices=[int(value) for value in (record.get("bottleneck_step_indices") or [])],
            inserted_at=str(record.get("inserted_at", "")),
            insertion_order=int(record.get("insertion_order", 0)),
            metadata=dict(record["metadata"]) if isinstance(record.get("metadata"), MappingABC) else {},
        )


@dataclass(slots=True)
class SimilarityMetadata:
    """Optional embedding/similarity metadata attached to archived states."""

    embedding_model: str = ""
    embedding_id: str = ""
    similarity_key: str = ""
    nearest_state_ids: list[str] = field(default_factory=list)
    similarity_score: float | None = None

    def to_record(self) -> dict[str, Any]:
        """Convert similarity metadata into a JSON-serializable record."""

        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SimilarityMetadata":
        """Construct similarity metadata from a serialized record."""

        return cls(
            embedding_model=str(record.get("embedding_model", "")),
            embedding_id=str(record.get("embedding_id", "")),
            similarity_key=str(record.get("similarity_key", "")),
            nearest_state_ids=_normalize_string_list(record.get("nearest_state_ids")),
            similarity_score=float(record["similarity_score"]) if record.get("similarity_score") is not None else None,
        )


@dataclass(slots=True)
class ArchivedState:
    """A stable archived state used by GLoW state selection."""

    state_id: str
    provenance_trajectory_id: str
    provenance_timestep: int
    achieved_value: float
    identity_strategy: str = ""
    first_seen_episode_id: str = ""
    first_seen_timestep: int = 0
    last_seen_episode_id: str = ""
    last_seen_timestep: int = 0
    provenance_trajectory_ids: list[str] = field(default_factory=list)
    visit_count: int = 0
    selection_count: int = 0
    restore_success_count: int = 0
    restore_failure_count: int = 0
    score_at_state: int = 0
    projected_potential_value: float | None = None
    frontier_support_count: int = 0
    supporting_frontier_trajectory_ids: list[str] = field(default_factory=list)
    observation_summary: str = ""
    inventory_summary: str = ""
    valid_action_summary: str = ""
    state_cluster_id: str = ""
    replay_metadata: ReplayMetadata = field(default_factory=ReplayMetadata)
    similarity_metadata: SimilarityMetadata | None = None
    native_snapshot_reference: str | None = None
    native_snapshot: tuple[Any, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_invariants(self) -> None:
        """Validate basic archived-state invariants."""

        if not self.state_id.strip():
            raise ValueError("ArchivedState.state_id must not be empty.")
        if not self.provenance_trajectory_id.strip():
            raise ValueError("ArchivedState.provenance_trajectory_id must not be empty.")
        if self.provenance_timestep < 0:
            raise ValueError("ArchivedState.provenance_timestep must be >= 0.")
        if (
            self.first_seen_timestep < 0
            or self.last_seen_timestep < 0
            or self.visit_count < 0
            or self.selection_count < 0
            or self.restore_success_count < 0
            or self.restore_failure_count < 0
            or self.frontier_support_count < 0
        ):
            raise ValueError("ArchivedState counters/timesteps must be >= 0.")
        if self.projected_potential_value is not None and self.projected_potential_value < self.achieved_value:
            raise ValueError("ArchivedState.projected_potential_value must be >= achieved_value when present.")

    def to_record(self, include_native_snapshot: bool = False) -> dict[str, Any]:
        """Convert the archived state into a JSON-serializable record."""

        record = {
            "state_id": self.state_id,
            "provenance_trajectory_id": self.provenance_trajectory_id,
            "provenance_timestep": self.provenance_timestep,
            "achieved_value": self.achieved_value,
            "identity_strategy": self.identity_strategy,
            "first_seen_episode_id": self.first_seen_episode_id,
            "first_seen_timestep": self.first_seen_timestep,
            "last_seen_episode_id": self.last_seen_episode_id,
            "last_seen_timestep": self.last_seen_timestep,
            "provenance_trajectory_ids": list(self.provenance_trajectory_ids),
            "visit_count": self.visit_count,
            "selection_count": self.selection_count,
            "restore_success_count": self.restore_success_count,
            "restore_failure_count": self.restore_failure_count,
            "score_at_state": self.score_at_state,
            "projected_potential_value": self.projected_potential_value,
            "frontier_support_count": self.frontier_support_count,
            "supporting_frontier_trajectory_ids": list(self.supporting_frontier_trajectory_ids),
            "observation_summary": self.observation_summary,
            "inventory_summary": self.inventory_summary,
            "valid_action_summary": self.valid_action_summary,
            "state_cluster_id": self.state_cluster_id,
            "replay_metadata": self.replay_metadata.to_record(include_native_snapshot=include_native_snapshot),
            "similarity_metadata": self.similarity_metadata.to_record() if self.similarity_metadata is not None else None,
            "native_snapshot_reference": self.native_snapshot_reference,
            "metadata": dict(self.metadata),
        }
        if include_native_snapshot and self.native_snapshot is not None:
            record["native_snapshot"] = list(self.native_snapshot)
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ArchivedState":
        """Construct an archived state from a serialized record."""

        native_snapshot = record.get("native_snapshot")
        normalized_snapshot: tuple[Any, ...] | None = None
        if isinstance(native_snapshot, list):
            normalized_snapshot = tuple(native_snapshot)
        elif isinstance(native_snapshot, tuple):
            normalized_snapshot = tuple(native_snapshot)

        archived_state = cls(
            state_id=str(record["state_id"]),
            provenance_trajectory_id=str(record["provenance_trajectory_id"]),
            provenance_timestep=int(record.get("provenance_timestep", 0)),
            achieved_value=float(record.get("achieved_value", 0.0)),
            identity_strategy=str(record.get("identity_strategy", "")),
            first_seen_episode_id=str(record.get("first_seen_episode_id", record.get("provenance_trajectory_id", ""))),
            first_seen_timestep=int(record.get("first_seen_timestep", record.get("provenance_timestep", 0))),
            last_seen_episode_id=str(record.get("last_seen_episode_id", record.get("provenance_trajectory_id", ""))),
            last_seen_timestep=int(record.get("last_seen_timestep", record.get("provenance_timestep", 0))),
            provenance_trajectory_ids=(
                _normalize_string_list(record.get("provenance_trajectory_ids"))
                or [str(record["provenance_trajectory_id"])]
            ),
            visit_count=int(record.get("visit_count", 0)),
            selection_count=int(record.get("selection_count", 0)),
            restore_success_count=int(record.get("restore_success_count", 0)),
            restore_failure_count=int(record.get("restore_failure_count", 0)),
            score_at_state=int(record.get("score_at_state", 0)),
            projected_potential_value=(
                float(record["projected_potential_value"])
                if record.get("projected_potential_value") is not None
                else None
            ),
            frontier_support_count=int(record.get("frontier_support_count", 0)),
            supporting_frontier_trajectory_ids=_normalize_string_list(record.get("supporting_frontier_trajectory_ids")),
            observation_summary=str(record.get("observation_summary", "")),
            inventory_summary=str(record.get("inventory_summary", "")),
            valid_action_summary=str(record.get("valid_action_summary", "")),
            state_cluster_id=str(record.get("state_cluster_id", "")),
            replay_metadata=ReplayMetadata.from_record(record["replay_metadata"])
            if isinstance(record.get("replay_metadata"), MappingABC)
            else ReplayMetadata(),
            similarity_metadata=SimilarityMetadata.from_record(record["similarity_metadata"])
            if isinstance(record.get("similarity_metadata"), MappingABC)
            else None,
            native_snapshot_reference=(
                str(record["native_snapshot_reference"]).strip()
                if record.get("native_snapshot_reference") not in (None, "")
                else None
            ),
            native_snapshot=normalized_snapshot,
            metadata=dict(record["metadata"]) if isinstance(record.get("metadata"), MappingABC) else {},
        )
        archived_state.validate_invariants()
        return archived_state


@dataclass(slots=True)
class CriticalStateAnnotation:
    """A critical state inferred by the global world model."""

    critical_state_id: str
    achieved_value: float
    potential_value: float
    textual_rationale: str
    source_frontier_trajectory_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    support_count: int = 0
    supporting_state_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert the critical-state annotation into a JSON-serializable record."""

        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CriticalStateAnnotation":
        """Construct a critical-state annotation from a serialized record."""

        return cls(
            critical_state_id=str(record["critical_state_id"]),
            achieved_value=float(record.get("achieved_value", 0.0)),
            potential_value=float(record.get("potential_value", 0.0)),
            textual_rationale=str(record.get("textual_rationale", "")),
            source_frontier_trajectory_ids=_normalize_string_list(record.get("source_frontier_trajectory_ids")),
            confidence=float(record.get("confidence", 0.0)),
            support_count=int(record.get("support_count", 0)),
            supporting_state_ids=_normalize_string_list(record.get("supporting_state_ids")),
            metadata=dict(record["metadata"]) if isinstance(record.get("metadata"), MappingABC) else {},
        )


@dataclass(slots=True)
class FrontierInsight:
    """A global-world-model analysis pass over the current trajectory frontier."""

    analysis_id: str
    frontier_trajectory_ids: list[str] = field(default_factory=list)
    inferred_bottlenecks: list[str] = field(default_factory=list)
    partial_solutions: list[str] = field(default_factory=list)
    missing_prerequisites: list[str] = field(default_factory=list)
    candidate_critical_states: list[CriticalStateAnnotation] = field(default_factory=list)
    generated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert the frontier insight into a JSON-serializable record."""

        return {
            "analysis_id": self.analysis_id,
            "frontier_trajectory_ids": list(self.frontier_trajectory_ids),
            "inferred_bottlenecks": list(self.inferred_bottlenecks),
            "partial_solutions": list(self.partial_solutions),
            "missing_prerequisites": list(self.missing_prerequisites),
            "candidate_critical_states": [item.to_record() for item in self.candidate_critical_states],
            "generated_at": self.generated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "FrontierInsight":
        """Construct a frontier insight from a serialized record."""

        return cls(
            analysis_id=str(record["analysis_id"]),
            frontier_trajectory_ids=_normalize_string_list(record.get("frontier_trajectory_ids")),
            inferred_bottlenecks=_normalize_string_list(record.get("inferred_bottlenecks")),
            partial_solutions=_normalize_string_list(record.get("partial_solutions")),
            missing_prerequisites=_normalize_string_list(record.get("missing_prerequisites")),
            candidate_critical_states=[
                CriticalStateAnnotation.from_record(item)
                for item in (record.get("candidate_critical_states") or [])
                if isinstance(item, MappingABC)
            ],
            generated_at=str(record.get("generated_at", "")),
            metadata=dict(record["metadata"]) if isinstance(record.get("metadata"), MappingABC) else {},
        )


@dataclass(slots=True)
class FrontierAnalysisResult:
    """Typed result for one frontier-analysis pass."""

    analysis_id: str
    insight: FrontierInsight
    prompt_input: str
    raw_completion: str = ""
    parse_error: str = ""
    used_fallback: bool = False
    artifact_directory: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert the analysis result into a JSON-serializable record."""

        return {
            "analysis_id": self.analysis_id,
            "insight": self.insight.to_record(),
            "prompt_input": self.prompt_input,
            "raw_completion": self.raw_completion,
            "parse_error": self.parse_error,
            "used_fallback": self.used_fallback,
            "artifact_directory": self.artifact_directory,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "FrontierAnalysisResult":
        """Construct a frontier-analysis result from a serialized record."""

        return cls(
            analysis_id=str(record["analysis_id"]),
            insight=FrontierInsight.from_record(record["insight"])
            if isinstance(record.get("insight"), MappingABC)
            else FrontierInsight(analysis_id=str(record["analysis_id"])),
            prompt_input=str(record.get("prompt_input", "")),
            raw_completion=str(record.get("raw_completion", "")),
            parse_error=str(record.get("parse_error", "")),
            used_fallback=bool(record.get("used_fallback", False)),
            artifact_directory=str(record.get("artifact_directory", "")),
            metadata=dict(record["metadata"]) if isinstance(record.get("metadata"), MappingABC) else {},
        )


@dataclass(slots=True)
class AdvantagePoint:
    """A key state-action point with inferred local advantage information."""

    state_id: str
    action: str
    inferred_advantage: float = 0.0
    outcome_delta: str = ""
    textual_rationale: str = ""
    support_trajectory_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_record(self) -> dict[str, Any]:
        """Convert the advantage point into a JSON-serializable record."""

        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "AdvantagePoint":
        """Construct an advantage point from a serialized record."""

        return cls(
            state_id=str(record["state_id"]),
            action=str(record.get("action", "")),
            inferred_advantage=float(record.get("inferred_advantage", 0.0)),
            outcome_delta=str(record.get("outcome_delta", "")),
            textual_rationale=str(record.get("textual_rationale", "")),
            support_trajectory_ids=_normalize_string_list(record.get("support_trajectory_ids")),
            confidence=float(record.get("confidence", 0.0)),
        )


@dataclass(slots=True)
class AdvantageHint:
    """Local-world-model hint distilled from multiple compared trajectories."""

    root_state_id: str
    compared_trajectory_ids: list[str] = field(default_factory=list)
    key_state_action_points: list[AdvantagePoint] = field(default_factory=list)
    intermediate_advantage_summaries: list[str] = field(default_factory=list)
    action_preferences: list[str] = field(default_factory=list)
    action_avoidances: list[str] = field(default_factory=list)
    textual_reasoning: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert the advantage hint into a JSON-serializable record."""

        return {
            "root_state_id": self.root_state_id,
            "compared_trajectory_ids": list(self.compared_trajectory_ids),
            "key_state_action_points": [item.to_record() for item in self.key_state_action_points],
            "intermediate_advantage_summaries": list(self.intermediate_advantage_summaries),
            "action_preferences": list(self.action_preferences),
            "action_avoidances": list(self.action_avoidances),
            "textual_reasoning": self.textual_reasoning,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "AdvantageHint":
        """Construct an advantage hint from a serialized record."""

        return cls(
            root_state_id=str(record["root_state_id"]),
            compared_trajectory_ids=_normalize_string_list(record.get("compared_trajectory_ids")),
            key_state_action_points=[
                AdvantagePoint.from_record(item)
                for item in (record.get("key_state_action_points") or [])
                if isinstance(item, MappingABC)
            ],
            intermediate_advantage_summaries=_normalize_string_list(record.get("intermediate_advantage_summaries")),
            action_preferences=_normalize_string_list(record.get("action_preferences")),
            action_avoidances=_normalize_string_list(record.get("action_avoidances")),
            textual_reasoning=str(record.get("textual_reasoning", "")),
            confidence=float(record.get("confidence", 0.0)),
            metadata=dict(record["metadata"]) if isinstance(record.get("metadata"), MappingABC) else {},
        )


@dataclass(slots=True)
class SubgoalRecord:
    """A discovered subgoal stored in the local world model."""

    subgoal_id: str
    description: str
    supporting_state_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_record(self) -> dict[str, Any]:
        """Convert the subgoal record into a JSON-serializable record."""

        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SubgoalRecord":
        """Construct a subgoal record from a serialized record."""

        return cls(
            subgoal_id=str(record["subgoal_id"]),
            description=str(record.get("description", "")),
            supporting_state_ids=_normalize_string_list(record.get("supporting_state_ids")),
            confidence=float(record.get("confidence", 0.0)),
        )


@dataclass(slots=True)
class AffordanceRecord:
    """A typed affordance discovered during local exploration."""

    object_text: str
    affordance: str
    supporting_action: str = ""
    confidence: float = 0.0

    def to_record(self) -> dict[str, Any]:
        """Convert the affordance record into a JSON-serializable record."""

        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "AffordanceRecord":
        """Construct an affordance record from a serialized record."""

        return cls(
            object_text=str(record.get("object_text", "")),
            affordance=str(record.get("affordance", "")),
            supporting_action=str(record.get("supporting_action", "")),
            confidence=float(record.get("confidence", 0.0)),
        )


@dataclass(slots=True)
class ActionBiasRecord:
    """A typed action prior or anti-prior in the local world model."""

    action: str
    weight: float
    rationale: str = ""
    support_count: int = 0

    def to_record(self) -> dict[str, Any]:
        """Convert the action bias record into a JSON-serializable record."""

        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ActionBiasRecord":
        """Construct an action bias record from a serialized record."""

        return cls(
            action=str(record.get("action", "")),
            weight=float(record.get("weight", 0.0)),
            rationale=str(record.get("rationale", "")),
            support_count=int(record.get("support_count", 0)),
        )


@dataclass(slots=True)
class LocalWorldModel:
    """Typed local-world-model artifact accumulated from MAR-style comparisons."""

    root_state_id: str
    accumulated_advantage_hints: list[AdvantageHint] = field(default_factory=list)
    discovered_subgoals: list[SubgoalRecord] = field(default_factory=list)
    inferred_affordances: list[AffordanceRecord] = field(default_factory=list)
    action_priors: list[ActionBiasRecord] = field(default_factory=list)
    action_antipriors: list[ActionBiasRecord] = field(default_factory=list)
    last_updated_timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_invariants(self) -> None:
        """Validate local-world-model root consistency."""

        if not self.root_state_id.strip():
            raise ValueError("LocalWorldModel.root_state_id must not be empty.")
        mismatched_roots = {
            hint.root_state_id
            for hint in self.accumulated_advantage_hints
            if hint.root_state_id != self.root_state_id
        }
        if mismatched_roots:
            raise ValueError(
                "All AdvantageHints in a LocalWorldModel must share the same root_state_id."
            )

    def to_record(self) -> dict[str, Any]:
        """Convert the local world model into a JSON-serializable record."""

        return {
            "root_state_id": self.root_state_id,
            "accumulated_advantage_hints": [item.to_record() for item in self.accumulated_advantage_hints],
            "discovered_subgoals": [item.to_record() for item in self.discovered_subgoals],
            "inferred_affordances": [item.to_record() for item in self.inferred_affordances],
            "action_priors": [item.to_record() for item in self.action_priors],
            "action_antipriors": [item.to_record() for item in self.action_antipriors],
            "last_updated_timestamp": self.last_updated_timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "LocalWorldModel":
        """Construct a local world model from a serialized record."""

        world_model = cls(
            root_state_id=str(record["root_state_id"]),
            accumulated_advantage_hints=[
                AdvantageHint.from_record(item)
                for item in (record.get("accumulated_advantage_hints") or [])
                if isinstance(item, MappingABC)
            ],
            discovered_subgoals=[
                SubgoalRecord.from_record(item)
                for item in (record.get("discovered_subgoals") or [])
                if isinstance(item, MappingABC)
            ],
            inferred_affordances=[
                AffordanceRecord.from_record(item)
                for item in (record.get("inferred_affordances") or [])
                if isinstance(item, MappingABC)
            ],
            action_priors=[
                ActionBiasRecord.from_record(item)
                for item in (record.get("action_priors") or [])
                if isinstance(item, MappingABC)
            ],
            action_antipriors=[
                ActionBiasRecord.from_record(item)
                for item in (record.get("action_antipriors") or [])
                if isinstance(item, MappingABC)
            ],
            last_updated_timestamp=str(record.get("last_updated_timestamp", "")),
            metadata=dict(record["metadata"]) if isinstance(record.get("metadata"), MappingABC) else {},
        )
        world_model.validate_invariants()
        return world_model


@dataclass(slots=True)
class LocalWorldModelPromptSummary:
    """Bounded prompt payload distilled from a typed local world model."""

    root_state_id: str
    summary_text: str = ""
    recent_advantage_hints: list[str] = field(default_factory=list)
    discovered_subgoals: list[str] = field(default_factory=list)
    inferred_affordances: list[str] = field(default_factory=list)
    action_priors: list[str] = field(default_factory=list)
    action_antipriors: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        """Convert the prompt summary into a JSON-serializable record."""

        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "LocalWorldModelPromptSummary":
        """Construct a prompt summary from a serialized record."""

        return cls(
            root_state_id=str(record.get("root_state_id", "")),
            summary_text=str(record.get("summary_text", "")),
            recent_advantage_hints=_normalize_string_list(record.get("recent_advantage_hints")),
            discovered_subgoals=_normalize_string_list(record.get("discovered_subgoals")),
            inferred_affordances=_normalize_string_list(record.get("inferred_affordances")),
            action_priors=_normalize_string_list(record.get("action_priors")),
            action_antipriors=_normalize_string_list(record.get("action_antipriors")),
        )

    def has_guidance(self) -> bool:
        """Return whether the summary contains any non-empty local guidance."""

        return any(
            (
                self.summary_text.strip(),
                self.recent_advantage_hints,
                self.discovered_subgoals,
                self.inferred_affordances,
                self.action_priors,
                self.action_antipriors,
            )
        )


@dataclass(slots=True)
class MARInferenceResult:
    """Typed result for one Multi-path Advantage Reflection inference pass."""

    root_state_id: str
    compared_branch_ids: list[str] = field(default_factory=list)
    advantage_hint: AdvantageHint | None = None
    discovered_subgoals: list[SubgoalRecord] = field(default_factory=list)
    inferred_affordances: list[AffordanceRecord] = field(default_factory=list)
    prompt_input: str = ""
    raw_completion: str = ""
    parse_error: str = ""
    used_fallback: bool = False
    artifact_directory: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert the MAR inference result into a JSON-serializable record."""

        return {
            "root_state_id": self.root_state_id,
            "compared_branch_ids": list(self.compared_branch_ids),
            "advantage_hint": self.advantage_hint.to_record() if self.advantage_hint is not None else None,
            "discovered_subgoals": [item.to_record() for item in self.discovered_subgoals],
            "inferred_affordances": [item.to_record() for item in self.inferred_affordances],
            "prompt_input": self.prompt_input,
            "raw_completion": self.raw_completion,
            "parse_error": self.parse_error,
            "used_fallback": self.used_fallback,
            "artifact_directory": self.artifact_directory,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "MARInferenceResult":
        """Construct a MAR inference result from a serialized record."""

        return cls(
            root_state_id=str(record.get("root_state_id", "")),
            compared_branch_ids=_normalize_string_list(record.get("compared_branch_ids")),
            advantage_hint=AdvantageHint.from_record(record["advantage_hint"])
            if isinstance(record.get("advantage_hint"), MappingABC)
            else None,
            discovered_subgoals=[
                SubgoalRecord.from_record(item)
                for item in (record.get("discovered_subgoals") or [])
                if isinstance(item, MappingABC)
            ],
            inferred_affordances=[
                AffordanceRecord.from_record(item)
                for item in (record.get("inferred_affordances") or [])
                if isinstance(item, MappingABC)
            ],
            prompt_input=str(record.get("prompt_input", "")),
            raw_completion=str(record.get("raw_completion", "")),
            parse_error=str(record.get("parse_error", "")),
            used_fallback=bool(record.get("used_fallback", False)),
            artifact_directory=str(record.get("artifact_directory", "")),
            metadata=dict(record["metadata"]) if isinstance(record.get("metadata"), MappingABC) else {},
        )


@dataclass(slots=True)
class GlowSelectionDecisionMetric:
    """Structured metrics for one archive-state selection decision."""

    cycle_index: int
    selected_state_id: str | None
    root_state_id: str | None
    replay_method: str = ""
    achieved_contribution: float = 0.0
    potential_contribution: float = 0.0
    rationale: str = ""
    frontier_size_before: int = 0
    frontier_size_after: int = 0
    branch_count: int = 0
    restore_succeeded: bool = False
    used_llm_adjudication: bool = False
    selected_frontier_trajectory_ids: list[str] = field(default_factory=list)
    selected_critical_state_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert the selection metric into a JSON-serializable record."""

        return asdict(self)


@dataclass(slots=True)
class GlowEpisodeMetrics:
    """Structured per-episode metrics for the GLoW control loop."""

    episode_id: str
    seed: int
    environment_interactions: int
    max_score: int
    final_score: int
    frontier_size_over_time: list[int] = field(default_factory=list)
    frontier_analysis_count: int = 0
    frontier_analysis_ids: list[str] = field(default_factory=list)
    mar_update_count: int = 0
    restore_attempt_count: int = 0
    restore_success_count: int = 0
    selected_state_decisions: list[GlowSelectionDecisionMetric] = field(default_factory=list)
    branch_counts_per_root_state: dict[str, int] = field(default_factory=dict)
    local_world_model_root_ids: list[str] = field(default_factory=list)
    frontier_analysis_artifact_directories: list[str] = field(default_factory=list)
    selection_artifact_directories: list[str] = field(default_factory=list)
    mar_artifact_directories: list[str] = field(default_factory=list)
    local_world_model_snapshot_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert the episode metrics into a JSON-serializable record."""

        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "environment_interactions": self.environment_interactions,
            "max_score": self.max_score,
            "final_score": self.final_score,
            "frontier_size_over_time": list(self.frontier_size_over_time),
            "frontier_analysis_count": self.frontier_analysis_count,
            "frontier_analysis_ids": list(self.frontier_analysis_ids),
            "mar_update_count": self.mar_update_count,
            "restore_attempt_count": self.restore_attempt_count,
            "restore_success_count": self.restore_success_count,
            "selected_state_decisions": [item.to_record() for item in self.selected_state_decisions],
            "branch_counts_per_root_state": dict(self.branch_counts_per_root_state),
            "local_world_model_root_ids": list(self.local_world_model_root_ids),
            "frontier_analysis_artifact_directories": list(self.frontier_analysis_artifact_directories),
            "selection_artifact_directories": list(self.selection_artifact_directories),
            "mar_artifact_directories": list(self.mar_artifact_directories),
            "local_world_model_snapshot_paths": list(self.local_world_model_snapshot_paths),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "GlowEpisodeMetrics":
        """Construct episode metrics from a serialized record."""

        return cls(
            episode_id=str(record.get("episode_id", "")),
            seed=int(record.get("seed", 0)),
            environment_interactions=int(record.get("environment_interactions", 0)),
            max_score=int(record.get("max_score", 0)),
            final_score=int(record.get("final_score", 0)),
            frontier_size_over_time=[int(value) for value in (record.get("frontier_size_over_time") or [])],
            frontier_analysis_count=int(record.get("frontier_analysis_count", 0)),
            frontier_analysis_ids=_normalize_string_list(record.get("frontier_analysis_ids")),
            mar_update_count=int(record.get("mar_update_count", 0)),
            restore_attempt_count=int(record.get("restore_attempt_count", 0)),
            restore_success_count=int(record.get("restore_success_count", 0)),
            selected_state_decisions=[
                GlowSelectionDecisionMetric(**dict(item))
                for item in (record.get("selected_state_decisions") or [])
                if isinstance(item, MappingABC)
            ],
            branch_counts_per_root_state={
                str(key): int(value)
                for key, value in dict(record.get("branch_counts_per_root_state", {})).items()
            },
            local_world_model_root_ids=_normalize_string_list(record.get("local_world_model_root_ids")),
            frontier_analysis_artifact_directories=_normalize_string_list(
                record.get("frontier_analysis_artifact_directories")
            ),
            selection_artifact_directories=_normalize_string_list(record.get("selection_artifact_directories")),
            mar_artifact_directories=_normalize_string_list(record.get("mar_artifact_directories")),
            local_world_model_snapshot_paths=_normalize_string_list(record.get("local_world_model_snapshot_paths")),
            metadata=dict(record["metadata"]) if isinstance(record.get("metadata"), MappingABC) else {},
        )


@dataclass(slots=True)
class GlowRunMetrics:
    """Structured run-level metrics and aggregate statistics for GLoW experiments."""

    run_id: str
    config_path: str
    episode_count: int
    seed_values: list[int]
    mean_final_score: float
    std_final_score: float
    mean_max_score: float
    std_max_score: float
    mean_environment_interactions: float
    std_environment_interactions: float
    total_restore_attempt_count: int = 0
    total_restore_success_count: int = 0
    total_frontier_analysis_count: int = 0
    total_mar_update_count: int = 0
    episode_metric_paths: list[str] = field(default_factory=list)
    episode_summary_paths: list[str] = field(default_factory=list)
    trajectory_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    summary_path: str = ""

    def to_record(self) -> dict[str, Any]:
        """Convert run-level metrics into a JSON-serializable record."""

        return asdict(self)


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
    metrics_path: Path | None = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert the episode summary into a serializable dictionary."""

        payload = asdict(self)
        payload["trajectory_path"] = str(self.trajectory_path)
        payload["summary_path"] = str(self.summary_path) if self.summary_path is not None else None
        payload["metrics_path"] = str(self.metrics_path) if self.metrics_path is not None else None
        return payload
