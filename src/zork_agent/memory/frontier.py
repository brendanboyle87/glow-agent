"""Heuristic frontier bookkeeping for replayable state expansion.

The scoring and deduplication logic here are explicit engineering approximations,
not a paper-faithful implementation of GLoW. The intent is to keep the revisit
policy easy to inspect and tune while the research loop is still changing.

TODO: revisit the heuristic weights once real Zork runs generate meaningful logs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from zork_agent.types import FrontierEntry, StateCandidate, derive_state_cluster_id, derive_state_family_key


@dataclass(slots=True)
class FrontierScoringConfig:
    """Weights and thresholds for the heuristic frontier ranking."""

    score_weight: float = 1.0
    novelty_weight: float = 0.75
    recent_gain_weight: float = 0.5
    depth_weight: float = 0.15
    strategic_score_weight: float = 1.0
    loop_penalty_weight: float = 1.0
    movement_penalty_weight: float = 1.0
    room_text_only_penalty_weight: float = 0.75
    reversible_state_penalty_weight: float = 1.0
    revisit_saturation_penalty_weight: float = 1.0
    cluster_revisit_saturation_weight: float = 0.75
    reversible_toggle_novelty_scale: float = 0.2
    trivial_observation_novelty_scale: float = 0.35
    oscillating_pair_novelty_scale: float = 0.2
    family_repeat_novelty_decay: float = 0.75
    family_repeat_penalty: float = 0.5
    family_revisit_saturation_weight: float = 0.5
    base_reversible_state_penalty: float = 0.75
    oscillating_pair_penalty: float = 1.0
    trivial_reversible_penalty: float = 0.5
    similarity_threshold: float = 0.85
    snapshot_retention_limit: int = 8

    def __post_init__(self) -> None:
        """Validate heuristic frontier settings."""

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0.")
        if self.snapshot_retention_limit < 0:
            raise ValueError("snapshot_retention_limit must be >= 0.")
        for name in (
            "score_weight",
            "novelty_weight",
            "recent_gain_weight",
            "depth_weight",
            "strategic_score_weight",
            "loop_penalty_weight",
            "movement_penalty_weight",
            "room_text_only_penalty_weight",
            "reversible_state_penalty_weight",
            "revisit_saturation_penalty_weight",
            "cluster_revisit_saturation_weight",
            "reversible_toggle_novelty_scale",
            "trivial_observation_novelty_scale",
            "oscillating_pair_novelty_scale",
            "family_repeat_novelty_decay",
            "family_repeat_penalty",
            "family_revisit_saturation_weight",
            "base_reversible_state_penalty",
            "oscillating_pair_penalty",
            "trivial_reversible_penalty",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0.0.")


class FrontierQueue:
    """Small sorted frontier with heuristic ranking and near-duplicate suppression."""

    def __init__(self, max_size: int, scoring: FrontierScoringConfig | None = None):
        # TODO: switch to a heap only if profiles show this list-based queue is too slow.
        if max_size <= 0:
            raise ValueError("FrontierQueue max_size must be greater than 0.")
        self.max_size = max_size
        self.scoring = scoring or FrontierScoringConfig()
        self._entries: list[FrontierEntry] = []
        self._state_no_progress_revisits: dict[str, int] = {}
        self._family_no_progress_revisits: dict[str, int] = {}
        self._cluster_no_progress_revisits: dict[str, int] = {}

    def add(self, entry: FrontierEntry) -> None:
        """Insert one entry, deduplicate conservatively, and keep the frontier clipped."""

        scored_entry = self._prepare_entry(entry)
        self._apply_dynamic_penalties(scored_entry)
        scored_entry.priority = self.score_entry(scored_entry)
        duplicate_index = self._find_duplicate_index(scored_entry)
        if duplicate_index is not None:
            existing = self._entries[duplicate_index]
            self._apply_dynamic_penalties(existing)
            existing.priority = self.score_entry(existing)
            if self._is_better(scored_entry, existing):
                self._entries[duplicate_index] = scored_entry
        else:
            self._entries.append(scored_entry)
        self._resort_and_trim()

    def add_candidate(self, candidate: StateCandidate, *, notes: str = "") -> None:
        """Convert a replayable state candidate into a frontier entry and add it."""

        self.add(FrontierEntry.from_candidate(candidate, notes=notes))

    def pop_best(self) -> FrontierEntry | None:
        """Pop and return the best frontier entry."""

        if not self._entries:
            return None
        return self._entries.pop(0)

    def peek_best(self) -> FrontierEntry | None:
        """Return the current best entry without removing it."""

        return self._entries[0] if self._entries else None

    def top_k(self, k: int) -> list[FrontierEntry]:
        """Return the top-ranked frontier entries."""

        return list(self._entries[: max(k, 0)])

    def top_k_replay_targets(self, k: int) -> list[StateCandidate]:
        """Return the top-ranked replayable state candidates."""

        return [entry.to_state_candidate() for entry in self.top_k(k)]

    def snapshot(self) -> list[FrontierEntry]:
        """Return a copy of the current frontier ordering."""

        return list(self._entries)

    def __len__(self) -> int:
        """Return the number of retained entries."""

        return len(self._entries)

    def score_entry(self, entry: FrontierEntry) -> float:
        """Score an entry with an explicit weighted heuristic formula.

        Formula:
        `priority = score_w * score + novelty_w * effective_novelty + gain_w * recent_gain
        - depth_w * depth - loop_w * loop_penalty - reversible_w * reversible_state_penalty
        - revisit_w * revisit_saturation_penalty`
        """

        return (
            self.scoring.score_weight * float(entry.score)
            + self.scoring.novelty_weight * float(entry.effective_novelty or entry.novelty)
            + self.scoring.recent_gain_weight * float(entry.recent_gain)
            + self.scoring.strategic_score_weight * float(entry.strategic_value)
            - self.scoring.depth_weight * float(entry.depth)
            - self.scoring.loop_penalty_weight * float(entry.loop_penalty)
            - self.scoring.movement_penalty_weight * float(entry.movement_penalty)
            - self.scoring.room_text_only_penalty_weight * float(entry.room_text_only_gain)
            - self.scoring.reversible_state_penalty_weight * float(entry.reversible_state_penalty)
            - self.scoring.revisit_saturation_penalty_weight * float(entry.revisit_saturation_penalty)
        )

    def note_revisit_outcome(
        self,
        entry_or_state_id: FrontierEntry | str,
        *,
        state_family_key: str | None = None,
        made_progress: bool,
    ) -> None:
        """Record whether revisiting a frontier node produced real progress.

        No-progress revisits saturate both the specific node and its broader state family,
        which lowers future priority for trivial reversible toggles.
        """

        if isinstance(entry_or_state_id, FrontierEntry):
            state_id = entry_or_state_id.state_id
            family_key = entry_or_state_id.state_family_key or state_family_key or ""
            cluster_key = entry_or_state_id.state_cluster_id or ""
        else:
            state_id = entry_or_state_id
            family_key = state_family_key or ""
            cluster_key = ""

        if made_progress:
            self._state_no_progress_revisits.pop(state_id, None)
            if family_key:
                self._family_no_progress_revisits.pop(family_key, None)
            if cluster_key:
                self._cluster_no_progress_revisits.pop(cluster_key, None)
        else:
            self._state_no_progress_revisits[state_id] = self._state_no_progress_revisits.get(state_id, 0) + 1
            if family_key:
                self._family_no_progress_revisits[family_key] = (
                    self._family_no_progress_revisits.get(family_key, 0) + 1
                )
            if cluster_key:
                self._cluster_no_progress_revisits[cluster_key] = (
                    self._cluster_no_progress_revisits.get(cluster_key, 0) + 1
                )
        self._resort_and_trim()

    def _find_duplicate_index(self, candidate: FrontierEntry) -> int | None:
        """Return the index of a near-identical entry if one already exists."""

        for index, existing in enumerate(self._entries):
            if self._are_duplicates(existing, candidate):
                return index
        return None

    def _are_duplicates(self, first: FrontierEntry, second: FrontierEntry) -> bool:
        """Detect near-identical entries using hashes and token overlap."""

        if first.world_state_hash != "unknown" and first.world_state_hash == second.world_state_hash:
            return True
        if first.state_id and first.state_id == second.state_id:
            return True
        if first.dedupe_key and first.dedupe_key == second.dedupe_key:
            return True
        similarity = self._text_similarity(first.dedupe_key, second.dedupe_key)
        return similarity >= self.scoring.similarity_threshold

    def _build_dedupe_key(self, entry: FrontierEntry) -> str:
        """Build the text fingerprint used for near-duplicate suppression."""

        if entry.summary_text.strip():
            return self._normalize_text(entry.summary_text)
        if entry.observation.strip() or entry.inventory_text.strip():
            return self._normalize_text(f"{entry.observation} {entry.inventory_text}")
        return self._normalize_text(entry.state_id or entry.world_state_hash)

    def _text_similarity(self, first: str, second: str) -> float:
        """Compute a simple token-overlap similarity between two texts."""

        first_tokens = set(first.split())
        second_tokens = set(second.split())
        if not first_tokens or not second_tokens:
            return 0.0
        return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for heuristic deduplication."""

        return " ".join(text.split()).strip().lower()

    def _is_better(self, candidate: FrontierEntry, existing: FrontierEntry) -> bool:
        """Return whether a duplicate candidate should replace an existing entry."""

        return self._entry_sort_key(candidate) < self._entry_sort_key(existing)

    def _entry_sort_key(self, entry: FrontierEntry) -> tuple[float, int, str, int, str]:
        """Build the deterministic ordering key for retained entries."""

        return (-entry.priority, entry.depth, entry.episode_id, entry.step_index, entry.state_id)

    def _resort_and_trim(self) -> None:
        """Sort entries by frontier priority and clip to max size."""

        self._refresh_dynamic_penalties()
        self._entries.sort(key=self._entry_sort_key)
        if len(self._entries) > self.max_size:
            self._entries = self._entries[: self.max_size]
        self._prune_native_snapshots()

    def _prune_native_snapshots(self) -> None:
        """Retain native snapshots only for the top configured frontier entries."""

        retained = 0
        for entry in self._entries:
            if entry.saved_node is None or not entry.saved_node.has_native_state:
                continue
            if retained < self.scoring.snapshot_retention_limit:
                retained += 1
                continue
            entry.saved_node = entry.saved_node.without_native_state()

    def _prepare_entry(self, entry: FrontierEntry) -> FrontierEntry:
        """Populate derived ranking fields used by the frontier."""

        if not entry.dedupe_key:
            entry.dedupe_key = self._build_dedupe_key(entry)
        if not entry.state_family_key:
            entry.state_family_key = derive_state_family_key(
                observation=entry.observation,
                inventory_text=entry.inventory_text,
                valid_actions=entry.valid_actions,
                summary_text=entry.summary_text or entry.state_id,
            )
        if not entry.state_cluster_id:
            entry.state_cluster_id = derive_state_cluster_id(
                observation=entry.observation,
                inventory_text=entry.inventory_text,
                valid_actions=entry.valid_actions,
                summary_text=entry.summary_text or entry.state_id,
            )
        entry.metadata.setdefault("base_reversible_state_penalty", float(entry.reversible_state_penalty))
        entry.effective_novelty = float(entry.novelty)
        entry.reversible_state_penalty = float(entry.reversible_state_penalty)
        entry.revisit_saturation_penalty = float(entry.revisit_saturation_penalty)
        entry.no_progress_revisit_count = int(entry.no_progress_revisit_count)
        entry.family_no_progress_revisit_count = int(entry.family_no_progress_revisit_count)
        entry.cluster_no_progress_revisit_count = int(entry.cluster_no_progress_revisit_count)
        entry.cluster_visit_count = int(entry.cluster_visit_count)
        entry.region_novelty_score = float(entry.region_novelty_score or 1.0)
        entry.strategic_value = float(entry.strategic_value)
        entry.unresolved_opportunity_count = int(entry.unresolved_opportunity_count)
        entry.movement_penalty = float(entry.movement_penalty)
        entry.affordance_gain = int(entry.affordance_gain)
        entry.room_text_only_gain = float(entry.room_text_only_gain)
        return entry

    def _refresh_dynamic_penalties(self) -> None:
        """Recompute dynamic novelty decay and revisit penalties for all entries."""

        for entry in self._entries:
            self._apply_dynamic_penalties(entry)
            entry.priority = self.score_entry(entry)
            if not math.isfinite(entry.priority):
                raise ValueError(f"Frontier entry priority must be finite: {entry.state_id}")

    def _apply_dynamic_penalties(self, entry: FrontierEntry) -> None:
        """Apply reversible-state and revisit-saturation adjustments to one entry."""

        family_occurrences = self._family_occurrence_count(entry)
        cluster_occurrences = self._cluster_occurrence_count(entry)
        similar_family_seen = self._similar_family_state_seen(entry)

        novelty_scale = 1.0
        reversible_penalty = float(entry.metadata.get("base_reversible_state_penalty", 0.0))
        if entry.state_family_key.startswith("toggle:"):
            novelty_scale *= self.scoring.reversible_toggle_novelty_scale
            reversible_penalty += self.scoring.base_reversible_state_penalty
        if similar_family_seen or entry.trivial_reversible_change:
            novelty_scale *= self.scoring.trivial_observation_novelty_scale
            reversible_penalty += self.scoring.trivial_reversible_penalty
        if entry.oscillating_pair_member:
            novelty_scale *= self.scoring.oscillating_pair_novelty_scale
            reversible_penalty += self.scoring.oscillating_pair_penalty
        if family_occurrences > 0:
            novelty_scale /= 1.0 + family_occurrences * self.scoring.family_repeat_novelty_decay
            reversible_penalty += family_occurrences * self.scoring.family_repeat_penalty
        if cluster_occurrences > 0:
            novelty_scale /= 1.0 + float(cluster_occurrences)

        state_failures = self._state_no_progress_revisits.get(entry.state_id, 0)
        family_failures = self._family_no_progress_revisits.get(entry.state_family_key, 0)
        cluster_failures = self._cluster_no_progress_revisits.get(entry.state_cluster_id, 0)
        entry.effective_novelty = float(entry.novelty) * novelty_scale * max(entry.region_novelty_score, 0.05)
        if entry.room_text_only_gain > 0.0 and entry.affordance_gain <= 0 and entry.recent_gain <= 0.0:
            entry.effective_novelty *= 0.25
        entry.reversible_state_penalty = reversible_penalty
        entry.no_progress_revisit_count = state_failures
        entry.family_no_progress_revisit_count = family_failures
        entry.cluster_no_progress_revisit_count = cluster_failures
        entry.revisit_saturation_penalty = float(state_failures) + (
            float(family_failures) * self.scoring.family_revisit_saturation_weight
        ) + (
            float(cluster_failures) * self.scoring.cluster_revisit_saturation_weight
        )

    def _family_occurrence_count(self, candidate: FrontierEntry) -> int:
        """Return how many other retained entries share the same family key."""

        if not candidate.state_family_key:
            return 0
        return sum(
            1
            for entry in self._entries
            if entry is not candidate and entry.state_family_key == candidate.state_family_key
        )

    def _cluster_occurrence_count(self, candidate: FrontierEntry) -> int:
        """Return how many retained entries share the same region cluster."""

        if not candidate.state_cluster_id:
            return 0
        return sum(
            1
            for entry in self._entries
            if entry is not candidate and entry.state_cluster_id == candidate.state_cluster_id
        )

    def _similar_family_state_seen(self, candidate: FrontierEntry) -> bool:
        """Return whether a near-identical family member is already retained."""

        if not candidate.state_family_key:
            return False
        for entry in self._entries:
            if entry is candidate or entry.state_family_key != candidate.state_family_key:
                continue
            if self._text_similarity(entry.dedupe_key, candidate.dedupe_key) >= self.scoring.similarity_threshold:
                return True
        return False
