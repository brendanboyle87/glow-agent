"""Persistent store for MAR-derived local world models.

The paper-faithful intent is to keep local guidance tied to a restored root state
and accumulate typed advantage hints over repeated local rollouts. This store keeps
that state inspectable on disk without changing the current CLI artifact layout.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

from zork_agent.types import LocalWorldModel, MARInferenceResult, RootDepthProgressionProfile
from zork_agent.utils.serialization import to_jsonable


class LocalWorldModelStore:
    """Persist local world models and MAR analysis bundles keyed by root state id."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def model_dir(self, root_state_id: str) -> Path:
        """Return the directory used to store one root state's local world model."""

        return self.root_dir / _safe_root_id(root_state_id)

    def model_path(self, root_state_id: str) -> Path:
        """Return the `model.json` path for one root state's local world model."""

        return self.model_dir(root_state_id) / "model.json"

    def read(self, root_state_id: str) -> LocalWorldModel | None:
        """Load a local world model if it has been written previously."""

        path = self.model_path(root_state_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return LocalWorldModel.from_record(payload)

    def write(self, model: LocalWorldModel) -> Path:
        """Persist the current local world model snapshot to disk."""

        model.validate_invariants()
        model_dir = self.model_dir(model.root_state_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / "model.json"
        path.write_text(
            json.dumps(to_jsonable(model.to_record()), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._write_depth_progression_summary(model_dir=model_dir, model=model)
        return path

    def _write_depth_progression_summary(self, *, model_dir: Path, model: LocalWorldModel) -> None:
        """Persist a concise per-root depth progression summary alongside the model."""

        profile = _depth_progression_profile(model)
        if profile is None:
            return
        summary_path = model_dir / "depth_progression_summary.json"
        summary_path.write_text(
            json.dumps(to_jsonable(_profile_summary_payload(model, profile)), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_mar_result(self, result: MARInferenceResult, updated_model: LocalWorldModel) -> Path:
        """Persist one MAR prompt/completion/result bundle plus the updated model snapshot."""

        model_dir = self.model_dir(result.root_state_id)
        updates_dir = model_dir / "updates"
        updates_dir.mkdir(parents=True, exist_ok=True)
        update_dir = updates_dir / _update_id(result)
        update_dir.mkdir(parents=True, exist_ok=True)
        result.artifact_directory = str(update_dir)

        (update_dir / "prompt.txt").write_text(result.prompt_input, encoding="utf-8")
        (update_dir / "raw_completion.txt").write_text(result.raw_completion, encoding="utf-8")
        (update_dir / "mar_result.json").write_text(
            json.dumps(to_jsonable(result.to_record()), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        self.write(updated_model)
        (update_dir / "updated_model.json").write_text(
            json.dumps(to_jsonable(updated_model.to_record()), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return update_dir


def _safe_root_id(root_state_id: str) -> str:
    """Convert a root state id into a filename-safe directory name."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", root_state_id.strip())
    return normalized or "unknown-root"


def _update_id(result: MARInferenceResult) -> str:
    """Return a deterministic-ish update id for one MAR result bundle."""

    metadata_id = str(result.metadata.get("analysis_id", "")).strip()
    if metadata_id:
        return metadata_id
    if result.compared_branch_ids:
        return f"branches-{'-'.join(result.compared_branch_ids[:4])}"
    return "mar-update"


def _depth_progression_profile(model: LocalWorldModel) -> RootDepthProgressionProfile | None:
    """Return the persisted depth-progression profile from model metadata, if present."""

    raw_profile = model.metadata.get("depth_progression")
    if not isinstance(raw_profile, dict):
        return None
    profile = RootDepthProgressionProfile.from_record(raw_profile)
    if not profile.root_state_id:
        profile.root_state_id = model.root_state_id
    return profile


def _top_bias_actions(biases, *, limit: int = 5) -> list[str]:
    """Return the strongest action biases for summary artifacts."""

    return [
        bias.action
        for bias in sorted(biases, key=lambda item: (-item.weight, item.action))[:limit]
        if bias.action.strip()
    ]


def _top_affordance_objects(model: LocalWorldModel, *, limit: int = 5) -> list[str]:
    """Return the strongest affordance object hints for summary artifacts."""

    return [
        record.object_text
        for record in model.inferred_affordances[:limit]
        if record.object_text.strip()
    ]


def _profile_summary_payload(model: LocalWorldModel, profile: RootDepthProgressionProfile) -> dict[str, object]:
    """Build a concise per-root depth progression summary artifact."""

    return {
        "root_state_id": model.root_state_id,
        "productive_root": profile.productive_root,
        "productive_commit_count": profile.productive_commit_count,
        "materially_distinct_commit_count": profile.materially_distinct_commit_count,
        "repeated_winning_continuation_count": profile.repeated_winning_continuation_count,
        "replay_saturation_level": profile.replay_saturation_level,
        "replay_saturation_state": (
            "saturated" if profile.replay_saturation_level > 0 else "unsaturated"
        ),
        "unique_committed_prefix_count": profile.unique_committed_prefix_count,
        "last_committed_prefix_signature": profile.last_committed_prefix_signature,
        "last_replay_saturation_penalty": profile.last_replay_saturation_penalty,
        "last_within_root_novelty_bonus": profile.last_within_root_novelty_bonus,
        "last_within_root_depth_bonus": profile.last_within_root_depth_bonus,
        "unexplored_continuation_indicators": {
            "saturated_try_actions": list(profile.saturated_try_actions),
            "saturated_prefix_signatures": list(profile.saturated_prefix_signatures),
            "diversity_lag": max(
                0,
                profile.repeated_winning_continuation_count
                - max(0, profile.materially_distinct_commit_count - 1),
            ),
        },
        "local_model_priors": _top_bias_actions(model.action_priors),
        "local_model_antipriors": _top_bias_actions(model.action_antipriors),
        "local_model_objects": _top_affordance_objects(model),
        "committed_branch_history": list(profile.committed_branch_history[-12:]),
    }
