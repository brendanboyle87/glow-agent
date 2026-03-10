"""Configuration models and YAML loading helpers for the scaffold.

TODO: add schema versioning and provider-specific config validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from zork_agent.types import StateSelectionMode, default_inverse_action_pairs, normalize_inverse_action_pairs


class StrictModel(BaseModel):
    """Base model that rejects unexpected keys."""

    model_config = ConfigDict(extra="forbid")


class RuntimeConfig(StrictModel):
    """Runtime details for Jericho and container execution."""

    jericho_game_path: Path
    use_live_jericho: bool = False
    docker_image: str = "glow-agent-jericho:latest"
    container_workdir: str = "/workspace"
    docker_notes: list[str] = Field(default_factory=list)

    @field_validator("jericho_game_path")
    @classmethod
    def validate_game_path(cls, value: Path) -> Path:
        """Reject empty runtime game paths."""

        if not str(value).strip():
            raise ValueError("runtime.jericho_game_path must not be empty.")
        return value


class LLMConfig(StrictModel):
    """Configuration for a local LLM backend."""

    provider: str = "lmstudio"
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = "lm-studio"
    model_name: str = "local-model"
    temperature: float = 0.2
    max_tokens: int = 256
    request_timeout_seconds: float = 60.0
    offline_stub: bool = True

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """Validate the LM Studio base URL early so config errors are obvious."""

        normalized = value.strip().rstrip("/")
        if not normalized:
            raise ValueError("llm.base_url must not be empty.")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("llm.base_url must start with http:// or https://.")
        return normalized

    @field_validator("model_name", "api_key")
    @classmethod
    def validate_nonempty_strings(cls, value: str, info) -> str:
        """Reject empty provider strings."""

        normalized = value.strip()
        if not normalized:
            raise ValueError(f"llm.{info.field_name} must not be empty.")
        return normalized

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Keep temperature in a conservative range."""

        if not 0.0 <= value <= 2.0:
            raise ValueError("llm.temperature must be between 0.0 and 2.0.")
        return value

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int) -> int:
        """Require a positive token budget."""

        if value <= 0:
            raise ValueError("llm.max_tokens must be greater than 0.")
        return value

    @field_validator("request_timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        """Require a positive request timeout."""

        if value <= 0:
            raise ValueError("llm.request_timeout_seconds must be greater than 0.")
        return value


class PromptConfig(StrictModel):
    """Prompt template locations."""

    directory: Path
    system_file: str = "system.txt"
    action_proposal_file: str = "action_proposal.txt"
    trajectory_analysis_file: str = "trajectory_analysis.txt"
    local_reflection_file: str = "local_reflection.txt"
    state_selection_file: str = "state_selection.txt"

    @field_validator(
        "system_file",
        "action_proposal_file",
        "trajectory_analysis_file",
        "local_reflection_file",
        "state_selection_file",
    )
    @classmethod
    def validate_prompt_filenames(cls, value: str, info) -> str:
        """Reject empty prompt filenames."""

        normalized = value.strip()
        if not normalized:
            raise ValueError(f"prompts.{info.field_name} must not be empty.")
        return normalized


class PolicyConfig(StrictModel):
    """Paper-inspired control knobs exposed for reproducibility."""

    rollout_count: int = 4
    rollout_depth: int = 6
    frontier_max_size: int = 32
    snapshot_retention_limit: int = 8
    archive_top_k: int = 8
    action_candidates: int = 4
    state_selection_mode: StateSelectionMode = StateSelectionMode.HEURISTIC
    inverse_action_pairs: dict[str, str] = Field(default_factory=default_inverse_action_pairs)
    immediate_inverse_penalty: float = 0.75
    repeated_pair_penalty: float = 1.0
    reversible_no_progress_penalty: float = 1.0
    repeated_movement_penalty: float = 1.0
    movement_cycle_penalty: float = 1.25
    same_region_repeat_penalty: float = 0.75
    action_untried_bonus: float = 2.0
    action_new_noun_bonus: float = 1.0
    action_new_noun_interaction_bonus: float = 1.25
    action_repeated_no_gain_penalty: float = 1.0
    branch_progress_score_weight: float = 1.0
    branch_progress_inventory_weight: float = 1.0
    branch_progress_affordance_weight: float = 0.35
    branch_progress_location_weight: float = 1.0
    branch_progress_object_weight: float = 0.25
    branch_progress_loop_reduction_weight: float = 0.5
    room_text_only_weight: float = 0.1
    movement_progress_cap: float = 0.5
    min_affordance_gain_for_movement_commit: int = 1
    branch_commit_min_progress_score: float = 1.0
    frontier_loop_penalty_weight: float = 1.0
    frontier_movement_penalty_weight: float = 1.0
    frontier_room_text_only_penalty_weight: float = 0.75
    frontier_reversible_state_penalty_weight: float = 1.0
    frontier_revisit_saturation_penalty_weight: float = 1.0
    frontier_cluster_revisit_saturation_weight: float = 0.75
    frontier_reversible_toggle_novelty_scale: float = 0.2
    frontier_trivial_observation_novelty_scale: float = 0.35
    frontier_oscillating_pair_novelty_scale: float = 0.2
    frontier_family_repeat_novelty_decay: float = 0.75
    frontier_family_repeat_penalty: float = 0.5
    frontier_family_revisit_saturation_weight: float = 0.5
    frontier_base_reversible_state_penalty: float = 0.75
    frontier_oscillating_pair_penalty: float = 1.0
    frontier_trivial_reversible_penalty: float = 0.5

    @field_validator(
        "rollout_count",
        "rollout_depth",
        "frontier_max_size",
        "archive_top_k",
        "action_candidates",
    )
    @classmethod
    def validate_positive_policy_values(cls, value: int, info) -> int:
        """Require positive heuristic budgets."""

        if value <= 0:
            raise ValueError(f"policy.{info.field_name} must be greater than 0.")
        return value

    @field_validator("snapshot_retention_limit")
    @classmethod
    def validate_snapshot_retention_limit(cls, value: int) -> int:
        """Require a non-negative retention limit."""

        if value < 0:
            raise ValueError("policy.snapshot_retention_limit must be >= 0.")
        return value

    @field_validator("inverse_action_pairs")
    @classmethod
    def validate_inverse_action_pairs(cls, value: dict[str, str]) -> dict[str, str]:
        """Normalize inverse-action mappings into a symmetric lower-case form."""

        normalized = normalize_inverse_action_pairs(value)
        if not normalized:
            raise ValueError("policy.inverse_action_pairs must define at least one inverse pair.")
        return normalized

    @field_validator(
        "immediate_inverse_penalty",
        "repeated_pair_penalty",
        "reversible_no_progress_penalty",
        "repeated_movement_penalty",
        "movement_cycle_penalty",
        "same_region_repeat_penalty",
        "action_untried_bonus",
        "action_new_noun_bonus",
        "action_new_noun_interaction_bonus",
        "action_repeated_no_gain_penalty",
        "branch_progress_score_weight",
        "branch_progress_inventory_weight",
        "branch_progress_affordance_weight",
        "branch_progress_location_weight",
        "branch_progress_object_weight",
        "branch_progress_loop_reduction_weight",
        "room_text_only_weight",
        "movement_progress_cap",
        "branch_commit_min_progress_score",
        "frontier_loop_penalty_weight",
        "frontier_movement_penalty_weight",
        "frontier_room_text_only_penalty_weight",
        "frontier_reversible_state_penalty_weight",
        "frontier_revisit_saturation_penalty_weight",
        "frontier_cluster_revisit_saturation_weight",
        "frontier_reversible_toggle_novelty_scale",
        "frontier_trivial_observation_novelty_scale",
        "frontier_oscillating_pair_novelty_scale",
        "frontier_family_repeat_novelty_decay",
        "frontier_family_repeat_penalty",
        "frontier_family_revisit_saturation_weight",
        "frontier_base_reversible_state_penalty",
        "frontier_oscillating_pair_penalty",
        "frontier_trivial_reversible_penalty",
    )
    @classmethod
    def validate_nonnegative_penalties(cls, value: float, info) -> float:
        """Require non-negative oscillation penalties."""

        if value < 0.0:
            raise ValueError(f"policy.{info.field_name} must be >= 0.0.")
        return value

    @field_validator("min_affordance_gain_for_movement_commit")
    @classmethod
    def validate_min_affordance_gain_for_movement_commit(cls, value: int) -> int:
        """Require a non-negative affordance threshold for movement commits."""

        if value < 0:
            raise ValueError("policy.min_affordance_gain_for_movement_commit must be >= 0.")
        return value


class PathsConfig(StrictModel):
    """Artifact, archive, and log locations."""

    artifacts_dir: Path
    trajectory_dir: Path
    archive_dir: Path
    log_dir: Path
    summary_dir: Path

    @model_validator(mode="after")
    def validate_unique_directories(self) -> "PathsConfig":
        """Reject duplicated artifact directories that would collapse outputs."""

        path_values = {
            "artifacts_dir": self.artifacts_dir,
            "trajectory_dir": self.trajectory_dir,
            "archive_dir": self.archive_dir,
            "log_dir": self.log_dir,
            "summary_dir": self.summary_dir,
        }
        duplicates = _find_duplicate_path_names(path_values)
        if duplicates:
            joined = ", ".join("/".join(group) for group in duplicates)
            raise ValueError(f"paths directories must be distinct; duplicates found: {joined}")
        return self


class LoggingConfig(StrictModel):
    """Standard logging configuration."""

    level: str = "INFO"
    run_log_filename: str = "agent.log"

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Restrict logging levels to standard names."""

        normalized = value.strip().upper()
        valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if normalized not in valid_levels:
            raise ValueError(f"logging.level must be one of: {', '.join(sorted(valid_levels))}.")
        return normalized


class ExperimentConfig(StrictModel):
    """Episode and evaluation settings."""

    game_id: str = "zork1"
    seed: int = 7
    batch_size: int = 1
    max_steps: int = 8
    max_replay_attempts: int = 2
    frontier_refresh_cadence: int = 1
    local_exploration_cadence: int = 3
    branch_commit_steps: int = 4
    stub_episode_length: int = 3
    deterministic: bool = True

    @field_validator(
        "seed",
        "batch_size",
        "max_steps",
        "max_replay_attempts",
        "frontier_refresh_cadence",
        "local_exploration_cadence",
        "branch_commit_steps",
        "stub_episode_length",
    )
    @classmethod
    def validate_nonnegative_values(cls, value: int, info) -> int:
        """Validate core experiment budgets."""

        minimum = 0 if info.field_name in {"seed", "max_replay_attempts"} else 1
        if value < minimum:
            raise ValueError(f"experiment.{info.field_name} must be >= {minimum}.")
        return value

    @model_validator(mode="after")
    def validate_budget_relationships(self) -> "ExperimentConfig":
        """Check simple budget consistency rules."""

        if self.max_replay_attempts > self.max_steps:
            raise ValueError("experiment.max_replay_attempts cannot exceed experiment.max_steps.")
        if self.branch_commit_steps > self.max_steps:
            raise ValueError("experiment.branch_commit_steps cannot exceed experiment.max_steps.")
        return self


class ProjectConfig(StrictModel):
    """Complete project configuration tree."""

    project_name: str = "zork-agent"
    runtime: RuntimeConfig
    llm: LLMConfig
    prompts: PromptConfig
    policy: PolicyConfig
    paths: PathsConfig
    logging: LoggingConfig
    experiment: ExperimentConfig
    source_config: Path | None = None

    @property
    def prompt_dir(self) -> Path:
        """Return the absolute prompt directory."""

        return self.prompts.directory

    @property
    def run_log_path(self) -> Path:
        """Return the run log file path."""

        return self.paths.log_dir / self.logging.run_log_filename

    def ensure_output_directories(self) -> None:
        """Create the configured artifact directories if needed."""

        # TODO: split high-volume and low-volume artifacts once experiments diversify.
        for path in (
            self.paths.artifacts_dir,
            self.paths.trajectory_dir,
            self.paths.archive_dir,
            self.paths.log_dir,
            self.paths.summary_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def validate_prompt_files(self) -> None:
        """Validate that the configured prompt directory and prompt files exist."""

        if not self.prompt_dir.exists():
            raise FileNotFoundError(f"Prompt directory does not exist: {self.prompt_dir}")
        if not self.prompt_dir.is_dir():
            raise NotADirectoryError(f"Prompt directory is not a directory: {self.prompt_dir}")

        missing_paths = [
            self.prompt_dir / filename
            for filename in (
                self.prompts.system_file,
                self.prompts.action_proposal_file,
                self.prompts.trajectory_analysis_file,
                self.prompts.local_reflection_file,
                self.prompts.state_selection_file,
            )
            if not (self.prompt_dir / filename).exists()
        ]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Configured prompt files are missing: {missing}")


def load_config(path: str | Path) -> ProjectConfig:
    """Load, merge, validate, and resolve a YAML configuration file."""

    # TODO: support explicit profile overlays beyond a single extends chain.
    load_dotenv(override=False)
    config_path = Path(path).expanduser().resolve()
    raw = _load_yaml_with_extends(config_path)
    raw = _apply_env_overrides(raw)
    raw.pop("extends", None)
    config = ProjectConfig.model_validate(raw)
    resolved = _resolve_relative_paths(config, base_dir=_config_base_dir(config_path))
    resolved.source_config = config_path
    resolved.validate_prompt_files()
    return resolved


def _load_yaml_with_extends(path: Path) -> dict[str, Any]:
    """Load a YAML file and recursively merge any parent config."""

    current = _load_yaml_file(path)
    extends = current.get("extends")
    if not extends:
        return current

    parent_path = (path.parent / str(extends)).resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"Config extends target does not exist: {parent_path}")
    parent = _load_yaml_with_extends(parent_path)
    return _deep_merge(parent, current)


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML document from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            loaded = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML config at {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected mapping at config root: {path}")
    return loaded


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two configuration mappings."""

    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply a small set of explicit environment overrides."""

    updated = _deep_merge({}, raw)
    overrides = {
        "LMSTUDIO_BASE_URL": ("llm", "base_url"),
        "LMSTUDIO_API_KEY": ("llm", "api_key"),
        "LMSTUDIO_MODEL_NAME": ("llm", "model_name"),
        "ZORK1_GAME_PATH": ("runtime", "jericho_game_path"),
        "ZORK_AGENT_TRAJECTORY_DIR": ("paths", "trajectory_dir"),
        "ZORK_AGENT_ARCHIVE_DIR": ("paths", "archive_dir"),
        "ZORK_AGENT_LOG_DIR": ("paths", "log_dir"),
    }
    for env_key, path_keys in overrides.items():
        value = os.getenv(env_key)
        if value:
            _set_nested_value(updated, path_keys, value)
    return updated


def _set_nested_value(payload: dict[str, Any], path_keys: tuple[str, ...], value: Any) -> None:
    """Set a nested config key, creating intermediate mappings as needed."""

    cursor = payload
    for key in path_keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path_keys[-1]] = value


def _resolve_relative_paths(config: ProjectConfig, base_dir: Path) -> ProjectConfig:
    """Resolve path-valued config fields relative to the config file."""

    data = config.model_dump()

    def resolve(value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute():
            return str(candidate)
        return str((base_dir / candidate).resolve())

    data["runtime"]["jericho_game_path"] = resolve(data["runtime"]["jericho_game_path"])
    data["prompts"]["directory"] = resolve(data["prompts"]["directory"])
    for key in ("artifacts_dir", "trajectory_dir", "archive_dir", "log_dir", "summary_dir"):
        data["paths"][key] = resolve(data["paths"][key])

    return ProjectConfig.model_validate(data)


def _config_base_dir(config_path: Path) -> Path:
    """Choose the base directory used to resolve relative project paths."""

    # TODO: replace this layout heuristic with an explicit project_root field if needed.
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def _find_duplicate_path_names(path_values: dict[str, Path]) -> list[tuple[str, ...]]:
    """Return groups of config field names that resolve to the same directory."""

    reverse_index: dict[Path, list[str]] = {}
    for name, path in path_values.items():
        reverse_index.setdefault(path, []).append(name)
    return [tuple(names) for names in reverse_index.values() if len(names) > 1]
