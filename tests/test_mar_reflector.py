"""Tests for MAR inference and persistent local-world-model updates.

TODO: add end-to-end evaluator coverage once the episode loop consumes archived roots directly.
"""

from __future__ import annotations

from pathlib import Path

from zork_agent.config import (
    ExperimentConfig,
    LLMConfig,
    LoggingConfig,
    PathsConfig,
    PolicyConfig,
    ProjectConfig,
    PromptConfig,
    RuntimeConfig,
)
from zork_agent.env.jericho_env import JerichoEnv
from zork_agent.llm.base import BaseLLMClient
from zork_agent.llm.prompts import PromptManager
from zork_agent.memory.local_world_model_store import LocalWorldModelStore
from zork_agent.policy.action_generator import ActionGenerator
from zork_agent.policy.local_explorer import LocalExplorer
from zork_agent.policy.mar_reflector import MultiPathAdvantageReflector
from zork_agent.types import (
    ActionBiasRecord,
    ActionProposal,
    AffordanceRecord,
    BranchTerminationReason,
    LLMChatRequest,
    LLMResponse,
    LocalBranchOutcome,
    LocalWorldModel,
)


class FakeLLMClient(BaseLLMClient):
    """Small fake LLM client for MAR parser tests."""

    def __init__(self, text: str):
        super().__init__(default_model="fake-model")
        self.text = text
        self.last_request: LLMChatRequest | None = None

    def list_models(self) -> list[str]:
        return ["fake-model"]

    def complete_chat(self, request: LLMChatRequest) -> LLMResponse:
        self.last_request = request
        return LLMResponse(
            text=self.text,
            model=request.model or "fake-model",
            latency_seconds=0.01,
            raw_payload={"messages": request.messages},
        )


class _FakeInventoryItem:
    """Small object that mimics Jericho inventory items."""

    def __init__(self, name: str):
        self.name = name


class BranchingBackend:
    """Fake backend with native snapshots and a small local branching structure."""

    def __init__(self, story_file: str, seed: int | None = None):
        self.story_file = story_file
        self.seed = seed
        self._snapshot_id = "west"
        self._state_table = {
            "west": {
                "observation": "West of House.",
                "inventory": [],
                "valid_actions": ["open mailbox", "look"],
                "score": 0,
                "moves": 0,
            },
            "mailbox": {
                "observation": "Opening the small mailbox reveals a leaflet.",
                "inventory": [_FakeInventoryItem("leaflet")],
                "valid_actions": ["read leaflet", "look"],
                "score": 1,
                "moves": 1,
            },
            "stale": {
                "observation": "West of House.",
                "inventory": [],
                "valid_actions": ["look"],
                "score": 0,
                "moves": 1,
            },
        }
        self._apply_snapshot("west")

    def reset(self) -> tuple[str, dict[str, int]]:
        self._apply_snapshot("west")
        return self.observation, {"score": self.score, "moves": self.moves}

    def step(self, action: str) -> tuple[str, float, bool, dict[str, int]]:
        normalized = action.strip().lower()
        if self._snapshot_id == "west" and normalized == "open mailbox":
            self._apply_snapshot("mailbox")
            return self.observation, 1.0, False, {"score": self.score, "moves": self.moves}
        if self._snapshot_id == "west" and normalized == "look":
            self._apply_snapshot("stale")
            return self.observation, 0.0, False, {"score": self.score, "moves": self.moves}
        if self._snapshot_id == "mailbox" and normalized == "read leaflet":
            self.score = 2
            self.moves += 1
            self.observation = "The leaflet welcomes you to Zork."
            self.valid_actions = ["look"]
            return self.observation, 1.0, True, {"score": self.score, "moves": self.moves}
        return self.observation, 0.0, False, {"score": self.score, "moves": self.moves}

    def get_score(self) -> int:
        return self.score

    def get_moves(self) -> int:
        return self.moves

    def get_inventory(self) -> list[_FakeInventoryItem]:
        return list(self.inventory)

    def get_valid_actions(self) -> list[str]:
        return list(self.valid_actions)

    def get_world_state_hash(self) -> str:
        return f"hash-{self._snapshot_id}"

    def get_state(self) -> tuple[object, ...]:
        return ("native", self._snapshot_id)

    def set_state(self, state: tuple[object, ...]) -> None:
        self._apply_snapshot(str(state[1]))

    def close(self) -> None:
        return None

    def _apply_snapshot(self, snapshot_id: str) -> None:
        payload = self._state_table[snapshot_id]
        self._snapshot_id = snapshot_id
        self.observation = str(payload["observation"])
        self.inventory = list(payload["inventory"])
        self.valid_actions = list(payload["valid_actions"])
        self.score = int(payload["score"])
        self.moves = int(payload["moves"])


class RecordingActionGenerator(ActionGenerator):
    """Action generator that records MAR guidance passed in from the local world model."""

    def __init__(self, config: ProjectConfig, prompt_manager: PromptManager):
        super().__init__(config, prompt_manager, llm_client=None)
        self.recorded_try_actions: list[list[str]] = []
        self.recorded_avoid_actions: list[list[str]] = []
        self.recorded_objects: list[list[str]] = []

    def propose_actions(self, state, **kwargs):  # type: ignore[override]
        self.recorded_try_actions.append(list(kwargs.get("supported_try_actions") or []))
        self.recorded_avoid_actions.append(list(kwargs.get("supported_avoid_actions") or []))
        self.recorded_objects.append(list(kwargs.get("supported_reflection_objects") or []))
        if "open mailbox" in state.valid_actions:
            return [ActionProposal(action="open mailbox"), ActionProposal(action="look")]
        return [ActionProposal(action="look")]


def _build_config(tmp_path: Path) -> ProjectConfig:
    """Create a minimal config for MAR tests."""

    rom_path = tmp_path / "games" / "zork1.z5"
    rom_path.parent.mkdir(parents=True, exist_ok=True)
    rom_path.write_text("fake rom", encoding="utf-8")
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "system.txt").write_text("System for {game_id}", encoding="utf-8")
    (prompt_dir / "action.txt").write_text("Action prompt", encoding="utf-8")
    (prompt_dir / "trajectory.txt").write_text("Trajectory prompt", encoding="utf-8")
    (prompt_dir / "frontier.txt").write_text("Frontier analysis prompt", encoding="utf-8")
    (prompt_dir / "mar_advantage.txt").write_text(
        "Root-local rollout comparison:\n{branch_comparison_block}\nExisting local world model:\n{existing_local_model_block}\nGrounding constraints:\n{grounding_constraints}",
        encoding="utf-8",
    )
    (prompt_dir / "reflection.txt").write_text("Reflection prompt", encoding="utf-8")
    (prompt_dir / "selection.txt").write_text("Frontier: {frontier_snapshot}", encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    return ProjectConfig(
        runtime=RuntimeConfig(jericho_game_path=rom_path, use_live_jericho=True),
        llm=LLMConfig(model_name="fake-model"),
        prompts=PromptConfig(
            directory=prompt_dir,
            system_file="system.txt",
            action_proposal_file="action.txt",
            trajectory_analysis_file="trajectory.txt",
            frontier_analysis_file="frontier.txt",
            local_reflection_file="reflection.txt",
            state_selection_file="selection.txt",
        ),
        policy=PolicyConfig(rollout_count=2, rollout_depth=2, action_candidates=3),
        paths=PathsConfig(
            artifacts_dir=artifacts_dir,
            trajectory_dir=artifacts_dir / "trajectories",
            archive_dir=artifacts_dir / "archive",
            log_dir=artifacts_dir / "logs",
            summary_dir=artifacts_dir / "summaries",
        ),
        logging=LoggingConfig(),
        experiment=ExperimentConfig(game_id="zork1"),
    )


def _mock_branches() -> list[LocalBranchOutcome]:
    """Create two mock rollouts from the same root."""

    return [
        LocalBranchOutcome(
            branch_index=0,
            actions_taken=["open mailbox", "read leaflet"],
            total_reward=2.0,
            score_change=2,
            final_score=2,
            final_observation="The leaflet welcomes you to Zork.",
            terminated=True,
            termination_reason=BranchTerminationReason.TERMINATED,
            persistent_inventory_gain_count=1,
            persistent_affordance_gain=1,
            persistent_exit_gain_count=0,
            branch_progress_score=4.0,
            metadata={
                "action_events": [
                    {
                        "action": "open mailbox",
                        "score_delta": 0,
                        "inventory_gained": True,
                        "persistent_affordance_gain": 1,
                        "persistent_exit_gain_count": 0,
                    },
                    {
                        "action": "read leaflet",
                        "score_delta": 2,
                        "inventory_gained": False,
                        "persistent_affordance_gain": 0,
                        "persistent_exit_gain_count": 0,
                    },
                ]
            },
        ),
        LocalBranchOutcome(
            branch_index=1,
            actions_taken=["look", "look"],
            total_reward=0.0,
            score_change=0,
            final_score=0,
            final_observation="West of House.",
            terminated=False,
            termination_reason=BranchTerminationReason.HORIZON_REACHED,
            appears_stuck=True,
            branch_progress_score=0.0,
            metadata={
                "action_events": [
                    {
                        "action": "look",
                        "score_delta": 0,
                        "inventory_gained": False,
                        "persistent_affordance_gain": 0,
                        "persistent_exit_gain_count": 0,
                        "movement_penalty": 0.5,
                    }
                ]
            },
        ),
    ]


def test_mar_reflector_generates_and_accumulates_advantage_hints(tmp_path: Path) -> None:
    """Heuristic MAR should emit a typed advantage hint and merge it into a local world model."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    reflector = MultiPathAdvantageReflector(config, prompt_manager, llm_client=None)
    store = LocalWorldModelStore(config.paths.summary_dir / "local_world_models")

    result = reflector.infer_advantage_hint(
        root_state_id="root:mailbox",
        branches=_mock_branches(),
        existing_local_world_model=None,
        analysis_id="mar-test-001",
    )

    assert result.used_fallback is True
    assert result.advantage_hint is not None
    assert "open mailbox" in result.advantage_hint.action_preferences
    assert "look" in result.advantage_hint.action_avoidances

    model = reflector.update_local_world_model(
        root_state_id="root:mailbox",
        existing_local_world_model=None,
        inference_result=result,
    )
    store.write_mar_result(result, model)
    reloaded = store.read("root:mailbox")

    assert reloaded is not None
    assert len(reloaded.accumulated_advantage_hints) == 1
    assert any(item.action == "open mailbox" for item in reloaded.action_priors)
    assert any(item.object_text == "mailbox" for item in reloaded.inferred_affordances)


def test_mar_reflector_parses_llm_output_into_typed_hint(tmp_path: Path) -> None:
    """LLM MAR output should parse into typed advantage points and action guidance."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient(
        """
        {
          "key_points": [
            {
              "action": "open mailbox",
              "advantage": 1.2,
              "outcome": "revealed leaflet",
              "rationale": "opened access to a new object",
              "support": ["branch-0"],
              "confidence": 0.8
            }
          ],
          "prefer": ["open mailbox", "read leaflet"],
          "avoid": ["look"],
          "subgoals": ["reveal the leaflet first"],
          "affordances": [{"object": "mailbox", "verb": "open"}],
          "reasoning": "opening the mailbox created the only useful affordance."
        }
        """.strip()
    )
    reflector = MultiPathAdvantageReflector(config, prompt_manager, llm_client)

    result = reflector.infer_advantage_hint(
        root_state_id="root:mailbox",
        branches=_mock_branches(),
        existing_local_world_model=None,
        analysis_id="mar-test-llm",
    )

    assert result.used_fallback is False
    assert result.advantage_hint is not None
    assert result.advantage_hint.action_preferences == ["open mailbox", "read leaflet"]
    assert result.advantage_hint.action_avoidances == ["look"]
    assert result.advantage_hint.key_state_action_points[0].action == "open mailbox"
    assert result.discovered_subgoals[0].description == "reveal the leaflet first"
    assert result.inferred_affordances[0].object_text == "mailbox"
    assert llm_client.last_request is not None
    assert llm_client.last_request.response_format is not None
    assert llm_client.last_request.response_format["type"] == "json_schema"


def test_mar_reflector_strips_thinking_preamble_and_code_fences(tmp_path: Path) -> None:
    """Parser hardening should recover MAR schema after a reasoning preamble."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    reflector = MultiPathAdvantageReflector(
        config,
        prompt_manager,
        FakeLLMClient(
            "Thinking Process:\n"
            "1. Compare branch outcomes.\n"
            "2. Identify the productive action.\n\n"
            "BEGIN_STRUCTURED_OUTPUT\n"
            "{\n"
            "  \"key_points\": [\n"
            "    {\n"
            "      \"action\": \"open mailbox\",\n"
            "      \"advantage\": 1.5,\n"
            "      \"outcome\": \"inventory gain\",\n"
            "      \"rationale\": \"revealed the leaflet\",\n"
            "      \"support\": [\"branch-0\"],\n"
            "      \"confidence\": 0.8\n"
            "    }\n"
            "  ],\n"
            "  \"prefer\": [\"open mailbox\", \"read leaflet\"],\n"
            "  \"avoid\": [\"look\"],\n"
            "  \"subgoals\": [\"reveal and inspect the leaflet\"],\n"
            "  \"affordances\": [{\"object\": \"leaflet\", \"verb\": \"read\"}],\n"
            "  \"reasoning\": \"open mailbox produced the only durable local gain.\"\n"
            "}\n"
            "END_STRUCTURED_OUTPUT"
        ),
    )

    result = reflector.infer_advantage_hint(
        root_state_id="root-west",
        branches=_mock_branches(),
        existing_local_world_model=None,
        analysis_id="mar-hardened",
    )

    assert result.used_fallback is False
    assert result.advantage_hint is not None
    assert result.advantage_hint.action_preferences[:2] == ["open mailbox", "read leaflet"]
    assert result.inferred_affordances[0].object_text == "leaflet"


def test_mar_reflector_debug_mode_does_not_force_json_schema(tmp_path: Path) -> None:
    """Debug mode should preserve free-form reasoning instead of enforcing backend schema."""

    config = _build_config(tmp_path)
    config.llm.analysis_debug_mode = True
    prompt_manager = PromptManager(config.prompts)
    llm_client = FakeLLMClient(
        "BEGIN_STRUCTURED_OUTPUT\n"
        "{\n"
        '  "key_points": [],\n'
        '  "prefer": ["open mailbox"],\n'
        '  "avoid": [],\n'
        '  "subgoals": [],\n'
        '  "affordances": [],\n'
        '  "reasoning": "debug mode."\n'
        "}\n"
        "END_STRUCTURED_OUTPUT"
    )
    reflector = MultiPathAdvantageReflector(config, prompt_manager, llm_client)

    result = reflector.infer_advantage_hint(
        root_state_id="root:mailbox",
        branches=_mock_branches(),
        existing_local_world_model=None,
        analysis_id="mar-debug-mode",
    )

    assert result.used_fallback is False
    assert llm_client.last_request is not None
    assert llm_client.last_request.response_format is None


def test_local_explorer_reuses_local_world_model_guidance(tmp_path: Path) -> None:
    """Repeated exploration from the same root should feed local-model priors back into action generation."""

    config = _build_config(tmp_path)
    prompt_manager = PromptManager(config.prompts)
    action_generator = RecordingActionGenerator(config, prompt_manager)
    store = LocalWorldModelStore(config.paths.summary_dir / "local_world_models")
    store.write(
        LocalWorldModel(
            root_state_id="hash-west",
            action_priors=[ActionBiasRecord(action="open mailbox", weight=1.0)],
            action_antipriors=[ActionBiasRecord(action="look", weight=1.0)],
            inferred_affordances=[AffordanceRecord(object_text="mailbox", affordance="open")],
        )
    )
    env = JerichoEnv(config, env_factory=BranchingBackend)
    base_state = env.reset(seed=7)
    explorer = LocalExplorer(
        action_generator,
        env=env,
        local_world_model_store=store,
    )

    result = explorer.explore_from_state(base_state, branch_count=1, branch_horizon=1)

    assert result.root_state_id == "hash-west"
    assert result.used_local_world_model is not None
    assert action_generator.recorded_try_actions[0] == ["open mailbox"]
    assert action_generator.recorded_avoid_actions[0] == ["look"]
    assert action_generator.recorded_objects[0] == ["mailbox"]
