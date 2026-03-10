# Zork Agent Scaffold

This repository scaffolds a local research project for parser-based interactive fiction, starting with Jericho + Zork I and a local LLM served by LM Studio. The current codebase is intentionally incomplete: it wires configuration, prompts, logging, JSONL trajectories, Docker/runtime conventions, and a minimal CLI without claiming to reproduce the full GLoW paper.

The design keeps two concerns separate:

- Paper-inspired surfaces live in `src/zork_agent/policy/` and describe hierarchical exploration, reflection, and state selection at a high level.
- Engineering approximations live in `src/zork_agent/env/`, `src/zork_agent/llm/`, `src/zork_agent/memory/`, and `src/zork_agent/experiment/`, where the current implementation is deliberately simple and testable.

## What Is Scaffolded Today

- Typed config loading from YAML with `extends` support and environment overrides.
- Editable prompt templates under `prompts/`.
- A CLI with `run-single`, `run-batch`, `inspect-trajectory`, and `validate-config`.
- JSONL trajectory writing and replay inspection helpers.
- Standard-library logging with deterministic seed helpers.
- A Docker runtime aimed at Linux + Jericho.
- An LM Studio client that targets the OpenAI-compatible local API.
- A Jericho-first wrapper with typed reset/step outputs, inventory/score helpers, native `get_state()` / `set_state()` restore, and replay fallback.
- Frontier memory that retains native snapshots for the best replay targets and prunes the rest down to action-prefix fallback nodes.
- A working baseline episode loop that ties together reset, frontier updates, replay selection, shallow local branching, reflection, and JSON/JSONL artifacts.
- Placeholder policy components with explicit TODOs rather than paper claims.

## Architecture

```text
Host macOS
  LM Studio server
    -> OpenAI-compatible HTTP API (:1234)

Linux Docker container
  zork-agent CLI / scripts
    -> config loading + prompt files
    -> LMStudioClient
         -> host.docker.internal:1234/v1
    -> EpisodeRunner
         -> JerichoEnv
              -> Jericho FrotzEnv
         -> ActionGenerator
         -> StateSelector
         -> LocalExplorer
         -> ReflectionEngine
         -> FrontierQueue
         -> TrajectoryStore

Artifacts
  artifacts/trajectories/*.jsonl
  artifacts/summaries/*.json
  artifacts/logs/*.log
```

## What Is Not Implemented Yet

- Full hierarchical exploration or archive management from GLoW.
- Learned or retrieval-based memory.
- Robust Jericho state branching, action pruning, or benchmarking.
- Model-specific prompt tuning or automatic evaluation metrics.
- A production-grade Docker orchestration story beyond a single local research container.

## Project Layout

```text
.
├── README.md
├── Makefile
├── pyproject.toml
├── docker-compose.yml
├── configs/
├── docker/
├── prompts/
├── scripts/
├── src/
└── tests/
```

## Prerequisites

- macOS host with Docker Desktop and LM Studio installed.
- `uv` for local Python workflow.
- A legally obtained Z-machine ROM such as `zork1.z5`.

The code in this repository runs on Python 3.11+, but the Docker runtime uses Python 3.12 because the current Jericho project README now lists Linux and Python 3.12+ as requirements as of March 8, 2026. Jericho’s current docs also show `python -m pip install jericho` plus `python -m spacy download en_core_web_sm`, and the quickstart uses `FrotzEnv(".../zork1.z5")` with `reset()` and `step(...)`. Sources: [Jericho GitHub README](https://github.com/microsoft/jericho), [Jericho Quick-start](https://jericho-py.readthedocs.io/en/latest/tutorial_quick.html), [Jericho FrotzEnv docs](https://jericho-py.readthedocs.io/en/latest/frotz_env.html).

## 1. Install Local Tooling

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create local project state:

```bash
cp .env.example .env
mkdir -p games
```

Put your Zork I ROM at `games/zork1.z5`, or edit `configs/zork1_local.yaml` / `configs/zork1_debug.yaml` to point somewhere else.

Install the Python environment on the host:

```bash
uv sync --extra dev
```

## 2. Start LM Studio

LM Studio’s current docs say you can either:

- Open LM Studio, load a model, go to the Developer tab, and toggle `Start server`.
- Or start the server from the CLI with `lms server start`.

LM Studio’s OpenAI-compatible docs currently list:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

and show `base_url="http://localhost:1234/v1"` in Python examples. Sources: [LM Studio local server docs](https://lmstudio.ai/docs/developer/core/server), [LM Studio OpenAI compatibility docs](https://lmstudio.ai/docs/developer/openai-compat).

Recommended host setup:

1. Load a model in LM Studio.
2. Start the local server.
3. Confirm the model identifier in LM Studio.
4. Update `.env`:

```dotenv
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_MODEL_NAME=your-loaded-model
LMSTUDIO_TIMEOUT_SECONDS=60
LMSTUDIO_MAX_RETRIES=2
```

If the Python process is running inside Docker on macOS, use:

```dotenv
LMSTUDIO_BASE_URL=http://host.docker.internal:1234/v1
```

## 3. Build and Run the Jericho Docker Runtime

Build the image:

```bash
make build-docker
```

Open a shell inside the runtime:

```bash
make run-docker-shell
```

Run a single Zork episode inside the container:

```bash
make run-zork-docker CONTAINER_CONFIG=configs/zork1_local.yaml
```

Run the Docker smoke test:

```bash
make smoke-test-docker CONTAINER_CONFIG=configs/zork1_local.yaml
```

Notes:

- The Docker container is where Jericho is intended to live.
- LM Studio stays on the host.
- On Docker Desktop for Mac, `host.docker.internal` is the expected bridge name for the host LM Studio server.
- The compose service mounts the whole repo at `/workspace`, so `games/zork1.z5` on the host becomes `/workspace/games/zork1.z5` in the container.
- If you keep ROMs elsewhere on the host, either copy/symlink them into `games/` or override `ZORK1_GAME_PATH` in `.env`.
- ROM files are not bundled with this repository.
- The Jericho wrapper prefers `get_state()` / `set_state()` for restoration and uses reset-plus-replay only as fallback when native restore is unavailable or fails validation.

### Docker Smoke Test

The `smoke-test` entrypoint path checks four things, in order:

1. Python imports for `zork_agent` and `jericho`
2. config loading through `configs/zork1_local.yaml` or `DOCKER_SMOKE_CONFIG`
3. reachability of `LMSTUDIO_BASE_URL` via `GET /v1/models`
4. Jericho environment initialization if the configured game file exists

If the ROM file is missing, the final Jericho init step is reported as `skipped` instead of failing the smoke test.

## 4. Run From the Host

Host-side validation and placeholder runs work too:

```bash
uv run zork-agent validate-config configs/zork1_debug.yaml
uv run zork-agent run-single configs/zork1_debug.yaml
uv run zork-agent run-batch configs/zork1_debug.yaml --episodes 2
```

Inspect a saved trajectory:

```bash
uv run zork-agent inspect-trajectory artifacts/trajectories/zork1-episode-000.jsonl
```

## 5. Makefile Shortcuts

```bash
make sync
make test
make validate-config CONFIG=configs/zork1_debug.yaml
make run-single CONFIG=configs/zork1_debug.yaml
make run-batch CONFIG=configs/zork1_debug.yaml
make build-docker
make run-docker-shell
make run-zork-docker CONTAINER_CONFIG=configs/zork1_local.yaml
make smoke-test-docker CONTAINER_CONFIG=configs/zork1_local.yaml
```

The older aliases `make docker-build`, `make docker-shell`, and `make docker-run-single` still work.

## Configs

- `configs/base.yaml`: shared defaults.
- `configs/zork1_local.yaml`: intended for Docker + live LM Studio on the host.
- `configs/zork1_debug.yaml`: offline-friendly placeholder mode for smoke tests.

Config loading supports:

- YAML inheritance via `extends`
- environment overrides from `.env`
- deterministic seeds where possible
- explicit artifact/log/archive directories
- early validation of prompt-file existence and common URL/budget mistakes
- configurable frontier snapshot retention via `policy.snapshot_retention_limit`
- configurable selection/revisit budgets via:
  - `policy.state_selection_mode`
  - `experiment.max_steps`
  - `experiment.max_replay_attempts`
  - `experiment.frontier_refresh_cadence`
  - `experiment.local_exploration_cadence`

## Episode Loop

The baseline runner in `src/zork_agent/experiment/episode_runner.py` is intentionally simple:

1. Reset Jericho and add the initial state to the frontier.
2. Take normal exploratory actions from the current state.
3. Periodically refresh the frontier with replayable saved nodes.
4. On the configured cadence, choose a promising frontier node to revisit.
5. Restore that node with Jericho-native snapshots first, replay fallback second.
6. Run shallow local rollouts from the restored node and compare their outcomes.
7. Reflect over those branches to extract compact action guidance.
8. Use either the best local branch's first action or a fresh action-generation call to continue the main episode.
9. Persist the episode trajectory as JSONL and a summary sidecar as JSON.

The local branching and reflection stages are heuristic/LLM-assisted research layers on top of an explicit deterministic control loop. The frontier ranking remains heuristic and inspectable, while action generation, optional selector assistance, and reflection are the LLM-mediated parts.

## Paper Fidelity Vs Approximation

This codebase is research scaffolding, not a paper reproduction.

- Jericho native snapshots are first-class and preferred. That is a pragmatic engineering choice for fast revisit/branching, not a claim that the underlying paper used the same restore mechanics.
- Frontier ranking is an explicit heuristic formula over score, novelty, recent gain, and depth. It is deliberately easy to tune and inspect rather than paper-faithful.
- Local exploration is shallow multi-branch rollout comparison, not full tree search.
- Reflection is compact operational guidance extracted from rollouts or trajectories. It is meant to bias future prompts, not stand in for a full learned memory system.
- Replay remains as a fallback for robustness when native restore is unavailable or fails validation.

## Prompts

Prompt templates are plain text files under `prompts/` and are loaded at runtime. They are meant to be edited frequently during research.

The LM Studio client uses plain Python `str.format(...)`-style placeholders and exposes helper methods for:

- action candidate generation
- trajectory summarization
- rollout reflection
- revisit scoring

Each helper ultimately sends a standard `POST /v1/chat/completions` request through the same lower-level completion path.

The client accepts either a full OpenAI-compatible base URL such as `http://127.0.0.1:1234/v1` or a host-only form such as `http://127.0.0.1:1234`; the latter is normalized to `/v1`.

## Jericho Wrapper Notes

The environment wrapper in `src/zork_agent/env/jericho_env.py` exposes:

- `reset()`
- `step(action)`
- `get_native_state()`
- `set_native_state(state)`
- `validate_restored_state(...)`
- `get_valid_actions()`
- `get_score()`
- `get_inventory_text()`
- `get_world_state_snapshot()`
- `restore_snapshot()`
- `close()`

Important limitations:

- Jericho itself is Linux-oriented, so the live runtime assumption is isolated to the Docker image and the wrapper's optional `FrotzEnv` import path.
- `get_valid_actions()` and `get_inventory()` depend on Jericho internals and may be unavailable or expensive for some games; the wrapper treats both as optional metadata.
- Jericho-native snapshots are preferred because they are faster and less drift-prone than re-running an action prefix.
- Replay remains in `src/zork_agent/env/replay.py` as a fallback path. The restore flow is: try `set_state()`, validate score/observation/inventory conservatively, then fall back to `reset()` plus action replay if native restore fails or looks suspicious.
- Post-`set_state()` observation validation is still partly constrained by Jericho's helper surface: score and inventory can usually be re-queried directly, but observation text may come from the best available live value or the cached saved-node text when Jericho does not expose a clean current-observation getter.
- Frontier memory keeps native states only for the top configured revisit targets and prunes the rest to action-prefix-only saved nodes via `policy.snapshot_retention_limit`.

## Known Limitations

- The live Jericho path is Linux-oriented and expected to run in Docker. macOS host runs are mainly for debug/stub workflows.
- Native Jericho snapshots are held in memory for frontier revisit; they are not fully serialized into JSONL trajectories, so persisted trajectories reload as replay-first restore targets.
- Replay validation is conservative but still heuristic. Observation matching can fail on innocuous wording drift, or pass when semantically different states look textually similar.
- `get_valid_actions()` and `get_inventory()` depend on Jericho internals and may be slow, unavailable, or inconsistent across games.
- The Docker entrypoint reapplies Docker-only Jericho/spaCy dependencies after `uv sync` so bind-mounted workspace updates do not strip them out. This is pragmatic, but not especially elegant.
- The current Make/CLI workflow is aimed at local research iteration rather than long-running experiment orchestration or resumable jobs.

## Docker Networking Notes

On macOS with Docker Desktop, the container should call the host LM Studio server through `http://host.docker.internal:1234/v1`. That hostname resolves from inside the Linux container back to the macOS host, which lets the Jericho runtime stay containerized while LM Studio continues to run natively on the host GPU/Metal stack.

## LM Studio Smoke Test

Once LM Studio is serving a loaded model, you can test the client directly:

```bash
uv run python - <<'PY'
from pathlib import Path

from zork_agent.config import load_config
from zork_agent.llm.lmstudio_client import LMStudioClient
from zork_agent.llm.prompts import PromptManager

config = load_config(Path("configs/zork1_local.yaml"))
client = LMStudioClient(
    base_url=config.llm.base_url,
    default_model=config.llm.model_name,
    api_key=config.llm.api_key,
    timeout_seconds=config.llm.request_timeout_seconds,
)
prompt_manager = PromptManager(config.prompts)
response = client.generate_action_candidates(
    prompt_manager=prompt_manager,
    game_id=config.experiment.game_id,
    observation="You are standing west of a white house near a mailbox.",
    inventory="You are empty-handed.",
    score=0,
    moves=0,
    action_candidates=3,
    model=config.llm.model_name,
    temperature=config.llm.temperature,
    max_tokens=config.llm.max_tokens,
    timeout_seconds=config.llm.request_timeout_seconds,
)
print(response.model)
print(response.latency_seconds)
print(response.text)
print(response.usage)
PY
```

## Testing

Run the tests:

```bash
uv run pytest
```

The automated tests currently cover config loading, frontier bookkeeping, replay loading, JSONL serialization, prompt rendering, LM Studio response parsing, HTTP error handling, Jericho wrapper behavior via fake backends, and stub-mode episode/evaluator smoke tests.

Additional robustness checks now cover:

- sparse replay `step_index` handling instead of list-position assumptions
- trajectory integrity checks for mixed episode ids or unsorted steps
- malformed JSONL detection with line-number context
- frontier tie-breaking stability
- LM Studio base-URL normalization and clearer backend error messages
- malformed LLM output fallback paths in selection/action-generation tests
