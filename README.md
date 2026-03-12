# GLoW for Jericho

This repository is a research implementation of GLoW-style exploration for Jericho text games, starting with Zork I and a local LM Studio model. The codebase is organized around the paper-facing concepts that now drive the implementation:

- complete-trajectory frontier memory
- independent persistent archive-state memory with achieved plus potential value
- global frontier analysis
- local world models updated through Multi-path Advantage Reflection (MAR)
- a GLoW-style global/local control loop

The implementation remains inspectable and practical:

- Jericho native snapshots are preferred for restore.
- Replay remains as a fallback for robustness.
- LLM outputs are cached and written as artifacts for offline inspection.
- Deterministic fallbacks remain available when parsing fails.

## Architecture

```text
Host macOS
  LM Studio server
    -> OpenAI-compatible HTTP API (:1234)

Linux Docker container
  zork-agent CLI
    -> EpisodeRunner (GLoW control loop)
         -> TrajectoryFrontier
         -> ArchiveStore / ArchiveUpdater
         -> FrontierAnalyzer
         -> StateSelector
         -> JerichoEnv
         -> LocalExplorer
              -> MAR reflector
              -> LocalWorldModelStore
         -> TrajectoryStore

Artifacts
  artifacts/trajectories/*.jsonl
  artifacts/archive/*.archive.jsonl
  artifacts/summaries/frontier_analysis/**/*
  artifacts/summaries/state_selection/**/*
  artifacts/summaries/local_world_models/**/*
  artifacts/metrics/episodes/*.json
  artifacts/metrics/runs/*.json
  artifacts/logs/*.log
```

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

Put your Zork I ROM at `games/zork1.z5`, or override `ZORK1_GAME_PATH` in `.env`.

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

Run the main GLoW config inside the container:

```bash
make run-zork-docker CONTAINER_CONFIG=configs/glow_core.yaml
```

Run the Docker smoke test:

```bash
make smoke-test-docker CONTAINER_CONFIG=configs/glow_core.yaml
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
2. config loading through `configs/glow_core.yaml` or `DOCKER_SMOKE_CONFIG`
3. reachability of `LMSTUDIO_BASE_URL` via `GET /v1/models`
4. Jericho environment initialization if the configured game file exists

If the ROM file is missing, the final Jericho init step is reported as `skipped` instead of failing the smoke test.

## 4. Run From the Host

Validate configs and run the GLoW implementation directly on the host:

```bash
uv run zork-agent validate-config configs/glow_smoke.yaml
uv run zork-agent run-single configs/glow_smoke.yaml
uv run zork-agent run-batch configs/glow_smoke.yaml --episodes 2
```

Inspect a saved trajectory:

```bash
uv run zork-agent inspect-trajectory artifacts/trajectories/zork1-episode-000.jsonl
```

## 5. Makefile Shortcuts

```bash
make sync
make test
make validate-config CONFIG=configs/glow_smoke.yaml
make run-single CONFIG=configs/glow_smoke.yaml
make run-batch CONFIG=configs/glow_smoke.yaml
make build-docker
make run-docker-shell
make run-zork-docker CONTAINER_CONFIG=configs/glow_core.yaml
make smoke-test-docker CONTAINER_CONFIG=configs/glow_core.yaml
```

## Configs

- `configs/base.yaml`: shared defaults.
- `configs/glow_core.yaml`: main GLoW run with LM Studio + Jericho.
- `configs/glow_smoke.yaml`: fast offline smoke-test config.
- `configs/glow_ablation_no_global_analysis.yaml`: disables frontier analysis.
- `configs/glow_ablation_no_mar.yaml`: disables MAR / local-world-model updates.
- `configs/glow_ablation_no_potential_value_selection.yaml`: removes potential-value contribution in archive-state selection.

Config loading supports:

- YAML inheritance via `extends`
- environment overrides from `.env`
- deterministic seeds where possible
- explicit artifact/log/archive/metrics directories
- early validation of prompt-file existence and common URL/budget mistakes
- clean ablation switches via:
  - `experiment.enable_global_frontier_analysis`
  - `experiment.enable_mar`
  - `experiment.enable_potential_value_selection`

## Main Run

Main GLoW run:

```bash
docker compose run --rm jericho uv run zork-agent run-single configs/glow_core.yaml
```

Main ablations:

```bash
uv run zork-agent run-batch configs/glow_ablation_no_global_analysis.yaml --episodes 3
uv run zork-agent run-batch configs/glow_ablation_no_mar.yaml --episodes 3
uv run zork-agent run-batch configs/glow_ablation_no_potential_value_selection.yaml --episodes 3
```

## Evaluation Artifacts

Per-episode outputs:

- `artifacts/trajectories/*.jsonl`
- `artifacts/summaries/*.summary.json`
- `artifacts/metrics/episodes/*.metrics.json`

Structured analysis artifacts:

- `artifacts/archive/*.archive.jsonl`
- `artifacts/summaries/frontier_analysis/<analysis_id>/`
- `artifacts/summaries/state_selection/<selection_id>/`
- `artifacts/summaries/local_world_models/<root_state_id>/`

Run-level metrics:

- `artifacts/metrics/runs/*.json`

These metrics include environment interactions, max/final score, restore counts, frontier sizes over time, frontier-analysis counts, MAR updates, selection decisions, and branch counts by root state.

## Archive vs Frontier

The repo now treats the archive and frontier as separate memories:

- `TrajectoryFrontier` is a bounded top-k memory of complete trajectories ranked by achieved value.
- The archive is a broader persistent state memory written independently under `artifacts/archive/`.

Archive ingestion walks every explored trajectory, not just the retained frontier. This means a state can remain selectable even after the trajectory that first discovered it falls out of the frontier.

Primary archive fields:

- identity and provenance:
  - `state_id`
  - `first_seen_episode_id` / `first_seen_timestep`
  - `last_seen_episode_id` / `last_seen_timestep`
  - `provenance_trajectory_id` / `provenance_timestep`
  - `provenance_trajectory_ids`
- restore handles:
  - native snapshot reference if available
  - replay prefix / restore metadata fallback
- counters:
  - `visit_count`
  - `selection_count`
  - `restore_success_count`
  - `restore_failure_count`
- local descriptors:
  - observation / inventory / valid-action summaries
  - state-cluster id

Derived archive fields:

- `achieved_value`
  - refreshed from the currently retained frontier trajectories
- `projected_potential_value`
  - cached from the latest frontier analysis / critical-state annotations
- `frontier_support_count` and `supporting_frontier_trajectory_ids`
  - derived from the current frontier, not intrinsic state properties

### State Identity

State identity is explicit and intentionally simple:

- prefer Jericho/native `world_state_hash` when available
- otherwise fall back to a deterministic normalized signature over:
  - observation text
  - inventory text
  - sorted valid actions

The fallback identity is an approximation and is documented as such in code and the fidelity ledger. It is used only when the environment does not provide a stable engine hash.

## Fidelity Ledger

The implementation is intentionally explicit about what is paper-faithful and what remains an implementation choice.

See:

- [docs/glow_fidelity_ledger.md](/Users/brendanboyle/repos/glow-agent/docs/glow_fidelity_ledger.md)
- [docs/archive_memory_design_note.md](/Users/brendanboyle/repos/glow-agent/docs/archive_memory_design_note.md)

## Prompts

Prompt templates are plain text files under `prompts/` and are loaded at runtime. They are meant to be edited frequently during research.

The LM Studio client uses plain Python `str.format(...)`-style placeholders and exposes helper methods for:

- action candidate generation
- frontier analysis
- MAR advantage reflection
- archive-state selection

Each helper ultimately sends a standard `POST /v1/chat/completions` request through the same lower-level completion path.

The client accepts either a full OpenAI-compatible base URL such as `http://127.0.0.1:1234/v1` or a host-only form such as `http://127.0.0.1:1234`; the latter is normalized to `/v1`.

## Jericho Runtime Notes

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
- Native snapshot references are kept available for archive states and replay metadata whenever the live runtime can provide them.

## Known Limitations

- The live Jericho path is Linux-oriented and expected to run in Docker.
- Native snapshots are preferred but not always fully serializable; replay fallback remains part of the implementation.
- The global world model, archive selection, and MAR all depend on prompt-mediated structured inference; deterministic fallbacks remain necessary when parsing or model quality is weak.
- The repo is a faithful research implementation target, not a claim of byte-for-byte reproduction of unpublished details or unstated training procedures.

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

config = load_config(Path("configs/glow_core.yaml"))
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

The automated tests cover:

- config loading and ablation toggles
- trajectory frontier and independent archive refresh
- frontier analysis parsing and artifacts
- archive-state selection
- MAR and local world model persistence
- action reranking and prompt construction
- replay / native restore behavior
- GLoW episode-runner and evaluator control flow with mocked env + mocked LLMs
