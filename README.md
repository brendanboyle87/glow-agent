# GLoW for Jericho

This repository is a Jericho-first research implementation of a GLoW-style agent for parser interactive fiction, centered on Zork I. The default path is `experiment.runner_mode: glow_faithful`: a complete-trajectory frontier, an independent archive of typed states, archive-state selection by achieved plus potential value, global frontier analysis, shallow local branching, MAR-updated local world models, and Jericho-native snapshot restore with replay fallback.

The code is faithful to the paper's control-loop shape, but it is still an inspectable research system rather than a learned end-to-end reproduction. The main deviations are explicit heuristics, typed symbolic world-model artifacts, and deterministic fallbacks around every LLM-mediated stage.

## Current Status

- Implemented today:
  - `TrajectoryFrontier` over complete `EpisodeTrajectory` artifacts
  - independent persistent `ArchiveStore` of `ArchivedState` records
  - archive-state selection with achieved and inferred-potential scoring
  - global frontier analysis into typed `FrontierInsight` and `CriticalStateAnnotation` artifacts
  - local exploration from restored roots, plus MAR updates into persistent `LocalWorldModel` artifacts
  - evidence-ranked action selection over Jericho valid actions when available
  - Jericho `get_state()` / `set_state()` as first-class restore handles, with replay fallback
  - per-run manifests and prompt/completion artifacts for offline inspection
- Still heuristic and implementation-specific:
  - frontier ranking, archive scoring, branch commit gating, and action reranking are explicit hand-tuned rules
  - local and global "world models" are typed symbolic artifacts, not trained neural models
  - LLM stages are optional and every structured stage has a deterministic fallback

## GLoW Fidelity Snapshot

- Close to paper intent:
  - complete-trajectory frontier memory
  - separate persistent state archive
  - achieved plus potential value selection
  - global/local control loop with frontier analysis and MAR
  - native Jericho snapshots as the preferred restoration path
- Still not paper-equivalent:
  - no learned world model
  - no learned value function or learned selector
  - action policy is constrained by Jericho valid actions plus explicit evidence heuristics
  - potential value comes from typed critical-state inference, not a trained critic
  - the CLI always rewrites outputs under a per-run artifact root, so cross-run archive reuse is available in the runner but not exposed as a turnkey shared-archive CLI workflow

See [docs/glow_fidelity_ledger.md](docs/glow_fidelity_ledger.md) for the detailed paper-component ledger, and [docs/archive_memory_design_note.md](docs/archive_memory_design_note.md) for the archive/frontier split.

## Control Loop

In `glow_faithful` mode, the runner first loads the latest archive snapshot from the configured archive directory, if one exists. It then repeats this cycle:

1. Select an archived state using achieved value, inferred potential value, revisit penalties, and restore-method bonuses.
2. Restore that state, preferring Jericho native snapshots and falling back to replay when needed.
3. Run shallow local exploration from the restored root.
4. Optionally run MAR over the compared local branches and update the root's persistent local world model.
5. Convert local branches into complete trajectories and insert them into the complete-trajectory frontier.
6. Refresh archive-derived fields and optionally analyze the retained frontier trajectories into a global frontier insight.

The repository still contains a legacy heuristic state frontier (`FrontierQueue`) and a `legacy` runner mode for comparison/debugging, but the README and configs are centered on the GLoW path.

## Key Modules

- `src/zork_agent/experiment/episode_runner.py`: main orchestration for `legacy` and `glow_faithful` runner modes
- `src/zork_agent/memory/frontier.py`: complete-trajectory frontier plus the older heuristic state frontier
- `src/zork_agent/memory/archive_store.py` and `src/zork_agent/memory/archive_updater.py`: persistent archive state storage and derived-field refresh
- `src/zork_agent/policy/state_selector.py`: archive-state selection and legacy frontier selection
- `src/zork_agent/policy/local_explorer.py`: shallow branch rollouts from a restored root
- `src/zork_agent/policy/mar_reflector.py`: MAR inference and local-world-model updates
- `src/zork_agent/policy/frontier_analyzer.py`: global frontier analysis over retained trajectories
- `src/zork_agent/policy/action_generator.py`: Jericho-valid-action-first candidate generation and evidence reranking
- `src/zork_agent/env/jericho_env.py` and `src/zork_agent/env/replay.py`: Jericho wrapper plus native-snapshot-first restore flow

## Prerequisites

- Python 3.11+ on the host for local workflows
- `uv`
- a legally obtained Z-machine ROM such as `zork1.z5`
- Docker Desktop if you want the recommended live Jericho runtime on macOS
- LM Studio only for configs with `llm.offline_stub: false`

The host project supports Python 3.11+, but the Docker image intentionally uses Python 3.12 because that is the runtime this repo currently uses for live Jericho.

## Quick Start

### 1. Local Setup

```bash
cp .env.example .env
mkdir -p games
uv sync --extra dev
```

Put your ROM at `games/zork1.z5`, or override `ZORK1_GAME_PATH` in `.env`.

### 2. Offline Smoke Run

`configs/glow_smoke.yaml` uses the deterministic stub environment and does not require Jericho or LM Studio.

```bash
uv run zork-agent validate-config configs/glow_smoke.yaml
uv run zork-agent run-single configs/glow_smoke.yaml
uv run zork-agent run-batch configs/glow_smoke.yaml --episodes 2
```

Inspect the single-episode trajectory:

```bash
uv run zork-agent inspect-trajectory \
  artifacts/zork1-episode-000/trajectories/zork1-episode-000.jsonl
```

### 3. Live Jericho in Docker

`configs/glow_core.yaml` is the main live profile. On macOS, the intended setup is:

- LM Studio on the host
- Jericho inside Docker
- `LMSTUDIO_BASE_URL=http://host.docker.internal:1234/v1`

Build and run:

```bash
make build-docker
make run-zork-docker CONTAINER_CONFIG=configs/glow_core.yaml
```

Open a shell in the runtime:

```bash
make run-docker-shell
```

Run the Docker smoke test:

```bash
make smoke-test-docker CONTAINER_CONFIG=configs/glow_core.yaml
```

The Docker smoke test checks:

1. Python imports for `zork_agent` and `jericho`
2. config loading for `SMOKE_CONFIG` (default `configs/glow_core.yaml`)
3. reachability of `GET /v1/models` on the configured LM Studio base URL
4. Jericho environment initialization if the configured ROM exists

If the ROM is missing, the Jericho initialization step is reported as skipped instead of failing the smoke test.

## CLI and Make Targets

CLI commands:

```bash
uv run zork-agent validate-config <config>
uv run zork-agent run-single <config>
uv run zork-agent run-batch <config> --episodes <n>
uv run zork-agent inspect-trajectory <trajectory.jsonl>
```

Equivalent Make targets:

```bash
make sync
make test
make validate-config CONFIG=configs/glow_smoke.yaml
make run-single CONFIG=configs/glow_smoke.yaml
make run-batch CONFIG=configs/glow_smoke.yaml
make inspect-trajectory TRAJECTORY=artifacts/zork1-episode-000/trajectories/zork1-episode-000.jsonl
make build-docker
make run-docker-shell
make run-zork-docker CONTAINER_CONFIG=configs/glow_core.yaml
make smoke-test-docker CONTAINER_CONFIG=configs/glow_core.yaml
```

## Config Profiles

Current config families:

- Core:
  - `configs/base.yaml`
  - `configs/glow_smoke.yaml`
  - `configs/glow_core.yaml`
- Main feature ablations:
  - `configs/glow_ablation_no_global_analysis.yaml`
  - `configs/glow_ablation_no_mar.yaml`
  - `configs/glow_ablation_no_potential_value_selection.yaml`
- Minimal end-to-end live profiles:
  - `configs/glow_integration_smoke.yaml`
  - `configs/glow_integration_live.yaml`
  - `configs/glow_coverage_run.yaml`
- Longer research and diagnostic presets:
  - `configs/glow_stability_long.yaml`
  - `configs/glow_stability_no_mar.yaml`
  - `configs/glow_progress_run*.yaml`
  - `configs/glow_mar_reuse*.yaml`
  - `configs/glow_egg_bottleneck*.yaml`
  - `configs/glow_persistence_*.yaml`
- Seed wrappers:
  - `configs/*_seedNN.yaml` files are thin seed-specific overrides for repeated runs

Important CLI behavior: the CLI scopes every run under one run root. Whatever `paths.artifacts_dir` resolves to in the base config becomes the parent directory, and `trajectory_dir`, `archive_dir`, `log_dir`, `summary_dir`, and `metrics_dir` are all rewritten under `<artifacts_dir>/<run_id>/...`.

## Artifact Layout

Every CLI run writes a self-contained run directory. A typical run directory looks like:

```text
artifacts/zork1-episode-000/
  run_manifest.json
  source_config.yaml
  resolved_config.json
  archive/
    zork1-episode-000.archive.jsonl
  logs/
    agent.log
  metrics/
    episodes/
      zork1-episode-000.metrics.json
  summaries/
    zork1-episode-000.summary.json
    frontier_analysis/
      <analysis_id>/
        prompt.txt
        raw_completion.txt
        parsed_insight.json
        result.json
    state_selection/
      <selection_id>/
        prompt.txt
        raw_completion.txt
        candidate_scores.json
        selection_result.json
    local_world_models/
      <root_state_id>/
        model.json
        depth_progression_summary.json
        updates/
          <analysis_id>/
            prompt.txt
            raw_completion.txt
            mar_result.json
            updated_model.json
  trajectories/
    zork1-episode-000.jsonl
```

`run_manifest.json` is the quickest way to find the primary files for a completed CLI run.

Batch runs also add run-level summaries under `metrics/runs/`.

## Archive, Frontier, and Restore Semantics

### Trajectory frontier vs archive

- `TrajectoryFrontier` is the paper-facing global memory:
  - stores complete trajectories
  - retains top-k by `max_cumulative_reward_achieved`
  - feeds frontier analysis
- The archive is broader persistent state memory:
  - stores typed `ArchivedState` records
  - ingests from explored trajectories independently of frontier retention
  - is the source of archive-state selection candidates

Derived archive fields are refreshed from other subsystems:

- `achieved_value` and frontier support come from the currently retained trajectory frontier
- `projected_potential_value` comes from the latest frontier analysis

The runner can preload the latest archive snapshot from its configured archive directory. In the default CLI flow that directory is run-scoped, so archive reuse across separate CLI invocations only happens if the caller deliberately shares the same run-root archive path outside the default scoping behavior.

### Jericho restore behavior

The environment wrapper is intentionally Jericho-first:

- use `get_state()` / `set_state()` when the backend exposes them
- validate restored score/observation/inventory conservatively
- fall back to reset plus replay when native restore is unavailable or fails validation

Replay is a fallback, not the primary design target.

### Action generation

In live Jericho mode, action generation is primarily constrained by Jericho's valid-action list when available. The LLM provides ordering hints, but the final ranking is driven by explicit evidence features and local/global guidance artifacts.

## Testing

Run the full test suite:

```bash
uv run pytest
```

The test suite covers:

- config loading and run-scoped path rewriting
- archive persistence and archive refresh logic
- trajectory frontier behavior
- archive-state selection
- frontier analysis parsing and artifacts
- MAR and local-world-model persistence
- action generation and reranking
- Jericho-native restore and replay fallback behavior
- GLoW episode-runner and evaluator control flow with mocked env/LLM components
