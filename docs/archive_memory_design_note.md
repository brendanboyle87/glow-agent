# Archive Memory Design Note

This note explains the archive implementation used by the current GLoW-oriented
repo and how it differs from the bounded trajectory frontier.

## Responsibilities

The implementation now treats the two memories as separate subsystems:

- `TrajectoryFrontier`
  - bounded top-k memory of complete trajectories
  - ranked by trajectory value (`max_cumulative_reward_achieved`)
  - used for global frontier analysis and achieved-value refresh
- archive
  - broader persistent state memory
  - ingested from all explored trajectories
  - used as the source of archive-state selection candidates

The archive is not rebuilt from the frontier.

## Modules

- `/Users/brendanboyle/repos/glow-agent/src/zork_agent/types.py`
  - `ArchivedState`
  - `ReplayMetadata`
- `/Users/brendanboyle/repos/glow-agent/src/zork_agent/memory/archive_store.py`
  - independent JSONL persistence for archive snapshots
- `/Users/brendanboyle/repos/glow-agent/src/zork_agent/memory/archive_updater.py`
  - state identity
  - per-trajectory ingestion
  - frontier-derived achieved-value refresh
  - frontier-analysis-derived potential-value refresh

## State Identity Strategy

The state identity strategy is explicit and deterministic:

1. Prefer `world_state_hash` when the environment provides one and it is not
   `"unknown"`.
2. Otherwise fall back to a normalized signature over:
   - observation text
   - inventory text
   - sorted valid actions

The fallback is intentionally simple and documented as an approximation. It is
good enough to keep the archive symbolic and inspectable without pretending to
perfectly canonicalize Jericho states when the environment does not expose a
stable engine hash.

## Primary vs Derived Fields

Primary archive fields are stored because they are intrinsic to the archived
state:

- state identity and provenance
- replay/native restore handles
- visit/selection/restore counters
- score-at-state
- observation / inventory / valid-action summaries
- state-cluster id

Derived fields are refreshed from other subsystems and should be read that way:

- `achieved_value`
  - refreshed from the currently retained frontier trajectories
- `projected_potential_value`
  - cached from the latest frontier analysis / critical-state annotations
- `frontier_support_count`
  - number of retained frontier trajectories currently supporting the state
- `supporting_frontier_trajectory_ids`
  - the retained frontier trajectories currently supporting the state

## Why This Improves Fidelity

The earlier behavior tied state memory to retained frontier trajectories. That
meant:

- states disappeared from selectable memory once their trajectory fell out of the
  frontier
- archive statistics implicitly tracked frontier retention rather than the full
  explored state space

The new archive fixes that deviation by:

- ingesting every explored state from every completed trajectory
- persisting that state memory independently from the frontier
- using the frontier only to refresh achieved-value support as a derived signal

This matches the intended conceptual split more closely:

- frontier = bounded high-value trajectory memory
- archive = broader long-lived state memory

## Remaining Approximation

Archived states keep exact replay/native restore handles, but the runner may not
always keep the full original provenance prefix text once a source trajectory has
fallen out of the frontier. Restore remains exact through replay/native snapshot
metadata; the limitation only affects how much original temporal text context can
be reconstructed for later frontier-facing branch trajectories.
