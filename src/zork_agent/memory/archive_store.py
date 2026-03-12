"""Persistent archive storage for GLoW archived states.

The archive is stored independently from the trajectory frontier. The current
implementation uses JSONL snapshots because they are simple, typed, and easy to
inspect in research workflows.
"""

from __future__ import annotations

from pathlib import Path

from zork_agent.types import ArchivedState
from zork_agent.utils.serialization import read_jsonl, write_jsonl


class ArchiveStore:
    """Read and write archive-state snapshots independently of the frontier."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def path_for_episode(self, episode_id: str) -> Path:
        """Return the archive snapshot path for one episode run."""

        return self.root_dir / f"{episode_id}.archive.jsonl"

    def snapshot_paths(self) -> list[Path]:
        """Return all persisted archive snapshots in deterministic order."""

        if not self.root_dir.exists():
            return []
        return sorted(self.root_dir.glob("*.archive.jsonl"))

    def latest_snapshot_path(self) -> Path | None:
        """Return the most recently modified archive snapshot, if any."""

        snapshots = self.snapshot_paths()
        if not snapshots:
            return None
        return max(snapshots, key=lambda path: (path.stat().st_mtime_ns, path.name))

    def write_states(
        self,
        episode_id: str,
        states: list[ArchivedState],
        *,
        include_native_snapshot: bool = False,
    ) -> Path:
        """Persist a sorted archive snapshot for one episode run."""

        path = self.path_for_episode(episode_id)
        records = [
            state.to_record(include_native_snapshot=include_native_snapshot)
            for state in states
        ]
        write_jsonl(path, records)
        return path

    def read_states(self, path: Path) -> list[ArchivedState]:
        """Load archived states from a JSONL snapshot path."""

        return [ArchivedState.from_record(record) for record in read_jsonl(path)]

    def read_episode(self, episode_id: str) -> list[ArchivedState]:
        """Load archived states for one episode run."""

        return self.read_states(self.path_for_episode(episode_id))

    def read_latest_snapshot(self) -> tuple[Path, list[ArchivedState]] | None:
        """Load the newest persisted archive snapshot, if any."""

        latest_path = self.latest_snapshot_path()
        if latest_path is None:
            return None
        return latest_path, self.read_states(latest_path)
