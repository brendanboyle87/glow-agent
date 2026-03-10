"""JSON and JSONL serialization helpers.

TODO: switch to a richer event schema before using these logs for long-term analysis.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


JsonDict = dict[str, Any]


class SerializationError(ValueError):
    """Raised when JSONL data is malformed or not record-shaped."""


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and common Python containers into JSON-compatible values."""

    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_jsonl(path: Path, records: Iterable[JsonDict]) -> Path:
    """Write an iterable of dictionaries to a JSONL file."""

    # TODO: support append-safe temp files if write volumes become large.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            normalized = _normalize_record(record, context=f"{path}:{index}")
            handle.write(json.dumps(normalized, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    return path


def append_jsonl(path: Path, record: JsonDict) -> Path:
    """Append one record to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        normalized = _normalize_record(record, context=str(path))
        handle.write(json.dumps(normalized, ensure_ascii=True, sort_keys=True))
        handle.write("\n")
    return path


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read all JSONL records from disk."""

    if not path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    records: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SerializationError(
                        f"Invalid JSONL at {path}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise SerializationError(
                        f"Expected a JSON object at {path}:{line_number}, got {type(parsed).__name__}."
                    )
                records.append({str(key): value for key, value in parsed.items()})
    return records


def _normalize_record(record: Mapping[str, Any] | JsonDict, *, context: str) -> JsonDict:
    """Validate one record and convert it into a JSON-compatible dictionary."""

    if not isinstance(record, Mapping):
        raise SerializationError(f"Expected a mapping record for JSONL output at {context}.")
    normalized = to_jsonable(dict(record))
    if not isinstance(normalized, dict):
        raise SerializationError(f"Expected a mapping record for JSONL output at {context}.")
    return normalized
