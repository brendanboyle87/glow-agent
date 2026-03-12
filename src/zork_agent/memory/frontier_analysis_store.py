"""Artifact store for frontier-analysis runs.

This keeps the paper-faithful analysis stage inspectable by writing:
- the exact prompt input
- the raw LLM completion
- the parsed FrontierInsight
- a typed wrapper record with parse/fallback metadata

TODO: add index/latest-manifest support if multiple analyses per run become common.
"""

from __future__ import annotations

import json
from pathlib import Path

from zork_agent.types import FrontierAnalysisResult, FrontierInsight
from zork_agent.utils.serialization import to_jsonable


class FrontierAnalysisStore:
    """Persist frontier-analysis prompt, raw output, and parsed artifacts."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def analysis_dir(self, analysis_id: str) -> Path:
        """Return the artifact directory for one analysis id."""

        return self.root_dir / analysis_id

    def write(self, result: FrontierAnalysisResult) -> Path:
        """Persist one frontier-analysis result bundle to disk."""

        analysis_dir = self.analysis_dir(result.analysis_id)
        analysis_dir.mkdir(parents=True, exist_ok=True)

        (analysis_dir / "prompt.txt").write_text(result.prompt_input, encoding="utf-8")
        (analysis_dir / "raw_completion.txt").write_text(result.raw_completion, encoding="utf-8")
        (analysis_dir / "parsed_insight.json").write_text(
            json.dumps(to_jsonable(result.insight.to_record()), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        result.artifact_directory = str(analysis_dir)
        (analysis_dir / "result.json").write_text(
            json.dumps(to_jsonable(result.to_record()), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return analysis_dir

    def read_result(self, analysis_id: str) -> FrontierAnalysisResult:
        """Load a stored frontier-analysis result from disk."""

        analysis_dir = self.analysis_dir(analysis_id)
        payload = json.loads((analysis_dir / "result.json").read_text(encoding="utf-8"))
        return FrontierAnalysisResult.from_record(payload)

    def read_insight(self, analysis_id: str) -> FrontierInsight:
        """Load the parsed FrontierInsight artifact from disk."""

        analysis_dir = self.analysis_dir(analysis_id)
        payload = json.loads((analysis_dir / "parsed_insight.json").read_text(encoding="utf-8"))
        return FrontierInsight.from_record(payload)
