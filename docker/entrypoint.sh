#!/usr/bin/env bash
set -euo pipefail

cd /workspace

sync_workspace() {
  if [[ -f pyproject.toml ]]; then
    uv sync --extra dev --quiet
  fi
}

ensure_jericho_runtime() {
  if uv run python - <<'PY'
import importlib.util
import sys

required = ("jericho", "spacy", "en_core_web_sm")
missing = [name for name in required if importlib.util.find_spec(name) is None]
sys.exit(0 if not missing else 1)
PY
  then
    return
  fi

  uv pip install --python /opt/venv/bin/python pip jericho spacy
  uv run python -m spacy download en_core_web_sm >/dev/null
}

run_smoke_test() {
  local smoke_config="${SMOKE_CONFIG:-configs/zork1_local.yaml}"

  sync_workspace
  ensure_jericho_runtime
  uv run python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import httpx

from zork_agent.config import load_config
from zork_agent.env.jericho_env import JerichoEnv

import jericho  # noqa: F401
import zork_agent  # noqa: F401

config_path = Path(os.environ.get("SMOKE_CONFIG", "configs/zork1_local.yaml"))
config = load_config(config_path)
print(f"[smoke] imports: ok")
print(f"[smoke] config: ok ({config_path})")

base_url = os.environ.get("LMSTUDIO_BASE_URL", config.llm.base_url).rstrip("/")
headers = {"Authorization": f"Bearer {os.environ.get('LMSTUDIO_API_KEY', config.llm.api_key)}"}
response = httpx.get(f"{base_url}/models", headers=headers, timeout=5.0)
response.raise_for_status()
print(f"[smoke] lmstudio: ok ({base_url}/models)")

game_path = config.runtime.jericho_game_path
if not game_path.exists():
    print(f"[smoke] jericho: skipped (game file not found at {game_path})")
else:
    env = JerichoEnv(config)
    try:
        state = env.reset(seed=config.experiment.seed)
        print(f"[smoke] jericho: ok (score={state.score}, hash={state.world_state_hash})")
    finally:
        env.close()
PY
}

case "${1:-}" in
  smoke-test)
    shift
    run_smoke_test "$@"
    exit 0
    ;;
esac

sync_workspace
ensure_jericho_runtime
exec "$@"
