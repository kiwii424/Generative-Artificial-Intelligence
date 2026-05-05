#!/usr/bin/env bash
# Launch the full-pipeline laptop demo (claude-v4: Snowflake embed + BGE reranker + HyDE).
# First run downloads ~4GB of models; subsequent runs take ~30s to warm up.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_DIR="$ROOT/loptop_demo"
PORT="${PORT:-8080}"

cd "$DEMO_DIR"

if [[ ! -d .venv ]]; then
  echo "[setup] Creating venv + installing deps (one-time, ~2 min)..."
  uv venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY not set."
  echo "  Put it in $DEMO_DIR/.env  or  export OPENROUTER_API_KEY=sk-or-..."
  exit 1
fi

echo "[run] Starting local demo on http://localhost:$PORT"
echo "[run] Browser opens automatically once models are loaded."
(
  # Wait for uvicorn to be ready, then open browser
  until curl -sf "http://localhost:$PORT/health" | grep -q '"ready": true'; do
    sleep 2
  done
  open "http://localhost:$PORT/" 2>/dev/null || true
) &

exec uvicorn demo_server:app --host 0.0.0.0 --port "$PORT"
