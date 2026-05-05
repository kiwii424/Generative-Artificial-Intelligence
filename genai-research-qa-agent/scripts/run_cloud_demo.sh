#!/usr/bin/env bash
# Open the deployed Cloud Run demo (lightweight BM25 + OpenRouter LLM).
# Checks /health first so you know if it's cold-starting or broken before showing it.

set -euo pipefail

URL="${CLOUD_URL:-https://genai-research-qa-api-232454355491.asia-east1.run.app}"

echo "[check] pinging $URL/health ..."
STATUS=$(curl -sS -o /tmp/cloud_health.json -w "%{http_code}" --max-time 30 "$URL/health" || echo "000")

if [[ "$STATUS" != "200" ]]; then
  echo "ERROR: /health returned HTTP $STATUS"
  cat /tmp/cloud_health.json 2>/dev/null || true
  echo ""
  echo "Possible causes: service not deployed, cold start > 30s, or region unreachable."
  exit 1
fi

echo "[ok]  Cloud service is live."
cat /tmp/cloud_health.json
echo ""
echo ""
echo "Opening browser → $URL/"
open "$URL/" 2>/dev/null || echo "(open browser manually: $URL/)"
