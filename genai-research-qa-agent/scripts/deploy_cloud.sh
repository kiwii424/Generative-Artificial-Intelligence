#!/usr/bin/env bash
# Build + push + deploy the lightweight cloud demo (app/) to Cloud Run.
# Uses docker buildx to cross-compile for linux/amd64 from Apple Silicon.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROJECT="${GCP_PROJECT:-genai-research-qa-agent}"
REGION="${GCP_REGION:-asia-east1}"
REPO="${GCP_REPO:-genai-repo}"
SERVICE="${CLOUD_RUN_SERVICE:-genai-research-qa-api}"
TAG="${IMAGE_TAG:-v$(date +%Y%m%d-%H%M%S)-amd64}"

IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:${TAG}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY not set."
  echo "  Put it in $ROOT/.env  or  export OPENROUTER_API_KEY=sk-or-..."
  exit 1
fi

LLM_MODEL="${OPENROUTER_MODEL:-meta-llama/llama-3.2-3b-instruct}"

echo "=================================================================="
echo " Deploying cloud demo"
echo "   Project : $PROJECT"
echo "   Region  : $REGION"
echo "   Service : $SERVICE"
echo "   Image   : $IMAGE"
echo "   LLM     : $LLM_MODEL"
echo "=================================================================="
read -r -p "Proceed? [y/N] " ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "aborted."; exit 1; }

echo "[1/3] Building + pushing image (linux/amd64)..."
docker buildx build \
  --platform linux/amd64 \
  -t "$IMAGE" \
  --push \
  .

echo "[2/3] Deploying to Cloud Run..."
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --timeout 300 \
  --set-env-vars "OPENROUTER_API_KEY=${OPENROUTER_API_KEY},OPENROUTER_MODEL=${LLM_MODEL}"

echo "[3/3] Health check..."
URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)')
echo "   URL: $URL"
sleep 3
curl -sS "$URL/health" | tee /tmp/deploy_health.json
echo ""
echo ""
echo "Done. Open: $URL/"
