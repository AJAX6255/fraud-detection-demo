#!/usr/bin/env bash
# Deploy the fraud-detection engine to Google Cloud Run.
# Usage: ./deploy_cloudrun.sh YOUR_PROJECT_ID [REGION] [SERVICE_MODE (api|streamlit)]
# Prereqs: gcloud CLI authenticated; artifacts/ and data/processed/transactions.parquet built.
set -euo pipefail

PROJECT_ID="${1:?Usage: ./deploy_cloudrun.sh YOUR_PROJECT_ID [REGION] [SERVICE_MODE]}"
REGION="${2:-us-central1}"
SERVICE_MODE="${3:-api}"
SERVICE="fraud-engine"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

echo "==> Deploying $SERVICE (mode: $SERVICE_MODE) to $REGION (source build)…"
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --set-env-vars "SERVICE_MODE=${SERVICE_MODE},LLM_PROVIDER=${LLM_PROVIDER:-},GEMINI_API_KEY=${GEMINI_API_KEY:-},OPENAI_API_KEY=${OPENAI_API_KEY:-}"

echo "==> Done. Service URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
