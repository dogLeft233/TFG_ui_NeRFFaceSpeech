#!/bin/bash
# Run the existing run_eval_pipeline.sh inside the Docker image
# using the docker-compose service (GPU-enabled).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"
SERVICE_NAME="nerffacespeech"

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "❌ docker-compose.yml not found: $COMPOSE_FILE"
  exit 1
fi

echo "Using compose file: $COMPOSE_FILE"
echo "Service: $SERVICE_NAME"
echo "Passing args to run_eval_pipeline.sh: $*"
echo "---------------------------------------"
echo "Volumes (from compose) will map host data/output/models/cache to /app/..."
echo "Input is expected under host: $PROJECT_ROOT/data"
echo "Outputs will be written to host: $PROJECT_ROOT/output and $PROJECT_ROOT/outputs"
echo "---------------------------------------"

# Build if needed (skip if already built)
docker compose -f "$COMPOSE_FILE" build "$SERVICE_NAME"

# Run the pipeline inside the container, reusing the service config (GPU + volumes).
docker compose -f "$COMPOSE_FILE" run --rm \
  "$SERVICE_NAME" \
  bash -lc "bash /app/run_eval_pipeline.sh $*"

