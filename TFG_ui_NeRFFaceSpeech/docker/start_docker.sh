#!/bin/bash
# Run the project start.sh inside the Docker image using docker-compose
# Exposes backend (8000) and frontend (7860) ports to host.

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
echo "Host ports -> container: 8000->8000 (backend), 7860->7860 (frontend)"
echo "Volumes (from compose) will map host data/output/models/cache to /app/..."
echo "---------------------------------------"

# Build if needed (skip if already built)
docker compose -f "$COMPOSE_FILE" build "$SERVICE_NAME"

# Run start.sh inside the container, reusing service config (GPU + volumes)
# Publish frontend/backend ports to the host.
docker compose -f "$COMPOSE_FILE" run --rm \
  --publish 8000:8000 \
  --publish 7860:7860 \
  "$SERVICE_NAME" \
  bash -lc "cd /app && bash start.sh $*"


