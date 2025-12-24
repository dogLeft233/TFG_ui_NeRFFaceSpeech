#!/bin/bash
# 使用 Docker 容器执行项目根目录下的 prepare_project.sh
# - 自动构建镜像（若尚未构建）
# - 复用 docker-compose 中的 GPU / 卷挂载配置
# - 在容器内激活 /app/environment/llm_talk 后运行 prepare_project.sh

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
echo "Running prepare_project.sh inside Docker container..."
echo "---------------------------------------"
echo "Volumes (from compose) will map host assets/models/cache 到 /app/..."
echo "---------------------------------------"

# 构建镜像（如已构建则跳过）
docker compose -f "$COMPOSE_FILE" build "$SERVICE_NAME"

# 在容器中运行项目准备脚本，执行完自动退出并删除容器
docker compose -f "$COMPOSE_FILE" run --rm \
  "$SERVICE_NAME" \
  bash -lc "set -e \
    && source /opt/conda/etc/profile.d/conda.sh \
    && conda activate /app/environment/llm_talk \
    && cd /app \
    && bash prepare_project.sh"


