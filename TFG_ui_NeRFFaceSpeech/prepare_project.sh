#!/bin/bash

# 项目准备脚本：
# - 从 assets/ 恢复模型权重软链接（tools/manage_assets.py restore）
# - 预下载必要的模型到默认缓存目录（tools/download_models.py）
#
# 用法（在项目根目录）：
#   bash prepare_project.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_ENV_DIR="$PROJECT_ROOT/environment/llm_talk"
PYTHON_CMD="python"

echo "==============================="
echo "0) 激活 environment/llm_talk 环境"
echo "==============================="

if [ -x "$LLM_ENV_DIR/bin/python" ]; then
  PYTHON_CMD="$LLM_ENV_DIR/bin/python"
  echo "使用环境 Python: $PYTHON_CMD"
else
  echo "⚠ 未找到 $LLM_ENV_DIR/bin/python，将使用当前系统 python"
fi

cd "$PROJECT_ROOT"

echo "==============================="
echo "1) 恢复模型权重软链接 (assets/)"
echo "==============================="

if [ -f "$PROJECT_ROOT/tools/manage_assets.py" ]; then
  "$PYTHON_CMD" "$PROJECT_ROOT/tools/manage_assets.py" restore
else
  echo "⚠ tools/manage_assets.py 不存在，跳过 restore 步骤"
fi

echo
echo "==============================="
echo "2) 预下载必需模型 (Whisper / HF)"
echo "==============================="

if [ -f "$PROJECT_ROOT/tools/download_models.py" ]; then
  "$PYTHON_CMD" "$PROJECT_ROOT/tools/download_models.py"
else
  echo "⚠ tools/download_models.py 不存在，跳过模型下载步骤"
fi

echo
echo "✅ 项目准备完成（如有跳过步骤，请根据提示检查）"


