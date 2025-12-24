#!/bin/bash
# 本地运行 eval_pipline 的启动脚本（不依赖 Docker，不修改系统全局路径）

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "==========================================="
echo "NeRFFaceSpeech 本地评估流程启动脚本"
echo "（使用本机 conda 的 syncnet 环境）"
echo "==========================================="
echo ""

# 基本配置
INPUT_DIR="$PROJECT_ROOT/data/geneface_datasets/data/raw/videos"
OUTPUT_DIR="$PROJECT_ROOT/output/eval_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="$PROJECT_ROOT/NeRFFaceSpeech_Code/pretrained_networks/ffhq_1024.pkl"
SEGMENT_SEC=8
MAX_SEGMENTS=8
CONDA_ENV_NAME="environment/syncnet"

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}❌ 错误: 输入目录不存在: $INPUT_DIR${NC}"
    echo ""
    echo "请确保以下目录存在："
    echo "  $INPUT_DIR"
    exit 1
fi

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ 错误: 模型文件不存在: $MODEL_PATH${NC}"
    echo ""
    echo "请确保模型文件存在，或修改脚本中的 MODEL_PATH 变量"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓${NC} 输出目录: $OUTPUT_DIR"
echo ""

echo "配置参数:"
echo "  输入目录: $INPUT_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  模型文件: $MODEL_PATH"
echo "  每段时长: ${SEGMENT_SEC}秒"
echo "  每视频段数: ${MAX_SEGMENTS}段（随机选择）"
echo "  对齐方式: FFHQFaceAlignment"
echo "  使用环境: $CONDA_ENV_NAME"
echo ""

# 激活本地 conda 环境（仅在当前脚本进程生效，不修改系统配置）
activate_conda() {
    # 优先使用已安装的 conda.sh
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "/opt/conda/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        # 通用方式：只在当前 shell 中注入，不改系统配置文件
        eval "$(conda shell.bash hook)"
    else
        echo -e "${RED}❌ 未找到 conda 命令，请确认已安装 Anaconda/Miniconda 并在 PATH 中${NC}"
        exit 1
    fi

    # 激活指定环境
    if ! conda activate "$CONDA_ENV_NAME" 2>/dev/null; then
        echo -e "${RED}❌ 无法激活 conda 环境: $CONDA_ENV_NAME${NC}"
        echo "请先创建该环境，或修改脚本中的 CONDA_ENV_NAME。"
        exit 1
    fi
}

echo "激活 conda 环境: $CONDA_ENV_NAME ..."
activate_conda
echo -e "${GREEN}✓${NC} 已激活 conda 环境: $CONDA_ENV_NAME"
echo ""

# 进入项目根目录并运行评估流程
cd "$PROJECT_ROOT"

echo "==========================================="
echo "开始运行本地评估流程（不使用 Docker）"
echo "==========================================="
echo ""

set -x
python -m eval_pipline \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --network "$MODEL_PATH" \
    --segment-sec "$SEGMENT_SEC" \
    --max-segments "$MAX_SEGMENTS" \
    --random-segments \
    --ffhq-alignment \
    --device cuda
set +x

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo -e "${GREEN}✅ 本地评估流程执行成功！${NC}"
    echo "==========================================="
    echo ""
    echo "结果保存在: $OUTPUT_DIR"
    echo "  - 切分视频: $OUTPUT_DIR/videos_split/"
    echo "  - 裁剪视频: $OUTPUT_DIR/videos_cropped/"
    echo "  - 推理结果: $OUTPUT_DIR/videos_infer/"
    echo "  - 指标结果: $OUTPUT_DIR/metrics.json"
    echo ""
    echo "查看结果:"
    echo "  ls -lh \"$OUTPUT_DIR\"/"
    echo ""
else
    echo ""
    echo "==========================================="
    echo -e "${RED}❌ 本地评估流程执行失败！${NC}"
    echo "==========================================="
    echo ""
    echo "退出码: $EXIT_CODE"
    echo "请检查上面的 Python 错误信息。"
    echo ""
    exit $EXIT_CODE
fi
