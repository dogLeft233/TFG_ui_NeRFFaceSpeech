#!/bin/bash

# NeRFFaceSpeech 一键启动脚本
# 使用方法: ./start.sh 或 bash start.sh

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$SCRIPT_DIR"

# 环境配置
CONDA_ENV="api"
BACKEND_HOST="0.0.0.0"
BACKEND_PORT="8000"
FRONTEND_PORT="7860"

# PID 文件
BACKEND_PID_FILE="$APP_DIR/.backend.pid"
FRONTEND_PID_FILE="$APP_DIR/.frontend.pid"

# 清理函数
cleanup() {
    echo -e "\n${YELLOW}正在停止服务...${NC}"
    
    # 停止后端
    if [ -f "$BACKEND_PID_FILE" ]; then
        BACKEND_PID=$(cat "$BACKEND_PID_FILE")
        if ps -p "$BACKEND_PID" > /dev/null 2>&1; then
            echo -e "${BLUE}停止后端服务 (PID: $BACKEND_PID)...${NC}"
            kill "$BACKEND_PID" 2>/dev/null || true
        fi
        rm -f "$BACKEND_PID_FILE"
    fi
    
    # 停止前端
    if [ -f "$FRONTEND_PID_FILE" ]; then
        FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
        if ps -p "$FRONTEND_PID" > /dev/null 2>&1; then
            echo -e "${BLUE}停止前端服务 (PID: $FRONTEND_PID)...${NC}"
            kill "$FRONTEND_PID" 2>/dev/null || true
        fi
        rm -f "$FRONTEND_PID_FILE"
    fi
    
    # 清理可能残留的进程
    pkill -f "uvicorn backend.main:app" 2>/dev/null || true
    pkill -f "simple_web.py" 2>/dev/null || true
    
    echo -e "${GREEN}所有服务已停止${NC}"
    exit 0
}

# 注册清理函数
trap cleanup SIGINT SIGTERM

# 检测 Python 命令和环境
USE_CONDA_RUN=false
USE_CONDA_PATH=false
CONDA_ENV_PATH=""
PYTHON_CMD="python"
UVICORN_CMD="uvicorn"

# 查找 conda 环境路径（支持非标准位置）
find_conda_env_path() {
    local env_name=$1
    # 方法1: 通过 conda env list 查找
    if command -v conda &> /dev/null; then
        local env_path=$(conda env list 2>/dev/null | grep -E "(^${env_name}[[:space:]]|/${env_name}$)" | awk '{print $NF}' | head -1)
        if [ -n "$env_path" ] && [ -d "$env_path" ]; then
            echo "$env_path"
            return 0
        fi
    fi
    # 方法2: 检查项目目录下的 environment 文件夹
    local project_env_path="$PROJECT_ROOT/environment/$env_name"
    if [ -d "$project_env_path" ] && [ -f "$project_env_path/bin/python" ]; then
        echo "$project_env_path"
        return 0
    fi
    # 方法3: 检查标准位置
    local standard_paths=(
        "$HOME/miniconda3/envs/$env_name"
        "$HOME/anaconda3/envs/$env_name"
        "/opt/conda/envs/$env_name"
    )
    for path in "${standard_paths[@]}"; do
        if [ -d "$path" ] && [ -f "$path/bin/python" ]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

# 检查是否已经在 conda 环境中
if [ -n "$CONDA_DEFAULT_ENV" ] || [ -n "$CONDA_PREFIX" ]; then
    # 已经在 conda 环境中
    CURRENT_ENV="${CONDA_DEFAULT_ENV:-$(basename "$CONDA_PREFIX")}"
    CURRENT_ENV_PATH="$CONDA_PREFIX"
    echo -e "${BLUE}检测到 Conda 环境: $CURRENT_ENV${NC}"
    echo -e "${BLUE}环境路径: $CURRENT_ENV_PATH${NC}"
    
    # 检查是否是目标环境（通过名称或路径）
    if [ "$CURRENT_ENV" = "$CONDA_ENV" ] || [[ "$CURRENT_ENV_PATH" == *"/$CONDA_ENV" ]] || [[ "$CURRENT_ENV_PATH" == *"/$CONDA_ENV/" ]]; then
        echo -e "${GREEN}✓ 已在目标环境 $CONDA_ENV 中，直接使用当前环境${NC}"
        USE_CONDA_RUN=false
        PYTHON_CMD="$CURRENT_ENV_PATH/bin/python"
    else
        echo -e "${YELLOW}当前环境 ($CURRENT_ENV) 不是目标环境 ($CONDA_ENV)${NC}"
        # 尝试查找目标环境路径
        CONDA_ENV_PATH=$(find_conda_env_path "$CONDA_ENV")
        if [ -n "$CONDA_ENV_PATH" ]; then
            echo -e "${GREEN}✓ 找到目标环境路径: $CONDA_ENV_PATH${NC}"
            USE_CONDA_PATH=true
            PYTHON_CMD="$CONDA_ENV_PATH/bin/python"
        else
            USE_CONDA_RUN=true
        fi
    fi
else
    # 不在 conda 环境中，尝试查找环境
    echo -e "${BLUE}未检测到 Conda 环境，查找目标环境...${NC}"
    
    # 初始化 conda（如果需要）
    if ! command -v conda &> /dev/null; then
        if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
            source "/opt/conda/etc/profile.d/conda.sh"
        fi
    fi
    
    # 查找目标环境路径
    CONDA_ENV_PATH=$(find_conda_env_path "$CONDA_ENV")
    if [ -n "$CONDA_ENV_PATH" ]; then
        echo -e "${GREEN}✓ 找到目标环境路径: $CONDA_ENV_PATH${NC}"
        USE_CONDA_PATH=true
        PYTHON_CMD="$CONDA_ENV_PATH/bin/python"
    elif command -v conda &> /dev/null; then
        # 尝试使用 conda run
        if conda env list 2>/dev/null | grep -qE "(^${CONDA_ENV}[[:space:]]|/${CONDA_ENV}$)"; then
            USE_CONDA_RUN=true
        else
            echo -e "${YELLOW}警告: 未找到 Conda 环境 '$CONDA_ENV'${NC}"
            echo -e "${YELLOW}尝试直接使用当前 Python 环境${NC}"
        fi
    else
        echo -e "${YELLOW}警告: 未找到 conda 命令和环境${NC}"
        echo -e "${YELLOW}尝试直接使用当前 Python 环境${NC}"
    fi
fi

# 确定使用的命令前缀
if [ "$USE_CONDA_PATH" = true ] && [ -n "$CONDA_ENV_PATH" ]; then
    # 直接使用环境路径中的 Python
    if [ ! -f "$PYTHON_CMD" ]; then
        echo -e "${RED}错误: Python 可执行文件不存在: $PYTHON_CMD${NC}"
        exit 1
    fi
    CMD_PREFIX=""
    echo -e "${GREEN}✓ 将使用环境 Python: $PYTHON_CMD${NC}"
elif [ "$USE_CONDA_RUN" = true ]; then
    if command -v conda &> /dev/null; then
        # 如果找到了环境路径，使用 -p 参数
        if [ -n "$CONDA_ENV_PATH" ]; then
            CMD_PREFIX="conda run -p $CONDA_ENV_PATH"
            echo -e "${GREEN}✓ 将使用: conda run -p $CONDA_ENV_PATH${NC}"
        else
            CMD_PREFIX="conda run -n $CONDA_ENV"
            echo -e "${GREEN}✓ 将使用: conda run -n $CONDA_ENV${NC}"
        fi
    else
        echo -e "${RED}错误: 需要 conda 但未找到 conda 命令${NC}"
        echo "请先激活环境或确保环境路径正确"
        exit 1
    fi
else
    CMD_PREFIX=""
    echo -e "${GREEN}✓ 将使用当前环境的 Python${NC}"
fi

# 如果使用环境路径，设置 Python 命令
if [ "$USE_CONDA_PATH" = true ] && [ -n "$PYTHON_CMD" ]; then
    export PATH="$CONDA_ENV_PATH/bin:$PATH"
fi

# 进入 app 目录
cd "$APP_DIR"
echo -e "${BLUE}工作目录: $(pwd)${NC}"

# 检查必要文件是否存在
if [ ! -f "backend/main.py" ]; then
    echo -e "${RED}错误: 未找到 backend/main.py${NC}"
    exit 1
fi

if [ ! -f "simple_web.py" ]; then
    echo -e "${RED}错误: 未找到 simple_web.py${NC}"
    exit 1
fi

# 检查端口是否被占用
check_port() {
    local port=$1
    local service=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}警告: 端口 $port ($service) 已被占用${NC}"
        echo "请先停止占用该端口的服务，或修改脚本中的端口配置"
        read -p "是否继续？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

check_port $BACKEND_PORT "后端"
check_port $FRONTEND_PORT "前端"

# 启动后端服务
echo -e "\n${BLUE}启动后端服务...${NC}"
if [ "$USE_CONDA_PATH" = true ] && [ -n "$PYTHON_CMD" ]; then
    BACKEND_CMD="$PYTHON_CMD -m uvicorn backend.main:app --host $BACKEND_HOST --port $BACKEND_PORT"
    echo -e "${BLUE}命令: $BACKEND_CMD${NC}"
    nohup $PYTHON_CMD -m uvicorn backend.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" > "$APP_DIR/.backend.log" 2>&1 &
elif [ -n "$CMD_PREFIX" ]; then
    BACKEND_CMD="$CMD_PREFIX uvicorn backend.main:app --host $BACKEND_HOST --port $BACKEND_PORT"
    echo -e "${BLUE}命令: $BACKEND_CMD${NC}"
    nohup $CMD_PREFIX uvicorn backend.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" > "$APP_DIR/.backend.log" 2>&1 &
else
    BACKEND_CMD="uvicorn backend.main:app --host $BACKEND_HOST --port $BACKEND_PORT"
    echo -e "${BLUE}命令: $BACKEND_CMD${NC}"
    nohup uvicorn backend.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" > "$APP_DIR/.backend.log" 2>&1 &
fi
BACKEND_PID=$!
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"

# 等待后端启动
echo -e "${YELLOW}等待后端服务启动...${NC}"
sleep 3

# 检查后端是否成功启动
if ! ps -p "$BACKEND_PID" > /dev/null 2>&1; then
    echo -e "${RED}错误: 后端服务启动失败${NC}"
    echo "查看日志: cat $APP_DIR/.backend.log"
    rm -f "$BACKEND_PID_FILE"
    exit 1
fi

echo -e "${GREEN}✓ 后端服务已启动 (PID: $BACKEND_PID)${NC}"
echo -e "${GREEN}  后端地址: http://$BACKEND_HOST:$BACKEND_PORT${NC}"
echo -e "${GREEN}  API 文档: http://$BACKEND_HOST:$BACKEND_PORT/docs${NC}"

# 启动前端服务
echo -e "\n${BLUE}启动前端服务...${NC}"
if [ "$USE_CONDA_PATH" = true ] && [ -n "$PYTHON_CMD" ]; then
    FRONTEND_CMD="$PYTHON_CMD simple_web.py"
    echo -e "${BLUE}命令: $FRONTEND_CMD${NC}"
    echo -e "${GREEN}✓ 前端服务正在启动...${NC}"
    echo -e "${GREEN}  前端地址: http://localhost:$FRONTEND_PORT${NC}"
    $PYTHON_CMD simple_web.py &
elif [ -n "$CMD_PREFIX" ]; then
    FRONTEND_CMD="$CMD_PREFIX python simple_web.py"
    echo -e "${BLUE}命令: $FRONTEND_CMD${NC}"
    echo -e "${GREEN}✓ 前端服务正在启动...${NC}"
    echo -e "${GREEN}  前端地址: http://localhost:$FRONTEND_PORT${NC}"
    $CMD_PREFIX python simple_web.py &
else
    FRONTEND_CMD="python simple_web.py"
    echo -e "${BLUE}命令: $FRONTEND_CMD${NC}"
    echo -e "${GREEN}✓ 前端服务正在启动...${NC}"
    echo -e "${GREEN}  前端地址: http://localhost:$FRONTEND_PORT${NC}"
    python simple_web.py &
fi
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"

# 等待前端进程
wait $FRONTEND_PID

