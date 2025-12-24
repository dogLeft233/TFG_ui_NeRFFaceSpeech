#!/usr/bin/env python3
"""
启动 TTS 服务的便捷脚本
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import (
    LLM_CONDA_PYTHON,
    TTS_SERVICE_SCRIPT
)

def main():
    """启动 TTS 服务"""
    import subprocess
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    
    # 构建命令
    cmd = [
        str(LLM_CONDA_PYTHON),
        str(TTS_SERVICE_SCRIPT),
        "--host", "0.0.0.0",
        "--port", "8001",
        "--workers", "1"
    ]
    
    print("=" * 60)
    print("启动 TTS 服务...")
    print("=" * 60)
    print(f"服务地址: http://0.0.0.0:8001")
    print(f"使用环境: {LLM_CONDA_PYTHON}")
    print("=" * 60)
    
    # 启动服务
    subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))

if __name__ == "__main__":
    main()

