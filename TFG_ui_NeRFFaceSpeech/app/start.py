#!/usr/bin/env python3
"""
NeRFFaceSpeech 一键启动脚本 (Python 版本)
使用方法: python start.py 或 ./start.py
"""

import os
import sys
import subprocess
import signal
import time
import atexit
from pathlib import Path

# 颜色输出
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

# 配置
CONDA_ENV = "api"
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = "8000"
FRONTEND_PORT = "7860"

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
APP_DIR = SCRIPT_DIR

# 进程列表
processes = []

def cleanup():
    """清理函数：停止所有启动的进程"""
    print(f"\n{Colors.YELLOW}正在停止服务...{Colors.NC}")
    
    for proc in processes:
        if proc.poll() is None:  # 进程仍在运行
            print(f"{Colors.BLUE}停止进程 (PID: {proc.pid})...{Colors.NC}")
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception as e:
                print(f"{Colors.RED}停止进程时出错: {e}{Colors.NC}")
    
    # 清理可能残留的进程
    try:
        subprocess.run(["pkill", "-f", "uvicorn backend.main:app"], 
                      capture_output=True, timeout=2)
        subprocess.run(["pkill", "-f", "simple_web.py"], 
                      capture_output=True, timeout=2)
    except:
        pass
    
    print(f"{Colors.GREEN}所有服务已停止{Colors.NC}")

def check_conda():
    """检查 conda 是否可用"""
    try:
        result = subprocess.run(["conda", "--version"], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def check_env_exists(env_name):
    """检查 conda 环境是否存在"""
    try:
        result = subprocess.run(["conda", "env", "list"], 
                              capture_output=True, text=True, timeout=10)
        return env_name in result.stdout
    except:
        return False

def check_port(port):
    """检查端口是否被占用"""
    try:
        result = subprocess.run(["lsof", "-Pi", f":{port}", "-sTCP:LISTEN", "-t"],
                              capture_output=True, timeout=2)
        return result.returncode == 0
    except:
        return False

def get_conda_python(env_name):
    """获取 conda 环境中 Python 的路径"""
    try:
        conda_base = subprocess.run(["conda", "info", "--base"],
                                   capture_output=True, text=True, timeout=5)
        if conda_base.returncode == 0:
            conda_base_path = conda_base.stdout.strip()
            python_path = Path(conda_base_path) / "envs" / env_name / "bin" / "python"
            if python_path.exists():
                return str(python_path)
    except:
        pass
    
    # 如果无法获取，尝试使用 conda run
    return None

def main():
    """主函数"""
    # 注册清理函数
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda s, f: cleanup())
    signal.signal(signal.SIGTERM, lambda s, f: cleanup())
    
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}NeRFFaceSpeech 一键启动脚本{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")
    
    # 检查 conda
    if not check_conda():
        print(f"{Colors.RED}错误: 未找到 conda 命令{Colors.NC}")
        print("请先安装 Anaconda 或 Miniconda")
        sys.exit(1)
    
    # 检查环境是否存在
    if not check_env_exists(CONDA_ENV):
        print(f"{Colors.RED}错误: Conda 环境 '{CONDA_ENV}' 不存在{Colors.NC}")
        print(f"请先创建环境: conda env create -f {PROJECT_ROOT}/environment/api.yaml")
        sys.exit(1)
    
    # 获取 conda 环境的 Python
    conda_python = get_conda_python(CONDA_ENV)
    if conda_python:
        print(f"{Colors.GREEN}✓ 使用 Python: {conda_python}{Colors.NC}")
        python_cmd = conda_python
    else:
        # 尝试使用 conda run
        print(f"{Colors.YELLOW}使用 conda run 方式启动{Colors.NC}")
        python_cmd = None  # 将使用 conda run
    
    # 检查必要文件
    if not (APP_DIR / "backend" / "main.py").exists():
        print(f"{Colors.RED}错误: 未找到 backend/main.py{Colors.NC}")
        sys.exit(1)
    
    if not (APP_DIR / "simple_web.py").exists():
        print(f"{Colors.RED}错误: 未找到 simple_web.py{Colors.NC}")
        sys.exit(1)
    
    # 检查端口
    if check_port(BACKEND_PORT):
        print(f"{Colors.YELLOW}警告: 端口 {BACKEND_PORT} (后端) 已被占用{Colors.NC}")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    if check_port(FRONTEND_PORT):
        print(f"{Colors.YELLOW}警告: 端口 {FRONTEND_PORT} (前端) 已被占用{Colors.NC}")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # 切换到 app 目录
    os.chdir(APP_DIR)
    print(f"{Colors.BLUE}工作目录: {os.getcwd()}{Colors.NC}\n")
    
    # 启动后端服务
    print(f"{Colors.BLUE}启动后端服务...{Colors.NC}")
    backend_cmd = [
        "uvicorn", "backend.main:app",
        "--host", BACKEND_HOST,
        "--port", BACKEND_PORT
    ]
    
    if python_cmd:
        # 使用指定的 Python
        backend_cmd = [python_cmd, "-m"] + backend_cmd[0:]
    else:
        # 使用 conda run
        backend_cmd = ["conda", "run", "-n", CONDA_ENV] + backend_cmd
    
    print(f"{Colors.BLUE}命令: {' '.join(backend_cmd)}{Colors.NC}")
    
    backend_log = APP_DIR / ".backend.log"
    with open(backend_log, "w") as f:
        backend_proc = subprocess.Popen(
            backend_cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=APP_DIR
        )
    
    processes.append(backend_proc)
    
    # 等待后端启动
    print(f"{Colors.YELLOW}等待后端服务启动...{Colors.NC}")
    time.sleep(3)
    
    # 检查后端是否成功启动
    if backend_proc.poll() is not None:
        print(f"{Colors.RED}错误: 后端服务启动失败{Colors.NC}")
        print(f"查看日志: cat {backend_log}")
        cleanup()
        sys.exit(1)
    
    print(f"{Colors.GREEN}✓ 后端服务已启动 (PID: {backend_proc.pid}){Colors.NC}")
    print(f"{Colors.GREEN}  后端地址: http://{BACKEND_HOST}:{BACKEND_PORT}{Colors.NC}")
    print(f"{Colors.GREEN}  API 文档: http://{BACKEND_HOST}:{BACKEND_PORT}/docs{Colors.NC}\n")
    
    # 启动前端服务
    print(f"{Colors.BLUE}启动前端服务...{Colors.NC}")
    frontend_cmd = ["python", "simple_web.py"]
    
    if python_cmd:
        frontend_cmd = [python_cmd, "simple_web.py"]
    else:
        frontend_cmd = ["conda", "run", "-n", CONDA_ENV, "python", "simple_web.py"]
    
    print(f"{Colors.BLUE}命令: {' '.join(frontend_cmd)}{Colors.NC}")
    print(f"{Colors.GREEN}✓ 前端服务正在启动...{Colors.NC}")
    print(f"{Colors.GREEN}  前端地址: http://localhost:{FRONTEND_PORT}{Colors.NC}\n")
    print(f"{Colors.YELLOW}按 Ctrl+C 停止所有服务{Colors.NC}\n")
    
    # 在前台启动前端
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=APP_DIR
    )
    processes.append(frontend_proc)
    
    # 等待前端进程结束
    try:
        frontend_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

if __name__ == "__main__":
    main()

