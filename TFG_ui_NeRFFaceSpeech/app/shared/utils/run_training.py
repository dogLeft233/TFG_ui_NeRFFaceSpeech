"""
模型训练工具模块
支持 StyleNeRF 模型微调训练
从 fastapi_server/utils/ 迁移而来
"""
import subprocess
import os
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import (
    PROJECT_ROOT,
    NERF_CONDA_ENV,
    NERF_CONDA_PYTHON,
    NERF_CODE_DIR,
    MODEL_DIR,
)

# 训练任务状态存储（实际应用中应使用数据库或文件系统）
TRAINING_TASKS: Dict[str, Dict] = {}

# 训练输出目录
TRAINING_OUTPUT_DIR = NERF_CODE_DIR / "training_outputs"
TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def start_training(
    data_path: str,
    base_model: str = "ffhq_1024.pkl",
    outdir: Optional[str] = None,
    kimg: int = 50,
    snap: int = 5,
    imgsnap: int = 1,
    aug: str = "noaug",
    mirror: bool = False,
    model_config: str = "style_ffhq_ae_basic"
) -> Dict:
    """
    启动模型训练任务
    
    Args:
        data_path: 训练数据路径（单张图像数据集目录）
        base_model: 基础模型文件名（在 pretrained_networks 目录中）
        outdir: 输出目录（可选，默认自动生成）
        kimg: 训练轮数（千图像单位，建议50左右）
        snap: 快照保存间隔
        imgsnap: 图像快照间隔
        aug: 数据增强选项（noaug/crop/等）
        mirror: 是否镜像翻转
        model_config: 模型配置名称
    
    Returns:
        dict: 包含任务ID和状态的响应
    """
    # 验证数据路径
    if not os.path.exists(data_path):
        return {
            "success": False,
            "error": f"训练数据路径不存在: {data_path}"
        }
    
    # 验证基础模型
    base_model_path = MODEL_DIR / base_model
    if not base_model_path.exists():
        return {
            "success": False,
            "error": f"基础模型不存在: {base_model_path}"
        }
    
    # 生成任务ID和输出目录
    task_id = str(uuid.uuid4())
    if outdir is None:
        outdir = str(TRAINING_OUTPUT_DIR / task_id)
    os.makedirs(outdir, exist_ok=True)
    
    # 构建训练命令
    # 注意：这里假设训练脚本在 StyleNeRF 目录中
    # 实际路径可能需要根据项目结构调整
    training_script = NERF_CODE_DIR / "StyleNeRF" / "run_train.py"
    
    # 如果 run_train.py 不存在，尝试使用其他训练脚本
    if not training_script.exists():
        # 尝试查找其他可能的训练脚本
        possible_scripts = [
            NERF_CODE_DIR / "StyleNeRF" / "training" / "training_loop.py",
            PROJECT_ROOT / "scripts" / "train_model.py",
        ]
        training_script = None
        for script in possible_scripts:
            if script.exists():
                training_script = script
                break
        
        if training_script is None:
            return {
                "success": False,
                "error": "未找到训练脚本，请确保训练脚本存在于项目中"
            }
    
    # 准备环境变量
    env = os.environ.copy()
    env["PATH"] = f"{NERF_CONDA_ENV / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(NERF_CODE_DIR)
    
    # 构建命令参数
    # 使用新创建的 run_train.py 脚本
    cmd = [
        str(NERF_CONDA_PYTHON),
        str(training_script),
        "--outdir", outdir,
        "--data", data_path,
        "--model", model_config,
        "--resume", str(base_model_path),
        "--kimg", str(kimg),
        "--snap", str(snap),
        "--imgsnap", str(imgsnap),
        "--aug", aug,
    ]
    
    # 如果启用镜像翻转，添加 --mirror 标志
    if mirror:
        cmd.append("--mirror")
    
    # 初始化任务状态
    task_info = {
        "task_id": task_id,
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "data_path": data_path,
        "base_model": base_model,
        "outdir": outdir,
        "config": {
            "kimg": kimg,
            "snap": snap,
            "imgsnap": imgsnap,
            "aug": aug,
            "mirror": mirror,
            "model_config": model_config
        },
        "log": [],
        "progress": 0,
        "error": None
    }
    
    TRAINING_TASKS[task_id] = task_info
    
    # 在后台线程中运行训练
    def run_training_thread():
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',  # 使用replace模式处理非UTF-8字节，避免解码错误
                bufsize=1,
                env=env,
                cwd=str(NERF_CODE_DIR)
            )
            
            # 实时读取输出
            for line in process.stdout:
                if line:
                    task_info["log"].append(line.strip())
                    # 保持日志不超过1000行
                    if len(task_info["log"]) > 1000:
                        task_info["log"] = task_info["log"][-1000:]
            
            process.wait()
            
            if process.returncode == 0:
                task_info["status"] = "completed"
                task_info["progress"] = 100
            else:
                task_info["status"] = "failed"
                task_info["error"] = f"训练进程退出，返回码: {process.returncode}"
                
        except Exception as e:
            task_info["status"] = "failed"
            task_info["error"] = f"训练过程出错: {str(e)}"
        finally:
            task_info["end_time"] = datetime.now().isoformat()
    
    thread = threading.Thread(target=run_training_thread, daemon=True)
    thread.start()
    
    return {
        "success": True,
        "task_id": task_id,
        "message": "训练任务已启动"
    }


def get_training_status(task_id: str) -> Dict:
    """
    获取训练任务状态
    
    Args:
        task_id: 任务ID
    
    Returns:
        dict: 任务状态信息
    """
    if task_id not in TRAINING_TASKS:
        return {
            "success": False,
            "error": "任务不存在"
        }
    
    task_info = TRAINING_TASKS[task_id].copy()
    # 只返回最近的日志（最后100行）
    task_info["recent_log"] = task_info["log"][-100:] if len(task_info["log"]) > 100 else task_info["log"]
    
    return {
        "success": True,
        "data": task_info
    }


def list_training_tasks() -> Dict:
    """
    列出所有训练任务
    
    Returns:
        dict: 任务列表
    """
    tasks = []
    for task_id, task_info in TRAINING_TASKS.items():
        tasks.append({
            "task_id": task_id,
            "status": task_info["status"],
            "start_time": task_info["start_time"],
            "data_path": task_info["data_path"],
            "base_model": task_info["base_model"],
            "progress": task_info["progress"]
        })
    
    return {
        "success": True,
        "data": tasks
    }


def stop_training(task_id: str) -> Dict:
    """
    停止训练任务（需要实现进程管理）
    
    Args:
        task_id: 任务ID
    
    Returns:
        dict: 操作结果
    """
    if task_id not in TRAINING_TASKS:
        return {
            "success": False,
            "error": "任务不存在"
        }
    
    task_info = TRAINING_TASKS[task_id]
    if task_info["status"] not in ["running", "pending"]:
        return {
            "success": False,
            "error": f"任务状态为 {task_info['status']}，无法停止"
        }
    
    # TODO: 实现进程终止逻辑
    # 这里需要保存进程ID以便终止
    
    task_info["status"] = "stopped"
    task_info["end_time"] = datetime.now().isoformat()
    
    return {
        "success": True,
        "message": "训练任务已停止"
    }

