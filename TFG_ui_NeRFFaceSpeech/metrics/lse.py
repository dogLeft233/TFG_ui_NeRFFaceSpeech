"""LSE (Lip Sync Error) 指标计算模块。

提供 LSE-C (Confidence) 和 LSE-D (Distance) 的计算功能，使用 SyncNet 模型。
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np


def compute_lse_from_video(
    video_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    batch_size: int = 20,
    vshift: int = 15,
    tmp_dir: Optional[Union[str, Path]] = None,
    syncnet_dir: Optional[Union[str, Path]] = None
) -> Tuple[float, float]:
    """从视频文件计算 LSE-C 和 LSE-D。
    
    LSE-C (Lip Sync Error - Confidence): 同步置信度，值越大越好
    LSE-D (Lip Sync Error - Distance): 同步距离，值越小越好
    
    Args:
        video_path: 视频文件路径（MP4）
        model_path: SyncNet 模型文件路径，如果为 None 则使用默认路径
        batch_size: 批处理大小
        vshift: 视频偏移范围
        tmp_dir: 临时目录，如果为 None 则使用系统临时目录
        syncnet_dir: SyncNet 代码目录，如果为 None 则使用默认路径
    
    Returns:
        (LSE-C, LSE-D) 元组
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 设置默认路径
    if syncnet_dir is None:
        # 默认使用 metrics/scores_LSE
        current_dir = Path(__file__).parent
        syncnet_dir = current_dir / "scores_LSE"
    
    syncnet_dir = Path(syncnet_dir)
    if not syncnet_dir.exists():
        raise FileNotFoundError(f"SyncNet 目录不存在: {syncnet_dir}")
    
    # 确定 syncnet_python 目录路径
    syncnet_python_dir = syncnet_dir / "syncnet_python"
    if not syncnet_python_dir.exists():
        # 如果 syncnet_dir 本身就是 syncnet_python 目录
        if (syncnet_dir / "SyncNetModel.py").exists():
            syncnet_python_dir = syncnet_dir
        else:
            raise FileNotFoundError(
                f"SyncNet Python 目录不存在: {syncnet_python_dir}\n"
                f"请检查 SyncNet 目录结构是否正确"
            )
    
    # 添加 syncnet_python 目录到 Python 路径（SyncNetModel 和 SyncNetInstance_calc_scores 都在这里）
    if str(syncnet_python_dir) not in sys.path:
        sys.path.insert(0, str(syncnet_python_dir))
    
    # 导入 SyncNet（需要在 syncnet conda 环境中）
    try:
        from SyncNetInstance_calc_scores import SyncNetInstance
    except ImportError as e:
        raise ImportError(
            f"无法导入 SyncNetInstance。请确保已激活 syncnet conda 环境。\n"
            f"错误: {e}\n"
            f"提示: 运行 'conda activate syncnet' 后再执行\n"
            f"当前 sys.path 中的相关路径: {[p for p in sys.path if 'syncnet' in p.lower()]}"
        )
    
    # 设置模型路径
    if model_path is None:
        # 尝试多个可能的模型路径
        possible_paths = [
            syncnet_dir / "syncnet_python" / "data" / "syncnet_v2.model",
            syncnet_dir / "data" / "syncnet_v2.model",
            syncnet_dir / "syncnet_v2.model",
        ]
        model_path = None
        for path in possible_paths:
            if Path(path).exists():
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(
                f"SyncNet 模型文件不存在。请检查以下路径之一：\n"
                f"  - {possible_paths[0]}\n"
                f"  - {possible_paths[1]}\n"
                f"  - {possible_paths[2]}\n"
                f"或运行 'sh {syncnet_dir}/syncnet_python/download_model.sh' 下载模型"
            )
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"SyncNet 模型文件不存在: {model_path}\n"
            f"请运行 'sh {syncnet_dir}/download_model.sh' 下载模型"
        )
    
    # 创建临时目录
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="lse_"))
    else:
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    reference = "lse_calc"
    tmp_subdir = tmp_dir / reference
    if tmp_subdir.exists():
        shutil.rmtree(tmp_subdir)
    tmp_subdir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 创建命名空间对象来传递参数
        class Opt:
            def __init__(self):
                self.initial_model = str(model_path)
                self.batch_size = batch_size
                self.vshift = vshift
                self.tmp_dir = str(tmp_dir)
                self.reference = reference
        
        opt = Opt()
        
        # 加载模型
        s = SyncNetInstance()
        s.loadParameters(str(model_path))
        
        # 计算 LSE
        offset, confidence, min_distance = s.evaluate(opt, videofile=str(video_path))
        
        # LSE-C 是 confidence，LSE-D 是 min_distance
        lse_c = float(confidence)
        lse_d = float(min_distance)
        
        return lse_c, lse_d
    
    finally:
        # 清理临时目录（可选，保留以便调试）
        # if tmp_dir.exists() and tmp_dir.name.startswith("lse_"):
        #     shutil.rmtree(tmp_dir)
        pass


def compute_lse_metric(
    video_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    batch_size: int = 20,
    vshift: int = 15,
    tmp_dir: Optional[Union[str, Path]] = None,
    syncnet_dir: Optional[Union[str, Path]] = None
) -> Tuple[float, float]:
    """计算视频的 LSE 指标（便捷函数）。
    
    Args:
        video_path: 视频文件路径（MP4）
        model_path: SyncNet 模型文件路径
        batch_size: 批处理大小
        vshift: 视频偏移范围
        tmp_dir: 临时目录
        syncnet_dir: SyncNet 代码目录
    
    Returns:
        (LSE-C, LSE-D) 元组
    """
    return compute_lse_from_video(
        video_path, model_path, batch_size, vshift, tmp_dir, syncnet_dir
    )


if __name__ == "__main__":
    """简单测试 / 示例。
    
    由于 LSE 依赖 SyncNet 模型和专用 conda 环境，这里提供一个命令行接口：
    
        python -m metrics.lse --video path/to/video.mp4 --syncnet_dir metrics/scores_LSE
    
    如果模型或环境未就绪，会给出清晰的错误提示，而不是直接崩溃。
    """
    import argparse

    parser = argparse.ArgumentParser(description="LSE (Lip Sync Error) 简单测试")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="待评估的视频路径（MP4）",
    )
    parser.add_argument(
        "--syncnet_dir",
        type=str,
        default=None,
        help="SyncNet 代码目录，默认使用 metrics/scores_LSE",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="SyncNet 模型文件路径（可选）",
    )
    args = parser.parse_args()

    print("=== LSE 简单测试 ===")
    print(f"视频: {args.video}")
    print(f"SyncNet 目录: {args.syncnet_dir or '(默认 metrics/scores_LSE)'}")

    try:
        lse_c, lse_d = compute_lse_metric(
            args.video,
            model_path=args.model_path,
            syncnet_dir=args.syncnet_dir,
        )
        print(f"LSE-C (越大越好): {lse_c:.4f}")
        print(f"LSE-D (越小越好): {lse_d:.4f}")
    except Exception as e:
        print("LSE 计算失败：")
        print(str(e))

