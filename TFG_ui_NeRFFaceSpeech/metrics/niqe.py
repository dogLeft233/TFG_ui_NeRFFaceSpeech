"""NIQE 指标计算模块。

提供无参考图像质量评估（NIQE）的计算功能，支持图像和视频输入。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

def calculate_niqe(img: np.ndarray) -> float:
    """计算 NIQE（无参考图像质量评估）。
    
    NIQE 是一个无参考图像质量评估指标，值越小表示图像质量越好。
    
    Args:
        img: 输入图像，形状为 [H, W, C] 或 [H, W]，值域 [0, 255]，RGB 格式
    
    Returns:
        NIQE 分数，值越小越好。
    """
    try:
        # 尝试使用 pyiqa 库（如果可用）
        import pyiqa
        import torch
        
        niqe_metric = pyiqa.create_metric('niqe', device='cpu')
        # 转换为 torch 张量 [1, C, H, W]
        if len(img.shape) == 2:
            # 灰度图，转换为 RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        score = niqe_metric(img_tensor)
        return float(score.item())
    except ImportError:
        # 如果没有 pyiqa，使用简化实现
        # 转换为灰度图
        logger.info(f"No pyiqa, using simplified implementation")
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # 计算梯度特征
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # 计算对比度
        contrast = np.std(gray)
        
        # 简化的 NIQE 估计
        niqe_score = np.std(gradient_magnitude) / (contrast + 1e-6)
        return float(niqe_score)


def extract_frames_from_video(
    video_path: Union[str, Path], 
    max_frames: Optional[int] = None
) -> List[np.ndarray]:
    """从视频中提取所有帧或采样帧。
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大提取帧数，如果为 None 则提取所有帧
    
    Returns:
        帧列表，每个元素为 RGB 格式的 numpy 数组 [H, W, 3]，值域 [0, 255]
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is not None and total_frames > max_frames:
        # 均匀采样
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        # 提取所有帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    return frames


def compute_niqe_from_video(
    video_path: Union[str, Path],
    max_frames: Optional[int] = None,
    batch_size: int = 100,
) -> Tuple[float, List[float]]:
    """从视频文件计算 NIQE。
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大处理帧数，如果为 None 则处理所有帧
    
    Returns:
        (平均 NIQE, 逐帧 NIQE 列表)
    """
    frames = extract_frames_from_video(video_path, max_frames)

    # 尽量在视频级别只创建一次 pyiqa 的 NIQE metric，避免重复构建网络与日志刷屏；
    # 如果有 CUDA，则使用 GPU 并按 batch 处理所有帧。
    try:
        import pyiqa
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        niqe_metric = pyiqa.create_metric("niqe", device=device)

        # 转为 [N, C, H, W]，按 batch 喂给网络
        tensors = []
        for img in frames:
            # frames 已经是 RGB，[H, W, 3]
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            t = (
                torch.from_numpy(img)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )
            tensors.append(t)

        imgs_tensor = torch.stack(tensors, dim=0).to(device)

        niqe_values: List[float] = []
        for start in range(0, imgs_tensor.size(0), batch_size):
            end = start + batch_size
            batch = imgs_tensor[start:end]
            scores = niqe_metric(batch)  # [B]
            niqe_values.extend([float(s) for s in scores.detach().cpu().view(-1)])
    except ImportError:
        # 没装 pyiqa 时，退回到单帧版本（内部会走简化实现）
        niqe_values = [calculate_niqe(frame) for frame in frames]

    avg_niqe = float(np.mean(niqe_values))
    return avg_niqe, niqe_values


if __name__ == "__main__":
    """简单测试用例。
    
    使用随机图像和临时视频文件测试 NIQE 相关接口是否可以正常运行。
    该测试不依赖外部模型文件（如果未安装 pyiqa，则会走简化实现）。
    """
    import tempfile
    import os

    print("=== NIQE 简单测试 ===")

    # 1) 测试单张图像的 NIQE
    img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    score_img = calculate_niqe(img)
    print(f"单张随机图像 NIQE: {score_img:.4f}")

    # 2) 测试从视频计算 NIQE（创建一个临时 mp4 视频）
    tmp_dir = tempfile.mkdtemp(prefix="niqe_test_")
    video_path = os.path.join(tmp_dir, "test_video.mp4")

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 10.0, (64, 64))

        # 写入若干帧随机图像
        for _ in range(10):
            frame_bgr = cv2.cvtColor(
                np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            writer.write(frame_bgr)
        writer.release()

        avg_niqe, per_frame = compute_niqe_from_video(video_path, max_frames=5)
        print(f"视频平均 NIQE: {avg_niqe:.4f}，前 5 帧 NIQE 数量: {len(per_frame)}")
    finally:
        # 清理临时文件
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            os.rmdir(tmp_dir)
        except Exception:
            # 清理失败不影响测试
            pass

