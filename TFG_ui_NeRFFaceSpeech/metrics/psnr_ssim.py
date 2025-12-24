"""PSNR 和 SSIM 指标计算模块。

提供逐帧计算 PSNR 和 SSIM 的功能，支持图像和视频输入。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
from skimage.metrics import structural_similarity as ssim_func


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 PSNR（峰值信噪比）。
    
    Args:
        img1: 第一张图像，形状为 [H, W, C] 或 [H, W]，值域 [0, 255]
        img2: 第二张图像，形状为 [H, W, C] 或 [H, W]，值域 [0, 255]
    
    Returns:
        PSNR 分数（dB），值越大越好。如果两张图像完全相同，返回 inf。
    """
    # 确保两张图像尺寸相同
    if img1.shape != img2.shape:
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
    
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 SSIM（结构相似性指数）。
    
    Args:
        img1: 第一张图像，形状为 [H, W, C] 或 [H, W]，值域 [0, 255]
        img2: 第二张图像，形状为 [H, W, C] 或 [H, W]，值域 [0, 255]
    
    Returns:
        SSIM 分数，范围 [0, 1]，值越大越好。
    """
    # 确保两张图像尺寸相同
    if img1.shape != img2.shape:
        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
    
    # 使用 skimage 的 ssim，支持多通道
    if len(img1.shape) == 3:
        return float(ssim_func(img1, img2, channel_axis=2, data_range=255))
    else:
        return float(ssim_func(img1, img2, data_range=255))


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


def compute_psnr_from_videos(
    video1_path: Union[str, Path],
    video2_path: Union[str, Path],
    max_frames: Optional[int] = None,
) -> Tuple[float, List[float]]:
    """从两个视频文件计算 PSNR。
    
    Args:
        video1_path: 第一个视频文件路径
        video2_path: 第二个视频文件路径
        max_frames: 最大处理帧数，如果为 None 则处理所有帧
    
    Returns:
        (平均 PSNR, 逐帧 PSNR 列表)
    """
    frames1 = extract_frames_from_video(video1_path, max_frames)
    frames2 = extract_frames_from_video(video2_path, max_frames)
    
    # 对齐帧数
    min_frames = min(len(frames1), len(frames2))
    frames1 = frames1[:min_frames]
    frames2 = frames2[:min_frames]

    # 串行逐帧计算 PSNR，避免并行带来的额外复杂度
    psnr_values: List[float] = []
    for frame1, frame2 in zip(frames1, frames2):
        if frame1.shape != frame2.shape:
            h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
            frame1 = frame1[:h, :w]
            frame2 = frame2[:h, :w]
        psnr_values.append(calculate_psnr(frame1, frame2))
    
    avg_psnr = float(np.mean(psnr_values))
    return avg_psnr, psnr_values


def compute_ssim_from_videos(
    video1_path: Union[str, Path],
    video2_path: Union[str, Path],
    max_frames: Optional[int] = None,
) -> Tuple[float, List[float]]:
    """从两个视频文件计算 SSIM。
    
    Args:
        video1_path: 第一个视频文件路径
        video2_path: 第二个视频文件路径
        max_frames: 最大处理帧数，如果为 None 则处理所有帧
    
    Returns:
        (平均 SSIM, 逐帧 SSIM 列表)
    """
    frames1 = extract_frames_from_video(video1_path, max_frames)
    frames2 = extract_frames_from_video(video2_path, max_frames)
    
    # 对齐帧数
    min_frames = min(len(frames1), len(frames2))
    frames1 = frames1[:min_frames]
    frames2 = frames2[:min_frames]

    # 串行逐帧计算 SSIM，避免并行带来的额外复杂度
    ssim_values: List[float] = []
    for frame1, frame2 in zip(frames1, frames2):
        if frame1.shape != frame2.shape:
            h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
            frame1 = frame1[:h, :w]
            frame2 = frame2[:h, :w]
        ssim_values.append(calculate_ssim(frame1, frame2))

    avg_ssim = float(np.mean(ssim_values))
    return avg_ssim, ssim_values


if __name__ == "__main__":
    """简单测试用例。
    
    使用随机图像和临时视频文件测试 PSNR / SSIM 相关接口是否可以正常运行。
    """
    import tempfile
    import os

    print("=== PSNR / SSIM 简单测试 ===")

    # 1) 测试单张图像的 PSNR / SSIM
    img1 = np.full((64, 64, 3), 128, dtype=np.uint8)
    noise = np.random.randint(-10, 10, size=(64, 64, 3)).astype(np.int16)
    img2 = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    psnr_val = calculate_psnr(img1, img2)
    ssim_val = calculate_ssim(img1, img2)
    print(f"单张图像 PSNR: {psnr_val:.4f} dB, SSIM: {ssim_val:.4f}")

    # 2) 测试从视频计算 PSNR / SSIM（创建两个临时 mp4 视频）
    tmp_dir = tempfile.mkdtemp(prefix="psnr_ssim_test_")
    video1_path = os.path.join(tmp_dir, "video1.mp4")
    video2_path = os.path.join(tmp_dir, "video2.mp4")

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer1 = cv2.VideoWriter(video1_path, fourcc, 10.0, (64, 64))
        writer2 = cv2.VideoWriter(video2_path, fourcc, 10.0, (64, 64))

        for _ in range(10):
            frame1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            frame2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            writer1.write(frame1_bgr)
            writer2.write(frame2_bgr)
        writer1.release()
        writer2.release()

        avg_psnr, psnr_list = compute_psnr_from_videos(video1_path, video2_path, max_frames=5)
        avg_ssim, ssim_list = compute_ssim_from_videos(video1_path, video2_path, max_frames=5)

        print(f"视频平均 PSNR: {avg_psnr:.4f} dB，逐帧数量: {len(psnr_list)}")
        print(f"视频平均 SSIM: {avg_ssim:.4f}，逐帧数量: {len(ssim_list)}")
    finally:
        # 清理临时文件
        try:
            for p in (video1_path, video2_path):
                if os.path.exists(p):
                    os.remove(p)
            os.rmdir(tmp_dir)
        except Exception:
            pass

