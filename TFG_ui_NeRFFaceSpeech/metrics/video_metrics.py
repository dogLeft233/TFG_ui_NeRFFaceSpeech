"""视频指标计算统一接口。

提供接受 MP4 文件作为输入的指标计算接口，包括 NIQE、PSNR、SSIM、FID、LSE。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .niqe import compute_niqe_from_video
from .psnr_ssim import compute_psnr_from_videos, compute_ssim_from_videos
from .FID_FVD import compute_fid_from_videos
from .lse import compute_lse_from_video


def compute_all_metrics(
    video1_path: Union[str, Path],
    video2_path: Optional[Union[str, Path]] = None,
    max_frames: Optional[int] = None,
    metrics: Optional[List[str]] = None,
    device: str = 'cuda',
    fid_batch_size: int = 50,
    lse_model_path: Optional[Union[str, Path]] = None,
    lse_batch_size: int = 20,
    lse_vshift: int = 15,
    lse_tmp_dir: Optional[Union[str, Path]] = None,
    lse_syncnet_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Union[float, List[float]]]:
    """计算视频的所有指标。
    
    Args:
        video1_path: 第一个视频文件路径（MP4）
        video2_path: 第二个视频文件路径（MP4），如果提供则计算 PSNR、SSIM 和 FID
        max_frames: 最大处理帧数，如果为 None 则处理所有帧（不适用于 LSE）
        metrics: 要计算的指标列表，可选值：['niqe', 'psnr', 'ssim', 'fid', 'lse']
                 如果为 None，则根据是否提供 video2_path 自动选择
        device: 计算设备（'cuda' 或 'cpu'），用于 FID 计算
        fid_batch_size: FID 计算的批处理大小
        lse_model_path: LSE 计算的 SyncNet 模型路径
        lse_batch_size: LSE 计算的批处理大小
        lse_vshift: LSE 计算的视频偏移范围
        lse_tmp_dir: LSE 计算的临时目录
        lse_syncnet_dir: LSE 计算的 SyncNet 代码目录
    
    Returns:
        包含所有指标结果的字典：
        - 'NIQE': 平均 NIQE 分数（如果计算）
        - 'NIQE_per_frame': 逐帧 NIQE 列表（如果计算）
        - 'PSNR': 平均 PSNR 分数（如果计算，需要 video2_path）
        - 'PSNR_per_frame': 逐帧 PSNR 列表（如果计算）
        - 'SSIM': 平均 SSIM 分数（如果计算，需要 video2_path）
        - 'SSIM_per_frame': 逐帧 SSIM 列表（如果计算）
        - 'FID': FID 分数（如果计算，需要 video2_path）
        - 'LSE_C': LSE-C 分数（如果计算）
        - 'LSE_D': LSE-D 分数（如果计算）
    """
    if metrics is None:
        # 自动选择要计算的指标
        metrics = []
        if video2_path is None:
            metrics = ['niqe', 'lse']  # 单视频指标
        else:
            metrics = ['niqe', 'psnr', 'ssim', 'fid', 'lse']  # 所有指标
    
    results = {}
    
    # 计算 NIQE（单视频指标）
    if 'niqe' in metrics:
        avg_niqe, niqe_per_frame = compute_niqe_from_video(video1_path, max_frames)
        results['NIQE'] = avg_niqe
        results['NIQE_per_frame'] = niqe_per_frame
        
        if video2_path is not None:
            # 也计算第二个视频的 NIQE
            avg_niqe2, niqe_per_frame2 = compute_niqe_from_video(video2_path, max_frames)
            results['NIQE_video2'] = avg_niqe2
            results['NIQE_video2_per_frame'] = niqe_per_frame2
    
    # 计算 LSE（单视频指标，但可以为两个视频都计算）
    if 'lse' in metrics:
        try:
            lse_c, lse_d = compute_lse_from_video(
                video1_path,
                model_path=lse_model_path,
                batch_size=lse_batch_size,
                vshift=lse_vshift,
                tmp_dir=lse_tmp_dir,
                syncnet_dir=lse_syncnet_dir
            )
            results['LSE_C'] = lse_c
            results['LSE_D'] = lse_d
            
            if video2_path is not None:
                # 也计算第二个视频的 LSE
                lse_c2, lse_d2 = compute_lse_from_video(
                    video2_path,
                    model_path=lse_model_path,
                    batch_size=lse_batch_size,
                    vshift=lse_vshift,
                    tmp_dir=lse_tmp_dir,
                    syncnet_dir=lse_syncnet_dir
                )
                results['LSE_C_video2'] = lse_c2
                results['LSE_D_video2'] = lse_d2
        except Exception as e:
            # LSE 计算可能失败（例如未激活 syncnet 环境），记录但不中断
            results['LSE_C'] = float('nan')
            results['LSE_D'] = float('nan')
            results['LSE_error'] = str(e)
            if video2_path is not None:
                results['LSE_C_video2'] = float('nan')
                results['LSE_D_video2'] = float('nan')
    
    # 计算 PSNR、SSIM 和 FID（需要两个视频）
    if video2_path is not None:
        if 'psnr' in metrics:
            avg_psnr, psnr_per_frame = compute_psnr_from_videos(
                video1_path, video2_path, max_frames
            )
            results['PSNR'] = avg_psnr
            results['PSNR_per_frame'] = psnr_per_frame
        
        if 'ssim' in metrics:
            avg_ssim, ssim_per_frame = compute_ssim_from_videos(
                video1_path, video2_path, max_frames
            )
            results['SSIM'] = avg_ssim
            results['SSIM_per_frame'] = ssim_per_frame
        
        if 'fid' in metrics:
            fid_score = compute_fid_from_videos(
                video1_path, video2_path, device=device, 
                batch_size=fid_batch_size, max_frames=max_frames
            )
            results['FID'] = fid_score
    else:
        if 'psnr' in metrics or 'ssim' in metrics or 'fid' in metrics:
            raise ValueError("计算 PSNR、SSIM 或 FID 需要提供两个视频文件（video2_path）")
    
    return results


def compute_niqe_metric(
    video_path: Union[str, Path],
    max_frames: Optional[int] = None
) -> float:
    """计算视频的 NIQE 指标（便捷函数）。
    
    Args:
        video_path: 视频文件路径（MP4）
        max_frames: 最大处理帧数，如果为 None 则处理所有帧
    
    Returns:
        平均 NIQE 分数
    """
    avg_niqe, _ = compute_niqe_from_video(video_path, max_frames)
    return avg_niqe


def compute_psnr_metric(
    video1_path: Union[str, Path],
    video2_path: Union[str, Path],
    max_frames: Optional[int] = None
) -> float:
    """计算两个视频之间的 PSNR 指标（便捷函数）。
    
    Args:
        video1_path: 第一个视频文件路径（MP4）
        video2_path: 第二个视频文件路径（MP4）
        max_frames: 最大处理帧数，如果为 None 则处理所有帧
    
    Returns:
        平均 PSNR 分数（dB）
    """
    avg_psnr, _ = compute_psnr_from_videos(video1_path, video2_path, max_frames)
    return avg_psnr


def compute_ssim_metric(
    video1_path: Union[str, Path],
    video2_path: Union[str, Path],
    max_frames: Optional[int] = None
) -> float:
    """计算两个视频之间的 SSIM 指标（便捷函数）。
    
    Args:
        video1_path: 第一个视频文件路径（MP4）
        video2_path: 第二个视频文件路径（MP4）
        max_frames: 最大处理帧数，如果为 None 则处理所有帧
    
    Returns:
        平均 SSIM 分数（范围 [0, 1]）
    """
    avg_ssim, _ = compute_ssim_from_videos(video1_path, video2_path, max_frames)
    return avg_ssim


def compute_fid_metric(
    video1_path: Union[str, Path],
    video2_path: Union[str, Path],
    device: str = 'cuda',
    batch_size: int = 50,
    max_frames: Optional[int] = None
) -> float:
    """计算两个视频之间的 FID 指标（便捷函数）。
    
    Args:
        video1_path: 第一个视频文件路径（MP4）
        video2_path: 第二个视频文件路径（MP4）
        device: 计算设备（'cuda' 或 'cpu'）
        batch_size: 批处理大小
        max_frames: 最大处理帧数，如果为 None 则处理所有帧
    
    Returns:
        FID 分数，值越小越好
    """
    return compute_fid_from_videos(
        video1_path, video2_path, device=device, 
        batch_size=batch_size, max_frames=max_frames
    )


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
    """统一视频指标接口的简单测试 / 示例。
    
    示例用法：
        python -m metrics.video_metrics --video1 a.mp4 --video2 b.mp4 --no_lse --no_fid
    默认会在给定的视频上计算可用的指标，并打印结果字典。
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="视频指标统一接口简单测试")
    parser.add_argument(
        "--video1",
        type=str,
        required=True,
        help="第一个视频文件路径（MP4）",
    )
    parser.add_argument(
        "--video2",
        type=str,
        default=None,
        help="第二个视频文件路径（MP4，可选）",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=10,
        help="最大处理帧数，用于加快测试速度",
    )
    parser.add_argument(
        "--no_fid",
        action="store_true",
        help="测试时关闭 FID（避免下载 Inception 权重）",
    )
    parser.add_argument(
        "--no_lse",
        action="store_true",
        help="测试时关闭 LSE（避免对 SyncNet 环境的依赖）",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="指定要计算的指标列表，例如：--metrics lse 或 --metrics niqe psnr",
    )
    args = parser.parse_args()

    # 自动构建要计算的指标列表
    if args.metrics is not None:
        # 如果用户指定了 --metrics，直接使用
        metric_list = args.metrics
    else:
        # 否则使用默认逻辑
        metric_list = []
        if args.video2 is None:
            # 单视频场景：默认 NIQE + LSE（可关）
            metric_list.append("niqe")
            if not args.no_lse:
                metric_list.append("lse")
        else:
            # 双视频场景：默认全指标，可按需关闭
            metric_list.extend(["niqe", "psnr", "ssim"])
            if not args.no_fid:
                metric_list.append("fid")
            if not args.no_lse:
                metric_list.append("lse")

    print("=== 视频指标统一接口简单测试 ===")
    print(f"video1: {args.video1}")
    print(f"video2: {args.video2}")
    print(f"metrics: {metric_list}")

    results = compute_all_metrics(
        args.video1,
        video2_path=args.video2,
        max_frames=args.max_frames,
        metrics=metric_list,
        device="cpu",  # 测试默认使用 CPU，避免 GPU 依赖
    )

    print("指标结果：")
    # 使用 JSON 友好格式，仅展示标量和列表长度，避免过长输出
    summary = {}
    for k, v in results.items():
        if isinstance(v, list):
            summary[k] = {"len": len(v)}
        else:
            summary[k] = v
    print(json.dumps(summary, indent=2, ensure_ascii=False))

