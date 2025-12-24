"""视频质量评估指标库。

提供 NIQE、PSNR、SSIM、FID 等指标的计算功能，支持 MP4 视频文件作为输入。
"""

from .FID_FVD import FIDCalculator, compute_fid, compute_fid_from_videos
from .lse import compute_lse_from_video
from .niqe import calculate_niqe, compute_niqe_from_video
from .psnr_ssim import (
    calculate_psnr,
    calculate_ssim,
    compute_psnr_from_videos,
    compute_ssim_from_videos,
)
from .video_metrics import (
    compute_all_metrics,
    compute_fid_metric,
    compute_lse_metric,
    compute_niqe_metric,
    compute_psnr_metric,
    compute_ssim_metric,
)

__all__ = [
    # FID
    'FIDCalculator',
    'compute_fid',
    'compute_fid_from_videos',
    'compute_fid_metric',
    # NIQE
    'calculate_niqe',
    'compute_niqe_from_video',
    'compute_niqe_metric',
    # PSNR
    'calculate_psnr',
    'compute_psnr_from_videos',
    'compute_psnr_metric',
    # SSIM
    'calculate_ssim',
    'compute_ssim_from_videos',
    'compute_ssim_metric',
    # LSE
    'compute_lse_from_video',
    'compute_lse_metric',
    # 统一接口
    'compute_all_metrics',
]

