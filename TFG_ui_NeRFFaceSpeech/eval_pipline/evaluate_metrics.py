"""视频质量评测脚本，计算 NIQE, PSNR, FID, SSIM, LSE-C, LSE-D 指标。

使用 metrics/ 模块中的统一接口进行计算。

输入结构：
- 真值文件夹：包含真值 mp4 文件（例如 May.mp4）
- 生成结果文件夹：包含子文件夹（例如 May/），子文件夹中有 output_NeRFFaceSpeech.mp4

用法示例：
    python scripts/evaluate_metrics.py \
        --gt-dir data/geneface_datasets/data/raw/videos \
        --pred-dir output/raw_video_infer \
        --output results/metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# 导入 metrics 模块
import sys
sys.path.append(str(Path(__file__).parent.parent))
from metrics import compute_all_metrics, compute_lse_metric

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算视频质量评测指标")
    parser.add_argument(
        "--gt-dir",
        type=Path,
        required=True,
        help="真值视频文件夹（包含 mp4 文件）",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="生成结果文件夹（包含子文件夹，子文件夹中有 output_NeRFFaceSpeech.mp4）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出结果 JSON 文件路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备（默认 cuda）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批处理大小（默认 32，用于 FID）",
    )
    parser.add_argument(
        "--skip-lse",
        action="store_true",
        help="跳过 LSE-C 和 LSE-D 计算（如果 SyncNet 模型不可用）",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="指定要计算的指标列表，例如：--metrics lse 或 --metrics niqe psnr ssim",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最大处理帧数（如果为 None 则处理所有帧）",
    )
    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser.add_argument(
        "--lse-use-preprocessing",
        nargs='?',
        const=True,
        default=True,
        type=str_to_bool,
        help="LSE 计算时使用完整预处理流程（人脸检测、跟踪、裁剪），与 run_pipeline.py 一致（默认启用）。使用 --lse-use-preprocessing=False 禁用",
    )
    parser.add_argument(
        "--lse-data-dir",
        type=Path,
        default=None,
        help="LSE 预处理数据目录（仅在 --lse-use-preprocessing 时使用）",
    )
    parser.add_argument(
        "--lse-reference",
        type=str,
        default=None,
        help="LSE 预处理参考名称（仅在 --lse-use-preprocessing 时使用）",
    )
    parser.add_argument(
        "--lse-min-track",
        type=int,
        default=100,
        help="LSE 预处理最小轨迹长度（帧数），默认 100。如果视频较短或人脸较少，可以降低此值（例如 50）",
    )
    parser.add_argument(
        "--lse-facedet-scale",
        type=float,
        default=0.25,
        help="LSE 预处理人脸检测时的图像缩放比例，默认 0.25",
    )
    parser.add_argument(
        "--lse-crop-scale",
        type=float,
        default=0.40,
        help="LSE 预处理裁剪边界框的扩展比例，默认 0.40",
    )
    parser.add_argument(
        "--lse-frame-rate",
        type=int,
        default=25,
        help="LSE 预处理视频帧率，默认 25",
    )
    parser.add_argument(
        "--lse-num-failed-det",
        type=int,
        default=25,
        help="LSE 预处理允许的最大连续检测失败帧数，默认 25",
    )
    parser.add_argument(
        "--lse-min-face-size",
        type=int,
        default=100,
        help="LSE 预处理最小人脸尺寸（像素），默认 100",
    )
    return parser.parse_args()




def extract_base_name(video_name: str) -> str:
    """从视频文件名中提取基础名称（组名称）。
    
    支持格式：
    - "May_seg000" -> "May"
    - "May_seg001" -> "May"
    - "May" -> "May"
    """
    # 处理 "_seg" 格式（例如 "May_seg000"）
    if "_seg" in video_name:
        # 使用简单分割来提取组名称
        # 例如 "May_seg000" -> "May"
        parts = video_name.split("_seg")
        if len(parts) > 1:
            # 检查分割后的第二部分是否是数字（段编号）
            try:
                int(parts[1])  # 尝试转换为整数，如果是数字则说明是段编号
                return parts[0]
            except ValueError:
                # 如果不是数字，说明 "_seg" 是名称的一部分，返回原名称
                return video_name
    return video_name


def match_videos(gt_dir: Path, pred_dir: Path) -> List[Tuple[Path, Path, str]]:
    """匹配真值视频和生成视频。
    
    Returns:
        List of (gt_video_path, pred_video_path, base_name) tuples
    """
    matches = []
    
    # 查找所有真值视频
    gt_videos = sorted(gt_dir.glob("*.mp4")) + sorted(gt_dir.glob("*.MP4"))
    
    for gt_video in gt_videos:
        # 根据真值视频名（不含扩展名）查找对应的生成结果文件夹
        name = gt_video.stem
        base_name = extract_base_name(name)
        pred_folder = pred_dir / name
        pred_video = pred_folder / "output_NeRFFaceSpeech.mp4"
        
        if pred_video.exists():
            matches.append((gt_video, pred_video, base_name))
        else:
            logger.warning(f"未找到生成视频: {pred_video}")
    
    return matches


def evaluate_video_pair(
    gt_video: Path,
    pred_video: Path,
    device: str,
    batch_size: int,
    max_frames: Optional[int],
    skip_lse: bool = False,
    metrics: Optional[List[str]] = None,
    lse_use_preprocessing: bool = True,
    lse_data_dir: Optional[Path] = None,
    lse_reference: Optional[str] = None,
    lse_min_track: int = 100,
    lse_facedet_scale: float = 0.25,
    lse_crop_scale: float = 0.40,
    lse_frame_rate: int = 25,
    lse_num_failed_det: int = 25,
    lse_min_face_size: int = 100,
) -> Dict[str, float]:
    """计算一对视频的所有指标。
    
    使用 metrics/ 模块中的统一接口进行计算。
    """
    logger.info(f"评测: {gt_video.name} vs {pred_video.name}")
    
    results = {}
    
    # 使用 metrics 模块计算所有指标
    logger.info("使用 metrics 模块计算指标...")
    try:
        # 确定要计算的指标
        if metrics is not None:
            # 如果用户指定了指标列表，直接使用
            metrics_to_compute = metrics
        else:
            # 否则使用默认逻辑
            metrics_to_compute = ['niqe', 'psnr', 'ssim', 'fid']
            if not skip_lse:
                metrics_to_compute.append('lse')
        
        # 调用统一接口
        metrics_results = compute_all_metrics(
            video1_path=gt_video,
            video2_path=pred_video,
            max_frames=max_frames,
            metrics=metrics_to_compute,
            device=device,
            fid_batch_size=batch_size,
            lse_use_preprocessing=lse_use_preprocessing,
            lse_data_dir=lse_data_dir,
            lse_reference=lse_reference,
            lse_min_track=lse_min_track,
            lse_facedet_scale=lse_facedet_scale,
            lse_crop_scale=lse_crop_scale,
            lse_frame_rate=lse_frame_rate,
            lse_num_failed_det=lse_num_failed_det,
            lse_min_face_size=lse_min_face_size,
        )
        
        # 调试：输出 metrics_results 中的键（仅当只计算 LSE 时）
        if metrics_to_compute == ['lse']:
            logger.info(f"compute_all_metrics 返回的键: {list(metrics_results.keys())}")
            if 'LSE_error' in metrics_results:
                logger.warning(f"LSE 计算错误: {metrics_results['LSE_error']}")
        
        # 适配原有的结果键名格式
        # PSNR, SSIM, FID 直接使用
        if 'PSNR' in metrics_results:
            results['PSNR'] = float(metrics_results['PSNR'])
        else:
            results['PSNR'] = float('nan')
        
        if 'SSIM' in metrics_results:
            results['SSIM'] = float(metrics_results['SSIM'])
        else:
            results['SSIM'] = float('nan')
        
        if 'FID' in metrics_results:
            results['FID'] = float(metrics_results['FID'])
        else:
            results['FID'] = float('nan')
        
        # NIQE: 使用 NIQE 和 NIQE_video2（如果存在）
        if 'NIQE' in metrics_results:
            results['NIQE_GT'] = float(metrics_results['NIQE'])
        else:
            results['NIQE_GT'] = float('nan')
        
        if 'NIQE_video2' in metrics_results:
            results['NIQE_PRED'] = float(metrics_results['NIQE_video2'])
        elif 'NIQE' in metrics_results:
            # 如果没有 video2 的 NIQE，使用 video1 的（这种情况不应该发生，但为了兼容性）
            results['NIQE_PRED'] = float(metrics_results['NIQE'])
        else:
            results['NIQE_PRED'] = float('nan')
        
        # LSE: 只计算 pred_video 的 LSE（保持原有行为）
        if 'lse' in metrics_to_compute:
            # 检查是否有 LSE 错误信息
            if 'LSE_error' in metrics_results:
                logger.warning(f"LSE 计算错误: {metrics_results['LSE_error']}")
            
            # 优先使用 compute_all_metrics 返回的 video2 的 LSE
            if 'LSE_C_video2' in metrics_results:
                lse_c_val = metrics_results['LSE_C_video2']
                lse_d_val = metrics_results['LSE_D_video2']
                results['LSE_C'] = float(lse_c_val)
                results['LSE_D'] = float(lse_d_val)
                if np.isnan(lse_c_val) or np.isnan(lse_d_val):
                    logger.warning(f"LSE 结果为 NaN，可能计算失败。检查 SyncNet 环境和模型是否可用。")
                else:
                    logger.info(f"LSE 计算成功: LSE_C={lse_c_val:.4f}, LSE_D={lse_d_val:.4f}")
            elif 'LSE_C' in metrics_results:
                # 如果只有 video1 的 LSE（单视频场景），也使用
                lse_c_val = metrics_results['LSE_C']
                lse_d_val = metrics_results['LSE_D']
                results['LSE_C'] = float(lse_c_val)
                results['LSE_D'] = float(lse_d_val)
                if np.isnan(lse_c_val) or np.isnan(lse_d_val):
                    logger.warning(f"LSE 结果为 NaN，可能计算失败。检查 SyncNet 环境和模型是否可用。")
                else:
                    logger.info(f"LSE 计算成功: LSE_C={lse_c_val:.4f}, LSE_D={lse_d_val:.4f}")
            else:
                # 如果没有，单独计算 pred_video 的 LSE
                logger.info("compute_all_metrics 未返回 LSE 结果，单独计算生成视频的 LSE...")
                try:
                    lse_c, lse_d = compute_lse_metric(
                        str(pred_video),
                        use_preprocessing=lse_use_preprocessing,
                        data_dir=lse_data_dir,
                        reference=lse_reference,
                        min_track=lse_min_track,
                        facedet_scale=lse_facedet_scale,
                        crop_scale=lse_crop_scale,
                        frame_rate=lse_frame_rate,
                        num_failed_det=lse_num_failed_det,
                        min_face_size=lse_min_face_size,
                    )
                    results['LSE_C'] = float(lse_c)
                    results['LSE_D'] = float(lse_d)
                    logger.info(f"单独计算 LSE 成功: LSE_C={lse_c:.4f}, LSE_D={lse_d:.4f}")
                except Exception as e:
                    logger.error(f"单独计算 LSE 失败: {e}")
                    logger.error("请确保：1) 已激活 syncnet conda 环境；2) SyncNet 模型文件存在")
                    results['LSE_C'] = float('nan')
                    results['LSE_D'] = float('nan')
        else:
            results['LSE_C'] = float('nan')
            results['LSE_D'] = float('nan')
    
    except Exception as e:
        logger.error(f"指标计算失败: {e}")
        # 返回所有指标为 nan
        results = {
            'PSNR': float('nan'),
            'SSIM': float('nan'),
            'FID': float('nan'),
            'NIQE_GT': float('nan'),
            'NIQE_PRED': float('nan'),
            'LSE_C': float('nan'),
            'LSE_D': float('nan'),
        }
    
    return results


def main() -> int:
    args = parse_args()
    
    if not args.gt_dir.exists():
        raise FileNotFoundError(f"真值目录不存在: {args.gt_dir}")
    if not args.pred_dir.exists():
        raise FileNotFoundError(f"生成结果目录不存在: {args.pred_dir}")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # 匹配视频
    logger.info("匹配真值视频和生成视频...")
    matches = match_videos(args.gt_dir, args.pred_dir)
    
    if not matches:
        raise ValueError("未找到任何匹配的视频对")
    
    logger.info(f"找到 {len(matches)} 对匹配的视频")
    
    # 显示分组信息（用于调试）
    from collections import Counter
    base_names = [base_name for _, _, base_name in matches]
    name_counts = Counter(base_names)
    logger.info(f"分组统计（共 {len(name_counts)} 个组）:")
    for base_name, count in sorted(name_counts.items()):
        logger.info(f"  {base_name}: {count} 个视频")
    
    # 确定计算设备
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 对每对视频计算指标
    all_results = {}
    summary = defaultdict(list)
    grouped_results = defaultdict(list)  # 按 base_name 分组
    
    for i, (gt_video, pred_video, base_name) in enumerate(matches):
        logger.info(f"\n{'='*60}")
        logger.info(f"处理 ({i+1}/{len(matches)}): {gt_video.name} (基础名称: {base_name})")
        logger.info(f"{'='*60}")
        
        try:
            results = evaluate_video_pair(
                gt_video=gt_video,
                pred_video=pred_video,
                device=device,
                batch_size=args.batch_size,
                max_frames=args.max_frames,
                skip_lse=args.skip_lse,
                metrics=args.metrics,
                lse_use_preprocessing=args.lse_use_preprocessing,
                lse_data_dir=args.lse_data_dir,
                lse_reference=args.lse_reference,
                lse_min_track=args.lse_min_track,
                lse_facedet_scale=args.lse_facedet_scale,
                lse_crop_scale=args.lse_crop_scale,
                lse_frame_rate=args.lse_frame_rate,
                lse_num_failed_det=args.lse_num_failed_det,
                lse_min_face_size=args.lse_min_face_size,
            )
            
            all_results[gt_video.stem] = results
            
            # 收集到 summary 中用于计算总平均值
            for key, value in results.items():
                if not np.isnan(value):
                    summary[key].append(value)
            
            # 收集到分组中用于计算每组平均值
            grouped_results[base_name].append(results)
        
        except Exception as e:
            logger.error(f"处理 {gt_video.name} 时出错: {e}")
            all_results[gt_video.stem] = {"error": str(e)}
    
    # 计算每组（按 base_name）的平均指标
    group_summary = {}
    for base_name, results_list in grouped_results.items():
        group_metrics = defaultdict(list)
        for result in results_list:
            if "error" not in result:
                for key, value in result.items():
                    if not np.isnan(value):
                        group_metrics[key].append(value)
        
        group_avg = {}
        for key, values in group_metrics.items():
            if values:
                group_avg[f"{key}_mean"] = float(np.mean(values))
                group_avg[f"{key}_std"] = float(np.std(values))
        
        if group_avg:
            group_summary[base_name] = {
                "metrics": group_avg,
                "num_segments": len(results_list),
            }
    
    # 计算总平均指标
    summary_avg = {}
    for key, values in summary.items():
        if values:
            summary_avg[f"{key}_mean"] = float(np.mean(values))
            summary_avg[f"{key}_std"] = float(np.std(values))
    
    # 保存结果
    output_data = {
        "individual_results": all_results,
        "group_summary": group_summary,  # 按名称分组的平均指标
        "overall_summary": summary_avg,  # 总平均指标
        "total_videos": len(matches),
        "total_groups": len(group_summary),
        "successful_evaluations": len([r for r in all_results.values() if "error" not in r]),
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("评测完成！")
    logger.info(f"{'='*60}")
    logger.info(f"结果已保存到: {args.output}")
    
    # 显示分组结果
    logger.info(f"\n按名称分组的平均指标 (共 {len(group_summary)} 组):")
    for base_name, group_data in sorted(group_summary.items()):
        logger.info(f"\n  {base_name} (共 {group_data['num_segments']} 段):")
        for key, value in sorted(group_data['metrics'].items()):
            logger.info(f"    {key}: {value:.4f}")
    
    # 显示总平均结果
    logger.info(f"\n总平均指标 (共 {len(matches)} 个视频):")
    for key, value in sorted(summary_avg.items()):
        logger.info(f"  {key}: {value:.4f}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

