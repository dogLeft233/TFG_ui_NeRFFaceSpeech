"""评估流程主入口。

完整的评估流程包括：
1. 将视频每8秒一划分
2. 对划分后的视频进行人脸检测和裁剪
3. 将处理完的视频送入模型推理
4. 用推理结果和真值视频计算指标

用法示例：
    # 基本用法
    python -m eval_pipline \
        --input-dir data/videos \
        --output-dir output/eval \
        --network path/to/network.pkl \
        --device cuda
    
    # 使用 FFHQFaceAlignment 对齐（推荐，方案 A）
    python -m eval_pipline \
        --input-dir data/videos \
        --output-dir output/eval \
        --network path/to/network.pkl \
        --device cuda \
        --ffhq-alignment
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="完整的模型评估流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
完整流程：
  1. 将输入视频每8秒一划分，保存到 {output_dir}/videos_split/
  2. 对划分后的视频进行人脸检测和裁剪，保存到 {output_dir}/videos_cropped/
  3. 将处理完的视频送入模型推理，保存到 {output_dir}/videos_infer/
  4. 用推理结果和真值视频计算指标，保存到 {output_dir}/metrics.json

FFHQFaceAlignment 使用说明（方案 A）：
  --ffhq-alignment: 使用 FFHQFaceAlignment 进行对齐
    - 从第一帧计算对齐参数（landmarks + affine matrix）
    - 对所有帧应用相同的对齐参数（不重新检测）
    - GT 和 GEN 使用相同的对齐参数，确保在同一坐标系下比较
    - 需要先安装依赖：pip install -r FFHQFaceAlignment/requirements.txt
    - 需要下载模型：python FFHQFaceAlignment/download.py
        """
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="包含原始 mp4 视频文件的输入目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出根目录，所有中间结果和最终结果将保存在此目录下",
    )
    parser.add_argument(
        "--network",
        type=Path,
        required=True,
        help="训练好的生成器模型文件路径（.pkl）",
    )
    parser.add_argument(
        "--segment-sec",
        type=int,
        default=8,
        help="视频切分的每段时长（秒），默认 8 秒",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="每个原始视频最多切分的片段数量上限（默认 None 表示无限制）",
    )
    parser.add_argument(
        "--random-segments",
        action="store_true",
        help="当设置了 --max-segments 时，随机选择时段进行切分（而不是从开头顺序切分）",
    )
    parser.add_argument(
        "--face-ratio",
        type=float,
        default=0.8,
        help="人脸占画面的比例（0.0-1.0，默认0.8）",
    )
    parser.add_argument(
        "--detect-interval",
        type=int,
        default=30,
        help="每隔多少帧检测一次人脸（默认30）",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1024, 1024),
        help="输出视频尺寸（默认1024x1024）",
    )
    parser.add_argument(
        "--gen-res",
        type=int,
        default=1024,
        help="生成器期望的输入分辨率（默认1024）",
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
        help="跳过 LSE-C 和 LSE-D 计算",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最大处理帧数（如果为 None 则处理所有帧）",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="跳过步骤1（视频切分），使用已存在的切分结果",
    )
    parser.add_argument(
        "--skip-crop",
        action="store_true",
        help="跳过步骤2（人脸裁剪），使用已存在的裁剪结果",
    )
    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="跳过步骤3（模型推理），使用已存在的推理结果",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="跳过步骤4（指标计算），只运行前面的步骤",
    )
    parser.add_argument(
        "--ffhq-style",
        action="store_true",
        help="使用 FFHQ-style 人脸对齐（需要 dlib 和 shape_predictor_68_face_landmarks.dat）",
    )
    parser.add_argument(
        "--landmark-model",
        type=Path,
        default=None,
        help="dlib 68点关键点模型路径（默认在 pretrained_networks 目录查找）",
    )
    parser.add_argument(
        "--resize-only",
        action="store_true",
        help="绕过人脸检测和裁剪，只对视频进行 resize（适用于已裁剪好的视频）",
    )
    parser.add_argument(
        "--ffhq-alignment",
        action="store_true",
        help="使用 FFHQFaceAlignment 进行对齐（从第一帧计算对齐参数，应用到所有帧）",
    )
    return parser.parse_args()


def run_step(step_name: str, script_path: Path, args: list[str], cwd: Path | None = None) -> bool:
    """运行一个步骤的脚本。"""
    logger.info(f"\n{'='*80}")
    logger.info(f"步骤: {step_name}")
    logger.info(f"{'='*80}")
    logger.info(f"运行命令: python {script_path} {' '.join(args)}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)] + args,
            cwd=str(cwd) if cwd else None,
            check=True,
        )
        logger.info(f"✓ {step_name} 完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {step_name} 失败: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ {step_name} 出错: {e}")
        return False


def main() -> int:
    args = parse_args()
    
    # 验证输入
    if not args.input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")
    if not args.network.exists():
        raise FileNotFoundError(f"模型文件不存在: {args.network}")
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义中间目录
    eval_pipline_dir = Path(__file__).parent
    videos_split_dir = args.output_dir / "videos_split"
    videos_cropped_dir = args.output_dir / "videos_cropped"
    videos_infer_dir = args.output_dir / "videos_infer"
    metrics_output = args.output_dir / "metrics.json"
    
    logger.info(f"\n{'='*80}")
    logger.info("开始评估流程")
    logger.info(f"{'='*80}")
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"模型文件: {args.network}")
    logger.info(f"\n中间目录:")
    logger.info(f"  - 切分视频: {videos_split_dir}")
    logger.info(f"  - 裁剪视频: {videos_cropped_dir}")
    logger.info(f"  - 推理结果: {videos_infer_dir}")
    logger.info(f"  - 指标结果: {metrics_output}")
    
    # 步骤1: 视频切分
    if not args.skip_split:
        split_args = [
            "--input-dir", str(args.input_dir),
            "--output-dir", str(videos_split_dir),
            "--segment-sec", str(args.segment_sec),
        ]
        if args.max_segments is not None:
            split_args.extend(["--max-segments", str(args.max_segments)])
        if args.random_segments:
            split_args.append("--random-segments")
        
        success = run_step(
            "步骤1: 视频切分（每8秒一段）",
            eval_pipline_dir / "split_videos_every_8s.py",
            split_args
        )
        if not success:
            logger.error("步骤1失败，终止流程")
            return 1
    else:
        logger.info("跳过步骤1: 视频切分")
        if not videos_split_dir.exists():
            logger.warning(f"警告: 切分目录不存在 {videos_split_dir}，但已设置 --skip-split")
    
    # 步骤2: 人脸检测和裁剪
    if not args.skip_crop:
        crop_args = [
            "--input-dir", str(videos_split_dir),
            "--output-dir", str(videos_cropped_dir),
            "--face-ratio", str(args.face_ratio),
            "--detect-interval", str(args.detect_interval),
            "--output-size", str(args.output_size[0]), str(args.output_size[1]),
        ]
        if args.ffhq_style:
            crop_args.append("--ffhq-style")
        if args.landmark_model:
            crop_args.extend(["--landmark-model", str(args.landmark_model)])
        if args.resize_only:
            crop_args.append("--resize-only")
        if args.ffhq_alignment:
            crop_args.append("--ffhq-alignment")
        
        success = run_step(
            "步骤2: 人脸检测和裁剪",
            eval_pipline_dir / "video_face_crop.py",
            crop_args
        )
        if not success:
            logger.error("步骤2失败，终止流程")
            return 1
    else:
        logger.info("跳过步骤2: 人脸裁剪")
        if not videos_cropped_dir.exists():
            logger.warning(f"警告: 裁剪目录不存在 {videos_cropped_dir}，但已设置 --skip-crop")
    
    # 步骤3: 模型推理
    if not args.skip_infer:
        success = run_step(
            "步骤3: 模型推理",
            eval_pipline_dir / "video_batch_infer_from_raw.py",
            [
                "--video-dir", str(videos_cropped_dir),
                "--network", str(args.network),
                "--outdir", str(videos_infer_dir),
                "--gen-res", str(args.gen_res),
            ]
        )
        if not success:
            logger.error("步骤3失败，终止流程")
            return 1
    else:
        logger.info("跳过步骤3: 模型推理")
        if not videos_infer_dir.exists():
            logger.warning(f"警告: 推理目录不存在 {videos_infer_dir}，但已设置 --skip-infer")
    
    # 步骤4: 指标计算
    if not args.skip_eval:
        # 真值视频目录：
        # - 如果使用了 FFHQ 对齐，GT 应该使用对齐后的视频（videos_cropped_dir）
        #   因为 GEN 视频也是基于对齐后的输入生成的，两者应该在同一坐标系下比较
        # - 否则使用切分后的原始视频（videos_split_dir）
        # 推理结果目录：使用推理输出（videos_infer_dir）
        gt_dir = videos_cropped_dir if (args.ffhq_alignment or args.ffhq_style or args.resize_only) else videos_split_dir
        eval_args = [
            "--gt-dir", str(gt_dir),
            "--pred-dir", str(videos_infer_dir),
            "--output", str(metrics_output),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
        ]
        if args.ffhq_alignment:
            logger.info(f"使用 FFHQ 对齐后的 GT 视频目录: {gt_dir}")
        if args.max_frames is not None:
            eval_args.extend(["--max-frames", str(args.max_frames)])
        if args.skip_lse:
            eval_args.append("--skip-lse")
        
        success = run_step(
            "步骤4: 指标计算",
            eval_pipline_dir / "evaluate_metrics.py",
            eval_args,
        )
        if not success:
            logger.error("步骤4失败")
            return 1
    else:
        logger.info("跳过步骤4: 指标计算")
    
    logger.info(f"\n{'='*80}")
    logger.info("评估流程完成！")
    logger.info(f"{'='*80}")
    logger.info(f"所有结果保存在: {args.output_dir}")
    logger.info(f"指标结果: {metrics_output}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

