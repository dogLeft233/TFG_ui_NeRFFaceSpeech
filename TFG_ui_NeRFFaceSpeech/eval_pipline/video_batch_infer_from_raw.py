"""对 raw 视频目录中的每个 mp4 文件：

1. 提取完整音频为 wav。
2. 随机抽取一帧图像作为关键帧。
3. 调用 `main_NeRFFaceSpeech_audio_driven_from_image.py` 进行模型推理。
4. 将推理结果保存在独立子目录中，方便后续评测。

典型目录结构：
    data/geneface_datasets/data/raw/videos/
        Macron.mp4
        May.mp4
        ...

用法示例：
    python scripts/video_batch_infer_from_raw.py \
        --video-dir data/geneface_datasets/data/raw/videos \
        --network /path/to/your_network.pkl \
        --outdir outputs/raw_video_infer

说明：
- 每个 mp4 会在 `--outdir` 下生成一个同名子目录，例如 `outputs/raw_video_infer/May/`。
- 子目录内会包含：
    - `audio.wav`          : 从 mp4 提取的音频
    - `keyframe.png`       : 随机抽取的一帧图像
    - `output_NeRFFaceSpeech.mp4` : 模型推理生成的视频
- 如果某个视频对应的输出目录下已经存在 `output_NeRFFaceSpeech.mp4`，则会跳过该视频。
"""

from __future__ import annotations

import argparse
import subprocess
import os
from pathlib import Path

import cv2

# 项目根目录 = 当前脚本所在目录的上级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# NeRFFaceSpeech 代码根目录
NERF_CODE_DIR = PROJECT_ROOT / "NeRFFaceSpeech_Code"
# NeRFFaceSpeech 专用环境的 python 可执行文件
NERF_ENV_PYTHON = PROJECT_ROOT / "environment" / "nerffacespeech" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 raw mp4 批量生成 NeRFFaceSpeech 推理结果")
    parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="包含多个 mp4 文件的目录（例如 data/geneface_datasets/data/raw/videos）",
    )
    parser.add_argument(
        "--network",
        type=Path,
        required=True,
        help="训练好的生成器 pkl（传给 --network）",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="批量输出根目录，每个视频一个子目录",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若指定，则即使已有 output_NeRFFaceSpeech.mp4 也会重新生成",
    )
    parser.add_argument(
        "--gen-res",
        type=int,
        default=1024,
        help="生成器期望的输入分辨率（会将关键帧等比例缩放+填充到此分辨率，默认1024，对应 ffhq_1024）",
    )
    return parser.parse_args()


def list_videos(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"video-dir 不存在: {video_dir}")
    if not video_dir.is_dir():
        raise NotADirectoryError(f"video-dir 不是目录: {video_dir}")

    videos: list[Path] = []
    for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
        videos.extend(video_dir.glob(ext))
    videos = sorted(videos)
    if not videos:
        raise FileNotFoundError(f"在目录中未找到任何视频文件: {video_dir}")
    return videos


def extract_audio(video_path: Path, out_wav: Path) -> None:
    """使用 ffmpeg 从视频中提取音频到 wav 文件。

    为了与大部分语音模型兼容，这里统一转为 16kHz 单声道 PCM。
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # -y 覆盖输出；-vn 去掉视频；-ar 16000 采样率；-ac 1 单声道；pcm_s16le 无压缩 PCM
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_wav),
    ]

    print(f"[音频] {video_path.name} -> {out_wav.name}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg 提取音频失败: {video_path}") from exc


def extract_first_frame_and_resize(video_path: Path, out_image: Path, target_size: int = 1024) -> None:
    """提取第一帧，等比例缩放 + 上下左右填充黑色到 target_size x target_size。
    
    如果视频已经是 target_size x target_size，则直接保存，避免不必要的处理。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"无法读取第一帧: {video_path}")

    orig_h, orig_w = frame.shape[:2]

    # 如果已经是目标尺寸，直接保存，避免不必要的处理
    if orig_h == target_size and orig_w == target_size:
        out_image.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_image), frame):
            raise RuntimeError(f"保存关键帧失败: {out_image}")
        print(
            f"[关键帧] {video_path.name} -> {out_image.name} "
            f"(first frame, already {target_size}x{target_size}, no resize/padding needed)"
        )
        return

    # 等比例缩放 + padding 到 target_size x target_size
    h, w = orig_h, orig_w
    scale = 1.0
    if max(h, w) > target_size:
        scale = target_size / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    pad_top = (target_size - h) // 2
    pad_bottom = target_size - h - pad_top
    pad_left = (target_size - w) // 2
    pad_right = target_size - w - pad_left

    # 如果不需要 padding，直接保存
    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        frame_out = frame
    else:
        frame_out = cv2.copyMakeBorder(
            frame,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],  # 黑色填充
        )

    out_image.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_image), frame_out):
        raise RuntimeError(f"保存关键帧失败: {out_image}")

    print(
        f"[关键帧] {video_path.name} -> {out_image.name} "
        f"(first frame, padded to {target_size}x{target_size}, orig={orig_h}x{orig_w})"
    )

def run_inference(network: Path, outdir: Path, keyframe: Path, audio_wav: Path) -> Path:
    """调用现有 CLI 进行推理，返回生成的视频路径。"""
    outdir.mkdir(parents=True, exist_ok=True)
    pred_mp4 = outdir / "output_NeRFFaceSpeech.mp4"

    # 统一转为绝对路径，避免切换 cwd 后找不到文件
    network_abs = network if network.is_absolute() else (PROJECT_ROOT / network).resolve()
    keyframe_abs = keyframe.resolve()
    audio_wav_abs = audio_wav.resolve()
    outdir_abs = outdir.resolve()

    # 如果存在 nerffacespeech 环境，则优先使用该环境的 python
    python_exe = NERF_ENV_PYTHON if NERF_ENV_PYTHON.exists() else "python"
    env = os.environ.copy()
    # 确保 nerffacespeech 环境的 bin 在 PATH 中（包含 ninja 等可执行文件）
    env["PATH"] = f"{NERF_ENV_PYTHON.parent}:{env.get('PATH', '')}"

    # 在 NeRFFaceSpeech 代码根目录下运行，从而让脚本中的相对路径（如 pretrained_networks/seg.pth）生效
    # 调用 audio-driven 脚本：main_NeRFFaceSpeech_audio_driven_from_image.py
    cmd = [
        str(python_exe),
        "StyleNeRF/main_NeRFFaceSpeech_audio_driven_from_image.py",
        "--network",
        str(network_abs),
        "--outdir",
        str(outdir_abs),
        "--test_img",
        str(keyframe_abs),
        "--test_data",
        str(audio_wav_abs),
        "--trunc",
        str(0.7)
    ]

    print(f"[推理] 输出目录: {outdir}")
    print(f"运行{cmd}")
    subprocess.run(cmd, check=True, cwd=str(NERF_CODE_DIR), env=env)

    if not pred_mp4.exists():
        raise RuntimeError(f"推理完成但未找到输出视频: {pred_mp4}")

    print(f"[完成] 推理结果: {pred_mp4}")
    return pred_mp4


def cleanup_outputs(outdir: Path) -> None:
    """只保留关键文件，删除其它中间产物。"""
    keep_names = {
        "output_NeRFFaceSpeech.mp4",
        "keyframe.png",
        "audio.wav",
    }
    for item in outdir.iterdir():
        if item.name in keep_names:
            continue
        if item.is_dir():
            # 递归删除目录
            import shutil
            shutil.rmtree(item, ignore_errors=True)
            print(f"[清理] 删除目录: {item}")
        else:
            try:
                item.unlink()
                print(f"[清理] 删除文件: {item}")
            except Exception:
                pass


def process_one_video(
    video_path: Path,
    network: Path,
    out_root: Path,
    overwrite: bool = False,
    gen_res: int = 1024,
) -> None:
    name = video_path.stem  # e.g. May, Macron
    video_outdir = out_root / name
    video_outdir.mkdir(parents=True, exist_ok=True)

    pred_mp4 = video_outdir / "output_NeRFFaceSpeech.mp4"
    if pred_mp4.exists() and not overwrite:
        print(f"[跳过] 已存在推理结果: {pred_mp4}")
        return

    # 1) 提取音频
    audio_wav = video_outdir / "audio.wav"
    extract_audio(video_path, audio_wav)

    # 2) 提取第一帧作为关键帧并处理到 1024x1024
    keyframe = video_outdir / "keyframe.png"
    extract_first_frame_and_resize(video_path, keyframe, target_size=gen_res)

    # 3) 运行推理
    run_inference(network=network, outdir=video_outdir, keyframe=keyframe, audio_wav=audio_wav)

    # 4) 清理，只保留关键帧、音频和生成视频
    cleanup_outputs(video_outdir)


def main() -> int:
    args = parse_args()

    if not args.network.exists():
        raise FileNotFoundError(f"network 不存在: {args.network}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(args.video_dir)
    print(f"[信息] 在目录 {args.video_dir} 中找到 {len(videos)} 个视频文件")

    for i, v in enumerate(videos):
        print("\n" + "=" * 60)
        print(f"[处理] ({i + 1}/{len(videos)}) {v.name}")
        print("=" * 60)
        try:
            process_one_video(
                video_path=v,
                network=args.network,
                out_root=args.outdir,
                overwrite=args.overwrite,
                gen_res=args.gen_res,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[错误] 处理视频 {v} 时出错: {exc}")

    print("\n" + "=" * 60)
    print("[完成] 全部视频处理结束")
    print("=" * 60)
    print(f"输出目录: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


