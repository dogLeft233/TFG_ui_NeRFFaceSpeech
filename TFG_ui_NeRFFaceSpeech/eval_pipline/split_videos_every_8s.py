"""将指定文件夹中的所有 mp4 视频按固定时长切段的小脚本。

功能：
- 遍历输入目录中的所有 `.mp4` 文件
- 每隔 N 秒（默认 8 秒）切一段
- 不足 N 秒的视频会完整保留
- 不足 N 秒的尾段会被舍弃（但至少保留一个片段）
- 将切好的片段保存到输出目录

依赖：
- 需要系统已安装 `ffmpeg` 和 `ffprobe`

用法示例：
    python scripts/split_videos_every_8s.py \
        --input-dir data/geneface_datasets/data/raw/videos \
        --output-dir data/geneface_datasets/data/raw/videos_split \
        --segment-sec 8

输出命名示例：
    输入：May.mp4
    输出：May_seg000.mp4, May_seg001.mp4, ...
"""

from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按固定时长切分目录中的 mp4 视频")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="包含原始 mp4 文件的目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="保存切分后 mp4 文件的目录",
    )
    parser.add_argument(
        "--segment-sec",
        type=int,
        default=8,
        help="每段的时长（秒），默认 8 秒",
    )
    parser.add_argument(
        "--min-sec",
        type=int,
        default=8,
        help="最小时长（秒），默认 8 秒；不足该时长的尾段会被舍弃",
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
    return parser.parse_args()


def list_mp4_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input-dir 不存在: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input-dir 不是目录: {input_dir}")

    files = sorted(input_dir.glob("*.mp4"))
    if not files:
        raise FileNotFoundError(f"在目录中未找到任何 mp4 文件: {input_dir}")
    return files


def get_video_duration_sec(video_path: Path) -> float:
    """使用 ffprobe 获取视频时长（秒）"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return float(out.decode("utf-8").strip())
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe 获取时长失败: {video_path}\n{exc.output.decode('utf-8', errors='ignore')}") from exc
    except ValueError as exc:
        raise RuntimeError(f"无法解析视频时长: {video_path}") from exc


def split_video(
    video_path: Path,
    output_dir: Path,
    segment_sec: int,
    min_sec: int,
    max_segments: Optional[int] = None,
    random_segments: bool = False,
) -> None:
    duration = get_video_duration_sec(video_path)
    stem = video_path.stem  # 去掉扩展名
    
    # 如果视频时长小于 segment_sec，直接复制整个视频
    if duration < segment_sec:
        out_name = f"{stem}_seg000.mp4"
        out_path = output_dir / out_name
        print(f"[保留] {video_path.name} 时长 {duration:.2f}s < {segment_sec}s，完整保留")
        
        # 直接复制整个视频
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-c",
            "copy",
            str(out_path),
        ]
        
        print(f"  [片段] {out_name} (完整视频)")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:
            print(f"  [错误] 复制视频失败: {out_name} -> {exc}")
        return

    # 计算可切分的段数（舍弃不足 min_sec 的尾巴）
    num_segments = int(duration // segment_sec)
    if num_segments == 0:
        # 如果无法切出完整段，但视频时长 >= segment_sec，保留整个视频
        out_name = f"{stem}_seg000.mp4"
        out_path = output_dir / out_name
        print(f"[保留] {video_path.name} 时长 {duration:.2f}s，无法切出完整 {segment_sec}s 段，完整保留")
        
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-c",
            "copy",
            str(out_path),
        ]
        
        print(f"  [片段] {out_name} (完整视频)")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:
            print(f"  [错误] 复制视频失败: {out_name} -> {exc}")
        return

    # 计算所有可能的时段起始时间
    all_possible_starts = []
    for idx in range(num_segments):
        start_time = idx * segment_sec
        # 保证最后一段也至少有 min_sec
        if duration - start_time >= min_sec:
            all_possible_starts.append(start_time)
    
    # 应用 max_segments 限制
    if max_segments is not None and max_segments > 0:
        if random_segments and len(all_possible_starts) > max_segments:
            # 随机选择时段
            selected_starts = sorted(random.sample(all_possible_starts, max_segments))
            print(f"[切分] {video_path.name}: 总时长 {duration:.2f}s, 每段 {segment_sec}s, 可切段数 {len(all_possible_starts)}, 随机选择 {max_segments} 段")
        else:
            # 顺序选择前 N 个时段（或全部，如果可切段数 <= max_segments）
            selected_starts = all_possible_starts[:max_segments]
            if random_segments and len(all_possible_starts) <= max_segments:
                print(f"[切分] {video_path.name}: 总时长 {duration:.2f}s, 每段 {segment_sec}s, 可切段数 {len(all_possible_starts)} (无需随机选择，全部使用)")
            else:
                print(f"[切分] {video_path.name}: 总时长 {duration:.2f}s, 每段 {segment_sec}s, 可切段数 {len(all_possible_starts)}, 限制后段数 {len(selected_starts)}")
    else:
        selected_starts = all_possible_starts
        print(f"[切分] {video_path.name}: 总时长 {duration:.2f}s, 每段 {segment_sec}s, 段数 {len(selected_starts)}")

    # 切分选中的时段
    for idx, start_time in enumerate(selected_starts):
        out_name = f"{stem}_seg{idx:03d}.mp4"
        out_path = output_dir / out_name

        # 使用 -ss + -t + -c copy，避免重新编码，速度快
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time}",
            "-i",
            str(video_path),
            "-t",
            f"{segment_sec}",
            "-c",
            "copy",
            str(out_path),
        ]

        print(f"  [片段] {out_name} (start={start_time:.2f}s)")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:
            print(f"  [错误] 切分片段失败: {out_name} -> {exc}")


def main() -> int:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = list_mp4_files(args.input_dir)
    print(f"[信息] 在目录 {args.input_dir} 中找到 {len(videos)} 个 mp4 文件")

    for i, v in enumerate(videos):
        print("\n" + "=" * 60)
        print(f"[处理] ({i + 1}/{len(videos)}) {v.name}")
        print("=" * 60)
        try:
            split_video(
                video_path=v,
                output_dir=args.output_dir,
                segment_sec=args.segment_sec,
                min_sec=args.min_sec,
                max_segments=args.max_segments,
                random_segments=args.random_segments,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[错误] 处理视频 {v} 时出错: {exc}")

    print("\n" + "=" * 60)
    print("[完成] 全部视频切分结束")
    print("=" * 60)
    print(f"输出目录: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


