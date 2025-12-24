"""LSE (Lip Sync Error) 指标计算模块。

提供 LSE-C (Confidence) 和 LSE-D (Distance) 的计算功能，使用 SyncNet 模型。
支持两种模式：
1. 直接模式：直接对 MP4 视频计算 LSE（简单快速）
2. 预处理模式：先运行完整预处理（人脸检测、跟踪、裁剪），再对裁剪后的人脸视频计算 LSE（与 syncnet_python/run_pipeline.py 一致）
"""

import os
import sys
import tempfile
import shutil
import subprocess
import glob
from pathlib import Path
from typing import Tuple, Optional, Union, List

import numpy as np


def run_preprocessing_pipeline(
    video_path: Union[str, Path],
    data_dir: Union[str, Path],
    reference: str = "lse_calc",
    syncnet_python_dir: Optional[Union[str, Path]] = None,
    facedet_scale: float = 0.25,
    crop_scale: float = 0.40,
    min_track: int = 100,
    frame_rate: int = 25,
    num_failed_det: int = 25,
    min_face_size: int = 100,
) -> List[Path]:
    """运行完整的预处理流程（与 run_pipeline.py 一致）。
    
    包括：视频转换、帧提取、音频提取、人脸检测、场景检测、人脸跟踪、人脸裁剪。
    
    Args:
        video_path: 输入视频文件路径（MP4）
        data_dir: 数据目录（将创建 pyavi, pyframes, pycrop 等子目录）
        reference: 参考名称（用于创建子目录）
        syncnet_python_dir: syncnet_python 目录路径
        facedet_scale: 人脸检测时的图像缩放比例
        crop_scale: 裁剪边界框的扩展比例
        min_track: 最小轨迹长度（帧数）
        frame_rate: 视频帧率
        num_failed_det: 允许的最大连续检测失败帧数
        min_face_size: 最小人脸尺寸（像素）
    
    Returns:
        裁剪后的人脸视频文件路径列表（AVI 格式，224x224）
    """
    video_path = Path(video_path)
    data_dir = Path(data_dir)
    syncnet_python_dir = Path(syncnet_python_dir) if syncnet_python_dir else None
    
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 确定 syncnet_python 目录
    if syncnet_python_dir is None:
        current_dir = Path(__file__).parent
        syncnet_python_dir = current_dir / "scores_LSE" / "syncnet_python"
        if not syncnet_python_dir.exists():
            if (current_dir / "scores_LSE" / "SyncNetModel.py").exists():
                syncnet_python_dir = current_dir / "scores_LSE"
            else:
                raise FileNotFoundError(f"SyncNet Python 目录不存在: {syncnet_python_dir}")
    
    if not syncnet_python_dir.exists():
        raise FileNotFoundError(f"SyncNet Python 目录不存在: {syncnet_python_dir}")
    
    # 检查 run_pipeline.py 是否存在
    run_pipeline_script = syncnet_python_dir / "run_pipeline.py"
    if not run_pipeline_script.exists():
        raise FileNotFoundError(f"run_pipeline.py 不存在: {run_pipeline_script}")
    
    # 创建数据目录
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建命令（使用绝对路径，确保 run_pipeline.py 能找到文件）
    video_path_abs = video_path.resolve()  # 转换为绝对路径
    data_dir_abs = data_dir.resolve()  # 转换为绝对路径
    
    cmd = [
        sys.executable,
        str(run_pipeline_script),
        "--videofile", str(video_path_abs),
        "--reference", reference,
        "--data_dir", str(data_dir_abs),
        "--facedet_scale", str(facedet_scale),
        "--crop_scale", str(crop_scale),
        "--min_track", str(min_track),
        "--frame_rate", str(frame_rate),
        "--num_failed_det", str(num_failed_det),
        "--min_face_size", str(min_face_size),
    ]
    
    # 运行预处理
    print(f"运行预处理流程: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(syncnet_python_dir),
            check=True,
            capture_output=False,  # 显示输出以便调试
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"预处理流程失败: {e}")
    
    # 查找生成的裁剪视频
    crop_dir = data_dir / "pycrop" / reference
    crop_videos = sorted(glob.glob(str(crop_dir / "0*.avi")))
    
    if not crop_videos:
        # 检查预处理中间结果，提供更详细的诊断信息
        work_dir = data_dir / "pywork" / reference
        faces_file = work_dir / "faces.pckl"
        tracks_file = work_dir / "tracks.pckl"
        
        diagnostic_info = [
            f"预处理完成，但未找到裁剪后的人脸视频。",
            f"检查目录: {crop_dir}",
            f"",
        ]
        
        # 检查人脸检测结果
        faces_data = None
        if faces_file.exists():
            try:
                import pickle
                with open(faces_file, 'rb') as f:
                    faces_data = pickle.load(f)
                total_faces = sum(len(frame_faces) for frame_faces in faces_data)
                frames_with_faces = sum(1 for frame_faces in faces_data if len(frame_faces) > 0)
                diagnostic_info.append(f"✓ 人脸检测完成: 共 {frames_with_faces}/{len(faces_data)} 帧检测到人脸，总计 {total_faces} 个人脸检测")
            except Exception as e:
                diagnostic_info.append(f"⚠ 无法读取人脸检测结果: {faces_file} ({e})")
        else:
            diagnostic_info.append(f"✗ 未找到人脸检测结果: {faces_file}")
        
        # 检查场景检测结果
        scene_file = work_dir / "scene.pckl"
        scene_data = None
        if scene_file.exists():
            try:
                import pickle
                with open(scene_file, 'rb') as f:
                    scenes = pickle.load(f)
                scene_data = scenes
                if scenes:
                    scene_info = []
                    for i, scene in enumerate(scenes):
                        try:
                            # FrameTimecode 对象有 frame_num 属性
                            start_frame = scene[0].frame_num
                            end_frame = scene[1].frame_num
                            scene_length = end_frame - start_frame
                            scene_info.append(f"场景 {i}: 帧 {start_frame}-{end_frame} (长度: {scene_length})")
                        except Exception as e:
                            scene_info.append(f"场景 {i}: 无法获取帧号 ({e})")
                    diagnostic_info.append(f"✓ 场景检测: 共 {len(scenes)} 个场景")
                    diagnostic_info.extend([f"  {s}" for s in scene_info])
                else:
                    diagnostic_info.append(f"⚠ 场景列表为空")
            except Exception as e:
                diagnostic_info.append(f"⚠ 无法读取场景结果: {scene_file} ({e})")
        
        # 检查轨迹结果
        if tracks_file.exists():
            try:
                import pickle
                with open(tracks_file, 'rb') as f:
                    tracks = pickle.load(f)
                if tracks:
                    track_lengths = [len(t['frame']) for t in tracks]
                    diagnostic_info.append(f"✓ 轨迹文件存在: 找到 {len(tracks)} 个轨迹，长度: {track_lengths}")
                    diagnostic_info.append(f"  但未生成裁剪视频，可能原因：轨迹长度 < {min_track} 帧（最小要求）")
                else:
                    diagnostic_info.append(f"⚠ 轨迹文件为空: {tracks_file}")
                    # 分析为什么没有轨迹：检查场景长度和跟踪参数
                    if scene_data:
                        for i, scene in enumerate(scene_data):
                            try:
                                start_frame = scene[0].frame_num
                                end_frame = scene[1].frame_num
                                scene_length = end_frame - start_frame
                                if scene_length < min_track:
                                    diagnostic_info.append(f"  场景 {i} 长度 ({scene_length} 帧) < min_track ({min_track} 帧)，跳过跟踪")
                                else:
                                    diagnostic_info.append(f"  场景 {i} 长度 ({scene_length} 帧) >= min_track ({min_track} 帧)，但未生成轨迹")
                                    # 检查该场景范围内的人脸检测情况
                                    if faces_data is not None:
                                        try:
                                            scene_faces = faces_data[start_frame:end_frame] if start_frame < len(faces_data) else []
                                            if scene_faces:
                                                total_faces_in_scene = sum(len(f) for f in scene_faces)
                                                frames_with_faces_in_scene = sum(1 for f in scene_faces if len(f) > 0)
                                                diagnostic_info.append(f"    场景内人脸检测: {frames_with_faces_in_scene}/{len(scene_faces)} 帧有检测，总计 {total_faces_in_scene} 个")
                                            else:
                                                diagnostic_info.append(f"    场景范围内无人脸检测数据（start_frame={start_frame}, len(faces)={len(faces_data)}）")
                                        except Exception as e:
                                            diagnostic_info.append(f"    无法分析场景内人脸检测: {e}")
                                    diagnostic_info.append(f"    可能原因：")
                                    diagnostic_info.append(f"      - 连续帧之间的人脸 IOU < 0.5（人脸位置变化太大，无法连接成轨迹）")
                                    diagnostic_info.append(f"      - 平均人脸尺寸 < min_face_size ({min_face_size} 像素）")
                                    diagnostic_info.append(f"      - 建议：降低 min_track 参数（例如：--min_track 50）或检查人脸检测质量")
                            except Exception as e:
                                diagnostic_info.append(f"  场景 {i}: 无法分析 ({e})")
            except:
                diagnostic_info.append(f"⚠ 无法读取轨迹结果: {tracks_file}")
        else:
            diagnostic_info.append(f"✗ 未找到轨迹结果: {tracks_file}")
        
        diagnostic_info.extend([
            f"",
            f"建议：",
            f"  1. 检查视频是否包含清晰的人脸",
            f"  2. 尝试降低 min_track 参数（当前: {min_track} 帧）",
            f"  3. 检查人脸检测是否成功（查看 {work_dir}/faces.pckl）",
        ])
        
        raise RuntimeError("\n".join(diagnostic_info))
    
    print(f"找到 {len(crop_videos)} 个人脸轨迹视频")
    return [Path(v) for v in crop_videos]


def compute_lse_from_video(
    video_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    batch_size: int = 20,
    vshift: int = 15,
    tmp_dir: Optional[Union[str, Path]] = None,
    syncnet_dir: Optional[Union[str, Path]] = None,
    use_preprocessing: bool = False,
    data_dir: Optional[Union[str, Path]] = None,
    reference: Optional[str] = None,
    return_all_tracks: bool = False,
    facedet_scale: float = 0.25,
    crop_scale: float = 0.40,
    min_track: int = 100,
    frame_rate: int = 25,
    num_failed_det: int = 25,
    min_face_size: int = 100,
) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
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
        use_preprocessing: 是否使用完整预处理流程（人脸检测、跟踪、裁剪）
                          如果为 True，将先运行 run_pipeline.py 的预处理，再对裁剪后的人脸视频计算 LSE
        data_dir: 预处理数据目录（仅在 use_preprocessing=True 时使用）
        reference: 预处理参考名称（仅在 use_preprocessing=True 时使用）
        return_all_tracks: 如果为 True 且 use_preprocessing=True，返回所有轨迹的 LSE 列表
                          否则返回平均 LSE
    
    Returns:
        如果 return_all_tracks=False: (LSE-C, LSE-D) 元组
        如果 return_all_tracks=True: [(LSE-C, LSE-D), ...] 列表（每个轨迹一个）
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
    
    # 如果使用预处理流程
    if use_preprocessing:
        # 设置数据目录和参考名称
        if data_dir is None:
            if tmp_dir is None:
                data_dir = Path(tempfile.mkdtemp(prefix="lse_preprocess_"))
            else:
                data_dir = Path(tmp_dir)
        else:
            data_dir = Path(data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
        
        if reference is None:
            reference = "lse_calc"
        
        # 运行预处理流程
        crop_videos = run_preprocessing_pipeline(
            video_path=video_path,
            data_dir=data_dir,
            reference=reference,
            syncnet_python_dir=syncnet_python_dir,
            facedet_scale=facedet_scale,
            crop_scale=crop_scale,
            min_track=min_track,
            frame_rate=frame_rate,
            num_failed_det=num_failed_det,
            min_face_size=min_face_size,
        )
        
        # 创建临时目录用于 LSE 计算
        if tmp_dir is None:
            tmp_dir = Path(tempfile.mkdtemp(prefix="lse_"))
        else:
            tmp_dir = Path(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # 对每个裁剪后的人脸视频计算 LSE
        all_lse_results = []
        for crop_video in crop_videos:
            offset, confidence, min_distance = s.evaluate(opt, videofile=str(crop_video))
            lse_c = float(confidence)
            lse_d = float(min_distance)
            all_lse_results.append((lse_c, lse_d))
            print(f"轨迹 {crop_video.name}: LSE-C={lse_c:.4f}, LSE-D={lse_d:.4f}")
        
        if return_all_tracks:
            return all_lse_results
        else:
            # 返回平均 LSE
            avg_lse_c = np.mean([r[0] for r in all_lse_results])
            avg_lse_d = np.mean([r[1] for r in all_lse_results])
            print(f"平均 LSE: LSE-C={avg_lse_c:.4f}, LSE-D={avg_lse_d:.4f}")
            return avg_lse_c, avg_lse_d
    
    else:
        # 直接模式：直接对原始视频计算 LSE
        # 创建临时目录
        if tmp_dir is None:
            tmp_dir = Path(tempfile.mkdtemp(prefix="lse_"))
        else:
            tmp_dir = Path(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        reference_name = reference or "lse_calc"
        tmp_subdir = tmp_dir / reference_name
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
                    self.reference = reference_name
            
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
    syncnet_dir: Optional[Union[str, Path]] = None,
    use_preprocessing: bool = False,
    data_dir: Optional[Union[str, Path]] = None,
    reference: Optional[str] = None,
    return_all_tracks: bool = False,
) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
    """计算视频的 LSE 指标（便捷函数）。
    
    Args:
        video_path: 视频文件路径（MP4）
        model_path: SyncNet 模型文件路径
        batch_size: 批处理大小
        vshift: 视频偏移范围
        tmp_dir: 临时目录
        syncnet_dir: SyncNet 代码目录
        use_preprocessing: 是否使用完整预处理流程
        data_dir: 预处理数据目录
        reference: 预处理参考名称
        return_all_tracks: 是否返回所有轨迹的 LSE
    
    Returns:
        如果 return_all_tracks=False: (LSE-C, LSE-D) 元组
        如果 return_all_tracks=True: [(LSE-C, LSE-D), ...] 列表
    """
    return compute_lse_from_video(
        video_path, model_path, batch_size, vshift, tmp_dir, syncnet_dir,
        use_preprocessing, data_dir, reference, return_all_tracks
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
    parser.add_argument(
        "--use_preprocessing",
        action="store_true",
        help="使用完整预处理流程（人脸检测、跟踪、裁剪），与 run_pipeline.py 一致",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="预处理数据目录（仅在 --use_preprocessing 时使用）",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="预处理参考名称（仅在 --use_preprocessing 时使用）",
    )
    parser.add_argument(
        "--return_all_tracks",
        action="store_true",
        help="返回所有轨迹的 LSE（仅在 --use_preprocessing 时使用）",
    )
    parser.add_argument(
        "--min_track",
        type=int,
        default=100,
        help="最小轨迹长度（帧数），默认 100。如果视频较短或人脸较少，可以降低此值（仅在 --use_preprocessing 时使用）",
    )
    parser.add_argument(
        "--facedet_scale",
        type=float,
        default=0.25,
        help="人脸检测时的图像缩放比例，默认 0.25（仅在 --use_preprocessing 时使用）",
    )
    parser.add_argument(
        "--crop_scale",
        type=float,
        default=0.40,
        help="裁剪边界框的扩展比例，默认 0.40（仅在 --use_preprocessing 时使用）",
    )
    args = parser.parse_args()

    print("=== LSE 简单测试 ===")
    print(f"视频: {args.video}")
    print(f"SyncNet 目录: {args.syncnet_dir or '(默认 metrics/scores_LSE)'}")
    print(f"使用预处理: {args.use_preprocessing}")

    try:
        result = compute_lse_metric(
            args.video,
            model_path=args.model_path,
            syncnet_dir=args.syncnet_dir,
            use_preprocessing=args.use_preprocessing,
            data_dir=args.data_dir,
            reference=args.reference,
            return_all_tracks=args.return_all_tracks,
        )
        
        if args.return_all_tracks and args.use_preprocessing:
            print(f"\n找到 {len(result)} 个人脸轨迹:")
            for i, (lse_c, lse_d) in enumerate(result):
                print(f"  轨迹 {i}: LSE-C={lse_c:.4f}, LSE-D={lse_d:.4f}")
            avg_lse_c = np.mean([r[0] for r in result])
            avg_lse_d = np.mean([r[1] for r in result])
            print(f"\n平均 LSE:")
            print(f"  LSE-C (越大越好): {avg_lse_c:.4f}")
            print(f"  LSE-D (越小越好): {avg_lse_d:.4f}")
        else:
            lse_c, lse_d = result
            print(f"LSE-C (越大越好): {lse_c:.4f}")
            print(f"LSE-D (越小越好): {lse_d:.4f}")
    except Exception as e:
        print("LSE 计算失败：")
        print(str(e))

