"""
角色训练工具模块
支持从视频中提取训练数据（图像和音频）
"""
import subprocess
import cv2
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import shutil
import tempfile

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import PROJECT_ROOT, NERF_CONDA_PYTHON, MODEL_DIR, NERF_CODE_DIR, NERF_CONDA_ENV

# 日志函数（与 run_nerffacespeech.py 保持一致）
def add_log(message, level="info"):
    """添加日志到日志系统"""
    logger = logging.getLogger()
    level_map = {
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "success": logging.INFO,
        "progress": logging.INFO,
    }
    log_level = level_map.get(level, logging.INFO)
    logger.log(log_level, message)

# 角色数据目录
CHARACTER_DIR = PROJECT_ROOT / "assets" / "charactor"


def process_video_for_training(
    video_path: Path,
    character_name: str,
    face_ratio: float = 0.6,
    output_size: Tuple[int, int] = (1024, 1024),
    ffhq_alignment: bool = True,
    overwrite: bool = False,
) -> Dict:
    """
    处理视频用于角色训练：
    1. 使用 video_face_crop.py 处理视频（人脸检测、对齐、裁剪）
    2. 从处理后的视频中提取帧（保存为图像）
    3. 从原始视频中提取音频
    
    Args:
        video_path: 输入视频文件路径
        character_name: 角色名称（用于创建输出目录）
        face_ratio: 人脸占画面的比例（默认0.6）
        output_size: 输出视频尺寸（默认1024x1024）
        ffhq_alignment: 是否使用 FFHQ 对齐（默认True）
        overwrite: 是否覆盖已存在的输出文件
    
    Returns:
        dict: 处理结果，包含状态、输出路径等信息
    """
    import traceback
    try:
        add_log(f"[处理] ========== 开始处理视频用于训练 ==========", "info")
        add_log(f"[处理] 输入视频: {video_path}", "info")
        add_log(f"[处理] 角色名称: {character_name}", "info")
        add_log(f"[处理] 参数: face_ratio={face_ratio}, output_size={output_size}, ffhq_alignment={ffhq_alignment}, overwrite={overwrite}", "info")
        
        # 创建角色目录
        character_output_dir = CHARACTER_DIR / character_name
        add_log(f"[处理] 创建角色输出目录: {character_output_dir}", "info")
        character_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已存在且不覆盖
        if not overwrite and character_output_dir.exists():
            images_dir = character_output_dir / "images"
            audio_file = character_output_dir / "audio.wav"
            if images_dir.exists() and audio_file.exists():
                add_log(f"[处理] ⚠️ 角色数据已存在，跳过处理", "warning")
                return {
                    "success": True,
                    "message": f"角色 {character_name} 的训练数据已存在",
                    "character_dir": str(character_output_dir),
                    "images_dir": str(images_dir),
                    "audio_file": str(audio_file),
                }
        
        # 创建临时目录用于处理
        temp_dir = Path(tempfile.mkdtemp(prefix="character_training_"))
        temp_video = temp_dir / "input_video.mp4"
        cropped_video = temp_dir / "cropped_video.mp4"
        add_log(f"[处理] 创建临时目录: {temp_dir}", "info")
        
        # 复制视频到临时目录
        add_log(f"[处理] 复制视频到临时目录...", "info")
        shutil.copy2(video_path, temp_video)
        add_log(f"[处理] 视频已复制: {temp_video}", "info")
        
        # 步骤1: 使用 video_face_crop.py 处理视频
        add_log(f"[处理] ========== 步骤1: 视频人脸处理 ==========", "info")
        add_log(f"[处理] 开始处理视频: {video_path.name}", "info")
        crop_result = crop_video_face(
            input_video=temp_video,
            output_video=cropped_video,
            face_ratio=face_ratio,
            output_size=output_size,
            ffhq_alignment=ffhq_alignment,
        )
        
        if not crop_result["success"]:
            error_msg = crop_result.get('error', '未知错误')
            add_log(f"[处理] ❌ 视频处理失败: {error_msg}", "error")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {
                "success": False,
                "error": f"视频处理失败: {error_msg}"
            }
        add_log(f"[处理] ✅ 视频处理成功", "info")
        
        # 步骤2: 从处理后的视频中提取帧
        add_log(f"[处理] ========== 步骤2: 提取视频帧 ==========", "info")
        images_dir = character_output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        add_log(f"[提取] 输出目录: {images_dir}", "info")
        
        add_log(f"[提取] 开始提取视频帧...", "info")
        extract_result = extract_frames_from_video(
            video_path=cropped_video,
            output_dir=images_dir,
            frame_prefix="frame",
        )
        
        if not extract_result["success"]:
            error_msg = extract_result.get('error', '未知错误')
            add_log(f"[提取] ❌ 帧提取失败: {error_msg}", "error")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {
                "success": False,
                "error": f"帧提取失败: {error_msg}"
            }
        num_frames = extract_result.get("num_frames", 0)
        add_log(f"[提取] ✅ 帧提取成功，共 {num_frames} 帧", "info")
        
        # 步骤3: 从原始视频中提取音频
        add_log(f"[处理] ========== 步骤3: 提取音频 ==========", "info")
        audio_file = character_output_dir / "audio.wav"
        add_log(f"[提取] 音频输出文件: {audio_file}", "info")
        add_log(f"[提取] 开始提取音频...", "info")
        audio_result = extract_audio_from_video(
            video_path=video_path,
            output_audio=audio_file,
        )
        
        if not audio_result["success"]:
            error_msg = audio_result.get('error', '未知错误')
            add_log(f"[提取] ❌ 音频提取失败: {error_msg}", "error")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {
                "success": False,
                "error": f"音频提取失败: {error_msg}"
            }
        add_log(f"[提取] ✅ 音频提取成功", "info")
        
        # 步骤4: 调用 PTI 训练脚本生成模型文件
        add_log(f"[处理] ========== 步骤4: 生成 PTI 模型 ==========", "info")
        add_log(f"[训练] 开始生成 PTI 模型文件...", "info")
        pti_result = generate_pti_models(
            character_dir=character_output_dir,
            images_dir=images_dir,
            audio_file=audio_file,
            base_model="ffhq_1024.pkl",
        )
        
        if not pti_result["success"]:
            # PTI 训练失败不影响数据提取，只记录警告
            error_msg = pti_result.get('error', '未知错误')
            add_log(f"[训练] ⚠️ PTI 模型生成失败: {error_msg}", "warning")
            add_log(f"[训练] ⚠️ 训练数据已提取，但模型文件未生成，可能需要手动运行训练", "warning")
        else:
            add_log(f"[训练] ✅ PTI 模型生成成功", "info")
        
        # 清理临时目录
        add_log(f"[处理] 清理临时目录...", "info")
        shutil.rmtree(temp_dir, ignore_errors=True)
        add_log(f"[处理] 临时目录已清理", "info")
        
        result = {
            "success": True,
            "message": f"角色 {character_name} 的训练数据已准备完成",
            "character_dir": str(character_output_dir),
            "images_dir": str(images_dir),
            "audio_file": str(audio_file),
            "num_frames": num_frames,
        }
        
        # 添加模型文件信息
        if pti_result.get("success"):
            result["pti_models"] = {
                "G_PTI": pti_result.get("G_PTI"),
                "w_PTI": pti_result.get("w_PTI"),
                "bg_PTI": pti_result.get("bg_PTI"),
            }
        
        add_log(f"[处理] ========== 处理完成 ==========", "info")
        return result
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        add_log(f"[处理] ========== 处理异常 ==========", "error")
        add_log(f"[处理] ❌ 异常类型: {type(e).__name__}", "error")
        add_log(f"[处理] ❌ 异常消息: {str(e)}", "error")
        add_log(f"[处理] ❌ 异常堆栈:\n{error_traceback}", "error")
        add_log(f"[处理] ========================================", "error")
        return {
            "success": False,
            "error": f"处理过程中出错: {str(e)}"
        }


def crop_video_face(
    input_video: Path,
    output_video: Path,
    face_ratio: float = 0.6,
    output_size: Tuple[int, int] = (1024, 1024),
    ffhq_alignment: bool = True,
) -> Dict:
    """
    使用 video_face_crop.py 处理视频（人脸检测、对齐、裁剪）
    
    Args:
        input_video: 输入视频路径
        output_video: 输出视频路径
        face_ratio: 人脸占画面的比例
        output_size: 输出视频尺寸
        ffhq_alignment: 是否使用 FFHQ 对齐
    
    Returns:
        dict: 处理结果
    """
    try:
        # video_face_crop.py 脚本路径
        crop_script = PROJECT_ROOT / "eval_pipline" / "video_face_crop.py"
        
        if not crop_script.exists():
            return {
                "success": False,
                "error": f"未找到视频处理脚本: {crop_script}"
            }
        
        # 创建临时输入和输出目录（video_face_crop.py 需要目录）
        temp_input_dir = input_video.parent / "temp_input"
        temp_output_dir = input_video.parent / "temp_output"
        temp_input_dir.mkdir(exist_ok=True)
        temp_output_dir.mkdir(exist_ok=True)
        
        # 复制视频到临时输入目录
        temp_input_video = temp_input_dir / input_video.name
        shutil.copy2(input_video, temp_input_video)
        
        # 构建命令（使用 NERF_CONDA_PYTHON 以确保有正确的依赖）
        python_cmd = str(NERF_CONDA_PYTHON) if NERF_CONDA_PYTHON.exists() else "python"
        
        cmd = [
            python_cmd,
            str(crop_script),
            "--input-dir", str(temp_input_dir),
            "--output-dir", str(temp_output_dir),
            "--face-ratio", str(face_ratio),
            "--output-size", str(output_size[0]), str(output_size[1]),
            "--overwrite",
        ]
        
        if ffhq_alignment:
            cmd.append("--ffhq-alignment")
        
        # 执行命令
        add_log(f"[视频处理] 执行命令: {' '.join(cmd)}", "info")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        
        # 检查输出文件
        output_file = temp_output_dir / temp_input_video.name
        
        # 检查脚本返回码和错误输出
        has_error = False
        error_keywords = ['错误', 'Error', 'ERROR', 'Exception', 'RuntimeError', 'FFHQFaceAlignment', '未检测到关键点', '处理视频', '时出错', 'Traceback']
        
        # 检查返回码
        if result.returncode != 0:
            has_error = True
            add_log(f"[视频处理] ❌ 脚本执行失败（返回码: {result.returncode}）", "error")
        
        # 检查 stderr 中是否有错误信息（即使返回码为0，也可能有错误）
        error_msg = result.stderr if result.stderr else ""
        stdout_msg = result.stdout if result.stdout else ""
        
        # 检查 stderr 或 stdout 中是否包含错误关键词
        if error_msg or stdout_msg:
            combined_output = (error_msg + "\n" + stdout_msg).lower()
            if any(keyword.lower() in combined_output for keyword in error_keywords):
                has_error = True
                add_log(f"[视频处理] ❌ 检测到错误信息", "error")
        
        # 如果检测到错误，记录详细信息
        if has_error:
            add_log(f"[视频处理] ========== 检测到错误信息 ==========", "error")
            add_log(f"[视频处理] 返回码: {result.returncode}", "error")
            
            if error_msg:
                add_log(f"[视频处理] stderr 输出:", "error")
                # 提取关键错误信息
                error_lines = error_msg.split('\n')
                for line in error_lines:
                    if line.strip():  # 只记录非空行
                        if any(keyword in line for keyword in error_keywords):
                            add_log(f"[视频处理]   ERROR: {line}", "error")
                        else:
                            add_log(f"[视频处理]   {line}", "info")
            
            if stdout_msg:
                add_log(f"[视频处理] stdout 输出:", "info")
                # 检查 stdout 中是否也有错误信息
                stdout_lines = stdout_msg.split('\n')
                for line in stdout_lines:
                    if line.strip():  # 只记录非空行
                        if any(keyword in line for keyword in error_keywords):
                            add_log(f"[视频处理]   ERROR: {line}", "error")
                        else:
                            add_log(f"[视频处理]   {line}", "info")
            
            # 如果错误信息太长，只显示最后部分
            full_error = (error_msg + "\n" + stdout_msg).strip()
            error_summary = full_error[-2000:] if len(full_error) > 2000 else full_error
            
            # 如果输出文件不存在，直接返回错误
            if not output_file.exists():
                add_log(f"[视频处理] ❌ 输出文件不存在: {output_file}", "error")
                add_log(f"[视频处理] ========================================", "error")
                # 清理临时目录
                shutil.rmtree(temp_input_dir, ignore_errors=True)
                shutil.rmtree(temp_output_dir, ignore_errors=True)
                
                return {
                    "success": False,
                    "error": f"视频处理失败（返回码: {result.returncode}）。错误信息: {error_summary}"
                }
            # 如果输出文件存在但检测到错误，记录警告但继续
            else:
                add_log(f"[视频处理] ⚠️  检测到错误但输出文件已生成，继续处理", "warning")
                add_log(f"[视频处理] 输出文件: {output_file}", "info")
                add_log(f"[视频处理] ========================================", "warning")
        
        # 检查输出文件是否存在
        if output_file.exists():
            # 移动到目标位置
            shutil.move(str(output_file), str(output_video))
            add_log(f"[视频处理] ✅ 视频处理成功: {output_video.name}", "success")
            # 清理临时目录
            shutil.rmtree(temp_input_dir, ignore_errors=True)
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            
            return {
                "success": True,
                "output_video": str(output_video),
            }
        else:
            # 输出脚本的标准输出和错误输出用于调试
            if result.stdout:
                add_log(f"[视频处理] 脚本标准输出: {result.stdout[-500:]}", "warning")
            if result.stderr:
                add_log(f"[视频处理] 脚本错误输出: {result.stderr[-500:]}", "error")
            
            # 清理临时目录
            shutil.rmtree(temp_input_dir, ignore_errors=True)
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            
            error_msg = result.stderr if result.stderr else result.stdout
            error_summary = error_msg[-1000:] if error_msg and len(error_msg) > 1000 else (error_msg or "未知错误")
            
            return {
                "success": False,
                "error": f"视频处理失败，未生成输出文件。错误信息: {error_summary}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"视频处理异常: {str(e)}"
        }


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    frame_prefix: str = "frame",
    image_format: str = "jpg",
) -> Dict:
    """
    从视频中提取所有帧并保存为图像
    
    Args:
        video_path: 输入视频路径
        output_dir: 输出图像目录
        frame_prefix: 帧文件名前缀
        image_format: 图像格式（jpg/png）
    
    Returns:
        dict: 提取结果，包含提取的帧数
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                "success": False,
                "error": f"无法打开视频文件: {video_path}"
            }
        
        frame_count = 0
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 保存帧
            frame_filename = f"{frame_prefix}_{frame_idx:06d}.{image_format}"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            frame_count += 1
            frame_idx += 1
        
        cap.release()
        
        return {
            "success": True,
            "num_frames": frame_count,
            "output_dir": str(output_dir),
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"帧提取异常: {str(e)}"
        }


def extract_audio_from_video(
    video_path: Path,
    output_audio: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Dict:
    """
    从视频中提取音频并保存为 WAV 文件
    
    Args:
        video_path: 输入视频路径
        output_audio: 输出音频路径
        sample_rate: 采样率（默认16000Hz）
        channels: 声道数（默认1，单声道）
    
    Returns:
        dict: 提取结果
    """
    try:
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用 ffmpeg 提取音频
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",  # 不包含视频
            "-acodec", "pcm_s16le",  # PCM 16位小端
            "-ar", str(sample_rate),  # 采样率
            "-ac", str(channels),  # 声道数
            str(output_audio),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"音频提取失败: {result.stderr[:500]}"
            }
        
        if not output_audio.exists():
            return {
                "success": False,
                "error": "音频文件未生成"
            }
        
        return {
            "success": True,
            "output_audio": str(output_audio),
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"音频提取异常: {str(e)}"
        }


def get_character_training_status(character_name: str) -> Dict:
    """
    获取角色训练数据的状态
    
    Args:
        character_name: 角色名称
    
    Returns:
        dict: 状态信息
    """
    character_dir = CHARACTER_DIR / character_name
    
    if not character_dir.exists():
        return {
            "exists": False,
            "character_name": character_name,
        }
    
    images_dir = character_dir / "images"
    audio_file = character_dir / "audio.wav"
    
    num_images = 0
    if images_dir.exists():
        num_images = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
    
    audio_exists = audio_file.exists()
    
    # 检查 PTI 模型文件
    g_pti = character_dir / "G_PTI.pt"
    w_pti = character_dir / "w_PTI.pt"
    bg_pti = character_dir / "bg_PTI.pt"
    
    pti_models_exist = {
        "G_PTI": g_pti.exists(),
        "w_PTI": w_pti.exists(),
        "bg_PTI": bg_pti.exists(),
    }
    all_pti_models_exist = all(pti_models_exist.values())
    
    return {
        "exists": True,
        "character_name": character_name,
        "character_dir": str(character_dir),
        "images_dir": str(images_dir) if images_dir.exists() else None,
        "num_images": num_images,
        "audio_file": str(audio_file) if audio_exists else None,
        "audio_exists": audio_exists,
        "pti_models": pti_models_exist,
        "pti_models_exist": all_pti_models_exist,
    }


def generate_pti_models(
    character_dir: Path,
    images_dir: Path,
    audio_file: Path,
    base_model: str = "ffhq_1024.pkl",
    truncation_psi: float = 1.0,
) -> Dict:
    """
    调用 main_NeRFFaceSpeech_audio_driven_w_given_poses.py 生成 PTI 模型文件
    
    Args:
        character_dir: 角色目录（输出目录）
        images_dir: 图像目录（用于 motion_guide_img_folder）
        audio_file: 音频文件路径
        base_model: 基础模型文件名（默认 ffhq_1024.pkl）
        truncation_psi: Truncation psi 参数（默认 1.0）
    
    Returns:
        dict: 处理结果，包含生成的模型文件路径
    """
    try:
        # 检查基础模型是否存在
        base_model_path = MODEL_DIR / base_model
        if not base_model_path.exists():
            return {
                "success": False,
                "error": f"基础模型不存在: {base_model_path}"
            }
        
        # 检查图像目录是否存在且有图像文件
        if not images_dir.exists():
            return {
                "success": False,
                "error": f"图像目录不存在: {images_dir}"
            }
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if not image_files:
            return {
                "success": False,
                "error": f"图像目录中没有图像文件: {images_dir}"
            }
        
        # 选择第一帧作为输入图像（用于 PTI）
        test_img = sorted(image_files)[0]
        
        # 验证输入图像文件
        if not test_img.exists():
            return {
                "success": False,
                "error": f"输入图像文件不存在: {test_img}"
            }
        
        # 检查音频文件是否存在
        if not audio_file.exists():
            return {
                "success": False,
                "error": f"音频文件不存在: {audio_file}"
            }
        
        # 验证所有路径都是绝对路径（避免相对路径问题）
        test_img = test_img.resolve()
        audio_file = audio_file.resolve()
        images_dir = images_dir.resolve()
        character_dir = character_dir.resolve()
        base_model_path = base_model_path.resolve()
        
        # 检查模型文件是否已存在
        g_pti = character_dir / "G_PTI.pt"
        w_pti = character_dir / "w_PTI.pt"
        bg_pti = character_dir / "bg_PTI.pt"
        
        if g_pti.exists() and w_pti.exists() and bg_pti.exists():
            add_log(f"[PTI训练] ✅ 模型文件已存在，跳过训练", "info")
            return {
                "success": True,
                "G_PTI": str(g_pti),
                "w_PTI": str(w_pti),
                "bg_PTI": str(bg_pti),
                "models_generated": ["G_PTI.pt", "w_PTI.pt", "bg_PTI.pt"],
                "skipped": True,
            }
        
        # PTI 训练脚本路径
        pti_script = NERF_CODE_DIR / "StyleNeRF" / "main_NeRFFaceSpeech_audio_driven_w_given_poses.py"
        if not pti_script.exists():
            return {
                "success": False,
                "error": f"PTI 训练脚本不存在: {pti_script}"
            }
        
        # 如果视频文件存在，删除它以确保脚本会重新运行并生成模型文件
        output_video = character_dir / "output_NeRFFaceSpeech.mp4"
        if output_video.exists():
            add_log(f"[PTI训练] 删除已存在的视频文件以重新生成模型: {output_video}", "info")
            try:
                output_video.unlink()
            except Exception as e:
                add_log(f"[PTI训练] ⚠️  删除视频文件失败: {e}", "warning")
        
        # 构建命令
        # 脚本路径使用相对于工作目录的路径（工作目录是 NERF_CODE_DIR）
        script_relative_path = Path("StyleNeRF") / pti_script.name
        cmd = [
            str(NERF_CONDA_PYTHON),
            str(script_relative_path),  # 使用相对路径，因为工作目录是 NERF_CODE_DIR
            "--network", str(base_model_path),
            "--outdir", str(character_dir),
            "--test_data", str(audio_file),
            "--test_img", str(test_img),
            "--motion_guide_img_folder", str(images_dir),
            "--trunc", str(truncation_psi),
            "--noise-mode", "const",
        ]
        
        add_log(f"[PTI训练] 执行命令: {' '.join(cmd)}", "info")
        add_log(f"[PTI训练] 参数验证:", "info")
        add_log(f"[PTI训练]   输入图像: {test_img} (存在: {test_img.exists()})", "info")
        add_log(f"[PTI训练]   音频文件: {audio_file} (存在: {audio_file.exists()})", "info")
        add_log(f"[PTI训练]   图像目录: {images_dir} (存在: {images_dir.exists()}, 图像数: {len(image_files)})", "info")
        add_log(f"[PTI训练]   输出目录: {character_dir} (存在: {character_dir.exists()})", "info")
        add_log(f"[PTI训练]   基础模型: {base_model_path} (存在: {base_model_path.exists()})", "info")
        
        # 再次验证关键文件是否存在
        if not test_img.exists():
            return {
                "success": False,
                "error": f"输入图像文件不存在: {test_img}"
            }
        if not audio_file.exists():
            return {
                "success": False,
                "error": f"音频文件不存在: {audio_file}"
            }
        if not base_model_path.exists():
            return {
                "success": False,
                "error": f"基础模型文件不存在: {base_model_path}"
            }
        
        # 执行命令（PTI 训练可能需要较长时间）
        add_log(f"[PTI训练] 开始训练，这可能需要较长时间...", "info")
        add_log(f"[PTI训练] 使用 Python 环境: {NERF_CONDA_PYTHON}", "info")
        
        # 脚本使用相对路径 pretrained_networks/seg.pth，需要从 NERF_CODE_DIR 运行
        # 而不是从 StyleNeRF 目录运行
        script_work_dir = NERF_CODE_DIR
        add_log(f"[PTI训练] 工作目录: {script_work_dir}", "info")
        
        # 设置环境变量，确保使用正确的 Python 路径和模块路径
        env = os.environ.copy()
        # 添加 StyleNeRF 目录到 PYTHONPATH，以便导入模块
        env["PYTHONPATH"] = str(NERF_CODE_DIR / "StyleNeRF") + os.pathsep + str(NERF_CODE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        # 确保 PATH 包含 nerffacespeech 环境的 bin 目录，以便找到 ninja 等工具
        nerf_env_bin = NERF_CONDA_ENV / "bin"
        if nerf_env_bin.exists():
            current_path = env.get("PATH", "")
            env["PATH"] = str(nerf_env_bin) + os.pathsep + current_path
            add_log(f"[PTI训练] 设置 PATH，包含: {nerf_env_bin}", "info")
        
        # 使用 Popen 实时输出日志（与推理时保持一致）
        add_log(f"[PTI训练] 执行命令: {' '.join(cmd)}", "info")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',  # 使用replace模式处理非UTF-8字节，避免解码错误
                bufsize=1,
                env=env,
                cwd=str(script_work_dir),
                universal_newlines=True
            )
            
            # 实时读取输出并添加到日志
            error_lines = []  # 收集错误信息
            for line in process.stdout:
                if line:
                    line = line.rstrip()
                    # 根据内容判断日志级别
                    if 'ERROR' in line or '错误' in line or 'Error' in line or 'Exception' in line or 'Traceback' in line or 'FileNotFoundError' in line or 'RuntimeError' in line:
                        add_log(f"[PTI训练] {line}", "error")
                        error_lines.append(line)
                    elif 'WARNING' in line or '警告' in line or 'Warning' in line or 'WARN' in line:
                        add_log(f"[PTI训练] {line}", "warning")
                    elif '%|' in line or 'it/s' in line or 'Processing frames' in line or 'tqdm' in line.lower():
                        # 训练进度信息
                        add_log(f"[PTI训练] {line}", "progress")
                    elif 'Loading' in line or 'loading' in line or 'Loaded' in line:
                        add_log(f"[PTI训练] {line}", "info")
                    elif 'epoch' in line.lower() or 'iteration' in line.lower() or 'step' in line.lower():
                        add_log(f"[PTI训练] {line}", "progress")
                    elif line.strip():  # 忽略空行
                        add_log(f"[PTI训练] {line}", "info")
            
            process.wait()
            result_code = process.returncode
            
            # 为了兼容后续代码，创建一个类似 subprocess.run 的结果对象
            # 将错误信息合并为字符串
            error_output = "\n".join(error_lines) if error_lines else ""
            
            # 创建一个简单的结果对象
            class ProcessResult:
                def __init__(self, returncode, stderr="", stdout=""):
                    self.returncode = returncode
                    self.stderr = stderr
                    self.stdout = stdout
            
            result = ProcessResult(result_code, stderr=error_output, stdout="")
            
        except Exception as e:
            add_log(f"[PTI训练] 执行过程异常: {str(e)}", "error")
            # 创建失败结果
            class ProcessResult:
                def __init__(self, returncode, stderr="", stdout=""):
                    self.returncode = returncode
                    self.stderr = stderr
                    self.stdout = stdout
            result = ProcessResult(1, stderr=str(e), stdout="")
        
        # 检查生成的模型文件
        models_generated = []
        if g_pti.exists():
            models_generated.append("G_PTI.pt")
        if w_pti.exists():
            models_generated.append("w_PTI.pt")
        if bg_pti.exists():
            models_generated.append("bg_PTI.pt")
        
        # 如果模型文件已生成，即使返回码不为0也认为成功（脚本可能在其他地方出错但模型已生成）
        if len(models_generated) >= 2:
            add_log(f"[PTI训练] ✅ 模型文件生成成功: {', '.join(models_generated)}", "success")
            if result.returncode != 0:
                add_log(f"[PTI训练] ⚠️  脚本返回码: {result.returncode}，但模型文件已生成", "warning")
            return {
                "success": True,
                "G_PTI": str(g_pti) if g_pti.exists() else None,
                "w_PTI": str(w_pti) if w_pti.exists() else None,
                "bg_PTI": str(bg_pti) if bg_pti.exists() else None,
                "models_generated": models_generated,
            }
        else:
            # 如果模型文件未生成，输出详细错误信息
            error_msg = ""
            if result.stderr:
                error_msg = result.stderr
            elif result.stdout:
                error_msg = result.stdout
            else:
                error_msg = "未知错误"
            
            add_log(f"[PTI训练] ❌ 训练失败（返回码: {result.returncode}）", "error")
            
            # 检查常见错误并提供解决建议
            error_lower = error_msg.lower()
            suggestions = []
            
            if "ninja is required" in error_lower or "ninja" in error_lower:
                suggestions.append("缺少 Ninja 构建工具。请安装: apt-get install ninja-build 或 conda install ninja")
            if "cuda" in error_lower and ("not found" in error_lower or "unavailable" in error_lower):
                suggestions.append("CUDA 不可用。请检查 GPU 和 CUDA 环境配置")
            if "no module named" in error_lower:
                suggestions.append("缺少 Python 模块。请检查 nerffacespeech 环境的依赖是否完整安装")
            if "file not found" in error_lower or "no such file" in error_lower:
                suggestions.append("文件未找到。请检查所有必需的预训练模型文件是否存在")
            
            # 输出错误信息到日志（错误信息已经在实时输出中记录，这里只输出摘要）
            if result.stderr:
                add_log(f"[PTI训练] 错误摘要: {result.stderr[:500]}", "error")
            
            if suggestions:
                add_log(f"[PTI训练] 💡 可能的解决方案:", "warning")
                for i, suggestion in enumerate(suggestions, 1):
                    add_log(f"[PTI训练]   {i}. {suggestion}", "warning")
            
            # 提取关键错误信息（最后1000字符用于返回）
            error_summary = error_msg[-1000:] if len(error_msg) > 1000 else error_msg
            
            return {
                "success": False,
                "error": f"PTI 训练失败（返回码: {result.returncode}）: {error_summary}",
                "models_generated": models_generated,
                "returncode": result.returncode,
                "full_error": error_msg,  # 保存完整错误信息
                "suggestions": suggestions,  # 提供解决建议
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"PTI 训练异常: {str(e)}"
        }


def list_characters() -> Dict:
    """
    列出所有已训练的角色
    
    Returns:
        dict: 角色列表
    """
    characters = []
    
    if not CHARACTER_DIR.exists():
        return {
            "success": True,
            "characters": [],
        }
    
    for char_dir in CHARACTER_DIR.iterdir():
        if char_dir.is_dir():
            status = get_character_training_status(char_dir.name)
            characters.append(status)
    
    return {
        "success": True,
        "characters": characters,
    }

