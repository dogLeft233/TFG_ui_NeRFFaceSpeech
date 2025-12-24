import subprocess
import os
from pathlib import Path

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NERF_CONDA_ENV,
    NERF_CONDA_PYTHON,
    NERF_SCRIPT,
    NERF_CODE_DIR as NERF_WORKDIR,
    MODEL_DIR,
    PROJECT_ROOT,
    get_character_test_image
)

# 使用缓存版本的脚本（支持缓存机制）
NERF_SCRIPT_CACHE = NERF_WORKDIR / "StyleNeRF" / "main_NeRFFaceSpeech_audio_driven_w_given_poses_cache.py"

# 使用logging模块添加日志（避免循环导入）
import logging
def add_log(message, level="info"):
    """添加日志到日志系统"""
    logger = logging.getLogger()
    level_map = {
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "success": logging.INFO,  # success级别使用INFO，但在消息前添加标记以便前端识别
        "progress": logging.INFO,
        "debug": logging.DEBUG
    }
    log_level = level_map.get(level, logging.INFO)
    # 对于success级别，在消息前添加特殊标记，BufferLogHandler会识别并正确处理
    if level == "success":
        logger.log(log_level, f"[SUCCESS] {message}")
    elif level == "progress":
        logger.log(log_level, f"[PROGRESS] {message}")
    else:
        logger.log(log_level, message)

def generate_video(
    audio_path: str,
    character: str,
    output_path: str,
    model_name: str
) -> bool:
    """生成视频文件"""
    try:
        test_img = get_character_test_image(character)
    except ValueError as e:
        error_msg = f"错误: {e}"
        print(error_msg)
        add_log(error_msg, "error")
        return False

    # ---------- 模型路径安全拼接 ----------
    network_path = MODEL_DIR / model_name
    if not model_name.endswith(".pkl") or not network_path.exists():
        error_msg = f"非法或不存在的模型：{network_path}"
        print(error_msg)
        add_log(error_msg, "error")
        return False

    env = os.environ.copy()
    # 与可运行版本一致：使用 PATH 和 PYTHONPATH
    env["PATH"] = f"{NERF_CONDA_ENV / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(NERF_WORKDIR)
    
    # 确保关键环境变量被设置（从父进程继承或使用默认值）
    # 这些环境变量对于模型下载非常重要，必须在子进程中明确设置
    if "TORCH_HOME" not in env:
        env["TORCH_HOME"] = "/root/autodl-tmp/weights"
        add_log(f"[NeRF] 设置 TORCH_HOME: {env['TORCH_HOME']}", "info")
    else:
        add_log(f"[NeRF] 使用继承的 TORCH_HOME: {env['TORCH_HOME']}", "info")
    
    if "HF_ENDPOINT" not in env:
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        add_log(f"[NeRF] 设置 HF_ENDPOINT: {env['HF_ENDPOINT']}", "info")
    
    if "HF_HOME" not in env:
        env["HF_HOME"] = "/root/autodl-tmp/Hugging_Face"
        add_log(f"[NeRF] 设置 HF_HOME: {env['HF_HOME']}", "info")
    
    if "PIP_INDEX_URL" not in env:
        env["PIP_INDEX_URL"] = "https://pypi.tuna.tsinghua.edu.cn/simple"
        add_log(f"[NeRF] 设置 PIP_INDEX_URL: {env['PIP_INDEX_URL']}", "info")
    
    # 设置模型缓存目录，避免重复下载
    # face_alignment 和 torch.hub 通常使用 TORCH_HOME 环境变量
    torch_home = env["TORCH_HOME"]
    torch_hub_dir = Path(torch_home) / "hub" / "checkpoints"
    
    # 检查是否已有3DFAN4模型文件
    fan4_file = torch_hub_dir / "3DFAN4-4a694010b9.zip"
    fan4_file_alt = Path.home() / ".face_alignment" / "3DFAN4-4a694010b9.zip"
    
    if fan4_file.exists():
        add_log(f"[NeRF] ✅ 检测到本地3DFAN4模型文件: {fan4_file}", "success")
        add_log(f"[NeRF] 将使用本地缓存，无需下载", "info")
    elif fan4_file_alt.exists():
        add_log(f"[NeRF] ✅ 检测到本地3DFAN4模型文件: {fan4_file_alt}", "success")
    else:
        add_log(f"[NeRF] ⚠️ 未检测到3DFAN4模型文件，将尝试自动下载", "warning")
        add_log(f"[NeRF] 下载地址: https://www.adrianbulat.com/downloads/python-fan/3DFAN4-4a694010b9.zip", "warning")
        add_log(f"[NeRF] 如果网络较慢，建议手动下载到: {torch_hub_dir}", "warning")
    
    if not torch_hub_dir.exists():
        try:
            torch_hub_dir.mkdir(parents=True, exist_ok=True)
            add_log(f"[NeRF] 创建模型缓存目录: {torch_hub_dir}", "info")
        except Exception as e:
            add_log(f"[NeRF] 无法创建模型缓存目录 {torch_hub_dir}: {e}", "warning")
    else:
        add_log(f"[NeRF] 使用模型缓存目录: {torch_hub_dir}", "info")

    # ---------- 构造缓存目录路径（PTI和3DMM缓存） ----------
    # 缓存目录：项目根目录/assets/charactor/{角色名}
    cache_dir = PROJECT_ROOT / "assets" / "charactor" / character
    cache_dir = cache_dir.resolve()  # 转换为绝对路径
    cache_dir.mkdir(parents=True, exist_ok=True)
    add_log(f"[NeRF] 使用缓存目录: {cache_dir}", "info")
    add_log(f"[NeRF] 缓存目录将存储PTI和3DMM拟合结果，可显著加速后续推理", "info")

    # 使用缓存版本的脚本
    script_path = NERF_SCRIPT_CACHE if NERF_SCRIPT_CACHE.exists() else NERF_SCRIPT
    if NERF_SCRIPT_CACHE.exists():
        add_log(f"[NeRF] 使用缓存版本脚本: {script_path}", "info")
    else:
        add_log(f"[NeRF] 警告: 缓存版本脚本不存在，使用默认脚本: {script_path}", "warning")
        add_log(f"[NeRF] 缓存功能可能不可用", "warning")

    cmd = [
        str(NERF_CONDA_PYTHON),
        str(script_path),
        f"--outdir={output_path}",
        "--trunc=0.7",
        f"--network={network_path}",
        f"--test_data={audio_path}",
        f"--test_img={test_img}",
        "--motion_guide_img_folder=frames",
        f"--cache_dir={cache_dir}"  # 添加缓存目录参数
    ]

    try:
        # 捕获标准输出和标准错误，实时添加到日志
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',  # 使用replace模式处理非UTF-8字节，避免解码错误
            bufsize=1,
            env=env,
            cwd=str(NERF_WORKDIR),
            universal_newlines=True
        )
        
        # 实时读取输出并添加到日志
        error_lines = []  # 收集错误信息
        for line in process.stdout:
            if line:
                line = line.rstrip()
                # 根据内容判断日志级别
                if 'ERROR' in line or '错误' in line or 'Error' in line or 'Exception' in line or 'Traceback' in line or 'TimeoutError' in line or 'URLError' in line or 'Connection timed out' in line:
                    add_log(line, "error")
                    error_lines.append(line)
                elif 'WARNING' in line or '警告' in line or 'Warning' in line or 'WARN' in line or 'DeprecationWarning' in line:
                    add_log(line, "warning")
                elif '%|' in line or 'it/s' in line or 'Sampling' in line:
                    # NeRF采样进度条信息（注意：这只是采样阶段，不是整个流程）
                    add_log(line, "progress")
                elif 'Downloading' in line or 'downloading' in line:
                    # 下载信息（重要提示）
                    add_log(f"[下载中] {line}", "info")
                    # 如果是下载3DFAN4模型，说明NeRF采样已完成，现在在下载后续步骤需要的模型
                    if '3DFAN4' in line or 'face_alignment' in line or 'adrianbulat' in line:
                        add_log("[重要提示] ⚠️ NeRF采样已完成（100%），现在进行后续步骤", "warning")
                        add_log("[重要提示] 正在下载 face_alignment 模型文件，这是正常流程的一部分", "warning")
                        add_log("[重要提示] 如果网络较慢，下载可能需要较长时间，请耐心等待...", "warning")
                elif 'Loading' in line and ('model' in line.lower() or 'network' in line.lower()):
                    add_log(f"[加载中] {line}", "info")
                elif line.strip():  # 忽略空行
                    add_log(line, "info")
        
        process.wait()
        
        if process.returncode != 0:
            error_msg = f"NeRF 视频生成失败，返回码: {process.returncode}"
            print(error_msg)
            add_log(error_msg, "error")
            
            # 检查是否是网络超时错误
            error_text = '\n'.join(error_lines)
            if 'TimeoutError' in error_text or 'Connection timed out' in error_text or 'URLError' in error_text:
                add_log("=" * 60, "error")
                add_log("[网络错误] 检测到网络连接超时，模型无法下载预训练权重文件", "error")
                add_log("[网络错误] 这通常是因为网络速度慢或无法访问外部下载源", "error")
                add_log("[网络错误] 建议解决方案：", "error")
                add_log("[网络错误] 1. 检查网络连接是否正常", "error")
                add_log("[网络错误] 2. 尝试手动下载文件到本地缓存目录", "error")
                add_log("[网络错误] 3. 文件应下载到: /root/autodl-tmp/weights/hub/checkpoints/", "error")
                add_log("[网络错误] 4. 或设置代理/使用学术加速", "error")
                add_log("=" * 60, "error")
            
            return False
        
        return True
    except Exception as e:
        error_msg = f"NeRF 视频生成错误：{e}"
        print(error_msg)
        add_log(error_msg, "error")
        return False
