"""
配置管理模块
使用相对路径，基于项目根目录
从 fastapi_server/config.py 迁移而来
"""
import os
from pathlib import Path

# 获取项目根目录（gradio_app 的父目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# ==================== 路径配置 ====================

# NeRFFaceSpeech 代码目录
NERF_CODE_DIR = PROJECT_ROOT / "NeRFFaceSpeech_Code"

# 输出目录
OUTPUT_VIDEO_DIR = NERF_CODE_DIR / "outputs" / "video"
OUTPUT_AUDIO_DIR = NERF_CODE_DIR / "outputs" / "audio"

# 模型目录
# 优先使用相对路径（基于项目根目录）
relative_model_dir = NERF_CODE_DIR / "pretrained_networks"
MODEL_DIR = relative_model_dir

# 如果相对路径不存在，尝试使用服务器上的绝对路径（需要权限）
if not MODEL_DIR.exists():
    try:
        absolute_model_dir = Path("/root/autodl-tmp/TFG_TALK_NeRFaceSpeech/NeRFFaceSpeech_Code/pretrained_networks")
        if absolute_model_dir.exists():
            MODEL_DIR = absolute_model_dir
    except (PermissionError, OSError):
        # 如果没有权限访问绝对路径，继续使用相对路径
        pass

# WebUI 目录（前端静态文件目录）
WEBUI_DIR = Path(__file__).parent.parent / "webui"

# 数据库目录（在项目根目录下，与fastapi_server平级）
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# 视频存储目录（在数据库目录下）
VIDEOS_STORAGE_DIR = DATABASE_DIR / "videos"
VIDEOS_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# 音频存储目录（在数据库目录下）
AUDIOS_STORAGE_DIR = DATABASE_DIR / "audios"
AUDIOS_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# 文本存储目录（在数据库目录下）
TEXTS_STORAGE_DIR = DATABASE_DIR / "texts"
TEXTS_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# 训练数据集目录
# 优先使用相对路径（基于项目根目录）
DATA_DIR = PROJECT_ROOT / "data"

# 如果相对路径不存在，尝试使用服务器上的绝对路径（需要权限）
if not DATA_DIR.exists():
    try:
        absolute_data_dir = Path("/root/autodl-tmp/TFG_TALK_NeRFaceSpeech/data")
        if absolute_data_dir.exists():
            DATA_DIR = absolute_data_dir
    except (PermissionError, OSError):
        # 如果没有权限访问绝对路径，继续使用相对路径
        pass

# 训练数据集默认路径（单张图像数据集目录，用于StyleNeRF训练）
# 可以根据实际情况修改
TRAINING_DATASET_DIR = DATA_DIR / "geneface_datasets" / "data" / "raw"
# 如果默认路径不存在，使用data目录作为后备
if not TRAINING_DATASET_DIR.exists():
    TRAINING_DATASET_DIR = DATA_DIR

# ==================== Conda 环境配置 ====================

# API 环境（用于运行 FastAPI 服务器）
API_CONDA_ENV = PROJECT_ROOT / "environment" / "api"
API_CONDA_PYTHON = API_CONDA_ENV / "bin" / "python"

# LLM Talk 环境（在 environment 文件夹中）
LLM_CONDA_ENV = PROJECT_ROOT / "environment" / "llm_talk"
LLM_CONDA_PYTHON = LLM_CONDA_ENV / "bin" / "python"

# NeRF 环境（在 environment 文件夹中）
NERF_CONDA_ENV = PROJECT_ROOT / "environment" / "nerffacespeech"
NERF_CONDA_PYTHON = NERF_CONDA_ENV / "bin" / "python"

# ==================== 脚本路径配置 ====================

# LLM Talk 脚本
LLM_TALK_SCRIPT = PROJECT_ROOT / "llm_talk" / "talk.py"

# NeRF 脚本
NERF_SCRIPT = NERF_CODE_DIR / "StyleNeRF" / "main_NeRFFaceSpeech_audio_driven_w_given_poses.py"

# TTS 服务脚本
TTS_SERVICE_SCRIPT = PROJECT_ROOT / "gradio_app" / "services" / "tts_service.py"

# ASR 服务脚本
ASR_SERVICE_SCRIPT = PROJECT_ROOT / "gradio_app" / "services" / "asr_service.py"

# ==================== 服务地址配置 ====================

# TTS 服务地址（可通过环境变量覆盖）
TTS_SERVICE_URL = os.environ.get("TTS_SERVICE_URL", "http://localhost:8001")

# ASR 服务地址（可通过环境变量覆盖）
ASR_SERVICE_URL = os.environ.get("ASR_SERVICE_URL", "http://localhost:8002")

# ==================== 资源路径配置 ====================

# 角色音频提示文件（硬编码的默认角色）
CHARACTER_AUDIO_PROMPTS = {
    "ayanami": PROJECT_ROOT / "assets" / "charactors" / "Ayanami" / "绫波丽.wav",
    "Aerith": PROJECT_ROOT / "assets" / "charactors" / "Aerith" / "Aerith.mp3",
}

# 角色测试图片（硬编码的默认角色）
CHARACTER_TEST_IMAGES = {
    "ayanami": PROJECT_ROOT / "assets" / "charactors" / "Ayanami" / "ayanami.png",
    "Aerith": PROJECT_ROOT / "assets" / "charactors" / "Aerith" / "Aerith.jpg",
}

# 角色训练数据目录（动态加载的角色）
CHARACTER_DIR = PROJECT_ROOT / "assets" / "charactor"

# ==================== 辅助函数 ====================

def ensure_dirs():
    """确保必要的目录存在"""
    OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    # 如果 MODEL_DIR 不存在且是相对路径，才创建目录（绝对路径可能不需要创建）
    if not MODEL_DIR.exists() and not MODEL_DIR.is_absolute():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # 确保文本存储目录存在
    TEXTS_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

def get_character_audio_prompt(character: str) -> Path:
    """获取角色的音频提示文件路径"""
    # 首先检查硬编码的角色（向后兼容）
    prompt = CHARACTER_AUDIO_PROMPTS.get(character)
    if prompt is not None and prompt.exists():
        return prompt
    
    # 如果不在硬编码列表中，尝试从角色训练目录查找
    character_dir = CHARACTER_DIR / character
    if character_dir.exists():
        # 查找音频文件（优先查找 audio.wav）
        audio_file = character_dir / "audio.wav"
        if audio_file.exists():
            return audio_file
        
        # 如果 audio.wav 不存在，查找其他音频文件
        for audio_ext in [".mp3", ".wav", ".m4a", ".ogg"]:
            audio_file = character_dir / f"audio{audio_ext}"
            if audio_file.exists():
                return audio_file
    
    # 如果都找不到，抛出错误
    raise ValueError(f"未知角色: {character}（未找到音频文件）")

def get_character_test_image(character: str) -> Path:
    """获取角色的测试图片路径"""
    # 首先检查硬编码的角色（向后兼容）
    img = CHARACTER_TEST_IMAGES.get(character)
    if img is not None and img.exists():
        return img
    
    # 如果不在硬编码列表中，尝试从角色训练目录查找
    character_dir = CHARACTER_DIR / character
    if character_dir.exists():
        images_dir = character_dir / "images"
        if images_dir.exists():
            # 查找第一张图片（按文件名排序）
            image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
            if image_files:
                return image_files[0]
        
        # 如果 images 目录不存在，尝试在角色目录下直接查找图片
        image_files = sorted(character_dir.glob("*.jpg")) + sorted(character_dir.glob("*.png"))
        if image_files:
            return image_files[0]
    
    # 如果都找不到，抛出错误
    raise ValueError(f"未知角色: {character}（未找到测试图片）")

def get_character_list() -> list:
    """获取可用角色列表（包括硬编码和动态加载的角色）"""
    characters = list(CHARACTER_AUDIO_PROMPTS.keys())
    
    # 添加动态加载的角色（从角色训练目录）
    if CHARACTER_DIR.exists():
        for char_dir in CHARACTER_DIR.iterdir():
            if char_dir.is_dir():
                char_name = char_dir.name
                # 检查是否有必要的文件（图片和音频）
                images_dir = char_dir / "images"
                audio_file = char_dir / "audio.wav"
                if images_dir.exists() and audio_file.exists():
                    # 只添加不在硬编码列表中的角色
                    if char_name not in characters:
                        characters.append(char_name)
    
    return characters

# 启动时确保目录存在
ensure_dirs()

