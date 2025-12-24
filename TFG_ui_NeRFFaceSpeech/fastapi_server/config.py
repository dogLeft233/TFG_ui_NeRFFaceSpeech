"""
配置管理模块
使用相对路径，基于项目根目录
"""
import os
from pathlib import Path

# 获取项目根目录（fastapi_server 的父目录）
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

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
WEBUI_DIR = Path(__file__).parent / "webui"

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

# ==================== 资源路径配置 ====================

# 角色音频提示文件
CHARACTER_AUDIO_PROMPTS = {
    "ayanami": PROJECT_ROOT / "assets" / "charactors" / "Ayanami" / "绫波丽.wav",
    "Aerith": PROJECT_ROOT / "assets" / "charactors" / "Aerith" / "Aerith.mp3",
}

# 角色测试图片
CHARACTER_TEST_IMAGES = {
    "ayanami": PROJECT_ROOT / "assets" / "charactors" / "Ayanami" / "ayanami.png",
    "Aerith": PROJECT_ROOT / "assets" / "charactors" / "Aerith" / "Aerith.jpg",
}

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
    prompt = CHARACTER_AUDIO_PROMPTS.get(character)
    if prompt is None:
        raise ValueError(f"未知角色: {character}")
    return prompt

def get_character_test_image(character: str) -> Path:
    """获取角色的测试图片路径"""
    img = CHARACTER_TEST_IMAGES.get(character)
    if img is None:
        raise ValueError(f"未知角色: {character}")
    return img

# 启动时确保目录存在
ensure_dirs()
