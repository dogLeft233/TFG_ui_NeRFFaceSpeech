#!/usr/bin/env python3
"""
配置模块测试程序
测试迁移后的配置模块是否正常工作
"""
import sys
from pathlib import Path

# 添加gradio_app到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from shared.config import (
    PROJECT_ROOT,
    NERF_CODE_DIR,
    OUTPUT_VIDEO_DIR,
    OUTPUT_AUDIO_DIR,
    MODEL_DIR,
    WEBUI_DIR,
    DATABASE_DIR,
    VIDEOS_STORAGE_DIR,
    AUDIOS_STORAGE_DIR,
    TEXTS_STORAGE_DIR,
    DATA_DIR,
    TRAINING_DATASET_DIR,
    API_CONDA_ENV,
    API_CONDA_PYTHON,
    LLM_CONDA_ENV,
    LLM_CONDA_PYTHON,
    NERF_CONDA_ENV,
    NERF_CONDA_PYTHON,
    LLM_TALK_SCRIPT,
    NERF_SCRIPT,
    CHARACTER_AUDIO_PROMPTS,
    CHARACTER_TEST_IMAGES,
    get_character_audio_prompt,
    get_character_test_image,
    ensure_dirs
)

def test_paths():
    """测试路径配置"""
    print("=" * 60)
    print("测试路径配置")
    print("=" * 60)
    
    paths = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "NERF_CODE_DIR": NERF_CODE_DIR,
        "OUTPUT_VIDEO_DIR": OUTPUT_VIDEO_DIR,
        "OUTPUT_AUDIO_DIR": OUTPUT_AUDIO_DIR,
        "MODEL_DIR": MODEL_DIR,
        "WEBUI_DIR": WEBUI_DIR,
        "DATABASE_DIR": DATABASE_DIR,
        "VIDEOS_STORAGE_DIR": VIDEOS_STORAGE_DIR,
        "AUDIOS_STORAGE_DIR": AUDIOS_STORAGE_DIR,
        "TEXTS_STORAGE_DIR": TEXTS_STORAGE_DIR,
        "DATA_DIR": DATA_DIR,
        "TRAINING_DATASET_DIR": TRAINING_DATASET_DIR,
    }
    
    for name, path in paths.items():
        exists = path.exists()
        status = "✅" if exists else "⚠️"
        print(f"{status} {name}: {path}")
        if not exists:
            print(f"    (路径不存在，但这是正常的，某些目录可能尚未创建)")
    
    print()

def test_conda_envs():
    """测试Conda环境配置"""
    print("=" * 60)
    print("测试Conda环境配置")
    print("=" * 60)
    
    envs = {
        "API_CONDA_ENV": API_CONDA_ENV,
        "API_CONDA_PYTHON": API_CONDA_PYTHON,
        "LLM_CONDA_ENV": LLM_CONDA_ENV,
        "LLM_CONDA_PYTHON": LLM_CONDA_PYTHON,
        "NERF_CONDA_ENV": NERF_CONDA_ENV,
        "NERF_CONDA_PYTHON": NERF_CONDA_PYTHON,
    }
    
    for name, path in envs.items():
        exists = path.exists()
        status = "✅" if exists else "⚠️"
        print(f"{status} {name}: {path}")
        if not exists:
            print(f"    (环境不存在，但这是正常的，某些环境可能尚未创建)")
    
    print()

def test_scripts():
    """测试脚本路径配置"""
    print("=" * 60)
    print("测试脚本路径配置")
    print("=" * 60)
    
    scripts = {
        "LLM_TALK_SCRIPT": LLM_TALK_SCRIPT,
        "NERF_SCRIPT": NERF_SCRIPT,
    }
    
    for name, path in scripts.items():
        exists = path.exists()
        status = "✅" if exists else "⚠️"
        print(f"{status} {name}: {path}")
        if not exists:
            print(f"    (脚本不存在，但这是正常的，某些脚本可能尚未创建)")
    
    print()

def test_character_resources():
    """测试角色资源配置"""
    print("=" * 60)
    print("测试角色资源配置")
    print("=" * 60)
    
    print("角色音频提示文件:")
    for character, path in CHARACTER_AUDIO_PROMPTS.items():
        exists = path.exists()
        status = "✅" if exists else "⚠️"
        print(f"  {status} {character}: {path}")
        if not exists:
            print(f"      (文件不存在，但这是正常的，某些资源可能尚未创建)")
    
    print("\n角色测试图片:")
    for character, path in CHARACTER_TEST_IMAGES.items():
        exists = path.exists()
        status = "✅" if exists else "⚠️"
        print(f"  {status} {character}: {path}")
        if not exists:
            print(f"      (文件不存在，但这是正常的，某些资源可能尚未创建)")
    
    print()

def test_helper_functions():
    """测试辅助函数"""
    print("=" * 60)
    print("测试辅助函数")
    print("=" * 60)
    
    # 测试 ensure_dirs
    print("1. 测试 ensure_dirs()...")
    try:
        ensure_dirs()
        print("   ✅ ensure_dirs() 执行成功")
    except Exception as e:
        print(f"   ❌ ensure_dirs() 执行失败: {e}")
    
    # 测试 get_character_audio_prompt
    print("\n2. 测试 get_character_audio_prompt()...")
    for character in ["ayanami", "Aerith"]:
        try:
            path = get_character_audio_prompt(character)
            print(f"   ✅ {character}: {path}")
        except ValueError as e:
            print(f"   ⚠️  {character}: {e} (这是正常的，如果角色不存在)")
        except Exception as e:
            print(f"   ❌ {character}: 错误 - {e}")
    
    # 测试 get_character_test_image
    print("\n3. 测试 get_character_test_image()...")
    for character in ["ayanami", "Aerith"]:
        try:
            path = get_character_test_image(character)
            print(f"   ✅ {character}: {path}")
        except ValueError as e:
            print(f"   ⚠️  {character}: {e} (这是正常的，如果角色不存在)")
        except Exception as e:
            print(f"   ❌ {character}: 错误 - {e}")
    
    # 测试无效角色
    print("\n4. 测试无效角色（应该抛出异常）...")
    try:
        get_character_audio_prompt("invalid_character")
        print("   ❌ 应该抛出异常但没有抛出")
    except ValueError as e:
        print(f"   ✅ 正确抛出异常: {e}")
    except Exception as e:
        print(f"   ⚠️  抛出异常但类型不对: {e}")
    
    print()

def test_directory_creation():
    """测试目录创建"""
    print("=" * 60)
    print("测试目录创建")
    print("=" * 60)
    
    directories = [
        DATABASE_DIR,
        VIDEOS_STORAGE_DIR,
        AUDIOS_STORAGE_DIR,
        TEXTS_STORAGE_DIR,
        OUTPUT_VIDEO_DIR,
        OUTPUT_AUDIO_DIR,
    ]
    
    for directory in directories:
        exists = directory.exists()
        is_dir = directory.is_dir() if exists else False
        status = "✅" if (exists and is_dir) else "⚠️"
        print(f"{status} {directory.name}: {directory}")
        if not exists:
            print(f"    (目录不存在，ensure_dirs()应该会创建它)")
        elif not is_dir:
            print(f"    (路径存在但不是目录)")
    
    print()

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("配置模块测试程序")
    print("=" * 60 + "\n")
    
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"当前工作目录: {Path.cwd()}")
    print()
    
    # 运行所有测试
    test_paths()
    test_conda_envs()
    test_scripts()
    test_character_resources()
    test_helper_functions()
    test_directory_creation()
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n说明:")
    print("- ✅ 表示路径存在或功能正常")
    print("- ⚠️  表示路径不存在，但这是正常的（某些目录/文件可能尚未创建）")
    print("- ❌ 表示出现错误，需要检查")
    print()

if __name__ == "__main__":
    main()

