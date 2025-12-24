#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型预下载脚本（使用系统默认缓存目录）
"""

import sys
import traceback

print("=" * 60)
print("模型预下载脚本（使用系统默认缓存目录）")
print("=" * 60)
print()

def download_huggingface_models():
    print("[core] 下载 HuggingFace 模型: ResembleAI/chatterbox...")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="ResembleAI/chatterbox",
        repo_type="model",
        local_dir_use_symlinks=False
    )

    print("✓ HuggingFace chatterbox 下载完成\n")


def download_whisper_models():
    print("[core] 下载 Whisper 模型 (base)...")
    import whisper

    whisper.load_model("base")

    print("✓ Whisper base 下载完成\n")

# ========================
# 主入口
# ========================
def main():

    try:
        download_whisper_models()
        download_huggingface_models()

    except Exception:
        print("\n❌ 模型下载过程中发生错误：")
        traceback.print_exc()
        sys.exit(0)

    print("=" * 60)
    print("🎉 模型下载完成！")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()