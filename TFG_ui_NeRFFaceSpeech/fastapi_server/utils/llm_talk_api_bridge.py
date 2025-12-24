#!/usr/bin/env python
"""
llm_talk API 桥接脚本
通过 subprocess 调用，用于在独立环境中运行 llm_talk API
"""
import sys
import os
import json
import argparse

# 添加项目根目录到路径
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# llm_talk 在项目根目录下
llm_talk_path = PROJECT_ROOT / "llm_talk"
if llm_talk_path.exists() and (llm_talk_path / "__init__.py").exists():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
else:
    # 如果没找到，至少添加项目根目录
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT)

try:
    from llm_talk import get_talk_response_api, get_llm_response_api
except ImportError as e:
    result = {
        "success": False,
        "error": {
            "code": "IMPORT_ERROR",
            "message": f"无法导入 llm_talk 模块: {str(e)}",
            "type": "ImportError"
        },
        "data": None
    }
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="llm_talk API Bridge")
    parser.add_argument("--mode", choices=["talk", "llm_only"], required=True, help="API模式")
    parser.add_argument("--user_input", required=True, help="用户输入")
    parser.add_argument("--character", default="ayanami", help="角色名称")
    parser.add_argument("--enable_audio", type=bool, default=True, help="是否生成音频")
    parser.add_argument("--language_id", default="zh", help="语言ID")
    parser.add_argument("--audio_prompt_path", default=None, help="音频提示路径")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "talk":
            kwargs = {
                "user_input": args.user_input,
                "language_id": args.language_id,
                "combine_audio": args.enable_audio,
                "release_tts_model": False,
                "split_sentences": True,
            }
            if args.audio_prompt_path:
                kwargs["audio_prompt_path"] = args.audio_prompt_path
            
            result = get_talk_response_api(**kwargs)
        else:  # llm_only
            result = get_llm_response_api(args.user_input)
        
        # 递归检查并转换bytes为字符串（防止JSON序列化错误）
        def ensure_json_serializable(obj):
            """递归检查并转换不可JSON序列化的对象为字符串"""
            if isinstance(obj, bytes):
                return obj.decode('utf-8')
            elif isinstance(obj, dict):
                return {k: ensure_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [ensure_json_serializable(item) for item in obj]
            elif isinstance(obj, (set, tuple)):
                return [ensure_json_serializable(item) for item in obj]
            else:
                # 尝试直接返回，如果无法序列化会在json.dumps时抛出异常
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # 确保结果可以JSON序列化
        result = ensure_json_serializable(result)
        
        # 输出 JSON 结果
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as e:
        error_result = {
            "success": False,
            "error": {
                "code": "EXECUTION_ERROR",
                "message": str(e),
                "type": type(e).__name__
            },
            "data": None
        }
        print(json.dumps(error_result, ensure_ascii=False))
        return 1

if __name__ == "__main__":
    sys.exit(main())

