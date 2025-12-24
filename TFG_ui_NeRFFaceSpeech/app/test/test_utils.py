#!/usr/bin/env python3
"""
工具模块测试程序
测试 utils 模块的导入和基本功能
"""
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_imports():
    """测试模块导入"""
    print("\n" + "="*60)
    print("测试模块导入")
    print("="*60)
    
    results = []
    
    # 测试 run_llm_talk
    try:
        from shared.utils import run_llm_talk
        print("   ✓ run_llm_talk 导入成功")
        results.append(("run_llm_talk", True))
    except Exception as e:
        print(f"   ✗ run_llm_talk 导入失败: {e}")
        results.append(("run_llm_talk", False))
    
    # 测试 run_nerffacespeech
    try:
        from shared.utils import run_nerffacespeech
        print("   ✓ run_nerffacespeech 导入成功")
        results.append(("run_nerffacespeech", True))
    except Exception as e:
        print(f"   ✗ run_nerffacespeech 导入失败: {e}")
        results.append(("run_nerffacespeech", False))
    
    # 测试 run_chat
    try:
        from shared.utils import run_chat
        print("   ✓ run_chat 导入成功")
        results.append(("run_chat", True))
    except Exception as e:
        print(f"   ✗ run_chat 导入失败: {e}")
        results.append(("run_chat", False))
    
    # 测试 run_training
    try:
        from shared.utils import run_training
        print("   ✓ run_training 导入成功")
        results.append(("run_training", True))
    except Exception as e:
        print(f"   ✗ run_training 导入失败: {e}")
        results.append(("run_training", False))
    
    return results


def test_config_paths():
    """测试配置路径是否正确"""
    print("\n" + "="*60)
    print("测试配置路径")
    print("="*60)
    
    try:
        from shared.config import PROJECT_ROOT, LLM_CONDA_PYTHON, NERF_CONDA_PYTHON
        
        print(f"   PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"   ✓ PROJECT_ROOT 存在: {PROJECT_ROOT.exists()}")
        
        print(f"   LLM_CONDA_PYTHON: {LLM_CONDA_PYTHON}")
        print(f"   ✓ LLM_CONDA_PYTHON 存在: {LLM_CONDA_PYTHON.exists()}")
        
        print(f"   NERF_CONDA_PYTHON: {NERF_CONDA_PYTHON}")
        print(f"   ✓ NERF_CONDA_PYTHON 存在: {NERF_CONDA_PYTHON.exists()}")
        
        return True
    except Exception as e:
        print(f"   ✗ 配置路径测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_function_existence():
    """测试函数是否存在"""
    print("\n" + "="*60)
    print("测试函数存在性")
    print("="*60)
    
    results = []
    
    # 测试 run_llm_talk 的函数
    try:
        from shared.utils import run_llm_talk
        assert hasattr(run_llm_talk, 'generate_audio'), "generate_audio 函数不存在"
        print("   ✓ run_llm_talk.generate_audio 存在")
        results.append(("run_llm_talk.generate_audio", True))
    except Exception as e:
        print(f"   ✗ run_llm_talk.generate_audio 测试失败: {e}")
        results.append(("run_llm_talk.generate_audio", False))
    
    # 测试 run_nerffacespeech 的函数
    try:
        from shared.utils import run_nerffacespeech
        assert hasattr(run_nerffacespeech, 'generate_video'), "generate_video 函数不存在"
        print("   ✓ run_nerffacespeech.generate_video 存在")
        results.append(("run_nerffacespeech.generate_video", True))
    except Exception as e:
        print(f"   ✗ run_nerffacespeech.generate_video 测试失败: {e}")
        results.append(("run_nerffacespeech.generate_video", False))
    
    # 测试 run_chat 的函数
    try:
        from shared.utils import run_chat
        assert hasattr(run_chat, 'chat_with_llm'), "chat_with_llm 函数不存在"
        assert hasattr(run_chat, 'get_llm_only'), "get_llm_only 函数不存在"
        print("   ✓ run_chat.chat_with_llm 存在")
        print("   ✓ run_chat.get_llm_only 存在")
        results.append(("run_chat.chat_with_llm", True))
        results.append(("run_chat.get_llm_only", True))
    except Exception as e:
        print(f"   ✗ run_chat 函数测试失败: {e}")
        results.append(("run_chat", False))
    
    # 测试 run_training 的函数
    try:
        from shared.utils import run_training
        assert hasattr(run_training, 'start_training'), "start_training 函数不存在"
        assert hasattr(run_training, 'get_training_status'), "get_training_status 函数不存在"
        assert hasattr(run_training, 'list_training_tasks'), "list_training_tasks 函数不存在"
        print("   ✓ run_training.start_training 存在")
        print("   ✓ run_training.get_training_status 存在")
        print("   ✓ run_training.list_training_tasks 存在")
        results.append(("run_training.start_training", True))
        results.append(("run_training.get_training_status", True))
        results.append(("run_training.list_training_tasks", True))
    except Exception as e:
        print(f"   ✗ run_training 函数测试失败: {e}")
        results.append(("run_training", False))
    
    return results


def test_bridge_scripts():
    """测试桥接脚本路径"""
    print("\n" + "="*60)
    print("测试桥接脚本路径")
    print("="*60)
    
    try:
        from pathlib import Path
        from shared.config import PROJECT_ROOT
        
        utils_dir = Path(__file__).parent / "shared" / "utils"
        
        bridge_scripts = [
            "llm_talk_api_bridge.py",
            "llm_talk_with_text_bridge.py"
        ]
        
        all_exist = True
        for script_name in bridge_scripts:
            script_path = utils_dir / script_name
            exists = script_path.exists()
            print(f"   {script_name}: {'✓ 存在' if exists else '✗ 不存在'}")
            if not exists:
                all_exist = False
        
        return all_exist
    except Exception as e:
        print(f"   ✗ 桥接脚本路径测试失败: {e}")
        return False


def test_project_root_calculation():
    """测试项目根目录计算"""
    print("\n" + "="*60)
    print("测试项目根目录计算")
    print("="*60)
    
    try:
        from pathlib import Path
        from shared.config import PROJECT_ROOT
        
        # 计算期望的项目根目录
        current_file = Path(__file__)
        expected_root = current_file.parent.parent  # gradio_app -> 项目根目录
        
        print(f"   当前文件: {current_file}")
        print(f"   期望根目录: {expected_root}")
        print(f"   配置根目录: {PROJECT_ROOT}")
        print(f"   ✓ 路径匹配: {PROJECT_ROOT.resolve() == expected_root.resolve()}")
        
        return PROJECT_ROOT.resolve() == expected_root.resolve()
    except Exception as e:
        print(f"   ✗ 项目根目录计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("工具模块测试程序")
    print("="*60)
    
    all_results = []
    
    # 测试模块导入
    import_results = test_imports()
    all_results.extend(import_results)
    
    # 测试配置路径
    config_ok = test_config_paths()
    all_results.append(("config_paths", config_ok))
    
    # 测试函数存在性
    function_results = test_function_existence()
    all_results.extend(function_results)
    
    # 测试桥接脚本
    bridge_ok = test_bridge_scripts()
    all_results.append(("bridge_scripts", bridge_ok))
    
    # 测试项目根目录计算
    root_ok = test_project_root_calculation()
    all_results.append(("project_root", root_ok))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for name, result in all_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:40s} : {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 个通过, {failed} 个失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
        print("注意：某些测试失败可能是因为缺少外部依赖（conda环境、模型文件等）")
        return 1


if __name__ == "__main__":
    sys.exit(main())

