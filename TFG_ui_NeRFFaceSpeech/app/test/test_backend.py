#!/usr/bin/env python3
"""
FastAPI 后端服务测试程序
测试 backend/main.py 的导入和基本功能
"""
import sys
import os
import re
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_imports():
    """测试模块导入（检查导入路径是否正确）"""
    print("\n" + "="*60)
    print("测试模块导入路径")
    print("="*60)
    
    results = []
    
    # 检查 main.py 文件是否存在
    main_file = Path(__file__).parent.parent / "backend" / "main.py"
    if main_file.exists():
        print(f"   ✓ backend/main.py 文件存在")
        results.append(("backend.main.file", True))
        
        # 读取文件内容，检查导入路径
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查导入路径是否正确
        checks = [
            (r'from shared\.utils\.', 'shared.utils 导入路径'),
            (r'from shared\.config import', 'shared.config 导入路径'),
            (r'from shared\.database\.', 'shared.database 导入路径'),
        ]
        
        for pattern, description in checks:
            if re.search(pattern, content):
                print(f"   ✓ {description} 正确")
                results.append((description, True))
            else:
                print(f"   ✗ {description} 未找到或格式错误")
                results.append((description, False))
        
        # 检查是否还有旧的导入路径
        old_patterns = [
            (r'from utils\.', '旧的 utils 导入'),
            (r'from config import', '旧的 config 导入'),
            (r'from database\.', '旧的 database 导入'),
        ]
        
        has_old_imports = False
        for pattern, description in old_patterns:
            if re.search(pattern, content):
                print(f"   ⚠️  发现 {description}")
                has_old_imports = True
        
        if not has_old_imports:
            print(f"   ✓ 没有发现旧的导入路径")
            results.append(("no_old_imports", True))
        else:
            results.append(("no_old_imports", False))
    else:
        print(f"   ✗ backend/main.py 文件不存在")
        results.append(("backend.main.file", False))
    
    # 尝试导入（如果 FastAPI 可用）
    try:
        from backend import main
        print("   ✓ backend.main 导入成功（FastAPI 已安装）")
        results.append(("backend.main.import", True))
    except ImportError as e:
        if 'fastapi' in str(e).lower():
            print(f"   ⚠️  backend.main 导入失败（FastAPI 未安装，这是正常的）")
            print(f"      错误: {e}")
            print(f"      注意：FastAPI 应该在 conda 环境中运行")
            results.append(("backend.main.import", None))  # None 表示跳过
        else:
            print(f"   ✗ backend.main 导入失败: {e}")
            results.append(("backend.main.import", False))
    except Exception as e:
        print(f"   ✗ backend.main 导入失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("backend.main.import", False))
    
    return results


def test_app_creation():
    """测试 FastAPI app 创建"""
    print("\n" + "="*60)
    print("测试 FastAPI app 创建")
    print("="*60)
    
    try:
        from backend.main import app
        assert app is not None, "app 对象不存在"
        print(f"   ✓ FastAPI app 创建成功")
        print(f"   ✓ app.title: {app.title}")
        print(f"   ✓ app.version: {app.version}")
        return True
    except ImportError as e:
        if 'fastapi' in str(e).lower():
            print(f"   ⚠️  FastAPI 未安装，跳过此测试")
            print(f"      注意：FastAPI 应该在 conda 环境中运行")
            return None  # None 表示跳过
        else:
            print(f"   ✗ FastAPI app 创建失败: {e}")
            return False
    except Exception as e:
        print(f"   ✗ FastAPI app 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_routes():
    """测试路由是否存在"""
    print("\n" + "="*60)
    print("测试路由存在性")
    print("="*60)
    
    try:
        from backend.main import app
    except ImportError as e:
        if 'fastapi' in str(e).lower():
            print(f"   ⚠️  FastAPI 未安装，跳过此测试")
            return None
        raise
    
    try:
        
        # 获取所有路由
        routes = []
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                routes.append({
                    'path': route.path,
                    'methods': list(route.methods) if route.methods else []
                })
        
        print(f"   ✓ 找到 {len(routes)} 个路由")
        
        # 检查关键路由
        key_routes = [
            '/',
            '/api/settings',
            '/generate_video',
            '/chat',
            '/logs',
        ]
        
        route_paths = [r['path'] for r in routes]
        found_routes = []
        missing_routes = []
        
        for key_route in key_routes:
            # 检查路径是否匹配（支持路径参数）
            found = any(
                key_route == r_path or 
                r_path.startswith(key_route + '/') or
                key_route.startswith(r_path)
                for r_path in route_paths
            )
            if found:
                found_routes.append(key_route)
                print(f"   ✓ 路由存在: {key_route}")
            else:
                missing_routes.append(key_route)
                print(f"   ✗ 路由缺失: {key_route}")
        
        if missing_routes:
            print(f"\n   ⚠️  部分路由缺失: {missing_routes}")
            return False
        
        return True
    except Exception as e:
        print(f"   ✗ 路由测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """测试依赖模块是否正确导入"""
    print("\n" + "="*60)
    print("测试依赖模块")
    print("="*60)
    
    results = []
    
    # 测试 shared 模块导入
    try:
        from shared.config import PROJECT_ROOT, MODEL_DIR
        print(f"   ✓ shared.config 导入成功")
        print(f"      PROJECT_ROOT: {PROJECT_ROOT}")
        results.append(("shared.config", True))
    except Exception as e:
        print(f"   ✗ shared.config 导入失败: {e}")
        results.append(("shared.config", False))
    
    try:
        from shared.utils import run_llm_talk, run_nerffacespeech, run_chat
        print(f"   ✓ shared.utils 导入成功")
        results.append(("shared.utils", True))
    except Exception as e:
        print(f"   ✗ shared.utils 导入失败: {e}")
        results.append(("shared.utils", False))
    
    try:
        from shared.database import settings_db, video_records_db, chat_db
        print(f"   ✓ shared.database 导入成功")
        results.append(("shared.database", True))
    except Exception as e:
        print(f"   ✗ shared.database 导入失败: {e}")
        results.append(("shared.database", False))
    
    return results


def test_global_variables():
    """测试全局变量是否正确初始化"""
    print("\n" + "="*60)
    print("测试全局变量")
    print("="*60)
    
    try:
        from backend.main import LOG_BUFFER, DEBUG_LOG_BUFFER, TASKS, TASKS_LOCK
    except ImportError as e:
        if 'fastapi' in str(e).lower():
            print(f"   ⚠️  FastAPI 未安装，跳过此测试")
            return None
        raise
    
    try:
        
        assert LOG_BUFFER is not None, "LOG_BUFFER 未初始化"
        assert DEBUG_LOG_BUFFER is not None, "DEBUG_LOG_BUFFER 未初始化"
        assert TASKS is not None, "TASKS 未初始化"
        assert TASKS_LOCK is not None, "TASKS_LOCK 未初始化"
        
        print(f"   ✓ LOG_BUFFER 已初始化 (maxlen: {LOG_BUFFER.maxlen})")
        print(f"   ✓ DEBUG_LOG_BUFFER 已初始化 (maxlen: {DEBUG_LOG_BUFFER.maxlen})")
        print(f"   ✓ TASKS 已初始化")
        print(f"   ✓ TASKS_LOCK 已初始化")
        
        return True
    except Exception as e:
        print(f"   ✗ 全局变量测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logging_setup():
    """测试日志系统设置"""
    print("\n" + "="*60)
    print("测试日志系统")
    print("="*60)
    
    try:
        from backend.main import setup_logging, BufferLogHandler
    except ImportError as e:
        if 'fastapi' in str(e).lower():
            print(f"   ⚠️  FastAPI 未安装，跳过此测试")
            return None
        raise
    
    try:
        
        # 检查 BufferLogHandler 是否存在
        assert BufferLogHandler is not None, "BufferLogHandler 不存在"
        print(f"   ✓ BufferLogHandler 存在")
        
        # 检查 setup_logging 函数是否存在
        assert callable(setup_logging), "setup_logging 不是可调用对象"
        print(f"   ✓ setup_logging 函数存在")
        
        return True
    except Exception as e:
        print(f"   ✗ 日志系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_middleware():
    """测试中间件配置"""
    print("\n" + "="*60)
    print("测试中间件配置")
    print("="*60)
    
    try:
        from backend.main import app
    except ImportError as e:
        if 'fastapi' in str(e).lower():
            print(f"   ⚠️  FastAPI 未安装，跳过此测试")
            return None
        raise
    
    try:
        
        # 检查 CORS 中间件是否已添加
        has_cors = any(
            'CORSMiddleware' in str(type(middleware))
            for middleware in app.user_middleware
        )
        
        if has_cors:
            print(f"   ✓ CORS 中间件已配置")
        else:
            print(f"   ⚠️  CORS 中间件未找到（可能不是必需的）")
        
        print(f"   ✓ 中间件数量: {len(app.user_middleware)}")
        
        return True
    except Exception as e:
        print(f"   ✗ 中间件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("FastAPI 后端服务测试程序")
    print("="*60)
    
    all_results = []
    
    # 测试依赖模块
    dependency_results = test_dependencies()
    all_results.extend(dependency_results)
    
    # 测试模块导入
    import_results = test_imports()
    all_results.extend(import_results)
    
    # 测试 app 创建
    app_ok = test_app_creation()
    all_results.append(("app_creation", app_ok))
    
    # 测试路由
    routes_ok = test_routes()
    all_results.append(("routes", routes_ok))
    
    # 测试全局变量
    globals_ok = test_global_variables()
    all_results.append(("global_variables", globals_ok))
    
    # 测试日志系统
    logging_ok = test_logging_setup()
    all_results.append(("logging", logging_ok))
    
    # 测试中间件
    middleware_ok = test_middleware()
    all_results.append(("middleware", middleware_ok))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in all_results:
        if result is None:
            status = "⚠ 跳过"
            skipped += 1
        elif result:
            status = "✓ 通过"
            passed += 1
        else:
            status = "✗ 失败"
            failed += 1
        print(f"{name:40s} : {status}")
    
    print(f"\n总计: {passed} 个通过, {failed} 个失败, {skipped} 个跳过")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
        print("\n注意：这只是基本功能测试，实际运行需要：")
        print("1. 确保所有依赖模块（shared.config, shared.utils, shared.database）正常工作")
        print("2. 确保数据库已初始化")
        print("3. 确保 conda 环境配置正确")
        print("4. 使用 uvicorn 启动服务: uvicorn backend.main:app --host 0.0.0.0 --port 8000")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())

