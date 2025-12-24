#!/usr/bin/env python3
"""
测试后端视频生成功能
启动后端服务并发送视频生成请求进行测试
"""
import sys
import os
import time
import subprocess
import requests
import signal
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import (
    API_CONDA_PYTHON, VIDEOS_STORAGE_DIR, 
    get_character_list, PROJECT_ROOT
)

# 配置
API_BASE_URL = "http://localhost:8000"
BACKEND_STARTUP_TIMEOUT = 30  # 后端启动超时时间（秒）
POLL_INTERVAL = 2  # 状态轮询间隔（秒）
MAX_WAIT_TIME = 600  # 最大等待时间（秒，10分钟）

# 全局变量
backend_process: Optional[subprocess.Popen] = None


def start_backend() -> subprocess.Popen:
    """启动后端服务"""
    print("=" * 60)
    print("启动后端服务...")
    print("=" * 60)
    
    # 检查 API conda 环境
    api_python = API_CONDA_PYTHON if API_CONDA_PYTHON.exists() else Path(sys.executable)
    print(f"使用 Python: {api_python}")
    
    backend_cmd = [
        str(api_python), "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    
    env = os.environ.copy()
    
    backend_process = subprocess.Popen(
        backend_cmd,
        cwd=str(Path(__file__).parent.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # 实时输出后端日志
    def log_backend_output():
        for line in backend_process.stdout:
            print(f"[后端] {line.rstrip()}")
    
    import threading
    log_thread = threading.Thread(target=log_backend_output, daemon=True)
    log_thread.start()
    
    return backend_process


def wait_for_backend(timeout: int = BACKEND_STARTUP_TIMEOUT) -> bool:
    """等待后端服务就绪"""
    print(f"\n等待后端服务启动（最多等待 {timeout} 秒）...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{API_BASE_URL}/docs", timeout=2)
            if response.status_code == 200:
                elapsed = time.time() - start_time
                print(f"✅ 后端服务已就绪（耗时 {elapsed:.1f} 秒）\n")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        print(".", end="", flush=True)
    
    print(f"\n❌ 后端服务启动超时（超过 {timeout} 秒）")
    return False


def get_models() -> list:
    """获取可用模型列表"""
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ 获取到 {len(models)} 个模型")
            return models
        else:
            print(f"❌ 获取模型列表失败（状态码: {response.status_code}）")
            return []
    except Exception as e:
        print(f"❌ 获取模型列表失败: {e}")
        return []


def get_characters() -> list:
    """获取可用角色列表"""
    try:
        characters = get_character_list()
        print(f"✅ 获取到 {len(characters)} 个角色: {characters}")
        return characters
    except Exception as e:
        print(f"❌ 获取角色列表失败: {e}")
        return []


def submit_video_generation(text: str, character: str, model_name: str) -> Optional[str]:
    """提交视频生成任务"""
    print("\n" + "=" * 60)
    print("提交视频生成任务")
    print("=" * 60)
    print(f"文本: {text}")
    print(f"角色: {character}")
    print(f"模型: {model_name}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate_video",
            json={
                "text": text,
                "character": character,
                "model_name": model_name
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                unique_id = result.get("unique_id")
                print(f"✅ 任务提交成功")
                print(f"任务ID: {unique_id}")
                print(f"状态: {result.get('status')}")
                return unique_id
            else:
                print(f"❌ 任务提交失败: {result.get('message', '未知错误')}")
                return None
        else:
            print(f"❌ 请求失败（状态码: {response.status_code}）")
            print(f"响应: {response.text}")
            return None
    except Exception as e:
        print(f"❌ 提交任务失败: {e}")
        return None


def get_task_status(unique_id: str) -> Optional[Dict[str, Any]]:
    """获取任务状态"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/generate_video/status/{unique_id}",
            timeout=5
        )
        if response.status_code == 200:
            result = response.json()
            # 如果返回格式是 {"success": True, "data": {...}}，提取 data
            if isinstance(result, dict) and "data" in result and result.get("success"):
                return result["data"]
            return result
        else:
            return None
    except Exception as e:
        print(f"❌ 获取任务状态失败: {e}")
        return None


def wait_for_completion(unique_id: str, max_wait: int = MAX_WAIT_TIME) -> Optional[Dict[str, Any]]:
    """等待任务完成"""
    print("\n" + "=" * 60)
    print("等待任务完成...")
    print("=" * 60)
    
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < max_wait:
        status = get_task_status(unique_id)
        
        if status is None:
            print("❌ 无法获取任务状态")
            return None
        
        current_status = status.get("status", "unknown")
        
        # 如果状态发生变化，打印新状态
        if current_status != last_status:
            print(f"\n[{time.strftime('%H:%M:%S')}] 状态: {current_status}")
            last_status = current_status
            
            if "progress" in status:
                print(f"  进度: {status['progress']}")
            if "message" in status:
                print(f"  消息: {status['message']}")
        
        # 检查是否完成
        if current_status == "completed":
            elapsed = time.time() - start_time
            print(f"\n✅ 任务完成（耗时 {elapsed:.1f} 秒）")
            return status
        elif current_status == "failed" or current_status == "error":
            print(f"\n❌ 任务失败")
            if "error" in status:
                print(f"  错误: {status['error']}")
            return status
        
        # 等待一段时间后再次查询
        time.sleep(POLL_INTERVAL)
        print(".", end="", flush=True)
    
    print(f"\n❌ 任务超时（超过 {max_wait} 秒）")
    return None


def check_video_file(unique_id: str) -> bool:
    """检查生成的视频文件"""
    print("\n" + "=" * 60)
    print("检查生成的视频文件")
    print("=" * 60)
    
    video_path = VIDEOS_STORAGE_DIR / f"{unique_id}.mp4"
    
    if video_path.exists():
        file_size = video_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        print(f"✅ 视频文件存在")
        print(f"  路径: {video_path}")
        print(f"  大小: {file_size_mb:.2f} MB ({file_size:,} 字节)")
        return True
    else:
        print(f"❌ 视频文件不存在")
        print(f"  期望路径: {video_path}")
        return False


def test_video_generation():
    """主测试函数"""
    global backend_process
    
    print("\n" + "=" * 60)
    print("视频生成功能测试")
    print("=" * 60 + "\n")
    
    try:
        # 1. 启动后端服务
        backend_process = start_backend()
        
        # 2. 等待后端就绪
        if not wait_for_backend():
            print("❌ 后端服务启动失败，测试终止")
            return False
        
        # 3. 获取可用模型和角色
        print("\n" + "=" * 60)
        print("获取系统信息")
        print("=" * 60)
        
        models = get_models()
        if not models:
            print("❌ 无法获取模型列表，测试终止")
            return False
        
        characters = get_characters()
        if not characters:
            print("❌ 无法获取角色列表，测试终止")
            return False
        
        # 选择第一个可用的模型和角色
        model_name = models[0] if models else "未找到模型"
        character = characters[0] if characters else "ayanami"
        
        print(f"\n使用配置:")
        print(f"  模型: {model_name}")
        print(f"  角色: {character}")
        
        # 4. 提交视频生成任务
        test_text = "你好，这是一个测试视频。"
        unique_id = submit_video_generation(
            text=test_text,
            character=character,
            model_name=model_name
        )
        
        if not unique_id:
            print("❌ 任务提交失败，测试终止")
            return False
        
        # 5. 等待任务完成
        final_status = wait_for_completion(unique_id)
        
        if not final_status:
            print("❌ 无法获取最终状态，测试终止")
            return False
        
        if final_status.get("status") != "completed":
            print(f"❌ 任务未成功完成（状态: {final_status.get('status')}）")
            return False
        
        # 6. 检查生成的视频文件
        if not check_video_file(unique_id):
            print("❌ 视频文件检查失败")
            return False
        
        # 7. 测试通过
        print("\n" + "=" * 60)
        print("✅ 测试通过！")
        print("=" * 60)
        print(f"任务ID: {unique_id}")
        print(f"视频文件: {VIDEOS_STORAGE_DIR / f'{unique_id}.mp4'}")
        print(f"视频URL: {API_BASE_URL}/videos/{unique_id}.mp4")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理：停止后端服务
        if backend_process:
            print("\n" + "=" * 60)
            print("停止后端服务...")
            print("=" * 60)
            try:
                backend_process.terminate()
                backend_process.wait(timeout=5)
                print("✅ 后端服务已停止")
            except subprocess.TimeoutExpired:
                backend_process.kill()
                print("⚠️  强制停止后端服务")
            except Exception as e:
                print(f"❌ 停止后端服务时出错: {e}")


def signal_handler(sig, frame):
    """处理中断信号"""
    print("\n\n⚠️  收到中断信号，正在清理...")
    global backend_process
    if backend_process:
        try:
            backend_process.terminate()
            backend_process.wait(timeout=3)
        except:
            pass
    sys.exit(0)


if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行测试source /etc/network_turbo
    success = test_video_generation()
    sys.exit(0 if success else 1)

