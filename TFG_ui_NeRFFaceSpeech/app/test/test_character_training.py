#!/usr/bin/env python3
"""
角色训练功能测试
包括：
1. 后端 API 角色训练接口
2. 训练任务状态查询
3. 角色列表查询
4. 角色状态查询
"""
import sys
import os
import time
import requests
import subprocess
import tempfile
import shutil
import signal
import threading
import argparse
from pathlib import Path
from typing import Optional

# 尝试导入 cv2（可选）
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV 未安装，将尝试使用现有测试视频或跳过视频创建")

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradio_app.shared.config import (
    NERF_CODE_DIR, MODEL_DIR, NERF_CONDA_PYTHON, API_CONDA_PYTHON, PROJECT_ROOT as CONFIG_PROJECT_ROOT
)

# 配置
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
CHARACTER_NAME = "test_student"  # 测试角色名称

# 后端启动配置
BACKEND_STARTUP_TIMEOUT = 30  # 后端启动超时时间（秒）
AUTO_START_BACKEND = os.environ.get("AUTO_START_BACKEND", "true").lower() == "true"

# 全局变量
_backend_process: Optional[subprocess.Popen] = None
_test_video_path: Optional[Path] = None
_user_video_path: Optional[Path] = None  # 用户指定的视频路径


def print_section(title: str):
    """打印测试章节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def create_test_video(output_path: Path, duration_seconds: int = 5, fps: int = 25, width: int = 640, height: int = 480):
    """
    创建一个简单的测试视频（包含一个移动的彩色矩形）
    
    Args:
        output_path: 输出视频路径
        duration_seconds: 视频时长（秒）
        fps: 帧率
        width: 视频宽度
        height: 视频高度
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) 未安装，无法创建测试视频")
    
    print(f"创建测试视频: {output_path}")
    print(f"  时长: {duration_seconds}秒, 帧率: {fps}fps, 分辨率: {width}x{height}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    for i in range(total_frames):
        # 创建彩色背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加一个移动的彩色矩形（模拟人脸区域）
        rect_size = min(width, height) // 3
        x = int((width - rect_size) * (i / total_frames))
        y = int((height - rect_size) / 2)
        
        # 绘制一个类似人脸的矩形（中心位置）
        center_x = width // 2
        center_y = height // 2
        face_size = min(width, height) // 2
        
        # 绘制"人脸"区域（肤色矩形）
        cv2.rectangle(frame, 
                     (center_x - face_size // 2, center_y - face_size // 2),
                     (center_x + face_size // 2, center_y + face_size // 2),
                     (200, 180, 160), -1)  # 肤色
        
        # 添加一些特征点（眼睛、嘴巴）
        eye_y = center_y - face_size // 4
        cv2.circle(frame, (center_x - face_size // 4, eye_y), 10, (0, 0, 0), -1)  # 左眼
        cv2.circle(frame, (center_x + face_size // 4, eye_y), 10, (0, 0, 0), -1)  # 右眼
        cv2.ellipse(frame, (center_x, center_y + face_size // 4), 
                   (face_size // 4, face_size // 8), 0, 0, 180, (0, 0, 0), 2)  # 嘴巴
        
        out.write(frame)
    
    out.release()
    print(f"✅ 测试视频创建完成: {output_path}")
    print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def start_backend() -> Optional[subprocess.Popen]:
    """启动后端服务"""
    global _backend_process
    
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
    env["PYTHONPATH"] = str(PROJECT_ROOT / "gradio_app") + os.pathsep + env.get("PYTHONPATH", "")
    
    print(f"执行命令: {' '.join(backend_cmd)}")
    
    try:
        process = subprocess.Popen(
            backend_cmd,
            cwd=str(PROJECT_ROOT / "gradio_app"),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        _backend_process = process
        
        # 启动日志输出线程
        def log_output():
            for line in process.stdout:
                print(f"[后端] {line.rstrip()}")
        
        log_thread = threading.Thread(target=log_output, daemon=True)
        log_thread.start()
        
        # 等待后端启动
        print("等待后端服务启动...")
        max_retries = BACKEND_STARTUP_TIMEOUT // 2
        for i in range(max_retries):
            time.sleep(2)
            try:
                response = requests.get(f"{API_BASE_URL}/docs", timeout=2)
                if response.status_code == 200:
                    print("✅ 后端服务启动成功！")
                    print(f"   后端地址: {API_BASE_URL}\n")
                    return process
            except requests.exceptions.ConnectionError:
                # 连接错误是正常的，如果服务还没启动
                pass
            except Exception as e:
                print(f"  ⚠️  健康检查失败: {e}")
            
            # 检查进程是否还在运行
            if process.poll() is not None:
                print(f"❌ 后端进程意外退出，返回码: {process.returncode}")
                return None
        
        # 如果超时，检查进程状态
        if process.poll() is None:
            print("⚠️  后端进程正在运行，但 HTTP 检查未通过")
            print("   请检查后端服务日志\n")
            return process
        else:
            print("❌ 后端服务启动失败\n")
            return None
            
    except Exception as e:
        print(f"❌ 启动后端服务时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def stop_backend():
    """停止后端服务"""
    global _backend_process
    
    if _backend_process is None:
        return
    
    print("\n" + "=" * 60)
    print("停止后端服务...")
    print("=" * 60)
    
    try:
        # 尝试优雅地终止进程
        _backend_process.terminate()
        
        # 等待进程结束（最多5秒）
        try:
            _backend_process.wait(timeout=5)
            print("✅ 后端服务已停止")
        except subprocess.TimeoutExpired:
            # 如果5秒内没有结束，强制终止
            print("⚠️  后端服务未在5秒内停止，强制终止...")
            _backend_process.kill()
            _backend_process.wait()
            print("✅ 后端服务已强制停止")
    except Exception as e:
        print(f"⚠️  停止后端服务时出错: {e}")
    finally:
        _backend_process = None


def signal_handler(signum, frame):
    """信号处理器，用于清理资源"""
    print("\n\n收到中断信号，正在清理...")
    stop_backend()
    cleanup_test_data()
    sys.exit(0)


def cleanup_test_data():
    """清理测试数据"""
    global _test_video_path
    
    # 只清理临时创建的测试视频，不删除用户指定的视频
    if _test_video_path and _test_video_path.exists() and _user_video_path is None:
        try:
            # 检查是否是临时目录中的文件
            if "test_character_training_" in str(_test_video_path.parent):
                _test_video_path.unlink()
                print(f"清理临时测试视频: {_test_video_path}")
            _test_video_path = None
        except Exception as e:
            print(f"⚠️  清理测试视频失败: {e}")


def test_backend_connection():
    """测试1: 测试后端连接"""
    print_section("测试1: 测试后端连接")
    
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print(f"✅ 后端连接成功: {API_BASE_URL}")
            return True
        else:
            print(f"⚠️  后端响应状态码: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到后端: {API_BASE_URL}")
        
        # 如果启用了自动启动，尝试启动后端
        if AUTO_START_BACKEND:
            print("   尝试自动启动后端服务...")
            backend_process = start_backend()
            if backend_process:
                # 再次尝试连接
                try:
                    time.sleep(2)  # 再等待一下
                    response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
                    if response.status_code == 200:
                        print(f"✅ 后端连接成功: {API_BASE_URL}")
                        return True
                except:
                    pass
            
            print("   自动启动后端失败或后端未就绪")
            print("   提示: 可以手动启动后端服务:")
            print("   cd gradio_app && uvicorn backend.main:app --host 0.0.0.0 --port 8000")
            return False
        else:
            print("   请确保后端服务已启动:")
            print("   cd gradio_app && uvicorn backend.main:app --host 0.0.0.0 --port 8000")
            print("   或设置 AUTO_START_BACKEND=true 自动启动后端")
            return False
    except Exception as e:
        print(f"❌ 测试后端连接时出错: {e}")
        return False


def test_list_characters_api():
    """测试2: 测试列出角色 API"""
    print_section("测试2: 测试列出角色 API")
    
    try:
        response = requests.get(f"{API_BASE_URL}/character/list", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                characters = data.get("characters", [])
                print(f"✅ 成功获取角色列表，共 {len(characters)} 个角色")
                for i, char in enumerate(characters[:5]):  # 只显示前5个
                    print(f"  {i+1}. {char.get('character_name', 'Unknown')}")
                    print(f"     图像数量: {char.get('num_images', 0)}")
                    print(f"     音频存在: {char.get('audio_exists', False)}")
                return True
            else:
                print(f"❌ API 返回失败: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ API 响应状态码: {response.status_code}")
            print(f"   响应内容: {response.text[:500]}")
            return False
    except Exception as e:
        print(f"❌ 测试列出角色 API 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_character_training_api():
    """测试3: 测试角色训练 API"""
    print_section("测试3: 测试角色训练 API")
    
    global _test_video_path
    
    # 优先使用用户指定的视频
    if _user_video_path:
        if not _user_video_path.exists():
            print(f"❌ 指定的视频文件不存在: {_user_video_path}")
            return None
        _test_video_path = _user_video_path
        print(f"✅ 使用指定的视频文件: {_test_video_path}")
    else:
        # 创建测试视频
        temp_dir = Path(tempfile.mkdtemp(prefix="test_character_training_"))
        _test_video_path = temp_dir / "test_video.mp4"
        
        try:
            # 尝试创建测试视频
            if CV2_AVAILABLE:
                create_test_video(_test_video_path, duration_seconds=3, fps=25)
            else:
                # 如果没有 cv2，尝试查找现有的测试视频
                print("⚠️  OpenCV 未安装，尝试查找现有测试视频...")
                possible_test_videos = [
                    PROJECT_ROOT / "data" / "geneface_datasets" / "data" / "raw" / "videos",
                    PROJECT_ROOT / "test_data",
                ]
                
                test_video_found = False
                for test_dir in possible_test_videos:
                    if test_dir.exists():
                        videos = list(test_dir.glob("*.mp4")) + list(test_dir.glob("*.MP4"))
                        if videos:
                            _test_video_path = videos[0]
                            print(f"✅ 找到测试视频: {_test_video_path}")
                            test_video_found = True
                            break
                
                if not test_video_found:
                    print("❌ 未找到测试视频，且无法创建（需要 OpenCV）")
                    print("   请安装 OpenCV: pip install opencv-python")
                    print("   或使用 --video 参数指定一个测试视频文件")
                    return None
        except Exception as e:
            print(f"❌ 创建或查找测试视频失败: {e}")
            return None
    
    try:
        
        # 准备请求
        print(f"\n提交角色训练请求:")
        print(f"  角色名称: {CHARACTER_NAME}")
        print(f"  视频文件: {_test_video_path}")
        
        with open(_test_video_path, "rb") as f:
            files = {"video": ("test_video.mp4", f, "video/mp4")}
            data = {
                "character_name": CHARACTER_NAME,
                "face_ratio": "0.6",
                "output_size_w": "1024",
                "output_size_h": "1024",
                "ffhq_alignment": "false",  # 测试时禁用 FFHQ 对齐（避免依赖问题）
                "overwrite": "true",
            }
            
            response = requests.post(
                f"{API_BASE_URL}/character/train",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                task_id = result.get("task_id")
                print(f"✅ 角色训练任务提交成功")
                print(f"   任务ID: {task_id}")
                print(f"   角色名称: {result.get('character_name')}")
                return task_id
            else:
                error = result.get("error", "Unknown error")
                print(f"❌ 角色训练任务提交失败: {error}")
                return None
        else:
            print(f"❌ API 响应状态码: {response.status_code}")
            print(f"   响应内容: {response.text[:500]}")
            return None
            
    except Exception as e:
        print(f"❌ 测试角色训练 API 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_training_status_api(task_id: str, max_wait_time: int = 300):
    """测试4: 测试训练任务状态查询 API"""
    print_section("测试4: 测试训练任务状态查询 API")
    
    if not task_id:
        print("⚠️  没有有效的任务ID，跳过状态查询测试")
        return False
    
    print(f"查询任务状态: {task_id}")
    print(f"最大等待时间: {max_wait_time}秒")
    
    start_time = time.time()
    poll_interval = 3  # 每3秒查询一次
    
    while True:
        try:
            response = requests.get(
                f"{API_BASE_URL}/character/train/status/{task_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    status = data.get("status", "unknown")
                    print(f"\n任务状态: {status}")
                    
                    if status == "completed":
                        result = data.get("result", {})
                        print(f"✅ 训练任务完成！")
                        print(f"   角色目录: {result.get('character_dir', 'N/A')}")
                        print(f"   图像目录: {result.get('images_dir', 'N/A')}")
                        print(f"   音频文件: {result.get('audio_file', 'N/A')}")
                        print(f"   提取帧数: {result.get('num_frames', 0)}")
                        
                        # 验证输出文件是否存在
                        character_dir = Path(result.get('character_dir', ''))
                        if character_dir.exists():
                            images_dir = character_dir / "images"
                            audio_file = character_dir / "audio.wav"
                            
                            if images_dir.exists():
                                num_images = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
                                print(f"   ✅ 图像目录存在，包含 {num_images} 张图像")
                            else:
                                print(f"   ⚠️  图像目录不存在: {images_dir}")
                            
                            if audio_file.exists():
                                print(f"   ✅ 音频文件存在: {audio_file.stat().st_size / 1024:.2f} KB")
                            else:
                                print(f"   ⚠️  音频文件不存在: {audio_file}")
                            
                            # 验证 PTI 模型文件是否存在
                            pti_models = result.get('pti_models', {})
                            if pti_models:
                                print(f"\n   📦 PTI 模型文件:")
                                g_pti = character_dir / "G_PTI.pt"
                                w_pti = character_dir / "w_PTI.pt"
                                bg_pti = character_dir / "bg_PTI.pt"
                                
                                if g_pti.exists():
                                    print(f"      ✅ G_PTI.pt 存在: {g_pti.stat().st_size / 1024 / 1024:.2f} MB")
                                else:
                                    print(f"      ⚠️  G_PTI.pt 不存在")
                                
                                if w_pti.exists():
                                    print(f"      ✅ w_PTI.pt 存在: {w_pti.stat().st_size / 1024 / 1024:.2f} MB")
                                else:
                                    print(f"      ⚠️  w_PTI.pt 不存在")
                                
                                if bg_pti.exists():
                                    print(f"      ✅ bg_PTI.pt 存在: {bg_pti.stat().st_size / 1024 / 1024:.2f} MB")
                                else:
                                    print(f"      ⚠️  bg_PTI.pt 不存在")
                                
                                # 检查是否所有模型文件都存在
                                all_models_exist = g_pti.exists() and w_pti.exists() and bg_pti.exists()
                                if all_models_exist:
                                    print(f"      ✅ 所有 PTI 模型文件已生成")
                                else:
                                    print(f"      ⚠️  部分 PTI 模型文件缺失（训练可能仍在进行中或失败）")
                            else:
                                print(f"\n   ⚠️  PTI 模型文件信息未返回（可能训练仍在进行中）")
                        
                        return True
                    elif status == "failed":
                        error = data.get("error", "Unknown error")
                        print(f"❌ 训练任务失败: {error}")
                        return False
                    elif status in ["pending", "processing"]:
                        elapsed = time.time() - start_time
                        if elapsed > max_wait_time:
                            print(f"⚠️  任务超时（超过 {max_wait_time} 秒）")
                            return False
                        print(f"   等待中... (已等待 {elapsed:.0f}秒)")
                        time.sleep(poll_interval)
                        continue
                    else:
                        print(f"⚠️  未知状态: {status}")
                        return False
                else:
                    error = data.get("error", "Unknown error")
                    print(f"❌ 查询训练状态失败: {error}")
                    return False
            else:
                print(f"❌ API 响应状态码: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 查询训练状态时出错: {e}")
            return False


def test_pti_models_exist(character_name: str) -> bool:
    """测试6: 验证 PTI 模型文件是否存在"""
    character_dir = PROJECT_ROOT / "assets" / "charactor" / character_name
    
    if not character_dir.exists():
        print(f"❌ 角色目录不存在: {character_dir}")
        return False
    
    g_pti = character_dir / "G_PTI.pt"
    w_pti = character_dir / "w_PTI.pt"
    bg_pti = character_dir / "bg_PTI.pt"
    
    models_status = {
        "G_PTI.pt": g_pti.exists(),
        "w_PTI.pt": w_pti.exists(),
        "bg_PTI.pt": bg_pti.exists(),
    }
    
    all_exist = all(models_status.values())
    
    print(f"验证 PTI 模型文件:")
    for model_name, exists in models_status.items():
        status = "✅" if exists else "❌"
        if exists:
            model_path = character_dir / model_name
            size_mb = model_path.stat().st_size / 1024 / 1024
            print(f"   {status} {model_name}: 存在 ({size_mb:.2f} MB)")
        else:
            print(f"   {status} {model_name}: 不存在")
    
    if all_exist:
        print(f"✅ 所有 PTI 模型文件已生成")
    else:
        print(f"⚠️  部分 PTI 模型文件缺失")
        print(f"   提示: PTI 训练可能需要较长时间，如果训练仍在进行中，请等待完成")
    
    return all_exist


def test_character_status_api(character_name: str):
    """测试5: 测试角色状态查询 API"""
    print_section("测试5: 测试角色状态查询 API")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/character/{character_name}/status",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                char_data = data.get("data", {})
                print(f"✅ 成功查询角色状态")
                print(f"   角色名称: {char_data.get('character_name', 'N/A')}")
                print(f"   存在: {char_data.get('exists', False)}")
                print(f"   图像数量: {char_data.get('num_images', 0)}")
                print(f"   音频存在: {char_data.get('audio_exists', False)}")
                print(f"   角色目录: {char_data.get('character_dir', 'N/A')}")
                
                # 验证 PTI 模型文件状态
                pti_models = char_data.get('pti_models', {})
                pti_models_exist = char_data.get('pti_models_exist', False)
                
                if pti_models:
                    print(f"\n   📦 PTI 模型文件状态:")
                    print(f"      G_PTI: {'✅' if pti_models.get('G_PTI') else '❌'}")
                    print(f"      w_PTI: {'✅' if pti_models.get('w_PTI') else '❌'}")
                    print(f"      bg_PTI: {'✅' if pti_models.get('bg_PTI') else '❌'}")
                    print(f"      所有模型存在: {'✅' if pti_models_exist else '❌'}")
                else:
                    print(f"   ⚠️  PTI 模型文件信息未返回")
                
                return True
            else:
                error = data.get("error", "Unknown error")
                print(f"❌ 查询角色状态失败: {error}")
                return False
        else:
            print(f"❌ API 响应状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 测试角色状态查询 API 时出错: {e}")
        return False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="角色训练功能测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认测试视频
  python test_character_training.py
  
  # 指定输入视频文件
  python test_character_training.py --video /path/to/video.mp4
  
  # 指定角色名称
  python test_character_training.py --video /path/to/video.mp4 --character my_character
  
  # 指定 API 地址和角色名称
  python test_character_training.py --video /path/to/video.mp4 --character my_character --api-url http://localhost:8000
        """
    )
    
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="输入视频文件路径（支持 .mp4, .avi, .mov, .mkv 格式）"
    )
    
    parser.add_argument(
        "--character", "-c",
        type=str,
        default=CHARACTER_NAME,
        help=f"角色名称（默认: {CHARACTER_NAME}）"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help=f"后端 API 地址（默认: {API_BASE_URL}）"
    )
    
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="不自动启动后端服务（需要手动启动后端）"
    )
    
    parser.add_argument(
        "--max-wait-time",
        type=int,
        default=600,
        help="最大等待时间（秒，默认: 600）"
    )
    
    return parser.parse_args()


def main():
    """主测试函数"""
    global CHARACTER_NAME, API_BASE_URL, _user_video_path
    
    # 解析命令行参数
    args = parse_args()
    
    # 更新配置
    if args.video:
        video_path = Path(args.video).resolve()
        if not video_path.exists():
            print(f"❌ 错误: 视频文件不存在: {video_path}")
            sys.exit(1)
        _user_video_path = video_path
        print(f"📹 使用指定的视频文件: {_user_video_path}")
    
    if args.character:
        CHARACTER_NAME = args.character
        print(f"👤 角色名称: {CHARACTER_NAME}")
    
    if args.api_url:
        API_BASE_URL = args.api_url
        print(f"🌐 API 地址: {API_BASE_URL}")
    
    global AUTO_START_BACKEND
    if args.no_auto_start:
        AUTO_START_BACKEND = False
        print("🚫 自动启动后端: 已禁用")
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\n" + "=" * 60)
    print("  角色训练功能测试")
    print("=" * 60)
    
    if AUTO_START_BACKEND:
        print(f"\n自动启动后端: 已启用 (AUTO_START_BACKEND={AUTO_START_BACKEND})")
    else:
        print(f"\n自动启动后端: 已禁用 (设置 AUTO_START_BACKEND=true 启用)")
    
    results = {}
    
    # 测试1: 测试后端连接
    results["backend_connection"] = test_backend_connection()
    
    if not results["backend_connection"]:
        print("\n❌ 后端未连接，无法继续测试")
        cleanup_test_data()
        return
    
    # 测试2: 列出角色
    results["list_characters"] = test_list_characters_api()
    
    # 测试3: 提交角色训练任务
    task_id = test_character_training_api()
    results["start_training"] = task_id is not None
    
    # 测试4: 查询训练状态
    if task_id:
        results["training_status"] = test_training_status_api(task_id, max_wait_time=args.max_wait_time)
    
    # 测试5: 查询角色状态（包含 PTI 模型文件验证）
    results["character_status"] = test_character_status_api(CHARACTER_NAME)
    
    # 测试6: 验证 PTI 模型文件（如果训练完成）
    if results.get("training_status"):
        print_section("测试6: 验证 PTI 模型文件")
        results["pti_models"] = test_pti_models_exist(CHARACTER_NAME)
    else:
        results["pti_models"] = None
    
    # 再次列出角色（验证新角色已添加）
    print_section("验证: 再次列出角色")
    test_list_characters_api()
    
    # 打印测试总结
    print_section("测试总结")
    
    total_tests = len([v for v in results.values() if v is not None])
    passed_tests = len([v for v in results.values() if v is True])
    failed_tests = len([v for v in results.values() if v is False])
    
    print(f"总测试数: {total_tests}")
    print(f"通过: {passed_tests}")
    print(f"失败: {failed_tests}")
    
    print("\n详细结果:")
    for test_name, result in results.items():
        if result is True:
            status = "✅ 通过"
        elif result is False:
            status = "❌ 失败"
        else:
            status = "⚠️  跳过"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试都通过了！")
    else:
        print("\n⚠️  部分测试失败，请检查上述输出")
    
    # 清理资源
    cleanup_test_data()
    
    # 如果自动启动了后端，停止它
    if AUTO_START_BACKEND and _backend_process is not None:
        stop_backend()


if __name__ == "__main__":
    main()

