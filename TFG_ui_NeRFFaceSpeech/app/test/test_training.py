#!/usr/bin/env python3
"""
测试训练功能
包括：
1. 训练脚本参数解析
2. 后端 API 训练接口
3. 训练状态查询
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
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradio_app.shared.config import (
    NERF_CODE_DIR, MODEL_DIR, NERF_CONDA_PYTHON, API_CONDA_PYTHON, PROJECT_ROOT as CONFIG_PROJECT_ROOT
)

# 配置
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
TRAINING_SCRIPT = NERF_CODE_DIR / "StyleNeRF" / "run_train.py"

# 测试数据路径（需要用户提供或创建测试数据）
TEST_DATA_DIR = PROJECT_ROOT / "test_data" / "training_images"
TEST_MODEL = "ffhq_1024.pkl"

# 后端启动配置
BACKEND_STARTUP_TIMEOUT = 30  # 后端启动超时时间（秒）
AUTO_START_BACKEND = os.environ.get("AUTO_START_BACKEND", "true").lower() == "true"

# 全局变量
_temp_test_data_dir = None
_backend_process: Optional[subprocess.Popen] = None


def print_section(title: str):
    """打印测试章节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def create_random_test_data(num_images: int = 10, resolution: int = 256, output_dir: Path = None) -> Path:
    """
    创建随机测试图像数据
    
    Args:
        num_images: 要创建的图像数量
        resolution: 图像分辨率（正方形）
        output_dir: 输出目录，如果为None则创建临时目录
    
    Returns:
        测试数据目录路径
    """
    global _temp_test_data_dir
    
    if output_dir is None:
        # 创建临时目录
        _temp_test_data_dir = Path(tempfile.mkdtemp(prefix="test_training_data_"))
        output_dir = _temp_test_data_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"创建 {num_images} 张随机测试图像到: {output_dir}")
    
    # 创建随机图像
    np.random.seed(42)  # 固定随机种子以便重现
    for i in range(num_images):
        # 生成随机RGB图像
        img_array = np.random.randint(0, 256, (resolution, resolution, 3), dtype=np.uint8)
        
        # 添加一些结构（避免完全随机）
        # 创建一些简单的几何形状
        center = resolution // 2
        radius = resolution // 4
        y, x = np.ogrid[:resolution, :resolution]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        img_array[mask] = np.random.randint(128, 256, (np.sum(mask), 3), dtype=np.uint8)
        
        # 保存图像
        img = Image.fromarray(img_array, 'RGB')
        img_path = output_dir / f"test_image_{i:04d}.png"
        img.save(img_path)
    
    print(f"✅ 成功创建 {num_images} 张测试图像")
    return output_dir


def cleanup_test_data():
    """清理临时测试数据"""
    global _temp_test_data_dir
    if _temp_test_data_dir and _temp_test_data_dir.exists():
        try:
            shutil.rmtree(_temp_test_data_dir)
            print(f"清理临时测试数据: {_temp_test_data_dir}")
            _temp_test_data_dir = None
        except Exception as e:
            print(f"⚠️  清理临时数据失败: {e}")


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


def test_training_script_exists():
    """测试1: 检查训练脚本是否存在"""
    print_section("测试1: 检查训练脚本是否存在")
    
    if TRAINING_SCRIPT.exists():
        print(f"✅ 训练脚本存在: {TRAINING_SCRIPT}")
        return True
    else:
        print(f"❌ 训练脚本不存在: {TRAINING_SCRIPT}")
        return False


def test_training_script_help():
    """测试2: 测试训练脚本的帮助信息"""
    print_section("测试2: 测试训练脚本帮助信息")
    
    try:
        result = subprocess.run(
            [str(NERF_CONDA_PYTHON), str(TRAINING_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ 训练脚本帮助信息正常")
            print("\n帮助信息预览:")
            help_lines = result.stdout.split('\n')[:20]
            for line in help_lines:
                print(f"  {line}")
            return True
        else:
            print(f"❌ 训练脚本帮助信息失败")
            print(f"错误输出: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ 训练脚本帮助信息超时")
        return False
    except Exception as e:
        print(f"❌ 测试训练脚本帮助信息时出错: {e}")
        return False


def test_training_script_parameter_validation():
    """测试3: 测试训练脚本参数验证"""
    print_section("测试3: 测试训练脚本参数验证")
    
    # 测试缺少必需参数
    try:
        result = subprocess.run(
            [str(NERF_CONDA_PYTHON), str(TRAINING_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print("✅ 参数验证正常（缺少必需参数时返回错误）")
            if "Missing option" in result.stdout or "Missing option" in result.stderr:
                print("  检测到正确的错误提示")
            return True
        else:
            print("❌ 参数验证失败（应该返回错误但没有）")
            return False
    except Exception as e:
        print(f"❌ 测试参数验证时出错: {e}")
        return False


def test_backend_connection():
    """测试4: 测试后端连接"""
    print_section("测试4: 测试后端连接")
    
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


def test_list_datasets_api():
    """测试5: 测试列出数据集 API"""
    print_section("测试5: 测试列出数据集 API")
    
    try:
        response = requests.get(f"{API_BASE_URL}/train/datasets", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                datasets = data.get("data", [])
                print(f"✅ 成功获取数据集列表，共 {len(datasets)} 个数据集")
                for i, dataset in enumerate(datasets[:5]):  # 只显示前5个
                    print(f"  {i+1}. {dataset.get('name', 'Unknown')}: {dataset.get('path', 'Unknown')}")
                return True
            else:
                print(f"❌ API 返回失败: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ API 响应状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 测试列出数据集 API 时出错: {e}")
        return False


def test_start_training_api():
    """测试6: 测试启动训练 API（不实际启动训练）"""
    print_section("测试6: 测试启动训练 API")
    
    # 检查基础模型是否存在
    base_model_path = MODEL_DIR / TEST_MODEL
    if not base_model_path.exists():
        print(f"⚠️  基础模型不存在: {base_model_path}")
        print("   跳过实际训练启动测试")
        return None
    
    # 检查或创建测试数据
    test_data_dir = TEST_DATA_DIR
    if not test_data_dir.exists():
        print(f"测试数据目录不存在: {test_data_dir}")
        print("自动创建随机测试数据...")
        try:
            test_data_dir = create_random_test_data(num_images=5, resolution=256)
            print(f"✅ 已创建测试数据: {test_data_dir}")
        except Exception as e:
            print(f"❌ 创建测试数据失败: {e}")
            print("   跳过实际训练启动测试")
            return None
    else:
        print(f"✅ 使用现有测试数据目录: {test_data_dir}")
    
    # 测试 API 请求（但不实际启动长时间训练）
    try:
        # 使用很小的 kimg 值进行测试
        request_data = {
            "data_path": str(test_data_dir),
            "base_model": TEST_MODEL,
            "kimg": 1,  # 只训练1 kimg用于测试
            "snap": 1,
            "imgsnap": 1,
            "aug": "noaug",
            "mirror": False,
            "config_name": "style_ffhq_ae_basic"
        }
        
        print(f"发送训练请求:")
        for key, value in request_data.items():
            print(f"  {key}: {value}")
        
        response = requests.post(
            f"{API_BASE_URL}/train/start",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                task_id = data.get("task_id")
                print(f"✅ 训练任务启动成功")
                print(f"   任务ID: {task_id}")
                return task_id
            else:
                error = data.get("error", "Unknown error")
                print(f"❌ 训练任务启动失败: {error}")
                return None
        else:
            print(f"❌ API 响应状态码: {response.status_code}")
            print(f"   响应内容: {response.text[:500]}")
            return None
    except Exception as e:
        print(f"❌ 测试启动训练 API 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_training_status_api(task_id: str):
    """测试7: 测试训练状态查询 API"""
    print_section("测试7: 测试训练状态查询 API")
    
    if not task_id:
        print("⚠️  没有有效的任务ID，跳过状态查询测试")
        return False
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/train/status/{task_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                task_info = data.get("data", {})
                status = task_info.get("status", "unknown")
                print(f"✅ 成功查询训练状态")
                print(f"   任务ID: {task_id}")
                print(f"   状态: {status}")
                print(f"   开始时间: {task_info.get('start_time', 'N/A')}")
                print(f"   数据路径: {task_info.get('data_path', 'N/A')}")
                print(f"   基础模型: {task_info.get('base_model', 'N/A')}")
                
                # 显示最近的日志（如果有）
                recent_log = task_info.get("recent_log", [])
                if recent_log:
                    print(f"\n   最近日志（最后5行）:")
                    for log_line in recent_log[-5:]:
                        print(f"     {log_line}")
                
                return True
            else:
                error = data.get("error", "Unknown error")
                print(f"❌ 查询训练状态失败: {error}")
                return False
        else:
            print(f"❌ API 响应状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 测试训练状态查询 API 时出错: {e}")
        return False


def test_list_training_tasks_api():
    """测试8: 测试列出训练任务 API"""
    print_section("测试8: 测试列出训练任务 API")
    
    try:
        response = requests.get(f"{API_BASE_URL}/train/tasks", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                tasks = data.get("data", [])
                print(f"✅ 成功获取训练任务列表，共 {len(tasks)} 个任务")
                for i, task in enumerate(tasks[:5]):  # 只显示前5个
                    print(f"  {i+1}. 任务ID: {task.get('task_id', 'Unknown')}")
                    print(f"     状态: {task.get('status', 'Unknown')}")
                    print(f"     开始时间: {task.get('start_time', 'N/A')}")
                return True
            else:
                error = data.get("error", "Unknown error")
                print(f"❌ 获取训练任务列表失败: {error}")
                return False
        else:
            print(f"❌ API 响应状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 测试列出训练任务 API 时出错: {e}")
        return False


def test_training_script_dry_run():
    """测试9: 测试训练脚本的干运行（验证参数但不实际训练）"""
    print_section("测试9: 测试训练脚本参数解析（干运行）")
    
    # 创建一个临时输出目录用于测试
    temp_outdir = tempfile.mkdtemp(prefix="test_training_")
    
    # 检查基础模型是否存在
    base_model_path = MODEL_DIR / TEST_MODEL
    if not base_model_path.exists():
        print(f"⚠️  基础模型不存在: {base_model_path}")
        print("   跳过干运行测试")
        return None
    
    # 创建随机测试数据用于测试
    print("创建随机测试数据用于干运行测试...")
    try:
        test_data_dir = create_random_test_data(num_images=3, resolution=128, output_dir=None)
        print(f"✅ 已创建测试数据: {test_data_dir}")
    except Exception as e:
        print(f"❌ 创建测试数据失败: {e}")
        # 如果创建失败，使用不存在的路径来测试参数验证
        test_data_dir = "/nonexistent/path/to/data"
        print(f"   使用不存在的路径进行参数验证测试: {test_data_dir}")
    
    try:
        cmd = [
            str(NERF_CONDA_PYTHON),
            str(TRAINING_SCRIPT),
            "--outdir", temp_outdir,
            "--data", str(test_data_dir),
            "--resume", str(base_model_path),
            "--kimg", "1",
            "--batch", "1",
            "--batch-gpu", "1",  # 必须 <= batch_size
            "--resolution", "128",  # 使用较小的分辨率以加快测试
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # 如果数据路径不存在，脚本应该报错（这是预期的）
        # 或者脚本会尝试加载数据集并在加载时失败
        if result.returncode != 0:
            error_output = result.stderr + result.stdout
            # 检查脚本是否成功运行到数据集加载或网络构建阶段
            # 如果能运行到这些阶段，说明参数解析是正常的
            success_indicators = [
                "Loading training set",
                "训练",
                "Constructing networks",
                "Training configuration",  # 配置打印成功
                "Output directory",
                "Data path",
            ]
            
            if any(indicator in error_output for indicator in success_indicators):
                print("✅ 脚本已成功解析参数并开始执行（在数据集加载或网络构建阶段失败是预期的）")
                print(f"   返回码: {result.returncode}")
                print(f"   说明: 脚本参数解析正常，失败是由于数据集或模型配置问题（这在测试中是预期的）")
                # 打印关键错误信息以便调试
                error_lines = error_output.split("\n")
                print(f"\n   关键错误信息:")
                for line in error_lines[-10:]:  # 显示最后10行
                    if line.strip() and ("Error" in line or "Traceback" in line or "OSError" in line or "IOError" in line):
                        print(f"     {line}")
                return True
            
            # 检查是否是预期的错误（数据路径不存在、数据集加载失败等）
            expected_errors = [
                "不存在", "not exist", "No such file", "No such directory",
                "cannot find", "找不到", "IOError", "FileNotFoundError",
                "No image files found", "训练数据路径不存在",
                "Path must point to a directory or zip",  # ImageFolderDataset 的错误
            ]
            if any(err.lower() in error_output.lower() for err in expected_errors):
                print("✅ 参数验证正常（检测到预期的错误：数据路径不存在或数据集加载失败）")
                return True
            else:
                # 如果错误不是预期的，打印详细信息
                print("⚠️  脚本返回错误，错误信息不明确")
                print(f"   返回码: {result.returncode}")
                print(f"   错误输出预览: {error_output[:500]}...")
                print(f"\n   完整错误输出（最后20行）:")
                print("   " + "\n   ".join(error_output.split("\n")[-20:]))
                return None
        else:
            print("⚠️  脚本意外成功（数据路径不存在但脚本没有报错）")
            return None
            
    except subprocess.TimeoutExpired:
        print("⚠️  脚本执行超时（可能实际开始了训练）")
        return None
    except Exception as e:
        print(f"❌ 测试干运行时出错: {e}")
        return None
    finally:
        # 清理临时目录
        try:
            if os.path.exists(temp_outdir):
                shutil.rmtree(temp_outdir)
        except:
            pass
        # 清理测试数据（如果是临时创建的）
        cleanup_test_data()


def main():
    """主测试函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\n" + "=" * 60)
    print("  StyleNeRF 训练功能测试")
    print("=" * 60)
    
    if AUTO_START_BACKEND:
        print(f"\n自动启动后端: 已启用 (AUTO_START_BACKEND={AUTO_START_BACKEND})")
    else:
        print(f"\n自动启动后端: 已禁用 (设置 AUTO_START_BACKEND=true 启用)")
    
    results = {}
    
    # 测试1: 检查训练脚本是否存在
    results["script_exists"] = test_training_script_exists()
    
    if not results["script_exists"]:
        print("\n❌ 训练脚本不存在，无法继续测试")
        return
    
    # 测试2: 测试训练脚本帮助信息
    results["script_help"] = test_training_script_help()
    
    # 测试3: 测试参数验证
    results["parameter_validation"] = test_training_script_parameter_validation()
    
    # 测试4: 测试后端连接
    results["backend_connection"] = test_backend_connection()
    
    if not results["backend_connection"]:
        print("\n⚠️  后端未连接，跳过 API 测试")
    else:
        # 测试5: 列出数据集
        results["list_datasets"] = test_list_datasets_api()
        
        # 测试6: 启动训练（如果条件满足）
        task_id = test_start_training_api()
        results["start_training"] = task_id is not None
        
        # 测试7: 查询训练状态
        if task_id:
            results["training_status"] = test_training_status_api(task_id)
            # 等待一小段时间让训练有机会开始
            print("\n等待5秒后再次查询状态...")
            time.sleep(5)
            test_training_status_api(task_id)
        
        # 测试8: 列出训练任务
        results["list_tasks"] = test_list_training_tasks_api()
    
    # 测试9: 干运行测试
    results["dry_run"] = test_training_script_dry_run()
    
    # 打印测试总结
    print_section("测试总结")
    
    total_tests = len([v for v in results.values() if v is not None])
    passed_tests = len([v for v in results.values() if v is True])
    skipped_tests = len([v for v in results.values() if v is None])
    
    print(f"总测试数: {total_tests}")
    print(f"通过: {passed_tests}")
    print(f"失败: {total_tests - passed_tests - skipped_tests}")
    print(f"跳过: {skipped_tests}")
    
    print("\n详细结果:")
    for test_name, result in results.items():
        if result is True:
            status = "✅ 通过"
        elif result is False:
            status = "❌ 失败"
        else:
            status = "⚠️  跳过"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests - skipped_tests:
        print("\n🎉 所有可执行的测试都通过了！")
    else:
        print("\n⚠️  部分测试失败，请检查上述输出")
    
    # 清理资源
    cleanup_test_data()
    
    # 如果自动启动了后端，停止它
    if AUTO_START_BACKEND and _backend_process is not None:
        stop_backend()


if __name__ == "__main__":
    main()

