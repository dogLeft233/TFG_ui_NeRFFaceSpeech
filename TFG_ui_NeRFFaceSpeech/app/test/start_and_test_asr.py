#!/usr/bin/env python3
"""
启动后端和ASR服务，并运行测试
"""
import subprocess
import sys
import os
import time
import signal
import argparse
from pathlib import Path

# 计算项目根目录
# start_and_test_asr.py 在 gradio_app/ 目录下
# 所以需要向上两级到达项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
BACKEND_DIR = PROJECT_ROOT / "gradio_app" / "backend"
SERVICES_DIR = PROJECT_ROOT / "gradio_app" / "services"

# 进程列表
processes = []


def signal_handler(sig, frame):
    """处理Ctrl+C信号，清理所有进程"""
    print("\n\n⚠️  收到中断信号，正在清理进程...")
    for process in processes:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            try:
                process.kill()
            except:
                pass
    print("✅ 所有进程已清理")
    sys.exit(0)


def check_port(port: int) -> bool:
    """检查端口是否被占用"""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except:
        return False


def start_backend(port: int = 8000) -> subprocess.Popen:
    """启动后端服务"""
    print(f"\n🚀 启动后端服务（端口: {port}）...")
    
    if check_port(port):
        print(f"⚠️  端口 {port} 已被占用，跳过启动后端服务")
        print("   假设后端服务已在运行")
        return None
    
    try:
        os.chdir(str(BACKEND_DIR))
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(process)
        
        # 等待服务启动
        print("   等待服务启动...")
        for i in range(30):  # 最多等待30秒
            if check_port(port):
                print(f"✅ 后端服务已启动: http://localhost:{port}")
                return process
            time.sleep(1)
        
        print(f"❌ 后端服务启动超时")
        return None
        
    except Exception as e:
        print(f"❌ 启动后端服务失败: {e}")
        return None


def start_asr_service(port: int = 8002, model: str = "base") -> subprocess.Popen:
    """启动ASR服务"""
    print(f"\n🚀 启动ASR服务（端口: {port}, 模型: {model}）...")
    
    if check_port(port):
        print(f"⚠️  端口 {port} 已被占用，跳过启动ASR服务")
        print("   假设ASR服务已在运行")
        return None
    
    try:
        asr_service_script = SERVICES_DIR / "asr_service.py"
        # 调试输出
        print(f"   PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"   SERVICES_DIR: {SERVICES_DIR}")
        print(f"   ASR脚本路径: {asr_service_script}")
        print(f"   路径是否存在: {asr_service_script.exists()}")
        
        if not asr_service_script.exists():
            print(f"❌ ASR服务脚本不存在: {asr_service_script}")
            print(f"   请检查路径是否正确")
            return None
        
        process = subprocess.Popen(
            [sys.executable, str(asr_service_script), "--port", str(port), "--model", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(process)
        
        # 等待服务启动
        print("   等待服务启动...")
        for i in range(60):  # 最多等待60秒（模型加载需要时间）
            if check_port(port):
                print(f"✅ ASR服务已启动: http://localhost:{port}")
                return process
            time.sleep(1)
        
        print(f"❌ ASR服务启动超时")
        return None
        
    except Exception as e:
        print(f"❌ 启动ASR服务失败: {e}")
        return None


def run_tests(backend_url: str, audio_file: str, model: str, language: str, test_type: str, character: str):
    """运行测试"""
    print(f"\n🧪 运行测试...")
    
    test_script = PROJECT_ROOT / "gradio_app" / "test_asr_api.py"
    # 调试输出
    print(f"   PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"   测试脚本路径: {test_script}")
    print(f"   路径是否存在: {test_script.exists()}")
    
    if not test_script.exists():
        print(f"❌ 测试脚本不存在: {test_script}")
        print(f"   请检查路径是否正确")
        return False
    
    cmd = [
        sys.executable,
        str(test_script),
        "--backend-url", backend_url,
        "--audio-file", audio_file,
        "--model", model,
        "--test", test_type,
        "--character", character
    ]
    
    if language:
        cmd.extend(["--language", language])
    
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 运行测试失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动后端和ASR服务，并运行测试")
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8000,
        help="后端服务端口（默认: 8000）"
    )
    parser.add_argument(
        "--asr-port",
        type=int,
        default=8002,
        help="ASR服务端口（默认: 8002）"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=str(PROJECT_ROOT / "assets" / "charactors" / "Ayanami" / "绫波丽.wav"),
        help="测试音频文件路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper模型名称（默认: base）"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="语言代码（如'zh', 'en'），None表示自动检测"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["all", "health", "base64", "file", "chat"],
        default="all",
        help="要运行的测试（默认: all）"
    )
    parser.add_argument(
        "--character",
        type=str,
        default="ayanami",
        help="聊天测试使用的角色（默认: ayanami）"
    )
    parser.add_argument(
        "--skip-backend",
        action="store_true",
        help="跳过启动后端服务（假设已在运行）"
    )
    parser.add_argument(
        "--skip-asr",
        action="store_true",
        help="跳过启动ASR服务（假设已在运行）"
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="测试完成后保持服务运行（不退出）"
    )
    
    args = parser.parse_args()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("="*60)
    print("启动服务并运行ASR测试")
    print("="*60)
    print(f"后端端口: {args.backend_port}")
    print(f"ASR端口: {args.asr_port}")
    print(f"模型: {args.model}")
    print(f"测试类型: {args.test}")
    print("="*60)
    
    backend_process = None
    asr_process = None
    
    try:
        # 启动后端服务
        if not args.skip_backend:
            backend_process = start_backend(args.backend_port)
        else:
            print("\n⏭️  跳过启动后端服务")
        
        # 启动ASR服务
        if not args.skip_asr:
            asr_process = start_asr_service(args.asr_port, args.model)
        else:
            print("\n⏭️  跳过启动ASR服务")
        
        # 等待服务完全就绪
        print("\n⏳ 等待服务就绪...")
        time.sleep(3)
        
        # 运行测试
        backend_url = f"http://localhost:{args.backend_port}"
        test_success = run_tests(
            backend_url=backend_url,
            audio_file=args.audio_file,
            model=args.model,
            language=args.language,
            test_type=args.test,
            character=args.character
        )
        
        # 输出结果
        print("\n" + "="*60)
        if test_success:
            print("✅ 测试完成")
        else:
            print("❌ 测试失败")
        print("="*60)
        
        # 如果设置了保持运行，等待用户中断
        if args.keep_running:
            print("\n💡 服务保持运行中，按 Ctrl+C 停止...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        
        return 0 if test_success else 1
        
    except Exception as e:
        print(f"\n💥 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # 清理进程
        if not args.keep_running:
            print("\n🧹 清理进程...")
            for process in processes:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    try:
                        process.kill()
                    except:
                        pass
            print("✅ 清理完成")


if __name__ == "__main__":
    sys.exit(main())

