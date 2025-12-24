#!/usr/bin/env python3
"""
一键启动脚本 - NeRFFaceSpeech 开发者模式
运行方式: python start.py
功能: 自动设置环境变量，启动后端和前端服务器，打开浏览器
"""
import subprocess
import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

# 获取当前脚本所在目录
CURRENT_DIR = Path(__file__).parent.resolve()

# 导入配置以获取API环境的Python路径
sys.path.insert(0, str(CURRENT_DIR))
API_CONDA_PYTHON = None
try:
    from config import API_CONDA_PYTHON
except (ImportError, AttributeError):
    # 如果导入失败或属性不存在，尝试从项目路径构造
    PROJECT_ROOT = CURRENT_DIR.parent
    api_env_python = PROJECT_ROOT / "environment" / "api" / "bin" / "python"
    if api_env_python.exists():
        API_CONDA_PYTHON = api_env_python
    else:
        API_CONDA_PYTHON = Path(sys.executable)

def set_environment_variables():
    """设置必要的环境变量（必须在启动服务器前设置）"""
    print("=" * 60)
    print("设置环境变量...")
    print("=" * 60)
    
    env_vars = {
        'PIP_INDEX_URL': 'https://pypi.tuna.tsinghua.edu.cn/simple',
        'TORCH_HOME': '/root/autodl-tmp/weights',
        'HF_ENDPOINT': 'https://hf-mirror.com',
        'HF_HOME': '/root/autodl-tmp/Hugging_Face'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        # 验证环境变量是否设置成功
        actual_value = os.environ.get(key)
        if actual_value == value:
            print(f"  ✅ {key} = {value}")
        else:
            print(f"  ❌ {key} 设置失败！期望: {value}, 实际: {actual_value}")
    
    # 确保权重目录存在
    torch_home = env_vars['TORCH_HOME']
    torch_hub_dir = Path(torch_home) / "hub" / "checkpoints"
    try:
        torch_hub_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ 模型缓存目录已准备: {torch_hub_dir}")
    except Exception as e:
        print(f"  ⚠️  无法创建模型缓存目录 {torch_hub_dir}: {e}")
    
    print("环境变量设置完成\n")

def check_environment(env_python: Path, env_name: str) -> tuple[bool, str]:
    """
    检查环境是否可用
    
    Returns:
        (是否可用, 错误信息)
    """
    if not env_python.exists():
        return False, f"Python可执行文件不存在: {env_python}"
    
    # 检查uvicorn是否可用
    try:
        result = subprocess.run(
            [str(env_python), "-m", "uvicorn", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return False, f"uvicorn不可用: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, "检查uvicorn时超时"
    except Exception as e:
        return False, f"检查环境时出错: {str(e)}"
    
    return True, ""

def start_backend_server():
    """启动后端服务器（FastAPI）"""
    print("=" * 60)
    print("启动后端服务器...")
    print("=" * 60)
    
    # 优先使用API环境的Python
    api_python = None
    env_error = None
    
    if API_CONDA_PYTHON:
        print(f"检查API环境: {API_CONDA_PYTHON}")
        is_available, error_msg = check_environment(API_CONDA_PYTHON, "API")
        if is_available:
            api_python = API_CONDA_PYTHON
            print(f"✅ API环境可用: {api_python}")
        else:
            env_error = error_msg
            print(f"❌ API环境检查失败: {error_msg}")
            print(f"   环境路径: {API_CONDA_PYTHON}")
            if API_CONDA_PYTHON.parent.parent.exists():
                print(f"   环境目录存在: {API_CONDA_PYTHON.parent.parent}")
            else:
                print(f"   环境目录不存在: {API_CONDA_PYTHON.parent.parent}")
    
    # 如果API环境不可用，尝试使用当前Python
    if api_python is None:
        current_python = Path(sys.executable)
        print(f"\n检查当前Python环境: {current_python}")
        is_available, error_msg = check_environment(current_python, "当前")
        if is_available:
            api_python = current_python
            print(f"✅ 当前Python环境可用: {api_python}")
        else:
            print(f"❌ 当前Python环境检查失败: {error_msg}")
            print("\n" + "=" * 60)
            print("❌ 错误: 无法启动后端服务器")
            print("=" * 60)
            if env_error:
                print(f"\nAPI环境错误: {env_error}")
            print(f"当前Python错误: {error_msg}")
            print("\n解决方案:")
            print("1. 确保API环境存在: PROJECT_ROOT/environment/api")
            print("2. 或在当前环境安装uvicorn: pip install uvicorn fastapi")
            print("=" * 60 + "\n")
            return None
    
    # 构建后端启动命令
    # 注意：--reload 会监视整个目录，如果文件太多可能超出系统限制
    # 使用 --reload-dir 只监视源代码目录，避免监视数据库、视频等大型目录
    # 构建后端启动命令
    # 注意：如果遇到 "OS file watch limit reached" 错误，说明文件太多超出了系统 inotify 限制
    # 解决方案：禁用 --reload（服务器环境通常不需要热重载）
    # 如果确实需要热重载，可以增加系统限制：echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p
    backend_cmd = [
        str(api_python), "-m", "uvicorn",
        "main:app",
        # "--reload",  # 已禁用，避免文件监视限制问题
        # "--reload-dir", str(CURRENT_DIR),  # 已禁用
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    
    # 设置工作目录和环境变量
    env = os.environ.copy()
    
    # 如果遇到文件监视限制，可以临时禁用 reload 模式（注释掉上面的 --reload 和 --reload-dir）
    # 或者增加系统的 inotify 限制：
    # echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf && sudo sysctl -p
    
    # 在后台启动后端服务器
    backend_process = subprocess.Popen(
        backend_cmd,
        cwd=str(CURRENT_DIR),
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
    
    log_thread = threading.Thread(target=log_backend_output, daemon=True)
    log_thread.start()
    
    # 等待后端启动并检查是否成功
    print("等待后端服务器启动...")
    
    # 多次尝试检查，因为服务器启动可能需要一些时间
    max_retries = 5
    backend_ready = False
    for i in range(max_retries):
        time.sleep(2 if i == 0 else 1)  # 第一次等待2秒，后续每次1秒
        
        try:
            import requests
            response = requests.get("http://localhost:8000/docs", timeout=3)
            if response.status_code == 200:
                print("✅ 后端服务器启动成功！")
                print("   后端地址: http://localhost:8000/")
                print("   API文档: http://localhost:8000/docs\n")
                backend_ready = True
                break
        except requests.exceptions.ConnectionError:
            # 连接错误说明服务器可能还在启动中，继续等待
            if i < max_retries - 1:
                continue
        except ImportError:
            # 如果没有requests库，使用简单的方式检查（检查进程是否还在运行）
            if backend_process.poll() is None:
                # 进程还在运行，假设启动成功
                print("✅ 后端服务器进程正在运行（无法验证HTTP响应，建议安装requests库）")
                print("   后端地址: http://localhost:8000/\n")
                backend_ready = True
            break
        except Exception as e:
            # 其他错误，可能是服务器还在启动中
            if i < max_retries - 1:
                continue
            # 最后一次尝试失败
            print(f"⚠️  后端服务器检查失败: {e}")
            print("   服务器可能仍在启动中，请稍候...\n")
    
    if not backend_ready:
        # 检查进程是否还在运行
        if backend_process.poll() is None:
            print("ℹ️  后端服务器进程正在运行，但HTTP检查未通过")
            print("   后端地址: http://localhost:8000/")
            print("   如果无法访问，请检查后端日志\n")
        else:
            print("⚠️  后端服务器进程可能已退出，请检查日志\n")
    
    return backend_process

def start_frontend_server():
    """启动前端服务器"""
    print("=" * 60)
    print("启动前端服务器...")
    print("=" * 60)
    
    frontend_cmd = [sys.executable, "simple_web.py"]
    
    # 设置工作目录
    env = os.environ.copy()
    
    # 在后台启动前端服务器
    frontend_process = subprocess.Popen(
        frontend_cmd,
        cwd=str(CURRENT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # 实时输出前端日志
    def log_frontend_output():
        for line in frontend_process.stdout:
            print(f"[前端] {line.rstrip()}")
    
    log_thread = threading.Thread(target=log_frontend_output, daemon=True)
    log_thread.start()
    
    # 等待前端启动
    print("等待前端服务器启动...")
    time.sleep(2)
    
    print("✅ 前端服务器启动成功！")
    print("   前端地址: http://localhost:7860/\n")
    
    return frontend_process

def open_browser():
    """打开浏览器显示选择页面"""
    url = "http://localhost:7860/start.html"
    print("=" * 60)
    print("打开浏览器...")
    print(f"访问地址: {url}")
    print("=" * 60)
    
    # 等待服务器完全启动
    time.sleep(1)
    
    try:
        webbrowser.open(url)
        print(f"✅ 浏览器已打开: {url}\n")
    except Exception as e:
        print(f"⚠️  无法自动打开浏览器: {e}")
        print(f"   请手动访问: {url}\n")

def enable_network_acceleration():
    """开启学术加速（读取配置文件并手动设置环境变量）"""
    print("=" * 60)
    print("开启学术加速...")
    print("=" * 60)
    
    network_turbo_file = Path("/etc/network_turbo")
    if not network_turbo_file.exists():
        print("⚠️  学术加速配置文件不存在: /etc/network_turbo")
        print("   （不影响使用，但网络可能较慢）\n")
        return
    
    try:
        # 读取配置文件内容
        with open(network_turbo_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析并设置环境变量
        # 通常格式是 export VAR=value 或 VAR=value
        # 可能包含 && 连接符，需要先分割
        env_vars_set = []
        
        # 先按 && 分割，处理可能的多命令情况
        lines = content.split('\n')
        for original_line in lines:
            line = original_line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 如果包含 &&，只处理第一个命令（通常是环境变量设置）
            if '&&' in line:
                # 只取第一个命令部分
                line = line.split('&&', 1)[0].strip()
            
            # 移除 export 关键字（如果有）
            line = line.replace('export', '').strip()
            
            # 解析 VAR=value 格式
            if '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    var_value = parts[1].strip()
                    # 移除可能的引号
                    var_value = var_value.strip('"').strip("'")
                    # 移除末尾可能的空格和特殊字符
                    var_value = var_value.rstrip().rstrip(';')
                    
                    # 处理代理相关变量
                    if 'proxy' in var_name.lower():
                        # no_proxy 变量不需要 URL 格式，它是一个逗号分隔的域名/IP列表，这是正常格式
                        if var_name.lower() == 'no_proxy':
                            # no_proxy 格式：逗号分隔的域名/IP列表，这是正常的，不需要验证URL格式
                            # 只检查是否包含非法字符（如 &&）
                            if '&&' in var_value or ';' in var_value:
                                # 移除 && 和 ; 后面的内容
                                var_value = var_value.split('&&')[0].split(';')[0].strip()
                        else:
                            # 其他代理变量（http_proxy, https_proxy等）需要URL格式
                            if var_value and not var_value.startswith(('http://', 'https://', 'socks5://')):
                                print(f"  ⚠️  跳过格式错误的代理配置: {var_name}={var_value[:50]}...")
                                continue
                            # 检查是否包含非法字符（如 &&）
                            if '&&' in var_value or ';' in var_value:
                                # 移除 && 和 ; 后面的内容
                                var_value = var_value.split('&&')[0].split(';')[0].strip()
                    
                    # 设置环境变量
                    os.environ[var_name] = var_value
                    env_vars_set.append(f"{var_name}={var_value[:80]}")  # 只显示前80个字符
        
        if env_vars_set:
            print("✅ 学术加速已开启，已设置以下环境变量:")
            for var in env_vars_set:
                print(f"   {var}")
            print()
        else:
            print("⚠️  学术加速配置文件存在但未解析到环境变量\n")
            
        # 同时执行 source 命令以确保脚本中的其他操作也执行
        result = subprocess.run(
            ["bash", "-c", "source /etc/network_turbo && echo '学术加速脚本已执行'"],
            capture_output=True,
            text=True,
            timeout=5,
            env=os.environ.copy()  # 传递当前环境变量
        )
        
    except FileNotFoundError:
        print("⚠️  学术加速配置文件不存在\n")
    except Exception as e:
        print(f"⚠️  开启学术加速时出错: {e}（但不影响使用）\n")

def init_all_databases():
    """初始化所有数据库（在启动服务器前必须完成）"""
    print("=" * 60)
    print("初始化数据库...")
    print("=" * 60)
    
    all_success = True
    errors = []
    
    try:
        # 1. 初始化设置数据库
        print("\n[1/3] 初始化设置数据库...")
        try:
            from database.settings_db import init_database, DB_FILE as SETTINGS_DB_FILE, DB_DIR, get_all_settings
            init_database()  # 强制初始化（即使文件存在也会验证表结构）
            if SETTINGS_DB_FILE.exists():
                settings = get_all_settings()
                print(f"  ✅ 设置数据库初始化成功: {SETTINGS_DB_FILE}")
                print(f"  ✅ 当前设置数量: {len(settings)}")
            else:
                raise FileNotFoundError(f"设置数据库文件未创建: {SETTINGS_DB_FILE}")
        except Exception as e:
            error_msg = f"设置数据库初始化失败: {e}"
            print(f"  ❌ {error_msg}")
            errors.append(error_msg)
            all_success = False
            import traceback
            traceback.print_exc()
        
        # 2. 初始化视频记录数据库
        print("\n[2/3] 初始化视频记录数据库...")
        try:
            from database.video_records_db import init_database, DB_FILE as VIDEO_RECORDS_DB_FILE
            init_database()  # 强制初始化
            if VIDEO_RECORDS_DB_FILE.exists():
                print(f"  ✅ 视频记录数据库初始化成功: {VIDEO_RECORDS_DB_FILE}")
            else:
                raise FileNotFoundError(f"视频记录数据库文件未创建: {VIDEO_RECORDS_DB_FILE}")
        except Exception as e:
            error_msg = f"视频记录数据库初始化失败: {e}"
            print(f"  ❌ {error_msg}")
            errors.append(error_msg)
            all_success = False
            import traceback
            traceback.print_exc()
        
        # 3. 初始化聊天数据库
        print("\n[3/3] 初始化聊天数据库...")
        try:
            from database.chat_db import init_database, DB_FILE as CHAT_DB_FILE
            init_database()  # 强制初始化
            if CHAT_DB_FILE.exists():
                print(f"  ✅ 聊天数据库初始化成功: {CHAT_DB_FILE}")
            else:
                raise FileNotFoundError(f"聊天数据库文件未创建: {CHAT_DB_FILE}")
        except Exception as e:
            error_msg = f"聊天数据库初始化失败: {e}"
            print(f"  ❌ {error_msg}")
            errors.append(error_msg)
            all_success = False
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        if all_success:
            print("✅ 所有数据库初始化成功！")
            print("=" * 60 + "\n")
            return True
        else:
            print("❌ 数据库初始化失败！")
            print("\n错误详情:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
            print("\n⚠️  数据库初始化失败，服务器将无法正常启动")
            print("   网页可能无法打开或功能异常")
            print("=" * 60 + "\n")
            return False
            
    except Exception as e:
        print(f"\n❌ 数据库初始化过程出错: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60 + "\n")
        return False


def verify_database_access():
    """核验数据库访问，确保网页可以正常使用"""
    print("=" * 60)
    print("核验数据库访问状态...")
    print("=" * 60)
    
    all_success = True
    errors = []
    
    try:
        # 1. 核验设置数据库访问
        print("\n[1/3] 核验设置数据库访问...")
        try:
            from database.settings_db import get_all_settings, DB_FILE as SETTINGS_DB_FILE
            from database.settings_db import get_setting, set_setting
            
            # 检查文件是否存在
            if not SETTINGS_DB_FILE.exists():
                raise FileNotFoundError(f"设置数据库文件不存在: {SETTINGS_DB_FILE}")
            
            # 尝试读取所有设置（模拟网页的 fetchSettings）
            settings = get_all_settings()
            if not settings:
                raise ValueError("设置数据库为空，无法读取设置")
            
            # 验证关键设置是否存在
            required_keys = ["nerf_theme", "nerf_font", "nerf_font_size"]
            missing_keys = [key for key in required_keys if key not in settings]
            if missing_keys:
                raise ValueError(f"缺少必需的设置项: {missing_keys}")
            
            # 尝试读取单个设置
            theme = get_setting("nerf_theme")
            if not theme:
                raise ValueError("无法读取设置项 'nerf_theme'")
            
            # 尝试写入设置（测试写入权限）
            test_key = "__test_write_access__"
            try:
                set_setting(test_key, "test")
                get_setting(test_key)  # 验证写入成功
                # 清理测试数据
                import sqlite3
                conn = sqlite3.connect(str(SETTINGS_DB_FILE), check_same_thread=False)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM settings WHERE key = ?", (test_key,))
                conn.commit()
                conn.close()
            except Exception as e:
                raise ValueError(f"数据库写入权限测试失败: {e}")
            
            print(f"  ✅ 设置数据库访问正常")
            print(f"  ✅ 当前设置数量: {len(settings)}")
            print(f"  ✅ 关键设置项: {', '.join([f'{k}={settings[k]}' for k in required_keys])}")
            
        except Exception as e:
            error_msg = f"设置数据库访问核验失败: {e}"
            print(f"  ❌ {error_msg}")
            errors.append(error_msg)
            all_success = False
            import traceback
            traceback.print_exc()
        
        # 2. 核验视频记录数据库访问
        print("\n[2/3] 核验视频记录数据库访问...")
        try:
            from database.video_records_db import DB_FILE as VIDEO_RECORDS_DB_FILE
            from database.video_records_db import list_generation_records
            
            # 检查文件是否存在
            if not VIDEO_RECORDS_DB_FILE.exists():
                raise FileNotFoundError(f"视频记录数据库文件不存在: {VIDEO_RECORDS_DB_FILE}")
            
            # 尝试查询记录（模拟网页的 loadHistory）
            records = list_generation_records(limit=10)
            # 即使没有记录，查询也应该成功（返回空列表）
            if records is None:
                raise ValueError("查询视频记录返回 None，数据库可能有问题")
            
            print(f"  ✅ 视频记录数据库访问正常")
            print(f"  ✅ 当前记录数量: {len(records)}")
            
        except Exception as e:
            error_msg = f"视频记录数据库访问核验失败: {e}"
            print(f"  ❌ {error_msg}")
            errors.append(error_msg)
            all_success = False
            import traceback
            traceback.print_exc()
        
        # 3. 核验聊天数据库访问
        print("\n[3/3] 核验聊天数据库访问...")
        try:
            from database.chat_db import DB_FILE as CHAT_DB_FILE
            from database.chat_db import list_chat_sessions
            
            # 检查文件是否存在
            if not CHAT_DB_FILE.exists():
                raise FileNotFoundError(f"聊天数据库文件不存在: {CHAT_DB_FILE}")
            
            # 尝试查询会话列表（模拟网页的 loadChatHistory）
            sessions = list_chat_sessions(limit=10)
            # 即使没有会话，查询也应该成功（返回空列表）
            if sessions is None:
                raise ValueError("查询聊天会话返回 None，数据库可能有问题")
            
            print(f"  ✅ 聊天数据库访问正常")
            print(f"  ✅ 当前会话数量: {len(sessions)}")
            
        except Exception as e:
            error_msg = f"聊天数据库访问核验失败: {e}"
            print(f"  ❌ {error_msg}")
            errors.append(error_msg)
            all_success = False
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        if all_success:
            print("✅ 所有数据库访问核验通过！")
            print("✅ 网页可以正常使用数据库功能")
            print("=" * 60 + "\n")
            return True
        else:
            print("❌ 数据库访问核验失败！")
            print("\n错误详情:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
            print("\n⚠️  数据库访问异常，网页可能无法正常加载或功能受限")
            print("=" * 60 + "\n")
            return False
            
    except Exception as e:
        print(f"\n❌ 数据库访问核验过程出错: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60 + "\n")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("NeRFFaceSpeech 开发者模式启动")
    print("=" * 60 + "\n")
    
    try:
        # 0. 开启学术加速
        enable_network_acceleration()
        
        # 1. 设置环境变量
        set_environment_variables()
        
        # 2. 初始化所有数据库（在启动服务器前必须完成，如果失败则阻止启动）
        if not init_all_databases():
            print("\n" + "=" * 60)
            print("❌ 数据库初始化失败，程序退出")
            print("=" * 60)
            print("\n解决方案:")
            print("1. 检查数据库目录权限")
            print("2. 检查磁盘空间")
            print("3. 查看上述错误信息")
            print("=" * 60 + "\n")
            return
        
        # 3. 核验数据库访问状态（确保网页可以正常使用）
        if not verify_database_access():
            print("\n" + "=" * 60)
            print("⚠️  数据库访问核验失败，但将继续启动服务器")
            print("=" * 60)
            print("\n提示:")
            print("1. 网页可能无法正常加载设置")
            print("2. 某些功能可能受限")
            print("3. 建议修复数据库问题后重新启动")
            print("=" * 60 + "\n")
            # 不阻止启动，但给出警告
        
        # 4. 启动后端服务器
        backend_process = start_backend_server()
        if backend_process is None:
            # 后端启动失败，退出
            print("启动失败，程序退出")
            return
        
        # 5. 启动前端服务器
        frontend_process = start_frontend_server()
        
        # 6. 等待服务器就绪后再打开浏览器
        print("=" * 60)
        print("等待服务器就绪...")
        print("=" * 60)
        time.sleep(3)  # 等待3秒确保服务器已启动
        
        # 7. 打开浏览器
        open_browser()
        
        # 8. 保持运行
        print("=" * 60)
        print("服务器运行中...")
        print("按 Ctrl+C 停止所有服务器")
        print("=" * 60 + "\n")
        
        # 等待用户中断
        try:
            while True:
                time.sleep(1)
                # 检查进程是否还在运行
                if backend_process.poll() is not None:
                    print("\n❌ 后端服务器意外停止")
                    break
                if frontend_process.poll() is not None:
                    print("\n❌ 前端服务器意外停止")
                    break
        except KeyboardInterrupt:
            print("\n\n正在停止服务器...")
            
    except KeyboardInterrupt:
        print("\n\n正在停止服务器...")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理进程
        try:
            if 'backend_process' in locals():
                backend_process.terminate()
                backend_process.wait(timeout=5)
        except:
            pass
        
        try:
            if 'frontend_process' in locals():
                frontend_process.terminate()
                frontend_process.wait(timeout=5)
        except:
            pass
        
        print("✅ 所有服务器已停止")
        print("感谢使用 NeRFFaceSpeech！")

if __name__ == "__main__":
    main()

