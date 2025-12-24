# 从 fastapi_server/main.py 迁移而来
# 已更新导入路径以使用 shared 模块
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Body, BackgroundTasks, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import uuid
import os
from pathlib import Path
import collections
from datetime import datetime
import logging
import sys
import warnings
import traceback
import threading
import subprocess

from shared.utils.run_llm_talk import generate_audio
from shared.utils.run_nerffacespeech import generate_video
from shared.utils.run_chat import chat_with_llm, get_llm_only
from shared.utils.run_training import start_training, get_training_status, list_training_tasks, stop_training
from shared.utils.character_training import (
    process_video_for_training,
    get_character_training_status,
    list_characters,
)
from shared.config import OUTPUT_VIDEO_DIR, OUTPUT_AUDIO_DIR, MODEL_DIR, WEBUI_DIR, VIDEOS_STORAGE_DIR, AUDIOS_STORAGE_DIR, TEXTS_STORAGE_DIR, DATA_DIR, TRAINING_DATASET_DIR, API_CONDA_ENV, NERF_CONDA_ENV, LLM_CONDA_ENV, LLM_CONDA_PYTHON, ASR_SERVICE_SCRIPT, ASR_SERVICE_URL

# 尝试导入 ASR 客户端（如果可用）
try:
    from shared.utils.asr_client import (
        check_asr_service_health,
        transcribe_audio_via_service,
        transcribe_audio_file_via_service,
        transcribe_base64_audio_via_service,
        transcribe_base64_audio_via_subprocess
    )
    from shared.config import ASR_SERVICE_URL
    ASR_SERVICE_AVAILABLE = True
except ImportError:
    ASR_SERVICE_AVAILABLE = False
    ASR_SERVICE_URL = None
    # 即使 ASR 服务客户端导入失败，也要尝试导入 subprocess 版本的函数
    # 因为它不依赖 ASR 服务，而是直接调用 llm_talk 环境
    try:
        from shared.utils.asr_client import transcribe_base64_audio_via_subprocess
    except ImportError:
        transcribe_base64_audio_via_subprocess = None
    # 注意：此时 add_log 还未定义，日志将在 startup_event 中记录
from shared.database.settings_db import get_setting, set_setting, get_all_settings, DB_DIR
from shared.database.video_records_db import add_video_record, list_generation_records, add_generation_record, delete_generation_record, get_generation_record
from shared.database.chat_db import (
    create_chat_session, add_chat_message, get_chat_session,
    list_chat_sessions, get_chat_messages, delete_chat_session,
    update_chat_message_video_path
)
from shared.database.video_records_db import (
    create_or_update_task, get_task, get_running_task,
    list_tasks as list_video_tasks, delete_task as delete_video_task
)
import sqlite3
import shutil
import time
import tempfile
from typing import Tuple

# 日志缓冲系统
LOG_BUFFER_SIZE = 1000  # 最大保存 1000 条日志（自动删除多余的，节省内存）
LOG_BUFFER = collections.deque(maxlen=LOG_BUFFER_SIZE)
DEBUG_LOG_BUFFER = collections.deque(maxlen=LOG_BUFFER_SIZE)

# 任务管理系统
TASKS: Dict[str, Dict] = {}  # 存储运行中的任务 {unique_id: {status, start_time, ...}}
TASKS_LOCK = threading.Lock()  # 保护TASKS字典的锁

# ASR 服务进程管理
ASR_SERVICE_PROCESS = None  # 存储 ASR 服务进程


# 自定义日志处理器，将日志添加到缓冲区
class BufferLogHandler(logging.Handler):
    """将日志输出添加到缓冲区的处理器"""
    
    def emit(self, record):
        try:
            # 确定日志级别
            level_map = {
                logging.DEBUG: "debug",
                logging.INFO: "info",
                logging.WARNING: "warning",
                logging.ERROR: "error",
                logging.CRITICAL: "error",
            }
            level = level_map.get(record.levelno, "info")
            
            # 格式化日志消息，保持完整格式
            message = record.getMessage()
            
            # 对于所有日志，保持完整的原始格式，不简化
            # 这样可以确保所有输出都能完整显示
            # 如果是Uvicorn的日志，保持原始格式
            if record.name.startswith("uvicorn"):
                message = record.getMessage()
            # 对于其他日志，也保持原始格式，但可以添加模块名前缀
            elif record.name and record.name != "root":
                # 只在模块名不是root时添加，避免重复
                message = record.getMessage()
            
            # 添加到日志缓冲区
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message
            }
            
            if level == "debug":
                DEBUG_LOG_BUFFER.append(log_entry)
            else:
                LOG_BUFFER.append(log_entry)
        except Exception:
            # 防止日志处理出错导致程序崩溃
            pass


# 配置日志系统
def setup_logging():
    """配置日志系统，同时输出到控制台和缓冲区"""
    # 创建控制台处理器（输出到VSCode终端）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                         datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    
    # 创建自定义缓冲区处理器（用于网页显示）
    buffer_handler = BufferLogHandler()
    buffer_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    buffer_handler.setFormatter(formatter)
    
    # 配置Uvicorn的日志记录器（同时输出到控制台和缓冲区）
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)
    # 添加控制台处理器（如果还没有）
    if not any(isinstance(h, logging.StreamHandler) for h in uvicorn_logger.handlers):
        uvicorn_logger.addHandler(console_handler)
    # 添加缓冲区处理器
    if not any(isinstance(h, BufferLogHandler) for h in uvicorn_logger.handlers):
        uvicorn_logger.addHandler(buffer_handler)
    
    # 配置Uvicorn的access日志（HTTP请求日志）
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in access_logger.handlers):
        access_logger.addHandler(console_handler)
    if not any(isinstance(h, BufferLogHandler) for h in access_logger.handlers):
        access_logger.addHandler(buffer_handler)
    
    # 配置FastAPI的日志记录器
    fastapi_logger = logging.getLogger("fastapi")
    fastapi_logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in fastapi_logger.handlers):
        fastapi_logger.addHandler(console_handler)
    if not any(isinstance(h, BufferLogHandler) for h in fastapi_logger.handlers):
        fastapi_logger.addHandler(buffer_handler)
    
    # 配置根日志记录器（捕获所有模块的日志）
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # 设置为 INFO 以捕获所有信息
    # 清除现有的处理器，避免重复
    root_logger.handlers.clear()
    # 添加控制台和缓冲区处理器
    root_logger.addHandler(console_handler)
    root_logger.addHandler(buffer_handler)
    
    # 配置所有子模块的日志记录器，确保捕获所有输出
    # 特别是捕获 warnings 模块的输出
    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: root_logger.warning(
        f"{filename}:{lineno}: {category.__name__}: {message}"
    )


# 初始化日志系统
setup_logging()

app = FastAPI()


def start_asr_service_background():
    """在后台启动 ASR 服务"""
    global ASR_SERVICE_PROCESS
    
    # 如果 ASR_SERVICE_URL 未配置，使用默认值
    asr_service_url = ASR_SERVICE_URL if ASR_SERVICE_URL else "http://localhost:8002"
    
    # 检查 ASR 服务是否已经在运行
    try:
        import requests
        response = requests.get(f"{asr_service_url}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get("model_loaded"):
                add_log(f"ASR 服务已在运行: {asr_service_url}", "info")
                return
    except:
        pass  # 服务未运行，继续启动
    
    # 检查 ASR 服务脚本是否存在
    if not ASR_SERVICE_SCRIPT.exists():
        add_log(f"ASR 服务脚本不存在: {ASR_SERVICE_SCRIPT}，跳过启动", "warning")
        return
    
    # 检查 Python 环境是否存在，优先使用 LLM conda 环境的 Python
    asr_python = LLM_CONDA_PYTHON if LLM_CONDA_PYTHON.exists() else Path(sys.executable)
    
    # 获取 conda 环境的 bin 目录（用于 PATH）
    llm_env_bin = LLM_CONDA_ENV / "bin" if LLM_CONDA_ENV.exists() else None
    
    add_log(f"正在启动 ASR 服务 (使用 Python: {asr_python})...", "info")
    if llm_env_bin:
        add_log(f"使用 Conda 环境: {LLM_CONDA_ENV}", "info")
    
    try:
        # 构建启动命令
        asr_cmd = [
            str(asr_python),
            str(ASR_SERVICE_SCRIPT),
            "--host", "0.0.0.0",
            "--port", "8002",
            "--workers", "1",
            "--model", "base"
        ]
        
        # 设置环境变量，确保使用 conda 环境的工具
        env = os.environ.copy()
        project_root = Path(__file__).parent.parent.parent.resolve()
        env["PYTHONPATH"] = str(project_root)
        
        # 将 conda 环境的 bin 目录添加到 PATH 的最前面，确保优先使用环境中的工具
        if llm_env_bin and llm_env_bin.exists():
            conda_bin_path = str(llm_env_bin)
            current_path = env.get("PATH", "")
            # 将 conda 环境的 bin 目录放在 PATH 最前面（使用 os.pathsep 确保跨平台兼容）
            path_sep = os.pathsep
            env["PATH"] = f"{conda_bin_path}{path_sep}{current_path}" if current_path else conda_bin_path
            add_log(f"已将 Conda 环境 bin 目录添加到 PATH: {conda_bin_path}", "info")
        
        # 启动 ASR 服务进程
        ASR_SERVICE_PROCESS = subprocess.Popen(
            asr_cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 在后台线程中输出 ASR 服务日志
        def log_asr_output():
            if ASR_SERVICE_PROCESS and ASR_SERVICE_PROCESS.stdout:
                for line in ASR_SERVICE_PROCESS.stdout:
                    add_log(f"[ASR服务] {line.rstrip()}", "info")
        
        log_thread = threading.Thread(target=log_asr_output, daemon=True)
        log_thread.start()
        
        add_log("ASR 服务进程已启动，正在后台加载模型...", "info")
        
        # 在后台等待服务就绪（不阻塞主线程）
        def wait_for_asr_service():
            max_retries = 60  # 最多等待 60 秒
            for i in range(max_retries):
                time.sleep(1)
                try:
                    import requests
                    response = requests.get(f"{asr_service_url}/health", timeout=2)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("model_loaded"):
                            add_log(f"✅ ASR 服务启动成功: {asr_service_url}", "success")
                            return
                except:
                    if i < max_retries - 1:
                        continue
            
            # 检查进程是否还在运行
            if ASR_SERVICE_PROCESS and ASR_SERVICE_PROCESS.poll() is None:
                add_log("⚠️ ASR 服务进程正在运行，但模型可能仍在加载中", "warning")
            else:
                add_log("❌ ASR 服务启动失败，将使用直接调用方式", "warning")
        
        wait_thread = threading.Thread(target=wait_for_asr_service, daemon=True)
        wait_thread.start()
        
    except Exception as e:
        add_log(f"启动 ASR 服务失败: {e}", "error")
        import traceback
        add_log(traceback.format_exc(), "error")
        ASR_SERVICE_PROCESS = None

@app.on_event("startup")
async def startup_event():
    """应用启动事件，确保日志系统正常工作"""
    # 在启动时确保日志处理器已配置
    setup_logging()
    # 添加启动日志
    add_log("FastAPI应用启动完成", "success")
    add_log("[初始化] ASR 功能使用 subprocess 方式，直接调用封装好的环境", "info")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件，清理资源"""
    add_log("FastAPI应用正在关闭...", "info")


# 允许前端跨域访问 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 使用自定义端点提供视频、音频、文本文件，确保正确的MIME类型
@app.get("/videos/{filename}")
@app.head("/videos/{filename}")
async def serve_video(filename: str, request: Request):
    """提供视频文件，设置正确的MIME类型，支持Range请求（用于视频seek）"""
    from fastapi import HTTPException
    from fastapi.responses import Response
    import os
    
    file_path = VIDEOS_STORAGE_DIR / filename
    
    # 调试输出：显示路径信息
    print(f"[后端] [视频服务] 请求文件: {filename}")
    print(f"[后端] [视频服务] 存储目录: {VIDEOS_STORAGE_DIR}")
    print(f"[后端] [视频服务] 完整路径: {file_path}")
    print(f"[后端] [视频服务] 路径存在: {file_path.exists()}")
    print(f"[后端] [视频服务] 是文件: {file_path.is_file() if file_path.exists() else False}")
    
    if not file_path.exists() or not file_path.is_file():
        error_msg = f"Video file not found: {filename} (路径: {file_path})"
        print(f"[后端] [视频服务] ❌ {error_msg}")
        raise HTTPException(status_code=404, detail=error_msg)
    
    # 获取文件大小
    file_size = file_path.stat().st_size
    
    # 检查Range请求头（用于视频seek功能）
    range_header = request.headers.get("range") if request else None
    
    if range_header:
        # 解析Range头，格式通常是 "bytes=start-end"
        try:
            byte_start, byte_end = range_header.replace("bytes=", "").split("-")
            byte_start = int(byte_start) if byte_start else 0
            byte_end = int(byte_end) if byte_end else file_size - 1
            
            # 读取文件的指定范围
            with open(file_path, "rb") as f:
                f.seek(byte_start)
                data = f.read(byte_end - byte_start + 1)
            
            # 返回206 Partial Content响应
            headers = {
                "Content-Range": f"bytes {byte_start}-{byte_end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(data)),
                "Content-Type": "video/mp4",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length"
            }
            return Response(content=data, status_code=206, headers=headers)
        except (ValueError, IOError) as e:
            # 如果Range请求解析失败，返回完整文件
            pass
    
    # 如果没有Range请求，返回完整文件（支持HEAD和GET）
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length"
        }
    )

@app.get("/audios/{filename}")
@app.head("/audios/{filename}")
async def serve_audio(filename: str, request: Request):
    """提供音频文件，设置正确的MIME类型，支持Range请求"""
    from fastapi import HTTPException
    
    file_path = AUDIOS_STORAGE_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")
    
    media_type = "audio/wav" if filename.endswith('.wav') else "audio/mpeg"
    file_size = file_path.stat().st_size
    
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length"
        }
    )

@app.get("/texts/{filename}")
@app.head("/texts/{filename}")
async def serve_text(filename: str):
    """提供文本文件"""
    from fastapi import HTTPException
    
    file_path = TEXTS_STORAGE_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Text file not found: {filename}")
    
    return FileResponse(
        path=str(file_path),
        media_type="text/plain; charset=utf-8",
        headers={
            "Access-Control-Allow-Origin": "*"
        }
)

# 挂载 webui 静态文件
app.mount(
    "/webui",
    StaticFiles(directory=str(WEBUI_DIR)),
    name="webui"
)

# ---------------------------
# 根路径 - 后端输出页面（后端显示屏）
# ---------------------------
from fastapi.responses import HTMLResponse, FileResponse

@app.get("/", response_class=HTMLResponse)
def root():
    """后端根路径 - 显示后端输出页面（后端显示屏）"""
    logs_html_path = WEBUI_DIR / "logs.html"
    if logs_html_path.exists():
        with open(logs_html_path, "r", encoding="utf-8") as f:
            content = f.read()
            # 替换settings.js路径为/webui/settings.js，确保从根路径访问时能找到文件
            # 替换settings.js路径为/webui/settings.js
            content = content.replace('src="settings.js"', 'src="/webui/settings.js"')
            return HTMLResponse(content=content)
    
    # 如果 logs.html 不存在，返回简单的 API 信息页面
    return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>TFG_TALK_NeRFaceSpeech API - 后端显示屏</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 40px; background: #0b1020; color: #e5e7eb; }
                .container { max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.06); padding: 30px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.14); }
                h1 { color: #e5e7eb; }
                .endpoint { margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.03); border-left: 4px solid #3b82f6; }
                a { color: #60a5fa; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .link-box { margin-top: 20px; padding: 15px; background: rgba(59,130,246,0.2); border-radius: 8px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>TFG_TALK_NeRFaceSpeech API - 后端显示屏</h1>
                <p>这是后端服务器的主页。请访问 <a href="/webui/logs.html">/webui/logs.html</a> 查看后端输出。</p>
                <p>API 文档: <a href="/docs">/docs</a></p>
                <div class="link-box">
                    <strong>前端应用入口：</strong> 请访问 <a href="http://localhost:7860/" target="_blank">http://localhost:7860/</a> 使用前端功能（需要先运行 python simple_web.py）。
                </div>
                <h2>可用端点:</h2>
                <div class="endpoint"><strong>GET</strong> /models - 获取模型列表</div>
                <div class="endpoint"><strong>POST</strong> /generate_video - 生成视频</div>
                <div class="endpoint"><strong>POST</strong> /chat - 聊天对话</div>
                <div class="endpoint"><strong>POST</strong> /llm_only - 仅LLM问答</div>
                <div class="endpoint"><strong>POST</strong> /train/start - 启动训练</div>
                <div class="endpoint"><strong>GET</strong> /train/status/{task_id} - 查询训练状态</div>
                <div class="endpoint"><strong>GET</strong> /train/tasks - 列出训练任务</div>
                <div class="endpoint"><strong>POST</strong> /train/stop/{task_id} - 停止训练</div>
                <div class="endpoint"><strong>GET</strong> /logs - 获取日志输出</div>
            </div>
        </body>
        </html>
        """)

# ---------------------------
# 设置管理 API
# ---------------------------

@app.get("/api/settings")
async def get_settings_api():
    """获取所有设置（数据库未初始化时返回错误，阻止网页打开）"""
    try:
        settings = get_all_settings()
        return {"success": True, "data": settings}
    except (FileNotFoundError, ValueError) as e:
        # 数据库未初始化，返回错误（阻止网页打开）
        error_msg = f"数据库未初始化: {str(e)}。请先运行 start.py 初始化数据库。"
        add_log(error_msg, "error")
        return {
            "success": False,
            "error": str(e),
            "message": error_msg
        }
    except Exception as e:
        # 其他数据库错误，也返回错误（阻止网页打开）
        error_msg = f"获取设置失败: {str(e)}"
        add_log(error_msg, "error")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": error_msg
        }


@app.get("/api/settings/{key}")
async def get_setting_by_key_api(key: str):
    """获取指定设置"""
    value = get_setting(key)
    return {"success": True, "data": {"key": key, "value": value}}


@app.post("/api/settings/{key}")
async def update_setting_api(key: str, request: dict = Body(...)):
    """更新设置"""
    # 支持两种格式：{"value": "xxx"} 或直接传递字符串
    value = request.get("value") if isinstance(request, dict) else str(request)
    if value is None:
        # 如果没有value字段，尝试直接使用请求体作为值
        value = str(request)
    set_setting(key, str(value))
    return {"success": True, "message": f"Setting {key} updated"}


@app.post("/api/settings")
async def update_settings_api(settings: dict = Body(...)):
    """批量更新设置"""
    for key, value in settings.items():
        set_setting(key, str(value))
    return {"success": True, "message": "Settings updated"}


# ---------------------------
# 数据库管理 API
# ---------------------------

@app.get("/api/databases")
async def list_databases_api():
    """列出database文件夹下的所有数据库文件"""
    try:
        db_files = []
        if DB_DIR.exists():
            for file in DB_DIR.iterdir():
                if file.is_file() and file.suffix.lower() == '.db':
                    db_files.append({
                        "name": file.name,
                        "path": str(file),
                        "size": file.stat().st_size
                    })
        return {"success": True, "data": sorted(db_files, key=lambda x: x["name"])}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/databases/{db_name}/content")
async def get_database_content_api(db_name: str, table: str = None):
    """获取指定数据库的内容
    如果指定table，返回该表的数据；否则返回所有表的结构和数据
    """
    try:
        db_path = DB_DIR / db_name
        if not db_path.exists() or not db_path.suffix.lower() == '.db':
            return {"success": False, "error": f"数据库文件 {db_name} 不存在"}
        
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row  # 返回字典格式
        cursor = conn.cursor()
        
        result = {}
        
        if table:
            # 获取指定表的数据
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            result[table] = {
                "columns": columns,
                "rows": [dict(row) for row in rows]
            }
        else:
            # 获取所有表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table_name in tables:
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                result[table_name] = {
                    "columns": columns,
                    "rows": [dict(row) for row in rows]
                }
        
        conn.close()
        
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/databases/{db_name}/update")
async def update_database_content_api(db_name: str, update_data: dict = Body(...)):
    """更新数据库内容
    update_data格式: {
        "table": "table_name",
        "operation": "update|insert|delete",
        "where": {"key": "value"},  # 用于update和delete
        "data": {"key": "value"}  # 用于update和insert
    }
    """
    try:
        db_path = DB_DIR / db_name
        if not db_path.exists() or not db_path.suffix.lower() == '.db':
            return {"success": False, "error": f"数据库文件 {db_name} 不存在"}
        
        table = update_data.get("table")
        operation = update_data.get("operation")
        where_clause = update_data.get("where", {})
        data = update_data.get("data", {})
        
        if not table or not operation:
            return {"success": False, "error": "缺少必需参数: table 和 operation"}
        
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        cursor = conn.cursor()
        
        if operation == "update":
            if not where_clause or not data:
                return {"success": False, "error": "update操作需要where和data参数"}
            set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
            where_conditions = " AND ".join([f"{k} = ?" for k in where_clause.keys()])
            values = list(data.values()) + list(where_clause.values())
            cursor.execute(f"UPDATE {table} SET {set_clause} WHERE {where_conditions}", values)
            
        elif operation == "insert":
            if not data:
                return {"success": False, "error": "insert操作需要data参数"}
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            cursor.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", list(data.values()))
            
        elif operation == "delete":
            if not where_clause:
                return {"success": False, "error": "delete操作需要where参数"}
            where_conditions = " AND ".join([f"{k} = ?" for k in where_clause.keys()])
            cursor.execute(f"DELETE FROM {table} WHERE {where_conditions}", list(where_clause.values()))
        else:
            return {"success": False, "error": f"不支持的操作: {operation}"}
        
        conn.commit()
        conn.close()
        
        return {"success": True, "message": f"{operation}操作成功"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------
# 返回所有 pkl 模型
# ---------------------------
@app.get("/models")
def list_models():
    try:
        # 检查模型目录是否存在
        if not MODEL_DIR.exists():
            error_msg = f"模型目录不存在: {MODEL_DIR}\n请确保目录 /root/autodl-tmp/TFG_TALK_NeRFaceSpeech/NeRFFaceSpeech_Code/pretrained_networks/ 存在"
            add_log(error_msg, "error")
            return {"success": False, "error": error_msg}
        
        # 检查是否有 .pkl 文件
        try:
            files = os.listdir(str(MODEL_DIR))
        except PermissionError:
            error_msg = f"无权限访问模型目录: {MODEL_DIR}"
            add_log(error_msg, "error")
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"读取模型目录失败: {str(e)}\n目录路径: {MODEL_DIR}"
            add_log(error_msg, "error")
            return {"success": False, "error": error_msg}
        
        # 只查找 .pkl 文件，排除子目录
        models = []
        for f in files:
            file_path = os.path.join(str(MODEL_DIR), f)
            if f.endswith(".pkl") and os.path.isfile(file_path):
                models.append(f)
        
        if not models:
            # 列出目录中的所有文件（最多20个）用于调试
            file_list = ', '.join(files[:20]) if files else '(空)'
            error_msg = f"模型目录中没有找到 .pkl 文件\n目录路径: {MODEL_DIR}\n目录内容: {file_list}\n请确保 ffhq_1024.pkl 等模型文件在该目录中"
            add_log(error_msg, "error")
            return {"success": False, "error": error_msg}
        
        add_log(f"找到 {len(models)} 个模型文件: {', '.join(models)}", "info")
        return models
    except Exception as e:
        error_msg = f"获取模型列表失败: {str(e)}\n目录路径: {MODEL_DIR}"
        add_log(error_msg, "error")
        return {"success": False, "error": error_msg}


# ---------------------------
# 生成视频接口（后台任务模式）
# ---------------------------
@app.post("/generate_video")
def generate_video_api(
    background_tasks: BackgroundTasks,
    text: str = Body(..., embed=True),
    character: str = Body(..., embed=True),
    model_name: str = Body(..., embed=True),
    is_from_chat: bool = Body(False, embed=True),
):
    """
    提交视频生成任务，立即返回任务ID
    实际生成在后台进行，可以通过 /generate_video/status/{unique_id} 查询状态
    
    【推荐使用此接口】异步模式，不会超时，可以显示进度
    
    Args:
        is_from_chat: 如果为True，表示来自聊天对话，视频将只保存到chat_messages表，不保存到video_records数据库
    """
    unique_id = str(uuid.uuid4())
    start_time_str = datetime.now().isoformat()
    
    # 初始化任务状态（内存）
    with TASKS_LOCK:
        TASKS[unique_id] = {
            "status": "pending",
            "start_time": time.time(),
            "text": text,
            "character": character,
            "model_name": model_name,
            "is_from_chat": is_from_chat  # 保存标识，用于后续判断
        }
    
    # 如果不是来自聊天，才保存任务到video_records数据库
    if not is_from_chat:
        create_or_update_task(
            task_id=unique_id,
            status="pending",
            text=text,
            character=character,
            model_name=model_name,
            config={"text": text, "character": character, "model_name": model_name},
            start_time=start_time_str
        )
    
    # 在后台运行生成任务
    background_tasks.add_task(run_video_generation_task, unique_id, text, character, model_name, is_from_chat)
    
    add_log(f"[任务] {unique_id}: 视频生成任务已提交 (is_from_chat={is_from_chat})", "info")
    
    # 立即返回任务ID
    return {
        "success": True,
        "unique_id": unique_id,
        "status": "pending",
        "message": "任务已提交，正在后台生成中..."
    }


@app.post("/generate_video_sync")
def generate_video_sync_api(
    text: str = Body(..., embed=True),
    character: str = Body(..., embed=True),
    model_name: str = Body(..., embed=True),
):
    """
    【同步模式】直接等待视频生成完成并返回视频路径
    
    ⚠️ 警告：此接口会阻塞等待 5-15 分钟，存在以下风险：
    1. HTTP 超时：浏览器/服务器/代理可能有超时限制（通常 30 秒 - 5 分钟）
    2. 用户体验：用户需要长时间等待，无法看到进度
    3. 服务器资源：占用工作线程长时间
    
    【推荐使用 /generate_video（异步模式）】
    
    如果需要使用此接口，请确保：
    - uvicorn 启动时设置 timeout_keep_alive 和 timeout_graceful_shutdown 足够大
    - 前端 fetch 请求设置足够长的超时时间
    - 前面没有 nginx 等反向代理（或配置了长超时）
    
    示例 uvicorn 启动命令：
    uvicorn main:app --timeout-keep-alive 900 --timeout-graceful-shutdown 900
    """
    unique_id = str(uuid.uuid4())
    start_time_str = datetime.now().isoformat()
    
    # 初始化任务状态（内存）
    with TASKS_LOCK:
        TASKS[unique_id] = {
            "status": "pending",
            "start_time": time.time(),
            "text": text,
            "character": character,
            "model_name": model_name
        }
    
    # 保存任务到数据库
    create_or_update_task(
        task_id=unique_id,
        status="pending",
        text=text,
        character=character,
        model_name=model_name,
        config={"text": text, "character": character, "model_name": model_name},
        start_time=start_time_str
    )
    
    add_log(f"[任务] {unique_id}: 同步模式视频生成任务开始（将阻塞等待完成）", "info")
    
    try:
        # 同步执行生成任务（直接调用，不放在后台）
        run_video_generation_task(unique_id, text, character, model_name)
        
        # 检查任务状态
        with TASKS_LOCK:
            task = TASKS.get(unique_id, {})
            status = task.get("status", "unknown")
            
            if status == "completed":
                # 成功：返回视频路径
                video_url = task.get("video_url", f"/videos/{unique_id}.mp4")
                return {
                    "success": True,
                    "unique_id": unique_id,
                    "status": "completed",
                    "video_url": video_url,
                    "audio_url": task.get("audio_url"),
                    "text_url": task.get("text_url"),
                    "generation_time": task.get("generation_time"),
                    "message": "视频生成成功"
                }
            elif status == "failed":
                # 失败：返回错误信息
                return {
                    "success": False,
                    "unique_id": unique_id,
                    "status": "failed",
                    "error": task.get("error", "视频生成失败"),
                    "message": "视频生成失败"
                }
            else:
                # 未知状态
                return {
                    "success": False,
                    "unique_id": unique_id,
                    "status": status,
                    "error": "任务状态异常",
                    "message": f"任务状态: {status}"
                }
    
    except Exception as e:
        error_msg = f"同步生成任务执行异常: {str(e)}"
        add_log(f"[任务] {unique_id}: {error_msg}", "error")
        add_log(f"[任务] {unique_id}: 错误详情: {traceback.format_exc()}", "error")
        return {
            "success": False,
            "unique_id": unique_id,
            "status": "failed",
            "error": error_msg,
            "message": "任务执行异常"
        }


def run_video_generation_task(unique_id: str, text: str, character: str, model_name: str, is_from_chat: bool = False):
    """
    在后台线程中运行视频生成任务
    
    Args:
        is_from_chat: 如果为True，表示来自聊天对话，不保存到video_records数据库
    """
    start_time = time.time()
    
    # 更新任务状态为运行中（内存和数据库）
    with TASKS_LOCK:
        TASKS[unique_id] = {
            "status": "running",
            "start_time": start_time,
            "text": text,
            "character": character,
            "model_name": model_name,
            "is_from_chat": is_from_chat
        }
    
    # 如果不是来自聊天，才更新video_records数据库中的任务状态
    if not is_from_chat:
        create_or_update_task(
            task_id=unique_id,
            status="running",
            text=text,
            character=character,
            model_name=model_name,
            config={"text": text, "character": character, "model_name": model_name},
            start_time=datetime.fromtimestamp(start_time).isoformat()
        )
    
    try:
        add_log(f"[任务] {unique_id}: 开始生成视频任务", "info")

        audio_output = f"{OUTPUT_AUDIO_DIR}/{unique_id}.wav"
        # 使用绝对路径作为输出目录
        video_dir = Path(OUTPUT_VIDEO_DIR) / unique_id
        video_dir = video_dir.resolve()  # 转换为绝对路径
        temp_video_output = video_dir / "output_NeRFFaceSpeech.mp4"

        video_dir.mkdir(parents=True, exist_ok=True)
        add_log(f"[任务] {unique_id}: 视频输出目录（绝对路径）: {video_dir}", "info")

        # 步骤1: 生成音频
        add_log("=== 阶段1: LLM + TTS 音频生成 ===", "info")
        ok1, llm_response = generate_audio(
            text=text,
            output_path=audio_output,
            character=character
        )
        if not ok1:
            error_msg = "LLM语音生成失败"
            end_time_str = datetime.now().isoformat()
            generation_time_val = time.time() - start_time
            # 更新任务状态为失败
            with TASKS_LOCK:
                if unique_id in TASKS:
                    TASKS[unique_id]["status"] = "failed"
                    TASKS[unique_id]["error"] = error_msg
            # 如果不是来自聊天，才更新video_records数据库
            if not is_from_chat:
                create_or_update_task(
                    task_id=unique_id,
                    status="failed",
                    text=text,
                    character=character,
                    model_name=model_name,
                    config={"text": text, "character": character, "model_name": model_name},
                    error_message=error_msg,
                    start_time=datetime.fromtimestamp(start_time).isoformat(),
                    end_time=end_time_str,
                    generation_time=generation_time_val
                )
            # 在数据库中记录失败状态
            add_generation_record(
                unique_id=unique_id,
                text=text,
                character=character,
                model_name=model_name,
                video_path=None,
                audio_path=None,
                text_path=None,
                llm_response=None,
                generation_time=generation_time_val,
                config={"text": text, "character": character, "model_name": model_name},
                status='failed'
            )
            return
        
        if llm_response:
            add_log(f"=== 阶段1完成: 音频生成成功，LLM回答长度: {len(llm_response)} ===", "success")
        else:
            add_log("=== 阶段1完成: 音频生成成功（但未获取到LLM回答） ===", "success")

        # 步骤2: 生成视频
        add_log("=== 阶段2: NeRF 视频生成 ===", "info")
        # 确保使用绝对路径
        ok2 = generate_video(
            audio_path=str(Path(audio_output).resolve()),
            character=character,
            output_path=str(video_dir),  # 已经是绝对路径
            model_name=model_name
        )
        if not ok2:
            error_msg = "NeRF 视频生成失败（可能是网络超时导致模型下载失败，请查看后端日志）"
            add_log(f"[任务] {unique_id}: {error_msg}", "error")
            end_time_str = datetime.now().isoformat()
            generation_time_val = time.time() - start_time
            # 更新任务状态为失败
            with TASKS_LOCK:
                if unique_id in TASKS:
                    TASKS[unique_id]["status"] = "failed"
                    TASKS[unique_id]["error"] = error_msg
            # 如果不是来自聊天，才更新video_records数据库
            if not is_from_chat:
                create_or_update_task(
                    task_id=unique_id,
                    status="failed",
                    text=text,
                    character=character,
                    model_name=model_name,
                    config={"text": text, "character": character, "model_name": model_name},
                    error_message=error_msg,
                    start_time=datetime.fromtimestamp(start_time).isoformat(),
                    end_time=end_time_str,
                    generation_time=generation_time_val
                )
            # 在数据库中记录失败状态
            add_generation_record(
                unique_id=unique_id,
                text=text,
                character=character,
                model_name=model_name,
                video_path=None,
                audio_path=None,
                text_path=None,
                llm_response=llm_response if 'llm_response' in locals() else None,
                generation_time=generation_time_val,
                config={"text": text, "character": character, "model_name": model_name},
                status='failed'
            )
            return
        add_log("=== 阶段2完成: 视频生成成功 ===", "success")

        # 步骤3: 直接使用脚本生成的视频文件（不转码）
        add_log("=== 阶段3: 保存视频文件 ===", "info")
        
        # 最终视频文件路径（直接存储在videos目录下，不创建子文件夹）
        final_video_path = VIDEOS_STORAGE_DIR / f"{unique_id}.mp4"
        
        try:
            if temp_video_output.exists():
                # 直接移动文件，不进行转码
                add_log(f"[文件存储] 直接使用脚本生成的视频文件（不转码）", "info")
                print(f"[后端] [文件存储] 直接使用脚本生成的视频文件（不转码）")
                print(f"[后端] [文件存储] 源文件: {temp_video_output}")
                print(f"[后端] [文件存储] 目标文件: {final_video_path}")
                
                try:
                    shutil.move(str(temp_video_output), str(final_video_path))
                    if final_video_path.exists():
                        video_size = final_video_path.stat().st_size
                        add_log(f"[文件存储] ✅ 视频文件移动成功: {final_video_path} (大小: {video_size} 字节)", "success")
                        print(f"[后端] [文件存储] ✅ 视频文件移动成功: {final_video_path} (大小: {video_size} 字节)")
                    else:
                        add_log(f"[文件存储] 警告: 视频文件移动后不存在: {final_video_path}", "warning")
                except Exception as e:
                    add_log(f"[文件存储] 移动视频文件失败: {temp_video_output} -> {final_video_path}, 错误={e}", "error")
                    raise
                
                # 清理临时目录（如果为空）
                try:
                    if video_dir.exists():
                        video_dir.rmdir()
                        add_log(f"[文件存储] 临时目录已清理: {video_dir}", "info")
                except OSError:
                    add_log(f"[文件存储] 临时目录不为空，保留: {video_dir}", "info")
            else:
                add_log(f"[文件存储] 错误: 临时视频文件不存在: {temp_video_output}", "error")
                print(f"[后端] [文件存储] 错误: 临时视频文件不存在: {temp_video_output}")
                final_video_path = None
        except Exception as e:
            add_log(f"[文件存储] 移动视频文件失败: {temp_video_output} -> {final_video_path}, 错误={e}", "error")
            try:
                add_log(f"[文件存储] 错误详情: {traceback.format_exc()}", "error")
            except Exception:
                add_log(f"[文件存储] 错误详情: 无法获取详细堆栈信息", "error")
            # 如果移动失败，使用原始路径
            if os.path.exists(temp_video_output):
                final_video_path = Path(temp_video_output)
                add_log(f"[文件存储] 使用原始路径: {final_video_path}", "warning")
            else:
                error_msg = f"视频文件移动失败且原始文件不存在: {e}"
                add_log(f"[任务] {unique_id}: {error_msg}", "error")
                end_time_str = datetime.now().isoformat()
                generation_time_val = time.time() - start_time
                with TASKS_LOCK:
                    if unique_id in TASKS:
                        TASKS[unique_id]["status"] = "failed"
                        TASKS[unique_id]["error"] = error_msg
                # 如果不是来自聊天，才更新video_records数据库
                if not is_from_chat:
                    create_or_update_task(
                        task_id=unique_id,
                        status="failed",
                        text=text,
                        character=character,
                        model_name=model_name,
                        config={"text": text, "character": character, "model_name": model_name},
                        error_message=error_msg,
                        start_time=datetime.fromtimestamp(start_time).isoformat(),
                        end_time=end_time_str,
                        generation_time=generation_time_val
                    )
                return
        
        # 移动音频文件（直接存储在audios目录下，不创建子文件夹）
        final_audio_path = AUDIOS_STORAGE_DIR / f"{unique_id}.wav"
        
        try:
            if os.path.exists(audio_output):
                audio_size = os.path.getsize(audio_output)
                add_log(f"[文件存储] 开始移动音频文件: {audio_output} -> {final_audio_path} (大小: {audio_size} 字节)", "info")
                shutil.move(audio_output, str(final_audio_path))
                # 验证文件是否成功移动
                if os.path.exists(final_audio_path):
                    add_log(f"[文件存储] 音频文件移动成功: {final_audio_path}", "success")
                else:
                    add_log(f"[文件存储] 警告: 音频文件移动后不存在: {final_audio_path}", "warning")
                    final_audio_path = None
            else:
                add_log(f"[文件存储] 警告: 音频文件不存在: {audio_output}", "warning")
                final_audio_path = None
        except Exception as e:
            add_log(f"[文件存储] 移动音频文件失败: {audio_output} -> {final_audio_path}, 错误={e}", "warning")
            add_log(f"[文件存储] 错误详情: {traceback.format_exc()}", "warning")
            # 如果移动失败，尝试复制
            try:
                if os.path.exists(audio_output):
                    add_log(f"[文件存储] 尝试复制音频文件: {audio_output} -> {final_audio_path}", "info")
                    shutil.copy2(audio_output, str(final_audio_path))
                    if os.path.exists(final_audio_path):
                        add_log(f"[文件存储] 音频文件复制成功: {final_audio_path}", "success")
                    else:
                        final_audio_path = None
                else:
                    final_audio_path = None
            except Exception as e2:
                add_log(f"[文件存储] 复制音频文件也失败: 错误={e2}", "warning")
                add_log(f"[文件存储] 错误详情: {traceback.format_exc()}", "warning")
                final_audio_path = Path(audio_output) if os.path.exists(audio_output) else None
                if final_audio_path:
                    add_log(f"[文件存储] 使用原始音频路径: {final_audio_path}", "warning")

        # 计算生成时间
        generation_time = time.time() - start_time

        # 保存文本文件（LLM回答，直接存储在texts目录下，不创建子文件夹）
        add_log("=== 阶段4: 保存文本文件 ===", "info")
        final_text_path = None
        if llm_response:
            try:
                final_text_path = TEXTS_STORAGE_DIR / f"{unique_id}.txt"
                
                # 写入文本内容（UTF-8编码）
                with open(final_text_path, 'w', encoding='utf-8') as f:
                    f.write(llm_response)
                
                add_log(f"[文件存储] 文本文件保存成功: {final_text_path}", "success")
            except Exception as e:
                add_log(f"[文件存储] 保存文本文件失败: {e}", "warning")
                import traceback
                add_log(f"[文件存储] 错误详情: {traceback.format_exc()}", "warning")

        # 保存记录到数据库（包含音频路径和LLM回答）
        add_log("=== 阶段5: 保存生成记录到数据库 ===", "info")
        config = {
            "text": text,
            "character": character,
            "model_name": model_name
        }
        
        video_path_str = str(final_video_path) if final_video_path and final_video_path.exists() else None
        audio_path_str = str(final_audio_path) if final_audio_path and final_audio_path.exists() else None
        text_path_str = str(final_text_path) if final_text_path and final_text_path.exists() else None
        
        add_log(f"[数据库] 准备保存记录: unique_id={unique_id}, video_path={video_path_str}, audio_path={audio_path_str}, text_path={text_path_str}", "info")
        
        # 生成URL（从路径计算）
        video_url_val = f"/videos/{unique_id}.mp4"
        audio_url_val = f"/audios/{unique_id}.wav" if final_audio_path else None
        text_url_val = f"/texts/{unique_id}.txt" if final_text_path and final_text_path.exists() else None
        
        # 如果不是来自聊天，才保存到video_records数据库
        # 来自聊天的视频只保存到chat_messages表，不保存到video_records数据库
        if not is_from_chat:
            # 保存记录到数据库（状态为completed，包含所有路径信息）
            # 任务完成时直接保存为视频记录，不再单独维护任务状态
            save_success = add_video_record(
                unique_id=unique_id,
                text=text,
                character=character,
                model_name=model_name,
                video_path=video_path_str,
                audio_path=audio_path_str,
                text_path=text_path_str,  # 文本文件路径
                llm_response=llm_response,  # 保留字段，向后兼容
                generation_time=generation_time,
                config=config,
                status='completed'  # 状态为completed，这就是视频记录
            )
            
            if save_success:
                add_log(f"[任务] {unique_id}: 视频记录保存成功 (状态: completed, 路径: {video_path_str})", "success")
            else:
                add_log(f"[任务] {unique_id}: 警告: 视频记录保存失败", "warning")
        else:
            add_log(f"[任务] {unique_id}: 来自聊天，跳过保存到video_records数据库", "info")
        
        # 更新内存中的任务状态（用于API响应）
        # 存储文件路径而不是URL，前端会根据路径构造URL
        end_time_str = datetime.now().isoformat()
        with TASKS_LOCK:
            if unique_id in TASKS:
                TASKS[unique_id]["status"] = "completed"
                TASKS[unique_id]["generation_time"] = generation_time
                TASKS[unique_id]["video_path"] = video_path_str  # 存储文件系统路径
                TASKS[unique_id]["audio_path"] = audio_path_str if final_audio_path else None
                TASKS[unique_id]["text_path"] = text_path_str if final_text_path else None
        
        add_log(f"[任务] {unique_id}: 视频生成任务完成，耗时 {generation_time:.2f} 秒", "success")
        
    except Exception as e:
        error_msg = f"任务执行出错: {str(e)}"
        add_log(f"[任务] {unique_id}: {error_msg}", "error")
        add_log(f"[任务] {unique_id}: 错误详情: {traceback.format_exc()}", "error")
        end_time_str = datetime.now().isoformat()
        generation_time_val = time.time() - start_time
        # 更新任务状态为失败（内存和数据库）
        with TASKS_LOCK:
            if unique_id in TASKS:
                TASKS[unique_id]["status"] = "failed"
                TASKS[unique_id]["error"] = error_msg
        
        # 如果不是来自聊天，才更新video_records数据库
        if not is_from_chat:
            create_or_update_task(
                task_id=unique_id,
                status="failed",
                text=text,
                character=character,
                model_name=model_name,
                config={"text": text, "character": character, "model_name": model_name},
                error_message=error_msg,
                start_time=datetime.fromtimestamp(start_time).isoformat(),
                end_time=end_time_str,
                generation_time=generation_time_val
            )
        
        # 在数据库中记录失败状态
        try:
            add_generation_record(
                unique_id=unique_id,
                text=text,
                character=character,
                model_name=model_name,
                video_path=None,
                audio_path=None,
                text_path=None,
                llm_response=None,
                generation_time=generation_time_val,
                config={"text": text, "character": character, "model_name": model_name},
                status='failed'
            )
        except:
            pass


@app.get("/generate_video/status/{unique_id}")
def get_generation_status(unique_id: str):
    """
    查询视频生成任务状态
    """
    # 优先从内存中获取（正在运行的任务）
    with TASKS_LOCK:
        if unique_id in TASKS:
            task = TASKS[unique_id].copy()
            # 如果任务已完成或失败，从数据库获取完整信息
            if task["status"] in ["completed", "failed"]:
                try:
                    db_task = get_task(unique_id)
                    if db_task:
                        # 优先使用数据库中的路径信息
                        task.update({
                            "video_path": db_task.get("video_path"),  # 返回文件系统路径
                            "audio_path": db_task.get("audio_path"),
                            "text_path": db_task.get("text_path"),
                            "generation_time": db_task.get("generation_time"),
                            "error": db_task.get("error_message")
                        })
                except Exception as e:
                    add_log(f"[API] 从数据库获取任务 {unique_id} 记录失败: {e}", "warning")
            return {"success": True, "data": task}
    
    # 如果任务不在内存中，尝试从数据库查找
    try:
        db_task = get_task(unique_id)
        if db_task:
            task_data = {
                "status": db_task.get("status", "completed"),
                "text": db_task.get("text"),
                "character": db_task.get("character"),
                "model_name": db_task.get("model_name"),
                "video_path": db_task.get("video_path"),  # 返回文件系统路径而不是URL
                "audio_path": db_task.get("audio_path"),
                "text_path": db_task.get("text_path"),
                "generation_time": db_task.get("generation_time"),
                "error": db_task.get("error_message"),
                "start_time": db_task.get("start_time")
            }
            return {"success": True, "data": task_data}
    except Exception as e:
        add_log(f"[API] 从数据库查找任务 {unique_id} 失败: {e}", "warning")
    
    return {"success": False, "error": f"任务 {unique_id} 不存在"}


@app.get("/generate_video/running")
def get_running_task_api():
    """
    获取正在运行的任务（用于页面初始化时检查）
    """
    try:
        # 优先从内存中获取
        with TASKS_LOCK:
            for task_id, task in TASKS.items():
                if task.get("status") in ["pending", "running"]:
                    return {
                        "success": True,
                        "data": {
                            "unique_id": task_id,
                            "status": task.get("status"),
                            "text": task.get("text"),
                            "character": task.get("character"),
                            "model_name": task.get("model_name"),
                            "start_time": task.get("start_time")
                        }
                    }
        
        # 如果内存中没有，从数据库查找
        db_task = get_running_task()
        if db_task:
            return {
                "success": True,
                "data": {
                    "unique_id": db_task.get("task_id"),
                    "status": db_task.get("status"),
                    "text": db_task.get("text"),
                    "character": db_task.get("character"),
                    "model_name": db_task.get("model_name"),
                    "start_time": db_task.get("start_time")
                }
            }
        
        return {"success": True, "data": None}
        
    except Exception as e:
        add_log(f"[API] 获取正在运行的任务失败: {e}", "error")
        return {"success": False, "error": str(e)}


@app.post("/generate_video/cancel/{unique_id}")
def cancel_video_task_api(unique_id: str):
    """
    终止视频生成任务
    """
    try:
        
        # 检查任务是否存在
        db_task = get_task(unique_id)
        if not db_task:
            return {"success": False, "error": f"任务 {unique_id} 不存在"}
        
        # 检查任务状态
        if db_task.get("status") not in ["pending", "running"]:
            return {"success": False, "error": f"任务 {unique_id} 不在运行中（当前状态: {db_task.get('status')}）"}
        
        # 更新内存中的任务状态
        with TASKS_LOCK:
            if unique_id in TASKS:
                TASKS[unique_id]["status"] = "cancelled"
                TASKS[unique_id]["error"] = "任务已被用户终止"
        
        # 更新数据库中的任务状态
        end_time_str = datetime.now().isoformat()
        start_time_str = db_task.get("start_time")
        generation_time_val = None
        if start_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str).timestamp()
                generation_time_val = time.time() - start_time
            except:
                pass
        
        create_or_update_task(
            task_id=unique_id,
            status="cancelled",
            text=db_task.get("text"),
            character=db_task.get("character"),
            model_name=db_task.get("model_name"),
            config={"text": db_task.get("text"), "character": db_task.get("character"), "model_name": db_task.get("model_name")},
            error_message="任务已被用户终止",
            start_time=start_time_str,
            end_time=end_time_str,
            generation_time=generation_time_val
        )
        
        add_log(f"[任务] {unique_id}: 任务已终止", "info")
        return {"success": True, "message": "任务已终止"}
        
    except Exception as e:
        add_log(f"[API] 终止任务失败: {e}", "error")
        import traceback
        add_log(f"[API] 错误详情: {traceback.format_exc()}", "error")
        return {"success": False, "error": str(e)}


# ---------------------------
# 聊天对话接口
# ---------------------------
class ChatRequest(BaseModel):
    text: Optional[str] = None
    audio_base64: Optional[str] = None
    character: str = "ayanami"
    enable_audio: bool = True
    enable_video: bool = False  # 是否生成视频
    response_mode: Optional[str] = None  # 响应模式: "text", "audio", "video"
    session_id: Optional[str] = None  # 会话ID，如果不存在则创建新会话

@app.post("/chat")
def chat_api(request: ChatRequest):
    """
    聊天对话接口
    支持文本输入和音频输入，返回LLM回答和音频
    
    Args:
        text: 用户输入的文本
        audio_base64: 音频文件的base64编码（如果提供，将进行语音识别）
        character: 角色名称（ayanami 或 Aerith）
        enable_audio: 是否生成音频回复
        session_id: 会话ID（可选，如果不提供则创建新会话）
    
    Returns:
        dict: 包含LLM回答和音频的响应
    """
    import uuid
    import base64
    import shutil
    
    add_log(f"[聊天] 收到聊天请求: text={request.text[:50] if request.text else None}..., audio_base64={'provided' if request.audio_base64 else 'not provided'}, character={request.character}, enable_audio={request.enable_audio}, session_id={request.session_id}", "info")
    
    # 处理音频输入：如果有音频输入，先进行语音识别
    user_text = request.text
    if request.audio_base64 and not user_text:
        add_log("[聊天] 检测到音频输入，使用 subprocess 方式进行语音识别...", "info")
        try:
            # 直接使用 subprocess 方式调用封装好的环境
            if not transcribe_base64_audio_via_subprocess:
                error_msg = "transcribe_base64_audio_via_subprocess 函数不可用，请检查 shared/utils/asr_client.py"
                add_log(f"[聊天] ❌ {error_msg}", "error")
                return {
                    "success": False,
                    "error": error_msg
                }
            
            result = transcribe_base64_audio_via_subprocess(
                audio_base64=request.audio_base64,
                model_name="base",
                language=None,  # 自动检测语言
                task="transcribe"
            )
            
            if result.get("success"):
                user_text = result.get("data", {}).get("text", "").strip()
                add_log(f"[聊天] ✅ 语音识别成功: {user_text[:100]}...", "success")
            else:
                error_msg = result.get("error", {}).get("message", "语音识别失败")
                add_log(f"[聊天] ❌ 语音识别失败: {error_msg}", "error")
                return {
                    "success": False,
                    "error": f"语音识别失败: {error_msg}"
                }
        except Exception as e:
            error_msg = f"语音识别过程中发生错误: {str(e)}"
            add_log(f"[聊天] ❌ {error_msg}", "error")
            import traceback
            add_log(f"[聊天] 错误详情: {traceback.format_exc()}", "error")
            return {
                "success": False,
                "error": error_msg
            }
    
    # 检查是否有有效的文本输入
    if not user_text:
        add_log("[聊天] 请求失败: 未提供文本输入或音频识别失败", "warning")
        return {
            "success": False,
            "error": "请提供文本输入或有效的音频输入"
        }
    
    # 确定会话ID
    session_id = request.session_id
    if not session_id:
        session_id = f"chat-{uuid.uuid4()}"
        # 创建新会话
        create_chat_session(
            session_id=session_id,
            title=user_text[:30] if len(user_text) > 30 else user_text,
            character=request.character,
            config={"enable_audio": request.enable_audio}
        )
    
    # 确保会话存在
    session = get_chat_session(session_id)
    if not session:
        create_chat_session(
            session_id=session_id,
            title=user_text[:30] if len(user_text) > 30 else user_text,
            character=request.character,
            config={"enable_audio": request.enable_audio}
        )
    
    # 根据response_mode确定需要生成的内容
    response_mode = request.response_mode or "text"
    
    # 根据response_mode设置enable_audio和enable_video
    if response_mode == "text":
        enable_audio = True   # 文本模式：使用语音生成的逻辑（enable_audio=True），但获取文本后终止，只返回文本
        enable_video = False
    elif response_mode == "audio":
        enable_audio = True   # 音频模式：生成文本+音频
        enable_video = False
    elif response_mode == "video":
        enable_audio = True   # 视频模式：生成文本+音频+视频
        enable_video = True
    else:
        # 兼容旧代码：使用request中的参数
        enable_audio = request.enable_audio
        enable_video = request.enable_video
    
    # 保存用户消息
    user_message_id = f"user-{uuid.uuid4()}"
    # 确定内容类型
    user_content_type = "text"
    if request.audio_base64:
        user_content_type = "audio" if not user_text else "audio+text"
    
    add_chat_message(
        session_id=session_id,
        message_id=user_message_id,
        message_type="user",
        content_type=user_content_type,
        text_content=user_text,
        text_path=None,  # 用户输入的文本直接存储在数据库中
        audio_path=None,
        audio_base64=request.audio_base64 if request.audio_base64 else None
    )
    
    # 调用聊天函数
    add_log(f"[聊天] 开始调用LLM API: session_id={session_id}, user_input={user_text[:100]}..., response_mode={response_mode}, enable_audio={enable_audio}, enable_video={enable_video}", "info")
    
    # 阶段1: LLM文本生成
    add_log("=== [聊天] 阶段1: LLM 文本生成 ===", "info")
    result = chat_with_llm(
        user_input=user_text,
        character=request.character,
        enable_audio=enable_audio  # 使用处理后的enable_audio
    )
    
    if not result.get("success"):
        error_msg = result.get("error", "未知错误")
        add_log(f"[聊天] 阶段1失败: {error_msg}", "error")
        return result
    
    # 确保正确获取llm_answer
    llm_answer = result.get("data", {}).get("llm_answer", "") if result.get("data") else ""
    
    # 根据response_mode决定是否在日志中输出文本内容
    if response_mode == "text":
        if llm_answer:
            add_log(f"=== [聊天] 阶段1完成: LLM 文本生成成功，回答: {llm_answer[:200]}... ===", "success")
        else:
            add_log(f"=== [聊天] 阶段1完成: LLM 文本生成成功，但回答为空 ===", "warning")
    else:
        add_log(f"=== [聊天] 阶段1完成: LLM 文本生成成功（回答长度: {len(llm_answer) if llm_answer else 0}，模式: {response_mode}，不显示内容） ===", "success")
    
    # 如果启用了音频，说明阶段2（TTS音频生成）也在chat_with_llm中完成了
    if enable_audio:
        audio_base64 = result.get("data", {}).get("audio_base64")
        if audio_base64:
            if response_mode == "text":
                add_log("=== [聊天] 阶段2完成: TTS 音频生成成功（文本模式：已生成但仅返回文本） ===", "success")
            else:
                add_log("=== [聊天] 阶段2完成: TTS 音频生成成功 ===", "success")
        else:
            add_log("=== [聊天] 阶段2警告: TTS 音频生成失败或无音频数据 ===", "warning")
    elif not enable_audio:
        add_log("=== [聊天] 跳过阶段2: 未启用音频生成 ===", "info")
    
    # 如果聊天成功，保存AI回复到数据库
    if result.get("success") and result.get("data"):
        try:
            # 确保llm_answer不为空（如果之前获取的为空，再次尝试获取）
            if not llm_answer and result.get("data", {}).get("llm_answer"):
                llm_answer = result["data"]["llm_answer"]
            audio_base64 = result["data"].get("audio_base64")
            
            # 确保audio_base64是字符串而不是bytes
            if audio_base64 and isinstance(audio_base64, bytes):
                audio_base64 = audio_base64.decode('utf-8')
            elif audio_base64 and not isinstance(audio_base64, str):
                # 如果不是字符串也不是bytes，尝试转换为字符串
                audio_base64 = str(audio_base64)
            
            # 根据response_mode决定是否在日志中输出文本内容（response_mode已在前面定义）
            if response_mode == "text":
                add_log(f"[聊天] LLM回答长度: {len(llm_answer) if llm_answer else 0}", "info")
            elif response_mode == "audio":
                add_log(f"[聊天] LLM回答长度: {len(llm_answer) if llm_answer else 0}, 音频base64长度: {len(audio_base64) if audio_base64 else 0}", "info")
                # 语音模式不打印文本内容，只显示长度
            elif response_mode == "video":
                add_log(f"[聊天] LLM回答长度: {len(llm_answer) if llm_answer else 0}, 音频base64长度: {len(audio_base64) if audio_base64 else 0} (视频模式：文本用于TTS，不显示)", "info")
            
            # 根据response_mode决定是否保存文本文件（所有模式都保存文本，用于数据库记录）
            assistant_text_path = None
            if llm_answer:
                try:
                    text_file = TEXTS_STORAGE_DIR / f"{user_message_id}.txt"
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(llm_answer)
                    assistant_text_path = str(text_file)
                    add_log(f"[文件存储] AI回复文本文件保存成功: {text_file}", "info")
                except Exception as e:
                    add_log(f"[文件存储] 保存AI回复文本文件失败: {e}", "warning")
            
            # 保存AI回复音频文件（仅音频模式和视频模式保存音频，文本模式不保存）
            assistant_audio_path = None
            if enable_audio and audio_base64 and response_mode != "text":
                # 文本模式虽然调用了enable_audio=True，但不保存音频文件
                try:
                    audio_file = AUDIOS_STORAGE_DIR / f"{user_message_id}.wav"
                    
                    # 将base64解码为音频文件
                    audio_data = base64.b64decode(audio_base64)
                    with open(audio_file, 'wb') as f:
                        f.write(audio_data)
                    
                    assistant_audio_path = str(audio_file)
                    add_log(f"[文件存储] AI回复音频文件保存成功: {audio_file}", "info")
                except Exception as e:
                    add_log(f"[文件存储] 保存AI回复音频文件失败: {e}", "warning")
            elif response_mode == "text" and audio_base64:
                # 文本模式：生成了音频但不保存，记录日志
                add_log(f"[文件存储] 文本模式：已生成音频但未保存（仅返回文本）", "info")
            elif response_mode == "text" and audio_base64:
                # 文本模式：生成了音频但不保存，记录日志
                add_log(f"[文件存储] 文本模式：已生成音频但未保存（仅返回文本）", "info")
            
            # 根据response_mode确定数据库存储的内容类型
            if response_mode == "text":
                content_type = "text"  # 文本模式：只存储文本（虽然生成了音频但不存储）
            elif response_mode == "audio":
                content_type = "text+audio"  # 音频模式：存储文本+音频
            elif response_mode == "video":
                content_type = "text+audio+video"  # 视频模式：存储文本+音频+视频
            else:
                # 默认行为（兼容旧代码）
                content_type = "text"
                if enable_audio and audio_base64 and response_mode != "text":
                    content_type = "text+audio"
            
            # 阶段3: 视频生成（仅视频模式）- 对于视频模式，chat API不处理视频生成
            # 视频生成应该在聊天完成后，由前端直接调用 /generate_video API（就像generate.html那样）
            # 但是为了保持一致性，我们仍然需要返回video_task_id，这里返回None，让前端自己调用generate_video API
            video_task_id = None
            assistant_video_path = None  # 初始化视频路径（文本和音频模式下为None）
            if enable_video:
                add_log("=== [聊天] 阶段3: 视频模式，前端将直接调用 /generate_video API ===", "info")
                # 视频模式：chat API只负责生成文本和音频，视频生成由前端调用 /generate_video API完成
                # 这样就和generate.html的方式完全一致了
            
            # 保存AI回复消息
            assistant_message_id = f"assistant-{uuid.uuid4()}"
            # 所有模式都存储文本内容（数据库格式要求）
            # 文本模式下，audio_path和audio_base64都为None（虽然生成了音频但不保存）
            audio_base64_to_store = None
            if response_mode != "text":
                # 非文本模式：如果保存了文件，不存储base64；否则存储base64
                audio_base64_to_store = audio_base64 if not assistant_audio_path else None
            # 文本模式：不存储音频，即使生成了也不存储
            add_chat_message(
                session_id=session_id,
                message_id=assistant_message_id,
                message_type="assistant",
                content_type=content_type,  # text, text+audio, text+audio+video
                text_content=llm_answer,  # 所有模式都存储文本
                text_path=assistant_text_path,
                audio_path=assistant_audio_path,  # 文本模式下为None
                audio_base64=audio_base64_to_store,  # 文本模式下为None
                video_path=assistant_video_path  # 视频路径（仅视频模式，其他模式为None）
            )
            
            add_log(f"[数据库] 聊天消息已保存: session_id={session_id}, user_message_id={user_message_id}, assistant_message_id={assistant_message_id}", "info")
            
            # 在响应中添加会话ID和消息ID
            result["data"]["session_id"] = session_id
            result["data"]["user_message_id"] = user_message_id
            result["data"]["assistant_message_id"] = assistant_message_id
            
            # 根据响应模式过滤返回的数据
            if response_mode == "text":
                # 文本模式：只返回文本，不返回音频和视频
                # 确保llm_answer存在且不为空（使用保存的llm_answer变量，确保有值）
                result["data"]["llm_answer"] = llm_answer if llm_answer else result["data"].get("llm_answer", "")
                if "audio_base64" in result["data"]:
                    del result["data"]["audio_base64"]
                if "audio_url" in result["data"]:
                    del result["data"]["audio_url"]
                if "video_url" in result["data"]:
                    del result["data"]["video_url"]
                if "video_task_id" in result["data"]:
                    del result["data"]["video_task_id"]
            elif response_mode == "audio":
                # 音频模式：只返回音频，不返回文本和视频
                if "llm_answer" in result["data"]:
                    del result["data"]["llm_answer"]  # 不返回文本
                if assistant_audio_path:
                    result["data"]["audio_url"] = f"/audios/{Path(assistant_audio_path).name}"
                    if "audio_base64" in result["data"]:
                        del result["data"]["audio_base64"]
                if "video_url" in result["data"]:
                    del result["data"]["video_url"]
                if "video_task_id" in result["data"]:
                    del result["data"]["video_task_id"]
            elif response_mode == "video":
                # 视频模式：返回文本和音频，但不返回视频（视频由前端调用 /generate_video API生成）
                # 保留llm_answer和audio_url，供前端调用generate_video API时使用
                # 删除video相关字段（因为还没有生成视频）
                if "video_url" in result["data"]:
                    del result["data"]["video_url"]
                if "video_task_id" in result["data"]:
                    del result["data"]["video_task_id"]
                # 不删除llm_answer和audio_url，因为前端需要这些数据来调用generate_video API
            else:
                # 默认行为：如果音频已保存为文件，返回URL而不是base64（节省带宽）
                if assistant_audio_path:
                    result["data"]["audio_url"] = f"/audios/{Path(assistant_audio_path).name}"
                # 移除base64数据，避免在响应中传输大量数据
                if "audio_base64" in result["data"]:
                    del result["data"]["audio_base64"]
            
            add_log(f"[聊天] 响应准备完成: session_id={session_id}", "success")
        except Exception as e:
            add_log(f"[数据库] 保存聊天消息失败: {e}", "warning")
            import traceback
            add_log(f"[数据库] 错误详情: {traceback.format_exc()}", "warning")
            # 即使保存失败，也不影响聊天功能
    
    # 确保返回的数据中audio_base64是字符串（如果存在）
    if result.get("success") and result.get("data") and result["data"].get("audio_base64"):
        audio_base64_val = result["data"]["audio_base64"]
        if isinstance(audio_base64_val, bytes):
            result["data"]["audio_base64"] = audio_base64_val.decode('utf-8')
        elif not isinstance(audio_base64_val, str):
            result["data"]["audio_base64"] = str(audio_base64_val)
    
    add_log(f"[聊天] 返回响应: success={result.get('success')}", "info")
    return result


# ---------------------------
# ASR 语音识别接口
# ---------------------------
class ASRRequest(BaseModel):
    """ASR 请求模型"""
    audio_base64: Optional[str] = None
    audio_path: Optional[str] = None
    model_name: str = "base"
    language: Optional[str] = None  # None表示自动检测
    task: str = "transcribe"  # transcribe 或 translate

@app.post("/asr/transcribe")
def asr_transcribe_api(request: ASRRequest):
    """
    语音识别接口
    支持Base64音频数据或文件路径
    
    Args:
        audio_base64: Base64编码的音频数据
        audio_path: 音频文件路径
        model_name: Whisper模型名称（tiny, base, small, medium, large）
        language: 语言代码（如'zh', 'en'），None表示自动检测
        task: 任务类型（transcribe或translate）
    
    Returns:
        dict: 包含识别文本的响应
    """
    add_log(f"[ASR] 收到语音识别请求: model={request.model_name}, language={request.language}, task={request.task}", "info")
    
    # 检查输入
    if not request.audio_base64 and not request.audio_path:
        return {
            "success": False,
            "error": "请提供audio_base64或audio_path"
        }
    
    if request.audio_base64 and request.audio_path:
        return {
            "success": False,
            "error": "请只提供audio_base64或audio_path之一"
        }
    
    try:
        # 直接使用 subprocess 方式调用封装好的环境
        if not transcribe_base64_audio_via_subprocess:
            error_msg = "transcribe_base64_audio_via_subprocess 函数不可用，请检查 shared/utils/asr_client.py"
            add_log(f"[ASR] ❌ {error_msg}", "error")
            return {
                "success": False,
                "error": error_msg
            }
        
        add_log("[ASR] 使用 subprocess 调用方式进行识别", "info")
        
        # 只支持 Base64 音频的 subprocess 调用
        if request.audio_base64:
            result = transcribe_base64_audio_via_subprocess(
                audio_base64=request.audio_base64,
                model_name=request.model_name,
                language=request.language,
                task=request.task
            )
        else:
            # 如果是文件路径，需要先读取并转换为 Base64
            import base64
            try:
                with open(request.audio_path, 'rb') as f:
                    audio_data = f.read()
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                result = transcribe_base64_audio_via_subprocess(
                    audio_base64=audio_base64,
                    model_name=request.model_name,
                    language=request.language,
                    task=request.task
                )
            except Exception as e:
                add_log(f"[ASR] 读取音频文件失败: {e}", "error")
                return {
                    "success": False,
                    "error": f"读取音频文件失败: {str(e)}"
                }
        
        if result.get("success"):
            data = result.get("data", {})
            return {
                "success": True,
                "data": {
                    "text": data.get("text", ""),
                    "language": data.get("language", "unknown"),
                    "model_name": request.model_name,
                    "task": request.task,
                    "segments": data.get("segments", [])
                }
            }
        else:
            error_msg = result.get("error", {}).get("message", "识别失败")
            add_log(f"[ASR] ❌ 识别失败: {error_msg}", "error")
            return {
                "success": False,
                "error": error_msg
            }
            
    except Exception as e:
        error_msg = f"语音识别过程中发生错误: {str(e)}"
        add_log(f"[ASR] ❌ {error_msg}", "error")
        import traceback
        add_log(f"[ASR] 错误详情: {traceback.format_exc()}", "error")
        return {
            "success": False,
            "error": error_msg
        }

@app.get("/asr/health")
def asr_health_check():
    """检查 ASR 功能健康状态（使用 subprocess 方式）"""
    if not transcribe_base64_audio_via_subprocess:
        return {
            "success": False,
            "asr_available": False,
            "method": "subprocess",
            "error": "transcribe_base64_audio_via_subprocess 函数不可用",
            "message": "请检查 shared/utils/asr_client.py 是否存在该函数"
        }
    
    # 检查 LLM_CONDA_PYTHON 是否存在
    from shared.config import LLM_CONDA_PYTHON, LLM_CONDA_ENV
    python_exists = LLM_CONDA_PYTHON.exists() if LLM_CONDA_PYTHON else False
    env_exists = LLM_CONDA_ENV.exists() if LLM_CONDA_ENV else False
    
    return {
        "success": True,
        "asr_available": True,
        "method": "subprocess",
        "message": "ASR 功能可用（使用 subprocess 调用封装好的环境）",
        "python_path": str(LLM_CONDA_PYTHON) if LLM_CONDA_PYTHON else None,
        "python_exists": python_exists,
        "env_path": str(LLM_CONDA_ENV) if LLM_CONDA_ENV else None,
        "env_exists": env_exists
    }

# ---------------------------
# 纯LLM问答接口（不生成音频，快速响应）
# ---------------------------
class LLMOnlyRequest(BaseModel):
    text: str

@app.post("/llm_only")
def llm_only_api(request: LLMOnlyRequest):
    """
    仅获取LLM回答，不生成音频（用于快速响应）
    
    Args:
        request: 包含用户输入文本的请求对象
    
    Returns:
        dict: 包含LLM回答的响应
    """
    result = get_llm_only(request.text)
    return result


# ---------------------------
# 聊天会话和消息管理接口
# ---------------------------
@app.get("/chat/sessions")
def list_chat_sessions_api(limit: int = 100, offset: int = 0):
    """获取所有聊天会话列表"""
    try:
        sessions = list_chat_sessions(limit=limit, offset=offset)
        return {
            "success": True,
            "data": sessions
        }
    except Exception as e:
        add_log(f"[API] 获取聊天会话列表失败: {e}", "error")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/chat/sessions/{session_id}/messages")
def get_chat_messages_api(session_id: str, limit: int = 1000, offset: int = 0):
    """获取指定会话的所有消息"""
    try:
        messages = get_chat_messages(session_id=session_id, limit=limit, offset=offset)
        
        # 转换路径为URL
        for msg in messages:
            # 转换音频路径为URL（音频文件存储在AUDIOS_STORAGE_DIR，通过/audios/{filename}访问）
            if msg.get('audio_path'):
                audio_filename = Path(msg['audio_path']).name
                msg['audio_url'] = f"/audios/{audio_filename}"
            # 转换视频路径为URL（视频文件存储在VIDEOS_STORAGE_DIR，通过/videos/{filename}访问）
            if msg.get('video_path'):
                video_filename = Path(msg['video_path']).name
                msg['video_url'] = f"/videos/{video_filename}"
            # 转换文本路径为URL（可选，文本通常直接存储在数据库中）
            if msg.get('text_path'):
                text_filename = Path(msg['text_path']).name
                msg['text_url'] = f"/texts/{text_filename}"
        
        return {
            "success": True,
            "data": messages
        }
    except Exception as e:
        add_log(f"[API] 获取聊天消息失败: {e}", "error")
        return {
            "success": False,
            "error": str(e)
        }


@app.delete("/chat/sessions/{session_id}")
def delete_chat_session_api(session_id: str):
    """删除聊天会话及其所有消息"""
    try:
        success = delete_chat_session(session_id)
        return {
            "success": success
        }
    except Exception as e:
        add_log(f"[API] 删除聊天会话失败: {e}", "error")
        return {
            "success": False,
            "error": str(e)
        }


@app.put("/chat/messages/{message_id}/video_path")
def update_message_video_path_api(message_id: str, request_body: dict = Body(...)):
    """更新聊天消息的视频路径（用于视频生成完成后更新数据库）"""
    try:
        # 获取video_path参数
        video_path = request_body.get("video_path", "") if isinstance(request_body, dict) else ""
        
        # 验证路径格式
        if not video_path:
            return {
                "success": False,
                "error": "视频路径不能为空"
            }
        
        # 如果路径是相对路径（如 "database/videos/task_id.mp4"），转换为绝对路径
        # 确保路径格式与generate_video API保存的格式一致（绝对路径字符串）
        video_path_obj = Path(video_path)
        if not video_path_obj.is_absolute():
            # 相对路径，提取文件名并转换为绝对路径
            filename = video_path_obj.name
            video_path = str(VIDEOS_STORAGE_DIR / filename)
        
        # 更新数据库
        success = update_chat_message_video_path(message_id, video_path)
        
        if success:
            add_log(f"[API] 消息视频路径已更新: message_id={message_id}, video_path={video_path}", "info")
            return {
                "success": True
            }
        else:
            return {
                "success": False,
                "error": "消息不存在或更新失败"
            }
    except Exception as e:
        add_log(f"[API] 更新消息视频路径失败: {e}", "error")
        return {
            "success": False,
            "error": str(e)
        }


# ---------------------------
# 训练相关接口
# ---------------------------
class TrainingRequest(BaseModel):
    data_path: str
    base_model: str = "ffhq_1024.pkl"
    kimg: int = 50
    snap: int = 5
    imgsnap: int = 1
    aug: str = "noaug"
    mirror: bool = False
    config_name: str = "style_ffhq_ae_basic"

@app.post("/train/start")
def start_training_api(request: TrainingRequest):
    """
    启动模型训练
    
    Args:
        request: 训练请求参数对象
    
    Returns:
        dict: 任务ID和状态
    """
    result = start_training(
        data_path=request.data_path,
        base_model=request.base_model,
        kimg=request.kimg,
        snap=request.snap,
        imgsnap=request.imgsnap,
        aug=request.aug,
        mirror=request.mirror,
        model_config=request.config_name
    )
    return result


@app.get("/train/status/{task_id}")
def get_training_status_api(task_id: str):
    """
    获取训练任务状态
    
    Args:
        task_id: 任务ID
    
    Returns:
        dict: 任务状态和日志
    """
    return get_training_status(task_id)


@app.get("/train/tasks")
def list_training_tasks_api():
    """
    列出所有训练任务
    
    Returns:
        dict: 任务列表
    """
    return list_training_tasks()


@app.post("/train/stop/{task_id}")
def stop_training_api(task_id: str):
    """
    停止训练任务
    
    Args:
        task_id: 任务ID
    
    Returns:
        dict: 操作结果
    """
    return stop_training(task_id)


@app.get("/train/datasets")
def list_datasets_api():
    """
    列出可用的训练数据集路径
    
    Returns:
        dict: 包含数据集路径列表的响应
    """
    try:
        datasets = []
        
        # 检查默认数据集目录
        if DATA_DIR.exists():
            # 列出data目录下的子目录作为可选数据集
            for item in DATA_DIR.iterdir():
                if item.is_dir():
                    datasets.append({
                        "name": item.name,
                        "path": str(item),
                        "exists": True
                    })
        
        # 添加默认训练数据集路径（如果存在）
        default_dataset = str(TRAINING_DATASET_DIR)
        if TRAINING_DATASET_DIR.exists():
            datasets.insert(0, {
                "name": "默认训练数据集",
                "path": default_dataset,
                "exists": True,
                "default": True
            })
        else:
            datasets.insert(0, {
                "name": "默认训练数据集（不存在）",
                "path": default_dataset,
                "exists": False,
                "default": True
            })
        
        return {
            "success": True,
            "data": datasets,
            "default_path": default_dataset
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"获取数据集列表失败: {str(e)}"
        }


# ---------------------------
# 角色训练接口
# ---------------------------
@app.post("/character/train")
async def train_character_api(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="人物讲话视频文件"),
    character_name: str = Form(..., description="角色名称"),
    face_ratio: float = Form(0.6, description="人脸占画面的比例（0.0-1.0）"),
    output_size_w: int = Form(1024, description="输出视频宽度"),
    output_size_h: int = Form(1024, description="输出视频高度"),
    ffhq_alignment: bool = Form(True, description="是否使用FFHQ对齐"),
    overwrite: bool = Form(False, description="是否覆盖已存在的训练数据"),
):
    """
    角色训练接口：处理用户上传的视频，提取训练数据（图像和音频）
    
    Args:
        video: 上传的视频文件
        character_name: 角色名称（用于创建输出目录）
        face_ratio: 人脸占画面的比例（默认0.6）
        output_size_w: 输出视频宽度（默认1024）
        output_size_h: 输出视频高度（默认1024）
        ffhq_alignment: 是否使用FFHQ对齐（默认True）
        overwrite: 是否覆盖已存在的训练数据（默认False）
    
    Returns:
        dict: 任务ID和状态
    """
    try:
        add_log(f"[角色训练] ========== 收到训练请求 ==========", "info")
        add_log(f"[角色训练] 角色名称: {character_name}", "info")
        add_log(f"[角色训练] 视频文件名: {video.filename if video.filename else '未提供'}", "info")
        add_log(f"[角色训练] 参数: face_ratio={face_ratio}, output_size=({output_size_w}, {output_size_h}), ffhq_alignment={ffhq_alignment}, overwrite={overwrite}", "info")
        
        # 验证角色名称
        if not character_name or not character_name.strip():
            add_log(f"[角色训练] ❌ 验证失败: 角色名称不能为空", "error")
            return {
                "success": False,
                "error": "角色名称不能为空"
            }
        
        character_name = character_name.strip()
        add_log(f"[角色训练] 角色名称已清理: '{character_name}'", "info")
        
        # 验证视频文件
        if not video.filename:
            add_log(f"[角色训练] ❌ 验证失败: 未提供视频文件", "error")
            return {
                "success": False,
                "error": "未提供视频文件"
            }
        
        # 检查文件扩展名
        video_ext = Path(video.filename).suffix.lower()
        add_log(f"[角色训练] 视频文件扩展名: {video_ext}", "info")
        if video_ext not in ['.mp4', '.avi', '.mov', '.mkv']:
            add_log(f"[角色训练] ❌ 验证失败: 不支持的视频格式 {video_ext}", "error")
            return {
                "success": False,
                "error": f"不支持的视频格式: {video_ext}，支持的格式: .mp4, .avi, .mov, .mkv"
            }
        
        # 创建临时文件保存上传的视频
        temp_dir = Path(tempfile.mkdtemp(prefix="character_train_"))
        temp_video = temp_dir / video.filename
        add_log(f"[角色训练] 创建临时目录: {temp_dir}", "info")
        add_log(f"[角色训练] 临时视频路径: {temp_video}", "info")
        
        # 保存上传的文件
        add_log(f"[角色训练] 开始保存上传的视频文件...", "info")
        file_size = 0
        with open(temp_video, "wb") as f:
            content = await video.read()
            file_size = len(content)
            f.write(content)
        add_log(f"[角色训练] 视频文件已保存，大小: {file_size / 1024 / 1024:.2f} MB", "info")
        
        # 验证文件是否成功保存
        if not temp_video.exists():
            add_log(f"[角色训练] ❌ 错误: 文件保存后不存在: {temp_video}", "error")
            return {
                "success": False,
                "error": "文件保存失败"
            }
        
        file_size_on_disk = temp_video.stat().st_size
        add_log(f"[角色训练] 磁盘文件大小: {file_size_on_disk / 1024 / 1024:.2f} MB", "info")
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        add_log(f"[角色训练] 生成任务ID: {task_id}", "info")
        
        # 初始化任务状态
        with TASKS_LOCK:
            TASKS[task_id] = {
                "status": "pending",
                "start_time": time.time(),
                "character_name": character_name,
                "video_filename": video.filename,
            }
        add_log(f"[角色训练] 任务状态已初始化: pending", "info")
        
        # 在后台运行处理任务
        add_log(f"[角色训练] 提交后台任务...", "info")
        background_tasks.add_task(
            run_character_training_task,
            task_id,
            temp_video,
            character_name,
            face_ratio,
            (output_size_w, output_size_h),
            ffhq_alignment,
            overwrite,
            temp_dir,
        )
        
        add_log(f"[角色训练] ✅ 任务已提交到后台处理", "info")
        add_log(f"[角色训练] ========================================", "info")
        
        return {
            "success": True,
            "task_id": task_id,
            "status": "pending",
            "character_name": character_name,
            "message": "角色训练任务已提交，正在后台处理中...",
        }
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        add_log(f"[角色训练] ========== 提交任务异常 ==========", "error")
        add_log(f"[角色训练] ❌ 异常类型: {type(e).__name__}", "error")
        add_log(f"[角色训练] ❌ 异常消息: {str(e)}", "error")
        add_log(f"[角色训练] ❌ 异常堆栈:\n{error_traceback}", "error")
        add_log(f"[角色训练] ========================================", "error")
        return {
            "success": False,
            "error": f"提交训练任务失败: {str(e)}"
        }


def run_character_training_task(
    task_id: str,
    video_path: Path,
    character_name: str,
    face_ratio: float,
    output_size: Tuple[int, int],
    ffhq_alignment: bool,
    overwrite: bool,
    temp_dir: Path,
):
    """
    后台运行角色训练任务
    """
    import traceback
    start_time = time.time()
    
    try:
        add_log(f"[角色训练] {task_id} ========== 开始处理任务 ==========", "info")
        add_log(f"[角色训练] {task_id} 任务参数:", "info")
        add_log(f"[角色训练] {task_id}   - 视频路径: {video_path}", "info")
        add_log(f"[角色训练] {task_id}   - 角色名称: {character_name}", "info")
        add_log(f"[角色训练] {task_id}   - 人脸占比: {face_ratio}", "info")
        add_log(f"[角色训练] {task_id}   - 输出尺寸: {output_size}", "info")
        add_log(f"[角色训练] {task_id}   - FFHQ对齐: {ffhq_alignment}", "info")
        add_log(f"[角色训练] {task_id}   - 覆盖模式: {overwrite}", "info")
        add_log(f"[角色训练] {task_id}   - 临时目录: {temp_dir}", "info")
        
        # 验证输入文件
        if not video_path.exists():
            error_msg = f"视频文件不存在: {video_path}"
            add_log(f"[角色训练] {task_id} ❌ {error_msg}", "error")
            with TASKS_LOCK:
                if task_id in TASKS:
                    TASKS[task_id]["status"] = "failed"
                    TASKS[task_id]["error"] = error_msg
            return
        
        video_size = video_path.stat().st_size
        add_log(f"[角色训练] {task_id} 视频文件大小: {video_size / 1024 / 1024:.2f} MB", "info")
        
        with TASKS_LOCK:
            if task_id in TASKS:
                TASKS[task_id]["status"] = "processing"
                TASKS[task_id]["start_time"] = start_time
        
        add_log(f"[角色训练] {task_id} 状态已更新为: processing", "info")
        add_log(f"[角色训练] {task_id} 开始调用 process_video_for_training...", "info")
        
        # 调用处理函数
        result = process_video_for_training(
            video_path=video_path,
            character_name=character_name,
            face_ratio=face_ratio,
            output_size=output_size,
            ffhq_alignment=ffhq_alignment,
            overwrite=overwrite,
        )
        
        elapsed_time = time.time() - start_time
        add_log(f"[角色训练] {task_id} process_video_for_training 完成，耗时: {elapsed_time:.2f} 秒", "info")
        add_log(f"[角色训练] {task_id} 处理结果: success={result.get('success', False)}", "info")
        
        if not result.get("success"):
            error_msg = result.get("error", "未知错误")
            add_log(f"[角色训练] {task_id} ❌ 处理失败: {error_msg}", "error")
        
        # 清理临时目录
        add_log(f"[角色训练] {task_id} 开始清理临时目录: {temp_dir}", "info")
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            add_log(f"[角色训练] {task_id} 临时目录已清理", "info")
        except Exception as cleanup_error:
            add_log(f"[角色训练] {task_id} ⚠️ 清理临时目录失败: {cleanup_error}", "warning")
        
        # 更新任务状态
        with TASKS_LOCK:
            if task_id in TASKS:
                if result["success"]:
                    TASKS[task_id]["status"] = "completed"
                    TASKS[task_id]["result"] = result
                    num_frames = result.get('num_frames', 0)
                    add_log(f"[角色训练] {task_id} ✅ 任务完成", "info")
                    add_log(f"[角色训练] {task_id}    - 提取帧数: {num_frames}", "info")
                    add_log(f"[角色训练] {task_id}    - 角色目录: {result.get('character_dir', 'N/A')}", "info")
                    add_log(f"[角色训练] {task_id}    - 图像目录: {result.get('images_dir', 'N/A')}", "info")
                    add_log(f"[角色训练] {task_id}    - 音频文件: {result.get('audio_file', 'N/A')}", "info")
                    if result.get('pti_models'):
                        add_log(f"[角色训练] {task_id}    - PTI模型: 已生成", "info")
                    else:
                        add_log(f"[角色训练] {task_id}    - PTI模型: 未生成", "warning")
                else:
                    TASKS[task_id]["status"] = "failed"
                    error_msg = result.get("error", "未知错误")
                    TASKS[task_id]["error"] = error_msg
                    add_log(f"[角色训练] {task_id} ❌ 任务失败: {error_msg}", "error")
        
        total_time = time.time() - start_time
        add_log(f"[角色训练] {task_id} ========== 任务结束，总耗时: {total_time:.2f} 秒 ==========", "info")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_traceback = traceback.format_exc()
        add_log(f"[角色训练] {task_id} ========== 任务执行异常 ==========", "error")
        add_log(f"[角色训练] {task_id} ❌ 异常类型: {type(e).__name__}", "error")
        add_log(f"[角色训练] {task_id} ❌ 异常消息: {str(e)}", "error")
        add_log(f"[角色训练] {task_id} ❌ 异常堆栈:\n{error_traceback}", "error")
        add_log(f"[角色训练] {task_id} ❌ 任务耗时: {elapsed_time:.2f} 秒", "error")
        add_log(f"[角色训练] {task_id} ========================================", "error")
        
        with TASKS_LOCK:
            if task_id in TASKS:
                TASKS[task_id]["status"] = "failed"
                TASKS[task_id]["error"] = f"{type(e).__name__}: {str(e)}"
        
        # 清理临时目录
        try:
            add_log(f"[角色训练] {task_id} 尝试清理临时目录...", "info")
            shutil.rmtree(temp_dir, ignore_errors=True)
            add_log(f"[角色训练] {task_id} 临时目录已清理", "info")
        except Exception as cleanup_error:
            add_log(f"[角色训练] {task_id} ⚠️ 清理临时目录失败: {cleanup_error}", "warning")


@app.get("/character/train/status/{task_id}")
def get_character_training_status_api(task_id: str):
    """
    获取角色训练任务状态
    
    Args:
        task_id: 任务ID
    
    Returns:
        dict: 任务状态信息
    """
    try:
        with TASKS_LOCK:
            if task_id not in TASKS:
                return {
                    "success": False,
                    "error": "任务不存在"
                }
            
            task_info = TASKS[task_id].copy()
        
        return {
            "success": True,
            "task_id": task_id,
            "status": task_info.get("status", "unknown"),
            "character_name": task_info.get("character_name"),
            "video_filename": task_info.get("video_filename"),
            "result": task_info.get("result"),
            "error": task_info.get("error"),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"获取任务状态失败: {str(e)}"
        }


@app.get("/character/list")
def list_characters_api():
    """
    列出所有已训练的角色
    
    Returns:
        dict: 角色列表
    """
    try:
        result = list_characters()
        return result
    except Exception as e:
        add_log(f"[角色训练] 获取角色列表失败: {e}", "error")
        return {
            "success": False,
            "error": f"获取角色列表失败: {str(e)}"
        }


@app.get("/character/{character_name}/status")
def get_character_status_api(character_name: str):
    """
    获取指定角色的训练数据状态
    
    Args:
        character_name: 角色名称
    
    Returns:
        dict: 角色状态信息
    """
    try:
        result = get_character_training_status(character_name)
        return {
            "success": True,
            "data": result,
        }
    except Exception as e:
        add_log(f"[角色训练] 获取角色状态失败: {e}", "error")
        return {
            "success": False,
            "error": f"获取角色状态失败: {str(e)}"
        }


@app.get("/character/{character_name}/video")
def get_character_video(character_name: str):
    """
    获取角色训练生成的视频文件
    
    Args:
        character_name: 角色名称
    
    Returns:
        FileResponse: 视频文件
    """
    try:
        from shared.config import PROJECT_ROOT
        
        character_dir = PROJECT_ROOT / "assets" / "charactor" / character_name
        video_file = character_dir / "output_NeRFFaceSpeech.mp4"
        
        if not video_file.exists():
            raise HTTPException(status_code=404, detail=f"视频文件不存在: {video_file}")
        
        return FileResponse(
            path=str(video_file),
            media_type="video/mp4",
            filename=f"{character_name}_output.mp4"
        )
    except HTTPException:
        raise
    except Exception as e:
        add_log(f"[角色训练] 获取视频文件失败: {e}", "error")
        raise HTTPException(status_code=500, detail=f"获取视频文件失败: {str(e)}")


# ---------------------------
# 日志查看接口
# ---------------------------
@app.get("/logs")
def get_logs(limit: int = 500, debug: bool = False):
    """
    获取日志输出
    
    Args:
        limit: 返回的日志条数限制
        debug: 是否只返回 debug 日志
    
    Returns:
        dict: 包含日志列表的响应
    """
    if debug:
        logs = list(DEBUG_LOG_BUFFER)[-limit:]
    else:
        logs = list(LOG_BUFFER)[-limit:]
    
    return {
        "success": True,
        "logs": logs,
        "total": len(DEBUG_LOG_BUFFER) if debug else len(LOG_BUFFER)
    }


@app.get("/logs/full")
def get_full_logs(debug: bool = False):
    """
    获取完整日志输出（无长度限制）
    
    Args:
        debug: 是否只返回 debug 日志
    
    Returns:
        dict: 包含完整日志列表的响应
    """
    if debug:
        logs = list(DEBUG_LOG_BUFFER)
    else:
        logs = list(LOG_BUFFER)
    
    return {
        "success": True,
        "logs": logs,
        "total": len(DEBUG_LOG_BUFFER) if debug else len(LOG_BUFFER)
    }


@app.post("/logs/clear")
def clear_logs(keep_count: int = 0):
    """
    清空日志缓冲区
    
    Args:
        keep_count: 保留的日志条数（0表示全部清空，大于0表示保留最新的N条）
    
    Returns:
        dict: 操作结果
    """
    try:
        if keep_count <= 0:
            # 全部清空
            original_count = len(LOG_BUFFER)
            original_debug_count = len(DEBUG_LOG_BUFFER)
            LOG_BUFFER.clear()
            DEBUG_LOG_BUFFER.clear()
            # 注意：由于清空后立即调用add_log，这条日志会被添加到缓冲区，所以清空后缓冲区会有1条日志
            # 但这是预期的行为，用户可以看到"日志已清空"的确认信息
            add_log(f"[日志管理] 所有日志已清空（正常日志: {original_count} 条，调试日志: {original_debug_count} 条）", "info")
            return {
                "success": True,
                "message": f"所有日志已清空（正常日志: {original_count} 条，调试日志: {original_debug_count} 条）",
                "remaining": 0,
                "removed_normal": original_count,
                "removed_debug": original_debug_count
            }
        else:
            # 保留最新的N条
            original_count = len(LOG_BUFFER)
            original_debug_count = len(DEBUG_LOG_BUFFER)
            
            # 将deque转换为列表，保留最后N条，然后清空并重新填充
            if len(LOG_BUFFER) > keep_count:
                kept_logs = list(LOG_BUFFER)[-keep_count:]
                LOG_BUFFER.clear()
                # 重新创建deque并添加保留的日志
                for log in kept_logs:
                    LOG_BUFFER.append(log)
            
            if len(DEBUG_LOG_BUFFER) > keep_count:
                kept_debug_logs = list(DEBUG_LOG_BUFFER)[-keep_count:]
                DEBUG_LOG_BUFFER.clear()
                # 重新创建deque并添加保留的日志
                for log in kept_debug_logs:
                    DEBUG_LOG_BUFFER.append(log)
            
            removed_normal = original_count - len(LOG_BUFFER)
            removed_debug = original_debug_count - len(DEBUG_LOG_BUFFER)
            add_log(f"[日志管理] 已清空 {removed_normal} 条正常日志和 {removed_debug} 条调试日志，保留最新 {len(LOG_BUFFER)} 条正常日志和 {len(DEBUG_LOG_BUFFER)} 条调试日志", "info")
            return {
                "success": True,
                "message": f"已清空 {removed_normal} 条正常日志和 {removed_debug} 条调试日志，保留最新 {len(LOG_BUFFER)} 条正常日志和 {len(DEBUG_LOG_BUFFER)} 条调试日志",
                "remaining": len(LOG_BUFFER),
                "remaining_debug": len(DEBUG_LOG_BUFFER),
                "removed": removed_normal,
                "removed_debug": removed_debug
            }
    except Exception as e:
        error_msg = f"清空日志失败: {e}"
        # 不使用add_log，因为可能日志系统本身有问题
        import traceback
        print(f"[日志管理] {error_msg}")
        print(f"[日志管理] 错误详情: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e)
    }


def add_log(message: str, level: str = "info", is_debug: bool = False):
    """
    添加日志到缓冲区
    
    Args:
        message: 日志消息
        level: 日志级别 (info, warning, error, debug, success)
        is_debug: 是否为 debug 日志
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message
    }
    
    if is_debug or level == "debug":
        DEBUG_LOG_BUFFER.append(log_entry)
    else:
        LOG_BUFFER.append(log_entry)


@app.get("/generation_records")
def get_generation_records_api(
    record_type: Optional[str] = None,  # 'video' 或 'chat'，None表示所有
    limit: int = 100,
    offset: int = 0
):
    """
    获取生成记录列表
    
    Args:
        record_type: 记录类型（'video' 或 'chat'），None表示所有类型
        limit: 返回记录数量限制
        offset: 偏移量
    
    Returns:
        dict: 包含记录列表的响应
    """
    try:
        add_log(f"[API] 查询生成记录: record_type={record_type}, limit={limit}, offset={offset}", "info")
        records = list_generation_records(record_type=record_type, limit=limit, offset=offset)
        add_log(f"[API] 查询到 {len(records)} 条记录", "info")
        
        # 直接返回路径，前端会根据路径构造URL
        # 不再在后端生成URL，统一使用路径
        
        return {
            "success": True,
            "data": records,
            "count": len(records)
        }
    except Exception as e:
        error_msg = f"查询生成记录失败: {str(e)}"
        add_log(f"[API] {error_msg}", "error")
        add_log(f"[API] 错误详情: {traceback.format_exc()}", "error")
        return {
            "success": False,
            "error": error_msg
        }


@app.delete("/generation_records/{unique_id}")
def delete_generation_record_api(unique_id: str):
    """
    删除生成记录
    """
    try:
        add_log(f"[API] 删除生成记录: unique_id={unique_id}", "info")
        success = delete_generation_record(unique_id)
        if success:
            return {"success": True, "message": "记录已删除"}
        else:
            return {"success": False, "error": "删除记录失败"}
    except Exception as e:
        error_msg = f"删除生成记录失败: {str(e)}"
        add_log(f"[API] {error_msg}", "error")
        return {"success": False, "error": error_msg}


@app.get("/generation_records/{unique_id}")
def get_generation_record_api(unique_id: str):
    """
    获取单个生成记录的详细信息
    """
    try:
        record = get_generation_record(unique_id)
        if not record:
            return {"success": False, "error": f"记录 {unique_id} 不存在"}
        
        # 直接返回路径，前端会根据路径构造URL
        # 不再在后端生成URL，统一使用路径
        
        return {"success": True, "data": record}
    except Exception as e:
        error_msg = f"获取生成记录失败: {str(e)}"
        add_log(f"[API] {error_msg}", "error")
        return {"success": False, "error": error_msg}


@app.get("/favicon.ico")
async def get_favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


# 初始化时添加一条日志
add_log("后端服务已启动", "success")
