#!/usr/bin/env python3
"""
独立的 ASR 服务
在启动时预加载模型，提供 HTTP API 接口供后端调用
避免每次请求都重新加载模型
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入 ASR 模块
try:
    from llm_talk.asr import (
        load_asr_model,
        transcribe_audio_file,
        transcribe_audio_data,
        transcribe_base64_audio,
        manage_asr_model,
        ASRError
    )
except ImportError as e:
    logger.error(f"无法导入 ASR 模块: {e}")
    sys.exit(1)

# 创建 FastAPI 应用
app = FastAPI(
    title="ASR Service",
    description="独立的 ASR 服务，模型常驻内存",
    version="1.0.0"
)

# 全局变量：模型加载状态
MODEL_LOADED = False
MODEL_NAME = "base"

# ==================== 请求/响应模型 ====================

class ASRRequest(BaseModel):
    """ASR 请求模型（Base64）"""
    audio_base64: str = Field(..., description="Base64编码的音频数据")
    model_name: str = Field(default="base", description="Whisper模型名称")
    language: Optional[str] = Field(default=None, description="语言代码，None表示自动检测")
    task: str = Field(default="transcribe", description="任务类型：transcribe或translate")

class ASRFileRequest(BaseModel):
    """ASR 文件请求模型"""
    audio_path: str = Field(..., description="音频文件路径")
    model_name: str = Field(default="base", description="Whisper模型名称")
    language: Optional[str] = Field(default=None, description="语言代码，None表示自动检测")
    task: str = Field(default="transcribe", description="任务类型：transcribe或translate")

# ==================== 辅助函数 ====================

def ensure_json_serializable(obj):
    """确保对象可以 JSON 序列化"""
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, (set, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    else:
        try:
            import json
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

# ==================== 启动和关闭事件 ====================

@app.on_event("startup")
async def startup_event():
    """服务启动时预加载模型"""
    global MODEL_LOADED, MODEL_NAME
    try:
        logger.info("=" * 60)
        logger.info("ASR 服务启动中...")
        logger.info("=" * 60)
        logger.info(f"正在加载 Whisper 模型 '{MODEL_NAME}'...")
        load_asr_model(MODEL_NAME)
        MODEL_LOADED = True
        logger.info(f"✅ Whisper 模型 '{MODEL_NAME}' 加载成功，服务已就绪")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"❌ Whisper 模型加载失败: {e}")
        MODEL_LOADED = False
        # 不退出，允许后续重试

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时释放模型"""
    global MODEL_LOADED
    try:
        logger.info("正在释放 Whisper 模型...")
        manage_asr_model('unload')
        MODEL_LOADED = False
        logger.info("Whisper 模型已释放")
    except Exception as e:
        logger.error(f"释放 Whisper 模型时出错: {e}")

# ==================== API 端点 ====================

@app.get("/")
async def root():
    """根端点，返回服务信息"""
    return {
        "service": "ASR Service",
        "version": "1.0.0",
        "model_loaded": MODEL_LOADED,
        "model_name": MODEL_NAME,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "model_name": MODEL_NAME
    }

@app.post("/api/asr/transcribe")
async def transcribe_audio(request: ASRRequest):
    """
    从 Base64 音频数据进行语音识别
    """
    global MODEL_LOADED, MODEL_NAME
    
    # 如果请求的模型与当前加载的不同，需要重新加载
    if request.model_name != MODEL_NAME or not MODEL_LOADED:
        try:
            logger.info(f"加载模型: {request.model_name}")
            load_asr_model(request.model_name)
            MODEL_LOADED = True
            MODEL_NAME = request.model_name
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Whisper 模型加载失败: {str(e)}"
            )
    
    try:
        result = transcribe_base64_audio(
            base64_data=request.audio_base64,
            model_name=request.model_name,
            language=request.language,
            task=request.task
        )
        result = ensure_json_serializable(result)
        return JSONResponse(content=result)
    except ASRError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "code": e.error_code,
                "message": e.message,
                "type": "ASRError"
            }
        )
    except Exception as e:
        logger.error(f"ASR 识别失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"ASR 识别时发生错误: {str(e)}"
        )

@app.post("/api/asr/transcribe_file")
async def transcribe_audio_file_endpoint(request: ASRFileRequest):
    """
    从音频文件进行语音识别
    """
    global MODEL_LOADED, MODEL_NAME
    
    # 检查文件是否存在
    if not os.path.exists(request.audio_path):
        raise HTTPException(
            status_code=404,
            detail=f"音频文件不存在: {request.audio_path}"
        )
    
    # 如果请求的模型与当前加载的不同，需要重新加载
    if request.model_name != MODEL_NAME or not MODEL_LOADED:
        try:
            logger.info(f"加载模型: {request.model_name}")
            load_asr_model(request.model_name)
            MODEL_LOADED = True
            MODEL_NAME = request.model_name
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Whisper 模型加载失败: {str(e)}"
            )
    
    try:
        result = transcribe_audio_file(
            audio_path=request.audio_path,
            model_name=request.model_name,
            language=request.language,
            task=request.task
        )
        result = ensure_json_serializable(result)
        return JSONResponse(content=result)
    except ASRError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "code": e.error_code,
                "message": e.message,
                "type": "ASRError"
            }
        )
    except Exception as e:
        logger.error(f"ASR 识别失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"ASR 识别时发生错误: {str(e)}"
        )

@app.post("/api/asr/transcribe_upload")
async def transcribe_audio_upload(
    file: UploadFile = File(...),
    model_name: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe"
):
    """
    上传音频文件进行语音识别
    """
    global MODEL_LOADED, MODEL_NAME
    
    # 如果请求的模型与当前加载的不同，需要重新加载
    if model_name != MODEL_NAME or not MODEL_LOADED:
        try:
            logger.info(f"加载模型: {model_name}")
            load_asr_model(model_name)
            MODEL_LOADED = True
            MODEL_NAME = model_name
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Whisper 模型加载失败: {str(e)}"
            )
    
    try:
        # 读取上传的文件内容
        audio_data = await file.read()
        
        # 使用临时文件保存音频数据
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(audio_data)
        
        try:
            # 调用文件识别函数
            result = transcribe_audio_file(
                audio_path=tmp_path,
                model_name=model_name,
                language=language,
                task=task
            )
            result = ensure_json_serializable(result)
            return JSONResponse(content=result)
        finally:
            # 清理临时文件
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
                
    except ASRError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "code": e.error_code,
                "message": e.message,
                "type": "ASRError"
            }
        )
    except Exception as e:
        logger.error(f"ASR 识别失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"ASR 识别时发生错误: {str(e)}"
        )

@app.post("/api/model/{action}")
async def manage_model(action: str, model_name: str = "base"):
    """
    管理 ASR 模型（load, unload, reload, status）
    """
    global MODEL_LOADED, MODEL_NAME
    
    if action not in ['load', 'unload', 'reload', 'status']:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的操作: {action}"
        )
    
    try:
        result = manage_asr_model(action, model_name=model_name)
        
        if action == 'load':
            MODEL_LOADED = True
            MODEL_NAME = model_name
        elif action == 'unload':
            MODEL_LOADED = False
            MODEL_NAME = "base"
        elif action == 'reload':
            MODEL_LOADED = True
            MODEL_NAME = model_name
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"模型管理操作失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"模型管理操作失败: {str(e)}"
        )

# ==================== 主函数 ====================

def main():
    """启动 ASR 服务"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ASR Service")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务监听地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8002,
        help="服务监听端口"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="工作进程数（建议为1，因为模型在内存中）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper模型名称（默认: base）"
    )
    
    args = parser.parse_args()
    
    # 设置全局模型名称
    global MODEL_NAME
    MODEL_NAME = args.model
    
    logger.info(f"启动 ASR 服务: http://{args.host}:{args.port}")
    logger.info(f"使用模型: {MODEL_NAME}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )

if __name__ == "__main__":
    main()

