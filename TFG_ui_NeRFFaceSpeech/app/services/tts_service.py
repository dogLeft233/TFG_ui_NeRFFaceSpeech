#!/usr/bin/env python3
"""
独立的 TTS 服务
在启动时预加载模型，提供 HTTP API 接口供后端调用
避免每次请求都重新加载模型
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException
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

# 导入 TTS 模块
try:
    from llm_talk.tts import (
        load_tts_model,
        convert_text_to_wav_chatterbox,
        save_wav_to_file,
        manage_tts_model,
        TTSError
    )
    from llm_talk.talk import (
        talk_with_audio,
        split_text_to_sentences
    )
except ImportError as e:
    logger.error(f"无法导入 TTS 模块: {e}")
    sys.exit(1)

# 创建 FastAPI 应用
app = FastAPI(
    title="TTS Service",
    description="独立的 TTS 服务，模型常驻内存",
    version="1.0.0"
)

# 全局变量：模型加载状态
MODEL_LOADED = False

# ==================== 请求/响应模型 ====================

class TTSRequest(BaseModel):
    """TTS 请求模型"""
    text: str = Field(..., description="要转换的文本", min_length=1, max_length=1000)
    language_id: str = Field(default="zh", description="语言ID")
    audio_prompt_path: Optional[str] = Field(default=None, description="音频提示文件路径")
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.0, description="夸张程度")
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="CFG权重")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="温度参数")
    repetition_penalty: float = Field(default=1.5, ge=1.0, le=3.0, description="重复惩罚")
    min_p: float = Field(default=0.05, ge=0.0, le=1.0, description="最小概率")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="顶部概率")

class TalkRequest(BaseModel):
    """Talk 请求模型（LLM + TTS）"""
    user_input: str = Field(..., description="用户输入文本", min_length=1)
    language_id: str = Field(default="zh", description="语言ID")
    audio_prompt_path: Optional[str] = Field(default=None, description="音频提示文件路径")
    combine_audio: bool = Field(default=True, description="是否合并音频")
    split_sentences: bool = Field(default=True, description="是否分句处理")
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.0, description="夸张程度")
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="CFG权重")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="温度参数")
    repetition_penalty: float = Field(default=1.5, ge=1.0, le=3.0, description="重复惩罚")
    min_p: float = Field(default=0.05, ge=0.0, le=1.0, description="最小概率")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="顶部概率")

class GenerateAudioRequest(BaseModel):
    """生成音频文件请求模型"""
    text: str = Field(..., description="要转换的文本", min_length=1)
    output_path: str = Field(..., description="输出音频文件路径")
    language_id: str = Field(default="zh", description="语言ID")
    audio_prompt_path: Optional[str] = Field(default=None, description="音频提示文件路径")
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.0, description="夸张程度")
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="CFG权重")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="温度参数")
    repetition_penalty: float = Field(default=1.5, ge=1.0, le=3.0, description="重复惩罚")
    min_p: float = Field(default=0.05, ge=0.0, le=1.0, description="最小概率")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="顶部概率")

# ==================== 辅助函数 ====================

def ensure_json_serializable(obj):
    """确保对象可以 JSON 序列化（将 bytes 转换为 base64）"""
    import base64
    
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode('utf-8')
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, (set, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    else:
        try:
            # 尝试直接返回，如果无法序列化会在 JSONResponse 时抛出异常
            import json
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

# ==================== 启动和关闭事件 ====================

@app.on_event("startup")
async def startup_event():
    """服务启动时预加载模型"""
    global MODEL_LOADED
    try:
        logger.info("=" * 60)
        logger.info("TTS 服务启动中...")
        logger.info("=" * 60)
        logger.info("正在加载 TTS 模型...")
        load_tts_model()
        MODEL_LOADED = True
        logger.info("✅ TTS 模型加载成功，服务已就绪")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"❌ TTS 模型加载失败: {e}")
        MODEL_LOADED = False
        # 不退出，允许后续重试

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时释放模型"""
    global MODEL_LOADED
    try:
        logger.info("正在释放 TTS 模型...")
        manage_tts_model('unload')
        MODEL_LOADED = False
        logger.info("TTS 模型已释放")
    except Exception as e:
        logger.error(f"释放 TTS 模型时出错: {e}")

# ==================== API 端点 ====================

@app.get("/")
async def root():
    """根端点，返回服务信息"""
    return {
        "service": "TTS Service",
        "version": "1.0.0",
        "model_loaded": MODEL_LOADED,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED
    }

@app.post("/api/tts/generate")
async def generate_tts(request: TTSRequest):
    """
    生成 TTS 音频（返回音频数据）
    """
    global MODEL_LOADED
    if not MODEL_LOADED:
        try:
            load_tts_model()
            MODEL_LOADED = True
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"TTS 模型未加载且加载失败: {str(e)}"
            )
    
    try:
        result = convert_text_to_wav_chatterbox(
            text=request.text,
            language_id=request.language_id,
            audio_prompt_path=request.audio_prompt_path,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            min_p=request.min_p,
            top_p=request.top_p
        )
        return JSONResponse(content=result)
    except TTSError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "code": e.error_code,
                "message": e.message,
                "type": "TTSError"
            }
        )
    except Exception as e:
        logger.error(f"TTS 生成失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"TTS 生成时发生错误: {str(e)}"
        )

@app.post("/api/tts/generate_file")
async def generate_tts_file(request: GenerateAudioRequest):
    """
    生成 TTS 音频并保存到文件
    """
    global MODEL_LOADED
    if not MODEL_LOADED:
        try:
            load_tts_model()
            MODEL_LOADED = True
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"TTS 模型未加载且加载失败: {str(e)}"
            )
    
    try:
        # 生成音频数据
        result = convert_text_to_wav_chatterbox(
            text=request.text,
            language_id=request.language_id,
            audio_prompt_path=request.audio_prompt_path,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            min_p=request.min_p,
            top_p=request.top_p
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'TTS 生成失败')
            )
        
        # 保存到文件
        wav_data = result['data']['wav_data']
        success = save_wav_to_file(wav_data, request.output_path)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="保存音频文件失败"
            )
        
        return {
            "success": True,
            "data": {
                "output_path": request.output_path,
                "duration": result['data']['duration'],
                "sample_rate": result['data']['sample_rate']
            },
            "error": None
        }
    except HTTPException:
        raise
    except TTSError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "code": e.error_code,
                "message": e.message,
                "type": "TTSError"
            }
        )
    except Exception as e:
        logger.error(f"生成音频文件失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"生成音频文件时发生错误: {str(e)}"
        )

@app.post("/api/talk")
async def talk(request: TalkRequest):
    """
    完整的对话功能：用户输入 -> LLM回答 -> TTS音频生成
    """
    global MODEL_LOADED
    if not MODEL_LOADED:
        try:
            load_tts_model()
            MODEL_LOADED = True
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"TTS 模型未加载且加载失败: {str(e)}"
            )
    
    try:
        # 准备 TTS 参数
        tts_kwargs = {
            "audio_prompt_path": request.audio_prompt_path,
            "exaggeration": request.exaggeration,
            "cfg_weight": request.cfg_weight,
            "temperature": request.temperature,
            "repetition_penalty": request.repetition_penalty,
            "min_p": request.min_p,
            "top_p": request.top_p
        }
        
        # 调用 talk_with_audio
        result = talk_with_audio(
            user_input=request.user_input,
            language_id=request.language_id,
            combine_audio=request.combine_audio,
            release_tts_model=False,  # 不释放模型，保持常驻
            split_sentences=request.split_sentences,
            **tts_kwargs
        )
        
        # 确保结果可以 JSON 序列化
        result = ensure_json_serializable(result)
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Talk 处理失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Talk 处理时发生错误: {str(e)}"
        )

@app.post("/api/model/{action}")
async def manage_model(action: str):
    """
    管理 TTS 模型（load, unload, reload, status）
    """
    if action not in ['load', 'unload', 'reload', 'status']:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的操作: {action}"
        )
    
    try:
        result = manage_tts_model(action)
        global MODEL_LOADED
        if action == 'load':
            MODEL_LOADED = True
        elif action == 'unload':
            MODEL_LOADED = False
        elif action == 'reload':
            MODEL_LOADED = True
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"模型管理操作失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"模型管理操作失败: {str(e)}"
        )

# ==================== 主函数 ====================

def main():
    """启动 TTS 服务"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TTS Service")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务监听地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="服务监听端口"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="工作进程数（建议为1，因为模型在内存中）"
    )
    
    args = parser.parse_args()
    
    logger.info(f"启动 TTS 服务: http://{args.host}:{args.port}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )

if __name__ == "__main__":
    main()

