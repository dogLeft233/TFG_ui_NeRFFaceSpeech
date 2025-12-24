"""
TTS 服务客户端
通过 HTTP 调用独立的 TTS 服务，避免每次重新加载模型
"""
import os
import sys
import json
import logging
import requests
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# 导入配置
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import TTS_SERVICE_URL, get_character_audio_prompt

logger = logging.getLogger(__name__)

# 请求超时设置（秒）
REQUEST_TIMEOUT = 300  # 5分钟，TTS生成可能需要较长时间

def check_tts_service_health() -> bool:
    """检查 TTS 服务是否可用"""
    try:
        response = requests.get(
            f"{TTS_SERVICE_URL}/health",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("model_loaded", False)
        return False
    except Exception as e:
        logger.warning(f"TTS 服务健康检查失败: {e}")
        return False

def generate_audio_via_service(
    text: str,
    output_path: str,
    character: str,
    language_id: str = "zh"
) -> Tuple[bool, Optional[str]]:
    """
    通过 TTS 服务生成音频文件
    
    Args:
        text: 要转换的文本
        output_path: 输出音频文件路径
        character: 角色名称（用于获取音频提示）
        language_id: 语言ID
    
    Returns:
        tuple[bool, str | None]: (成功标志, LLM回答文本)
    """
    try:
        # 获取角色音频提示路径
        try:
            audio_prompt = get_character_audio_prompt(character)
            audio_prompt_path = str(audio_prompt.resolve())
        except ValueError as e:
            logger.error(f"获取角色音频提示失败: {e}")
            return False, None
        
        # 准备请求数据
        request_data = {
            "text": text,
            "output_path": output_path,
            "language_id": language_id,
            "audio_prompt_path": audio_prompt_path
        }
        
        logger.info(f"[TTS Client] 调用 TTS 服务生成音频")
        logger.info(f"[TTS Client] 文本长度: {len(text)}")
        logger.info(f"[TTS Client] 输出路径: {output_path}")
        logger.info(f"[TTS Client] 角色: {character}")
        
        # 调用 TTS 服务
        response = requests.post(
            f"{TTS_SERVICE_URL}/api/tts/generate_file",
            json=request_data,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                logger.info(f"[TTS Client] ✅ 音频生成成功: {output_path}")
                # 注意：TTS 服务只生成音频，不返回 LLM 回答
                # 如果需要 LLM 回答，需要使用 /api/talk 接口
                return True, None
            else:
                error_msg = result.get("error", {}).get("message", "未知错误")
                logger.error(f"[TTS Client] ❌ 音频生成失败: {error_msg}")
                return False, None
        else:
            error_detail = response.text
            logger.error(f"[TTS Client] ❌ TTS 服务返回错误: {response.status_code}")
            logger.error(f"[TTS Client] 错误详情: {error_detail}")
            return False, None
            
    except requests.exceptions.Timeout:
        logger.error(f"[TTS Client] ❌ 请求超时（>{REQUEST_TIMEOUT}秒）")
        return False, None
    except requests.exceptions.ConnectionError:
        logger.error(f"[TTS Client] ❌ 无法连接到 TTS 服务: {TTS_SERVICE_URL}")
        logger.error(f"[TTS Client] 请确保 TTS 服务正在运行")
        return False, None
    except Exception as e:
        logger.error(f"[TTS Client] ❌ 调用 TTS 服务时发生错误: {e}")
        import traceback
        logger.error(f"[TTS Client] 错误详情: {traceback.format_exc()}")
        return False, None

def talk_with_audio_via_service(
    user_input: str,
    output_path: str,
    character: str,
    language_id: str = "zh",
    combine_audio: bool = True,
    split_sentences: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    通过 TTS 服务进行完整对话（LLM + TTS）
    
    Args:
        user_input: 用户输入文本
        output_path: 输出音频文件路径
        character: 角色名称（用于获取音频提示）
        language_id: 语言ID
        combine_audio: 是否合并音频
        split_sentences: 是否分句处理
    
    Returns:
        tuple[bool, str | None]: (成功标志, LLM回答文本)
    """
    try:
        # 获取角色音频提示路径
        try:
            audio_prompt = get_character_audio_prompt(character)
            audio_prompt_path = str(audio_prompt.resolve())
        except ValueError as e:
            logger.error(f"获取角色音频提示失败: {e}")
            return False, None
        
        # 准备请求数据
        request_data = {
            "user_input": user_input,
            "language_id": language_id,
            "audio_prompt_path": audio_prompt_path,
            "combine_audio": combine_audio,
            "split_sentences": split_sentences
        }
        
        logger.info(f"[TTS Client] 调用 TTS 服务进行对话")
        logger.info(f"[TTS Client] 用户输入: {user_input[:100]}...")
        logger.info(f"[TTS Client] 输出路径: {output_path}")
        logger.info(f"[TTS Client] 角色: {character}")
        
        # 调用 TTS 服务的 talk 接口
        response = requests.post(
            f"{TTS_SERVICE_URL}/api/talk",
            json=request_data,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                # 获取 LLM 回答
                llm_answer = result.get("data", {}).get("llm_answer", "")
                
                # 保存音频文件
                audio_saved = False
                combined_audio = result.get("data", {}).get("combined_audio")
                if combined_audio and combined_audio.get("success"):
                    # 优先使用二进制数据，如果没有则使用 base64
                    import base64
                    wav_data = None
                    if combined_audio.get("combined_audio_data"):
                        # 如果是二进制数据（bytes），直接使用
                        wav_data = combined_audio.get("combined_audio_data")
                        if isinstance(wav_data, str):
                            # 如果是 base64 字符串，需要解码
                            wav_data = base64.b64decode(wav_data)
                    elif combined_audio.get("combined_base64_data"):
                        # 使用 base64 数据
                        base64_data = combined_audio.get("combined_base64_data")
                        wav_data = base64.b64decode(base64_data)
                    
                    if wav_data:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        with open(output_path, 'wb') as f:
                            f.write(wav_data)
                        audio_saved = True
                
                if audio_saved:
                    logger.info(f"[TTS Client] ✅ 对话完成，音频已保存: {output_path}")
                    logger.info(f"[TTS Client] LLM 回答长度: {len(llm_answer)}")
                    return True, llm_answer
                else:
                    logger.warning(f"[TTS Client] ⚠️ 对话完成但音频保存失败")
                    return False, llm_answer
            else:
                error_msg = result.get("error", {}).get("message", "未知错误")
                logger.error(f"[TTS Client] ❌ 对话失败: {error_msg}")
                return False, None
        else:
            error_detail = response.text
            logger.error(f"[TTS Client] ❌ TTS 服务返回错误: {response.status_code}")
            logger.error(f"[TTS Client] 错误详情: {error_detail}")
            return False, None
            
    except requests.exceptions.Timeout:
        logger.error(f"[TTS Client] ❌ 请求超时（>{REQUEST_TIMEOUT}秒）")
        return False, None
    except requests.exceptions.ConnectionError:
        logger.error(f"[TTS Client] ❌ 无法连接到 TTS 服务: {TTS_SERVICE_URL}")
        logger.error(f"[TTS Client] 请确保 TTS 服务正在运行")
        return False, None
    except Exception as e:
        logger.error(f"[TTS Client] ❌ 调用 TTS 服务时发生错误: {e}")
        import traceback
        logger.error(f"[TTS Client] 错误详情: {traceback.format_exc()}")
        return False, None

