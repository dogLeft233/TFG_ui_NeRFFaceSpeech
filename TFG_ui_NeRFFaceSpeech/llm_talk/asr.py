import torch
import io
import base64
import logging
import tempfile
import os
from typing import Optional, Dict, Any, Union
import soundfile as sf
import numpy as np

# 配置日志
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ASR_MODEL = None  # 延迟加载
MODEL_NAME = "base"  # 默认模型：tiny, base, small, medium, large

#---------------------------------------------------------------------

class ASRError(Exception):
    """自定义ASR异常类"""
    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        self.message = message
        self.error_code = error_code
        self.original_error = original_error
        super().__init__(self.message)
        
#---------------------------------------------------------------------

def load_asr_model(model_name: str = "base"):
    """
    延迟加载Whisper ASR模型
    
    Args:
        model_name: Whisper模型名称 (tiny, base, small, medium, large)
    
    Returns:
        加载的Whisper模型
    """
    global ASR_MODEL, MODEL_NAME
    
    # 如果模型已加载且模型名称相同，直接返回
    if ASR_MODEL is not None and MODEL_NAME == model_name:
        return ASR_MODEL
    
    try:
        import whisper
        
        logger.info(f"正在加载Whisper模型 '{model_name}' 到设备: {DEVICE}")
        
        # 如果已加载了不同模型，先释放
        if ASR_MODEL is not None:
            unload_asr_model()
        
        # 加载新模型
        ASR_MODEL = whisper.load_model(model_name, device=DEVICE)
        MODEL_NAME = model_name
        
        logger.info(f"Whisper模型 '{model_name}' 加载成功")
        return ASR_MODEL
        
    except ImportError:
        error_msg = "未安装whisper库，请运行: pip install openai-whisper"
        logger.error(error_msg)
        raise ASRError(error_msg, "WHISPER_NOT_INSTALLED")
    except Exception as e:
        error_msg = f"Whisper模型加载失败: {str(e)}"
        logger.error(error_msg)
        raise ASRError(error_msg, "MODEL_LOAD_ERROR", e)

def unload_asr_model():
    """
    从内存中释放ASR模型
    
    Returns:
        bool: 是否成功释放
    """
    global ASR_MODEL, MODEL_NAME
    
    try:
        if ASR_MODEL is not None:
            logger.info("正在释放Whisper模型...")
            
            # 删除模型引用
            del ASR_MODEL
            ASR_MODEL = None
            MODEL_NAME = "base"
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 如果使用CUDA，清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Whisper模型已成功释放")
            return True
        else:
            logger.info("Whisper模型未加载，无需释放")
            return True
            
    except Exception as e:
        error_msg = f"Whisper模型释放失败: {str(e)}"
        logger.error(error_msg)
        # 即使释放失败，也尝试重置全局变量
        ASR_MODEL = None
        MODEL_NAME = "base"
        return False

def reload_asr_model(model_name: str = "base"):
    """
    重新加载ASR模型（先释放再加载）
    
    Args:
        model_name: Whisper模型名称
    
    Returns:
        bool: 是否成功重新加载
    """
    try:
        logger.info(f"开始重新加载Whisper模型 '{model_name}'...")
        
        # 先释放现有模型
        unload_success = unload_asr_model()
        if not unload_success:
            logger.warning("模型释放失败，但继续尝试重新加载")
        
        # 重新加载模型
        load_asr_model(model_name)
        
        logger.info(f"Whisper模型 '{model_name}' 重新加载成功")
        return True
        
    except Exception as e:
        error_msg = f"Whisper模型重新加载失败: {str(e)}"
        logger.error(error_msg)
        raise ASRError(error_msg, "MODEL_RELOAD_ERROR", e)

def get_model_status():
    """
    获取ASR模型状态
    
    Returns:
        dict: 模型状态信息
    """
    global ASR_MODEL, MODEL_NAME
    
    status = {
        'loaded': ASR_MODEL is not None,
        'model_name': MODEL_NAME,
        'device': DEVICE,
        'cuda_available': torch.cuda.is_available(),
        'memory_info': {}
    }
    
    if ASR_MODEL is not None:
        status['model_type'] = type(ASR_MODEL).__name__
        
        # 获取内存使用情况
        if torch.cuda.is_available():
            status['memory_info'] = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated(),
                'max_reserved': torch.cuda.max_memory_reserved()
            }
    
    return status

#---------------------------------------------------------------------

def transcribe_audio_file(audio_path: str,
                         model_name: str = "base",
                         language: Optional[str] = None,
                         task: str = "transcribe",
                         **kwargs) -> Dict[str, Any]:
    """
    从音频文件进行语音识别
    
    Args:
        audio_path: 音频文件路径
        model_name: Whisper模型名称
        language: 语言代码 (如 'zh', 'en')，None表示自动检测
        task: 任务类型 ('transcribe' 或 'translate')
        **kwargs: 其他Whisper参数
    
    Returns:
        Dict[str, Any]: 识别结果
        {
            'success': bool,
            'data': {
                'text': str,  # 识别的文本
                'language': str,  # 检测到的语言
                'segments': list,  # 分段信息
                'info': dict  # 其他信息
            },
            'error': None or dict
        }
    """
    try:
        # 输入验证
        if not audio_path or not isinstance(audio_path, str):
            raise ASRError("音频文件路径不能为空且必须是字符串", "INVALID_INPUT")
        
        if not os.path.exists(audio_path):
            raise ASRError(f"音频文件不存在: {audio_path}", "FILE_NOT_FOUND")
        
        logger.info(f"开始语音识别，音频文件: {audio_path}")
        logger.info(f"使用模型: {model_name}, 语言: {language or '自动检测'}, 任务: {task}")
        
        # 加载模型
        model = load_asr_model(model_name)
        
        # 执行识别
        result = model.transcribe(
            audio_path,
            language=language,
            task=task,
            **kwargs
        )
        
        # 验证结果
        if not result or 'text' not in result:
            raise ASRError("Whisper返回空结果", "EMPTY_RESULT")
        
        text = result.get('text', '').strip()
        detected_language = result.get('language', 'unknown')
        segments = result.get('segments', [])
        
        logger.info(f"语音识别成功，文本长度: {len(text)}")
        logger.info(f"检测到的语言: {detected_language}")
        logger.info(f"识别文本: {text[:100]}...")
        
        return {
            'success': True,
            'data': {
                'text': text,
                'language': detected_language,
                'segments': segments,
                'info': {k: v for k, v in result.items() if k not in ['text', 'language', 'segments']}
            },
            'error': None
        }
        
    except ASRError:
        # 重新抛出ASR错误
        raise
    except Exception as e:
        # 捕获其他异常并转换为ASR错误
        error_msg = f"语音识别时发生未知错误: {str(e)}"
        logger.error(error_msg)
        raise ASRError(error_msg, "TRANSCRIPTION_ERROR", e)

def transcribe_audio_data(audio_data: Union[bytes, np.ndarray],
                         sample_rate: Optional[int] = None,
                         model_name: str = "base",
                         language: Optional[str] = None,
                         task: str = "transcribe",
                         **kwargs) -> Dict[str, Any]:
    """
    从音频数据进行语音识别
    
    Args:
        audio_data: 音频数据 (bytes格式的WAV数据，或numpy数组)
        sample_rate: 采样率（如果audio_data是numpy数组）
        model_name: Whisper模型名称
        language: 语言代码 (如 'zh', 'en')，None表示自动检测
        task: 任务类型 ('transcribe' 或 'translate')
        **kwargs: 其他Whisper参数
    
    Returns:
        Dict[str, Any]: 识别结果
    """
    try:
        # 输入验证
        if audio_data is None:
            raise ASRError("音频数据不能为空", "INVALID_INPUT")
        
        logger.info(f"开始语音识别，数据类型: {type(audio_data)}")
        
        # 创建临时文件保存音频数据
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
            try:
                # 处理不同类型的音频数据
                if isinstance(audio_data, bytes):
                    # 如果是bytes，直接写入文件
                    tmp_file.write(audio_data)
                    tmp_file.flush()
                elif isinstance(audio_data, np.ndarray):
                    # 如果是numpy数组，需要指定采样率
                    if sample_rate is None:
                        raise ASRError("numpy数组格式需要提供sample_rate参数", "MISSING_SAMPLE_RATE")
                    
                    # 确保音频数据是1D数组
                    if audio_data.ndim > 1:
                        audio_data = audio_data.flatten()
                    
                    # 保存为WAV文件
                    sf.write(tmp_path, audio_data, sample_rate, format='WAV')
                else:
                    raise ASRError(f"不支持的音频数据类型: {type(audio_data)}", "UNSUPPORTED_AUDIO_TYPE")
                
                # 调用文件识别函数
                result = transcribe_audio_file(
                    tmp_path,
                    model_name=model_name,
                    language=language,
                    task=task,
                    **kwargs
                )
                
                return result
                
            finally:
                # 清理临时文件
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {str(e)}")
        
    except ASRError:
        # 重新抛出ASR错误
        raise
    except Exception as e:
        # 捕获其他异常并转换为ASR错误
        error_msg = f"音频数据识别时发生未知错误: {str(e)}"
        logger.error(error_msg)
        raise ASRError(error_msg, "AUDIO_DATA_TRANSCRIPTION_ERROR", e)

def transcribe_base64_audio(base64_data: str,
                           model_name: str = "base",
                           language: Optional[str] = None,
                           task: str = "transcribe",
                           **kwargs) -> Dict[str, Any]:
    """
    从Base64编码的音频数据进行语音识别
    
    Args:
        base64_data: Base64编码的音频数据
        model_name: Whisper模型名称
        language: 语言代码 (如 'zh', 'en')，None表示自动检测
        task: 任务类型 ('transcribe' 或 'translate')
        **kwargs: 其他Whisper参数
    
    Returns:
        Dict[str, Any]: 识别结果
    """
    try:
        # 输入验证
        if not base64_data or not isinstance(base64_data, str):
            raise ASRError("Base64数据不能为空且必须是字符串", "INVALID_INPUT")
        
        logger.info("开始解码Base64音频数据...")
        
        # 解码Base64数据
        try:
            audio_bytes = base64.b64decode(base64_data)
        except Exception as e:
            raise ASRError(f"Base64解码失败: {str(e)}", "BASE64_DECODE_ERROR")
        
        # 调用音频数据识别函数
        result = transcribe_audio_data(
            audio_bytes,
            model_name=model_name,
            language=language,
            task=task,
            **kwargs
        )
        
        return result
        
    except ASRError:
        # 重新抛出ASR错误
        raise
    except Exception as e:
        # 捕获其他异常并转换为ASR错误
        error_msg = f"Base64音频识别时发生未知错误: {str(e)}"
        logger.error(error_msg)
        raise ASRError(error_msg, "BASE64_TRANSCRIPTION_ERROR", e)

#---------------------------------------------------------------------

def get_asr_response_api(audio_input: Union[str, bytes, np.ndarray, str],
                         model_name: str = "base",
                         language: Optional[str] = None,
                         task: str = "transcribe",
                         sample_rate: Optional[int] = None,
                         **kwargs) -> Dict[str, Any]:
    """
    为前端提供的ASR API接口，支持多种输入格式
    
    Args:
        audio_input: 音频输入，可以是：
            - str: 文件路径或Base64编码的字符串
            - bytes: WAV文件的二进制数据
            - np.ndarray: 音频numpy数组
        model_name: Whisper模型名称
        language: 语言代码
        task: 任务类型 ('transcribe' 或 'translate')
        sample_rate: 采样率（仅当audio_input是numpy数组时需要）
        **kwargs: 其他Whisper参数
    
    Returns:
        Dict[str, Any]: 标准化的API响应
    """
    try:
        # 根据输入类型选择相应的处理函数
        if isinstance(audio_input, str):
            # 判断是文件路径还是Base64字符串
            if os.path.exists(audio_input):
                # 文件路径
                result = transcribe_audio_file(
                    audio_input,
                    model_name=model_name,
                    language=language,
                    task=task,
                    **kwargs
                )
            else:
                # Base64字符串
                result = transcribe_base64_audio(
                    audio_input,
                    model_name=model_name,
                    language=language,
                    task=task,
                    **kwargs
                )
        elif isinstance(audio_input, bytes):
            # 二进制数据
            result = transcribe_audio_data(
                audio_input,
                model_name=model_name,
                language=language,
                task=task,
                **kwargs
            )
        elif isinstance(audio_input, np.ndarray):
            # numpy数组
            result = transcribe_audio_data(
                audio_input,
                sample_rate=sample_rate,
                model_name=model_name,
                language=language,
                task=task,
                **kwargs
            )
        else:
            raise ASRError(f"不支持的输入类型: {type(audio_input)}", "UNSUPPORTED_INPUT_TYPE")
        
        return result
        
    except ASRError as e:
        return {
            'success': False,
            'data': None,
            'error': {
                'code': e.error_code,
                'message': e.message,
                'type': 'ASRError'
            }
        }
    except Exception as e:
        return {
            'success': False,
            'data': None,
            'error': {
                'code': 'UNKNOWN_ERROR',
                'message': f"未知错误: {str(e)}",
                'type': 'Exception'
            }
        }

def manage_asr_model(action: str, model_name: str = "base") -> Dict[str, Any]:
    """
    管理ASR模型的API接口
    
    Args:
        action: 操作类型 ('load', 'unload', 'reload', 'status')
        model_name: 模型名称（仅在load和reload时需要）
    
    Returns:
        Dict[str, Any]: 操作结果
    """
    try:
        if action == 'load':
            load_asr_model(model_name)
            return {
                'success': True,
                'message': f'Whisper模型 {model_name} 加载成功',
                'action': 'load'
            }
            
        elif action == 'unload':
            success = unload_asr_model()
            return {
                'success': success,
                'message': 'Whisper模型释放成功' if success else 'Whisper模型释放失败',
                'action': 'unload'
            }
            
        elif action == 'reload':
            reload_asr_model(model_name)
            return {
                'success': True,
                'message': f'Whisper模型 {model_name} 重新加载成功',
                'action': 'reload'
            }
            
        elif action == 'status':
            status = get_model_status()
            return {
                'success': True,
                'data': status,
                'action': 'status'
            }
            
        else:
            return {
                'success': False,
                'error': {
                    'code': 'INVALID_ACTION',
                    'message': f'无效的操作: {action}。支持的操作: load, unload, reload, status',
                    'type': 'ValueError'
                }
            }
            
    except ASRError as e:
        return {
            'success': False,
            'error': {
                'code': e.error_code,
                'message': e.message,
                'type': 'ASRError'
            }
        }
    except Exception as e:
        return {
            'success': False,
            'error': {
                'code': 'UNKNOWN_ERROR',
                'message': f'未知错误: {str(e)}',
                'type': 'Exception'
            }
        }

#---------------------------------------------------------------------

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    print("=== ASR测试开始 ===")
    print("注意：此测试需要提供音频文件路径")
    print("使用示例: python asr.py <audio_file_path>")
    
    import sys
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        try:
            result = get_asr_response_api(audio_file)
            if result['success']:
                print(f"✅ 识别成功")
                print(f"📝 识别文本: {result['data']['text']}")
                print(f"🌐 检测语言: {result['data']['language']}")
            else:
                print(f"❌ 识别失败: {result['error']['message']}")
        except Exception as e:
            print(f"💥 测试过程中发生异常: {str(e)}")
    else:
        print("请提供音频文件路径作为参数")

