"""
ASR 服务客户端
通过 HTTP 调用独立的 ASR 服务，避免每次重新加载模型
也支持通过 subprocess 调用封装好的 llm_talk 环境
"""
import os
import sys
import json
import logging
import requests
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# 导入配置
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.config import ASR_SERVICE_URL, LLM_CONDA_PYTHON, LLM_CONDA_ENV, PROJECT_ROOT

logger = logging.getLogger(__name__)

# 请求超时设置（秒）
REQUEST_TIMEOUT = 300  # 5分钟，ASR识别可能需要较长时间

def check_asr_service_health() -> bool:
    """检查 ASR 服务是否可用"""
    try:
        response = requests.get(
            f"{ASR_SERVICE_URL}/health",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("model_loaded", False)
        return False
    except Exception as e:
        logger.warning(f"ASR 服务健康检查失败: {e}")
        return False

def transcribe_audio_via_service(
    audio_input: str,
    model_name: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe"
) -> Tuple[bool, Optional[str]]:
    """
    通过 ASR 服务进行语音识别
    
    Args:
        audio_input: 音频输入，可以是文件路径或Base64字符串
        model_name: Whisper模型名称
        language: 语言代码，None表示自动检测
        task: 任务类型（transcribe或translate）
    
    Returns:
        tuple[bool, str | None]: (成功标志, 识别文本)
    """
    try:
        # 判断输入类型
        if os.path.exists(audio_input):
            # 文件路径
            request_data = {
                "audio_path": audio_input,
                "model_name": model_name,
                "language": language,
                "task": task
            }
            
            logger.info(f"[ASR Client] 调用 ASR 服务识别音频文件")
            logger.info(f"[ASR Client] 文件路径: {audio_input}")
            logger.info(f"[ASR Client] 模型: {model_name}")
            
            # 调用 ASR 服务的文件识别接口
            response = requests.post(
                f"{ASR_SERVICE_URL}/api/asr/transcribe_file",
                json=request_data,
                timeout=REQUEST_TIMEOUT
            )
        else:
            # Base64字符串
            request_data = {
                "audio_base64": audio_input,
                "model_name": model_name,
                "language": language,
                "task": task
            }
            
            logger.info(f"[ASR Client] 调用 ASR 服务识别Base64音频")
            logger.info(f"[ASR Client] Base64长度: {len(audio_input)}")
            logger.info(f"[ASR Client] 模型: {model_name}")
            
            # 调用 ASR 服务的Base64识别接口
            response = requests.post(
                f"{ASR_SERVICE_URL}/api/asr/transcribe",
                json=request_data,
                timeout=REQUEST_TIMEOUT
            )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                text = result.get("data", {}).get("text", "")
                detected_language = result.get("data", {}).get("language", "unknown")
                logger.info(f"[ASR Client] ✅ 识别成功")
                logger.info(f"[ASR Client] 识别文本: {text[:100]}...")
                logger.info(f"[ASR Client] 检测语言: {detected_language}")
                return True, text
            else:
                error_msg = result.get("error", {}).get("message", "未知错误")
                logger.error(f"[ASR Client] ❌ 识别失败: {error_msg}")
                return False, None
        else:
            error_detail = response.text
            logger.error(f"[ASR Client] ❌ ASR 服务返回错误: {response.status_code}")
            logger.error(f"[ASR Client] 错误详情: {error_detail}")
            return False, None
            
    except requests.exceptions.Timeout:
        logger.error(f"[ASR Client] ❌ 请求超时（>{REQUEST_TIMEOUT}秒）")
        return False, None
    except requests.exceptions.ConnectionError:
        logger.error(f"[ASR Client] ❌ 无法连接到 ASR 服务: {ASR_SERVICE_URL}")
        logger.error(f"[ASR Client] 请确保 ASR 服务正在运行")
        return False, None
    except Exception as e:
        logger.error(f"[ASR Client] ❌ 调用 ASR 服务时发生错误: {e}")
        import traceback
        logger.error(f"[ASR Client] 错误详情: {traceback.format_exc()}")
        return False, None

def transcribe_audio_file_via_service(
    audio_path: str,
    model_name: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe"
) -> Tuple[bool, Optional[str]]:
    """
    通过 ASR 服务识别音频文件
    
    Args:
        audio_path: 音频文件路径
        model_name: Whisper模型名称
        language: 语言代码，None表示自动检测
        task: 任务类型（transcribe或translate）
    
    Returns:
        tuple[bool, str | None]: (成功标志, 识别文本)
    """
    return transcribe_audio_via_service(
        audio_input=audio_path,
        model_name=model_name,
        language=language,
        task=task
    )

def transcribe_base64_audio_via_service(
    audio_base64: str,
    model_name: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe"
) -> Tuple[bool, Optional[str]]:
    """
    通过 ASR 服务识别Base64音频
    
    Args:
        audio_base64: Base64编码的音频数据
        model_name: Whisper模型名称
        language: 语言代码，None表示自动检测
        task: 任务类型（transcribe或translate）
    
    Returns:
        tuple[bool, str | None]: (成功标志, 识别文本)
    """
    return transcribe_audio_via_service(
        audio_input=audio_base64,
        model_name=model_name,
        language=language,
        task=task
    )


def transcribe_base64_audio_via_subprocess(
    audio_base64: str,
    model_name: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe"
) -> Dict[str, Any]:
    """
    通过 subprocess 调用封装好的 llm_talk 环境进行语音识别
    
    Args:
        audio_base64: Base64编码的音频数据
        model_name: Whisper模型名称
        language: 语言代码，None表示自动检测
        task: 任务类型（transcribe或translate）
    
    Returns:
        dict: ASR API 响应结果（包含 success, error, data 字段）
    """
    logger.info(f"[ASR Subprocess] 开始语音识别: model={model_name}, language={language}, task={task}, audio_base64_length={len(audio_base64)}")
    
    if not LLM_CONDA_PYTHON.exists():
        error_msg = f"LLM conda 环境不存在: {LLM_CONDA_PYTHON}"
        logger.error(f"[ASR Subprocess] {error_msg}")
        return {
            "success": False,
            "error": {
                "code": "ENV_NOT_FOUND",
                "message": error_msg,
                "type": "FileNotFoundError"
            },
            "data": None
        }
    
    logger.info(f"[ASR Subprocess] 使用 Python: {LLM_CONDA_PYTHON}")
    
    # 使用 subprocess 调用 Python 代码执行 ASR 识别
    # 由于 base64 数据可能很长，使用临时文件存储参数
    import tempfile
    
    logger.info(f"[ASR Subprocess] 创建临时文件存储参数...")
    # 创建临时文件存储参数
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
        params = {
            "audio_base64": audio_base64,
            "model_name": model_name,
            "language": language,
            "task": task
        }
        json.dump(params, tmp_file, ensure_ascii=False)
        tmp_params_file = tmp_file.name
        logger.info(f"[ASR Subprocess] 临时文件已创建: {tmp_params_file}")
    
    # 构建 Python 代码，添加完善的错误处理
    python_code = f"""
import sys
import json
import os
import traceback

try:
    sys.path.insert(0, r'{PROJECT_ROOT}')
    
    # 读取参数文件
    with open(r'{tmp_params_file}', 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    from llm_talk.asr import get_asr_response_api
    
    result = get_asr_response_api(
        audio_input=params['audio_base64'],
        model_name=params['model_name'],
        language=params['language'],
        task=params['task']
    )
    
    # 确保结果是字典格式
    if not isinstance(result, dict):
        result = {{"success": False, "error": {{"message": "ASR 函数返回了非字典类型", "type": "TypeError"}}, "data": None}}
    
    print(json.dumps(result, ensure_ascii=False))
    
except ImportError as e:
    error_result = {{
        "success": False,
        "error": {{
            "code": "IMPORT_ERROR",
            "message": f"导入错误: {{str(e)}}",
            "type": "ImportError",
            "traceback": traceback.format_exc()
        }},
        "data": None
    }}
    print(json.dumps(error_result, ensure_ascii=False))
except Exception as e:
    error_result = {{
        "success": False,
        "error": {{
            "code": "EXECUTION_ERROR",
            "message": f"执行错误: {{str(e)}}",
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }},
        "data": None
    }}
    print(json.dumps(error_result, ensure_ascii=False))
finally:
    # 清理临时文件
    try:
        os.unlink(r'{tmp_params_file}')
    except:
        pass
"""
    
    # 构建命令
    cmd = [
        str(LLM_CONDA_PYTHON),
        "-c",
        python_code
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    
    # 将 conda 环境的 bin 目录添加到 PATH 的最前面
    llm_env_bin = LLM_CONDA_ENV / "bin" if LLM_CONDA_ENV.exists() else None
    if llm_env_bin and llm_env_bin.exists():
        conda_bin_path = str(llm_env_bin)
        current_path = env.get("PATH", "")
        path_sep = os.pathsep
        env["PATH"] = f"{conda_bin_path}{path_sep}{current_path}" if current_path else conda_bin_path
        logger.info(f"[ASR Subprocess] Conda 环境 bin 目录已添加到 PATH: {conda_bin_path}")
    
    logger.info(f"[ASR Subprocess] 执行命令: {' '.join(cmd[:3])}... (Python代码已省略)")
    logger.info(f"[ASR Subprocess] 工作目录: {PROJECT_ROOT}")
    logger.info(f"[ASR Subprocess] 开始执行 subprocess，超时时间: 300秒")
    
    try:
        # 使用Popen来实时捕获输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=str(PROJECT_ROOT),
            env=env
        )
        
        logger.info(f"[ASR Subprocess] 进程已启动，PID: {process.pid}")
        stdout, stderr = process.communicate(timeout=300)  # 5分钟超时
        logger.info(f"[ASR Subprocess] 进程执行完成，返回码: {process.returncode}")
        
        # 确保清理临时文件
        try:
            os.unlink(tmp_params_file)
        except:
            pass
        
        if process.returncode != 0:
            logger.error(f"[ASR Subprocess] 执行失败，返回码: {process.returncode}")
            if stderr:
                logger.error(f"[ASR Subprocess] 错误输出: {stderr}")
            if stdout:
                logger.error(f"[ASR Subprocess] 标准输出: {stdout[:500]}")
            error_msg = f"ASR 执行失败（返回码: {process.returncode}）"
            if stderr:
                error_msg += f": {stderr[:500]}"
            return {
                "success": False,
                "error": {
                    "code": "SUBPROCESS_ERROR",
                    "message": error_msg,
                    "type": "SubprocessError"
                },
                "data": None
            }
        
        # 解析 JSON 输出
        try:
            if not stdout or not stdout.strip():
                logger.error(f"[ASR Subprocess] 没有收到任何输出")
                if stderr:
                    logger.error(f"[ASR Subprocess] stderr 输出: {stderr[:1000]}")
                return {
                    "success": False,
                    "error": {
                        "code": "NO_OUTPUT",
                        "message": "ASR subprocess 没有产生任何输出",
                        "type": "EmptyOutputError"
                    },
                    "data": None
                }
            result = json.loads(stdout)
            logger.info(f"[ASR Subprocess] 成功解析 JSON 输出，result.success = {result.get('success')}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"[ASR Subprocess] JSON 解析失败: {e}")
            logger.error(f"[ASR Subprocess] stdout 输出 (前1000字符): {stdout[:1000] if stdout else '(空)'}")
            logger.error(f"[ASR Subprocess] stderr 输出 (前1000字符): {stderr[:1000] if stderr else '(空)'}")
            return {
                "success": False,
                "error": {
                    "code": "JSON_DECODE_ERROR",
                    "message": f"无法解析 ASR 输出: {str(e)}。stdout: {stdout[:200] if stdout else '(空)'}",
                    "type": "JSONDecodeError",
                    "raw_stdout": stdout[:500] if stdout else None,
                    "raw_stderr": stderr[:500] if stderr else None
                },
                "data": None
            }
            
    except subprocess.TimeoutExpired:
        logger.error(f"[ASR Subprocess] 执行超时（>300秒）")
        process.kill()
        # 确保清理临时文件
        try:
            os.unlink(tmp_params_file)
        except:
            pass
        return {
            "success": False,
            "error": {
                "code": "TIMEOUT",
                "message": "ASR 识别超时（>300秒）",
                "type": "TimeoutError"
            },
            "data": None
        }
    except Exception as e:
        logger.error(f"[ASR Subprocess] 调用失败: {e}")
        import traceback
        logger.error(f"[ASR Subprocess] 错误详情: {traceback.format_exc()}")
        # 确保清理临时文件
        try:
            os.unlink(tmp_params_file)
        except:
            pass
        return {
            "success": False,
            "error": {
                "code": "EXCEPTION",
                "message": f"调用 ASR 时发生错误: {str(e)}",
                "type": type(e).__name__
            },
            "data": None
        }

