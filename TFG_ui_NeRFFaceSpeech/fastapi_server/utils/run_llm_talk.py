import subprocess
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

# 导入配置
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROJECT_ROOT,
    LLM_CONDA_PYTHON,
    LLM_TALK_SCRIPT as TALK_SCRIPT,
    get_character_audio_prompt,
)

# 使用logging模块添加日志（避免循环导入）
import logging
def add_log(message, level="info"):
    """添加日志到日志系统"""
    logger = logging.getLogger()
    level_map = {
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "success": logging.INFO,  # success级别使用INFO，但在消息前添加标记以便前端识别
        "progress": logging.INFO,
        "debug": logging.DEBUG
    }
    log_level = level_map.get(level, logging.INFO)
    # 对于success级别，在消息前添加特殊标记，BufferLogHandler会识别并正确处理
    if level == "success":
        logger.log(log_level, f"[SUCCESS] {message}")
    elif level == "progress":
        logger.log(log_level, f"[PROGRESS] {message}")
    else:
        logger.log(log_level, message)

def generate_audio(text: str, output_path: str, character: str) -> Tuple[bool, Optional[str]]:
    """
    生成音频文件，并返回LLM回答文本
    使用LLM conda环境中的桥接脚本来调用llm_talk模块
    
    Returns:
        tuple[bool, str | None]: (成功标志, LLM回答文本)
    """
    try:
        audio_prompt = get_character_audio_prompt(character)
    except ValueError as e:
        error_msg = f"错误: {e}"
        print(error_msg)
        add_log(error_msg, "error")
        return False, None

    # 使用桥接脚本在LLM conda环境中运行
    try:
        import tempfile
        import json
        
        # 创建临时JSON文件用于存储结果
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            result_json_path = f.name
        
        bridge_script = Path(__file__).parent / "llm_talk_with_text_bridge.py"
        
        if not bridge_script.exists():
            error_msg = f"桥接脚本不存在: {bridge_script}"
            add_log(error_msg, "error")
            return False, None

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        
        cmd = [
            str(LLM_CONDA_PYTHON),
            str(bridge_script),
            "--input_text", text,
            "--audio_prompt_path", audio_prompt,
            "--output_path", output_path,
            "--result_json", result_json_path,
        ]
        
        add_log(f"[LLM+TTS] 开始生成音频，输入文本: {text[:100]}...", "info")
        add_log(f"[LLM+TTS] 使用LLM conda环境: {LLM_CONDA_PYTHON}", "info")
        
        # 运行桥接脚本
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',  # 使用replace模式处理非UTF-8字节，避免解码错误
            bufsize=1,
            env=env,
            cwd=str(PROJECT_ROOT),
            universal_newlines=True
        )
        
        # 实时读取输出并添加到日志
        for line in process.stdout:
            if line:
                line = line.rstrip()
                # 根据内容判断日志级别
                if 'ERROR' in line or '错误' in line or 'Error' in line or 'Exception' in line:
                    add_log(f"[LLM+TTS] {line}", "error")
                elif 'WARNING' in line or '警告' in line or 'Warning' in line or 'WARN' in line:
                    add_log(f"[LLM+TTS] {line}", "warning")
                elif 'SUCCESS' in line or '成功' in line:
                    add_log(f"[LLM+TTS] {line}", "success")
                elif line.strip():  # 忽略空行
                    add_log(f"[LLM+TTS] {line}", "info")
        
        process.wait()
        
        # 读取结果JSON文件
        llm_answer = None
        success = False
        
        try:
            if os.path.exists(result_json_path):
                with open(result_json_path, 'r', encoding='utf-8') as f:
                    result_info = json.load(f)
                    success = result_info.get('success', False)
                    llm_answer = result_info.get('llm_answer')
                    
                    if success and llm_answer:
                        add_log(f"[LLM+TTS] LLM回答生成成功，长度: {len(llm_answer)}", "success")
                        add_log(f"[LLM+TTS] LLM回答内容: {llm_answer[:200]}...", "info")
                    elif not success:
                        error_msg = result_info.get('error', '未知错误')
                        add_log(f"[LLM+TTS] 生成失败: {error_msg}", "error")
            else:
                add_log("[LLM+TTS] 结果JSON文件不存在", "error")
        finally:
            # 清理临时文件
            try:
                if os.path.exists(result_json_path):
                    os.remove(result_json_path)
            except:
                pass
        
        if process.returncode != 0:
            error_msg = f"LLM 语音生成失败，返回码: {process.returncode}"
            add_log(error_msg, "error")
            return False, llm_answer  # 即使失败，也返回可能获取到的LLM回答
        
        if success:
            return True, llm_answer
        else:
            return False, llm_answer
            
    except Exception as e:
        error_msg = f"LLM 语音生成错误: {e}"
        print(error_msg)
        add_log(error_msg, "error")
        import traceback
        add_log(f"[LLM+TTS] 错误详情: {traceback.format_exc()}", "error")
        return False, None
