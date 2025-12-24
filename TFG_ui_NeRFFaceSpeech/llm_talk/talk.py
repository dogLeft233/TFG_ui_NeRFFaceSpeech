import argparse
import logging
import re
import sys
from .llm import get_llm_response_api, LLMError
from .tts import get_tts_response_api, TTSError, manage_tts_model, save_wav_to_file
from typing import List, Dict, Any

# 配置日志
logger = logging.getLogger(__name__)

class TalkError(Exception):
    """自定义Talk异常类"""
    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        self.message = message
        self.error_code = error_code
        self.original_error = original_error
        super().__init__(self.message)

#---------------------------------------------------------------------

def split_text_to_sentences(text: str) -> List[str]:
    """
    将文本分割成句子
    
    Args:
        text: 输入文本
    
    Returns:
        List[str]: 句子列表
    """
    try:
        if not text or not isinstance(text, str):
            return []
        
        # 清理文本
        text = text.strip()
        if not text:
            return []
        
        # 使用正则表达式分割句子
        # 支持中文和英文的句号、问号、感叹号
        sentence_endings = r'[。！？.!?]+'
        sentences = re.split(sentence_endings, text)
        
        # 过滤空句子并添加标点符号
        result = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:
                # 如果不是最后一个句子，添加适当的标点符号
                if i < len(sentences) - 1:
                    # 根据原文本中的标点符号添加
                    original_text = text
                    sentence_end_pos = original_text.find(sentence) + len(sentence)
                    if sentence_end_pos < len(original_text):
                        next_char = original_text[sentence_end_pos]
                        if next_char in '。！？.!?':
                            sentence += next_char
                        else:
                            sentence += '。'  # 默认添加句号
                result.append(sentence)
        
        # 如果没有分割出句子，返回原文本
        if not result:
            result = [text]
        
        logger.info(f"文本分割完成，共 {len(result)} 个句子")
        return result
        
    except Exception as e:
        logger.warning(f"文本分割失败，返回原文本: {str(e)}")
        return [text]

def generate_audio_for_sentences(sentences: List[str], 
                               language_id: str = 'zh',
                               **tts_kwargs) -> List[Dict[str, Any]]:
    """
    为句子列表生成音频
    
    Args:
        sentences: 句子列表
        language_id: 语言ID
        **tts_kwargs: TTS参数
    
    Returns:
        List[Dict[str, Any]]: 音频结果列表
    """
    audio_results = []
    
    try:
        logger.info(f"开始为 {len(sentences)} 个句子生成音频")
        
        for i, sentence in enumerate(sentences):
            logger.info(f"正在处理第 {i+1}/{len(sentences)} 个句子: {sentence[:50]}...")
            
            # 调用TTS API
            tts_result = get_tts_response_api(sentence, language_id=language_id, **tts_kwargs)
            
            if tts_result['success']:
                audio_results.append({
                    'sentence': sentence,
                    'sentence_index': i,
                    'audio_data': tts_result['data']['wav_data'],
                    'base64_data': tts_result['data']['base64_data'],
                    'duration': tts_result['data']['duration'],
                    'sample_rate': tts_result['data']['sample_rate'],
                    'success': True,
                    'error': None
                })
                logger.info(f"第 {i+1} 个句子音频生成成功，时长: {tts_result['data']['duration']:.2f}秒")
            else:
                audio_results.append({
                    'sentence': sentence,
                    'sentence_index': i,
                    'audio_data': None,
                    'base64_data': None,
                    'duration': 0,
                    'sample_rate': 0,
                    'success': False,
                    'error': tts_result['error']
                })
                logger.error(f"第 {i+1} 个句子音频生成失败: {tts_result['error']['message']}")
        
        successful_count = sum(1 for result in audio_results if result['success'])
        logger.info(f"音频生成完成，成功: {successful_count}/{len(sentences)}")
        
        return audio_results
        
    except Exception as e:
        error_msg = f"批量音频生成失败: {str(e)}"
        logger.error(error_msg)
        raise TalkError(error_msg, "AUDIO_GENERATION_ERROR", e)

def combine_audio_data(audio_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    合并多个音频数据
    
    Args:
        audio_results: 音频结果列表
    
    Returns:
        Dict[str, Any]: 合并后的音频信息
    """
    try:
        import numpy as np
        import soundfile as sf
        import io
        import base64
        
        # 过滤成功的音频结果
        successful_results = [result for result in audio_results if result['success']]
        
        if not successful_results:
            return {
                'combined_audio_data': None,
                'combined_base64_data': None,
                'total_duration': 0,
                'sample_rate': 0,
                'success': False,
                'error': '没有成功的音频数据可合并'
            }
        
        # 获取采样率（假设所有音频采样率相同）
        sample_rate = successful_results[0]['sample_rate']
        
        # 合并音频数据
        combined_audio = []
        total_duration = 0
        
        for result in successful_results:
            # 从WAV数据中提取音频
            wav_buffer = io.BytesIO(result['audio_data'])
            audio_data, sr = sf.read(wav_buffer)
            wav_buffer.close()
            
            # 确保采样率一致
            if sr != sample_rate:
                logger.warning(f"采样率不一致: {sr} vs {sample_rate}")
            
            combined_audio.append(audio_data)
            total_duration += result['duration']
        
        # 拼接音频
        if combined_audio:
            combined_audio_array = np.concatenate(combined_audio)
            
            # 创建合并后的WAV文件
            combined_wav_buffer = io.BytesIO()
            sf.write(combined_wav_buffer, combined_audio_array, sample_rate, format='WAV')
            combined_wav_data = combined_wav_buffer.getvalue()
            combined_wav_buffer.close()
            
            # 转换为Base64
            combined_base64_data = base64.b64encode(combined_wav_data).decode('utf-8')
            
            logger.info(f"音频合并完成，总时长: {total_duration:.2f}秒")
            
            return {
                'combined_audio_data': combined_wav_data,
                'combined_base64_data': combined_base64_data,
                'total_duration': total_duration,
                'sample_rate': sample_rate,
                'success': True,
                'error': None
            }
        else:
            return {
                'combined_audio_data': None,
                'combined_base64_data': None,
                'total_duration': 0,
                'sample_rate': 0,
                'success': False,
                'error': '没有有效的音频数据'
            }
            
    except Exception as e:
        error_msg = f"音频合并失败: {str(e)}"
        logger.error(error_msg)
        return {
            'combined_audio_data': None,
            'combined_base64_data': None,
            'total_duration': 0,
            'sample_rate': 0,
            'success': False,
            'error': error_msg
        }

def talk_with_audio(user_input: str, 
                   language_id: str = 'zh',
                   combine_audio: bool = True,
                   release_tts_model: bool = False,
                   split_sentences: bool = True,
                   **tts_kwargs) -> Dict[str, Any]:
    """
    完整的对话功能：用户输入 -> LLM回答 -> TTS音频生成
    
    Args:
        user_input: 用户输入文本
        language_id: TTS语言ID
        combine_audio: 是否合并音频
        release_tts_model: 是否在完成后释放TTS模型
        split_sentences: 是否分句处理
        **tts_kwargs: TTS参数
    
    Returns:
        Dict[str, Any]: 完整的对话结果
    """
    try:
        logger.info(f"开始处理用户输入: {user_input[:100]}...")
        
        # 步骤1: 调用LLM获取回答
        logger.info("步骤1: 调用LLM生成回答...")
        llm_result = get_llm_response_api(user_input)
        
        if not llm_result['success']:
            return {
                'success': False,
                'error': {
                    'code': 'LLM_ERROR',
                    'message': f"LLM调用失败: {llm_result['error']['message']}",
                    'type': 'LLMError'
                },
                'data': None
            }
        
        llm_answer = llm_result['data']['answer']
        logger.info(f"LLM回答生成成功，长度: {len(llm_answer)}")
        logger.info(f"LLM回答生成成功，内容: {llm_answer[:100]}...")
        
        # 步骤2: 分句处理（可选）
        if split_sentences:
            logger.info("步骤2: 对回答进行分句...")
            sentences = split_text_to_sentences(llm_answer)
            logger.info(f"分句完成，共 {len(sentences)} 个句子")
        else:
            logger.info("步骤2: 跳过分句，直接使用完整回答...")
            sentences = [llm_answer]
            logger.info("使用完整回答作为单个句子")
        
        # 步骤3: 生成音频
        logger.info("步骤3: 生成TTS音频...")
        audio_results = generate_audio_for_sentences(sentences, language_id=language_id, **tts_kwargs)
        
        # 检查是否有成功的音频
        successful_audio = [result for result in audio_results if result['success']]
        if not successful_audio:
            return {
                'success': False,
                'error': {
                    'code': 'TTS_ERROR',
                    'message': '所有句子的TTS生成都失败了',
                    'type': 'TTSError'
                },
                'data': {
                    'llm_answer': llm_answer,
                    'sentences': sentences,
                    'audio_results': audio_results
                }
            }
        
        # 步骤4:
        combined_audio_info = None
        if combine_audio:
            logger.info("步骤4: 处理合并音频...")
            combined_audio_info = combine_audio_data(audio_results)
        
        # 步骤5: 释放TTS模型（可选）
        tts_model_released = False
        if release_tts_model:
            logger.info("步骤5: 释放TTS模型...")
            try:
                unload_result = manage_tts_model('unload')
                tts_model_released = unload_result['success']
                if tts_model_released:
                    logger.info("TTS模型已成功释放")
                else:
                    logger.warning(f"TTS模型释放失败: {unload_result.get('message', '未知错误')}")
            except Exception as e:
                logger.error(f"TTS模型释放过程中发生异常: {str(e)}")
                tts_model_released = False
        
        # 构建返回结果
        result = {
            'success': True,
            'error': None,
            'data': {
                'user_input': user_input,
                'llm_answer': llm_answer,
                'sentences': sentences,
                'audio_results': audio_results,
                'successful_audio_count': len(successful_audio),
                'total_audio_count': len(audio_results),
                'combined_audio': combined_audio_info,
                'processing_info': {
                    'total_sentences': len(sentences),
                    'successful_audio': len(successful_audio),
                    'failed_audio': len(audio_results) - len(successful_audio),
                    'combine_audio_enabled': combine_audio,
                    'tts_model_released': tts_model_released,
                    'release_tts_model_enabled': release_tts_model,
                    'split_sentences_enabled': split_sentences
                }
            }
        }
        
        logger.info("对话处理完成")
        return result
        
    except TalkError:
        # 重新抛出TalkError
        raise
    except Exception as e:
        error_msg = f"对话处理过程中发生未知错误: {str(e)}"
        logger.error(error_msg)
        raise TalkError(error_msg, "TALK_PROCESSING_ERROR", e)

def get_talk_response_api(user_input: str, 
                         language_id: str = 'zh',
                         combine_audio: bool = True,
                         release_tts_model: bool = False,
                         split_sentences: bool = True,
                         **kwargs) -> Dict[str, Any]:
    """
    为前端提供的Talk API接口
    
    Args:
        user_input: 用户输入
        language_id: TTS语言ID
        combine_audio: 是否合并音频
        release_tts_model: 是否在完成后释放TTS模型
        split_sentences: 是否分句处理
        **kwargs: 其他参数
    
    Returns:
        Dict[str, Any]: 标准化的API响应
    """
    try:
        result = talk_with_audio(user_input, language_id=language_id, combine_audio=combine_audio, release_tts_model=release_tts_model, split_sentences=split_sentences, **kwargs)
        return result
    except TalkError as e:
        return {
            'success': False,
            'error': {
                'code': e.error_code,
                'message': e.message,
                'type': 'TalkError'
            },
            'data': None
        }
    except Exception as e:
        return {
            'success': False,
            'error': {
                'code': 'UNKNOWN_ERROR',
                'message': f'未知错误: {str(e)}',
                'type': 'Exception'
            },
            'data': None
        }

#---------------------------------------------------------------------

def _cli_generate_audio(args: argparse.Namespace) -> int:
    """
    命令行入口：先走 LLM 生成回答，再进行 TTS 并保存音频
    """
    logging.basicConfig(level=logging.INFO)

    result = talk_with_audio(
        args.input_text,
        language_id=args.language_id,
        combine_audio=True,
        release_tts_model=False,
        split_sentences=True,
        audio_prompt_path=args.audio_prompt_path,
    )

    if not result["success"]:
        err = result["error"]
        sys.stderr.write(f"生成失败: {err.get('message', 'unknown error')}\n")
        return 1

    data = result["data"]
    # 优先保存合并后的音频，否则保存第一段成功的音频
    combined = data.get("combined_audio")
    if combined and combined.get("success"):
        wav_data = combined["combined_audio_data"]
    else:
        successes = [a for a in data["audio_results"] if a["success"]]
        if not successes:
            sys.stderr.write("没有可用音频数据可保存\n")
            return 1
        wav_data = successes[0]["audio_data"]

    ok = save_wav_to_file(wav_data, args.output_path)
    if not ok:
        sys.stderr.write("保存音频文件失败\n")
        return 1

    print(f"音频已生成: {args.output_path}")
    return 0


def _cli():
    parser = argparse.ArgumentParser(description="LLM Talk CLI")
    parser.add_argument("--input_text", required=True, help="要合成的文本")
    parser.add_argument("--audio_prompt_path", required=False, help="音频提示文件路径")
    parser.add_argument("--output_path", required=True, help="输出 WAV 文件路径")
    parser.add_argument("--language_id", default="zh", help="语言ID，默认 zh")
    parser.add_argument(
        "--run_tests", action="store_true",
        help="运行内置测试用例（调试用）"
    )
    args = parser.parse_args()

    if args.run_tests:
        # 原有测试逻辑，可手动开启
        _run_tests()
        return

    exit_code = _cli_generate_audio(args)
    sys.exit(exit_code)


def _run_tests():
    # 保留原有测试逻辑，便于手动验证
    logging.basicConfig(level=logging.INFO)
    test_cases = [
        {"input": "你好，请介绍一下人工智能的发展历史。", "split_sentences": True},
        {"input": "今天天气怎么样？", "split_sentences": False},
        {"input": "请用一句话总结机器学习的重要性。", "split_sentences": True}
    ]

    print("=== Talk功能测试开始 ===")

    try:
        for i, test_case in enumerate(test_cases):
            print(f"\n--- 测试用例 {i+1} ---")
            print(f"用户输入: {test_case['input']}")
            print(f"分句处理: {'是' if test_case['split_sentences'] else '否'}")

            # 调用Talk API
            result = get_talk_response_api(
                test_case['input'],
                combine_audio=True,
                release_tts_model=True,
                split_sentences=test_case['split_sentences']
            )

            if result['success']:
                data = result['data']
                print(f"✅ 处理成功")
                print(f"📝 LLM回答: {data['llm_answer']}")
                print(f"📊 句子数量: {data['processing_info']['total_sentences']}")
                print(f"🎵 成功音频: {data['processing_info']['successful_audio']}")
                print(f"❌ 失败音频: {data['processing_info']['failed_audio']}")
                print(f"✂️ 分句处理: {'是' if data['processing_info']['split_sentences_enabled'] else '否'}")
                print(f"🧠 TTS模型释放: {'是' if data['processing_info']['tts_model_released'] else '否'}")

                if data['combined_audio'] and data['combined_audio']['success']:
                    print(f"🔗 合并音频时长: {data['combined_audio']['total_duration']:.2f}秒")

                # 保存合并音频（如果存在）
                if data['combined_audio'] and data['combined_audio']['success']:
                    filename = f'talk_output_{i+1}.wav'
                    if save_wav_to_file(data['combined_audio']['combined_audio_data'], filename):
                        print(f"💾 合并音频已保存到: {filename}")

            else:
                print(f"❌ 处理失败: {result['error']['message']}")

    except Exception as e:
        print(f"💥 测试过程中发生异常: {str(e)}")

    print("\n=== Talk功能测试结束 ===")


if __name__ == "__main__":
    _cli()
