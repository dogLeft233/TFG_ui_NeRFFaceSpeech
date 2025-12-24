#!/usr/bin/env python3
"""
LLM Talk 桥接脚本 - 在LLM conda环境中运行，返回包含LLM回答的结果
用于从非LLM环境中调用llm_talk模块并获取LLM回答文本
"""
import sys
import json
import argparse
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="LLM Talk Bridge - 生成音频并返回LLM回答")
    parser.add_argument("--input_text", required=True, help="输入文本")
    parser.add_argument("--audio_prompt_path", required=True, help="音频提示文件路径")
    parser.add_argument("--output_path", required=True, help="输出音频文件路径")
    parser.add_argument("--result_json", required=True, help="结果JSON文件路径（包含LLM回答）")
    
    args = parser.parse_args()
    
    try:
        from llm_talk.talk import talk_with_audio
        from llm_talk.tts import save_wav_to_file
        
        # 调用talk_with_audio
        result = talk_with_audio(
            user_input=args.input_text,
            language_id='zh',
            combine_audio=True,
            release_tts_model=False,
            split_sentences=True,
            audio_prompt_path=args.audio_prompt_path
        )
        
        if not result['success']:
            # 失败情况
            error_info = {
                'success': False,
                'error': result['error'],
                'llm_answer': None
            }
            with open(args.result_json, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, ensure_ascii=False, indent=2)
            print(f"ERROR: {result['error'].get('message', 'LLM+TTS生成失败')}", file=sys.stderr)
            sys.exit(1)
        
        # 获取LLM回答
        llm_answer = result['data'].get('llm_answer', '')
        
        # 保存音频文件
        audio_saved = False
        combined_audio = result['data'].get('combined_audio')
        if combined_audio and combined_audio.get('success'):
            wav_data = combined_audio.get('combined_audio_data')
            if wav_data:
                import os
                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                audio_saved = save_wav_to_file(wav_data, args.output_path)
        
        if not audio_saved:
            # 尝试使用第一个成功的音频
            audio_results = result['data'].get('audio_results', [])
            successful_audio = [a for a in audio_results if a.get('success')]
            if successful_audio:
                wav_data = successful_audio[0].get('audio_data')
                if wav_data:
                    import os
                    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                    audio_saved = save_wav_to_file(wav_data, args.output_path)
        
        # 保存结果到JSON文件
        result_info = {
            'success': audio_saved,
            'llm_answer': llm_answer,
            'output_path': args.output_path if audio_saved else None
        }
        
        with open(args.result_json, 'w', encoding='utf-8') as f:
            json.dump(result_info, f, ensure_ascii=False, indent=2)
        
        if audio_saved:
            print(f"SUCCESS: 音频已保存到 {args.output_path}", file=sys.stderr)
            sys.exit(0)
        else:
            print("ERROR: 音频保存失败", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        error_info = {
            'success': False,
            'error': str(e),
            'llm_answer': None
        }
        try:
            with open(args.result_json, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, ensure_ascii=False, indent=2)
        except:
            pass
        print(f"ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

