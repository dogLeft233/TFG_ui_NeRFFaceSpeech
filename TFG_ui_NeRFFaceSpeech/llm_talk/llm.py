from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError
import logging
import time
from typing import Optional, Union

logger = logging.getLogger(__name__)

API_KEY = "sk-0f7140cd662b480f9c69f5b528e01d67"

class LLMError(Exception):
    """自定义LLM异常类"""
    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        self.message = message
        self.error_code = error_code
        self.original_error = original_error
        super().__init__(self.message)
        
#---------------------------------------------------------------------

def get_llm_completion(question: str, max_retries: int = 3, timeout: int = 30) -> ChatCompletion:
    """
    获取LLM回复，包含完整的错误处理机制
    
    Args:
        question: 用户问题
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
    
    Returns:
        ChatCompletion: LLM回复对象
        
    Raises:
        LLMError: 各种错误情况
    """
    # 输入验证
    if not question or not isinstance(question, str):
        raise LLMError("问题不能为空且必须是字符串", "INVALID_INPUT")
    
    if len(question.strip()) == 0:
        raise LLMError("问题内容不能为空", "EMPTY_QUESTION")
    
    if len(question) > 10000:  # 限制问题长度
        raise LLMError("问题长度超过限制（10000字符）", "QUESTION_TOO_LONG")
    
    logger.info(f"正在使用API Key向大模型发送请求，问题长度: {len(question)}")
    
    # API Key验证
    if not API_KEY or API_KEY == "your-api-key-here":
        raise LLMError("API Key未配置或无效", "INVALID_API_KEY")
    
    client = None
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # 创建客户端
            client = OpenAI(
                api_key=API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=timeout
            )
            
            # 发送请求
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Please answer each question in concise Chinese, strictly within 100 characters."},
                    {"role": "user", "content": question},
                ],
                # Qwen3模型通过enable_thinking参数控制思考过程
                # extra_body={"enable_thinking": False},
            )
            
            # 验证响应
            if not completion or not completion.choices:
                raise LLMError("LLM返回空响应", "EMPTY_RESPONSE")
            
            if not completion.choices[0].message.content:
                raise LLMError("LLM返回内容为空", "EMPTY_CONTENT")
            
            logger.info("LLM请求成功")
            return completion
            
        except RateLimitError as e:
            error_msg = f"API请求频率超限: {str(e)}"
            logger.warning(error_msg)
            if attempt < max_retries:
                wait_time = (2 ** attempt) + 1  # 指数退避
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            raise LLMError(error_msg, "RATE_LIMIT_EXCEEDED", e)
            
        except APITimeoutError as e:
            error_msg = f"API请求超时: {str(e)}"
            logger.warning(error_msg)
            if attempt < max_retries:
                logger.info(f"第 {attempt + 1} 次重试...")
                time.sleep(1)
                continue
            raise LLMError(error_msg, "TIMEOUT_ERROR", e)
            
        except APIConnectionError as e:
            error_msg = f"API连接错误: {str(e)}"
            logger.warning(error_msg)
            if attempt < max_retries:
                logger.info(f"第 {attempt + 1} 次重试...")
                time.sleep(2)
                continue
            raise LLMError(error_msg, "CONNECTION_ERROR", e)
            
        except APIError as e:
            error_msg = f"API错误: {str(e)}"
            logger.error(error_msg)
            
            # 根据错误类型分类处理
            if "401" in str(e) or "unauthorized" in str(e).lower():
                raise LLMError("API Key无效或已过期", "UNAUTHORIZED", e)
            elif "403" in str(e) or "forbidden" in str(e).lower():
                raise LLMError("API访问被禁止", "FORBIDDEN", e)
            elif "429" in str(e):
                raise LLMError("请求过于频繁，请稍后重试", "TOO_MANY_REQUESTS", e)
            elif "500" in str(e) or "internal" in str(e).lower():
                if attempt < max_retries:
                    logger.info(f"服务器内部错误，第 {attempt + 1} 次重试...")
                    time.sleep(3)
                    continue
                raise LLMError("服务器内部错误", "SERVER_ERROR", e)
            else:
                raise LLMError(error_msg, "API_ERROR", e)
                
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg, "UNKNOWN_ERROR", e)
    
    # 如果所有重试都失败了
    raise LLMError(f"经过 {max_retries} 次重试后仍然失败", "MAX_RETRIES_EXCEEDED", last_error)

def exceed_str_from_completion(completion: ChatCompletion) -> str:
    """
    从ChatCompletion对象中提取文本内容，包含错误处理
    
    Args:
        completion: ChatCompletion对象
    
    Returns:
        str: 提取的文本内容
        
    Raises:
        LLMError: 各种错误情况
    """
    try:
        # 验证输入
        if not completion:
            raise LLMError("Completion对象为空", "EMPTY_COMPLETION")
        
        if not hasattr(completion, 'choices') or not completion.choices:
            raise LLMError("Completion对象缺少choices属性或为空", "INVALID_COMPLETION")
        
        if not completion.choices[0]:
            raise LLMError("Completion.choices[0]为空", "EMPTY_CHOICE")
        
        if not hasattr(completion.choices[0], 'message'):
            raise LLMError("Choice对象缺少message属性", "INVALID_CHOICE")
        
        if not completion.choices[0].message:
            raise LLMError("Message对象为空", "EMPTY_MESSAGE")
        
        if not hasattr(completion.choices[0].message, 'content'):
            raise LLMError("Message对象缺少content属性", "INVALID_MESSAGE")
        
        content = completion.choices[0].message.content
        
        if content is None:
            raise LLMError("Message content为None", "NULL_CONTENT")
        
        if not isinstance(content, str):
            raise LLMError(f"Message content类型错误，期望str，实际{type(content)}", "INVALID_CONTENT_TYPE")
        
        if len(content.strip()) == 0:
            raise LLMError("Message content为空字符串", "EMPTY_CONTENT")
        
        logger.info(f"成功提取内容，长度: {len(content)}")
        return content
        
    except LLMError:
        # 重新抛出LLMError
        raise
    except Exception as e:
        # 捕获其他异常并转换为LLMError
        error_msg = f"提取内容时发生未知错误: {str(e)}"
        logger.error(error_msg)
        raise LLMError(error_msg, "EXTRACTION_ERROR", e)

def ask_llm(question: str) -> str:
    """
    简化的LLM问答接口，包含完整错误处理
    
    Args:
        question: 用户问题
    
    Returns:
        str: LLM回复内容
        
    Raises:
        LLMError: 各种错误情况
    """
    try:
        completion = get_llm_completion(question)
        return exceed_str_from_completion(completion)
    except LLMError:
        # 重新抛出LLMError
        raise
    except Exception as e:
        # 捕获其他异常并转换为LLMError
        error_msg = f"ask_llm执行时发生未知错误: {str(e)}"
        logger.error(error_msg)
        raise LLMError(error_msg, "ASK_LLM_ERROR", e)

#---------------------------------------------------------------------

# 前端API接口
def get_llm_response_api(question: str) -> dict:
    """
    为前端提供的API接口，返回标准化的响应格式
    
    Args:
        question: 用户问题
    
    Returns:
        dict: 包含success, data, error信息的响应
    """
    try:
        response = ask_llm(question)
        return {
            "success": True,
            "data": {
                "answer": response,
                "question": question
            },
            "error": None
        }
    except LLMError as e:
        return {
            "success": False,
            "data": None,
            "error": {
                "code": e.error_code,
                "message": e.message,
                "type": "LLMError"
            }
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": {
                "code": "UNKNOWN_ERROR",
                "message": f"未知错误: {str(e)}",
                "type": "Exception"
            }
        }
        
        
#---------------------------------------------------------------------

if __name__ == "__main__":
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    # 测试用例
    test_cases = [
        "请用中文介绍一下百炼智能的主要产品和服务。",
        "",  # 空字符串测试
        "A" * 10001,  # 超长字符串测试
        "正常问题测试"
    ]
    
    for i, question in enumerate(test_cases):
        print(f"\n=== 测试用例 {i+1} ===")
        try:
            result = get_llm_response_api(question)
            if result["success"]:
                print(f"成功: {result['data']['answer'][:100]}...")
            else:
                print(f"失败: {result['error']['message']}")
        except Exception as e:
            print(f"异常: {str(e)}")