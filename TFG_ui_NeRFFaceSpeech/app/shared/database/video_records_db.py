"""
视频生成记录数据库管理模块
使用SQLite存储视频生成记录
从 fastapi_server/database/video_records_db.py 迁移而来
"""
import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import traceback

# 数据库文件路径 - 使用和settings_db相同的目录
# 从 gradio_app/shared/database/ 到项目根目录需要向上3级
SHARED_DIR = Path(__file__).parent.parent.resolve()
GRADIO_APP_DIR = SHARED_DIR.parent.resolve()
PROJECT_ROOT = GRADIO_APP_DIR.parent.resolve()
DB_DIR = PROJECT_ROOT / "database"
DB_FILE = DB_DIR / "video_records.db"

# 视频文件存储目录（与config.py中的VIDEOS_STORAGE_DIR保持一致）
VIDEOS_DIR = DB_DIR / "videos"
# 音频文件存储目录
AUDIOS_DIR = DB_DIR / "audios"

# 确保数据库目录、视频目录和音频目录存在
logger = logging.getLogger()
try:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"[数据库] 数据库目录已准备: {DB_DIR}")
except Exception as e:
    logger.error(f"[数据库] 创建数据库目录失败 {DB_DIR}: {e}")
    raise

try:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"[数据库] 视频目录已准备: {VIDEOS_DIR}")
except Exception as e:
    logger.error(f"[数据库] 创建视频目录失败 {VIDEOS_DIR}: {e}")

try:
    AUDIOS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"[数据库] 音频目录已准备: {AUDIOS_DIR}")
except Exception as e:
    logger.error(f"[数据库] 创建音频目录失败 {AUDIOS_DIR}: {e}")


def init_database():
    """初始化数据库，创建表结构（只在数据库文件不存在时初始化）"""
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        
        # 检查数据库文件是否已存在
        if DB_FILE.exists():
            logger.info(f"[数据库] 数据库文件已存在，跳过初始化: {db_path}")
            # 即使文件存在，也验证表结构是否存在（向后兼容）
            conn = sqlite3.connect(db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # 创建视频生成记录表（仅存储视频生成记录，聊天记录存储在chat_messages表）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generation_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    unique_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    record_type TEXT NOT NULL DEFAULT 'video',  -- 固定为'video'，聊天记录不存储在此表
                    text TEXT NOT NULL,  -- 用户输入的文本（保留字段，兼容旧代码）
                    llm_response TEXT,  -- LLM生成的回答文本（保留字段，兼容旧代码，建议使用text_path）
                    character TEXT NOT NULL,
                    model_name TEXT,  -- 视频生成时使用，聊天时可能为空
                    video_path TEXT,  -- 视频文件路径
                    audio_path TEXT,  -- 音频文件路径
                    text_path TEXT,  -- 文本文件路径（LLM回答的文本文件）
                    generation_time REAL,  -- 生成耗时（秒）
                    config_json TEXT,  -- 其他配置信息（JSON格式）
                    status TEXT DEFAULT 'completed'  -- 状态：completed, failed, cancelled
                )
            """)
            
            # 创建索引以提高查询性能
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_unique_id ON generation_records(unique_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON generation_records(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_record_type ON generation_records(record_type)
            """)
            
            conn.commit()
            conn.close()
            return
        
        # 数据库文件不存在，执行初始化
        logger.info(f"[数据库] 数据库文件不存在，开始初始化: {db_path}")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        # 创建视频生成记录表（仅存储视频生成记录，聊天记录存储在chat_messages表）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generation_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unique_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                record_type TEXT NOT NULL DEFAULT 'video',  -- 固定为'video'，聊天记录不存储在此表
                text TEXT NOT NULL,  -- 用户输入的文本
                llm_response TEXT,  -- LLM生成的回答文本（保留字段，兼容旧代码）
                character TEXT NOT NULL,
                model_name TEXT,  -- 视频生成时使用，聊天时可能为空
                video_path TEXT,  -- 视频文件路径
                audio_path TEXT,  -- 音频文件路径
                text_path TEXT,  -- 文本文件路径（LLM回答的文本文件）
                generation_time REAL,  -- 生成耗时（秒）
                config_json TEXT,  -- 其他配置信息（JSON格式）
                status TEXT DEFAULT 'completed'  -- 状态：completed, failed, cancelled
            )
        """)
        logger.info("[数据库] generation_records 表已创建")
        
        # 创建索引以提高查询性能
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_unique_id ON generation_records(unique_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON generation_records(created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_record_type ON generation_records(record_type)
        """)
        logger.info("[数据库] 索引已创建")
        
        conn.commit()
        conn.close()
        logger.info("[数据库] 数据库初始化成功")
    except Exception as e:
        logger.error(f"[数据库] 数据库初始化失败: {e}")
        logger.error(f"[数据库] 错误详情: {traceback.format_exc()}")
        raise


def add_generation_record(
    unique_id: str,
    text: str,
    character: str,
    record_type: str = 'video',  # 固定为'video'，聊天记录使用chat_db.add_chat_message
    llm_response: Optional[str] = None,
    model_name: Optional[str] = None,
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    text_path: Optional[str] = None,  # 文本文件路径
    generation_time: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    status: str = 'completed'
) -> bool:
    """添加或更新生成记录（视频生成或聊天记录）"""
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        logger.info(f"[数据库] 开始保存生成记录: unique_id={unique_id}, type={record_type}, status={status}")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        created_at = datetime.now().isoformat()
        config_json = json.dumps(config, ensure_ascii=False) if config else None
        
        # 检查text_path字段是否存在（向后兼容）
        cursor.execute("PRAGMA table_info(generation_records)")
        columns = [col[1] for col in cursor.fetchall()]
        has_text_path = 'text_path' in columns
        
        # 检查记录是否已存在
        cursor.execute("SELECT created_at FROM generation_records WHERE unique_id = ?", (unique_id,))
        existing = cursor.fetchone()
        
        if existing:
            # 记录已存在，执行更新操作
            logger.info(f"[数据库] 记录已存在，执行更新: unique_id={unique_id}")
            update_fields = []
            update_values = []
            
            if text:
                update_fields.append("text = ?")
                update_values.append(text)
            if llm_response is not None:
                update_fields.append("llm_response = ?")
                update_values.append(llm_response)
            if character:
                update_fields.append("character = ?")
                update_values.append(character)
            if model_name is not None:
                update_fields.append("model_name = ?")
                update_values.append(model_name)
            if video_path is not None:
                update_fields.append("video_path = ?")
                update_values.append(video_path)
            if audio_path is not None:
                update_fields.append("audio_path = ?")
                update_values.append(audio_path)
            if text_path is not None and has_text_path:
                update_fields.append("text_path = ?")
                update_values.append(text_path)
            if generation_time is not None:
                update_fields.append("generation_time = ?")
                update_values.append(generation_time)
            if config_json is not None:
                update_fields.append("config_json = ?")
                update_values.append(config_json)
            if status:
                update_fields.append("status = ?")
                update_values.append(status)
            
            if update_fields:
                update_values.append(unique_id)
                sql = f"UPDATE generation_records SET {', '.join(update_fields)} WHERE unique_id = ?"
                cursor.execute(sql, update_values)
                logger.info(f"[数据库] 记录更新成功: unique_id={unique_id}")
        else:
            # 记录不存在，执行插入操作
            logger.info(f"[数据库] 记录不存在，执行插入: unique_id={unique_id}")
            if has_text_path:
                cursor.execute("""
                    INSERT INTO generation_records 
                    (unique_id, created_at, record_type, text, llm_response, character, model_name, video_path, audio_path, text_path, generation_time, config_json, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (unique_id, created_at, record_type, text, llm_response, character, model_name, video_path, audio_path, text_path, generation_time, config_json, status))
            else:
                # 向后兼容：如果字段不存在，先添加字段
                try:
                    cursor.execute("ALTER TABLE generation_records ADD COLUMN text_path TEXT")
                    logger.info("[数据库] 为 generation_records 表添加 text_path 字段")
                    cursor.execute("""
                        INSERT INTO generation_records 
                        (unique_id, created_at, record_type, text, llm_response, character, model_name, video_path, audio_path, text_path, generation_time, config_json, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (unique_id, created_at, record_type, text, llm_response, character, model_name, video_path, audio_path, text_path, generation_time, config_json, status))
                except sqlite3.OperationalError:
                    # 如果字段添加失败，使用旧的插入方式
                    cursor.execute("""
                        INSERT INTO generation_records 
                        (unique_id, created_at, record_type, text, llm_response, character, model_name, video_path, audio_path, generation_time, config_json, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (unique_id, created_at, record_type, text, llm_response, character, model_name, video_path, audio_path, generation_time, config_json, status))
            logger.info(f"[数据库] 记录插入成功: unique_id={unique_id}")
        
        conn.commit()
        conn.close()
        logger.info(f"[数据库] 生成记录保存成功: unique_id={unique_id}")
        return True
    except sqlite3.IntegrityError as e:
        logger.error(f"[数据库] 添加生成记录失败（唯一性约束）: unique_id={unique_id}, 错误={e}")
        logger.debug(f"[数据库] 错误详情: {traceback.format_exc()}")
        return False
    except Exception as e:
        logger.error(f"[数据库] 添加生成记录失败: unique_id={unique_id}, 错误={e}")
        logger.error(f"[数据库] 错误详情: {traceback.format_exc()}")
        return False


def add_video_record(
    unique_id: str,
    text: str,
    character: str,
    model_name: str,
    video_path: str,
    audio_path: Optional[str] = None,
    llm_response: Optional[str] = None,
    text_path: Optional[str] = None,
    generation_time: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    status: str = 'completed'
) -> bool:
    """添加视频生成记录（向后兼容函数）"""
    return add_generation_record(
        unique_id=unique_id,
        text=text,
        character=character,
        record_type='video',
        llm_response=llm_response,
        model_name=model_name,
        video_path=video_path,
        audio_path=audio_path,
        text_path=text_path,
        generation_time=generation_time,
        config=config,
        status=status
    )


def get_generation_record(unique_id: str) -> Optional[Dict[str, Any]]:
    """获取指定ID的生成记录"""
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        logger.debug(f"[数据库] 查询生成记录: unique_id={unique_id}")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM generation_records WHERE unique_id = ?", (unique_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            record = dict(row)
            if record.get('config_json'):
                try:
                    record['config'] = json.loads(record['config_json'])
                except json.JSONDecodeError as e:
                    logger.warning(f"[数据库] 解析config_json失败: unique_id={unique_id}, 错误={e}")
            logger.debug(f"[数据库] 成功查询到记录: unique_id={unique_id}")
            return record
        logger.debug(f"[数据库] 未找到记录: unique_id={unique_id}")
        return None
    except Exception as e:
        logger.error(f"[数据库] 获取生成记录失败: unique_id={unique_id}, 错误={e}")
        logger.error(f"[数据库] 错误详情: {traceback.format_exc()}")
        return None


def list_generation_records(
    record_type: Optional[str] = None,  # None表示所有类型，'video'或'chat'表示特定类型
    limit: int = 100, 
    offset: int = 0,
    unique_id: Optional[str] = None  # 可选：按unique_id查询
) -> List[Dict[str, Any]]:
    """列出生成记录（按创建时间倒序）"""
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        logger.debug(f"[数据库] 查询生成记录列表: type={record_type}, limit={limit}, offset={offset}")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if record_type:
            cursor.execute("""
                SELECT * FROM generation_records 
                WHERE record_type = ?
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """, (record_type, limit, offset))
        else:
            cursor.execute("""
                SELECT * FROM generation_records 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            record = dict(row)
            if record.get('config_json'):
                try:
                    record['config'] = json.loads(record['config_json'])
                except json.JSONDecodeError as e:
                    logger.warning(f"[数据库] 解析config_json失败: unique_id={record.get('unique_id')}, 错误={e}")
            records.append(record)
        
        logger.info(f"[数据库] 成功查询到 {len(records)} 条记录")
        return records
    except Exception as e:
        logger.error(f"[数据库] 列出生成记录失败: 错误={e}")
        logger.error(f"[数据库] 错误详情: {traceback.format_exc()}")
        return []


def get_video_record(unique_id: str) -> Optional[Dict[str, Any]]:
    """获取指定ID的视频记录（向后兼容函数）"""
    return get_generation_record(unique_id)


def list_video_records(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """列出视频记录（按创建时间倒序，向后兼容函数）"""
    return list_generation_records(record_type='video', limit=limit, offset=offset)


def delete_generation_record(unique_id: str) -> bool:
    """删除生成记录"""
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        logger.info(f"[数据库] 删除生成记录: unique_id={unique_id}")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM generation_records WHERE unique_id = ?", (unique_id,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            logger.info(f"[数据库] 成功删除 {deleted_count} 条记录: unique_id={unique_id}")
        else:
            logger.warning(f"[数据库] 未找到要删除的记录: unique_id={unique_id}")
        return deleted_count > 0
    except Exception as e:
        logger.error(f"[数据库] 删除生成记录失败: unique_id={unique_id}, 错误={e}")
        logger.error(f"[数据库] 错误详情: {traceback.format_exc()}")
        return False


def delete_video_record(unique_id: str) -> bool:
    """删除视频记录（向后兼容函数）"""
    return delete_generation_record(unique_id)


def create_or_update_task(
    task_id: str,
    status: str,
    text: str,
    character: str,
    model_name: Optional[str] = None,
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    text_path: Optional[str] = None,
    video_url: Optional[str] = None,  # 保留参数以兼容，但不保存到数据库（从path计算）
    audio_url: Optional[str] = None,  # 保留参数以兼容，但不保存到数据库（从path计算）
    text_url: Optional[str] = None,  # 保留参数以兼容，但不保存到数据库（从path计算）
    error_message: Optional[str] = None,  # 保留参数以兼容，但不保存到数据库
    start_time: Optional[str] = None,  # 保留参数以兼容，但不保存到数据库
    end_time: Optional[str] = None,  # 保留参数以兼容，但不保存到数据库
    generation_time: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    创建或更新视频生成任务（兼容video_tasks_db的接口）
    使用generation_records表存储，status可以是: pending, running, completed, failed, cancelled
    """
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        created_at = datetime.now().isoformat()
        config_json = json.dumps(config, ensure_ascii=False) if config else None
        
        # 检查记录是否已存在
        cursor.execute("SELECT created_at FROM generation_records WHERE unique_id = ?", (task_id,))
        existing = cursor.fetchone()
        
        if existing:
            # 更新现有记录
            existing_created_at = existing[0]
            update_fields = []
            update_values = []
            
            if status:
                update_fields.append("status = ?")
                update_values.append(status)
            if text:
                update_fields.append("text = ?")
                update_values.append(text)
            if character:
                update_fields.append("character = ?")
                update_values.append(character)
            if model_name is not None:
                update_fields.append("model_name = ?")
                update_values.append(model_name)
            if video_path is not None:
                update_fields.append("video_path = ?")
                update_values.append(video_path)
            if audio_path is not None:
                update_fields.append("audio_path = ?")
                update_values.append(audio_path)
            if text_path is not None:
                update_fields.append("text_path = ?")
                update_values.append(text_path)
            if generation_time is not None:
                update_fields.append("generation_time = ?")
                update_values.append(generation_time)
            if config_json is not None:
                update_fields.append("config_json = ?")
                update_values.append(config_json)
            
            if update_fields:
                update_values.append(task_id)
                sql = f"UPDATE generation_records SET {', '.join(update_fields)} WHERE unique_id = ?"
                cursor.execute(sql, update_values)
        
        else:
            # 创建新记录
            cursor.execute("""
                INSERT INTO generation_records 
                (unique_id, created_at, record_type, text, character, model_name, video_path, audio_path, text_path, generation_time, config_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (task_id, created_at, 'video', text, character, model_name, video_path, audio_path, text_path, generation_time, config_json, status))
        
        conn.commit()
        conn.close()
        logger.debug(f"[数据库] 任务已保存/更新: task_id={task_id}, status={status}")
        return True
    except Exception as e:
        logger.error(f"[数据库] 保存/更新任务失败: task_id={task_id}, 错误={e}")
        logger.error(f"[数据库] 错误详情: {traceback.format_exc()}")
        return False


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """
    获取指定任务（兼容video_tasks_db的接口）
    """
    record = get_generation_record(task_id)
    if record:
        # 转换为兼容video_tasks_db的格式
        record['task_id'] = record['unique_id']
        if record.get('config_json'):
            try:
                record['config'] = json.loads(record['config_json'])
            except:
                pass
        
        # 根据路径生成URL（如果路径存在）
        if record.get('video_path'):
            # 从路径中提取文件名（例如：database/videos/{task_id}.mp4）
            video_path = Path(record['video_path'])
            record['video_url'] = f"/videos/{video_path.name}"
        if record.get('audio_path'):
            audio_path = Path(record['audio_path'])
            record['audio_url'] = f"/audios/{audio_path.name}"
        if record.get('text_path'):
            text_path = Path(record['text_path'])
            record['text_url'] = f"/texts/{text_path.name}"
    return record


def get_running_task() -> Optional[Dict[str, Any]]:
    """
    获取正在运行的任务（status为pending或running）
    返回最新的一个任务（兼容video_tasks_db的接口）
    """
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM generation_records 
            WHERE record_type = 'video' AND status IN ('pending', 'running')
            ORDER BY created_at DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            record = dict(row)
            record['task_id'] = record['unique_id']
            if record.get('config_json'):
                try:
                    record['config'] = json.loads(record['config_json'])
                except:
                    pass
            
            # 根据路径生成URL（如果路径存在）
            if record.get('video_path'):
                video_path = Path(record['video_path'])
                record['video_url'] = f"/videos/{video_path.name}"
            if record.get('audio_path'):
                audio_path = Path(record['audio_path'])
                record['audio_url'] = f"/audios/{audio_path.name}"
            if record.get('text_path'):
                text_path = Path(record['text_path'])
                record['text_url'] = f"/texts/{text_path.name}"
            
            return record
        return None
    except Exception as e:
        logger.error(f"[数据库] 查询运行中任务失败: 错误={e}")
        logger.error(f"[数据库] 错误详情: {traceback.format_exc()}")
        return None


def list_tasks(status: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    列出任务（兼容video_tasks_db的接口）
    """
    records = list_generation_records(record_type='video', limit=limit, offset=offset)
    # 转换为兼容格式
    for record in records:
        record['task_id'] = record['unique_id']
        if record.get('config_json'):
            try:
                record['config'] = json.loads(record['config_json'])
            except:
                pass
    
    # 如果指定了status，进行过滤
    if status:
        records = [r for r in records if r.get('status') == status]
    
    return records


def delete_task(task_id: str) -> bool:
    """
    删除任务（兼容video_tasks_db的接口）
    """
    return delete_generation_record(task_id)


# 初始化数据库
init_database()

