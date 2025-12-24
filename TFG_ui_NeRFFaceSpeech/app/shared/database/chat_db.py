"""
聊天记录数据库管理模块
使用SQLite存储聊天会话和消息
从 fastapi_server/database/chat_db.py 迁移而来
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
DB_FILE = DB_DIR / "chat_records.db"

# 统一使用config.py中定义的存储目录（不再单独创建聊天目录）
# 音频和文本文件统一存储在 database/audios 和 database/texts 目录下


def init_database():
    """初始化数据库，创建表结构（只在数据库文件不存在时初始化）"""
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        
        # 检查数据库文件是否已存在
        if DB_FILE.exists():
            logger.info(f"[数据库] 聊天数据库文件已存在，跳过初始化: {db_path}")
            # 即使文件存在，也验证表结构是否存在（向后兼容）
            conn = sqlite3.connect(db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # 创建聊天会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT NOT NULL PRIMARY KEY,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    character TEXT,
                    config_json TEXT  -- 其他配置信息（JSON格式）
                )
            """)
            
            # 创建聊天消息表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT NOT NULL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,  -- 'user' 或 'assistant'
                    content_type TEXT NOT NULL,  -- 'text', 'text+audio', 'text+audio+video'
                    text_content TEXT,  -- 文本内容（如果是文本消息）
                    text_path TEXT,  -- 文本文件路径（如果文本保存在文件中）
                    audio_path TEXT,  -- 音频文件路径（如果音频保存在文件中）
                    audio_base64 TEXT,  -- 音频base64（可选，用于小音频文件）
                    video_path TEXT,  -- 视频文件路径（如果视频保存在文件中）
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                )
            """)
            
            # 添加video_path字段（如果不存在）
            try:
                cursor.execute("ALTER TABLE chat_messages ADD COLUMN video_path TEXT")
                logger.info("[数据库] 已添加 video_path 字段到 chat_messages 表")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
                    raise
            
            # 创建索引以提高查询性能
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id ON chat_messages(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON chat_messages(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_updated_at ON chat_sessions(updated_at)
            """)
            
            conn.commit()
            conn.close()
            return
        
        # 数据库文件不存在，执行初始化
        logger.info(f"[数据库] 聊天数据库文件不存在，开始初始化: {db_path}")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        # 创建聊天会话表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT NOT NULL PRIMARY KEY,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                character TEXT,
                config_json TEXT  -- 其他配置信息（JSON格式）
            )
        """)
        logger.info("[数据库] chat_sessions 表已创建")
        
        # 创建聊天消息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id TEXT NOT NULL PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_type TEXT NOT NULL,  -- 'user' 或 'assistant'
                content_type TEXT NOT NULL,  -- 'text', 'text+audio', 'text+audio+video'
                text_content TEXT,  -- 文本内容（如果是文本消息）
                text_path TEXT,  -- 文本文件路径（如果文本保存在文件中）
                audio_path TEXT,  -- 音频文件路径（如果音频保存在文件中）
                audio_base64 TEXT,  -- 音频base64（可选，用于小音频文件）
                video_path TEXT,  -- 视频文件路径（如果视频保存在文件中）
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
            )
        """)
        logger.info("[数据库] chat_messages 表已创建")
        
        # 创建索引以提高查询性能
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id ON chat_messages(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON chat_messages(created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_updated_at ON chat_sessions(updated_at)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"[数据库] 聊天数据库初始化完成: {db_path}")
        
    except Exception as e:
        logger.error(f"[数据库] 初始化聊天数据库失败: {e}")
        logger.error(f"[数据库] 错误详情: {traceback.format_exc()}")
        raise


def create_chat_session(session_id: str, title: Optional[str] = None, 
                       character: str = "ayanami", config: Optional[Dict] = None) -> bool:
    """创建或更新聊天会话"""
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        created_at = datetime.now().isoformat()
        updated_at = created_at
        config_json = json.dumps(config) if config else None
        
        # 使用 INSERT OR REPLACE 来创建或更新会话
        cursor.execute("""
            INSERT OR REPLACE INTO chat_sessions 
            (session_id, title, created_at, updated_at, character, config_json)
            VALUES (?, ?, COALESCE((SELECT created_at FROM chat_sessions WHERE session_id = ?), ?), ?, ?, ?)
        """, (session_id, title, session_id, created_at, updated_at, character, config_json))
        
        conn.commit()
        conn.close()
        logger.debug(f"[数据库] 聊天会话已创建/更新: session_id={session_id}")
        return True
        
    except Exception as e:
        logger.error(f"[数据库] 创建聊天会话失败: session_id={session_id}, 错误={e}")
        logger.error(f"[数据库] 错误详情: {traceback.format_exc()}")
        return False


def add_chat_message(session_id: str, message_id: str, message_type: str, 
                    content_type: str, text_content: Optional[str] = None,
                    text_path: Optional[str] = None, audio_path: Optional[str] = None,
                    audio_base64: Optional[str] = None, video_path: Optional[str] = None) -> bool:
    """
    添加聊天消息
    
    Args:
        session_id: 会话ID
        message_id: 消息ID
        message_type: 'user' 或 'assistant'
        content_type: 'text', 'text+audio', 'text+audio+video'
        text_content: 文本内容
        text_path: 文本文件路径
        audio_path: 音频文件路径
        audio_base64: 音频base64（可选）
        video_path: 视频文件路径（可选）
    """
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        created_at = datetime.now().isoformat()
        
        # 更新会话的updated_at时间
        cursor.execute("""
            UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?
        """, (created_at, session_id))
        
        # 插入消息
        cursor.execute("""
            INSERT OR REPLACE INTO chat_messages 
            (message_id, session_id, message_type, content_type, text_content, 
             text_path, audio_path, audio_base64, video_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (message_id, session_id, message_type, content_type, text_content,
              text_path, audio_path, audio_base64, video_path, created_at))
        
        conn.commit()
        conn.close()
        logger.debug(f"[数据库] 聊天消息已添加: message_id={message_id}, session_id={session_id}")
        return True
        
    except Exception as e:
        logger.error(f"[数据库] 添加聊天消息失败: message_id={message_id}, 错误={e}")
        logger.error(f"[数据库] 错误详情: {traceback.format_exc()}")
        return False


def get_chat_session(session_id: str) -> Optional[Dict[str, Any]]:
    """获取聊天会话信息"""
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            session = dict(row)
            if session.get('config_json'):
                try:
                    session['config'] = json.loads(session['config_json'])
                except json.JSONDecodeError:
                    pass
            return session
        return None
        
    except Exception as e:
        logger.error(f"[数据库] 获取聊天会话失败: session_id={session_id}, 错误={e}")
        return None


def list_chat_sessions(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """列出所有聊天会话（按更新时间倒序）"""
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM chat_sessions 
            ORDER BY updated_at DESC 
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        sessions = []
        for row in rows:
            session = dict(row)
            if session.get('config_json'):
                try:
                    session['config'] = json.loads(session['config_json'])
                except json.JSONDecodeError:
                    pass
            sessions.append(session)
        
        return sessions
        
    except Exception as e:
        logger.error(f"[数据库] 列出聊天会话失败: 错误={e}")
        return []


def get_chat_messages(session_id: str, limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
    """获取指定会话的所有消息（按创建时间正序）"""
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM chat_messages 
            WHERE session_id = ?
            ORDER BY created_at ASC
            LIMIT ? OFFSET ?
        """, (session_id, limit, offset))
        
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            messages.append(dict(row))
        
        return messages
        
    except Exception as e:
        logger.error(f"[数据库] 获取聊天消息失败: session_id={session_id}, 错误={e}")
        return []


def delete_chat_session(session_id: str) -> bool:
    """删除聊天会话及其所有消息（级联删除）"""
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        # 启用外键约束以支持级联删除
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # 先手动删除消息（确保级联删除工作）
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        deleted_messages = cursor.rowcount
        
        # 删除会话
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        deleted_sessions = cursor.rowcount
        
        conn.commit()
        conn.close()
        logger.info(f"[数据库] 聊天会话已删除: session_id={session_id}, 删除了 {deleted_messages} 条消息")
        return deleted_sessions > 0
        
    except Exception as e:
        logger.error(f"[数据库] 删除聊天会话失败: session_id={session_id}, 错误={e}")
        return False


def update_chat_message_video_path(message_id: str, video_path: str) -> bool:
    """更新聊天消息的视频路径"""
    logger = logging.getLogger()
    try:
        db_path = str(DB_FILE.resolve())
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE chat_messages SET video_path = ? WHERE message_id = ?
        """, (video_path, message_id))
        
        conn.commit()
        conn.close()
        logger.debug(f"[数据库] 聊天消息视频路径已更新: message_id={message_id}, video_path={video_path}")
        return cursor.rowcount > 0
        
    except Exception as e:
        logger.error(f"[数据库] 更新聊天消息视频路径失败: message_id={message_id}, 错误={e}")
        logger.error(f"[数据库] 错误详情: {traceback.format_exc()}")
        return False


# 初始化数据库（只在模块导入时执行一次）
try:
    init_database()
except Exception as e:
    logger = logging.getLogger()
    logger.error(f"[数据库] 聊天数据库初始化失败: {e}")

