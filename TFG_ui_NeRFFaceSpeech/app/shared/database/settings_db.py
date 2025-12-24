"""
设置数据库管理模块
使用SQLite存储所有页面的共享设置
从 fastapi_server/database/settings_db.py 迁移而来
"""
import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any

# 数据库文件路径 - 存储在项目根目录的 database 目录下
# 从 gradio_app/shared/database/ 到项目根目录需要向上3级
SHARED_DIR = Path(__file__).parent.parent.resolve()
GRADIO_APP_DIR = SHARED_DIR.parent.resolve()
PROJECT_ROOT = GRADIO_APP_DIR.parent.resolve()
DB_DIR = PROJECT_ROOT / "database"
DB_FILE = DB_DIR / "settings.db"

# 确保数据库目录存在
try:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[数据库] 数据库目录: {DB_DIR}")
    print(f"[数据库] 数据库文件: {DB_FILE}")
except Exception as e:
    print(f"[数据库] 警告: 无法创建数据库目录 {DB_DIR}: {e}")

# 默认设置
DEFAULT_SETTINGS = {
    "nerf_theme": "tech",
    "nerf_font": "Inter",
    "nerf_font_size": "medium",
    "nerf_custom_font_size": "14"
}


def init_database():
    """初始化数据库，创建表结构（只在数据库文件不存在时初始化）"""
    try:
        db_path = str(DB_FILE.resolve())
        
        # 检查数据库文件是否已存在
        db_exists = DB_FILE.exists()
        
        if db_exists:
            # 数据库文件已存在，只验证表结构，不进行初始化
            print(f"[数据库] 设置数据库文件已存在，跳过初始化: {db_path}")
            conn = sqlite3.connect(db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # 验证表结构是否存在（向后兼容）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            
            conn.commit()
            conn.close()
            return
        
        # 数据库文件不存在，执行初始化
        print(f"[数据库] 设置数据库文件不存在，开始初始化: {db_path}")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        # 创建表结构
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        # 插入所有默认设置
        inserted_count = 0
        for key, value in DEFAULT_SETTINGS.items():
            cursor.execute((key, value))
            inserted_count += 1
        
        conn.commit()
        conn.close()
        
        print(f"[数据库] 已插入 {inserted_count} 条默认设置")
        print(f"[数据库] 设置数据库初始化成功: {db_path}")
    except Exception as e:
        print(f"[数据库] 错误: 设置数据库初始化失败: {e}")
        print(f"[数据库] 数据库路径: {DB_FILE}")
        raise


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """获取设置值"""
    db_path = str(DB_FILE.resolve())
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return result[0]
    return default if default is not None else DEFAULT_SETTINGS.get(key)


def set_setting(key: str, value: str):
    """设置值"""
    db_path = str(DB_FILE.resolve())
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO settings (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = ?
    """, (key, value, value))
    
    conn.commit()
    conn.close()


def get_all_settings() -> Dict[str, str]:
    """获取所有设置（网页只负责读取，不负责初始化）"""
    db_path = str(DB_FILE.resolve())
    
    # 不在这里初始化数据库！数据库应该已经在 start.py 中初始化好了
    # 如果数据库不存在，直接抛出异常，让网页无法打开
    if not DB_FILE.exists():
        raise FileNotFoundError(f"数据库文件不存在: {db_path}。请先运行 start.py 初始化数据库。")
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    # 验证表是否存在
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='settings'
    """)
    if not cursor.fetchone():
        conn.close()
        raise ValueError(f"数据库表 'settings' 不存在。请先运行 start.py 初始化数据库。")
    
    cursor.execute("SELECT key, value FROM settings")
    results = cursor.fetchall()
    
    settings = {key: value for key, value in results}
    
    # 确保所有默认设置都存在（如果数据库中缺失，则插入默认值）
    need_update = False
    for key, default_value in DEFAULT_SETTINGS.items():
        if key not in settings:
            # 数据库中缺失此设置，插入默认值
            cursor.execute("""
                INSERT INTO settings (key, value)
                VALUES (?, ?)
            """, (key, default_value))
            settings[key] = default_value
            need_update = True
    
    if need_update:
        conn.commit()
        print(f"[数据库] 已将缺失的默认设置写入数据库")
    
    conn.close()
    
    return settings


def reset_to_defaults():
    """重置为默认设置"""
    db_path = str(DB_FILE.resolve())
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    for key, value in DEFAULT_SETTINGS.items():
        cursor.execute("""
            INSERT INTO settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = ?
        """, (key, value, value))
    
    conn.commit()
    conn.close()


# 初始化数据库（只在模块导入时执行一次）
try:
    init_database()
except Exception as e:
    print(f"[数据库] 设置数据库初始化失败: {e}")
    # 不抛出异常，允许其他模块继续加载

