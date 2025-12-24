#!/usr/bin/env python3
"""
数据库模块测试程序
测试 settings_db, video_records_db, chat_db 三个模块的功能
"""
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

import logging
from datetime import datetime
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 导入数据库模块
from shared.database import settings_db
from shared.database import video_records_db
from shared.database import chat_db

def test_settings_db():
    """测试设置数据库"""
    print("\n" + "="*60)
    print("测试 settings_db 模块")
    print("="*60)
    
    try:
        # 初始化数据库
        print("\n1. 初始化数据库...")
        settings_db.init_database()
        print("   ✓ 数据库初始化成功")
        
        # 测试设置和获取
        print("\n2. 测试设置和获取...")
        test_key = "test_setting_key"
        test_value = "test_value_123"
        
        settings_db.set_setting(test_key, test_value)
        retrieved = settings_db.get_setting(test_key)
        assert retrieved == test_value, f"设置值不匹配: {retrieved} != {test_value}"
        print(f"   ✓ 设置和获取成功: {test_key} = {retrieved}")
        
        # 测试获取所有设置
        print("\n3. 测试获取所有设置...")
        all_settings = settings_db.get_all_settings()
        assert test_key in all_settings, "新设置的键不在所有设置中"
        assert all_settings[test_key] == test_value, "新设置的值不正确"
        print(f"   ✓ 获取所有设置成功，共 {len(all_settings)} 个设置")
        
        # 测试默认值
        print("\n4. 测试默认值...")
        default_theme = settings_db.get_setting("nerf_theme")
        assert default_theme is not None, "默认主题不存在"
        print(f"   ✓ 默认主题: {default_theme}")
        
        # 测试重置为默认值
        print("\n5. 测试重置为默认值...")
        settings_db.reset_to_defaults()
        all_settings_after_reset = settings_db.get_all_settings()
        print(f"   ✓ 重置成功，当前设置数: {len(all_settings_after_reset)}")
        
        print("\n✓ settings_db 测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ settings_db 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_records_db():
    """测试视频记录数据库"""
    print("\n" + "="*60)
    print("测试 video_records_db 模块")
    print("="*60)
    
    try:
        # 初始化数据库
        print("\n1. 初始化数据库...")
        video_records_db.init_database()
        print("   ✓ 数据库初始化成功")
        
        # 测试添加生成记录
        print("\n2. 测试添加生成记录...")
        unique_id = f"test_{uuid.uuid4().hex[:8]}"
        test_record = {
            "unique_id": unique_id,
            "text": "测试文本",
            "character": "ayanami",
            "record_type": "video",
            "model_name": "test_model.pkl",
            "video_path": f"database/videos/{unique_id}.mp4",
            "audio_path": f"database/audios/{unique_id}.wav",
            "text_path": f"database/texts/{unique_id}.txt",
            "generation_time": 12.5,
            "status": "completed"
        }
        
        result = video_records_db.add_generation_record(
            unique_id=test_record["unique_id"],
            text=test_record["text"],
            character=test_record["character"],
            record_type=test_record["record_type"],
            model_name=test_record["model_name"],
            video_path=test_record["video_path"],
            audio_path=test_record["audio_path"],
            text_path=test_record["text_path"],
            generation_time=test_record["generation_time"],
            status=test_record["status"]
        )
        assert result, "添加生成记录失败"
        print(f"   ✓ 添加生成记录成功: {unique_id}")
        
        # 测试获取生成记录
        print("\n3. 测试获取生成记录...")
        retrieved = video_records_db.get_generation_record(unique_id)
        assert retrieved is not None, "获取生成记录失败"
        assert retrieved["unique_id"] == unique_id, "记录ID不匹配"
        assert retrieved["text"] == test_record["text"], "记录文本不匹配"
        print(f"   ✓ 获取生成记录成功: {retrieved['unique_id']}")
        
        # 测试列出生成记录
        print("\n4. 测试列出生成记录...")
        records = video_records_db.list_generation_records(record_type="video", limit=10)
        assert len(records) > 0, "没有找到任何记录"
        assert any(r["unique_id"] == unique_id for r in records), "新添加的记录不在列表中"
        print(f"   ✓ 列出生成记录成功，共 {len(records)} 条")
        
        # 测试更新记录
        print("\n5. 测试更新记录...")
        video_records_db.add_generation_record(
            unique_id=unique_id,
            text="更新后的文本",
            character="ayanami",
            record_type="video",
            status="completed"
        )
        updated = video_records_db.get_generation_record(unique_id)
        assert updated["text"] == "更新后的文本", "更新失败"
        print(f"   ✓ 更新记录成功")
        
        # 测试删除记录
        print("\n6. 测试删除记录...")
        result = video_records_db.delete_generation_record(unique_id)
        assert result, "删除记录失败"
        deleted_check = video_records_db.get_generation_record(unique_id)
        assert deleted_check is None, "记录未被删除"
        print(f"   ✓ 删除记录成功")
        
        # 测试向后兼容函数
        print("\n7. 测试向后兼容函数...")
        test_id = f"test_video_{uuid.uuid4().hex[:8]}"
        video_records_db.add_video_record(
            unique_id=test_id,
            text="测试视频",
            character="ayanami",
            model_name="test.pkl",
            video_path=f"database/videos/{test_id}.mp4"
        )
        video_record = video_records_db.get_video_record(test_id)
        assert video_record is not None, "获取视频记录失败"
        video_records_db.delete_video_record(test_id)
        print(f"   ✓ 向后兼容函数测试成功")
        
        print("\n✓ video_records_db 测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ video_records_db 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chat_db():
    """测试聊天数据库"""
    print("\n" + "="*60)
    print("测试 chat_db 模块")
    print("="*60)
    
    try:
        # 初始化数据库
        print("\n1. 初始化数据库...")
        chat_db.init_database()
        print("   ✓ 数据库初始化成功")
        
        # 测试创建聊天会话
        print("\n2. 测试创建聊天会话...")
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        result = chat_db.create_chat_session(
            session_id=session_id,
            title="测试会话",
            character="ayanami"
        )
        assert result, "创建聊天会话失败"
        print(f"   ✓ 创建聊天会话成功: {session_id}")
        
        # 测试获取聊天会话
        print("\n3. 测试获取聊天会话...")
        session = chat_db.get_chat_session(session_id)
        assert session is not None, "获取聊天会话失败"
        assert session["session_id"] == session_id, "会话ID不匹配"
        assert session["title"] == "测试会话", "会话标题不匹配"
        print(f"   ✓ 获取聊天会话成功: {session['title']}")
        
        # 测试添加聊天消息
        print("\n4. 测试添加聊天消息...")
        message_id_1 = f"msg_{uuid.uuid4().hex[:8]}"
        message_id_2 = f"msg_{uuid.uuid4().hex[:8]}"
        
        # 添加用户消息
        result = chat_db.add_chat_message(
            session_id=session_id,
            message_id=message_id_1,
            message_type="user",
            content_type="text",
            text_content="你好"
        )
        assert result, "添加用户消息失败"
        
        # 添加助手消息
        result = chat_db.add_chat_message(
            session_id=session_id,
            message_id=message_id_2,
            message_type="assistant",
            content_type="text+audio",
            text_content="你好！很高兴见到你",
            text_path=f"database/texts/{message_id_2}.txt",
            audio_path=f"database/audios/{message_id_2}.wav"
        )
        assert result, "添加助手消息失败"
        print(f"   ✓ 添加聊天消息成功")
        
        # 测试获取聊天消息
        print("\n5. 测试获取聊天消息...")
        messages = chat_db.get_chat_messages(session_id)
        assert len(messages) == 2, f"消息数量不匹配: {len(messages)} != 2"
        assert messages[0]["message_type"] == "user", "第一条消息类型不正确"
        assert messages[1]["message_type"] == "assistant", "第二条消息类型不正确"
        print(f"   ✓ 获取聊天消息成功，共 {len(messages)} 条消息")
        
        # 测试列出聊天会话
        print("\n6. 测试列出聊天会话...")
        sessions = chat_db.list_chat_sessions(limit=10)
        assert len(sessions) > 0, "没有找到任何会话"
        assert any(s["session_id"] == session_id for s in sessions), "新创建的会话不在列表中"
        print(f"   ✓ 列出聊天会话成功，共 {len(sessions)} 个会话")
        
        # 测试删除聊天会话
        print("\n7. 测试删除聊天会话...")
        result = chat_db.delete_chat_session(session_id)
        assert result, "删除聊天会话失败"
        deleted_session = chat_db.get_chat_session(session_id)
        assert deleted_session is None, "会话未被删除"
        deleted_messages = chat_db.get_chat_messages(session_id)
        assert len(deleted_messages) == 0, "会话消息未被级联删除"
        print(f"   ✓ 删除聊天会话成功（级联删除消息）")
        
        print("\n✓ chat_db 测试全部通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ chat_db 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("数据库模块测试程序")
    print("="*60)
    
    results = []
    
    # 测试各个模块
    results.append(("settings_db", test_settings_db()))
    results.append(("video_records_db", test_video_records_db()))
    results.append(("chat_db", test_chat_db()))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s} : {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 个通过, {failed} 个失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())

