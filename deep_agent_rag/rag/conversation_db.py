"""
對話歷史數據庫管理模組
使用 SQLite 持久化存儲對話歷史，支持會話管理和智能檢索
"""
import sqlite3
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import hashlib


class ConversationDB:
    """
    對話歷史數據庫管理器
    
    功能：
    - 創建和管理會話（每個會話對應一組文件）
    - 存儲和檢索對話消息
    - 根據查詢智能檢索相關歷史對話
    - 支持會話切換和恢復
    """
    
    def __init__(self, db_path: str = "./conversation_history.db"):
        """
        初始化對話歷史數據庫
        
        Args:
            db_path: 數據庫文件路徑，默認為 "./conversation_history.db"
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化數據庫表結構"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 創建會話表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                session_id TEXT PRIMARY KEY,
                file_paths TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 創建消息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id) ON DELETE CASCADE
            )
        """)
        
        # 創建索引以提升查詢性能
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_timestamp 
            ON conversation_messages(session_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_role 
            ON conversation_messages(session_id, role)
        """)
        
        conn.commit()
        conn.close()
        print(f"✓ 對話歷史數據庫已初始化: {self.db_path}")
    
    def _generate_session_id(self, file_paths: List[str]) -> str:
        """
        根據文件路徑生成會話 ID
        
        Args:
            file_paths: 文件路徑列表
            
        Returns:
            會話 ID（SHA256 hash）
        """
        # 排序文件路徑以確保一致性
        sorted_paths = sorted(file_paths)
        # 生成 hash
        paths_str = json.dumps(sorted_paths, sort_keys=True)
        session_id = hashlib.sha256(paths_str.encode()).hexdigest()[:16]
        return session_id
    
    def get_or_create_session(self, file_paths: List[str]) -> str:
        """
        獲取或創建會話
        
        Args:
            file_paths: 文件路徑列表
            
        Returns:
            會話 ID
        """
        session_id = self._generate_session_id(file_paths)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 檢查會話是否存在
        cursor.execute("SELECT session_id FROM conversation_sessions WHERE session_id = ?", (session_id,))
        if cursor.fetchone() is None:
            # 創建新會話
            cursor.execute("""
                INSERT INTO conversation_sessions (session_id, file_paths, created_at, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (session_id, json.dumps(file_paths)))
            conn.commit()
            print(f"✓ 創建新會話: {session_id}")
        else:
            # 更新會話時間戳
            cursor.execute("""
                UPDATE conversation_sessions 
                SET updated_at = CURRENT_TIMESTAMP 
                WHERE session_id = ?
            """, (session_id,))
            conn.commit()
        
        conn.close()
        return session_id
    
    def save_message(self, session_id: str, role: str, content: str):
        """
        保存消息到數據庫
        
        Args:
            session_id: 會話 ID
            role: 角色（'user' 或 'assistant'）
            content: 消息內容
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversation_messages (session_id, role, content, timestamp)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (session_id, role, content))
        
        # 更新會話時間戳
        cursor.execute("""
            UPDATE conversation_sessions 
            SET updated_at = CURRENT_TIMESTAMP 
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
    
    def get_recent_history(
        self, 
        session_id: str, 
        limit: int = 10
    ) -> List[Tuple[str, str]]:
        """
        獲取最近的對話歷史
        
        Args:
            session_id: 會話 ID
            limit: 返回的對話輪數（每輪包含用戶問題和 AI 回答）
            
        Returns:
            對話歷史列表，格式為 List[Tuple[str, str]]，每個元組為 (用戶問題, AI回答)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 獲取最近的 limit * 2 條消息（每輪對話包含用戶和 AI 各一條）
        cursor.execute("""
            SELECT role, content 
            FROM conversation_messages 
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit * 2))
        
        messages = cursor.fetchall()
        conn.close()
        
        # 反轉順序（從舊到新）
        messages.reverse()
        
        # 配對用戶問題和 AI 回答
        conversation_history = []
        current_user_msg = None
        
        for role, content in messages:
            if role == "user":
                current_user_msg = content
            elif role == "assistant" and current_user_msg is not None:
                conversation_history.append((current_user_msg, content))
                current_user_msg = None
        
        return conversation_history
    
    def search_relevant_history(
        self,
        session_id: str,
        query: str,
        limit: int = 5,
        use_semantic_search: bool = True
    ) -> List[Tuple[str, str]]:
        """
        根據查詢智能檢索相關的歷史對話
        
        Args:
            session_id: 會話 ID
            query: 當前查詢問題
            limit: 返回的相關對話輪數
            use_semantic_search: 是否使用語義搜索（需要 LLM，較慢但更準確）
            
        Returns:
            相關的對話歷史列表
        """
        if not use_semantic_search:
            # 簡單的關鍵詞匹配
            return self._keyword_search_history(session_id, query, limit)
        else:
            # 使用 LLM 進行語義搜索（更智能但需要額外 LLM 調用）
            return self._semantic_search_history(session_id, query, limit)
    
    def _keyword_search_history(
        self,
        session_id: str,
        query: str,
        limit: int = 5
    ) -> List[Tuple[str, str]]:
        """
        使用關鍵詞匹配檢索歷史對話
        
        Args:
            session_id: 會話 ID
            query: 查詢問題
            limit: 返回的對話輪數
            
        Returns:
            相關的對話歷史列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 提取查詢中的關鍵詞（簡單實現：使用空格分割）
        keywords = query.lower().split()
        
        # 構建 SQL 查詢：查找包含關鍵詞的消息
        # 使用 LIKE 進行模糊匹配
        conditions = []
        params = [session_id]
        
        for keyword in keywords:
            if len(keyword) > 2:  # 只匹配長度大於 2 的關鍵詞
                conditions.append("LOWER(content) LIKE ?")
                params.append(f"%{keyword}%")
        
        if not conditions:
            # 如果沒有有效的關鍵詞，返回最近的歷史
            conn.close()
            return self.get_recent_history(session_id, limit)
        
        # 構建完整的 SQL 查詢
        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT role, content, timestamp
            FROM conversation_messages
            WHERE session_id = ? AND ({where_clause})
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit * 2)  # 每輪對話包含兩條消息
        
        cursor.execute(sql, params)
        messages = cursor.fetchall()
        conn.close()
        
        # 配對用戶問題和 AI 回答
        conversation_history = []
        seen_timestamps = set()
        
        # 按時間戳分組
        message_dict = {}
        for role, content, timestamp in messages:
            if timestamp not in message_dict:
                message_dict[timestamp] = {}
            message_dict[timestamp][role] = content
        
        # 配對並添加到歷史
        for timestamp in sorted(message_dict.keys()):
            if "user" in message_dict[timestamp] and "assistant" in message_dict[timestamp]:
                conversation_history.append((
                    message_dict[timestamp]["user"],
                    message_dict[timestamp]["assistant"]
                ))
        
        return conversation_history[:limit]
    
    def _semantic_search_history(
        self,
        session_id: str,
        query: str,
        limit: int = 5
    ) -> List[Tuple[str, str]]:
        """
        使用語義搜索檢索歷史對話（需要 LLM）
        
        Args:
            session_id: 會話 ID
            query: 查詢問題
            limit: 返回的對話輪數
            
        Returns:
            相關的對話歷史列表
        """
        # 獲取所有歷史對話
        all_history = self.get_all_history(session_id)
        
        if not all_history or len(all_history) <= limit:
            # 如果歷史不多，直接返回
            return all_history
        
        try:
            # 使用 LLM 來選擇最相關的歷史
            from ..utils.llm_utils import get_llm
            from langchain_core.messages import HumanMessage
            
            llm = get_llm()
            
            # 構建 prompt
            history_text = "\n\n".join([
                f"問題: {user_q}\n回答: {ai_a[:200]}..." if len(ai_a) > 200 else f"問題: {user_q}\n回答: {ai_a}"
                for user_q, ai_a in all_history[-20:]  # 只考慮最近 20 輪
            ])
            
            prompt = f"""根據當前查詢，從以下對話歷史中選擇最相關的 {limit} 輪對話。

當前查詢：{query}

對話歷史（按時間順序，從舊到新）：
{history_text}

請只返回最相關的對話輪數的索引（從 0 開始，用逗號分隔），例如：0,3,5,7,9
如果沒有相關的，請返回 "無"。
"""
            
            messages = [HumanMessage(content=prompt)]
            response = llm.invoke(messages)
            result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            if result.lower() in ["無", "无", "none", "no", ""]:
                # 如果沒有相關的，返回最近的
                return all_history[-limit:]
            
            # 解析索引
            try:
                indices = [int(i.strip()) for i in result.split(",") if i.strip().isdigit()]
                # 只保留有效的索引
                indices = [i for i in indices if 0 <= i < len(all_history)]
                
                if indices:
                    # 返回選中的歷史
                    selected = [all_history[i] for i in indices]
                    return selected[:limit]
            except:
                pass
            
            # 如果解析失敗，返回最近的
            return all_history[-limit:]
            
        except Exception as e:
            print(f"⚠️ 語義搜索歷史失敗: {e}，使用關鍵詞搜索")
            return self._keyword_search_history(session_id, query, limit)
    
    def get_all_history(self, session_id: str) -> List[Tuple[str, str]]:
        """
        獲取會話的所有對話歷史
        
        Args:
            session_id: 會話 ID
            
        Returns:
            所有對話歷史列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT role, content 
            FROM conversation_messages 
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        
        messages = cursor.fetchall()
        conn.close()
        
        # 配對用戶問題和 AI 回答
        conversation_history = []
        current_user_msg = None
        
        for role, content in messages:
            if role == "user":
                current_user_msg = content
            elif role == "assistant" and current_user_msg is not None:
                conversation_history.append((current_user_msg, content))
                current_user_msg = None
        
        return conversation_history
    
    def clear_session(self, session_id: str):
        """
        清除會話的所有對話歷史
        
        Args:
            session_id: 會話 ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM conversation_messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM conversation_sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
        print(f"✓ 已清除會話: {session_id}")
    
    def list_sessions(self) -> List[Dict]:
        """
        列出所有會話
        
        Returns:
            會話列表，每個會話包含 session_id, file_paths, created_at, updated_at, message_count
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                s.session_id,
                s.file_paths,
                s.created_at,
                s.updated_at,
                COUNT(m.id) as message_count
            FROM conversation_sessions s
            LEFT JOIN conversation_messages m ON s.session_id = m.session_id
            GROUP BY s.session_id
            ORDER BY s.updated_at DESC
        """)
        
        sessions = []
        for row in cursor.fetchall():
            session_id, file_paths_json, created_at, updated_at, message_count = row
            try:
                file_paths = json.loads(file_paths_json) if file_paths_json else []
            except:
                file_paths = []
            
            sessions.append({
                "session_id": session_id,
                "file_paths": file_paths,
                "created_at": created_at,
                "updated_at": updated_at,
                "message_count": message_count
            })
        
        conn.close()
        return sessions
    
    def delete_old_sessions(self, days: int = 30):
        """
        刪除指定天數之前的會話
        
        Args:
            days: 保留最近多少天的會話
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM conversation_sessions
            WHERE updated_at < datetime('now', '-' || ? || ' days')
        """, (days,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"✓ 已刪除 {deleted_count} 個舊會話（超過 {days} 天）")
        return deleted_count

