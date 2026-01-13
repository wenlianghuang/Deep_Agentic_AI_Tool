"""
Hybrid Guardrail Manager
混合式內容過濾管理器，受 NeMo Guardrails 啟發

整合關鍵字密度檢查（快速層）和語義主題過濾（深度層）
支援輸入/輸出雙向過濾
"""

import os
import yaml
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import jieba

# 獲取配置文件路徑
GUARDRAILS_CONFIG_DIR = Path(__file__).parent / "config"
CONFIG_FILE = GUARDRAILS_CONFIG_DIR / "config.yml"
RAILS_FILE = GUARDRAILS_CONFIG_DIR / "rails.txt"


class SemanticTopic:
    """語義主題定義"""
    def __init__(self, name: str, display_name: str, examples: List[str], blocked_message: str):
        self.name = name
        self.display_name = display_name
        self.examples = examples
        self.blocked_message = blocked_message
        self.embeddings: Optional[np.ndarray] = None


class HybridGuardrailManager:
    """
    混合式 Guardrail 管理器
    
    功能：
    1. 快速關鍵字密度檢查（毫秒級）
    2. 語義主題匹配（使用 sentence-transformers）
    3. 輸入/輸出雙向過濾
    4. 可配置的啟用/停用選項
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化 Guardrail 管理器
        
        Args:
            config_path: 配置文件路徑（默認使用內建配置）
        """
        self.config_path = config_path or CONFIG_FILE
        self.config: Dict = {}
        self.topics: List[SemanticTopic] = []
        self.model: Optional[SentenceTransformer] = None
        self._initialized = False
        
        # 載入配置
        self._load_config()
        
        # 初始化 jieba
        self._init_jieba()
        
        # 懶加載 embedding 模型（只在需要時初始化）
        if self.config.get("enabled", {}).get("semantic_filter", False):
            self._init_semantic_model()
    
    def _load_config(self):
        """載入配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print(f"✅ 載入 Guardrails 配置: {self.config_path}")
        except Exception as e:
            print(f"⚠️  無法載入 Guardrails 配置，使用默認設定: {e}")
            self._load_default_config()
    
    def _load_default_config(self):
        """載入默認配置"""
        self.config = {
            "enabled": {
                "keyword_filter": True,
                "semantic_filter": False,  # 默認關閉語義過濾
                "input_rails": True,
                "output_rails": True
            },
            "keyword_filter": {
                "threshold": 0.05,
                "blocked_keywords": [
                    "伊斯蘭教", "阿拉", "回教徒", "默罕默德",
                    "Islam", "Allah", "Muslim", "Muhammad"
                ],
                "blocked_message": "抱歉，您的問題包含敏感內容，無法回答。請換個話題或重新表述您的問題。"
            },
            "semantic_filter": {
                "similarity_threshold": 0.75,
                "topics": []
            },
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "cache_embeddings": True
            }
        }
    
    def _init_jieba(self):
        """初始化 jieba 分詞"""
        keywords = self.config.get("keyword_filter", {}).get("blocked_keywords", [])
        for keyword in keywords:
            jieba.add_word(keyword, freq=10000, tag='sensitive')
    
    def _init_semantic_model(self):
        """初始化語義模型（懶加載）"""
        if self._initialized:
            return
        
        try:
            model_name = self.config.get("embeddings", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
            print(f"🔄 正在載入語義模型: {model_name}")
            self.model = SentenceTransformer(model_name)
            
            # 載入主題定義
            self._load_topics()
            
            # 預計算主題 embeddings
            self._precompute_topic_embeddings()
            
            self._initialized = True
            print(f"✅ 語義模型載入完成，共 {len(self.topics)} 個主題")
        except Exception as e:
            print(f"⚠️  無法載入語義模型: {e}")
            self.config["enabled"]["semantic_filter"] = False
    
    def _load_topics(self):
        """從配置載入主題定義"""
        self.topics = []
        
        # 從 YAML 配置載入
        topics_config = self.config.get("semantic_filter", {}).get("topics", [])
        for topic_data in topics_config:
            topic = SemanticTopic(
                name=topic_data.get("name", ""),
                display_name=topic_data.get("display_name", ""),
                examples=topic_data.get("examples", []),
                blocked_message=topic_data.get("blocked_message", "抱歉，無法回答此問題。")
            )
            self.topics.append(topic)
        
        print(f"📋 載入了 {len(self.topics)} 個語義主題")
    
    def _precompute_topic_embeddings(self):
        """預計算所有主題的 embeddings"""
        if not self.model:
            return
        
        for topic in self.topics:
            if topic.examples:
                topic.embeddings = self.model.encode(topic.examples, convert_to_numpy=True)
    
    def _check_keyword_density(self, text: str) -> Tuple[bool, float, str]:
        """
        檢查關鍵字密度
        
        Returns:
            (should_block, density, message)
        """
        if not text or not text.strip():
            return False, 0.0, ""
        
        # 使用 jieba 進行斷詞
        words = list(jieba.cut(text))
        total_words = len(words)
        
        if total_words == 0:
            return False, 0.0, ""
        
        # 建立小寫敏感詞集合
        blocked_keywords = self.config.get("keyword_filter", {}).get("blocked_keywords", [])
        blocked_keywords_lower = {k.lower() for k in blocked_keywords}
        
        # 計算敏感詞數量
        sensitive_word_count = sum(
            1 for word in words
            if word.strip().lower() in blocked_keywords_lower
        )
        
        # 計算密度
        density = sensitive_word_count / total_words
        threshold = self.config.get("keyword_filter", {}).get("threshold", 0.05)
        
        should_block = density >= threshold
        message = self.config.get("keyword_filter", {}).get("blocked_message", "") if should_block else ""
        
        return should_block, density, message
    
    def _check_semantic_topic(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        檢查語義主題匹配
        
        Returns:
            (should_block, topic_name, blocked_message)
        """
        if not self.model or not self.topics:
            return False, None, None
        
        # 計算輸入文本的 embedding
        text_embedding = self.model.encode([text], convert_to_numpy=True)[0]
        
        # 獲取相似度門檻
        threshold = self.config.get("semantic_filter", {}).get("similarity_threshold", 0.75)
        
        # 檢查每個主題
        for topic in self.topics:
            if topic.embeddings is None or len(topic.embeddings) == 0:
                continue
            
            # 計算與所有範例的相似度
            similarities = np.dot(topic.embeddings, text_embedding) / (
                np.linalg.norm(topic.embeddings, axis=1) * np.linalg.norm(text_embedding)
            )
            
            # 取最大相似度
            max_similarity = np.max(similarities)
            
            # 如果超過門檻，阻擋
            if max_similarity >= threshold:
                print(f"🚫 語義匹配: {topic.display_name} (相似度: {max_similarity:.2%})")
                return True, topic.name, topic.blocked_message
        
        return False, None, None
    
    def check_input(self, text: str) -> Tuple[bool, str]:
        """
        檢查用戶輸入
        
        Args:
            text: 用戶輸入文本
        
        Returns:
            (should_block, message): 是否阻擋, 阻擋訊息（如果阻擋）
        """
        if not self.config.get("enabled", {}).get("input_rails", True):
            return False, ""
        
        # 1. 快速關鍵字檢查
        if self.config.get("enabled", {}).get("keyword_filter", True):
            should_block, density, message = self._check_keyword_density(text)
            if should_block:
                print(f"🚫 關鍵字過濾: 密度 {density:.2%}")
                return True, message
        
        # 2. 語義主題檢查
        if self.config.get("enabled", {}).get("semantic_filter", False):
            should_block, topic, message = self._check_semantic_topic(text)
            if should_block:
                return True, message or "抱歉，無法回答此問題。"
        
        return False, ""
    
    def check_output(self, text: str) -> Tuple[bool, str]:
        """
        檢查 LLM 輸出
        
        Args:
            text: LLM 輸出文本
        
        Returns:
            (should_block, filtered_text): 是否阻擋, 過濾後的文本
        """
        if not self.config.get("enabled", {}).get("output_rails", True):
            return False, text
        
        # 1. 快速關鍵字檢查
        if self.config.get("enabled", {}).get("keyword_filter", True):
            should_block, density, message = self._check_keyword_density(text)
            if should_block:
                print(f"🚫 輸出過濾: 密度 {density:.2%}")
                return True, message
        
        # 2. 語義主題檢查
        if self.config.get("enabled", {}).get("semantic_filter", False):
            should_block, topic, message = self._check_semantic_topic(text)
            if should_block:
                return True, message or "抱歉，無法提供此回應。"
        
        return False, text
    
    def get_status(self) -> Dict:
        """獲取當前 Guardrails 狀態"""
        return {
            "enabled": self.config.get("enabled", {}),
            "keyword_filter": {
                "threshold": self.config.get("keyword_filter", {}).get("threshold", 0.05),
                "keywords_count": len(self.config.get("keyword_filter", {}).get("blocked_keywords", []))
            },
            "semantic_filter": {
                "initialized": self._initialized,
                "topics_count": len(self.topics),
                "threshold": self.config.get("semantic_filter", {}).get("similarity_threshold", 0.75)
            }
        }
    
    def get_topics_info(self) -> List[Dict]:
        """獲取主題資訊"""
        return [
            {
                "name": topic.name,
                "display_name": topic.display_name,
                "examples_count": len(topic.examples)
            }
            for topic in self.topics
        ]


# 全局單例
_guardrail_manager: Optional[HybridGuardrailManager] = None


def get_guardrail_manager() -> HybridGuardrailManager:
    """獲取全局 Guardrail 管理器單例"""
    global _guardrail_manager
    if _guardrail_manager is None:
        _guardrail_manager = HybridGuardrailManager()
    return _guardrail_manager
