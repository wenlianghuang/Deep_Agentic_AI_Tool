"""
LLM 適配器：將 LangChain ChatModel 包裝成 OllamaLLM 接口
用於兼容 Learn_RAG 項目中的進階 RAG 方法
"""
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
import logging

logger = logging.getLogger(__name__)


class LangChainLLMAdapter:
    """
    將 LangChain ChatModel 適配為 OllamaLLM 接口
        
    這個適配器允許 Learn_RAG 項目中的進階 RAG 方法（需要 OllamaLLM）
    使用 Deep_Agentic_AI_Tool 的統一 LLM 系統（Groq -> Ollama -> MLX）
    """    
    def __init__(self, langchain_llm: BaseChatModel):
        """
        初始化適配器
                
        Args:
            langchain_llm: LangChain ChatModel 實例（來自 get_llm()）
        """        
        self.llm = langchain_llm
        self.model_name = self._detect_model_name()
        self.base_url = "http://localhost:11434"  # 默認值，實際不使用
        self.timeout = 120  # 默認值，實際不使用
        
        logger.info(f"✅ LLM 適配器初始化完成 (模型類型: {self.model_name})")
    
    def _detect_model_name(self) -> str:
        """
        檢測 LLM 類型和模型名稱
                
        Returns:
            模型名稱字符串
        """        
        llm_type = type(self.llm).__name__
        
        # 檢測 Groq
        if "Groq" in llm_type or "ChatGroq" in llm_type:
            model_name = getattr(self.llm, 'model_name', 'groq-unknown')
            return f"groq:{model_name}"
        
        # 檢測 Ollama
        if "Ollama" in llm_type or "ChatOllama" in llm_type:
            model_name = getattr(self.llm, 'model', 'ollama-unknown')
            return f"ollama:{model_name}"
        
        # 檢測 MLX
        if "MLX" in llm_type or "MLXChatModel" in llm_type:
            return "mlx:qwen2.5"
        
        # 默認
        return f"langchain:{llm_type}"
    
    def _check_ollama_connection(self) -> bool:
        """
        檢查 Ollama 服務是否可用（兼容性方法，實際不使用）
                
        Returns:
            總是返回 True（因為我們使用的是統一的 LLM 系統）
        """        
        return True
    
    def _check_model_available(self) -> bool:
        """
        檢查模型是否可用（兼容性方法，實際不使用）
                
        Returns:
            總是返回 True（因為我們使用的是統一的 LLM 系統）
        """        
        return True
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        生成回答（兼容 OllamaLLM.generate 接口）
                
        Args:
            prompt: 輸入 prompt
            temperature: 溫度參數（0.0-1.0），控制隨機性
            max_tokens: 最大生成 token 數（None 表示使用模型預設）
            stream: 是否使用流式輸出（當前不支持，總是返回完整結果）
                
        Returns:
            生成的回答字符串
        """        
        try:
            # 將 prompt 轉換為 LangChain 消息格式
            messages = [HumanMessage(content=prompt)]
            
            # 準備調用參數
            invoke_kwargs = {}
            
            # 如果 LLM 支持 temperature 參數
            if hasattr(self.llm, 'temperature'):
                # 臨時設置 temperature（如果支持）
                original_temp = getattr(self.llm, 'temperature', None)
                try:
                    self.llm.temperature = temperature
                except:
                    pass  # 如果不支持設置，忽略
            
            # 如果 LLM 支持 max_tokens 參數
            if max_tokens and hasattr(self.llm, 'max_tokens'):
                original_max_tokens = getattr(self.llm, 'max_tokens', None)
                try:
                    self.llm.max_tokens = max_tokens
                except:
                    pass  # 如果不支持設置，忽略
            
            # 調用 LangChain LLM
            response = self.llm.invoke(messages, **invoke_kwargs)
            
            # 恢復原始參數（如果之前修改過）
            if hasattr(self.llm, 'temperature') and 'original_temp' in locals():
                try:
                    self.llm.temperature = original_temp
                except:
                    pass
            
            if hasattr(self.llm, 'max_tokens') and 'original_max_tokens' in locals():
                try:
                    self.llm.max_tokens = original_max_tokens
                except:
                    pass
            
            # 提取回答內容
            if hasattr(response, 'content'):
                answer = response.content
            elif isinstance(response, str):
                answer = response
            else:
                answer = str(response)
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"⚠️ LLM 生成回答時出錯: {e}")
            raise RuntimeError(f"LLM 生成失敗: {e}") from e

