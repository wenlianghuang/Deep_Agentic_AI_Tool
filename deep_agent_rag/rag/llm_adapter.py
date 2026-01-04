"""
LLM 适配器：将 LangChain ChatModel 包装成 OllamaLLM 接口
用于兼容 Learn_RAG 项目中的进阶 RAG 方法
"""
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
import logging

logger = logging.getLogger(__name__)


class LangChainLLMAdapter:
    """
    将 LangChain ChatModel 适配为 OllamaLLM 接口
    
    这个适配器允许 Learn_RAG 项目中的进阶 RAG 方法（需要 OllamaLLM）
    使用 Deep_Agentic_AI_Tool 的统一 LLM 系统（Groq -> Ollama -> MLX）
    """
    
    def __init__(self, langchain_llm: BaseChatModel):
        """
        初始化适配器
        
        Args:
            langchain_llm: LangChain ChatModel 实例（来自 get_llm()）
        """
        self.llm = langchain_llm
        self.model_name = self._detect_model_name()
        self.base_url = "http://localhost:11434"  # 默认值，实际不使用
        self.timeout = 120  # 默认值，实际不使用
        
        logger.info(f"✅ LLM 适配器初始化完成 (模型类型: {self.model_name})")
    
    def _detect_model_name(self) -> str:
        """
        检测 LLM 类型和模型名称
        
        Returns:
            模型名称字符串
        """
        llm_type = type(self.llm).__name__
        
        # 检测 Groq
        if "Groq" in llm_type or "ChatGroq" in llm_type:
            model_name = getattr(self.llm, 'model_name', 'groq-unknown')
            return f"groq:{model_name}"
        
        # 检测 Ollama
        if "Ollama" in llm_type or "ChatOllama" in llm_type:
            model_name = getattr(self.llm, 'model', 'ollama-unknown')
            return f"ollama:{model_name}"
        
        # 检测 MLX
        if "MLX" in llm_type or "MLXChatModel" in llm_type:
            return "mlx:qwen2.5"
        
        # 默认
        return f"langchain:{llm_type}"
    
    def _check_ollama_connection(self) -> bool:
        """
        检查 Ollama 服务是否可用（兼容性方法，实际不使用）
        
        Returns:
            总是返回 True（因为我们使用的是统一的 LLM 系统）
        """
        return True
    
    def _check_model_available(self) -> bool:
        """
        检查模型是否可用（兼容性方法，实际不使用）
        
        Returns:
            总是返回 True（因为我们使用的是统一的 LLM 系统）
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
            prompt: 输入 prompt
            temperature: 温度参数（0.0-1.0），控制随机性
            max_tokens: 最大生成 token 数（None 表示使用模型预设）
            stream: 是否使用流式输出（当前不支持，总是返回完整结果）
        
        Returns:
            生成的回答字符串
        """
        try:
            # 将 prompt 转换为 LangChain 消息格式
            messages = [HumanMessage(content=prompt)]
            
            # 准备调用参数
            invoke_kwargs = {}
            
            # 如果 LLM 支持 temperature 参数
            if hasattr(self.llm, 'temperature'):
                # 临时设置 temperature（如果支持）
                original_temp = getattr(self.llm, 'temperature', None)
                try:
                    self.llm.temperature = temperature
                except:
                    pass  # 如果不支持设置，忽略
            
            # 如果 LLM 支持 max_tokens 参数
            if max_tokens and hasattr(self.llm, 'max_tokens'):
                original_max_tokens = getattr(self.llm, 'max_tokens', None)
                try:
                    self.llm.max_tokens = max_tokens
                except:
                    pass  # 如果不支持设置，忽略
            
            # 调用 LangChain LLM
            response = self.llm.invoke(messages, **invoke_kwargs)
            
            # 恢复原始参数（如果之前修改过）
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
            
            # 提取回答内容
            if hasattr(response, 'content'):
                answer = response.content
            elif isinstance(response, str):
                answer = response
            else:
                answer = str(response)
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"⚠️ LLM 生成回答时出错: {e}")
            raise RuntimeError(f"LLM 生成失败: {e}") from e

