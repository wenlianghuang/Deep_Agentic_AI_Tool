"""
MLX 模型包裝器
將 MLX 模型整合到 LangChain 生態系統中
"""
from typing import List, Optional, Any
import mlx.core as mx
from mlx_lm import load, generate as mlx_generate

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from ..config import MLX_MODEL_ID, MLX_MAX_TOKENS, MLX_TEMPERATURE


class MLXChatModel(BaseChatModel):
    """
    MLX 模型的 LangChain 包裝器
    將 MLX 模型整合到 LangChain 生態系統中
    """
    model: Any = None
    tokenizer: Any = None
    max_tokens: int = MLX_MAX_TOKENS
    temperature: float = MLX_TEMPERATURE
    
    def __init__(self, model, tokenizer, max_tokens=MLX_MAX_TOKENS, temperature=MLX_TEMPERATURE, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    @property
    def _llm_type(self) -> str:
        return "mlx"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成回答"""
        # 將 LangChain 消息轉換為模型格式
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        # 使用 tokenizer 格式化對話
        try:
            prompt = self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # 如果 apply_chat_template 失敗，使用手動格式
            prompt_parts = []
            for msg in formatted_messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                elif role == "user":
                    prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                elif role == "assistant":
                    prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            prompt_parts.append("<|im_start|>assistant\n")
            prompt = "\n".join(prompt_parts)
        
        # 使用 MLX 的 generate 函數一次性生成（更快）
        # 注意：MLX 的 generate 不支援 temperature 參數，但速度更快
        try:
            response_text = mlx_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                verbose=False
            )# 【修復】清理輸出中的特殊標記
            response_text = response_text.strip()
            # 移除 <|im_end|> 和 <|im_start|> 標記
            response_text = response_text.replace("<|im_end|>", "").replace("<|im_start|>", "")
            # 移除多餘的空白行
            response_text = "\n".join(line for line in response_text.split("\n") if line.strip())
        except Exception as e:
            # 如果 generate 失敗，回退到逐個 token 生成
            print(f"   ⚠️ MLX generate 失敗，使用逐個 token 生成: {e}")
            tokens = self.tokenizer.encode(prompt)
            tokens = mx.array(tokens)
            
            generated_tokens = []
            for _ in range(self.max_tokens):
                # 前向傳播
                logits = self.model(tokens[None, :])
                logits = logits[0, -1, :]
                
                # 使用貪婪解碼（最快）
                next_token = mx.argmax(logits)
                next_token = int(next_token.item())
                
                # 檢查結束符
                if next_token == self.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token)
                tokens = mx.concatenate([tokens, mx.array([next_token])])
            
            # 解碼回答
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # 【額外保險】再次清理輸出，確保沒有遺漏的特殊標記
        response_text = response_text.strip()
        response_text = response_text.replace("<|im_end|>", "").replace("<|im_start|>", "")
        response_text = response_text.strip()
        
        # 創建 ChatResult
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def bind_tools(self, tools: List[Any], **kwargs: Any):
        """
        綁定工具（簡化版本）
        注意：MLX 模型可能不直接支援工具調用，這裡返回自身
        如果需要工具調用，可能需要額外的後處理
        """
        # 將工具信息添加到系統提示中
        self._tools = tools
        return self


# 全域 MLX 模型變數（延遲載入）
_mlx_model = None
_mlx_tokenizer = None


def load_mlx_model():
    """載入 MLX 模型（只載入一次）"""
    global _mlx_model, _mlx_tokenizer
    
    if _mlx_model is None or _mlx_tokenizer is None:
        print(f"📦 正在載入 MLX 模型 {MLX_MODEL_ID}...")
        _mlx_model, _mlx_tokenizer = load(MLX_MODEL_ID)
        print("✅ MLX 模型載入完成！")
    
    return _mlx_model, _mlx_tokenizer

