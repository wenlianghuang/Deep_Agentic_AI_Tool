"""
LLM 工具函數
提供 LLM 實例的創建和管理
"""
from ..models import MLXChatModel, load_mlx_model
from ..config import MLX_MAX_TOKENS, MLX_TEMPERATURE


def get_llm():
    """
    獲取 LLM 實例
    使用本地 MLX 模型替代 Groq API
    """
    # 載入 MLX 模型
    model, tokenizer = load_mlx_model()
    
    # 創建 MLX ChatModel 包裝器
    return MLXChatModel(
        model=model,
        tokenizer=tokenizer,
        max_tokens=MLX_MAX_TOKENS,
        temperature=MLX_TEMPERATURE
    )

