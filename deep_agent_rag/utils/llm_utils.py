"""
LLM 工具函數
提供 LLM 實例的創建和管理
優先使用 Groq API，額度用完後自動切換到本地 MLX 模型
"""
import warnings
from typing import Optional
from langchain_groq import ChatGroq
from ..models import MLXChatModel, load_mlx_model
from ..config import (
    MLX_MAX_TOKENS, 
    MLX_TEMPERATURE,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE,
    USE_GROQ_FIRST
)

# 全局變量：跟踪當前使用的 LLM 類型
_current_llm_type = None
_groq_quota_exceeded = False


def get_llm_type() -> str:
    """獲取當前使用的 LLM 類型"""
    return _current_llm_type or "unknown"


def is_using_local_llm() -> bool:
    """檢查是否正在使用本地 LLM"""
    return _current_llm_type == "mlx" or _groq_quota_exceeded


def get_llm():
    """
    獲取 LLM 實例
    優先使用 Groq API，額度用完後自動切換到本地 MLX 模型
    """
    global _current_llm_type, _groq_quota_exceeded
    
    # 如果已經知道 Groq 額度用完，直接使用本地模型
    if _groq_quota_exceeded:
        if _current_llm_type != "mlx":
            print("⚠️ 警告：Groq API 額度已用完，已切換到本地 MLX 模型 (Qwen2.5)")
        _current_llm_type = "mlx"
        model, tokenizer = load_mlx_model()
        return MLXChatModel(
            model=model,
            tokenizer=tokenizer,
            max_tokens=MLX_MAX_TOKENS,
            temperature=MLX_TEMPERATURE
        )
    
    # 嘗試使用 Groq API
    if USE_GROQ_FIRST and GROQ_API_KEY:
        try:
            groq_llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=GROQ_MODEL,
                max_tokens=GROQ_MAX_TOKENS,
                temperature=GROQ_TEMPERATURE
            )
            # 測試連接（通過一個簡單的調用來驗證）
            # 注意：這裡不實際調用，只是創建實例
            _current_llm_type = "groq"
            print("✅ 使用 Groq API (優先)")
            return groq_llm
        except Exception as e:
            # 如果創建失敗，可能是 API key 無效
            print(f"⚠️ Groq API 初始化失敗: {e}")
            _groq_quota_exceeded = True
            _current_llm_type = "mlx"
            print("⚠️ 警告：已切換到本地 MLX 模型 (Qwen2.5)")
            model, tokenizer = load_mlx_model()
            return MLXChatModel(
                model=model,
                tokenizer=tokenizer,
                max_tokens=MLX_MAX_TOKENS,
                temperature=MLX_TEMPERATURE
            )
    else:
        # 如果沒有配置 Groq 或選擇不使用，直接使用本地模型
        if not GROQ_API_KEY:
            print("ℹ️ 未配置 GROQ_API_KEY，使用本地 MLX 模型")
        _current_llm_type = "mlx"
        model, tokenizer = load_mlx_model()
        return MLXChatModel(
            model=model,
            tokenizer=tokenizer,
            max_tokens=MLX_MAX_TOKENS,
            temperature=MLX_TEMPERATURE
        )


def handle_groq_error(error: Exception) -> Optional[MLXChatModel]:
    """
    處理 Groq API 錯誤
    如果是額度用完錯誤，切換到本地模型
    
    Args:
        error: 捕獲的異常
    
    Returns:
        如果切換到本地模型，返回 MLXChatModel；否則返回 None
    """
    global _current_llm_type, _groq_quota_exceeded
    
    error_str = str(error).lower()
    
    # 檢查是否為額度相關錯誤
    quota_indicators = [
        "quota",
        "rate limit",
        "429",
        "insufficient",
        "exceeded",
        "limit reached"
    ]
    
    if any(indicator in error_str for indicator in quota_indicators):
        if not _groq_quota_exceeded:
            _groq_quota_exceeded = True
            warning_msg = "⚠️ 警告：Groq API 額度已用完，正在切換到本地 MLX 模型 (Qwen2.5)"
            print(warning_msg)
            warnings.warn(warning_msg, UserWarning)
        
        _current_llm_type = "mlx"
        model, tokenizer = load_mlx_model()
        return MLXChatModel(
            model=model,
            tokenizer=tokenizer,
            max_tokens=MLX_MAX_TOKENS,
            temperature=MLX_TEMPERATURE
        )
    
    return None

