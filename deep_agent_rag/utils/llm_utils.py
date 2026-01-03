"""
LLM 工具函數
提供 LLM 實例的創建和管理
優先順序：Groq API > Ollama > MLX 模型
"""
import warnings
from typing import Optional
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from ..models import MLXChatModel, load_mlx_model
from ..config import (
    MLX_MAX_TOKENS, 
    MLX_TEMPERATURE,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE,
    USE_GROQ_FIRST,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_MAX_TOKENS,
    OLLAMA_TEMPERATURE,
    USE_OLLAMA,
)

# 全局變量：跟踪當前使用的 LLM 類型
_current_llm_type = None
_groq_quota_exceeded = False


def get_llm_type() -> str:
    """獲取當前使用的 LLM 類型"""
    return _current_llm_type or "unknown"


def is_using_local_llm() -> bool:
    """檢查是否正在使用本地 LLM"""
    return _current_llm_type in ["mlx", "ollama"] or _groq_quota_exceeded


def get_llm():
    """
    獲取 LLM 實例
    優先順序：Groq API > Ollama > MLX 模型
    """
    global _current_llm_type, _groq_quota_exceeded
    
    # 優先順序 1: Groq API
    if USE_GROQ_FIRST and GROQ_API_KEY:
        try:
            groq_llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=GROQ_MODEL,
                max_tokens=GROQ_MAX_TOKENS,
                temperature=GROQ_TEMPERATURE
            )
            _current_llm_type = "groq"
            print("✅ 使用 Groq API (優先)")
            return groq_llm
        except Exception as e:
            # 如果創建失敗，繼續嘗試其他選項
            print(f"⚠️ Groq API 初始化失敗: {e}")
            # 不立即設置 _groq_quota_exceeded，先嘗試 Ollama
    
    # 優先順序 2: Ollama (Llama 3.2 或其他模型)
    if USE_OLLAMA:
        try:
            ollama_llm = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                num_predict=OLLAMA_MAX_TOKENS,
                temperature=OLLAMA_TEMPERATURE,
            )
            _current_llm_type = "ollama"
            print(f"✅ 使用 Ollama 模型 ({OLLAMA_MODEL})")
            return ollama_llm
        except Exception as e:
            print(f"⚠️ Ollama 初始化失敗: {e}")
            print("   請確保 Ollama 服務正在運行: ollama serve")
            print("   或檢查模型是否已下載: ollama pull " + OLLAMA_MODEL)
    
    # 優先順序 3: MLX 模型（備援）
    # 如果 Groq 額度用完，記錄狀態
    if _groq_quota_exceeded and _current_llm_type != "mlx":
        print("⚠️ 警告：Groq API 額度已用完，已切換到本地 MLX 模型 (Qwen2.5)")
    elif _current_llm_type != "mlx":
        if not GROQ_API_KEY and not USE_OLLAMA:
            print("ℹ️ 未配置 Groq API 或 Ollama，使用本地 MLX 模型")
        elif not USE_OLLAMA:
            print("ℹ️ Ollama 未啟用，使用本地 MLX 模型作為備援")
    
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
    如果是額度用完錯誤，先嘗試切換到 Ollama，否則切換到 MLX 模型
    
    Args:
        error: 捕獲的異常
    
    Returns:
        如果切換到本地模型，返回 ChatOllama 或 MLXChatModel；否則返回 None
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
            warning_msg = "⚠️ 警告：Groq API 額度已用完"
            print(warning_msg)
            warnings.warn(warning_msg, UserWarning)
        
        # 先嘗試使用 Ollama
        if USE_OLLAMA:
            try:
                ollama_llm = ChatOllama(
                    base_url=OLLAMA_BASE_URL,
                    model=OLLAMA_MODEL,
                    num_predict=OLLAMA_MAX_TOKENS,
                    temperature=OLLAMA_TEMPERATURE,
                )
                _current_llm_type = "ollama"
                print(f"✅ 已切換到 Ollama 模型 ({OLLAMA_MODEL})")
                return ollama_llm
            except Exception as e:
                print(f"⚠️ Ollama 切換失敗: {e}")
                print("   回退到 MLX 模型")
        
        # 回退到 MLX 模型
        _current_llm_type = "mlx"
        model, tokenizer = load_mlx_model()
        return MLXChatModel(
            model=model,
            tokenizer=tokenizer,
            max_tokens=MLX_MAX_TOKENS,
            temperature=MLX_TEMPERATURE
        )
    
    return None

