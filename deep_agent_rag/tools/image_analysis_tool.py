"""
圖片分析工具
使用多模態 LLM 分析圖片並返回描述
支持多個 API 提供商，自動切換優先級
"""
import os
import base64
from typing import Optional, Tuple
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from PIL import Image
import requests

from ..config import (
    GOOGLE_GEMINI_API_KEY,
    GOOGLE_GEMINI_MODEL,
    USE_GEMINI_FIRST, 
    OLLAMA_BASE_URL,
    OLLAMA_VISION_MODEL,
    USE_OLLAMA_VISION,
)


def validate_image_file(image_path: str) -> bool:
    """
    驗證圖片文件是否存在且格式有效
    
    Args:
        image_path: 圖片文件路徑
    
    Returns:
        如果圖片有效則返回 True
    """
    if not image_path or not image_path.strip():
        return False
    
    image_path = image_path.strip()
    
    # 檢查文件是否存在
    if not os.path.exists(image_path):
        return False
    
    # 檢查是否為圖片文件
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_mime_type(image_path: str) -> str:
    """
    獲取圖片的 MIME 類型
    
    Args:
        image_path: 圖片文件路徑
    
    Returns:
        MIME 類型字符串
    """
    image_ext = Path(image_path).suffix[1:].lower()
    mime_type_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp',
        'bmp': 'image/bmp'
    }
    return mime_type_map.get(image_ext, 'image/jpeg')


def prepare_image_message(image_path: str, prompt: str) -> HumanMessage:
    """
    準備包含圖片的消息
    
    Args:
        image_path: 圖片文件路徑
        prompt: 文字提示
    
    Returns:
        HumanMessage 對象
    """
    # 讀取圖片並編碼為 base64
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    mime_type = get_image_mime_type(image_path)
    
    return HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                }
            }
        ]
    )

def get_gemini_llm():
    """獲取 Google Gemini LLM 實例"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        if not GOOGLE_GEMINI_API_KEY:
            return None
        
        return ChatGoogleGenerativeAI(
            model=GOOGLE_GEMINI_MODEL,
            google_api_key=GOOGLE_GEMINI_API_KEY,
            temperature=0.7,
            max_tokens=2048,
        )
    except ImportError:
        return None
    except Exception as e:
        print(f"⚠️ Google Gemini 初始化失敗: {e}")
        return None




def get_ollama_vision_llm():
    """獲取 Ollama LLaVA LLM 實例（本地多模態模型）"""
    try:
        from langchain_ollama import ChatOllama
        import requests
        
        if not USE_OLLAMA_VISION:
            return None
        
        # 先檢查 Ollama 服務是否可用
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            if response.status_code != 200:
                print(f"⚠️ Ollama 服務不可用（狀態碼：{response.status_code}）")
                return None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"⚠️ 無法連接到 Ollama 服務 ({OLLAMA_BASE_URL})：{e}")
            print(f"   提示：請確保 Ollama 服務正在運行（執行 'ollama serve'）")
            return None
        except Exception as e:
            print(f"⚠️ 檢查 Ollama 服務時發生錯誤：{e}")
            # 繼續嘗試創建 LLM，可能只是檢查失敗
        
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_VISION_MODEL,
            num_predict=2048,
            temperature=0.7,
            timeout=30,  # 設置超時時間
        )
    except ImportError:
        return None
    except Exception as e:
        print(f"⚠️ Ollama LLaVA 初始化失敗: {e}")
        return None


def get_multimodal_llm() -> Tuple[Optional[object], str]:
    """
    獲取多模態 LLM 實例（按優先級嘗試）
    
    Returns:
        (LLM 實例, 提供商名稱) 或 (None, 錯誤訊息)
    """
    # 優先順序：OpenAI > Gemini > Anthropic > Ollama
    
    
    # 2. 嘗試 Google Gemini（默認優先，免費額度較高）
    if USE_GEMINI_FIRST:
        llm = get_gemini_llm()
        if llm:
            print("✅ 使用 Google Gemini")
            return llm, "gemini"
    
    
    # 4. 嘗試 Ollama LLaVA（本地，完全免費）
    if USE_OLLAMA_VISION:
        llm = get_ollama_vision_llm()
        if llm:
            print("✅ 使用 Ollama LLaVA (本地)")
            return llm, "ollama"
    
    
    llm = get_gemini_llm()
    if llm:
        print("✅ 使用 Google Gemini (備援)")
        return llm, "gemini"
    
    
    
    llm = get_ollama_vision_llm()
    if llm:
        print("✅ 使用 Ollama LLaVA (備援)")
        return llm, "ollama"
    
    # 所有都失敗
    error_msg = (
        "❌ 未找到可用的多模態 LLM API。\n\n"
        "請至少配置以下之一：\n"
        "1. OpenAI GPT-4 Vision: 設置 OPENAI_API_KEY\n"
        "2. Google Gemini: 設置 GOOGLE_GEMINI_API_KEY（推薦，免費額度較高）\n"
        "3. Anthropic Claude: 設置 ANTHROPIC_API_KEY\n"
        "4. Ollama LLaVA: 安裝並運行 Ollama，然後執行 'ollama pull llava'（完全免費，本地運行）\n\n"
        "在 .env 文件中設置相應的 API Key，或啟用 USE_OLLAMA_VISION=true 使用本地模型。"
    )
    return None, error_msg


def _analyze_image_internal(image_path: str, question: Optional[str] = None) -> str:
    """
    內部函數：使用多模態 LLM 分析圖片並返回描述
    這個函數可以被直接調用，不依賴 @tool 裝飾器
    
    Args:
        image_path: 圖片文件路徑（支持常見圖片格式：jpg, png, gif, webp 等）
        question: 可選的特定問題，例如："這張圖片中有什麼？"、"描述圖片中的場景"等。
                  如果不提供，將進行通用圖片分析
    
    Returns:
        圖片分析的結果描述
    """
    try:
        # 驗證圖片文件
        if not validate_image_file(image_path):
            return (
                f"❌ 錯誤：無法讀取圖片文件。\n"
                f"請確保：\n"
                f"1. 文件路徑正確：{image_path}\n"
                f"2. 文件存在且可讀取\n"
                f"3. 文件格式為有效的圖片格式（jpg, png, gif, webp 等）"
            )
        
        # 獲取多模態 LLM
        llm, provider = get_multimodal_llm()
        if llm is None:
            return provider  # 返回錯誤訊息
        
        # 構建提示詞
        if question:
            prompt = f"{question}\n\n請詳細分析這張圖片。"
        else:
            prompt = (
                "請詳細分析這張圖片，包括：\n"
                "1. 圖片的主要內容和場景\n"
                "2. 圖片中的物體、人物、文字等元素\n"
                "3. 圖片的風格、色彩、構圖等視覺特徵\n"
                "4. 圖片可能表達的意義或情感\n"
                "請用中文回答。"
            )
        
        # 準備消息
        message = prepare_image_message(image_path, prompt)
        
        # 調用 LLM 分析圖片
        try:
            response = llm.invoke([message])
            
            # 提取回答內容
            if hasattr(response, 'content'):
                answer = response.content
            elif isinstance(response, str):
                answer = response
            else:
                answer = str(response)
            
            return answer.strip()
            
        except Exception as e:
            error_msg = str(e).lower()
            error_str = str(e)
            
            # 處理連接錯誤（特別是 Ollama）
            if "connection refused" in error_msg or "errno 61" in error_msg or "errno 111" in error_msg:
                if provider == "ollama":
                    return (
                        f"❌ 錯誤：無法連接到 Ollama 服務。\n\n"
                        f"**解決方法：**\n"
                        f"1. 確保 Ollama 服務正在運行：\n"
                        f"   在終端執行：ollama serve\n\n"
                        f"2. 或者禁用 Ollama，使用其他 API 提供商：\n"
                        f"   在 .env 文件中設置：USE_OLLAMA_VISION=false\n\n"
                        f"3. 檢查 Ollama 服務地址：{OLLAMA_BASE_URL}\n\n"
                        f"**提示：** 如果不想使用 Ollama，請配置 Google Gemini API（免費額度較高）"
                    )
                else:
                    return (
                        f"❌ 錯誤：無法連接到 {provider.upper()} API。\n"
                        f"請檢查網絡連接和 API 服務狀態。\n"
                        f"錯誤詳情：{error_str}"
                    )
            
            # 處理 API 錯誤
            elif "api key" in error_msg or "authentication" in error_msg:
                return (
                    f"❌ 錯誤：{provider.upper()} API 認證失敗。\n"
                    f"請檢查 API Key 是否正確設置。"
                )
            elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                # 如果額度用完，嘗試下一個提供商
                print(f"⚠️ {provider} API 額度已用完，嘗試下一個提供商...")
                # 這裡可以實現自動切換邏輯，但為了簡化，先返回錯誤
                return (
                    f"❌ 錯誤：{provider.upper()} API 額度已用完或達到速率限制。\n"
                    f"請稍後再試，或配置其他 API 提供商（如 Ollama LLaVA，完全免費）。"
                )
            elif "model" in error_msg and "not found" in error_msg:
                if provider == "ollama":
                    return (
                        f"❌ 錯誤：Ollama 模型 '{OLLAMA_VISION_MODEL}' 未找到。\n"
                        f"請執行：ollama pull {OLLAMA_VISION_MODEL}"
                    )
                else:
                    return f"❌ 錯誤：{provider.upper()} 模型未找到或不可用。"
            elif "timeout" in error_msg:
                return (
                    f"❌ 錯誤：連接 {provider.upper()} API 超時。\n"
                    f"請檢查網絡連接或稍後再試。"
                )
            else:
                return (
                    f"❌ 分析圖片時發生錯誤：{error_str}\n\n"
                    f"**提供商：** {provider.upper()}\n"
                    f"**建議：** 請檢查 API 配置或嘗試使用其他提供商"
                )
                
    except Exception as e:
        error_msg = str(e)
        print(f"Image Analysis Tool 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return f"❌ 圖片分析工具發生錯誤：{error_msg}"


@tool
def analyze_image(image_path: str, question: Optional[str] = None) -> str:
    """
    使用多模態 LLM 分析圖片並返回描述
    自動選擇可用的 API 提供商（OpenAI > Gemini > Anthropic > Ollama）
    
    Args:
        image_path: 圖片文件路徑（支持常見圖片格式：jpg, png, gif, webp 等）
        question: 可選的特定問題，例如："這張圖片中有什麼？"、"描述圖片中的場景"等。
                  如果不提供，將進行通用圖片分析
    
    Returns:
        圖片分析的結果描述
    """
    return _analyze_image_internal(image_path, question)
