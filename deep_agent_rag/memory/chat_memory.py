"""
Simple Chatbot 長期記憶：使用 Chroma 儲存對話摘要，下次提問時自動檢索。
- 對話結束（按「清除對話」）時將當前對話摘要寫入 Chroma。
- 每次回覆前依用戶問題做語義檢索，將相關記憶注入 context。
"""
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Embedding：與專案 RAG 一致（優先使用 langchain_huggingface）
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    Chroma = None

# 預設路徑（專案根目錄下，與 chroma_db / chroma_db_private 並列）
_DEFAULT_PERSIST_DIR = "./chroma_db_chat_memory"
_COLLECTION_NAME = "simple_chatbot_memory"
_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None
_vectorstore = None


def _get_embeddings():
    """懶加載 HuggingFace embeddings（與 RAG 同模型）。"""
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    if HuggingFaceEmbeddings is None:
        raise ImportError("需要安裝 langchain-community 以使用長期記憶")
    _embeddings = HuggingFaceEmbeddings(
        model_name=_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return _embeddings


def _get_vectorstore(persist_directory: str = _DEFAULT_PERSIST_DIR):
    """取得或建立 Chroma 向量庫（持久化）。"""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    if Chroma is None:
        raise ImportError("需要安裝 chromadb / langchain-community 以使用長期記憶")
    os.makedirs(persist_directory, exist_ok=True)
    _vectorstore = Chroma(
        collection_name=_COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=persist_directory,
    )
    return _vectorstore


def _stringify_content(value: Any) -> str:
    """
    將 Gradio Chatbot 可能出現的 content 形態安全轉為文字。
    - str / number / bool
    - dict（例如 {type/text/...}）
    - list/tuple（多段內容或 multimodal 片段）
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        # 常見：{"type": "text", "text": "..."} 或 {"text": "..."}
        if value.get("type") == "text" and "text" in value:
            return _stringify_content(value.get("text"))
        if "text" in value:
            return _stringify_content(value.get("text"))
        if "content" in value:
            return _stringify_content(value.get("content"))
        return str(value)
    if isinstance(value, (list, tuple)):
        parts = []
        for v in value:
            s = _stringify_content(v).strip()
            if s:
                parts.append(s)
        return "\n".join(parts)
    return str(value)


def _iter_history_messages(history: List[Any]) -> Iterable[Tuple[str, str]]:
    """
    支援兩種常見 Gradio `gr.Chatbot` history 格式：
    1) messages: [{"role": "user"/"assistant", "content": ...}, ...]
    2) tuples: [[user, assistant], [user, assistant], ...]
    """
    for item in history or []:
        # messages 格式
        if isinstance(item, dict):
            role = str(item.get("role", "") or "")
            content = _stringify_content(item.get("content")).strip()
            if content:
                yield role, content
            continue

        # tuples 格式：一輪 [user, assistant]
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user, assistant = item
            user_text = _stringify_content(user).strip()
            if user_text:
                yield "user", user_text
            assistant_text = _stringify_content(assistant).strip()
            if assistant_text:
                yield "assistant", assistant_text
            continue

        # 其他非預期格式：盡量保留
        text = _stringify_content(item).strip()
        if text:
            yield "unknown", text


def _format_history_for_summary(history: List[Any]) -> str:
    """將 Gradio history 轉成純文字，供摘要或儲存（兼容 messages/tuples）。"""
    lines: List[str] = []
    for role, content in _iter_history_messages(history):
        if role == "user":
            prefix = "用戶："
        elif role == "assistant":
            prefix = "助手："
        else:
            prefix = "內容："
        lines.append(f"{prefix}{content}")
    return "\n".join(lines)


def save_conversation_summary(
    history: List[Any],
    user_id: str = "default",
    persist_directory: str = _DEFAULT_PERSIST_DIR,
    use_llm_summary: bool = True,
) -> None:
    """
    將一輪對話摘要後寫入 Chroma（對話結束時呼叫，例如按「清除對話」）。
    
    Args:
        history: Gradio 對話歷史（兼容 messages 或 tuples 格式）
        user_id: 用戶/會話 ID（目前無登入，用 "default"）
        persist_directory: Chroma 持久化目錄
        use_llm_summary: 是否用 LLM 壓縮成簡短摘要（True 較精簡，False 則存原始對話文字）
    """
    if not history or len(history) == 0:
        return
    try:
        vs = _get_vectorstore(persist_directory)
        raw = _format_history_for_summary(history)
        if not raw.strip():
            return
        if use_llm_summary:
            summary = _summarize_with_llm(raw)
            if not summary or not summary.strip():
                summary = raw[:2000]  # fallback：截斷
        else:
            summary = raw[:2000]
        meta = {
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "episodic",
        }
        vs.add_texts(texts=[summary], metadatas=[meta])
        vs.persist()
        print(f"📝 長期記憶已儲存（{len(summary)} 字）")
    except Exception as e:
        print(f"⚠️ 長期記憶儲存失敗: {e}")


def _summarize_with_llm(raw_conversation: str, max_chars: int = 800) -> Optional[str]:
    """用 LLM 將對話壓成簡短摘要（關鍵事實、偏好、未解問題）。"""
    try:
        from ..utils.llm_utils import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = get_llm()
        prompt = f"""請將以下對話壓縮成一段簡短摘要（2～5 句），用繁體中文。
摘要應包含：關鍵事實、用戶偏好或需求、尚未解決的問題。
不要加入「根據對話」等廢話，直接寫摘要內容。

對話內容：
---
{raw_conversation[:4000]}
---
摘要："""
        msg = [HumanMessage(content=prompt)]
        out = llm.invoke(msg)
        text = out.content if hasattr(out, "content") else str(out)
        return (text or "").strip()[:max_chars]
    except Exception as e:
        print(f"⚠️ 摘要生成失敗，改存原始片段: {e}")
        return raw_conversation[:max_chars]


def retrieve_memories(
    query: str,
    user_id: str = "default",
    k: int = 5,
    persist_directory: str = _DEFAULT_PERSIST_DIR,
) -> str:
    """
    依當前問題檢索相關長期記憶，組成一串可放入 system 的文案。
    
    Args:
        query: 用戶當前問題
        user_id: 用戶/會話 ID
        k: 最多取幾筆
        persist_directory: Chroma 目錄
    
    Returns:
        多段記憶用換行拼接的字串；若無記憶或出錯則回傳空字串。
    """
    if not query or not query.strip():
        return ""
    try:
        vs = _get_vectorstore(persist_directory)
        # 可選：用 metadata 過濾 user_id；Chroma 的 filter 依版本而異，這裡先不過濾
        docs = vs.similarity_search(query, k=k)
        if not docs:
            return ""
        parts = [doc.page_content.strip() for doc in docs if doc.page_content.strip()]
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        print(f"⚠️ 長期記憶檢索失敗: {e}")
        return ""


def clear_chat_memory(persist_directory: str = _DEFAULT_PERSIST_DIR) -> None:
    """清空該目錄下 Simple Chatbot 的長期記憶（刪除 collection 或目錄）。僅供需要「重置記憶」時使用。"""
    global _vectorstore
    try:
        import shutil
        if os.path.isdir(persist_directory):
            shutil.rmtree(persist_directory)
            _vectorstore = None
            print("🗑️ 長期記憶已清空")
    except Exception as e:
        print(f"⚠️ 清空長期記憶失敗: {e}")
