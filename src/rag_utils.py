"""
RAG 工具函數模組：提取重複的工具方法
"""
from typing import List, Dict
import hashlib
import logging
from .prompt_formatter import PromptFormatter
from .llm_integration import OllamaLLM

logger = logging.getLogger(__name__)


def get_doc_id(doc: Dict) -> str:
    """
    生成文檔的唯一標識符
    
    Args:
        doc: 文檔字典
        
    Returns:
        唯一 ID
    """
    metadata = doc.get("metadata", {})
    content = doc.get("content", "")
    
    # 使用 metadata 中的唯一標識（如果有的話）
    if "arxiv_id" in metadata and "chunk_index" in metadata:
        return f"{metadata['arxiv_id']}_{metadata['chunk_index']}"
    elif "file_path" in metadata and "chunk_index" in metadata:
        return f"{metadata['file_path']}_{metadata['chunk_index']}"
    else:
        # 回退到內容的 hash
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        return f"doc_{content_hash}"


def generate_sub_queries(
    llm: OllamaLLM,
    question: str,
    max_sub_queries: int = 3,
    temperature: float = 0.3
) -> List[str]:
    """
    將原始問題拆解成子問題
    
    Args:
        llm: LLM 實例
        question: 原始問題
        max_sub_queries: 最多生成的子問題數量
        temperature: 生成溫度參數
        
    Returns:
        子問題列表
    """
    # 檢測語言
    is_chinese = PromptFormatter.detect_language(question) == "zh"
    
    if is_chinese:
        prompt = f"""你是一個專業助理。請將以下原始問題拆解成最多 {max_sub_queries} 個具體的子問題，以便進行資料搜尋。
每個子問題應專注於原始問題的一個特定面向。請以換行符號分隔問題。

原始問題: {question}

子問題清單:"""
    else:
        prompt = f"""You are a professional assistant. Please decompose the following original question into at most {max_sub_queries} specific sub-questions for information retrieval.
Each sub-question should focus on a specific aspect of the original question. Please separate questions with newlines.

Original question: {question}

Sub-question list:"""
    
    try:
        response = llm.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=500
        )
        
        # 解析子問題
        sub_queries = [
            q.strip() 
            for q in response.strip().split("\n") 
            if q.strip() and not q.strip().startswith("#")
        ]
        
        # 移除編號前綴（如 "1. ", "1) " 等）
        cleaned_queries = []
        for q in sub_queries:
            # 移除開頭的編號
            q = q.lstrip("0123456789. )")
            q = q.strip()
            if q:
                cleaned_queries.append(q)
        
        # 限制數量
        cleaned_queries = cleaned_queries[:max_sub_queries]
        
        # 如果沒有生成子問題，使用原始問題
        if not cleaned_queries:
            logger.warning("⚠️  未生成子問題，使用原始問題")
            cleaned_queries = [question]
        
        return cleaned_queries
        
    except Exception as e:
        logger.error(f"⚠️  生成子問題時出錯: {e}")
        # 回退到原始問題
        return [question]


def generate_hypothetical_document(
    llm: OllamaLLM,
    question: str,
    hypothetical_length: int = 200,
    temperature: float = 0.7,
    enable_logging: bool = True
) -> str:
    """
    生成假設性文檔（Hypothetical Document）
    
    Args:
        llm: LLM 實例
        question: 用戶問題
        hypothetical_length: 假設性文檔的目標長度（字符數）
        temperature: 生成假設性文檔時的溫度參數
        enable_logging: 是否啟用日誌記錄
        
    Returns:
        假設性文檔文本
    """
    # 檢測語言
    is_chinese = PromptFormatter.detect_language(question) == "zh"
    
    if is_chinese:
        prompt = f"""請針對以下問題，寫出一段約 {hypothetical_length} 字的專業技術檔案內容。
這段內容應包含該領域常見的專業術語與原理說明，以便用於後續的語義檢索。
請使用專業的術語和概念，即使你對某些細節不確定，也要包含相關的專業詞彙。

問題: {question}

專業技術內容："""
    else:
        prompt = f"""Please write a professional technical document of approximately {hypothetical_length} words in response to the following question.
This content should include common professional terminology and principle explanations in this field, to be used for subsequent semantic retrieval.
Please use professional terms and concepts, and include relevant professional vocabulary even if you are uncertain about some details.

Question: {question}

Professional technical content:"""
    
    try:
        hypothetical_doc = llm.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=500
        )
        
        # 清理輸出
        hypothetical_doc = hypothetical_doc.strip()
        
        if not hypothetical_doc:
            if enable_logging:
                logger.warning("⚠️  生成的假設性文檔為空，使用原始問題")
            return question
        
        if enable_logging:
            logger.info(f"✅ 生成假設性文檔（長度: {len(hypothetical_doc)} 字符）")
        return hypothetical_doc
        
    except Exception as e:
        logger.error(f"⚠️  生成假設性文檔時出錯: {e}")
        # 回退到使用原始問題
        return question


def generate_step_back_question(
    llm: OllamaLLM,
    question: str,
    temperature: float = 0.3,
    enable_logging: bool = True
) -> str:
    """
    生成 Step-back 抽象問題
    
    Args:
        llm: LLM 實例
        question: 原始具體問題
        temperature: 生成抽象問題的溫度參數
        enable_logging: 是否啟用日誌記錄
        
    Returns:
        抽象問題
    """
    is_chinese = PromptFormatter.detect_language(question) == "zh"
    
    if is_chinese:
        prompt = f"""你是一個資深專家。請將以下具體問題轉換為一個更抽象、更基礎的原理性問題。
這個抽象問題應該幫助理解該領域的基礎概念和原理，而不是直接回答具體問題。

具體問題: {question}

請生成一個抽象問題，用於檢索相關的原理和背景知識：
"""
    else:
        prompt = f"""You are a senior expert. Please convert the following specific question into a more abstract, fundamental question about principles and concepts.
This abstract question should help understand the basic concepts and principles in this field, rather than directly answering the specific question.

Specific question: {question}

Please generate an abstract question for retrieving relevant principles and background knowledge:
"""
    
    try:
        abstract_question = llm.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=200
        )
        
        abstract_question = abstract_question.strip()
        
        if not abstract_question:
            if enable_logging:
                logger.warning("⚠️  生成的抽象問題為空，使用原始問題")
            return question
        
        if enable_logging:
            logger.info(f"✅ 生成抽象問題: '{abstract_question}'")
        return abstract_question
        
    except Exception as e:
        logger.error(f"⚠️  生成抽象問題時出錯: {e}")
        return question
