"""
Step-back Prompting 雙軌 RAG：結合具體事實與抽象原理
使用 Step-back Prompting 技術，同時檢索具體事實和抽象原理，提升回答質量
"""
from typing import List, Dict, Optional
from .retrievers.reranker import RAGPipeline
from .retrievers.vector_retriever import VectorRetriever
from .prompt_formatter import PromptFormatter
from .llm_integration import OllamaLLM
from .rag_utils import get_doc_id, generate_step_back_question
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class StepBackRAG:
    """使用 Step-back Prompting 的雙軌 RAG 系統"""
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        vector_retriever: VectorRetriever,
        llm: OllamaLLM,
        step_back_temperature: float = 0.3,  # 生成抽象問題時使用較低溫度
        answer_temperature: float = 0.7,
        enable_parallel: bool = True
    ):
        """
        初始化 Step-back RAG
        
        Args:
            rag_pipeline: RAG 管線實例（用於最終答案生成）
            vector_retriever: 向量檢索器
            llm: LLM 實例
            step_back_temperature: 生成抽象問題的溫度（較低，更穩定）
            answer_temperature: 生成答案的溫度
            enable_parallel: 是否並行執行雙軌檢索
        """
        self.rag_pipeline = rag_pipeline
        self.vector_retriever = vector_retriever
        self.llm = llm
        self.step_back_temperature = step_back_temperature
        self.answer_temperature = answer_temperature
        self.enable_parallel = enable_parallel
    
    def _generate_step_back_question(self, question: str) -> str:
        """
        生成 Step-back 抽象問題
        
        Args:
            question: 原始具體問題
            
        Returns:
            抽象問題
        """
        return generate_step_back_question(
            llm=self.llm,
            question=question,
            temperature=self.step_back_temperature,
            enable_logging=True
        )
    
    def _retrieve_direct(self, question: str, top_k: int, metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """直接檢索原始問題（具體事實）"""
        return self.vector_retriever.retrieve(
            query=question,
            top_k=top_k,
            metadata_filter=metadata_filter
        )
    
    def _retrieve_step_back(self, question: str, top_k: int, metadata_filter: Optional[Dict] = None) -> tuple:
        """Step-back 檢索（抽象原理）"""
        abstract_question = self._generate_step_back_question(question)
        results = self.vector_retriever.retrieve(
            query=abstract_question,
            top_k=top_k,
            metadata_filter=metadata_filter
        )
        return results, abstract_question
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
        return_abstract_question: bool = False
    ) -> Dict:
        """
        執行雙軌檢索（不生成答案）
        
        Args:
            question: 原始問題
            top_k: 每軌返回的結果數量
            metadata_filter: 可選的 metadata 過濾條件
            return_abstract_question: 是否返回抽象問題
            
        Returns:
            包含雙軌檢索結果的字典
        """
        start_time = time.time()
        
        if self.enable_parallel:
            # 並行執行雙軌檢索
            logger.info(f"🔄 並行執行雙軌檢索: '{question}'")
            with ThreadPoolExecutor(max_workers=2) as executor:
                direct_future = executor.submit(
                    self._retrieve_direct, question, top_k, metadata_filter
                )
                step_back_future = executor.submit(
                    self._retrieve_step_back, question, top_k, metadata_filter
                )
                
                specific_results = direct_future.result()
                abstract_results, abstract_question = step_back_future.result()
        else:
            # 串行執行
            logger.info(f"🔄 串行執行雙軌檢索: '{question}'")
            specific_results = self._retrieve_direct(question, top_k, metadata_filter)
            abstract_results, abstract_question = self._retrieve_step_back(question, top_k, metadata_filter)
        
        elapsed_time = time.time() - start_time
        logger.info(
            f"✅ 雙軌檢索完成（耗時: {elapsed_time:.2f}s）\n"
            f"   具體事實: {len(specific_results)} 個結果\n"
            f"   抽象原理: {len(abstract_results)} 個結果"
        )
        
        return {
            "specific_context": specific_results,
            "abstract_context": abstract_results,
            "abstract_question": abstract_question if return_abstract_question else None,
            "question": question,
            "elapsed_time": elapsed_time
        }
    
    def generate_answer(
        self,
        question: str,
        formatter: PromptFormatter,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
        document_type: str = "general",
        return_abstract_question: bool = False
    ) -> Dict:
        """
        完整的 Step-back RAG 流程：雙軌檢索 -> 生成答案
        
        Args:
            question: 原始問題
            formatter: Prompt 格式化器
            top_k: 每軌用於生成答案的文檔數量
            metadata_filter: 可選的 metadata 過濾條件
            document_type: 文檔類型 ("paper", "cv", "general")
            return_abstract_question: 是否返回抽象問題
            
        Returns:
            包含檢索結果、生成的答案和統計資訊的字典
        """
        start_time = time.time()
        
        # 第一步：雙軌檢索
        retrieval_result = self.query(
            question=question,
            top_k=top_k,
            metadata_filter=metadata_filter,
            return_abstract_question=return_abstract_question
        )
        
        specific_results = retrieval_result["specific_context"]
        abstract_results = retrieval_result["abstract_context"]
        
        if not specific_results and not abstract_results:
            return {
                **retrieval_result,
                "answer": "抱歉，未找到相關文檔來回答此問題。",
                "formatted_context": None,
                "answer_time": 0.0,
                "total_time": retrieval_result["elapsed_time"]
            }
        
        # 第二步：格式化雙軌上下文
        specific_context = formatter.format_context(
            specific_results,
            document_type=document_type
        ) if specific_results else "未找到相關的具體事實資料。"
        
        abstract_context = formatter.format_context(
            abstract_results,
            document_type=document_type
        ) if abstract_results else "未找到相關的基礎原理資料。"
        
        # 第三步：創建融合提示詞（關鍵步驟）
        is_chinese = PromptFormatter.detect_language(question) == "zh"
        
        if is_chinese:
            final_prompt = f"""你是一個資深專家。請結合以下兩類資訊來回答使用者的具體問題。

【基礎原理與背景】
{abstract_context}

【具體事實資料】
{specific_context}

使用者問題：{question}

請根據原理推導並結合事實，給出一個專業且具備邏輯的回答：
"""
        else:
            final_prompt = f"""You are a senior expert. Please answer the user's specific question by combining the following two types of information.

【Fundamental Principles and Background】
{abstract_context}

【Specific Facts and Data】
{specific_context}

User question: {question}

Please provide a professional and logical answer based on principles and facts:
"""
        
        # 第四步：生成回答
        logger.info("🤖 生成回答中...")
        answer_start = time.time()
        try:
            answer = self.llm.generate(
                prompt=final_prompt,
                temperature=self.answer_temperature,
                max_tokens=2048
            )
            answer_time = time.time() - answer_start
            logger.info(f"✅ 回答生成完成（耗時: {answer_time:.2f}s）")
        except Exception as e:
            logger.error(f"❌ 生成回答時出錯: {e}")
            answer = f"生成回答時出錯: {e}"
            answer_time = time.time() - answer_start
        
        total_time = time.time() - start_time
        
        return {
            **retrieval_result,
            "answer": answer,
            "formatted_context": {
                "specific": specific_context,
                "abstract": abstract_context
            },
            "answer_time": answer_time,
            "total_time": total_time
        }

