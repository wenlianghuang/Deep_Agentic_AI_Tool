"""
Triple Hybrid RAG：融合 SubQuery + HyDE + Step-back Prompting
結合三種技術的優勢，實現最強大的 RAG 系統
"""
from typing import List, Dict, Optional
from .retrievers.reranker import RAGPipeline
from .retrievers.vector_retriever import VectorRetriever
from .prompt_formatter import PromptFormatter
from .llm_integration import OllamaLLM
from .rag_utils import get_doc_id, generate_sub_queries, generate_hypothetical_document, generate_step_back_question
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class TripleHybridRAG:
    """融合 SubQuery + HyDE + Step-back 的三重混合 RAG 系統"""
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        vector_retriever: VectorRetriever,
        llm: OllamaLLM,
        max_sub_queries: int = 3,
        top_k_per_subquery: int = 5,
        hypothetical_length: int = 200,
        temperature_subquery: float = 0.3,
        temperature_hyde: float = 0.7,
        temperature_stepback: float = 0.3,
        answer_temperature: float = 0.7,
        enable_parallel: bool = True
    ):
        """
        初始化三重混合 RAG
        
        Args:
            rag_pipeline: RAG 管線實例
            vector_retriever: 向量檢索器
            llm: LLM 實例
            max_sub_queries: 最多生成的子問題數量
            top_k_per_subquery: 每個子問題檢索的結果數量
            hypothetical_length: 假設性文檔目標長度（字符數）
            temperature_subquery: 生成子問題的溫度（較低，更穩定）
            temperature_hyde: 生成假設性文檔的溫度（較高，更多專業術語）
            temperature_stepback: 生成抽象問題的溫度（較低，更穩定）
            answer_temperature: 生成答案的溫度
            enable_parallel: 是否並行處理
        """
        self.rag_pipeline = rag_pipeline
        self.vector_retriever = vector_retriever
        self.llm = llm
        self.max_sub_queries = max_sub_queries
        self.top_k_per_subquery = top_k_per_subquery
        self.hypothetical_length = hypothetical_length
        self.temperature_subquery = temperature_subquery
        self.temperature_hyde = temperature_hyde
        self.temperature_stepback = temperature_stepback
        self.answer_temperature = answer_temperature
        self.enable_parallel = enable_parallel
    
    def _generate_sub_queries(self, question: str) -> List[str]:
        """生成子問題（SubQuery）"""
        return generate_sub_queries(
            llm=self.llm,
            question=question,
            max_sub_queries=self.max_sub_queries,
            temperature=self.temperature_subquery
        )
    
    def _generate_hypothetical_document(self, sub_query: str) -> str:
        """為子問題生成假設性文檔（HyDE）"""
        return generate_hypothetical_document(
            llm=self.llm,
            question=sub_query,
            hypothetical_length=self.hypothetical_length,
            temperature=self.temperature_hyde,
            enable_logging=False
        )
    
    def _generate_step_back_question(self, question: str) -> str:
        """生成 Step-back 抽象問題"""
        return generate_step_back_question(
            llm=self.llm,
            question=question,
            temperature=self.temperature_stepback,
            enable_logging=False
        )
    
    def _process_subquery_with_hyde(
        self, 
        sub_query: str, 
        metadata_filter: Optional[Dict] = None
    ) -> tuple:
        """處理單個子問題：生成假設性文檔並檢索"""
        try:
            hypothetical_doc = self._generate_hypothetical_document(sub_query)
            results = self.vector_retriever.retrieve(
                query=hypothetical_doc,
                top_k=self.top_k_per_subquery,
                metadata_filter=metadata_filter
            )
            return results, hypothetical_doc
        except Exception as e:
            logger.error(f"⚠️  處理子問題 '{sub_query}' 時出錯: {e}")
            return [], ""
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
        return_sub_queries: bool = False,
        return_hypothetical: bool = False,
        return_abstract_question: bool = False
    ) -> Dict:
        """
        執行三重混合 RAG 檢索
        
        流程：
        1. 拆解成子問題（SubQuery）
        2. 對每個子問題生成假設性文檔並檢索（HyDE）
        3. 直接檢索原始問題（具體事實）
        4. 生成抽象問題並檢索（Step-back，抽象原理）
        5. 合併所有結果並去重
        """
        start_time = time.time()
        
        # 第一步：生成子問題
        logger.info(f"🔍 [SubQuery] 拆解問題: '{question}'")
        sub_queries = self._generate_sub_queries(question)
        logger.info(f"✅ 生成 {len(sub_queries)} 個子問題")
        
        # 第二步：為每個子問題生成假設性文檔並檢索（HyDE）
        logger.info(f"📚 [HyDE] 為每個子問題生成假設性文檔並檢索...")
        subquery_results = []
        hypothetical_docs = {}
        
        if self.enable_parallel and len(sub_queries) > 1:
            with ThreadPoolExecutor(max_workers=min(len(sub_queries), 5)) as executor:
                future_to_query = {
                    executor.submit(self._process_subquery_with_hyde, sq, metadata_filter): sq
                    for sq in sub_queries
                }
                
                for future in as_completed(future_to_query):
                    sub_query = future_to_query[future]
                    try:
                        results, hypo_doc = future.result()
                        hypothetical_docs[sub_query] = hypo_doc
                        subquery_results.extend(results)
                    except Exception as e:
                        logger.error(f"⚠️  處理子問題 '{sub_query}' 時出錯: {e}")
        else:
            for sub_query in sub_queries:
                results, hypo_doc = self._process_subquery_with_hyde(sub_query, metadata_filter)
                hypothetical_docs[sub_query] = hypo_doc
                subquery_results.extend(results)
        
        # 第三步：Step-back 雙軌檢索
        logger.info(f"🔍 [Step-back] 執行雙軌檢索...")
        
        if self.enable_parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                direct_future = executor.submit(
                    self.vector_retriever.retrieve,
                    question, top_k, metadata_filter
                )
                abstract_question = self._generate_step_back_question(question)
                step_back_future = executor.submit(
                    self.vector_retriever.retrieve,
                    abstract_question, top_k, metadata_filter
                )
                
                specific_results = direct_future.result()
                abstract_results = step_back_future.result()
        else:
            specific_results = self.vector_retriever.retrieve(
                query=question,
                top_k=top_k,
                metadata_filter=metadata_filter
            )
            abstract_question = self._generate_step_back_question(question)
            abstract_results = self.vector_retriever.retrieve(
                query=abstract_question,
                top_k=top_k,
                metadata_filter=metadata_filter
            )
        
        # 第四步：合併所有結果並去重
        logger.info(f"🔄 合併並去重所有檢索結果...")
        all_results = subquery_results + specific_results + abstract_results
        unique_docs = {}
        
        for doc in all_results:
            doc_id = get_doc_id(doc)
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc
            else:
                # 保留分數更高的
                existing_score = unique_docs[doc_id].get('score', 0)
                new_score = doc.get('score', 0)
                if new_score > existing_score:
                    unique_docs[doc_id] = doc
        
        # 排序並返回前 top_k
        result_list = list(unique_docs.values())
        result_list.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_results = result_list[:top_k]
        
        elapsed_time = time.time() - start_time
        logger.info(
            f"✅ 三重混合檢索完成（耗時: {elapsed_time:.2f}s）\n"
            f"   子問題檢索: {len(subquery_results)} 個結果\n"
            f"   具體事實: {len(specific_results)} 個結果\n"
            f"   抽象原理: {len(abstract_results)} 個結果\n"
            f"   去重後總計: {len(result_list)} 個，返回前 {len(final_results)} 個"
        )
        
        return {
            "results": final_results,
            "total_docs_found": len(result_list),
            "sub_queries": sub_queries if return_sub_queries else None,
            "hypothetical_documents": hypothetical_docs if return_hypothetical else None,
            "abstract_question": abstract_question if return_abstract_question else None,
            "subquery_results": subquery_results,
            "specific_context": specific_results,
            "abstract_context": abstract_results,
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
        return_sub_queries: bool = False,
        return_hypothetical: bool = False,
        return_abstract_question: bool = False
    ) -> Dict:
        """
        完整的三重混合 RAG 流程：檢索 + 生成答案
        """
        start_time = time.time()
        
        # 檢索
        retrieval_result = self.query(
            question=question,
            top_k=top_k,
            metadata_filter=metadata_filter,
            return_sub_queries=return_sub_queries,
            return_hypothetical=return_hypothetical,
            return_abstract_question=return_abstract_question
        )
        
        if not retrieval_result["results"]:
            return {
                **retrieval_result,
                "answer": "抱歉，未找到相關文檔來回答此問題。",
                "formatted_context": None,
                "answer_time": 0.0,
                "total_time": retrieval_result["elapsed_time"]
            }
        
        # 格式化三類上下文
        subquery_context = formatter.format_context(
            retrieval_result["subquery_results"][:top_k],
            document_type=document_type
        ) if retrieval_result.get("subquery_results") else "未找到相關的子問題檢索結果。"
        
        specific_context = formatter.format_context(
            retrieval_result["specific_context"],
            document_type=document_type
        ) if retrieval_result.get("specific_context") else "未找到相關的具體事實資料。"
        
        abstract_context = formatter.format_context(
            retrieval_result["abstract_context"],
            document_type=document_type
        ) if retrieval_result.get("abstract_context") else "未找到相關的基礎原理資料。"
        
        # 創建融合提示詞（關鍵步驟）
        is_chinese = PromptFormatter.detect_language(question) == "zh"
        
        if is_chinese:
            final_prompt = f"""你是一個資深專家。請結合以下三類資訊來回答使用者的具體問題。

【基礎原理與背景】（來自 Step-back 抽象問題檢索）
{abstract_context}

【具體事實資料】（來自直接問題檢索）
{specific_context}

【子問題相關資料】（來自 SubQuery + HyDE 檢索）
{subquery_context}

使用者問題：{question}

請根據原理推導、結合具體事實，並參考子問題的相關資料，給出一個專業、全面且具備邏輯的回答：
"""
        else:
            final_prompt = f"""You are a senior expert. Please answer the user's specific question by combining the following three types of information.

【Fundamental Principles and Background】(from Step-back abstract question retrieval)
{abstract_context}

【Specific Facts and Data】(from direct question retrieval)
{specific_context}

【Sub-question Related Information】(from SubQuery + HyDE retrieval)
{subquery_context}

User question: {question}

Please provide a professional, comprehensive, and logical answer based on principles, facts, and sub-question related information:
"""
        
        # 生成回答
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
                "subquery": subquery_context,
                "specific": specific_context,
                "abstract": abstract_context
            },
            "answer_time": answer_time,
            "total_time": total_time
        }

