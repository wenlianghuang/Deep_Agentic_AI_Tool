"""
Hybrid Sub-query + HyDE RAG：融合 Sub-query Decomposition 和 HyDE
結合兩種方法的優勢，提升檢索精度
"""
from typing import List, Dict, Optional
from .retrievers.reranker import RAGPipeline
from .retrievers.vector_retriever import VectorRetriever
from .prompt_formatter import PromptFormatter
from .llm_integration import OllamaLLM
from .rag_utils import get_doc_id, generate_sub_queries, generate_hypothetical_document
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class HybridSubqueryHyDERAG:
    """融合 Sub-query Decomposition 和 HyDE 的 RAG 系統"""
    
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
        enable_parallel: bool = True
    ):
        """
        初始化融合 RAG
        
        Args:
            rag_pipeline: RAG 管線實例
            vector_retriever: 向量檢索器
            llm: LLM 實例
            max_sub_queries: 最多生成的子問題數量
            top_k_per_subquery: 每個子問題檢索的結果數量
            hypothetical_length: 假設性文檔目標長度（字符數）
            temperature_subquery: 生成子問題的溫度（較低，更穩定）
            temperature_hyde: 生成假設性文檔的溫度（較高，更多專業術語）
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
        self.enable_parallel = enable_parallel
    
    def _generate_sub_queries(self, question: str) -> List[str]:
        """
        生成子問題（與 SubQueryDecompositionRAG 相同）
        
        Args:
            question: 原始問題
            
        Returns:
            子問題列表
        """
        return generate_sub_queries(
            llm=self.llm,
            question=question,
            max_sub_queries=self.max_sub_queries,
            temperature=self.temperature_subquery
        )
    
    def _generate_hypothetical_document(self, sub_query: str) -> str:
        """
        為子問題生成假設性文檔（與 HyDERAG 相同）
        
        Args:
            sub_query: 子問題
            
        Returns:
            假設性文檔文本
        """
        return generate_hypothetical_document(
            llm=self.llm,
            question=sub_query,
            hypothetical_length=self.hypothetical_length,
            temperature=self.temperature_hyde,
            enable_logging=False  # 使用 debug 級別日誌
        )
    
    def _process_subquery_with_hyde(
        self, 
        sub_query: str, 
        metadata_filter: Optional[Dict] = None
    ) -> tuple:
        """
        處理單個子問題：生成假設性文檔並檢索
        
        Args:
            sub_query: 子問題
            metadata_filter: 可選的 metadata 過濾條件
            
        Returns:
            (檢索結果列表, 假設性文檔)
        """
        try:
            # 生成假設性文檔
            hypothetical_doc = self._generate_hypothetical_document(sub_query)
            
            # 使用假設性文檔檢索
            results = self.vector_retriever.retrieve(
                query=hypothetical_doc,  # 使用假設性文檔而不是子問題
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
        return_hypothetical: bool = False
    ) -> Dict:
        """
        執行融合 RAG 檢索（不生成答案）
        
        Args:
            question: 原始問題
            top_k: 返回前 k 個結果
            metadata_filter: 可選的 metadata 過濾條件
            return_sub_queries: 是否返回子問題列表
            return_hypothetical: 是否返回假設性文檔字典（子問題 -> 假設性文檔）
            
        Returns:
            包含檢索結果和統計資訊的字典
        """
        start_time = time.time()
        
        # 第一步：生成子問題
        logger.info(f"🔍 拆解問題: '{question}'")
        sub_queries = self._generate_sub_queries(question)
        logger.info(f"✅ 生成 {len(sub_queries)} 個子問題")
        
        # 第二步：為每個子問題生成假設性文檔並檢索
        logger.info(f"📚 為每個子問題生成假設性文檔並檢索...")
        unique_docs = {}
        hypothetical_docs = {}
        
        if self.enable_parallel and len(sub_queries) > 1:
            # 並行處理
            logger.info(f"🔄 並行處理 {len(sub_queries)} 個子問題...")
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
                        
                        logger.debug(f"✅ 子問題 '{sub_query}' 找到 {len(results)} 個結果")
                        
                        for doc in results:
                            doc_id = get_doc_id(doc)
                            if doc_id not in unique_docs:
                                unique_docs[doc_id] = doc
                            else:
                                # 保留分數更高的
                                existing_score = unique_docs[doc_id].get('score', 0)
                                new_score = doc.get('score', 0)
                                if new_score > existing_score:
                                    unique_docs[doc_id] = doc
                    except Exception as e:
                        logger.error(f"⚠️  處理子問題 '{sub_query}' 時出錯: {e}")
        else:
            # 串行處理
            logger.info(f"🔄 串行處理 {len(sub_queries)} 個子問題...")
            for sub_query in sub_queries:
                results, hypo_doc = self._process_subquery_with_hyde(sub_query, metadata_filter)
                hypothetical_docs[sub_query] = hypo_doc
                
                logger.debug(f"✅ 子問題 '{sub_query}' 找到 {len(results)} 個結果")
                
                for doc in results:
                    doc_id = get_doc_id(doc)
                    if doc_id not in unique_docs:
                        unique_docs[doc_id] = doc
                    else:
                        existing_score = unique_docs[doc_id].get('score', 0)
                        new_score = doc.get('score', 0)
                        if new_score > existing_score:
                            unique_docs[doc_id] = doc
        
        # 第三步：排序並返回前 top_k
        result_list = list(unique_docs.values())
        result_list.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_results = result_list[:top_k]
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ 找到 {len(final_results)} 個唯一文檔（去重後，總共 {len(result_list)} 個）")
        
        return {
            "results": final_results,
            "total_docs_found": len(result_list),
            "sub_queries": sub_queries if return_sub_queries else None,
            "hypothetical_documents": hypothetical_docs if return_hypothetical else None,
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
        return_hypothetical: bool = False
    ) -> Dict:
        """
        完整的融合 RAG 流程：檢索 + 生成答案
        
        Args:
            question: 原始問題
            formatter: Prompt 格式化器
            top_k: 用於生成答案的文檔數量
            metadata_filter: 可選的 metadata 過濾條件
            document_type: 文檔類型 ("paper", "cv", "general")
            return_sub_queries: 是否返回子問題列表
            return_hypothetical: 是否返回假設性文檔字典
            
        Returns:
            包含檢索結果、生成的答案和統計資訊的字典
        """
        start_time = time.time()
        
        # 檢索
        retrieval_result = self.query(
            question=question,
            top_k=top_k,
            metadata_filter=metadata_filter,
            return_sub_queries=return_sub_queries,
            return_hypothetical=return_hypothetical
        )
        
        if not retrieval_result["results"]:
            return {
                **retrieval_result,
                "answer": "抱歉，未找到相關文檔來回答此問題。",
                "formatted_context": None,
                "answer_time": 0.0,
                "total_time": retrieval_result["elapsed_time"]
            }
        
        # 格式化上下文
        formatted_context = formatter.format_context(
            retrieval_result["results"],
            document_type=document_type
        )
        
        # 創建 prompt（使用原始問題）
        prompt = formatter.create_prompt(
            question,
            formatted_context,
            document_type=document_type
        )
        
        # 生成回答
        logger.info("🤖 生成回答中...")
        answer_start = time.time()
        try:
            answer = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
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
            "formatted_context": formatted_context,
            "answer_time": answer_time,
            "total_time": total_time
        }

