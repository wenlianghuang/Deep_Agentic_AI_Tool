"""
Sub-query Decomposition RAG：將複雜問題拆解成子問題後檢索
"""
from typing import List, Dict, Optional
from .retrievers.reranker import RAGPipeline
from .prompt_formatter import PromptFormatter
from .llm_integration import OllamaLLM
from .rag_utils import get_doc_id, generate_sub_queries
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class SubQueryDecompositionRAG:
    """使用子問題拆解的 RAG 系統"""
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        llm: OllamaLLM,
        max_sub_queries: int = 3,
        top_k_per_subquery: int = 5,
        enable_parallel: bool = True
    ):
        """
        初始化 Sub-query Decomposition RAG
        
        Args:
            rag_pipeline: 現有的 RAG 管線實例
            llm: LLM 實例（用於生成子問題）
            max_sub_queries: 最多生成的子問題數量
            top_k_per_subquery: 每個子問題檢索的結果數量
            enable_parallel: 是否並行處理子查詢
        """
        self.rag_pipeline = rag_pipeline
        self.llm = llm
        self.max_sub_queries = max_sub_queries
        self.top_k_per_subquery = top_k_per_subquery
        self.enable_parallel = enable_parallel
    
    def _generate_sub_queries(self, question: str) -> List[str]:
        """
        將原始問題拆解成子問題
        
        Args:
            question: 原始問題
            
        Returns:
            子問題列表
        """
        return generate_sub_queries(
            llm=self.llm,
            question=question,
            max_sub_queries=self.max_sub_queries,
            temperature=0.3
        )
    
    def _retrieve_for_subquery(
        self, 
        sub_query: str, 
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        針對單個子問題進行檢索
        
        Args:
            sub_query: 子問題
            metadata_filter: 可選的 metadata 過濾條件
            
        Returns:
            檢索結果列表
        """
        try:
            results = self.rag_pipeline.query(
                text=sub_query,
                top_k=self.top_k_per_subquery,
                metadata_filter=metadata_filter,
                enable_rerank=True
            )
            return results
        except Exception as e:
            logger.error(f"⚠️  檢索子問題 '{sub_query}' 時出錯: {e}")
            return []
    
    def _get_unique_documents(
        self, 
        sub_queries: List[str],
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        針對所有子問題進行檢索，並移除重複的檔案
        
        Args:
            sub_queries: 子問題列表
            metadata_filter: 可選的 metadata 過濾條件
            
        Returns:
            去重後的文檔列表
        """
        unique_docs = {}
        
        if self.enable_parallel and len(sub_queries) > 1:
            # 並行處理子查詢
            logger.info(f"🔄 並行處理 {len(sub_queries)} 個子查詢...")
            with ThreadPoolExecutor(max_workers=min(len(sub_queries), 5)) as executor:
                future_to_query = {
                    executor.submit(self._retrieve_for_subquery, q, metadata_filter): q
                    for q in sub_queries
                }
                
                for future in as_completed(future_to_query):
                    sub_query = future_to_query[future]
                    try:
                        docs = future.result()
                        logger.debug(f"✅ 子問題 '{sub_query}' 找到 {len(docs)} 個結果")
                        for doc in docs:
                            doc_id = get_doc_id(doc)
                            if doc_id not in unique_docs:
                                unique_docs[doc_id] = doc
                            else:
                                # 如果已存在，保留分數更高的
                                existing_score = unique_docs[doc_id].get(
                                    'rerank_score', 
                                    unique_docs[doc_id].get('hybrid_score', 0)
                                )
                                new_score = doc.get(
                                    'rerank_score',
                                    doc.get('hybrid_score', 0)
                                )
                                if new_score > existing_score:
                                    unique_docs[doc_id] = doc
                    except Exception as e:
                        logger.error(f"⚠️  處理子問題 '{sub_query}' 時出錯: {e}")
        else:
            # 串行處理
            logger.info(f"🔄 串行處理 {len(sub_queries)} 個子查詢...")
            for sub_query in sub_queries:
                docs = self._retrieve_for_subquery(sub_query, metadata_filter)
                logger.debug(f"✅ 子問題 '{sub_query}' 找到 {len(docs)} 個結果")
                for doc in docs:
                    doc_id = get_doc_id(doc)
                    if doc_id not in unique_docs:
                        unique_docs[doc_id] = doc
                    else:
                        # 保留分數更高的
                        existing_score = unique_docs[doc_id].get(
                            'rerank_score',
                            unique_docs[doc_id].get('hybrid_score', 0)
                        )
                        new_score = doc.get(
                            'rerank_score',
                            doc.get('hybrid_score', 0)
                        )
                        if new_score > existing_score:
                            unique_docs[doc_id] = doc
        
        # 按分數排序
        result_list = list(unique_docs.values())
        result_list.sort(
            key=lambda x: x.get('rerank_score', x.get('hybrid_score', 0)),
            reverse=True
        )
        
        return result_list
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
        return_sub_queries: bool = False
    ) -> Dict:
        """
        執行 Sub-query Decomposition RAG 查詢
        
        Args:
            question: 原始問題
            top_k: 返回前 k 個結果
            metadata_filter: 可選的 metadata 過濾條件
            return_sub_queries: 是否在結果中包含子問題列表
            
        Returns:
            包含檢索結果和統計資訊的字典
        """
        start_time = time.time()
        
        # 第一步：產生子問題
        logger.info(f"🔍 拆解問題: '{question}'")
        sub_queries = self._generate_sub_queries(question)
        logger.info(f"✅ 生成 {len(sub_queries)} 個子問題:")
        for i, sq in enumerate(sub_queries, 1):
            logger.info(f"   {i}. {sq}")
        
        # 第二步：檢索並去重
        logger.info(f"📚 檢索相關文檔...")
        docs = self._get_unique_documents(sub_queries, metadata_filter)
        logger.info(f"✅ 找到 {len(docs)} 個唯一文檔（去重後）")
        
        # 第三步：返回前 top_k 個結果
        final_results = docs[:top_k]
        
        elapsed_time = time.time() - start_time
        
        result = {
            "results": final_results,
            "total_docs_found": len(docs),
            "sub_queries": sub_queries if return_sub_queries else None,
            "elapsed_time": elapsed_time
        }
        
        return result
    
    def generate_answer(
        self,
        question: str,
        formatter: PromptFormatter,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
        document_type: str = "general",
        return_sub_queries: bool = False
    ) -> Dict:
        """
        完整的 Sub-query Decomposition RAG 流程：檢索 + 生成答案
        
        Args:
            question: 原始問題
            formatter: Prompt 格式化器
            top_k: 返回前 k 個結果用於生成答案
            metadata_filter: 可選的 metadata 過濾條件
            document_type: 文檔類型 ("paper", "cv", "general")
            return_sub_queries: 是否在結果中包含子問題列表
            
        Returns:
            包含檢索結果、生成的答案和統計資訊的字典
        """
        # 檢索
        retrieval_result = self.query(
            question=question,
            top_k=top_k,
            metadata_filter=metadata_filter,
            return_sub_queries=return_sub_queries
        )
        
        if not retrieval_result["results"]:
            return {
                **retrieval_result,
                "answer": "抱歉，未找到相關文檔來回答此問題。",
                "formatted_context": None
            }
        
        # 格式化上下文
        formatted_context = formatter.format_context(
            retrieval_result["results"],
            document_type=document_type
        )
        
        # 創建 prompt
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
        
        return {
            **retrieval_result,
            "answer": answer,
            "formatted_context": formatted_context,
            "answer_time": answer_time,
            "total_time": retrieval_result["elapsed_time"] + answer_time
        }

