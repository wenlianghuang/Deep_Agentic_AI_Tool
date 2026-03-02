"""
自適應 RAG 方法選擇器
根據查詢和檔案特徵自動選擇最佳的 RAG 方法
"""
from typing import Dict, List, Optional
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class RAGMethod(Enum):
    """可用的 RAG 方法"""
    BASIC = "basic"  # 基礎 RAG（當前使用的）
    SUBQUERY = "subquery"  # 子查詢分解
    HYDE = "hyde"  # 假設文檔嵌入
    STEP_BACK = "step_back"  # 後退推理
    HYBRID_SUBQUERY_HYDE = "hybrid_subquery_hyde"  # 混合子查詢+HyDE
    TRIPLE_HYBRID = "triple_hybrid"  # 三重混合


class QueryComplexity(Enum):
    """查詢複雜度"""
    SIMPLE = "simple"  # 簡單查詢（單問題，短句）
    MODERATE = "moderate"  # 中等複雜度（包含多個概念）
    COMPLEX = "complex"  # 複雜查詢（多部分問題，需要分解）
    VERY_COMPLEX = "very_complex"  # 非常複雜（多個相關問題）


class QueryType(Enum):
    """查詢類型"""
    FACTUAL = "factual"  # 事實性查詢（「什麼是X」）
    CONCEPTUAL = "conceptual"  # 概念性查詢（「如何理解X」）
    COMPARATIVE = "comparative"  # 比較性查詢（「X和Y的區別」）
    PRINCIPLE = "principle"  # 原理性查詢（「X的工作原理」）
    MULTI_ASPECT = "multi_aspect"  # 多面向查詢（包含多個問題）


class AdaptiveRAGSelector:
    """
    自適應 RAG 方法選擇器
    
    根據以下特徵選擇最佳 RAG 方法：
    1. 查詢複雜度
    2. 查詢類型
    3. 檔案數量和類型
    4. 文檔複雜度
    """
    
    def __init__(self):
        """初始化選擇器"""
        pass
    
    def analyze_query(self, query: str) -> Dict:
        """
        分析查詢特徵
        
        Args:
            query: 用戶查詢問題
        
        Returns:
            包含查詢特徵的字典
        """
        query_lower = query.lower()
        query_len = len(query)
        word_count = len(query.split())
        
        # 檢測查詢複雜度
        complexity = self._detect_complexity(query, word_count)
        
        # 檢測查詢類型
        query_type = self._detect_query_type(query, query_lower)
        
        # 檢測是否包含多個問題
        question_count = query.count('?') + query.count('？')
        has_multiple_questions = question_count > 1
        
        # 檢測是否包含比較性詞彙（含簡繁）
        comparison_keywords = ['vs', 'versus', 'difference', '区别', '區別', '比较', '比較', 'compare', '对比', '對比', '和', 'and', '与', '與']
        is_comparative = any(kw in query_lower for kw in comparison_keywords)
        
        # 檢測是否包含專業術語
        technical_indicators = [
            '原理', 'mechanism', 'algorithm', 'architecture', 'model', 'system',
            '原理', '機制', '算法', '架構', '模型', '系統', '方法', 'method',
            '如何工作', 'how does', 'how do', 'work', 'function'
        ]
        has_technical_terms = any(ind in query_lower for ind in technical_indicators)
        
        # 檢測是否包含「為什麼」、「如何」等需要解釋的詞彙（含簡繁）
        explanation_keywords = ['为什么', '為什麼', 'why', '如何', 'how', 'explain', '解释', '解釋', '说明', '說明']
        needs_explanation = any(kw in query_lower for kw in explanation_keywords)
        
        return {
            'complexity': complexity,
            'type': query_type,
            'word_count': word_count,
            'length': query_len,
            'has_multiple_questions': has_multiple_questions,
            'is_comparative': is_comparative,
            'has_technical_terms': has_technical_terms,
            'needs_explanation': needs_explanation,
            'question_count': question_count
        }
    
    def _detect_complexity(self, query: str, word_count: int) -> QueryComplexity:
        """檢測查詢複雜度"""
        # 簡單查詢：短句，單問題
        if word_count <= 10 and query.count('?') + query.count('？') <= 1:
            return QueryComplexity.SIMPLE
        
        # 中等複雜度：中等長度，可能包含多個概念
        if word_count <= 25:
            return QueryComplexity.MODERATE
        
        # 複雜查詢：長句，多個問題或概念
        if word_count <= 50:
            return QueryComplexity.COMPLEX
        
        # 非常複雜：很長，多個問題
        return QueryComplexity.VERY_COMPLEX
    
    def _detect_query_type(self, query: str, query_lower: str) -> QueryType:
        """檢測查詢類型"""
        # 比較性查詢（含簡繁）
        if any(kw in query_lower for kw in ['vs', 'versus', 'difference', '区别', '區別', '比较', '比較', 'compare', '对比', '對比', '和', 'and', '与', '與']):
            return QueryType.COMPARATIVE
        
        # 原理性查詢
        if any(kw in query_lower for kw in ['原理', 'principle', 'how does', 'how do', 'mechanism', '如何工作', '工作原理']):
            return QueryType.PRINCIPLE
        
        # 概念性查詢（含簡繁）
        if any(kw in query_lower for kw in ['什么是', '什麼是', 'what is', '理解', 'understand', 'explain', '解释', '解釋']):
            return QueryType.CONCEPTUAL
        
        # 多面向查詢
        if query.count('?') + query.count('？') > 1:
            return QueryType.MULTI_ASPECT
        
        # 預設：事實性查詢
        return QueryType.FACTUAL
    
    def analyze_files(self, file_paths: List[str], documents: Optional[List[Dict]] = None) -> Dict:
        """
        分析檔案特徵
        
        Args:
            file_paths: 檔案路徑列表
            documents: 文檔列表（可選，如果已處理）
        
        Returns:
            包含檔案特徵的字典
        """
        file_count = len(file_paths)
        
        # 檢測檔案類型
        file_types = []
        for path in file_paths:
            if path.endswith('.pdf'):
                file_types.append('pdf')
            elif path.endswith(('.docx', '.doc')):
                file_types.append('docx')
            else:
                file_types.append('txt')
        
        # 分析文檔複雜度（如果有文檔）
        total_chunks = len(documents) if documents else 0
        avg_chunk_size = 0
        if documents:
            chunk_sizes = [len(doc.get('content', '')) for doc in documents]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        # 檢測是否為學術論文（基於檔案名稱或內容）
        is_academic = any('paper' in path.lower() or 'arxiv' in path.lower() or 
                         path.endswith('.pdf') for path in file_paths)
        
        return {
            'file_count': file_count,
            'file_types': file_types,
            'total_chunks': total_chunks,
            'avg_chunk_size': avg_chunk_size,
            'is_academic': is_academic,
            'is_single_file': file_count == 1,
            'is_multi_file': file_count > 1
        }
    
    def select_best_method(
        self, 
        query_features: Dict, 
        file_features: Dict,
        enable_advanced: bool = True
    ) -> RAGMethod:
        """
        根據特徵選擇最佳 RAG 方法
        
        Args:
            query_features: 查詢特徵（來自 analyze_query）
            file_features: 檔案特徵（來自 analyze_files）
            enable_advanced: 是否啟用進階方法（如果 False，只使用基礎方法）
        
        Returns:
            選擇的 RAG 方法
        """
        if not enable_advanced:
            return RAGMethod.BASIC
        
        complexity = query_features['complexity']
        query_type = query_features['type']
        has_multiple_questions = query_features['has_multiple_questions']
        is_comparative = query_features['is_comparative']
        has_technical_terms = query_features['has_technical_terms']
        needs_explanation = query_features['needs_explanation']
        file_count = file_features['file_count']
        is_multi_file = file_features['is_multi_file']
        
        # 決策樹
        
        # 1. 非常複雜的查詢 + 多檔案 → Triple Hybrid（最強）
        if complexity == QueryComplexity.VERY_COMPLEX and is_multi_file:
            return RAGMethod.TRIPLE_HYBRID
        
        # 2. 複雜查詢 + 多問題 → SubQuery 或 Hybrid Subquery+HyDE
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            if has_multiple_questions or query_type == QueryType.MULTI_ASPECT:
                if is_multi_file:
                    return RAGMethod.HYBRID_SUBQUERY_HYDE
                else:
                    return RAGMethod.SUBQUERY
        
        # 3. 原理性查詢 → Step-back（需要背景知識）
        if query_type == QueryType.PRINCIPLE:
            if complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]:
                return RAGMethod.STEP_BACK
        
        # 4. 專業術語查詢 → HyDE（生成假設文檔）
        if has_technical_terms and complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]:
            return RAGMethod.HYDE
        
        # 5. 比較性查詢 + 多檔案 → SubQuery（需要分別檢索）
        if is_comparative and is_multi_file:
            return RAGMethod.SUBQUERY
        
        # 6. 中等複雜度 + 多檔案 → Hybrid Subquery+HyDE
        if complexity == QueryComplexity.MODERATE and is_multi_file:
            return RAGMethod.HYBRID_SUBQUERY_HYDE
        
        # 7. 簡單查詢 → 基礎 RAG 或 HyDE
        if complexity == QueryComplexity.SIMPLE:
            if has_technical_terms:
                return RAGMethod.HYDE
            else:
                return RAGMethod.BASIC
        
        # 8. 需要解釋的查詢 → Step-back（提供背景知識）
        if needs_explanation and complexity == QueryComplexity.MODERATE:
            return RAGMethod.STEP_BACK
        
        # 9. 預設：中等複雜度使用 Step-back
        if complexity == QueryComplexity.MODERATE:
            return RAGMethod.STEP_BACK
        
        # 10. 預設：複雜查詢使用 SubQuery
        return RAGMethod.SUBQUERY
    
    def get_method_reason(self, method: RAGMethod, query_features: Dict, file_features: Dict) -> str:
        """
        取得選擇該方法的理由
        
        Args:
            method: 選擇的 RAG 方法
            query_features: 查詢特徵
            file_features: 檔案特徵
        
        Returns:
            選擇理由的字串
        """
        complexity = query_features['complexity'].value
        query_type = query_features['type'].value
        file_count = file_features['file_count']
        
        reasons = {
            RAGMethod.BASIC: f"簡單查詢（{complexity}），使用基礎 RAG 方法即可",
            RAGMethod.SUBQUERY: f"查詢包含多個方面（{query_features['question_count']}個問題，{complexity}），使用子查詢分解以全面檢索",
            RAGMethod.HYDE: f"查詢包含專業術語（{complexity}），使用假設文檔嵌入以改善語義檢索",
            RAGMethod.STEP_BACK: f"原理性查詢（{query_type}，{complexity}），使用後退推理取得背景知識和原理",
            RAGMethod.HYBRID_SUBQUERY_HYDE: f"複雜查詢（{complexity}）+ {file_count}個檔案，使用混合子查詢+HyDE方法以全面檢索",
            RAGMethod.TRIPLE_HYBRID: f"非常複雜的查詢（{complexity}）+ {file_count}個檔案，使用三重混合方法（SubQuery+HyDE+Step-back）以獲得最佳效果"
        }
        return reasons.get(method, f"使用 {method.value} 方法")
