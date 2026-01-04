"""
自适应 RAG 方法选择器
根据查询和文件特征自动选择最佳的 RAG 方法
"""
from typing import Dict, List, Optional
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class RAGMethod(Enum):
    """可用的 RAG 方法"""
    BASIC = "basic"  # 基础 RAG（当前使用的）
    SUBQUERY = "subquery"  # 子查询分解
    HYDE = "hyde"  # 假设文档嵌入
    STEP_BACK = "step_back"  # 后退推理
    HYBRID_SUBQUERY_HYDE = "hybrid_subquery_hyde"  # 混合子查询+HyDE
    TRIPLE_HYBRID = "triple_hybrid"  # 三重混合


class QueryComplexity(Enum):
    """查询复杂度"""
    SIMPLE = "simple"  # 简单查询（单问题，短句）
    MODERATE = "moderate"  # 中等复杂度（包含多个概念）
    COMPLEX = "complex"  # 复杂查询（多部分问题，需要分解）
    VERY_COMPLEX = "very_complex"  # 非常复杂（多个相关问题）


class QueryType(Enum):
    """查询类型"""
    FACTUAL = "factual"  # 事实性查询（"什么是X"）
    CONCEPTUAL = "conceptual"  # 概念性查询（"如何理解X"）
    COMPARATIVE = "comparative"  # 比较性查询（"X和Y的区别"）
    PRINCIPLE = "principle"  # 原理性查询（"X的工作原理"）
    MULTI_ASPECT = "multi_aspect"  # 多面向查询（包含多个问题）


class AdaptiveRAGSelector:
    """
    自适应 RAG 方法选择器
    
    根据以下特征选择最佳 RAG 方法：
    1. 查询复杂度
    2. 查询类型
    3. 文件数量和类型
    4. 文档复杂度
    """
    
    def __init__(self):
        """初始化选择器"""
        pass
    
    def analyze_query(self, query: str) -> Dict:
        """
        分析查询特征
        
        Args:
            query: 用户查询问题
        
        Returns:
            包含查询特征的字典
        """
        query_lower = query.lower()
        query_len = len(query)
        word_count = len(query.split())
        
        # 检测查询复杂度
        complexity = self._detect_complexity(query, word_count)
        
        # 检测查询类型
        query_type = self._detect_query_type(query, query_lower)
        
        # 检测是否包含多个问题
        question_count = query.count('?') + query.count('？')
        has_multiple_questions = question_count > 1
        
        # 检测是否包含比较性词汇
        comparison_keywords = ['vs', 'versus', 'difference', '区别', '比较', 'compare', '对比', '和', 'and', '与']
        is_comparative = any(kw in query_lower for kw in comparison_keywords)
        
        # 检测是否包含专业术语
        technical_indicators = [
            '原理', 'mechanism', 'algorithm', 'architecture', 'model', 'system',
            '原理', '机制', '算法', '架构', '模型', '系统', '方法', 'method',
            '如何工作', 'how does', 'how do', 'work', 'function'
        ]
        has_technical_terms = any(ind in query_lower for ind in technical_indicators)
        
        # 检测是否包含"为什么"、"如何"等需要解释的词汇
        explanation_keywords = ['为什么', 'why', '如何', 'how', 'explain', '解释', '说明']
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
        """检测查询复杂度"""
        # 简单查询：短句，单问题
        if word_count <= 10 and query.count('?') + query.count('？') <= 1:
            return QueryComplexity.SIMPLE
        
        # 中等复杂度：中等长度，可能包含多个概念
        if word_count <= 25:
            return QueryComplexity.MODERATE
        
        # 复杂查询：长句，多个问题或概念
        if word_count <= 50:
            return QueryComplexity.COMPLEX
        
        # 非常复杂：很长，多个问题
        return QueryComplexity.VERY_COMPLEX
    
    def _detect_query_type(self, query: str, query_lower: str) -> QueryType:
        """检测查询类型"""
        # 比较性查询
        if any(kw in query_lower for kw in ['vs', 'versus', 'difference', '区别', '比较', 'compare', '对比', '和', 'and', '与']):
            return QueryType.COMPARATIVE
        
        # 原理性查询
        if any(kw in query_lower for kw in ['原理', 'principle', 'how does', 'how do', 'mechanism', '如何工作', '工作原理']):
            return QueryType.PRINCIPLE
        
        # 概念性查询
        if any(kw in query_lower for kw in ['什么是', 'what is', '理解', 'understand', 'explain', '解释']):
            return QueryType.CONCEPTUAL
        
        # 多面向查询
        if query.count('?') + query.count('？') > 1:
            return QueryType.MULTI_ASPECT
        
        # 默认：事实性查询
        return QueryType.FACTUAL
    
    def analyze_files(self, file_paths: List[str], documents: Optional[List[Dict]] = None) -> Dict:
        """
        分析文件特征
        
        Args:
            file_paths: 文件路径列表
            documents: 文档列表（可选，如果已处理）
        
        Returns:
            包含文件特征的字典
        """
        file_count = len(file_paths)
        
        # 检测文件类型
        file_types = []
        for path in file_paths:
            if path.endswith('.pdf'):
                file_types.append('pdf')
            elif path.endswith(('.docx', '.doc')):
                file_types.append('docx')
            else:
                file_types.append('txt')
        
        # 分析文档复杂度（如果有文档）
        total_chunks = len(documents) if documents else 0
        avg_chunk_size = 0
        if documents:
            chunk_sizes = [len(doc.get('content', '')) for doc in documents]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        # 检测是否为学术论文（基于文件名或内容）
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
        根据特征选择最佳 RAG 方法
        
        Args:
            query_features: 查询特征（来自 analyze_query）
            file_features: 文件特征（来自 analyze_files）
            enable_advanced: 是否启用高级方法（如果 False，只使用基础方法）
        
        Returns:
            选择的 RAG 方法
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
        
        # 决策树
        
        # 1. 非常复杂的查询 + 多文件 → Triple Hybrid（最强）
        if complexity == QueryComplexity.VERY_COMPLEX and is_multi_file:
            return RAGMethod.TRIPLE_HYBRID
        
        # 2. 复杂查询 + 多问题 → SubQuery 或 Hybrid Subquery+HyDE
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            if has_multiple_questions or query_type == QueryType.MULTI_ASPECT:
                if is_multi_file:
                    return RAGMethod.HYBRID_SUBQUERY_HYDE
                else:
                    return RAGMethod.SUBQUERY
        
        # 3. 原理性查询 → Step-back（需要背景知识）
        if query_type == QueryType.PRINCIPLE:
            if complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]:
                return RAGMethod.STEP_BACK
        
        # 4. 专业术语查询 → HyDE（生成假设文档）
        if has_technical_terms and complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]:
            return RAGMethod.HYDE
        
        # 5. 比较性查询 + 多文件 → SubQuery（需要分别检索）
        if is_comparative and is_multi_file:
            return RAGMethod.SUBQUERY
        
        # 6. 中等复杂度 + 多文件 → Hybrid Subquery+HyDE
        if complexity == QueryComplexity.MODERATE and is_multi_file:
            return RAGMethod.HYBRID_SUBQUERY_HYDE
        
        # 7. 简单查询 → 基础 RAG 或 HyDE
        if complexity == QueryComplexity.SIMPLE:
            if has_technical_terms:
                return RAGMethod.HYDE
            else:
                return RAGMethod.BASIC
        
        # 8. 需要解释的查询 → Step-back（提供背景知识）
        if needs_explanation and complexity == QueryComplexity.MODERATE:
            return RAGMethod.STEP_BACK
        
        # 9. 默认：中等复杂度使用 Step-back
        if complexity == QueryComplexity.MODERATE:
            return RAGMethod.STEP_BACK
        
        # 10. 默认：复杂查询使用 SubQuery
        return RAGMethod.SUBQUERY
    
    def get_method_reason(self, method: RAGMethod, query_features: Dict, file_features: Dict) -> str:
        """
        获取选择该方法的理由
        
        Args:
            method: 选择的 RAG 方法
            query_features: 查询特征
            file_features: 文件特征
        
        Returns:
            选择理由的字符串
        """
        complexity = query_features['complexity'].value
        query_type = query_features['type'].value
        file_count = file_features['file_count']
        
        reasons = {
            RAGMethod.BASIC: f"简单查询（{complexity}），使用基础 RAG 方法即可",
            RAGMethod.SUBQUERY: f"查询包含多个方面（{query_features['question_count']}个问题，{complexity}），使用子查询分解以全面检索",
            RAGMethod.HYDE: f"查询包含专业术语（{complexity}），使用假设文档嵌入以改善语义检索",
            RAGMethod.STEP_BACK: f"原理性查询（{query_type}，{complexity}），使用后退推理获取背景知识和原理",
            RAGMethod.HYBRID_SUBQUERY_HYDE: f"复杂查询（{complexity}）+ {file_count}个文件，使用混合子查询+HyDE方法以全面检索",
            RAGMethod.TRIPLE_HYBRID: f"非常复杂的查询（{complexity}）+ {file_count}个文件，使用三重混合方法（SubQuery+HyDE+Step-back）以获得最佳效果"
        }
        return reasons.get(method, f"使用 {method.value} 方法")

