"""
圖片分析 LangGraph 工作流
包含：分析 → 反思 → 改進（可迭代，有最大迭代次數限制）
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage
import operator

from ..tools.image_analysis_tool import _analyze_image_internal
from ..agents.image_reflection_agent import reflect_on_image_analysis, generate_improved_analysis
from ..config import MAX_REFLECTION_ITERATION


class ImageAnalysisState(TypedDict):
    """圖片分析狀態"""
    question: Optional[str]          # 用戶的問題（可選）
    image_path: str                  # 圖片路徑
    analysis_result: str             # 當前分析結果
    reflection_result: str           # 反思結果
    improvement_suggestions: str     # 改進建議
    needs_revision: bool             # 是否需要改進
    iteration: int                   # 當前迭代次數
    messages: Annotated[list[BaseMessage], operator.add]  # 消息歷史


def analyze_image_node(state: ImageAnalysisState) -> ImageAnalysisState:
    """
    分析圖片節點
    執行圖片分析
    """
    question = state.get("question")
    image_path = state["image_path"]
    
    print(f"   🔍 [ImageAnalysis] 第 {state.get('iteration', 0) + 1} 輪：正在分析圖片...")
    
    # 執行圖片分析
    result = _analyze_image_internal(image_path, question=question)
    
    return {
        **state,
        "analysis_result": result,
        "iteration": state.get("iteration", 0) + 1
    }


def reflection_node(state: ImageAnalysisState) -> ImageAnalysisState:
    """
    反思節點
    評估分析結果質量並提供改進建議
    """
    question = state.get("question", "")
    image_path = state["image_path"]
    analysis_result = state["analysis_result"]
    iteration = state.get("iteration", 0)
    
    print(f"   🔍 [ImageReflection] 第 {iteration} 輪：正在反思分析結果...")
    
    # 執行反思
    reflection_result, improvement_suggestions, needs_revision = reflect_on_image_analysis(
        question, image_path, analysis_result
    )
    
    return {
        **state,
        "reflection_result": reflection_result,
        "improvement_suggestions": improvement_suggestions,
        "needs_revision": needs_revision
    }


def improvement_node(state: ImageAnalysisState) -> ImageAnalysisState:
    """
    改進節點
    根據改進建議生成改進後的分析
    """
    question = state.get("question", "")
    image_path = state["image_path"]
    original_analysis = state["analysis_result"]
    improvement_suggestions = state["improvement_suggestions"]
    iteration = state.get("iteration", 0)
    
    print(f"   ✨ [ImageImprovement] 第 {iteration} 輪：正在生成改進版本...")
    
    # 生成改進後的分析
    improved_analysis = generate_improved_analysis(
        question, image_path, original_analysis, improvement_suggestions
    )
    
    return {
        **state,
        "analysis_result": improved_analysis,
        "iteration": iteration + 1
    }


def route_after_reflection(state: ImageAnalysisState) -> str:
    """
    反思後的路由邏輯
    決定是否需要改進，或是否達到最大迭代次數
    """
    needs_revision = state.get("needs_revision", False)
    iteration = state.get("iteration", 0)
    
    # 檢查是否達到最大迭代次數（注意：iteration 從 1 開始，所以需要 >= MAX_REFLECTION_ITERATION）
    # 因為第一次分析是 iteration=1，第一次改進後是 iteration=2，所以最多允許 MAX_REFLECTION_ITERATION 次改進
    if iteration >= MAX_REFLECTION_ITERATION + 1:  # +1 因為初始分析也算一次
        print(f"   ⚠️ [ImageAnalysis] 已達到最大反思迭代次數 ({MAX_REFLECTION_ITERATION})，停止改進")
        return "end"
    
    # 如果需要改進且未達到最大迭代次數，進入改進節點
    if needs_revision:
        print(f"   ✅ [ImageAnalysis] 需要改進，進入改進節點（當前迭代：{iteration}/{MAX_REFLECTION_ITERATION + 1}）")
        return "improvement"
    
    # 否則結束
    print(f"   ✅ [ImageAnalysis] 分析質量良好，無需改進")
    return "end"


def build_image_analysis_graph():
    """
    構建圖片分析 LangGraph 圖表
    
    Returns:
        編譯後的圖表
    """
    builder = StateGraph(ImageAnalysisState)
    
    # 添加節點
    builder.add_node("analyze", analyze_image_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("improvement", improvement_node)
    
    # 定義流程
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "reflection")
    
    # 條件路由：反思後決定是否需要改進
    builder.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "improvement": "improvement",
            "end": END
        }
    )
    
    # 改進後回到反思（迭代）
    builder.add_edge("improvement", "reflection")
    
    # 編譯圖表（使用內存檢查點）
    graph = builder.compile(checkpointer=MemorySaver())
    return graph
