"""
圖片分析 LangGraph 工作流
包含：分析 → 反思 → 改進（可迭代，有最大迭代次數限制）
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Optional

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired
from langchain_core.messages import BaseMessage
import operator

from ..tools.image_analysis_tool import _analyze_image_internal
from ..agents.image_reflection_agent import reflect_on_image_analysis, generate_improved_analysis
from ..config import MAX_REFLECTION_ITERATION, MAX_ANALYZE_RETRIES


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
    # 圖級重試（做法 B）：分析節點失敗時由條件邊決定是否回到 analyze 重試
    analyze_retry_count: NotRequired[int]         # 已重試次數（0 = 第一次）
    analyze_succeeded: NotRequired[bool]          # 本輪分析是否成功
    analyze_error: NotRequired[Optional[str]]      # 最後一次失敗的錯誤訊息


def analyze_image_node(state: ImageAnalysisState) -> dict:
    """
    分析圖片節點
    執行圖片分析；失敗時不拋錯，寫入 state 供路由決定重試或結束。
    """
    question = state.get("question")
    image_path = state["image_path"]
    retry_count = state.get("analyze_retry_count", 0)
    current_iter = state.get("iteration", 0)

    print(f"   🔍 [ImageAnalysis] 第 {current_iter + 1} 輪：正在分析圖片...（重試計數：{retry_count}）")

    try:
        result = _analyze_image_internal(image_path, question=question)
        return {
            **state,
            "analysis_result": result,
            "iteration": current_iter + 1,
            "analyze_succeeded": True,
            "analyze_retry_count": 0,
        }
    except Exception as e:
        print(f"   ⚠️ [ImageAnalysis] 分析失敗（第 {retry_count + 1} 次）：{e}")
        return {
            **state,
            "analyze_succeeded": False,
            "analyze_retry_count": retry_count + 1,
            "analyze_error": str(e),
        }


def route_after_analyze(state: ImageAnalysisState) -> str:
    """
    分析後的路由邏輯（圖級重試）
    成功 → reflection；失敗且未達重試上限 → 回到 analyze；否則 → error_report
    """
    if state.get("analyze_succeeded", True):
        return "reflection"
    retry_count = state.get("analyze_retry_count", 0)
    if retry_count < MAX_ANALYZE_RETRIES:
        print(f"   🔄 [ImageAnalysis] 將重試分析（{retry_count}/{MAX_ANALYZE_RETRIES}）")
        return "retry_analyze"
    return "error_report"


def error_report_node(state: ImageAnalysisState) -> dict:
    """
    分析重試用盡時：將錯誤訊息寫入 analysis_result，方便 UI 顯示。
    """
    retry_count = state.get("analyze_retry_count", 0)
    err = state.get("analyze_error", "未知錯誤")
    msg = f"分析失敗（已重試 {retry_count} 次）：{err}"
    print(f"   ❌ [ImageAnalysis] {msg}")
    return {**state, "analysis_result": msg}


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
    builder.add_node("error_report", error_report_node)
    
    # 定義流程
    builder.add_edge(START, "analyze")
    # 分析後：成功 → reflection；失敗且未達重試上限 → 回 analyze；否則 → error_report
    builder.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {
            "reflection": "reflection",
            "retry_analyze": "analyze",
            "error_report": "error_report",
        },
    )
    builder.add_edge("error_report", END)
    
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
