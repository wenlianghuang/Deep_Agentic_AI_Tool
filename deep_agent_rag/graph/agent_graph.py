"""
Agent 圖表構建
定義節點之間的連接和路由邏輯；含圖級重試（planner / research_agent 失敗時重試）
"""
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

from ..agents.state import DeepAgentState
from ..agents.planner import planner_node, get_planner_fallback_tasks
from ..agents.researcher import research_agent_node
from ..agents.note_taker import note_taking_node
from ..agents.reporter import final_report_node
from ..tools import get_tools_list
from ..utils.llm_utils import get_llm
from ..config import MAX_RESEARCH_ITERATIONS, MAX_PLANNER_RETRIES, MAX_RESEARCH_AGENT_RETRIES


def route_after_planner(state: DeepAgentState) -> str:
    """規劃後路由：成功 → research_agent；失敗且未達重試上限 → 回 planner；否則 → planner_fallback"""
    if state.get("planner_succeeded", True):
        return "research_agent"
    retry_count = state.get("planner_retry_count", 0)
    if retry_count < MAX_PLANNER_RETRIES:
        print(f"   🔄 [Planner] 將重試規劃（{retry_count}/{MAX_PLANNER_RETRIES}）")
        return "retry_planner"
    return "planner_fallback"


def route_after_agent(state: DeepAgentState) -> str:
    """決定是要呼叫工具、進入筆記階段、重試 research_agent、或走錯誤結束節點"""
    # 研究節點失敗且需重試或走錯誤結束
    if state.get("research_agent_succeeded", True) is False:
        retry_count = state.get("research_agent_retry_count", 0)
        if retry_count < MAX_RESEARCH_AGENT_RETRIES:
            print(f"   🔄 [Researcher] 將重試研究（{retry_count}/{MAX_RESEARCH_AGENT_RETRIES}）")
            return "research_agent"
        return "research_agent_error_finish"
    # 原有邏輯：工具調用 vs 筆記
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    if state.get("iteration", 0) >= MAX_RESEARCH_ITERATIONS:
        return "note_taking"
    return "note_taking"


def route_after_note(state: DeepAgentState) -> str:
    """決定是否還有下一個任務要跑"""
    if len(state["completed_tasks"]) < len(state["tasks"]):
        return "research_agent"
    return "final_report"


def planner_fallback_node(state: DeepAgentState) -> dict:
    """規劃重試用盡時：寫入預設任務，繼續流程"""
    query = state.get("query", "")
    tasks = get_planner_fallback_tasks(query)
    print(f"   ⚠️ [Planner] 重試用盡，使用預設計畫: {tasks}")
    return {
        "tasks": tasks,
        "completed_tasks": [],
        "research_notes": [],
        "iteration": 0,
    }


def research_agent_error_finish_node(state: DeepAgentState) -> dict:
    """研究重試用盡時：將錯誤寫入 messages 後進入筆記階段"""
    retry_count = state.get("research_agent_retry_count", 0)
    err = state.get("research_agent_error", "未知錯誤")
    msg = AIMessage(
        content=f"研究過程中發生錯誤（已重試 {retry_count} 次）: {err}"
    )
    print(f"   ❌ [Researcher] {msg.content}")
    return {"messages": [msg]}


def build_agent_graph(rag_retriever=None):
    """
    構建 Deep Agent 圖表
    
    Args:
        rag_retriever: RAG 檢索器（可選）
    
    Returns:
        編譯後的圖表
    """
    # 獲取工具列表
    tools_list = get_tools_list(rag_retriever)
    
    # 創建節點函數的包裝器，傳入必要的依賴；圖級重試：失敗時寫入 state 供路由重試
    def planner_wrapper(state):
        llm = get_llm()
        try:
            out = planner_node(state, llm=llm)
            return {**out, "planner_succeeded": True, "planner_retry_count": 0}
        except Exception as e:
            retry_count = state.get("planner_retry_count", 0)
            print(f"   ⚠️ [Planner] 規劃失敗（第 {retry_count + 1} 次）: {e}")
            return {
                **state,
                "planner_succeeded": False,
                "planner_retry_count": retry_count + 1,
                "planner_error": str(e),
            }

    def researcher_wrapper(state):
        llm = get_llm()
        llm_with_tools = llm.bind_tools(tools_list)
        try:
            out = research_agent_node(state, llm_with_tools=llm_with_tools)
            return {**out, "research_agent_succeeded": True, "research_agent_retry_count": 0}
        except Exception as e:
            retry_count = state.get("research_agent_retry_count", 0)
            print(f"   ⚠️ [Researcher] 研究失敗（第 {retry_count + 1} 次）: {e}")
            return {
                **state,
                "research_agent_succeeded": False,
                "research_agent_retry_count": retry_count + 1,
                "research_agent_error": str(e),
            }
    
    def note_taker_wrapper(state):
        llm = get_llm()
        return note_taking_node(state, llm=llm)
    
    def reporter_wrapper(state):
        llm = get_llm()
        return final_report_node(state, llm=llm)
    
    # 構建圖表
    builder = StateGraph(DeepAgentState)
    
    builder.add_node("planner", planner_wrapper)
    builder.add_node("planner_fallback", planner_fallback_node)
    builder.add_node("research_agent", researcher_wrapper)
    builder.add_node("research_agent_error_finish", research_agent_error_finish_node)
    builder.add_node("tools", ToolNode(tools_list))
    builder.add_node("note_taking", note_taker_wrapper)
    builder.add_node("final_report", reporter_wrapper)
    
    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "research_agent": "research_agent",
            "retry_planner": "planner",
            "planner_fallback": "planner_fallback",
        },
    )
    builder.add_edge("planner_fallback", "research_agent")
    
    builder.add_conditional_edges(
        "research_agent",
        route_after_agent,
        {
            "tools": "tools",
            "note_taking": "note_taking",
            "research_agent_error_finish": "research_agent_error_finish",
        },
    )
    builder.add_edge("research_agent_error_finish", "note_taking")
    builder.add_edge("tools", "research_agent")
    
    builder.add_conditional_edges(
        "note_taking",
        route_after_note,
        {"research_agent": "research_agent", "final_report": "final_report"}
    )
    builder.add_edge("final_report", END)
    
    graph = builder.compile(checkpointer=MemorySaver())
    return graph

