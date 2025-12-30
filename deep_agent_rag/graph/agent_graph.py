"""
Agent 圖表構建
定義節點之間的連接和路由邏輯
"""
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from ..agents.state import DeepAgentState
from ..agents.planner import planner_node
from ..agents.researcher import research_agent_node
from ..agents.note_taker import note_taking_node
from ..agents.reporter import final_report_node
from ..tools import get_tools_list
from ..utils.llm_utils import get_llm
from ..config import MAX_RESEARCH_ITERATIONS


def route_after_agent(state: DeepAgentState):
    """決定是要呼叫工具，還是進入筆記階段"""
    last_msg = state["messages"][-1]
    # 檢查是否有工具調用
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tools"
    # 檢查是否達到最大迭代次數
    if state.get("iteration", 0) >= MAX_RESEARCH_ITERATIONS:
        return "note_taking"
    return "note_taking"


def route_after_note(state: DeepAgentState):
    """決定是否還有下一個任務要跑"""
    if len(state["completed_tasks"]) < len(state["tasks"]):
        return "research_agent"
    return "final_report"


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
    
    # 創建節點函數的包裝器，傳入必要的依賴
    # 注意：為了支持動態切換 LLM，我們在每個節點中動態獲取 LLM
    def planner_wrapper(state):
        llm = get_llm()
        return planner_node(state, llm=llm)
    
    def researcher_wrapper(state):
        # 動態獲取 LLM 並綁定工具，以支持運行時切換
        llm = get_llm()
        llm_with_tools = llm.bind_tools(tools_list)
        return research_agent_node(state, llm_with_tools=llm_with_tools)
    
    def note_taker_wrapper(state):
        llm = get_llm()
        return note_taking_node(state, llm=llm)
    
    def reporter_wrapper(state):
        llm = get_llm()
        return final_report_node(state, llm=llm)
    
    # 構建圖表
    builder = StateGraph(DeepAgentState)
    
    builder.add_node("planner", planner_wrapper)
    builder.add_node("research_agent", researcher_wrapper)
    builder.add_node("tools", ToolNode(tools_list))
    builder.add_node("note_taking", note_taker_wrapper)
    builder.add_node("final_report", reporter_wrapper)
    
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "research_agent")
    
    builder.add_conditional_edges(
        "research_agent",
        route_after_agent,
        {"tools": "tools", "note_taking": "note_taking"}
    )
    builder.add_edge("tools", "research_agent")
    
    builder.add_conditional_edges(
        "note_taking",
        route_after_note,
        {"research_agent": "research_agent", "final_report": "final_report"}
    )
    builder.add_edge("final_report", END)
    
    graph = builder.compile(checkpointer=MemorySaver())
    return graph

