"""
Agent 圖表構建
Multi-Agent：Supervisor 依任務類型派單給學術 / 股票 / 網路專長 researcher，各自使用專屬 tools。
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
from ..tools import get_tools_list_academic, get_tools_list_stock, get_tools_list_web
from ..utils.llm_utils import get_llm
from ..config import MAX_RESEARCH_ITERATIONS, MAX_PLANNER_RETRIES, MAX_RESEARCH_AGENT_RETRIES

# 任務類型關鍵字（與 planner/researcher 一致）
ACADEMIC_KEYWORDS = ["pdf", "知識庫", "理論", "論文", "學術", "方法", "arxiv", "研究"]
STOCK_KEYWORDS = ["股票", "財報", "營運", "公司", "投資", "股價", "市值", "ticker", "msft", "googl", "aapl", "tsla", "nvda"]


def route_after_planner(state: DeepAgentState) -> str:
    """規劃後路由：成功 → supervisor；失敗且未達重試上限 → 回 planner；否則 → planner_fallback"""
    if state.get("planner_succeeded", True):
        return "supervisor"
    retry_count = state.get("planner_retry_count", 0)
    if retry_count < MAX_PLANNER_RETRIES:
        print(f"   🔄 [Planner] 將重試規劃（{retry_count}/{MAX_PLANNER_RETRIES}）")
        return "retry_planner"
    return "planner_fallback"


def _task_type(current_task: str) -> str:
    """依任務內容回傳專長類型：academic_researcher | stock_researcher | web_researcher"""
    t = (current_task or "").lower()
    if any(k in t for k in ACADEMIC_KEYWORDS):
        return "academic_researcher"
    if any(k in t for k in STOCK_KEYWORDS):
        return "stock_researcher"
    return "web_researcher"


def supervisor_node(state: DeepAgentState) -> dict:
    """Supervisor：更新 current_agent，派單由 route_supervisor 決定"""
    tasks = state.get("tasks", [])
    completed = state.get("completed_tasks", [])
    print(f"   🎯 [Supervisor] 任務進度 {len(completed)}/{len(tasks)}")
    return {"current_agent": "supervisor"}


def route_supervisor(state: DeepAgentState) -> str:
    """Supervisor 路由：尚有任務 → 依任務類型派給對應專長；否則 → final_report"""
    tasks = state.get("tasks", [])
    completed = state.get("completed_tasks", [])
    if len(completed) >= len(tasks):
        print("   🎯 [Supervisor] 派單 → final_report")
        return "final_report"
    current_task = tasks[len(completed)]
    specialist = _task_type(current_task)
    print(f"   🎯 [Supervisor] 派單 → {specialist}（任務: {current_task[:40]}…）")
    return specialist


def route_after_specialist(state: DeepAgentState) -> str:
    """專長節點後路由：工具調用 → 對應 tools 節點；失敗 → 重試該專長或 error_finish；否則 → note_taking"""
    if state.get("research_agent_succeeded", True) is False:
        retry_count = state.get("research_agent_retry_count", 0)
        if retry_count < MAX_RESEARCH_AGENT_RETRIES:
            agent = state.get("current_agent", "web_researcher")
            print(f"   🔄 [{agent}] 將重試（{retry_count}/{MAX_RESEARCH_AGENT_RETRIES}）")
            return agent
        return "research_agent_error_finish"
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        agent = state.get("current_agent", "web_researcher")
        tools_node = {"academic_researcher": "tools_academic", "stock_researcher": "tools_stock", "web_researcher": "tools_web"}.get(agent, "tools_web")
        return tools_node
    if state.get("iteration", 0) >= MAX_RESEARCH_ITERATIONS:
        return "note_taking"
    return "note_taking"


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
    構建 Multi-Agent 圖表：Supervisor → 學術/股票/網路專長 researcher → 各自 tools → note_taking → supervisor → final_report
    """
    tools_academic = get_tools_list_academic(rag_retriever)
    tools_stock = get_tools_list_stock()
    tools_web = get_tools_list_web()

    def _specialist_wrapper(tools_list: list, agent_name: str):
        def wrapper(state):
            llm = get_llm()
            llm_with_tools = llm.bind_tools(tools_list)
            try:
                out = research_agent_node(state, llm_with_tools=llm_with_tools)
                return {**out, "research_agent_succeeded": True, "research_agent_retry_count": 0, "current_agent": agent_name}
            except Exception as e:
                retry_count = state.get("research_agent_retry_count", 0)
                print(f"   ⚠️ [{agent_name}] 研究失敗（第 {retry_count + 1} 次）: {e}")
                return {
                    **state,
                    "research_agent_succeeded": False,
                    "research_agent_retry_count": retry_count + 1,
                    "research_agent_error": str(e),
                    "current_agent": agent_name,
                }
        return wrapper

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

    academic_wrapper = _specialist_wrapper(tools_academic, "academic_researcher")
    stock_wrapper = _specialist_wrapper(tools_stock, "stock_researcher")
    web_wrapper = _specialist_wrapper(tools_web, "web_researcher")

    def note_taker_wrapper(state):
        llm = get_llm()
        return note_taking_node(state, llm=llm)

    def reporter_wrapper(state):
        llm = get_llm()
        return final_report_node(state, llm=llm)

    builder = StateGraph(DeepAgentState)

    builder.add_node("planner", planner_wrapper)
    builder.add_node("planner_fallback", planner_fallback_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("academic_researcher", academic_wrapper)
    builder.add_node("stock_researcher", stock_wrapper)
    builder.add_node("web_researcher", web_wrapper)
    builder.add_node("tools_academic", ToolNode(tools_academic))
    builder.add_node("tools_stock", ToolNode(tools_stock))
    builder.add_node("tools_web", ToolNode(tools_web))
    builder.add_node("research_agent_error_finish", research_agent_error_finish_node)
    builder.add_node("note_taking", note_taker_wrapper)
    builder.add_node("final_report", reporter_wrapper)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "supervisor": "supervisor",
            "retry_planner": "planner",
            "planner_fallback": "planner_fallback",
        },
    )
    builder.add_edge("planner_fallback", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "academic_researcher": "academic_researcher",
            "stock_researcher": "stock_researcher",
            "web_researcher": "web_researcher",
            "final_report": "final_report",
        },
    )

    specialist_routes = {
        "tools_academic": "tools_academic",
        "tools_stock": "tools_stock",
        "tools_web": "tools_web",
        "note_taking": "note_taking",
        "research_agent_error_finish": "research_agent_error_finish",
        "academic_researcher": "academic_researcher",
        "stock_researcher": "stock_researcher",
        "web_researcher": "web_researcher",
    }
    for node in ("academic_researcher", "stock_researcher", "web_researcher"):
        builder.add_conditional_edges(node, route_after_specialist, specialist_routes)

    builder.add_edge("research_agent_error_finish", "note_taking")
    builder.add_edge("tools_academic", "academic_researcher")
    builder.add_edge("tools_stock", "stock_researcher")
    builder.add_edge("tools_web", "web_researcher")

    builder.add_edge("note_taking", "supervisor")
    builder.add_edge("final_report", END)

    graph = builder.compile(checkpointer=MemorySaver())
    return graph

