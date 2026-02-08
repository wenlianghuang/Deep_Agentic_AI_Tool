"""
研究節點
根據任務清單使用工具進行深度研究
"""
from langchain_core.messages import SystemMessage, AIMessage

from .state import DeepAgentState
from ..utils.llm_utils import get_llm, handle_groq_error
from ..config import MAX_ITERATIONS
from ..guidelines import get_guideline


def research_agent_node(state: DeepAgentState, llm_with_tools=None):
    """
    執行節點：根據目前的任務清單，使用工具進行深度研究
    
    【重要改進】根據任務內容智能指導工具選擇，避免調用無關工具
    """
    # 檢查迭代次數，防止無限循環
    current_iteration = state.get("iteration", 0)
    if current_iteration >= MAX_ITERATIONS:
        return {"messages": [AIMessage(content="已達最大迭代次數，停止研究。")]}
    
    current_task_idx = len(state.get("completed_tasks", []))
    tasks = state.get("tasks", [])
    
    if current_task_idx >= len(tasks):
        return {"messages": [AIMessage(content="所有研究任務已完成。")]}
    
    current_task = tasks[current_task_idx]
    print(f"   🕵️ [Researcher] 正在執行任務 {current_task_idx + 1}/{len(tasks)}: {current_task}")
    
    try:
        # 【Parlant 整合】使用指南系統取代硬編碼邏輯
        # 獲取工具選擇指南和研究行為指南
        tool_selection_guideline = get_guideline("research", "tool_selection")
        research_behavior_guideline = get_guideline("research", "research_behavior")
        
        # 根據任務內容提供上下文提示（保留關鍵字檢測作為輔助判斷）
        task_lower = current_task.lower()
        task_context = ""
        
        if any(keyword in task_lower for keyword in ["pdf", "知識庫", "理論", "論文", "學術", "方法"]):
            task_context = "【任務上下文】此任務涉及學術理論或 PDF 內容，應優先使用 PDF 知識庫工具。"
        elif any(keyword in task_lower for keyword in ["股票", "財報", "營運", "公司", "投資", "股價", "市值"]):
            task_context = "【任務上下文】此任務涉及股票或公司財務，應使用股票查詢工具。"
        elif any(keyword in task_lower for keyword in ["搜尋", "網路", "新聞", "動態", "資訊", "資料"]):
            task_context = "【任務上下文】此任務需要最新資訊，應使用網路搜尋工具。"
        else:
            task_context = "【任務上下文】請根據任務內容和指南選擇最合適的工具。"
        
        # 構建系統提示，使用指南系統
        system_msg = SystemMessage(content=(
            f"你是一位深度研究員。當前目標任務是：{current_task}\n\n"
            f"{task_context}\n\n"
            f"【工具選擇指南】\n{tool_selection_guideline}\n\n"
            f"【研究行為指南】\n{research_behavior_guideline}\n\n"
            f"可用的工具詳細說明：\n"
            f"- query_pdf_knowledge(query: str): 查詢 PDF 知識庫，用於學術理論、論文內容、研究方法等\n"
            f"- extract_keywords_from_pdf(query: str): 從 PDF 知識庫中提取學術關鍵字，用於 arXiv 搜尋\n"
            f"- search_arxiv_papers(keywords_json: str, max_results: int): 使用 arXiv API 搜尋相關論文\n"
            f"- add_arxiv_papers_to_rag(arxiv_ids_json: str): 下載 arXiv 論文並添加到 RAG 系統中，擴展知識庫\n"
            f"- search_web(query: str): 網路搜尋，用於獲取最新資訊、新聞、一般知識等\n"
            f"- get_company_deep_info(ticker: str): 股票資訊查詢，僅用於查詢股票代碼對應的公司財務數據\n\n"
            f"【學術研究工作流程建議】\n"
            f"當進行深度學術研究時，建議按以下順序使用工具：\n"
            f"1. 先使用 query_pdf_knowledge 查詢本地 PDF 知識庫\n"
            f"2. 如果本地資料不足，使用 extract_keywords_from_pdf 提取關鍵字\n"
            f"3. 使用 search_arxiv_papers 搜尋相關論文\n"
            f"4. 使用 add_arxiv_papers_to_rag 將相關論文添加到知識庫\n"
            f"5. 再次使用 query_pdf_knowledge 查詢擴展後的知識庫，獲得更全面的答案\n"
        ))
        
        # 構建上下文：包含原始問題、已完成任務和研究筆記
        context_messages = [system_msg]
        
        # 如果有研究筆記，加入上下文
        if state.get("research_notes"):
            notes_summary = "\n".join(state["research_notes"][-3:])  # 只取最近3條筆記
            context_messages.append(SystemMessage(
                content=f"先前的研究發現：\n{notes_summary}"
            ))
        
        # 加入原始問題，幫助 LLM 理解整體目標
        original_query = state.get("query", "")
        if original_query:
            context_messages.append(SystemMessage(
                content=f"用戶的原始問題：{original_query}"
            ))
        
        # 加入歷史消息
        context_messages.extend(state["messages"][-10:])  # 只保留最近10條消息避免上下文過長
        
        if llm_with_tools is None:
            from ..utils.llm_utils import get_llm
            from ..tools import get_tools_list
            llm = get_llm()
            tools_list = get_tools_list()
            llm_with_tools = llm.bind_tools(tools_list)
        
        try:
            response = llm_with_tools.invoke(context_messages)
        except Exception as e:
            # 處理 Groq API 錯誤，如果額度用完則切換到本地模型
            fallback_llm = handle_groq_error(e)
            if fallback_llm:
                print("   ⚠️ [Researcher] Groq API 額度已用完，已切換到本地 MLX 模型")
                from ..tools import get_tools_list
                tools_list = get_tools_list()
                fallback_llm_with_tools = fallback_llm.bind_tools(tools_list)
                response = fallback_llm_with_tools.invoke(context_messages)
            else:
                raise
        return {
            "messages": [response],
            "iteration": current_iteration + 1
        }
    except Exception as e:
        # 研究失敗時拋出，由圖級重試（agent_graph）捕獲並重試或走 research_agent_error_finish
        print(f"   ⚠️ [Researcher] 研究失敗: {e}")
        raise

