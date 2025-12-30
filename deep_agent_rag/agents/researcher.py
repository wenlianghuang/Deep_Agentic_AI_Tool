"""
研究節點
根據任務清單使用工具進行深度研究
"""
from langchain_core.messages import SystemMessage, AIMessage

from .state import DeepAgentState
from ..utils.llm_utils import get_llm, handle_groq_error
from ..config import MAX_ITERATIONS


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
        # 【關鍵改進點 5】根據任務內容判斷應該使用哪些工具，提供明確指導
        task_lower = current_task.lower()
        tool_guidance = ""
        
        # 檢測任務類型並提供對應的工具使用建議
        if any(keyword in task_lower for keyword in ["pdf", "知識庫", "理論", "論文", "學術", "方法"]):
            tool_guidance = (
                "\n【工具選擇指導】此任務應優先使用 PDF 知識庫查詢工具（query_pdf_knowledge）。"
                "\n如果任務涉及學術理論、論文內容或研究方法，請使用 query_pdf_knowledge。"
                "\n請勿使用股票查詢工具（get_company_deep_info），除非任務明確要求。"
            )
        elif any(keyword in task_lower for keyword in ["股票", "財報", "營運", "公司", "投資", "股價", "市值"]):
            tool_guidance = (
                "\n【工具選擇指導】此任務應使用股票資訊查詢工具（get_company_deep_info）。"
                "\n請從任務描述中提取股票代碼（如 MSFT, GOOGL），然後使用 get_company_deep_info 查詢。"
            )
        elif any(keyword in task_lower for keyword in ["搜尋", "網路", "新聞", "動態", "資訊", "資料"]):
            tool_guidance = (
                "\n【工具選擇指導】此任務應使用網路搜尋工具（search_web）。"
                "\n請使用 search_web 獲取最新的網路資訊、新聞或一般知識。"
            )
        else:
            # 通用指導：根據任務內容選擇合適的工具
            tool_guidance = (
                "\n【工具選擇指導】請根據任務內容選擇最合適的工具："
                "\n- 如果任務涉及學術理論、論文或 PDF 內容 → 使用 query_pdf_knowledge"
                "\n- 如果任務涉及股票、公司財務 → 使用 get_company_deep_info"
                "\n- 如果任務需要最新資訊、新聞 → 使用 search_web"
                "\n請只使用與任務相關的工具，不要使用不相關的工具。"
            )
        
        # 【關鍵改進點 6】構建更智能的系統提示，明確工具使用規則
        system_msg = SystemMessage(content=(
            f"你是一位深度研究員。當前目標任務是：{current_task}\n"
            f"{tool_guidance}\n"
            f"\n可用的工具詳細說明：\n"
            f"- query_pdf_knowledge(query: str): 查詢 PDF 知識庫，用於學術理論、論文內容、研究方法等\n"
            f"- search_web(query: str): 網路搜尋，用於獲取最新資訊、新聞、一般知識等\n"
            f"- get_company_deep_info(ticker: str): 股票資訊查詢，僅用於查詢股票代碼對應的公司財務數據\n"
            f"\n【重要原則】"
            f"\n1. 請根據任務內容選擇最合適的工具"
            f"\n2. 如果任務與股票無關，請勿使用 get_company_deep_info"
            f"\n3. 如果任務涉及學術理論，請優先使用 query_pdf_knowledge"
            f"\n4. 你可以進行多輪工具調用來深入挖掘資訊"
            f"\n5. 當你認為資訊已經足夠時，請總結你的發現並回覆"
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
        print(f"   ⚠️ [Researcher] 研究失敗: {e}")
        error_msg = AIMessage(content=f"研究過程中發生錯誤: {str(e)}")
        return {
            "messages": [error_msg],
            "iteration": current_iteration + 1
        }

