"""
報告節點
將所有研究筆記彙整成最終報告
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

from .state import DeepAgentState
from ..utils.llm_utils import get_llm


def final_report_node(state: DeepAgentState, llm=None):
    """
    總結節點：將所有研究筆記彙整成最終報告 (這就是 Deep Agent 的最終產出)
    
    【重要改進】根據問題類型動態調整報告結構，避免要求不相關的內容
    """
    if llm is None:
        llm = get_llm()
    
    try:
        research_notes = state.get("research_notes", [])
        if not research_notes:
            return {"messages": [AIMessage(content="未收集到足夠的研究資料，無法生成報告。")]}
        
        all_notes = "\n\n".join(research_notes)
        completed_tasks = state.get("completed_tasks", [])
        query = state.get("query", "")
        query_lower = query.lower()
        
        # 【關鍵改進點 7】根據問題類型動態生成報告模板
        # 檢測問題類型
        is_stock_related = any(keyword in query_lower for keyword in [
            '股票', 'ticker', '公司', '營運', '財報', '投資', '股價'
        ])
        is_academic_related = any(keyword in query_lower for keyword in [
            '論文', '理論', '方法', '研究', '學術', 'tree of thoughts', 'chain of thought'
        ])
        
        # 根據問題類型選擇報告結構
        if is_academic_related and not is_stock_related:
            # 學術理論問題的報告結構
            report_structure = (
                "請撰寫一份專業的學術分析報告，包含以下部分：\n"
                "1. 執行摘要（Executive Summary）- 概述主要發現和結論\n"
                "2. 理論基礎與概念說明 - 詳細解釋相關理論和方法\n"
                "3. 比較分析 - 深入比較不同概念或方法的差異\n"
                "4. 學術參考與文獻 - 引用 PDF 知識庫和網路搜尋到的相關資料\n"
                "5. 優缺點分析 - 評估不同方法的優缺點\n"
                "6. 應用場景與實務考量 - 說明實際應用情況\n"
                "7. 結論與建議 - 總結要點並提供建議\n\n"
                "【重要】如果研究筆記中沒有財務數據或股票資訊，請不要強行加入這些內容。"
            )
        elif is_stock_related:
            # 股票相關問題的報告結構
            report_structure = (
                "請撰寫一份專業的投資分析報告，包含以下部分：\n"
                "1. 執行摘要（Executive Summary）\n"
                "2. 數據分析與財務狀況\n"
                "3. 近期動態與市場表現\n"
                "4. 理論基礎與學術參考（如適用）\n"
                "5. 產業競爭力分析\n"
                "6. 投資風險評估\n"
                "7. 結論與建議\n"
            )
        else:
            # 通用問題的報告結構
            report_structure = (
                "請撰寫一份專業的分析報告，包含以下部分：\n"
                "1. 執行摘要（Executive Summary）- 概述主要發現\n"
                "2. 核心內容分析 - 根據研究筆記詳細分析問題\n"
                "3. 資料來源與參考 - 說明使用的資料來源（PDF 知識庫、網路搜尋等）\n"
                "4. 深入探討 - 進一步分析相關議題\n"
                "5. 結論與建議 - 總結要點並提供建議\n\n"
                "【重要】請根據實際收集到的資料撰寫報告，不要添加未收集到的資訊。"
            )
        
        prompt = ChatPromptTemplate.from_template(
            "你是一位專業分析師。請根據以下收集到的研究筆記，為用戶問題 '{query}' 撰寫一份結構完整的深度報告。\n\n"
            "已完成的研究任務：\n{completed_tasks}\n\n"
            "研究筆記內容：\n{notes}\n\n"
            "{report_structure}\n\n"
            "請確保報告內容詳實、邏輯清晰，並基於實際收集到的數據和資料。"
            "如果某些部分沒有相關資料，請明確說明，不要編造資訊。"
        )
        chain = prompt | llm | StrOutputParser()
        report = chain.invoke({
            "query": query, 
            "notes": all_notes,
            "completed_tasks": "\n".join([f"- {task}" for task in completed_tasks]),
            "report_structure": report_structure
        })
        print(f"   📊 [FinalReport] 報告生成完成（問題類型：學術={is_academic_related}, 股票={is_stock_related}）")
        return {"messages": [AIMessage(content=report)]}
    except Exception as e:
        print(f"   ⚠️ [FinalReport] 報告生成失敗: {e}")
        return {"messages": [AIMessage(content=f"報告生成過程中發生錯誤: {str(e)}")]}

