"""
規劃節點
將複雜問題拆解為具體的研究計畫
"""
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .state import DeepAgentState
from ..utils.llm_utils import get_llm, handle_groq_error
from ..guidelines import get_guideline, get_customer_journey


def get_planner_fallback_tasks(query: str) -> list:
    """重試用盡時回傳的預設任務清單（供圖的 planner_fallback 節點使用）"""
    query_lower = (query or "").lower()
    is_stock_related = any(
        kw in query_lower
        for kw in ["股票", "ticker", "公司", "營運", "財報", "投資", "股價"]
    )
    if is_stock_related:
        return [
            "查詢基礎財務數據和營運狀況",
            "搜尋近期重大新聞和市場動態",
            "查詢 PDF 知識庫中的相關理論（如適用）",
            "分析產業競爭力和未來前景",
        ]
    return [
        "查詢 PDF 知識庫中的相關理論和方法",
        "搜尋網路上相關的學術資料",
        "整理和分析收集到的資訊",
    ]


def planner_node(state: DeepAgentState, llm=None):
    """
    規劃節點：將複雜問題拆解為具體的研究計畫
    
    【重要改進】根據問題類型動態生成任務，避免無關工具調用
    - 學術理論問題 → 專注 PDF 知識庫和網路搜尋
    - 股票相關問題 → 包含股票查詢、新聞、PDF 知識庫
    - 通用問題 → 根據問題內容智能選擇工具
    """
    if llm is None:
        llm = get_llm()
    
    try:
        query = state["query"]
        query_lower = query.lower()
        
        # 【關鍵改進點 1】問題類型檢測：分析問題是否與股票或學術相關
        # 檢測股票相關關鍵字
        stock_keywords = [
            '股票', 'ticker', '公司', '營運', '財報', '投資', '股價', '市值',
            'msft', 'googl', 'aapl', 'tsla', 'nvda', 'amzn', 'meta', 'nflx'  # 常見股票代碼
        ]
        is_stock_related = any(keyword in query_lower for keyword in stock_keywords)
        
        # 檢測學術理論相關關鍵字
        academic_keywords = [
            '論文', '理論', '方法', '研究', '學術', 'tree of thoughts', 
            'chain of thought', 'cot', 'tot', 'methodology', 'framework',
            '概念', '比較', '差異', '分析', 'approach'
        ]
        is_academic_related = any(keyword in query_lower for keyword in academic_keywords)
        
        # 【Parlant 整合】使用指南系統取代硬編碼的提示詞
        task_planning_guideline = get_guideline("research", "task_planning")
        customer_journey = get_customer_journey("research")
        
        # 根據問題類型提供上下文
        query_context = ""
        if is_academic_related and not is_stock_related:
            query_context = "【問題類型】這是一個學術理論問題，請專注於 PDF 知識庫和學術搜尋，不要包含股票查詢任務。"
        elif is_stock_related:
            query_context = "【問題類型】這是一個股票相關問題，應包含財務數據查詢、市場新聞等任務。"
        else:
            query_context = "【問題類型】這是一個通用問題，請根據問題內容智能選擇研究方式。"
        
        # 構建統一的提示詞模板，使用指南系統
        journey_steps = customer_journey.get("steps", [""])[0] if customer_journey else ""
        prompt_template = (
            "你是一個資深研究規劃員。請針對用戶的問題：'{query}'\n"
            "拆解出 3-5 個具體的研究步驟。\n\n"
            f"{query_context}\n\n"
            "【任務規劃指南】\n{task_planning_guideline}\n\n"
            f"【客戶旅程】{journey_steps}\n\n"
            "請只輸出清單，每行一個任務，格式為：數字. 任務描述"
        )
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        try:
            result = chain.invoke({"query": query})
        except Exception as e:
            # 處理 Groq API 錯誤，如果額度用完則切換到本地模型
            fallback_llm = handle_groq_error(e)
            if fallback_llm:
                print("   ⚠️ [Planner] Groq API 額度已用完，已切換到本地 MLX 模型")
                chain = prompt | fallback_llm | StrOutputParser()
                result = chain.invoke({"query": query})
            else:
                raise
        
        # 更健壯的任務解析：提取數字開頭或列表項
        tasks = []
        for line in result.split("\n"):
            line = line.strip()
            if not line:
                continue
            # 移除編號（如 "1. " 或 "- "）
            cleaned = re.sub(r'^[\d\-•]\s*\.?\s*', '', line)
            if cleaned:
                tasks.append(cleaned)
        
        # 【關鍵改進點 3】根據問題類型生成備用任務（避免硬編碼股票任務）
        if not tasks:
            if is_academic_related and not is_stock_related:
                tasks = [
                    "查詢 PDF 知識庫中的相關理論和方法",
                    "搜尋網路上相關的學術資料和論文",
                    "比較和分析不同概念或方法的差異",
                    "總結理論要點、優缺點和應用場景"
                ]
            elif is_stock_related:
                tasks = [
                    "查詢基礎財務數據和營運狀況",
                    "搜尋近期重大新聞和市場動態",
                    "查詢 PDF 知識庫中的相關理論（如適用）",
                    "分析產業競爭力和未來前景"
                ]
            else:
                # 通用問題的預設任務
                tasks = [
                    "搜尋網路上相關資訊",
                    "查詢 PDF 知識庫（如適用）",
                    "整理和分析收集到的資訊"
                ]
        
        print(f"   📝 [Planner] 問題類型檢測：學術={is_academic_related}, 股票={is_stock_related}")
        print(f"   📝 [Planner] 生成計畫: {tasks}")
        return {
            "tasks": tasks, 
            "completed_tasks": [], 
            "research_notes": [],
            "iteration": 0
        }
    except Exception as e:
        # 規劃失敗時拋出，由圖級重試（agent_graph）捕獲並重試或走 planner_fallback
        print(f"   ⚠️ [Planner] 規劃失敗: {e}")
        raise

