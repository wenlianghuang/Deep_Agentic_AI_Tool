"""
規劃節點
將複雜問題拆解為具體的研究計畫
"""
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .state import DeepAgentState
from ..utils.llm_utils import get_llm


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
        
        # 【關鍵改進點 2】根據問題類型動態生成提示詞
        if is_academic_related and not is_stock_related:
            # 純學術理論問題：專注於 PDF 知識庫和學術搜尋
            prompt_template = (
                "你是一個資深研究規劃員。請針對用戶的問題：'{query}'\n"
                "拆解出 3-5 個具體的研究步驟。\n\n"
                "【重要】這是一個學術理論問題，請專注於：\n"
                "1. 查詢 PDF 知識庫中的相關理論、方法和概念\n"
                "2. 搜尋網路上相關的學術資料、論文和最新研究\n"
                "3. 比較和分析不同概念或方法的差異\n"
                "4. 總結理論要點、優缺點和應用場景\n\n"
                "【請勿使用】股票查詢工具，因為問題與股票無關。\n\n"
                "請只輸出清單，每行一個任務，格式為：數字. 任務描述"
            )
        elif is_stock_related:
            # 股票相關問題：包含股票查詢、新聞、PDF 知識庫（如果涉及理論）
            prompt_template = (
                "你是一個資深研究規劃員。請針對用戶的問題：'{query}'\n"
                "拆解出 3-5 個具體的研究步驟，例如：\n"
                "1. 查詢基礎財報數據和營運狀況\n"
                "2. 搜尋近期重大新聞和市場動態\n"
                "3. 查詢 PDF 知識庫中的相關理論或方法（如適用）\n"
                "4. 分析產業競爭力和未來前景\n"
                "請只輸出清單，每行一個任務，格式為：數字. 任務描述"
            )
        else:
            # 通用問題：根據問題內容智能選擇工具
            prompt_template = (
                "你是一個資深研究規劃員。請針對用戶的問題：'{query}'\n"
                "拆解出 3-5 個具體的研究步驟。\n\n"
                "可用的研究方式包括：\n"
                "- 查詢 PDF 知識庫（如果問題涉及學術理論、論文內容或研究方法）\n"
                "- 搜尋網路（獲取最新資訊、新聞或一般知識）\n"
                "- 查詢股票資訊（僅當問題明確涉及股票代碼、公司名稱或財務數據時）\n\n"
                "【重要】請根據問題的實際需求，選擇合適的研究方式。\n"
                "如果問題與股票無關，請不要包含股票查詢任務。\n\n"
                "請只輸出清單，每行一個任務，格式為：數字. 任務描述"
            )
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"query": query})
        
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
        print(f"   ⚠️ [Planner] 規劃失敗: {e}，使用預設計畫")
        # 【關鍵改進點 4】異常處理時也根據問題類型選擇預設任務
        query = state.get("query", "")
        query_lower = query.lower()
        is_stock_related = any(keyword in query_lower for keyword in [
            '股票', 'ticker', '公司', '營運', '財報'
        ])
        
        if is_stock_related:
            default_tasks = [
                "查詢基礎財務數據和營運狀況",
                "搜尋近期重大新聞和市場動態",
                "查詢 PDF 知識庫中的相關理論（如適用）",
                "分析產業競爭力和未來前景"
            ]
        else:
            # 非股票問題的預設任務
            default_tasks = [
                "查詢 PDF 知識庫中的相關理論和方法",
                "搜尋網路上相關的學術資料",
                "整理和分析收集到的資訊"
            ]
        
        return {
            "tasks": default_tasks,
            "completed_tasks": [],
            "research_notes": [],
            "iteration": 0
        }

