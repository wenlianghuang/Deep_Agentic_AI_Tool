"""
Agent 工具定義
包含股票查詢、網路搜尋、PDF 知識庫查詢等工具
"""
import yfinance as yf
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults

from ..utils.llm_utils import get_llm


@tool
def get_company_deep_info(ticker: str) -> str:
    """查詢股票的詳細營運狀況，包括現價、市值、本益比、營收增長等深度數據。"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        summary = (
            f"股票: {info.get('longName')} ({ticker})\n"
            f"現價: {info.get('currentPrice')} {info.get('currency')}\n"
            f"市值: {info.get('marketCap')}\n"
            f"本益比 (PE): {info.get('trailingPE')}\n"
            f"營收增長: {info.get('revenueGrowth')}\n"
            f"業務摘要: {info.get('longBusinessSummary')[:500]}..."
        )
        return summary
    except Exception as e:
        return f"數據查詢失敗: {e}"


@tool
def search_web(query: str) -> str:
    """搜尋網際網路以獲取最新新聞或一般知識。"""
    try:
        tool = TavilySearchResults(k=5)  # 增加搜尋量以獲取深度資訊
        return str(tool.invoke(query))
    except Exception as e:
        return f"搜尋錯誤: {e}"


def query_pdf_knowledge(query: str, rag_retriever=None) -> str:
    """
    查詢 PDF 知識庫（Tree of Thoughts 論文）中的相關資訊。
    當問題涉及論文內容、研究概念、方法論或學術理論時使用此工具。
    """
    if not rag_retriever:
        return "PDF 知識庫未載入，無法查詢。"
    
    try:
        print(f"   🔍 [RAG] 正在查詢 PDF 知識庫: {query}")
        
        # 檢索相關文檔
        docs = rag_retriever.invoke(query)
        
        if not docs:
            return "在 PDF 知識庫中未找到相關資訊。"
        
        # 格式化檢索結果
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 使用 LLM 基於檢索到的內容回答問題
        llm_rag = get_llm()
        prompt = ChatPromptTemplate.from_template(
            "請根據以下從 PDF 知識庫中檢索到的上下文片段，回答使用者的問題。\n\n"
            "上下文：\n{context}\n\n"
            "問題：{question}\n\n"
            "請基於上下文回答，如果上下文中沒有相關資訊，請明確說明。回答請保持簡潔且準確。"
        )
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | llm_rag
            | StrOutputParser()
        )
        result = chain.invoke(query)
        return result
    except Exception as e:
        return f"PDF 知識庫查詢失敗: {e}"


def get_tools_list(rag_retriever=None):
    """
    獲取工具列表
    注意：query_pdf_knowledge 需要 rag_retriever，所以需要動態創建
    """
    # 創建一個帶有 rag_retriever 的 query_pdf_knowledge 工具
    if rag_retriever:
        # 創建一個包裝函數，將 rag_retriever 綁定進去
        def query_pdf_wrapper(query: str) -> str:
            """
            查詢 PDF 知識庫（Tree of Thoughts 論文）中的相關資訊。
            當問題涉及論文內容、研究概念、方法論或學術理論時使用此工具。
            
            Args:
                query: 查詢問題
            
            Returns:
                基於 PDF 知識庫的回答
            """
            return query_pdf_knowledge(query, rag_retriever=rag_retriever)
        
        # 使用 tool 裝飾器創建工具
        pdf_tool = tool(query_pdf_wrapper)
        pdf_tool.name = "query_pdf_knowledge"
        return [get_company_deep_info, search_web, pdf_tool]
    else:
        return [get_company_deep_info, search_web]

