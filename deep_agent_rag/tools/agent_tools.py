"""
Agent 工具定義
包含股票查詢、網路搜尋、PDF 知識庫查詢、arXiv 論文搜尋等工具
"""
import yfinance as yf
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


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


def get_product_names_from_files(data_dir: str = "data") -> list:
    """
    從 data 文件夾中的 PDF 文件名動態提取產品名稱。
    
    Args:
        data_dir: PDF 文件所在的文件夾路徑
        
    Returns:
        產品名稱列表（包含帶破折號和空格的版本）
    """
    import os
    
    product_names = []
    
    try:
        # 獲取絕對路徑
        if not os.path.isabs(data_dir):
            # 假設相對於專案根目錄
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, data_dir)
        
        if not os.path.exists(data_dir):
            print(f"   ⚠️ 資料夾不存在: {data_dir}")
            return []
        
        # 掃描 PDF 文件
        for filename in os.listdir(data_dir):
            if filename.endswith('.pdf'):
                # 從文件名中提取產品名稱（例如 "Lumina-Grid 智慧能源控制器.pdf" -> "Lumina-Grid"）
                # 提取第一個空格前的部分作為產品名稱
                product_name = filename.split()[0] if ' ' in filename else filename.replace('.pdf', '')
                
                # 移除可能的擴展名（如果沒有空格的話）
                product_name = product_name.replace('.pdf', '')
                
                if product_name:
                    # 添加原始名稱（帶破折號）
                    product_names.append(product_name)
                    
                    # 添加空格版本（將破折號替換為空格）
                    if '-' in product_name:
                        product_names.append(product_name.replace('-', ' '))
        
        if product_names:
            print(f"   ✅ 從 {len(set(product_names))//2} 個 PDF 文件中提取產品名稱: {', '.join(set([p for p in product_names if '-' in p]))}")
        
    except Exception as e:
        print(f"   ⚠️ 讀取產品名稱失敗: {e}")
        # 返回空列表，讓調用者決定是否使用備用列表
    
    return product_names


def query_pdf_knowledge(query: str, rag_retriever=None) -> str:
    """
    查詢 PDF 知識庫中的相關資訊。
    當問題涉及論文內容、研究概念、方法論或學術理論時使用此工具。
    
    現在使用 Private File RAG 系統，支持多文件、進階 RAG 方法。
    
    這個函數會智能擴展查詢：
    1. 如果查詢中沒有明確的產品名稱，會先進行初步檢索
    2. 從初步檢索結果和查詢本身推斷可能的產品名稱
    3. 使用擴展後的查詢進行完整檢索
    """
    if not rag_retriever:
        return "PDF 知識庫未載入，無法查詢。"
    
    try:
        print(f"   🔍 [RAG] 正在查詢 PDF 知識庫: {query}")
        
        # 檢查是否是 Private File RAG 實例
        from ..rag.private_file_rag import PrivateFileRAG
        from ..utils.llm_utils import get_llm
        from langchain_core.messages import HumanMessage
        
        if not isinstance(rag_retriever, PrivateFileRAG):
            return "PDF 知識庫格式不正確，請重新初始化。"
        
        # 已知的產品名稱列表 - 從 data 文件夾動態載入
        product_names = get_product_names_from_files()
        
        # 如果動態載入失敗，使用備用列表
        if not product_names:
            print("   ⚠️ 無法從文件載入產品名稱，使用備用列表")
            product_names = [
                "Lumina-Grid", "Gaia-7", "Nebula-X", "Deep-Void", "Synapse-Link",
                "Lumina Grid", "Gaia 7", "Nebula X", "Deep Void", "Synapse Link"
            ]
        
        # 檢查查詢中是否已經包含產品名稱
        query_lower = query.lower()
        has_product_in_query = any(
            name.lower() in query_lower for name in product_names
        )
        
        # 如果查詢中沒有產品名稱，嘗試智能擴展
        expanded_query = query
        if not has_product_in_query:
            print(f"   🔍 [查詢擴展] 查詢中沒有明確的產品名稱，嘗試智能擴展...")
            
            # 策略 1: 先進行一次初步檢索，查看 PDF 內容
            # 使用較大的 top_k 來獲取更多候選結果
            preliminary_result = rag_retriever.query(
                query=query,
                top_k=10,  # 獲取更多結果以便分析
                use_llm=False  # 只檢索，不生成回答
            )
            
            if preliminary_result.get("success") and preliminary_result.get("results"):
                # 從初步檢索結果中提取文本
                contexts = []
                for res in preliminary_result.get("results", [])[:5]:  # 只取前5個結果
                    contexts.append(res.get("content", ""))
                
                combined_context = "\n\n".join(contexts)
                # 限制長度避免過長
                context_snippet = combined_context[:2000]
                
                # 使用 LLM 從查詢和檢索結果中推斷產品名稱
                try:
                    llm = get_llm()
                    infer_prompt = f"""根據以下查詢和 PDF 內容片段，推斷用戶可能想查詢哪個產品的信息。

查詢：{query}

PDF 內容片段：
{context_snippet}

已知產品列表：{', '.join(product_names)}

請根據查詢內容和 PDF 片段推斷最可能的產品名稱。
如果能夠確定產品名稱，請只返回產品名稱（例如："Lumina-Grid"）。
如果無法確定，請返回 "無"。
只返回產品名稱或"無"，不要其他解釋。"""
                    
                    messages = [HumanMessage(content=infer_prompt)]
                    response = llm.invoke(messages)
                    inferred_product = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                    
                    # 檢查推斷的產品是否在已知列表中
                    if inferred_product and inferred_product.lower() not in ["無", "无", "none", "no", ""]:
                        # 找到匹配的產品名稱
                        matched_product = None
                        for name in product_names:
                            if name.lower() in inferred_product.lower() or inferred_product.lower() in name.lower():
                                matched_product = name
                                break
                        
                        if matched_product:
                            expanded_query = f"{matched_product} {query}"
                            print(f"   ✅ [查詢擴展] 從 PDF 內容推斷產品名稱 '{matched_product}'，擴展查詢為：{expanded_query}")
                        else:
                            # 如果推斷的產品不在列表中，但看起來像產品名稱，也可以嘗試
                            # 檢查是否包含常見的產品名稱模式
                            for name in product_names:
                                if any(word.lower() in inferred_product.lower() for word in name.split() if len(word) > 2):
                                    expanded_query = f"{name} {query}"
                                    print(f"   ✅ [查詢擴展] 從推斷結果 '{inferred_product}' 匹配到產品 '{name}'，擴展查詢為：{expanded_query}")
                                    break
                except Exception as e:
                    print(f"   ⚠️ [查詢擴展] LLM 推斷產品名稱失敗: {e}，使用原始查詢")
            
            # 策略 2: 如果初步檢索沒有幫助，嘗試直接從查詢推斷
            if expanded_query == query:
                try:
                    llm = get_llm()
                    # 檢查查詢中是否包含版本號、技術規格等關鍵詞
                    version_keywords = ["版本", "version", "v1", "v2", "v3", "v1.", "v2.", "v3.", "v4", "v5"]
                    spec_keywords = ["時脈", "頻率", "clock", "GHz", "核心", "晶片", "chip", "core", "能源", "轉換率"]
                    
                    has_version_or_spec = any(
                        keyword in query_lower for keyword in version_keywords + spec_keywords
                    )
                    
                    if has_version_or_spec:
                        infer_prompt = f"""根據以下查詢，推斷用戶可能想查詢哪個產品的信息。

查詢：{query}

已知產品列表：{', '.join(product_names)}

請根據查詢內容推斷最可能的產品名稱。如果查詢中沒有明確的產品信息，請返回 "無"。
只返回產品名稱或"無"，不要其他解釋。"""
                        
                        messages = [HumanMessage(content=infer_prompt)]
                        response = llm.invoke(messages)
                        inferred_product = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                        
                        if inferred_product and inferred_product.lower() not in ["無", "无", "none", "no", ""]:
                            # 找到匹配的產品名稱
                            matched_product = None
                            for name in product_names:
                                if name.lower() in inferred_product.lower() or inferred_product.lower() in name.lower():
                                    matched_product = name
                                    break
                            
                            if matched_product:
                                expanded_query = f"{matched_product} {query}"
                                print(f"   ✅ [查詢擴展] 從查詢推斷產品名稱 '{matched_product}'，擴展查詢為：{expanded_query}")
                except Exception as e:
                    print(f"   ⚠️ [查詢擴展] 從查詢推斷產品名稱失敗: {e}，使用原始查詢")
        
        # 使用擴展後的查詢進行完整檢索
        result = rag_retriever.query(
            query=expanded_query,  # 使用擴展後的查詢
            top_k=5,  # 檢索前 5 個相關片段
            use_llm=True  # 使用 LLM 生成回答
        )
        
        if result.get("success"):
            answer = result.get("answer", "")
            if answer:
                # 可選：添加使用的 RAG 方法信息（用於調試）
                rag_method = result.get("rag_method", "basic")
                if rag_method != "basic":
                    print(f"   📊 [RAG] 使用 {rag_method} 方法")
                return answer
            else:
                return "在 PDF 知識庫中未找到相關資訊。"
        else:
            error = result.get("error", "未知錯誤")
            return f"PDF 知識庫查詢失敗: {error}"
            
    except Exception as e:
        return f"PDF 知識庫查詢失敗: {e}"


def extract_keywords_from_pdf(query: str, rag_retriever=None) -> str:
    """
    從 PDF 知識庫中提取學術關鍵字，用於 arXiv 搜尋。
    當需要查找相關學術論文時使用此工具。
    
    Args:
        query: 查詢問題
        rag_retriever: RAG 檢索器（PrivateFileRAG 實例）
    
    Returns:
        提取的關鍵字列表（JSON 格式）
    """
    if not rag_retriever:
        return "PDF 知識庫未載入，無法提取關鍵字。"
    
    try:
        from ..rag.private_file_rag import PrivateFileRAG
        from ..utils.llm_utils import get_llm
        from langchain_core.messages import HumanMessage
        import json
        
        if not isinstance(rag_retriever, PrivateFileRAG):
            return "PDF 知識庫格式不正確。"
        
        # 先查詢 PDF 獲取相關內容
        result = rag_retriever.query(
            query=query,
            top_k=10,
            use_llm=False  # 只檢索，不生成回答
        )
        
        if not result.get("success") or not result.get("results"):
            return "在 PDF 中未找到相關內容，無法提取關鍵字。"
        
        # 從檢索結果中提取文本
        contexts = []
        for res in result.get("results", [])[:3]:
            contexts.append(res.get("content", ""))
        
        combined_context = "\n\n".join(contexts)
        
        # 使用 LLM 提取關鍵字
        llm = get_llm()
        prompt = f"""從以下 PDF 內容中提取學術關鍵字，這些關鍵字將用於在 arXiv 上搜尋相關論文。

PDF 內容：
{combined_context}

原始查詢：{query}

請提取：
1. 核心學術概念和術語（英文）
2. 研究方法和技術名稱
3. 相關領域關鍵詞

返回格式：JSON 陣列，例如：["keyword1", "keyword2", "keyword3"]
只返回 JSON 陣列，不要其他解釋。"""
        
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        keywords_text = response.content if hasattr(response, 'content') else str(response)
        
        # 嘗試解析 JSON
        try:
            # 清理響應，提取 JSON 部分
            keywords_text = keywords_text.strip()
            if keywords_text.startswith("```"):
                # 移除程式碼塊標記
                keywords_text = keywords_text.split("```")[1]
                if keywords_text.startswith("json"):
                    keywords_text = keywords_text[4:]
            keywords_text = keywords_text.strip()
            
            keywords = json.loads(keywords_text)
            if isinstance(keywords, list) and keywords:
                return json.dumps(keywords, ensure_ascii=False)
            else:
                return "未能提取有效關鍵字。"
        except json.JSONDecodeError:
            # 如果 JSON 解析失敗，嘗試提取引號中的內容
            import re
            keywords = re.findall(r'"([^"]+)"', keywords_text)
            if keywords:
                return json.dumps(keywords, ensure_ascii=False)
            return f"關鍵字提取失敗，LLM 返回：{keywords_text}"
            
    except Exception as e:
        return f"提取關鍵字失敗: {e}"


def search_arxiv_papers(keywords_json: str, max_results: int = 5) -> str:
    """
    使用 arXiv API 搜尋相關論文。
    
    Args:
        keywords_json: 關鍵字 JSON 陣列字串，例如：'["machine learning", "neural networks"]'
        max_results: 最大返回結果數
    
    Returns:
        論文列表的格式化字串
    """
    try:
        import json
        import arxiv
        from src.document_processor import DocumentProcessor
        
        # 解析關鍵字
        keywords = json.loads(keywords_json)
        if not isinstance(keywords, list) or not keywords:
            return "無效的關鍵字格式。"
        
        # 構建 arXiv 搜尋查詢
        # 使用 OR 連接多個關鍵字
        query = " OR ".join([f'all:"{kw}"' for kw in keywords[:5]])  # 限制最多 5 個關鍵字
        
        print(f"   📚 [arXiv] 正在搜尋論文，關鍵字: {', '.join(keywords[:5])}")
        
        # 搜尋論文
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for paper in search.results():
            papers.append({
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary[:500],  # 限制摘要長度
                "published": str(paper.published),
                "arxiv_id": paper.entry_id.split('/')[-1],
                "arxiv_url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "categories": [str(cat) for cat in paper.categories],
            })
        
        if not papers:
            return "未找到相關論文。"
        
        # 返回格式化的論文列表
        result_text = f"找到 {len(papers)} 篇相關論文：\n\n"
        for i, paper in enumerate(papers, 1):
            result_text += f"{i}. {paper['title']}\n"
            result_text += f"   arXiv ID: {paper['arxiv_id']}\n"
            result_text += f"   作者: {', '.join(paper['authors'][:3])}"
            if len(paper['authors']) > 3:
                result_text += f" 等 {len(paper['authors'])} 位作者"
            result_text += "\n"
            result_text += f"   摘要: {paper['summary']}...\n"
            result_text += f"   連結: {paper['pdf_url']}\n"
            result_text += f"   分類: {', '.join(paper['categories'][:3])}\n\n"
        
        print(f"   ✅ [arXiv] 成功找到 {len(papers)} 篇論文")
        return result_text
        
    except Exception as e:
        return f"arXiv 搜尋失敗: {e}"


def add_arxiv_papers_to_rag(arxiv_ids_json: str, rag_retriever=None) -> str:
    """
    下載 arXiv 論文並添加到 RAG 系統中。
    
    Args:
        arxiv_ids_json: arXiv ID 的 JSON 陣列，例如：'["2305.10601", "2301.12345"]'
        rag_retriever: RAG 檢索器（PrivateFileRAG 實例）
    
    Returns:
        添加結果的狀態資訊
    """
    if not rag_retriever:
        return "RAG 系統未初始化。"
    
    try:
        import json
        import arxiv
        import tempfile
        import os
        from ..rag.private_file_rag import PrivateFileRAG
        
        if not isinstance(rag_retriever, PrivateFileRAG):
            return "RAG 系統格式不正確。"
        
        # 解析 arXiv IDs
        arxiv_ids = json.loads(arxiv_ids_json)
        if not isinstance(arxiv_ids, list) or not arxiv_ids:
            return "無效的 arXiv ID 格式。"
        
        print(f"   📥 [arXiv] 正在下載 {len(arxiv_ids)} 篇論文...")
        
        # 下載論文 PDF
        downloaded_files = []
        for arxiv_id in arxiv_ids[:5]:  # 限制最多 5 篇
            try:
                # 搜尋論文
                search = arxiv.Search(id_list=[arxiv_id])
                paper = next(search.results(), None)
                
                if not paper:
                    print(f"   ⚠️ 找不到論文 {arxiv_id}")
                    continue
                
                # 下載 PDF 到臨時檔案
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    paper.download_pdf(dirpath=os.path.dirname(tmp_file.name), filename=os.path.basename(tmp_file.name))
                    downloaded_files.append(tmp_file.name)
                    print(f"   ✓ 已下載論文 {arxiv_id}: {paper.title[:50]}...")
                    
            except Exception as e:
                print(f"   ⚠️ 下載論文 {arxiv_id} 失敗: {e}")
                continue
        
        if not downloaded_files:
            return "未能下載任何論文。"
        
        print(f"   🔄 [RAG] 正在將 {len(downloaded_files)} 篇論文添加到 RAG 系統...")
        
        # 將論文添加到 RAG 系統
        documents, status_msg = rag_retriever.process_files(downloaded_files)
        
        # 清理臨時檔案
        for file_path in downloaded_files:
            try:
                os.unlink(file_path)
            except:
                pass
        
        if documents:
            return f"✅ 成功添加 {len(downloaded_files)} 篇論文到 RAG 系統。{status_msg}"
        else:
            return f"⚠️ 論文下載成功但處理失敗：{status_msg}"
            
    except Exception as e:
        return f"添加論文失敗: {e}"


def get_tools_list(rag_retriever=None):
    """
    獲取工具列表
    注意：部分工具需要 rag_retriever，所以需要動態創建
    """
    # 創建帶有 rag_retriever 的工具包裝器
    if rag_retriever:
        def query_pdf_wrapper(query: str) -> str:
            """
            查詢 PDF 知識庫中的相關資訊。
            當問題涉及論文內容、研究概念、方法論或學術理論時使用此工具。
            支持多文件檢索和進階 RAG 方法。
            
            Args:
                query: 查詢問題
            
            Returns:
                基於 PDF 知識庫的回答
            """
            return query_pdf_knowledge(query, rag_retriever=rag_retriever)
        
        def extract_keywords_wrapper(query: str) -> str:
            """
            從 PDF 知識庫中提取學術關鍵字，用於 arXiv 搜尋。
            當需要查找相關學術論文時使用此工具。
            
            Args:
                query: 查詢問題
            
            Returns:
                關鍵字 JSON 陣列字串
            """
            return extract_keywords_from_pdf(query, rag_retriever=rag_retriever)
        
        def add_arxiv_papers_wrapper(arxiv_ids_json: str) -> str:
            """
            下載 arXiv 論文並添加到 RAG 系統中。
            
            Args:
                arxiv_ids_json: arXiv ID 的 JSON 陣列字串
            
            Returns:
                添加結果的狀態資訊
            """
            return add_arxiv_papers_to_rag(arxiv_ids_json, rag_retriever=rag_retriever)
        
        # 創建工具
        pdf_tool = tool(query_pdf_wrapper)
        pdf_tool.name = "query_pdf_knowledge"
        
        keywords_tool = tool(extract_keywords_wrapper)
        keywords_tool.name = "extract_keywords_from_pdf"
        
        arxiv_search_tool = tool(search_arxiv_papers)
        arxiv_search_tool.name = "search_arxiv_papers"
        
        add_papers_tool = tool(add_arxiv_papers_wrapper)
        add_papers_tool.name = "add_arxiv_papers_to_rag"
        
        return [
            get_company_deep_info, 
            search_web, 
            pdf_tool,
            keywords_tool,
            arxiv_search_tool,
            add_papers_tool
        ]
    else:
        return [get_company_deep_info, search_web]

