"""
Parlant SDK 管理器
提供同步接口來訪問 Parlant 指南
"""
import asyncio
from typing import Dict, Optional, List
import parlant.sdk as p

# 指南緩存：存儲從 Parlant SDK 加載的指南文本
_guideline_cache: Dict[str, Dict[str, str]] = {
    "research": {},
    "email": {},
    "calendar": {}
}

# Parlant 代理實例
_parlant_agents: Dict[str, p.Agent] = {}
_parlant_server: Optional[p.Server] = None
_initialized = False


async def _initialize_parlant():
    """初始化 Parlant Server 和代理"""
    global _parlant_server, _parlant_agents, _initialized
    
    if _initialized:
        return
    
    print("🚀 正在初始化 Parlant SDK...")
    
    try:
        # Parlant Server 需要作為異步上下文管理器使用
        # 但我們只需要初始化指南，不需要運行服務器
        # 所以我們直接加載指南到緩存，而不實際啟動 Server
        
        # 從 Parlant SDK 加載指南到緩存
        await _load_guidelines_to_cache()
        
        # 如果需要實際使用 Parlant Server，可以在需要時啟動
        # 但對於我們的用例，指南緩存已經足夠
        
        _initialized = True
        print("✅ Parlant SDK 指南系統初始化完成")
    except Exception as e:
        print(f"⚠️ Parlant SDK 初始化警告: {e}")
        print("   將使用內置指南緩存...")
        # 即使出錯，也加載緩存
        await _load_guidelines_to_cache()
        _initialized = True


async def _load_guidelines_to_cache():
    """從 Parlant 代理加載指南到緩存"""
    # 由於 Parlant SDK 的指南是動態匹配的，我們需要手動構建指南文本
    # 這裡我們基於創建的指南構建文本
    
    # 研究代理指南
    _guideline_cache["research"]["tool_selection"] = """
工具選擇指南：

1. 當任務涉及以下內容時，使用 query_pdf_knowledge：
   - 學術理論、論文內容、研究方法
   - Tree of Thoughts、Chain of Thought 等概念
   - 方法論、框架、學術概念比較
   - PDF 知識庫中的任何內容

2. 當任務涉及以下內容時，使用 get_company_deep_info：
   - 股票代碼（如 MSFT, GOOGL, AAPL, TSLA, NVDA）
   - 公司財務數據、營運狀況
   - 股價、市值、本益比等財務指標
   - 公司業務摘要、營收增長

3. 當任務需要以下內容時，使用 search_web：
   - 最新新聞、市場動態
   - 一般知識、網路資訊
   - 無法從 PDF 或股票工具獲取的資訊
   - 即時資訊、時事新聞

重要原則：
- 只使用與任務相關的工具，不要使用不相關的工具
- 如果任務與股票無關，絕不使用 get_company_deep_info
- 如果任務涉及學術理論，優先使用 query_pdf_knowledge
- 可以進行多輪工具調用來深入挖掘資訊
- 當資訊已經足夠時，總結發現並回覆
    """
    
    _guideline_cache["research"]["research_behavior"] = """
研究行為指南：

1. 深度挖掘：
   - 進行多輪工具調用來深入挖掘資訊
   - 不要只調用一次工具就停止
   - 根據初步結果決定是否需要進一步查詢

2. 資訊整合：
   - 將不同來源的資訊整合分析
   - 比較不同工具的結果
   - 識別資訊之間的關聯性

3. 適時停止：
   - 當資訊足夠回答問題時，總結發現並回覆
   - 避免過度調用工具造成浪費
   - 如果資訊不足，明確說明限制

4. 避免重複：
   - 不要重複調用相同工具獲取相同資訊
   - 記錄已查詢的內容，避免重複
   - 如果工具返回相同結果，停止該方向的查詢
    """
    
    _guideline_cache["research"]["task_planning"] = """
任務規劃指南：

1. 學術理論問題（包含論文、理論、方法等關鍵字）：
   - 專注於 PDF 知識庫和學術搜尋
   - 包含理論比較、方法分析、應用場景
   - 不包含股票查詢任務

2. 股票相關問題（包含股票、ticker、公司、財報等關鍵字）：
   - 包含財務數據查詢、市場新聞、理論分析（如適用）
   - 包含產業競爭力和前景分析

3. 通用問題：
   - 根據問題內容智能選擇研究方式
   - 避免無關工具的使用

規劃原則：
- 每個問題拆解為 3-5 個具體的研究步驟
- 任務描述要清晰、可執行
- 避免生成與問題無關的任務
    """
    
    # 郵件代理指南
    _guideline_cache["email"]["email_writing"] = """
郵件撰寫指南：

1. 結構要求：
   - 適當的問候語（根據收件人和關係選擇）
   - 清晰的主體內容（根據用戶提示）
   - 適當的結尾
   - 簽名使用佔位符 [您的姓名] 和 [公司名稱]

2. 風格要求：
   - 專業、禮貌、簡潔明瞭
   - 根據用戶提示調整語氣（正式/非正式）
   - 中英文郵件都應符合對應語言習慣
   - 避免使用過於口語化的表達

3. 主題生成：
   - 簡潔、專業
   - 不超過 50 個字
   - 準確反映郵件內容
   - 避免使用過於籠統的主題
    """
    
    # 行事曆代理指南
    _guideline_cache["calendar"]["event_creation"] = """
事件創建指南：

1. 必填資訊：
   - 事件標題（清晰描述事件內容）
   - 開始時間（明確的日期和時間）
   - 結束時間（合理的持續時間）
   - 地點（如適用，需經過驗證）

2. 地點處理：
   - 使用 Google Maps 驗證地點有效性
   - 標準化地點格式（完整地址）
   - 計算交通時間（如配置了常用位置）
   - 建議出發時間（考慮交通時間）

3. 時間處理：
   - 明確指定日期和時間
   - 考慮時區（如適用）
   - 建議合理的持續時間
   - 避免時間衝突

4. 描述要求：
   - 事件描述要清晰、完整
   - 包含必要的背景資訊
   - 如有參與者，明確列出
    """


def initialize_parlant_sync():
    """
    同步初始化 Parlant（在應用啟動時調用）
    這個函數會在新的事件循環中運行異步初始化
    """
    try:
        # 使用 asyncio.run 創建新的事件循環
        # 這是最安全的方式，避免與現有事件循環衝突
        asyncio.run(_initialize_parlant())
    except RuntimeError as e:
        # 如果已經有運行中的事件循環，嘗試其他方式
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果循環正在運行，使用 create_task（但這可能不會立即執行）
                print("⚠️ 檢測到運行中的事件循環，指南將在後台初始化")
                asyncio.create_task(_initialize_parlant())
            else:
                loop.run_until_complete(_initialize_parlant())
        except Exception:
            # 最後的備用方案：直接加載緩存
            print("⚠️ 無法初始化 Parlant SDK，使用內置指南緩存")
            _load_guidelines_to_cache_sync()
    except Exception as e:
        print(f"⚠️ Parlant SDK 初始化錯誤: {e}")
        print("   使用內置指南緩存...")
        _load_guidelines_to_cache_sync()


def _load_guidelines_to_cache_sync():
    """同步加載指南到緩存（備用方案）"""
    global _initialized
    # 直接調用異步函數的邏輯（但不使用 await）
    # 這裡我們直接設置緩存內容
    asyncio.run(_load_guidelines_to_cache())
    _initialized = True


def get_guideline(agent_type: str, guideline_type: str) -> str:
    """
    獲取指南（同步接口）
    
    Args:
        agent_type: 'research', 'email', 'calendar'
        guideline_type: 'tool_selection', 'task_planning', 'research_behavior', etc.
    
    Returns:
        指南文本
    """
    if not _initialized:
        # 如果還沒初始化，先初始化
        initialize_parlant_sync()
    
    return _guideline_cache.get(agent_type, {}).get(guideline_type, "")


def get_customer_journey(agent_type: str) -> dict:
    """
    獲取客戶旅程定義（同步接口）
    
    Args:
        agent_type: 'research', 'email', 'calendar'
    
    Returns:
        客戶旅程字典
    """
    journeys = {
        "research": {
            "steps": ["問題理解 → 任務規劃 → 工具選擇 → 資訊收集 → 筆記整理 → 報告生成"],
            "checkpoints": [
                "任務規劃完成後，檢查任務是否合理且與問題相關",
                "工具調用後，檢查結果是否相關且有用",
                "報告生成前，檢查資訊是否充分且完整"
            ]
        },
        "email": {
            "steps": ["提示理解 → 草稿生成 → 反思評估 → 改進優化 → 用戶確認 → 發送"],
            "checkpoints": [
                "草稿生成後，進行反思評估質量",
                "改進後，再次評估是否達到標準",
                "發送前，用戶確認內容是否滿意"
            ]
        },
        "calendar": {
            "steps": ["提示理解 → 地點驗證 → 草稿生成 → 反思評估 → 資訊補全 → 用戶確認 → 創建"],
            "checkpoints": [
                "地點驗證後，檢查地點是否有效",
                "反思評估後，檢查是否有缺失資訊",
                "創建前，用戶確認所有資訊是否正確"
            ]
        }
    }
    
    return journeys.get(agent_type, {})

