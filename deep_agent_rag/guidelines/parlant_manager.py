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
   - 事件標題（summary）：清晰描述事件內容，不超過 50 字
   - 開始時間（start_datetime）：明確的日期和時間，ISO 8601 格式
   - 結束時間（end_datetime）：合理的結束時間，考慮持續時間
   - 事件描述（description）：詳細說明事件內容、目的、議程等
   - 地點（location，如適用）：需經過 Google Maps 驗證
   - 參與者（attendees，如適用）：有效的郵箱地址列表

2. 地點處理：
   - 使用 Google Maps 驗證地點有效性
   - 標準化地點格式（完整地址）
   - 計算交通時間（如配置了常用位置）
   - 建議出發時間（考慮交通時間）
   - 如果驗證失敗，保留原始地址但記錄警告

3. 時間處理：
   - 明確指定日期和時間
   - 支援相對日期（今天、明天、後天）和絕對日期
   - 支援多種時間格式（24 小時制、12 小時制、中文時間）
   - 考慮時區（預設 Asia/Taipei）
   - 建議合理的持續時間（預設 1 小時）
   - 避免時間衝突

4. 描述要求：
   - 事件描述要清晰、完整
   - 包含必要的背景資訊
   - 如有參與者，明確列出
   - 如有議程，詳細說明
   - 根據事件類型調整描述的詳細程度

5. 參與者處理：
   - 只提取有效的郵箱地址（格式：user@domain.com）
   - 如果提示中只有名字沒有郵箱，則留空
   - 多個參與者用逗號分隔
   - 驗證郵箱格式的正確性

6. 資訊提取原則：
   - 從單一用戶提示中提取所有資訊
   - 如果資訊缺失，標記為缺失（不編造）
   - 優先使用用戶明確提供的資訊
   - 如果無法確定，使用合理的預設值

7. JSON 輸出格式：
   - 嚴格按照 JSON 格式輸出
   - 所有欄位都應該是字串類型
   - 缺失的欄位使用空字串
   - 不要包含額外的說明文字
    """
    
    _guideline_cache["calendar"]["event_reflection"] = """
事件反思評估指南：

1. 資訊完整性檢查：
   - 事件是否完整回應了用戶的原始提示？
   - 是否遺漏關鍵資訊（時間、地點、參與者、描述）？
   - 提取的資訊是否準確？

2. 時間合理性評估：
   - 開始和結束時間是否合理？
   - 持續時間是否適當（根據事件類型）？
   - 是否考慮了時區？
   - 是否有時間衝突的風險？

3. 描述清晰度評估：
   - 事件描述是否清晰、詳細？
   - 是否包含必要的議程或目的說明？
   - 是否提供了足夠的背景資訊？

4. 標題準確性評估：
   - 事件標題是否準確反映事件內容？
   - 是否簡潔明瞭（不超過 50 字）？
   - 是否專業且易於理解？

5. 參與者正確性評估：
   - 參與者郵箱是否正確提取？
   - 郵箱格式是否正確（user@domain.com）？
   - 是否只包含有效的郵箱地址？

6. 地點適配性評估：
   - 如果有地點，是否與事件類型匹配？
   - 地點是否經過驗證和標準化？
   - 是否提供了交通時間建議（如適用）？

評估原則：
- 只有在事件有嚴重問題（如遺漏關鍵資訊、時間不合理、描述不清楚）時才建議重新生成
- 輕微問題可以通過改進建議解決
- 如果事件品質很好，明確說明為什麼
    """
    
    _guideline_cache["calendar"]["event_improvement"] = """
事件改進生成指南：

1. 基於反思建議改進：
   - 仔細分析反思評估中提出的改進建議
   - 優先解決嚴重問題（資訊缺失、時間不合理等）
   - 逐步優化細節問題

2. 保持原始資訊：
   - 如果原始資訊正確，不要隨意更改
   - 只在必要時修改或補充資訊
   - 確保改進後的版本仍然符合用戶原始提示

3. 資訊補全原則：
   - 如果缺少關鍵資訊，嘗試從用戶提示中推斷
   - 如果無法推斷，保持為空（讓用戶補充）
   - 不要編造不存在的資訊

4. 格式標準化：
   - 日期格式：使用標準格式（2026-01-25）或相對日期（明天、今天）
   - 時間格式：使用 24 小時制（14:00）或 12 小時制（2:00 PM）
   - 郵箱格式：只包含有效的郵箱地址（user@domain.com）

5. 描述優化：
   - 確保描述清晰、完整
   - 包含必要的背景資訊和議程
   - 根據事件類型調整描述的詳細程度
    """
    
    _guideline_cache["calendar"]["location_handling"] = """
地點處理指南：

1. 地點驗證：
   - 使用 Google Maps API 驗證地點有效性
   - 標準化地點格式（使用完整地址）
   - 如果驗證失敗，保留原始地址但記錄警告

2. 地點標準化：
   - 優先使用標準化的完整地址
   - 包含城市、街道、門牌號等完整資訊
   - 確保地址格式符合 Google Calendar 要求

3. 交通時間計算（如配置了常用位置）：
   - 計算從常用位置到事件地點的交通時間
   - 提供出發時間建議（考慮交通時間）
   - 考慮交通方式（駕車、公共交通等）

4. 地點建議：
   - 如果地址驗證失敗，提供可能的替代地址建議
   - 如果地址模糊，提供澄清建議
   - 記錄驗證過程中的任何警告或錯誤

5. 地點缺失處理：
   - 如果用戶提示中沒有地點，保持為空
   - 不要為所有事件都添加地點
   - 根據事件類型判斷是否需要地點
    """
    
    _guideline_cache["calendar"]["time_parsing"] = """
時間解析指南：

重要：你必須直接輸出 ISO 8601 格式的日期時間，而不是原始字符串。

1. 日期解析和格式化：
   - 支援絕對日期：2026-01-25、2026/01/25 → 轉換為 2026-01-25
   - 支援相對日期：今天、明天、後天、today、tomorrow → 計算實際日期
   - 支援星期（重要計算規則）：
     * 中文：下週一、下週二、下週三、下週四、下週五、下週六、下週日
     * 英文：next Monday, next Tuesday, next Wednesday, next Thursday, next Friday, next Saturday, next Sunday
     * 計算規則（關鍵）：
       - "下週X" 永遠是指下一個指定的星期幾，不包括今天
       - 如果今天是週三，說"下週三"是指下一個週三（7天後），不是今天
       - 如果今天是週五，說"下週一"是指下一個週一（3天後）
       - 計算方法：從今天開始往後找，找到第一個匹配的星期幾（如果今天就是這個星期幾，則找下週的）
       - 星期對應：週一=Monday=0, 週二=Tuesday=1, 週三=Wednesday=2, 週四=Thursday=3, 週五=Friday=4, 週六=Saturday=5, 週日=Sunday=6
     * 範例計算：
       - 今天是 2026-01-27（週一），"下週三" = 2026-02-04（下一個週三，8天後）
       - 今天是 2026-01-29（週三），"下週三" = 2026-02-05（下一個週三，7天後，不是今天）
       - 今天是 2026-01-31（週五），"下週一" = 2026-02-03（下一個週一，3天後）
   - 如果無法確定日期，使用今天的日期

2. 時間解析和格式化：
   - 支援 24 小時制：14:00、09:00 → 轉換為 14:00:00、09:00:00
   - 支援 12 小時制：2:00 PM、9:00 AM → 轉換為 14:00:00、09:00:00
   - 支援中文時間：下午 2 點、上午 9 點 → 轉換為 14:00:00、09:00:00
   - 如果無法確定時間，使用 09:00:00（上午 9 點）

3. ISO 8601 格式要求：
   - 格式：YYYY-MM-DDTHH:MM:SS+08:00
   - 範例：2026-01-25T14:00:00+08:00
   - 時區：預設使用 Asia/Taipei (+08:00)
   - 必須包含完整的日期、時間和時區資訊
   - start_datetime 和 end_datetime 都必須是完整的 ISO 8601 格式

4. 持續時間處理：
   - 預設持續時間為 1 小時
   - 根據事件類型調整持續時間（會議通常 1-2 小時，活動可能更長）
   - 考慮用戶提示中提到的持續時間
   - end_datetime = start_datetime + 持續時間

5. 時間合理性檢查：
   - 確保結束時間晚於開始時間
   - 檢查是否有明顯的時間錯誤（如凌晨 3 點的會議）
   - 考慮工作時間（通常為 9:00-18:00）
   - 如果時間不合理，調整為合理的工作時間

6. 輸出要求：
   - 必須在 JSON 中直接輸出 start_datetime 和 end_datetime（ISO 8601 格式）
   - 同時保留 date 和 time 欄位（原始字符串，用於 UI 顯示）
   - 如果無法確定日期或時間，start_datetime 和 end_datetime 可以為空字符串
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

