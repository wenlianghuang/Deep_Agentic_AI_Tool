# Multi-Agent 協作架構說明

本文件說明 Deep Agentic AI Tool 的 **Multi-Agent 協作** 設計：從單一研究 Agent 演進為「Supervisor + 多個專長 Researcher」，各專長使用專屬工具，最後經筆記彙總與報告產出。

---

## 一、演進歷程（討論與實作）

### 1. 原始架構：單一線性流程

最初為單一研究流程：

- **Planner** → **Research Agent**（綁定所有工具）→ **Tools** → **Note Taker** → **Final Report**
- 一個 Research Agent 負責所有任務類型，工具全綁在一起，由 LLM 自行選擇。

### 2. 第一階段：Supervisor + 單一 Research Agent

為引入「派單」概念、方便日後擴充，先做了最小改動：

- 在狀態中新增 **`current_agent`**，標示目前執行的節點。
- 新增 **Supervisor** 節點：規劃完成後先進入 Supervisor，由它決定下一動是「繼續研究」或「產出報告」。
- **Note Taking** 完成後改為回到 **Supervisor**，由 Supervisor 依任務是否做完，決定再派給 Research Agent 或進入 Final Report。
- 流程變為：`Planner → Supervisor → Research Agent ⇄ Tools → Note Taking → Supervisor → … → Final Report`。

此時仍只有一個 Research Agent，但「誰來決定下一步」的邏輯已集中在 Supervisor。

### 3. 第二階段：完整 Multi-Agent（多專長 Researcher）

在 Supervisor 架構穩定後，改為多個專長 Researcher，各自使用專屬工具：

- **學術專長**（`academic_researcher`）：PDF 知識庫、arXiv 論文搜尋與擴展、必要時輔助網路搜尋。
- **股票專長**（`stock_researcher`）：公司深度資訊（yfinance）、網路搜尋（新聞/動態）。
- **網路專長**（`web_researcher`）：僅網路搜尋，處理一般資訊與新聞類任務。

Supervisor 依「當前未完成任務」的**任務內容**（關鍵字）派單給對應專長；每個專長只會接到與自己工具相符的任務，避免無關工具被呼叫。

---

## 二、架構總覽

### 流程圖

```
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Planner   │ 拆解問題為多個子任務
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │ 成功        │ 失敗/重試   │ planner_fallback
              ▼             ▼             ▼
        ┌──────────┐  (retry/fallback) ┌──────────┐
        │Supervisor│◄──────────────────┤  Fallback│
        └────┬─────┘                   └──────────┘
             │
             │ 依「下一個任務」類型派單
             ├──────────────────────────────────────────┐
             │                                          │
    ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
    │academic_researcher│  │ stock_researcher │  │ web_researcher  │
    │  (學術/PDF/arXiv) │  │ (股票/公司/新聞)  │  │  (網路搜尋)      │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
             │ 需呼叫工具時         │                    │
             ▼                    ▼                    ▼
    tools_academic       tools_stock          tools_web
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │ 工具結果回到對應 researcher（可多輪）
                                  │ 完成一任務後
                    ┌─────────────▼─────────────┐
                    │       Note Taking         │ 整理筆記、更新 completed_tasks
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │       Supervisor          │ 還有任務？→ 派給對應專長
                    └─────────────┬─────────────┘  沒有 → Final Report
                                  │
                    ┌─────────────▼─────────────┐
                    │      Final Report         │ 彙總所有筆記產出報告
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │          END              │
                    └───────────────────────────┘
```

### 狀態（DeepAgentState）

- **messages**：對話與工具呼叫紀錄。
- **tasks** / **completed_tasks**：待執行與已完成的子任務。
- **research_notes**：各任務的研究筆記（Note Taker 寫入）。
- **query**：使用者原始問題。
- **iteration**：迭代計數，防止無限循環。
- **current_agent**（Multi-Agent 新增）：目前執行的節點名稱（如 `supervisor`、`academic_researcher`、`stock_researcher`、`web_researcher`），供路由與日誌使用。
- 其餘為圖級重試用欄位（planner / research_agent 的 retry_count、succeeded、error）。

---

## 三、專長與工具對照

| 專長節點 | 用途 | 綁定工具 |
|----------|------|----------|
| **academic_researcher** | 學術理論、論文、PDF 知識庫、arXiv | `query_pdf_knowledge`、`extract_keywords_from_pdf`、`search_arxiv_papers`、`add_arxiv_papers_to_rag`、`search_web` |
| **stock_researcher** | 股票、公司財報、股價、營運、新聞 | `get_company_deep_info`、`search_web` |
| **web_researcher** | 一般網路搜尋、新聞、最新資訊 | `search_web` |

任務類型由 **關鍵字** 判定（與 Planner / 原 Researcher 邏輯一致）：

- **學術**：pdf、知識庫、理論、論文、學術、方法、arxiv、研究
- **股票**：股票、財報、營運、公司、投資、股價、市值、ticker、常見股票代碼（如 aapl, nvda）等
- **其餘**：歸類為 **web_researcher**

實作位置：`deep_agent_rag/graph/agent_graph.py` 的 `_task_type()` 與 `ACADEMIC_KEYWORDS` / `STOCK_KEYWORDS`；工具列表在 `deep_agent_rag/tools/agent_tools.py` 的 `get_tools_list_academic`、`get_tools_list_stock`、`get_tools_list_web`。

---

## 四、相關檔案

| 檔案 | 說明 |
|------|------|
| `deep_agent_rag/agents/state.py` | `DeepAgentState` 定義，含 `current_agent` |
| `deep_agent_rag/graph/agent_graph.py` | 圖表建構：Supervisor、三個專長節點、三個 Tool 節點、路由邏輯 |
| `deep_agent_rag/tools/agent_tools.py` | `get_tools_list_academic/stock/web`、`_make_pdf_arxiv_tools` |
| `deep_agent_rag/agents/researcher.py` | 共用 `research_agent_node`，由圖表以不同 `llm_with_tools` 綁定各專長工具 |
| `deep_agent_rag/agents/note_taker.py` | 筆記節點，將單一任務結果寫入 `research_notes` |
| `deep_agent_rag/agents/reporter.py` | 最終報告節點 |
| `deep_agent_rag/ui/gradio_interface.py` | UI 中各節點顯示名稱（含 supervisor、三個專長、三個 tools） |

---

## 五、如何執行與測試

### 啟動主程式

```bash
# 在專案根目錄
.venv/bin/python Deep_Agent_Gradio_RAG_localLLM_main.py
```

啟動後會建構 Multi-Agent 圖（含 RAG 與各專長工具），並開啟 Gradio 介面。在研究介面輸入問題即可觀察 Planner → Supervisor → 專長 Researcher → Note Taking → Supervisor → Final Report 的流程。

### 建議測試問題

1. **同時觸發學術 + 股票（推薦）**  
   「比較 Chain of Thought 和 Tree of Thoughts 的差異，並查一下 NVIDIA 最近股價與營運狀況。」  
   → 預期會派給 `academic_researcher` 與 `stock_researcher`，並在筆記與報告中彙總兩類結果。

2. **學術 + 網路**  
   「什麼是 RAG？並搜尋 2024 年 RAG 或檢索增強生成相關的重要新聞或進展。」  
   → 學術任務走 academic，新聞/進展走 web_researcher。

3. **純學術（單一專長、多輪工具）**  
   「從 PDF 知識庫查詢 Tree of Thoughts 的相關內容，若不足再從 arXiv 找相關論文並整理重點。」  
   → 僅派給 academic_researcher，可觀察同一專長內多次呼叫 PDF/arXiv 工具。

終端日誌會印出 Supervisor 派單目標（例如 `派單 → academic_researcher`、`派單 → stock_researcher`）以及各專長的執行與重試訊息，方便確認 Multi-Agent 流程是否正確。

---

## 六、小結

- **Supervisor** 負責「下一個要誰做」：依 `tasks` / `completed_tasks` 與任務內容派給對應專長或 Final Report。
- **三個專長 Researcher** 共用同一套研究邏輯（`research_agent_node`），但各自綁定不同工具，避免無關 API 被呼叫。
- 每個任務完成後經 **Note Taking** 寫入筆記，再回到 **Supervisor** 決定下一個任務由哪個專長執行，全部完成後進入 **Final Report** 產出最終報告。

此設計方便日後擴充更多專長（例如新增節點與 `get_tools_list_xxx`），只需在 Supervisor 的任務分類與圖的節點/邊上做對應擴充即可。
