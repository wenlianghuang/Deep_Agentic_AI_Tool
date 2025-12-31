# Deep Agentic AI Tool

一個功能完整的深度研究代理系統，整合 RAG（檢索增強生成）、本地 MLX 模型支援，以及多種智能工具。系統基於 LangGraph 構建，提供智能研究、郵件生成、行事曆管理等全方位 AI 助手功能。

## 🚀 核心功能

### 🔍 深度研究代理 (Deep Research Agent)
- 智能多步驟研究規劃與執行
- 自動任務分解（學術、股票、一般查詢）
- 動態工具選擇（股票查詢、網路搜尋、PDF 知識庫）
- 即時進度追蹤與筆記整理
- 完整研究報告生成

**詳細說明請參考：[功能詳述](FEATURES.md#深度研究代理)**

### 📧 智能郵件助手 (Email Agent)
- AI 自動生成郵件草稿
- Gmail API 整合（避免垃圾郵件分類）
- 支援中英文郵件
- AI 反思評估與自動改進
- 可編輯草稿後發送

**詳細說明請參考：[功能詳述](FEATURES.md#智能郵件助手)**  
**設置指南請參考：[Gmail API 設置](GMAIL_API_SETUP.md)**

### 📅 智能行事曆管理 (Calendar Agent)
- AI 自動生成行事曆事件
- Google Calendar API 整合
- AI 反思評估與自動改進（最多 3 輪迭代）
- Google Maps 地點驗證與標準化
- 自動計算交通時間與建議出發時間
- 支援創建、更新、刪除事件

**詳細說明請參考：[功能詳述](FEATURES.md#智能行事曆管理)**  
**Google Maps 整合說明請參考：[Google Maps 整合](GOOGLE_MAPS_INTEGRATION.md)**

### 📊 研究工具集
- **股票查詢**：即時股票數據（財務狀況、市值、P/E 比等）
- **網路搜尋**：Tavily API 整合，獲取最新資訊
- **PDF 知識庫**：RAG 系統，向量化語義搜尋（目前包含 "Tree of Thoughts" 論文）

## 🛠️ 技術特色

- **本地 MLX 模型**：使用 Qwen2.5-Coder-7B-Instruct-4bit，保護隱私，無需 API 金鑰
- **Groq API 備援**：自動切換到 Groq API（當本地模型不可用或額度用完時）
- **LangGraph 編排**：複雜的代理工作流程管理
- **Gradio Web 界面**：現代化、用戶友好的 Web UI，即時更新
- **模組化架構**：清晰、可擴展的代碼結構
- **外部 SSD 支援**：可配置模型緩存到外部 SSD（節省本地空間）

**詳細架構說明請參考：[系統架構](ARCHITECTURE.md)**

## 📋 系統需求

- Python >= 3.13
- macOS（MLX 支援）或 Linux
- Google Cloud 帳號（用於 Gmail/Calendar API - 可選，僅郵件/行事曆功能需要）
- Tavily API 金鑰（用於網路搜尋 - 可選，可在 `.env` 中配置）
- Google Maps API 金鑰（用於行事曆地點驗證 - 可選，可在 `.env` 中配置）

## 🛠️ 快速開始

### 1. 安裝依賴

```bash
# 使用 uv（推薦）
uv sync

# 或使用 pip
pip install -e .
```

### 2. 環境變數配置

在專案根目錄創建 `.env` 文件：

```env
# 可選：Groq API（用於更快的推理）
GROQ_API_KEY=your_groq_api_key_here

# 可選：Tavily API（用於網路搜尋）
TAVILY_API_KEY=your_tavily_api_key_here

# 可選：Google Maps API（用於行事曆地點驗證）
NORMAL_GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# 可選：用戶常用位置（用於計算交通時間）
USER_HOME_ADDRESS=台北市信義區信義路五段7號
USER_OFFICE_ADDRESS=台北市大安區敦化南路二段216號

# 可選：Gmail/Calendar API 憑證文件路徑
GMAIL_CREDENTIALS_FILE=credentials.json
GMAIL_TOKEN_FILE=token.json
CALENDAR_CREDENTIALS_FILE=credentials.json
CALENDAR_TOKEN_FILE=token.json
```

**詳細配置說明請參考：[配置指南](CONFIGURATION.md)**

### 3. 準備數據（可選）

- 將 PDF 文件放在 `data/` 目錄
- 系統預設使用 `data/Tree_of_Thoughts.pdf`
- 可在 `deep_agent_rag/config.py` 中修改路徑

### 4. 設置 API（可選）

- **Gmail API**：參考 [Gmail API 設置指南](GMAIL_API_SETUP.md)
- **Google Calendar API**：與 Gmail API 共用相同的 OAuth2 憑證
- **Google Maps API**：參考 [Google Maps 整合說明](GOOGLE_MAPS_INTEGRATION.md)

### 5. 啟動應用

```bash
python Deep_Agent_Gradio_RAG_localLLM_main.py
```

Gradio 界面將在以下地址可用：
- 本地：`http://localhost:7860`
- 網路：`http://0.0.0.0:7860`

## 🎯 使用指南

### 深度研究代理

1. 切換到 **"🔍 Deep Research Agent"** 標籤
2. 輸入研究問題
3. 點擊 **"🔍 開始研究"**
4. 查看即時更新：
   - **當前狀態**：顯示正在執行的節點
   - **任務列表**：顯示研究計劃與進度
   - **研究筆記**：即時研究過程筆記
   - **最終報告**：完整研究報告（逐句流式顯示）

**使用範例請參考：[功能詳述](FEATURES.md#使用範例)**

### 郵件工具

1. 切換到 **"📧 Email Tool"** 標籤
2. 輸入郵件提示（例如："寫一封感謝信"）
3. 輸入收件人郵箱
4. 點擊 **"📝 生成郵件草稿"**
5. 查看 AI 反思評估結果（如有改進建議）
6. 檢查並編輯生成的主題和正文
7. 點擊 **"📧 發送郵件"** 發送

**詳細使用說明請參考：[功能詳述](FEATURES.md#智能郵件助手)**

### 行事曆工具

1. 切換到 **"📅 Calendar Tool"** 標籤
2. 使用快速選擇按鈕或輸入完整事件提示
3. 點擊 **"📝 生成事件草稿"**
4. 查看 AI 反思評估結果和地點驗證資訊
5. 如有缺失資訊，系統會顯示下拉選單供選擇
6. 檢查並修改事件內容
7. 點擊 **"創建事件"** 確認

**詳細使用說明請參考：[功能詳述](FEATURES.md#智能行事曆管理)**

## 🏗️ 專案結構

```
Deep_Agentic_AI_Tool/
├── deep_agent_rag/              # 主套件
│   ├── agents/                  # 代理節點
│   │   ├── planner.py          # 任務規劃節點
│   │   ├── researcher.py       # 研究執行節點
│   │   ├── note_taker.py       # 筆記整理節點
│   │   ├── reporter.py         # 最終報告生成
│   │   ├── email_agent.py      # 郵件生成代理
│   │   ├── email_reflection_agent.py  # 郵件反思代理
│   │   ├── calendar_agent.py    # 行事曆事件生成代理
│   │   ├── calendar_reflection_agent.py  # 行事曆反思代理
│   │   └── state.py            # 狀態定義
│   ├── graph/                   # LangGraph 編排
│   │   └── agent_graph.py      # 圖表構建與路由
│   ├── models/                  # 模型包裝器
│   │   └── mlx_chat_model.py   # MLX 模型整合
│   ├── rag/                     # RAG 系統
│   │   └── rag_system.py       # PDF 載入與檢索
│   ├── tools/                   # 工具定義
│   │   ├── agent_tools.py      # 股票、網路搜尋、PDF 工具
│   │   ├── email_tool.py       # 郵件發送工具
│   │   ├── calendar_tool.py    # 行事曆管理工具
│   │   └── googlemaps_tool.py  # Google Maps 整合工具
│   ├── ui/                      # 用戶界面
│   │   └── gradio_interface.py # Gradio Web UI
│   ├── utils/                   # 工具函數
│   │   └── llm_utils.py        # LLM 實例管理
│   └── config.py                # 配置
├── data/                        # 數據目錄
│   └── Tree_of_Thoughts.pdf
├── Deep_Agent_Gradio_RAG_localLLM_main.py  # 主入口點
├── main.py                      # 簡單入口點
├── pyproject.toml               # 專案依賴
└── README.md                    # 本文件
```

**詳細架構說明請參考：[系統架構](ARCHITECTURE.md)**

## ⚙️ 配置

主要配置選項在 `deep_agent_rag/config.py`：

- **MLX 模型設定**：模型 ID、最大 tokens、溫度
- **RAG 設定**：PDF 路徑、嵌入模型、分塊大小
- **代理設定**：最大迭代次數、反思迭代次數
- **API 設定**：Groq、Gmail、Calendar、Google Maps
- **外部 SSD 路徑**：模型緩存目錄

**詳細配置說明請參考：[配置指南](CONFIGURATION.md)**

## 🔧 開發指南

### 添加新工具

1. 在 `deep_agent_rag/tools/` 中創建新工具函數
2. 使用 `@tool` 裝飾器
3. 在 `get_tools_list()` 中添加工具
4. 代理會自動發現並使用新工具

### 修改代理邏輯

- **規劃邏輯**：編輯 `deep_agent_rag/agents/planner.py`
- **研究邏輯**：編輯 `deep_agent_rag/agents/researcher.py`
- **報告生成**：編輯 `deep_agent_rag/agents/reporter.py`

### 自定義 UI

編輯 `deep_agent_rag/ui/gradio_interface.py` 修改 Web 界面。

**詳細開發指南請參考：[系統架構](ARCHITECTURE.md#開發指南)**

## 📦 主要依賴

- **LangChain & LangGraph**：代理框架與工作流程管理
- **MLX/MLX-LM**：本地模型推理（Apple Silicon 優化）
- **Gradio**：Web 界面
- **ChromaDB**：RAG 向量數據庫
- **Tavily**：網路搜尋 API
- **yfinance**：股票數據檢索
- **Google API Client**：Gmail/Calendar API 整合
- **googlemaps**：Google Maps API 整合

完整依賴列表請參考 `pyproject.toml`。

## 🐛 故障排除

### MLX 模型問題
- **模型無法載入**：確保有足夠的磁碟空間和記憶體
- **推理速度慢**：這是本地模型的正常現象，可考慮使用 Groq API 獲得更快結果

### Groq API 問題
- **額度用完**：系統會自動切換到本地 MLX 模型
- **API 錯誤**：檢查 `.env` 文件中的 `GROQ_API_KEY`

### RAG 系統問題
- **PDF 找不到**：確保 PDF 文件存在於 `config.py` 中指定的路徑
- **嵌入模型錯誤**：系統會嘗試重新下載模型（如果緩存損壞）

### Gmail/Calendar API 問題
- **授權錯誤**：刪除 `token.json` 並重新授權
- **憑證找不到**：確保 `credentials.json` 在專案根目錄
- 詳細說明請參考 [Gmail API 設置指南](GMAIL_API_SETUP.md)

### Google Maps API 問題
- **地點驗證失敗**：檢查 API 金鑰是否正確設置
- **交通時間無法計算**：確保設置了 `USER_HOME_ADDRESS` 或 `USER_OFFICE_ADDRESS`
- 詳細說明請參考 [Google Maps 整合說明](GOOGLE_MAPS_INTEGRATION.md)

## 📚 相關文檔

- **[功能詳述](FEATURES.md)** - 所有功能的詳細說明與使用範例
- **[系統架構](ARCHITECTURE.md)** - 系統架構、工作流程與開發指南
- **[配置指南](CONFIGURATION.md)** - 完整的配置選項說明
- **[Gmail API 設置](GMAIL_API_SETUP.md)** - Gmail API 設置步驟
- **[Google Maps 整合](GOOGLE_MAPS_INTEGRATION.md)** - Google Maps API 整合說明

## 📝 授權

[添加您的授權資訊]

## 🤝 貢獻

[添加貢獻指南]

## 📧 聯絡

[添加聯絡資訊]

## 🙏 致謝

- **LangChain & LangGraph**：優秀的代理框架
- **MLX Team**：高效的本地模型推理
- **Qwen Team**：Qwen2.5 模型
- **Jina AI**：嵌入模型

---

**注意**：本系統主要設計用於 macOS（Apple Silicon）以獲得最佳 MLX 性能。Linux 支援可用，但性能特徵可能不同。
