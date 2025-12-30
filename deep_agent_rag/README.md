# Deep Agent RAG System - 模組化架構

## 📁 文件結構

```
deep_agent_rag/
├── __init__.py              # 包初始化
├── config.py                # 配置和常量
├── models/                  # MLX 模型相關
│   ├── __init__.py
│   └── mlx_chat_model.py    # MLX 模型包裝器
├── rag/                     # RAG 系統
│   ├── __init__.py
│   └── rag_system.py        # RAG 初始化和檢索
├── tools/                   # 工具定義
│   ├── __init__.py
│   └── agent_tools.py       # 股票、網路搜尋、PDF 查詢工具
├── agents/                  # Agent 節點
│   ├── __init__.py
│   ├── state.py             # 狀態定義
│   ├── planner.py           # 規劃節點
│   ├── researcher.py        # 研究節點
│   ├── note_taker.py        # 筆記節點
│   └── reporter.py          # 報告節點
├── graph/                   # 圖表構建
│   ├── __init__.py
│   └── agent_graph.py       # 圖表構建和路由
├── ui/                      # UI 相關
│   ├── __init__.py
│   └── gradio_interface.py # Gradio 界面
└── utils/                   # 工具函數
    ├── __init__.py
    └── llm_utils.py         # LLM 工具函數
```

## 🚀 使用方式

運行主程序：

```bash
python Deep_Agent_Gradio_RAG_localLLM_main.py
```

## 📝 模組說明

### config.py
- 系統配置和常量
- 路徑設定
- 模型參數配置

### models/
- **mlx_chat_model.py**: MLX 模型的 LangChain 包裝器，將 MLX 模型整合到 LangChain 生態系統

### rag/
- **rag_system.py**: RAG 系統初始化，處理 PDF 載入、向量化和檢索

### tools/
- **agent_tools.py**: 定義所有工具函數（股票查詢、網路搜尋、PDF 查詢）

### agents/
- **state.py**: 定義 DeepAgentState 狀態結構
- **planner.py**: 規劃節點，將問題拆解為研究任務
- **researcher.py**: 研究節點，使用工具進行深度研究
- **note_taker.py**: 筆記節點，將研究結果轉化為筆記
- **reporter.py**: 報告節點，生成最終報告

### graph/
- **agent_graph.py**: 構建 LangGraph 圖表，定義節點連接和路由邏輯

### ui/
- **gradio_interface.py**: Gradio Web 界面，提供流式更新功能

### utils/
- **llm_utils.py**: LLM 實例的創建和管理

## 🔧 開發建議

1. **添加新工具**: 在 `tools/agent_tools.py` 中添加新的 `@tool` 函數
2. **修改 Agent 邏輯**: 在 `agents/` 目錄下對應的文件中修改
3. **調整配置**: 在 `config.py` 中修改配置參數
4. **自定義 UI**: 在 `ui/gradio_interface.py` 中修改界面

## 📦 依賴

- langchain
- langgraph
- mlx-lm
- gradio
- yfinance
- tavily-python
- chromadb

