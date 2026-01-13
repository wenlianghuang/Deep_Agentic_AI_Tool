# 🛡️ 內容過濾 Guardrails 文件

## 概述
本系統為 Simple Chatbot 實作了內容過濾 Guardrails 功能。它使用 `jieba` 進行精確的中文斷詞，並支援英文詞彙的不分大小寫比對，能自動檢測並攔截包含敏感內容的 AI 回應。

## 主要特點
*   **雙語支援 (Dual-Language Support)**：精確處理繁體中文與英文。
*   **基於密度的過濾 (Density-Based Filtering)**：僅在敏感詞彙密度超過 **5%** 時進行攔截（允許學術討論或低頻率出現的情境）。
*   **零侵入性 (Zero-Intrusion)**：透過 LangChain `RunnableLambda` 無縫整合，不影響對話流程。
*   **高度可自訂 (Customizable)**：可輕鬆設定關鍵字、門檻值與攔截訊息。

## 配置設定
所有設定皆定義於 `deep_agent_rag/ui/simple_chatbot_interface.py`。

### 1. 敏感關鍵字列表 (Blocked Keywords)
系統預設配置了以下敏感詞彙：
```python
BLOCKED_KEYWORDS = [
    "伊斯蘭教", "阿拉", "回教徒", "默罕默德",  # 中文
    "Islam", "Allah", "Muslim", "Muhammad"      # 英文
]
```

### 2. 門檻設定 (Thresholds)
*   **密度門檻 (Density Threshold)**：`0.05` (5%)
*   **計算方式**：`敏感詞數量 / 總詞數`

### 3. 攔截訊息 (Blocking Message)
> "抱歉，您的問題包含敏感內容，無法回答。請換個話題或重新表述您的問題。"

## 運作原理
1.  **斷詞 (Tokenization)**：將文本切分為詞彙，中文使用 `jieba`，英文使用空白/標準方式分隔。
2.  **比對 (Matching)**：將詞彙與 `BLOCKED_KEYWORDS` 列表進行比對（英文不區分大小寫）。
3.  **密度計算 (Density Calculation)**：計算敏感詞彙佔總詞彙的比例。
4.  **執行動作 (Action)**：
    *   **若密度 ≥ 5%**：將完整回應替換為預設的攔截訊息。
    *   **若密度 < 5%**：保留並輸出原始回應。

## 使用方法

### 啟動聊天機器人
```bash
uv run python main.py
```
在 Gradio 介面中打開 **Simple Chatbot** 標籤頁。您可以在「🛡️ 內容過濾 Guardrails」展開區塊中查看目前的 Guardrails 設定。

### 執行測試
驗證 Guardrails 邏輯：
```bash
uv run python test_guardrails.py
```

## 自訂指南

### 新增關鍵字
編輯 `deep_agent_rag/ui/simple_chatbot_interface.py` 中的 `BLOCKED_KEYWORDS` 列表：
```python
BLOCKED_KEYWORDS = [
    "新關鍵字1",
    "新關鍵字2",
    # ...
]
```
*注意：`jieba` 自定義詞典會在初始化時自動更新。*

### 調整靈敏度
修改 `KEYWORD_DENSITY_THRESHOLD`：
```python
KEYWORD_DENSITY_THRESHOLD = 0.10  # 提高至 10%
```

## 疑難排解
*   **jieba 分詞不準確**：請確認 `_init_jieba_custom_dict()` 是否已被呼叫以註冊新關鍵字。
*   **誤判 (False Positives)**：調整密度門檻或檢視關鍵字列表。
*   **效能**：系統使用 `jieba` 快取機制；首次載入可能稍慢，後續檢查時間 `< 1ms`。

## 相關檔案
*   **實作檔案**：`deep_agent_rag/ui/simple_chatbot_interface.py`
*   **測試檔案**：`test_guardrails.py`
*   **依賴套件**：需要 `jieba`（已配置於 `pyproject.toml`）。

---
**最後更新**：2026-01-13