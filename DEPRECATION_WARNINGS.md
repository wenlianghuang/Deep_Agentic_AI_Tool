# DeprecationWarning 處理說明

## 概述

應用啟動時可能會看到一些 `DeprecationWarning`，這些警告來自第三方依賴，**不影響功能**，但會讓輸出看起來很亂。

## 警告來源

### 1. httplib2 警告

```
DeprecationWarning: 'setName' deprecated - use 'set_name'
DeprecationWarning: 'leaveWhitespace' deprecated - use 'leave_whitespace'
```

**來源**：`httplib2` 包（Google API 客戶端的依賴）
- **原因**：`httplib2` 使用了舊版本的 `pyparsing` API
- **影響**：無，只是 API 命名變更的警告
- **解決**：等待 `httplib2` 更新或 Google 更新依賴

### 2. websockets 警告

```
DeprecationWarning: websockets.legacy is deprecated
DeprecationWarning: websockets.server.WebSocketServerProtocol is deprecated
```

**來源**：`websockets` 包（uvicorn 的依賴）
- **原因**：`uvicorn` 使用了舊版本的 `websockets` API
- **影響**：無，只是 API 遷移的警告
- **解決**：等待 `uvicorn` 更新到新版本的 `websockets`

## 處理方式

### 已實施的解決方案

在 `Deep_Agent_Gradio_RAG_localLLM_main.py` 中添加了警告過濾：

```python
import warnings

# 抑制第三方依賴的 DeprecationWarning（不影響功能）
warnings.filterwarnings("ignore", category=DeprecationWarning, module="httplib2")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvicorn.protocols.websockets")
```

這樣可以：
- ✅ 保持輸出整潔
- ✅ 不影響功能
- ✅ 只抑制特定的第三方依賴警告
- ✅ 仍然會顯示我們自己代碼的警告

## 為什麼不直接修復？

這些警告來自**第三方依賴**，我們無法直接修改：

1. **httplib2**：由 Google 維護，是 `google-auth-httplib2` 的依賴
2. **websockets**：由 uvicorn 使用，是 `uvicorn` 的依賴

我們只能：
- 等待依賴包更新
- 或者抑制這些特定的警告（已實施）

## 驗證

運行應用時，這些警告應該不再顯示：

```bash
python Deep_Agent_Gradio_RAG_localLLM_main.py
```

應該看到乾淨的輸出，沒有 DeprecationWarning。

## 注意事項

- ⚠️ 我們只抑制了**第三方依賴**的警告
- ✅ 我們自己代碼的警告仍然會顯示
- ✅ 其他類型的警告（如 UserWarning）仍然會顯示
- ✅ 錯誤（Error）仍然會顯示

## 未來更新

當以下依賴更新後，這些警告可能會自然消失：

1. `google-auth-httplib2` 更新 `httplib2` 依賴
2. `uvicorn` 更新到新版本的 `websockets` API

屆時可以移除警告過濾代碼。

---

**結論**：這些警告是無害的，已通過警告過濾處理，不影響應用功能。

