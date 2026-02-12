# Calendar MCP 測試

將 Calendar 工具以 MCP Server 暴露，並用 LangChain MCP 客戶端測試。

## 依賴

- 專案已包含 `mcp`、`langchain-mcp-adapters`；Calendar MCP server 使用 FastMCP，已加入 `fastmcp` 至 `pyproject.toml`。
- 若尚未鎖定：在專案根目錄執行 `uv lock`。

## 1. 啟動 Calendar MCP Server（可選：手動測試）

**stdio 模式**（預設，供 client 以子進程啟動時使用）：

```bash
uv run python scripts/calendar_mcp_server.py
# 或
uv run python scripts/calendar_mcp_server.py --transport stdio
```

**HTTP 模式**（可搭配其他 MCP 客戶端，例如 Cursor）：

```bash
uv run python scripts/calendar_mcp_server.py --transport http --port 8010
```

之後在 MCP 設定中可指向 `http://localhost:8010/mcp`（依 FastMCP HTTP 端點為準）。

## 2. 執行客戶端測試

測試會自動以 stdio 啟動 `calendar_mcp_server.py`，載入 MCP tools 並呼叫「列出行事曆事件」：

```bash
# 在專案根目錄
uv run python scripts/test_calendar_mcp_client.py
```

預期輸出包含：
- 已載入 4 個 MCP 工具（create / update / delete / list）
- 呼叫 `list_calendar_events_mcp` 的結果（行事曆事件列表或「目前沒有找到任何行事曆事件」）

## 3. 暴露的工具

| MCP 工具名稱 | 說明 |
|-------------|------|
| `create_calendar_event_mcp` | 創建行事曆事件 |
| `update_calendar_event_mcp` | 更新現有事件 |
| `delete_calendar_event_mcp` | 刪除事件 |
| `list_calendar_events_mcp` | 列出行事曆事件 |

底層仍使用 `deep_agent_rag.tools.calendar_tool` 的 Google Calendar API（需已設定 `credentials.json` / `token.json`）。

## 4. 與現有 Calendar UI 的關係

- **原有流程不變**：Gradio 的「Calendar Tool」分頁仍直接使用 `calendar_agent` 與 `calendar_tool`，無需改動。
- **MCP 為額外介面**：同一組工具可同時給 Cursor、LangGraph 或其他 MCP 客戶端使用。
