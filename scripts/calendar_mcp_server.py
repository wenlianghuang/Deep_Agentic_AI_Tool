"""
Calendar MCP Server
將現有 Calendar 工具以 MCP 協議暴露，供 MCP client（如 Cursor、LangChain）呼叫。
執行方式（在專案根目錄）：
  uv run python scripts/calendar_mcp_server.py
或 stdio 模式（預設）：
  uv run python scripts/calendar_mcp_server.py --transport stdio
HTTP 模式（可選，方便測試）：
  uv run python scripts/calendar_mcp_server.py --transport http --port 8010
"""
import sys
from pathlib import Path

# 確保專案根目錄在 path 中
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# stdio 模式下 MCP 用 stdout 傳 JSON-RPC，任何 print() 會破壞協定。
# 在 import 可能產生輸出的模組前，先把 stdout 導到 stderr，匯入完成後再還原。
_original_stdout = sys.stdout
sys.stdout = sys.stderr
try:
    # FastMCP：langchain-mcp-adapters 範例用 mcp.server.fastmcp；專案 lock 內有 fastmcp 套件
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        try:
            from fastmcp import FastMCP
        except ImportError:
            from fastmcp.server import FastMCP

    from deep_agent_rag.tools.calendar_tool import (
        create_calendar_event,
        update_calendar_event,
        delete_calendar_event,
        list_calendar_events,
    )
finally:
    sys.stdout = _original_stdout

# FastMCP(name) 僅接受 name，不支援 description 等關鍵字
mcp = FastMCP("Calendar")


def _invoke_lc_tool(tool, **kwargs):
    """呼叫 LangChain tool，過濾掉 None 的參數以符合 schema。工具內 print 導向 stderr 以免破壞 stdio JSON-RPC。"""
    inp = {k: v for k, v in kwargs.items() if v is not None}
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        return tool.invoke(inp)
    finally:
        sys.stdout = old_stdout


@mcp.tool()
def create_calendar_event_mcp(
    summary: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
    location: str = "",
    attendees: str = "",
    timezone: str = "Asia/Taipei",
) -> str:
    """
    創建行事曆事件。
    summary: 事件標題
    start_datetime: 開始時間 (格式: YYYY-MM-DDTHH:MM:SS，例如: 2026-01-25T09:00:00)
    end_datetime: 結束時間 (格式: YYYY-MM-DDTHH:MM:SS)
    description: 事件描述（可選）
    location: 事件地點（可選）
    attendees: 參與者郵箱，多個用逗號分隔（可選）
    timezone: 時區（預設: Asia/Taipei）
    """
    return _invoke_lc_tool(
        create_calendar_event,
        summary=summary,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        description=description or "",
        location=location or "",
        attendees=attendees or "",
        timezone=timezone,
    )


@mcp.tool()
def update_calendar_event_mcp(
    event_id: str,
    summary: str = None,
    start_datetime: str = None,
    end_datetime: str = None,
    description: str = None,
    location: str = None,
    attendees: str = None,
    timezone: str = "Asia/Taipei",
) -> str:
    """
    更新現有行事曆事件。
    event_id: 要更新的事件 ID
    summary / start_datetime / end_datetime / description / location / attendees: 可選，不提供則不更新
    timezone: 時區（預設: Asia/Taipei）
    """
    return _invoke_lc_tool(
        update_calendar_event,
        event_id=event_id,
        summary=summary,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        description=description,
        location=location,
        attendees=attendees,
        timezone=timezone,
    )


@mcp.tool()
def delete_calendar_event_mcp(event_id: str) -> str:
    """刪除行事曆事件。event_id: 要刪除的事件 ID"""
    return _invoke_lc_tool(delete_calendar_event, event_id=event_id)


@mcp.tool()
def list_calendar_events_mcp(
    max_results: int = 10,
    time_min: str = None,
    time_max: str = None,
) -> str:
    """
    列出行事曆事件。
    max_results: 最大返回結果數（預設: 10）
    time_min: 開始時間過濾（可選，格式: YYYY-MM-DDTHH:MM:SS）
    time_max: 結束時間過濾（可選）
    """
    return _invoke_lc_tool(
        list_calendar_events,
        max_results=max_results,
        time_min=time_min,
        time_max=time_max,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calendar MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8010, help="Port for HTTP transport")
    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="http", port=args.port)
