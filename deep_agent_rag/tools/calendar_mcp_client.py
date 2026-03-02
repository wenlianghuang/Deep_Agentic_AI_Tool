"""
Calendar MCP Client
供 Gradio / calendar_agent 透過 MCP 呼叫行事曆工具。
使用 stdio 模式：不依賴 HTTP，由 client 以子進程啟動 Calendar MCP Server（與 test_calendar_mcp_client 相同），
取得 tools 後提供與原 calendar_tool 相容的 .invoke() 介面。
"""
import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Any

from ..config import USE_CALENDAR_MCP

# 專案根目錄（scripts/ 與 deep_agent_rag/ 的父目錄）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SERVER_SCRIPT = _PROJECT_ROOT / "scripts" / "calendar_mcp_server.py"

_tools: Optional[List[Any]] = None
_tool_name_to_tool: Optional[dict] = None


def _normalize_result(result: Any) -> str:
    """將 MCP tool 回傳的 list/dict 轉成字串，與原 calendar_tool 回傳一致。"""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return result.get("output", str(result))
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else str(result)
    return str(result)


def _sync_ainvoke(tool: Any, input_dict: dict) -> str:
    """在 sync 環境下呼叫 async tool.ainvoke，並將結果正規化為 str。"""
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(tool.ainvoke(input_dict))
        return _normalize_result(result)
    finally:
        loop.close()


class _CalendarToolWrapper:
    """提供與 LangChain tool 相容的 .invoke(input_dict) -> str 介面。"""
    def __init__(self, mcp_tool: Any):
        self._tool = mcp_tool

    def invoke(self, input_dict: dict) -> str:
        return _sync_ainvoke(self._tool, input_dict)


def initialize() -> bool:
    """
    以 stdio 模式載入 Calendar MCP tools（client 會 spawn server 子進程，無需 HTTP）。
    從 sync 呼叫，內部會 run async get_tools。
    """
    global _tools, _tool_name_to_tool
    if not USE_CALENDAR_MCP:
        return False
    if _tools is not None:
        return True
    if not _SERVER_SCRIPT.exists():
        print(f"⚠️ [Calendar MCP] 找不到 server 腳本: {_SERVER_SCRIPT}")
        return False

    async def _get_tools_stdio():
        from langchain_mcp_adapters.client import MultiServerMCPClient
        # stdio：不依賴 http_app()，client 會以子進程啟動 server，每次需要時由 adapter 管理
        client = MultiServerMCPClient({
            "calendar": {
                "command": sys.executable,
                "args": [str(_SERVER_SCRIPT), "--transport", "stdio"],
                "transport": "stdio",
            }
        })
        return await client.get_tools()

    try:
        _tools = asyncio.run(_get_tools_stdio())
        _tool_name_to_tool = {t.name: t for t in _tools}
        return True
    except Exception as e:
        print(f"⚠️ [Calendar MCP] 載入 tools 失敗（stdio）: {e}")
        return False


def get_create_calendar_event_tool():
    """回傳具 .invoke(input_dict) -> str 的 create 工具；若 MCP 未啟用則回傳 None，由 caller 改用地區 calendar_tool。"""
    if not USE_CALENDAR_MCP or not initialize():
        return None
    t = _tool_name_to_tool.get("create_calendar_event_mcp")
    if t is None:
        return None
    return _CalendarToolWrapper(t)


def get_update_calendar_event_tool():
    """回傳具 .invoke(input_dict) -> str 的 update 工具。"""
    if not USE_CALENDAR_MCP or not initialize():
        return None
    t = _tool_name_to_tool.get("update_calendar_event_mcp")
    if t is None:
        return None
    return _CalendarToolWrapper(t)


def get_delete_calendar_event_tool():
    """回傳具 .invoke(input_dict) -> str 的 delete 工具。"""
    if not USE_CALENDAR_MCP or not initialize():
        return None
    t = _tool_name_to_tool.get("delete_calendar_event_mcp")
    if t is None:
        return None
    return _CalendarToolWrapper(t)
