"""
Image Analysis MCP Client
供 Gradio / agent / graph 透過 MCP 呼叫圖片分析工具。
使用 stdio 模式，取得 tools 後提供與原 image_analysis_tool 相容的介面；失敗則 fallback 到 local。
"""
import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Any

from ..config import USE_IMAGE_ANALYSIS_MCP

# 專案根目錄
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SERVER_SCRIPT = _PROJECT_ROOT / "scripts" / "image_analysis_mcp_server.py"

_tools: Optional[List[Any]] = None
_tool_name_to_tool: Optional[dict] = None


def _normalize_result(result: Any) -> str:
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
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(tool.ainvoke(input_dict))
        return _normalize_result(result)
    finally:
        loop.close()


def _get_raw_mcp_tool():
    """內部使用：回傳 MCP adapter 的原始 LangChain tool（供 ToolNode 或 sync 呼叫）。"""
    if not USE_IMAGE_ANALYSIS_MCP or not initialize():
        return None
    t = _tool_name_to_tool.get("analyze_image_mcp")
    return t


def initialize() -> bool:
    """以 stdio 模式載入 Image Analysis MCP tools。"""
    global _tools, _tool_name_to_tool
    if not USE_IMAGE_ANALYSIS_MCP:
        return False
    if _tools is not None:
        return True
    if not _SERVER_SCRIPT.exists():
        print(f"⚠️ [ImageAnalysis MCP] 找不到 server 腳本: {_SERVER_SCRIPT}")
        return False

    async def _get_tools_stdio():
        from langchain_mcp_adapters.client import MultiServerMCPClient
        client = MultiServerMCPClient({
            "image_analysis": {
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
        print(f"⚠️ [ImageAnalysis MCP] 載入 tools 失敗（stdio）: {e}")
        return False


def get_analyze_image_tool():
    """
    回傳 analyze 工具供 agent_tools / ToolNode 使用。
    回傳 MCP adapter 的原始 LangChain tool（ToolNode 只接受 BaseTool），並將 name 設為 analyze_image。
    若 MCP 未啟用或失敗則回傳 None（caller 改用地區 tool）。
    """
    t = _get_raw_mcp_tool()
    if t is None:
        return None
    try:
        t.name = "analyze_image"
    except Exception:
        pass
    return t


def analyze_image_result(image_path: str, question: Optional[str] = None) -> str:
    """
    分析圖片：優先使用 MCP tool，失敗則使用本地 _analyze_image_internal。
    供 image_analysis_graph、image_reflection_agent 呼叫。
    """
    tool = _get_raw_mcp_tool()
    if tool is not None:
        try:
            return _sync_ainvoke(tool, {
                "image_path": image_path,
                "question": question or "",
            })
        except Exception as e:
            print(f"⚠️ [ImageAnalysis MCP] 呼叫失敗，改用本地: {e}")
    from .image_analysis_tool import _analyze_image_internal
    return _analyze_image_internal(image_path, question=question)
