"""
測試 Calendar MCP 客戶端
透過 langchain-mcp-adapters 連到 Calendar MCP server，取得 tools 並呼叫 list_calendar_events_mcp。
執行方式（在專案根目錄）：
  uv run python scripts/test_calendar_mcp_client.py
會自動以 stdio 啟動 scripts/calendar_mcp_server.py 並呼叫 list 工具。
"""
import asyncio
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# 使用 stdio 模式：client 會 spawn calendar_mcp_server.py 子進程
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools


async def main():
    server_script = _root / "scripts" / "calendar_mcp_server.py"
    if not server_script.exists():
        print(f"❌ 找不到 server 腳本: {server_script}")
        return

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script), "--transport", "stdio"],
        env=None,
    )

    print("🔌 正在連接 Calendar MCP Server (stdio)...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            print(f"✅ 已載入 {len(tools)} 個 MCP 工具:", [t.name for t in tools])

            # 找 list_calendar_events_mcp 並呼叫
            list_tool = None
            for t in tools:
                if "list_calendar" in t.name.lower() or t.name == "list_calendar_events_mcp":
                    list_tool = t
                    break
            if not list_tool:
                print("⚠️ 未找到 list_calendar_events 工具，改用第一個 tool 做測試")
                list_tool = tools[0]

            print(f"\n📋 呼叫工具: {list_tool.name}")
            try:
                result = await list_tool.ainvoke({"max_results": 5})
                print("--- 結果 ---")
                if isinstance(result, str):
                    print(result)
                elif isinstance(result, dict):
                    print(result.get("output", result))
                elif isinstance(result, list):
                    # MCP/LangChain 有時回傳 content 陣列，例如 [{"type": "text", "text": "..."}]
                    parts = []
                    for item in result:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and "text" in item:
                            parts.append(item["text"])
                        else:
                            parts.append(str(item))
                    print("\n".join(parts) if parts else result)
                else:
                    print(result)
                print("--- 結束 ---")
            except Exception as e:
                print(f"❌ 呼叫失敗: {e}")
                raise

    print("\n✅ Calendar MCP 客戶端測試完成。")


if __name__ == "__main__":
    asyncio.run(main())
