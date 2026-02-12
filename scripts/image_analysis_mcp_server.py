"""
Image Analysis MCP Server
將圖片分析工具以 MCP 協議暴露，供 MCP client（如 LangChain、Gradio）呼叫。
執行方式（在專案根目錄）：
  uv run python scripts/image_analysis_mcp_server.py
或 stdio 模式（預設）：
  uv run python scripts/image_analysis_mcp_server.py --transport stdio
"""
import sys
from pathlib import Path
from typing import Optional

# 確保專案根目錄在 path 中
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# stdio 模式下 MCP 用 stdout 傳 JSON-RPC，任何 print() 會破壞協定
_original_stdout = sys.stdout
sys.stdout = sys.stderr
try:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        try:
            from fastmcp import FastMCP
        except ImportError:
            from fastmcp.server import FastMCP

    from deep_agent_rag.tools.image_analysis_tool import _analyze_image_internal
finally:
    sys.stdout = _original_stdout

mcp = FastMCP("ImageAnalysis")


@mcp.tool()
def analyze_image_mcp(image_path: str, question: Optional[str] = None) -> str:
    """
    使用多模態 LLM 分析圖片並返回描述。
    image_path: 圖片文件路徑（支持 jpg, png, gif, webp 等）
    question: 可選的特定問題，例如：「這張圖片中有什麼？」；不提供則進行通用分析
    """
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        return _analyze_image_internal(image_path, question=question or None)
    finally:
        sys.stdout = old_stdout


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Image Analysis MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    args = parser.parse_args()
    mcp.run(transport="stdio")
