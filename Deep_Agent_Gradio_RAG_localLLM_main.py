"""
Deep Agent RAG System - 主程序
模組化的深度研究代理系統，整合 RAG、工具調用和本地 MLX 模型

使用方式：
    python Deep_Agent_Gradio_RAG_localLLM_main.py
"""
import gradio as gr

# 導入模組化組件
from deep_agent_rag.rag import init_rag_system
from deep_agent_rag.graph import build_agent_graph
from deep_agent_rag.ui import create_gradio_interface


def main():
    """主函數：初始化系統並啟動 Gradio 界面"""
    print("\n🚀 Deep Research Agent with RAG (Local MLX Edition) 啟動！")
    print("💡 本系統整合了：股票查詢、網路搜尋、PDF 知識庫查詢功能\n")
    print("📦 使用本地 MLX 模型，保護隱私，無需 API 金鑰\n")
    
    # 初始化 RAG 系統
    print("🔧 正在初始化系統組件...")
    rag_retriever = init_rag_system()
    
    # 構建 Agent 圖表
    print("🔧 正在構建 Agent 圖表...")
    graph = build_agent_graph(rag_retriever=rag_retriever)
    print("✅ 系統初始化完成！\n")
    
    # 創建 Gradio 界面
    print("🌐 正在啟動 Gradio 界面...\n")
    demo = create_gradio_interface(graph)
    
    # 啟動 Gradio 服務
    demo.launch(
        server_name="0.0.0.0",  # 允許外部訪問
        server_port=7860,        # 端口號
        share=False,            # 設為 True 可生成公開連結（需要 Gradio 帳號）
        show_error=True,       # 顯示錯誤詳情
        theme=gr.themes.Soft(),  # 主題設置（Gradio 6.0+ 必須在 launch() 中）
        css="""
        .gradio-container {
            font-family: 'Microsoft JhengHei', 'PingFang TC', Arial, sans-serif;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """  # CSS 樣式（Gradio 6.0+ 必須在 launch() 中）
    )


if __name__ == "__main__":
    main()

