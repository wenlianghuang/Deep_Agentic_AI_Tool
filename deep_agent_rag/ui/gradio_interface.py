"""
Gradio 界面模組
提供 Web UI 和流式更新功能
"""
import uuid
import re
import time
import json
import os
from typing import Iterator, Tuple
import gradio as gr
from langchain_core.messages import HumanMessage

# graph 和 rag_retriever 將從外部傳入，不在這裡導入
from ..utils.llm_utils import get_llm_type, is_using_local_llm
from .email_interface import _create_email_interface
from .calendar_interface import _create_calendar_interface
from .private_file_rag_interface import _create_private_file_rag_interface
from .simple_chatbot_interface import create_simple_chatbot_interface
from .image_analysis_interface import _create_image_analysis_interface


def run_research_agent(query: str, graph, thread_id: str = None) -> Iterator[Tuple[str, str, str, str, str]]:
    """
    執行研究代理並實時返回狀態（用於 Gradio 流式更新）
    
    【Gradio 整合】返回生成器，讓 Gradio 可以實時更新 UI
    返回格式: (當前節點狀態, 任務列表, 研究筆記, 最終報告, 警告訊息)
    
    Args:
        query: 用戶輸入的研究問題
        graph: 編譯後的 Agent 圖表
        thread_id: 可選的會話 ID，用於區分不同的查詢會話
    
    Yields:
        Tuple[str, str, str, str, str]: (狀態, 任務列表, 研究筆記, 報告, 警告訊息)
    """
    if not query or not query.strip():
        yield "❌ 請輸入問題", "", "", "", ""
        return
    
    # 檢查 LLM 類型並生成警告訊息
    warning_msg = ""
    if is_using_local_llm():
        warning_msg = "⚠️ **警告：Groq API 額度已用完，已切換到本地 MLX 模型 (Qwen2.5)**\n\n本地模型處理速度可能較慢，請耐心等待。"
    else:
        llm_type = get_llm_type()
        if llm_type == "groq":
            warning_msg = "✅ **當前使用：Groq API**"
        else:
            warning_msg = "ℹ️ **當前使用：本地 MLX 模型 (Qwen2.5)**"
    
    # 生成唯一的 thread_id（如果未提供）
    if not thread_id:
        thread_id = f"deep-research-{uuid.uuid4().hex[:8]}"
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # 初始化完整狀態
    initial_state = {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "tasks": [],
        "completed_tasks": [],
        "research_notes": [],
        "iteration": 0
    }
    
    # 初始化顯示變數
    current_node = "🔄 初始化中..."
    tasks_display = ""
    notes_display = ""
    report_display = ""
    full_report = ""  # 儲存完整報告，用於逐步顯示
    
    # 在開始時顯示警告訊息
    yield current_node, tasks_display, notes_display, report_display, warning_msg
    
    try:
        # 開始執行圖表
        events = graph.stream(
            initial_state,
            config,
            stream_mode="updates"
        )
        
        # 遍歷事件流，實時更新 UI
        for event in events:
            for node, data in event.items():
                # 更新當前節點狀態
                node_emoji = {
                    "planner": "📝",
                    "supervisor": "🎯",
                    "academic_researcher": "📚",
                    "stock_researcher": "📈",
                    "web_researcher": "🌐",
                    "tools_academic": "🔧",
                    "tools_stock": "🔧",
                    "tools_web": "🔧",
                    "note_taking": "📌",
                    "final_report": "📊"
                }.get(node, "🔄")
                
                current_node = f"{node_emoji} 正在執行: {node}"
                
                # 檢查 LLM 狀態變化（可能在執行過程中切換）
                if is_using_local_llm():
                    warning_msg = "⚠️ **警告：Groq API 額度已用完，已切換到本地 MLX 模型 (Qwen2.5)**\n\n本地模型處理速度可能較慢，請耐心等待。"
                else:
                    llm_type = get_llm_type()
                    if llm_type == "groq":
                        warning_msg = "✅ **當前使用：Groq API**"
                    else:
                        warning_msg = "ℹ️ **當前使用：本地 MLX 模型 (Qwen2.5)**"
                
                # 更新任務列表顯示
                if "tasks" in data:
                    tasks = data.get("tasks", [])
                    if tasks:
                        tasks_display = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
                
                # 更新完成任務計數
                if "completed_tasks" in data:
                    completed = data.get("completed_tasks", [])
                    tasks = data.get("tasks", [])
                    if completed and tasks:
                        completed_count = len(completed)
                        total_count = len(tasks)
                        progress = f"\n\n✅ 進度: {completed_count}/{total_count} 個任務已完成"
                        tasks_display = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)]) + progress
                
                # 更新研究筆記顯示（只顯示最近5條，避免過長）
                if "research_notes" in data:
                    notes = data.get("research_notes", [])
                    if notes:
                        # 只取最近5條筆記
                        recent_notes = notes[-5:] if len(notes) > 5 else notes
                        notes_display = "\n\n" + "="*50 + "\n\n".join(recent_notes)
                
                # 檢查是否是最終報告
                if node == "final_report" and "messages" in data:
                    full_report = data["messages"][-1].content
                    current_node = "📊 正在生成報告..."
                    
                    # 按句子分割並逐步顯示（支持中英文標點）
                    # 使用正則表達式分割句子（支持中文標點：。！？和英文標點：. ! ?）
                    sentence_pattern = r'([。！？\n\n]+|\.\s+|!\s+|\?\s+)'
                    parts = re.split(sentence_pattern, full_report)
                    
                    # 重新組合句子（保留標點）
                    sentence_parts = []
                    i = 0
                    while i < len(parts):
                        if i + 1 < len(parts) and re.match(sentence_pattern, parts[i + 1]):
                            # 句子 + 標點
                            sentence_parts.append(parts[i] + parts[i + 1])
                            i += 2
                        else:
                            # 單獨的句子或標點
                            if parts[i].strip():
                                sentence_parts.append(parts[i])
                            i += 1
                    
                    # 如果分割失敗，使用簡單的字符塊方式
                    if not sentence_parts or len(sentence_parts) == 1:
                        # 按字符塊逐步顯示（每20個字符）
                        chunk_size = 20
                        accumulated_text = ""
                        for i in range(0, len(full_report), chunk_size):
                            accumulated_text = full_report[:i + chunk_size]
                            report_display = accumulated_text
                            yield current_node, tasks_display, notes_display, report_display, warning_msg
                            time.sleep(0.03)  # 每塊之間的延遲（30毫秒）
                    else:
                        # 逐步顯示每個句子
                        accumulated_text = ""
                        for sentence in sentence_parts:
                            accumulated_text += sentence
                            report_display = accumulated_text
                            yield current_node, tasks_display, notes_display, report_display, warning_msg
                            time.sleep(0.1)  # 每句之間的延遲（100毫秒）
                    
                    # 確保完整報告顯示
                    report_display = full_report
                    current_node = "✅ 報告生成完成！"
                    yield current_node, tasks_display, notes_display, report_display, warning_msg
                    continue  # 跳過後面的 yield，避免重複
                
                # 實時返回狀態（讓 Gradio 更新 UI）
                yield current_node, tasks_display, notes_display, report_display, warning_msg
        
        # 最終狀態
        yield "✅ 研究完成！", tasks_display, notes_display, report_display, warning_msg
        
    except Exception as e:
        error_msg = f"❌ 發生錯誤: {str(e)}"
        print(f"錯誤詳情: {e}")
        import traceback
        traceback.print_exc()
        # 檢查是否是因為 Groq 額度問題
        if is_using_local_llm():
            warning_msg = "⚠️ **警告：Groq API 額度已用完，已切換到本地 MLX 模型 (Qwen2.5)**\n\n本地模型處理速度可能較慢，請耐心等待。"
        yield error_msg, tasks_display, notes_display, report_display, warning_msg


def create_gradio_interface(graph):
    """
    創建 Gradio 界面
    
    【Gradio 6.x 兼容】使用最新的 Gradio API 創建美觀的 Web 界面
    """
    with gr.Blocks(
        title="Deep Research Agent with RAG (Local MLX)"
    ) as demo:
        # 標題區域
        gr.Markdown(
            """
            <div class="header">
            <h1>🚀 Deep Research Agent with RAG</h1>
            <p><strong>功能特色：</strong></p>
            <p>💬 簡單聊天機器人 | 🔍 Deep Research Agent | 📧 智能郵件助手 | 📅 智能行事曆管理 | 📄 私有文件 RAG 問答 | 🖼️ 智能圖片分析</p>
            <p><strong>智能規劃：</strong> 系統會根據問題類型自動選擇合適的研究工具</p>
            <p><strong>本地模型：</strong> 使用 MLX 本地模型，保護隱私，無需 API 金鑰</p>
            </div>
            """,
            elem_classes=["header"]
        )
        
        # 使用 Tabs 分離不同功能
        with gr.Tabs() as tabs:
            # Tab 1: Simple Chatbot
            with gr.Tab("💬 Simple Chatbot"):
                _create_simple_chatbot_tab()
            
            # Tab 2: Deep Research Agent
            with gr.Tab("🔍 Deep Research Agent"):
                _create_research_interface(graph)
            
            # Tab 3: Email Tool
            with gr.Tab("📧 Email Tool"):
                _create_email_interface()
            
            # Tab 4: Calendar Tool
            with gr.Tab("📅 Calendar Tool"):
                _create_calendar_interface()
            
            # Tab 5: Private File RAG
            with gr.Tab("📚 Private File RAG"):
                _create_private_file_rag_interface()
            
            # Tab 6: Image Analysis
            with gr.Tab("🖼️ Image Analysis"):
                _create_image_analysis_interface()
    
    return demo


def _create_research_interface(graph):
    """創建 Deep Research Agent 界面"""
    with gr.Row():
        with gr.Column(scale=2):
            # 輸入區域
            query_input = gr.Textbox(
                label="📝 請輸入您的研究問題",
                placeholder="例如：說明Tree of Thoughts，並深度比較他跟Chain of Thought的差距在哪裡？",
                lines=3,
                value="比較微軟(MSFT)和谷歌(GOOGL)在AI領域的佈局，並結合 Tree of Thoughts 論文中的方法論進行分析"
            )
            
            # 按鈕區域
            with gr.Row():
                submit_btn = gr.Button("🔍 開始研究", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️ 清除", variant="secondary", scale=1)
            
            # 狀態顯示
            status_display = gr.Textbox(
                label="📊 當前狀態",
                value="等待開始...",
                interactive=False,
                lines=2
            )
            
            # 警告訊息顯示
            warning_display = gr.Markdown(
                value="",
                elem_classes=["warning-box"]
            )
        
        with gr.Column(scale=1):
            # 任務列表
            tasks_display = gr.Textbox(
                label="📋 研究任務列表",
                lines=12,
                interactive=False
            )
    
    with gr.Row():
        # 研究筆記（實時更新）
        notes_display = gr.Textbox(
            label="📌 研究筆記（實時更新）",
            lines=15,
            interactive=False
        )
    
    with gr.Row():
        # 最終報告
        report_display = gr.Textbox(
            label="📄 最終深度報告",
            lines=20,
            interactive=False
        )
    
    # 事件處理函數
    def process_query(query):
        """處理查詢並返回流式更新"""
        if not query or not query.strip():
            return "❌ 請輸入問題", "", "", "", ""
        
        # 使用生成器函數實時更新（Gradio 6.x 支持流式輸出）
        for status, tasks, notes, report, warning in run_research_agent(query, graph):
            yield status, tasks, notes, report, warning
    
    def clear_all():
        """清除所有輸入和輸出"""
        # 檢查當前 LLM 狀態
        warning_msg = ""
        if is_using_local_llm():
            warning_msg = "⚠️ **警告：Groq API 額度已用完，已切換到本地 MLX 模型 (Qwen2.5)**\n\n本地模型處理速度可能較慢，請耐心等待。"
        else:
            llm_type = get_llm_type()
            if llm_type == "groq":
                warning_msg = "✅ **當前使用：Groq API**"
            else:
                warning_msg = "ℹ️ **當前使用：本地 MLX 模型 (Qwen2.5)**"
        return "", "", "", "", "等待開始...", warning_msg
    
    # 綁定事件
    submit_btn.click(
        fn=process_query,
        inputs=query_input,
        outputs=[status_display, tasks_display, notes_display, report_display, warning_display]
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[query_input, tasks_display, notes_display, report_display, status_display, warning_display]
    )
    
    # 初始化時顯示當前 LLM 狀態
    def get_initial_warning():
        warning_msg = ""
        if is_using_local_llm():
            warning_msg = "⚠️ **警告：Groq API 額度已用完，已切換到本地 MLX 模型 (Qwen2.5)**\n\n本地模型處理速度可能較慢，請耐心等待。"
        else:
            llm_type = get_llm_type()
            if llm_type == "groq":
                warning_msg = "✅ **當前使用：Groq API**"
            else:
                warning_msg = "ℹ️ **當前使用：本地 MLX 模型 (Qwen2.5)**"
        return warning_msg
    
    # 在界面載入時顯示初始警告
    warning_display.value = get_initial_warning()
    
    # 示例問題（快速測試）
    gr.Examples(
        examples=[
            "說明Tree of Thoughts，並深度比較他跟Chain of Thought的差距在哪裡？",
            "比較微軟(MSFT)和谷歌(GOOGL)在AI領域的佈局",
            "分析 Tree of Thoughts 方法的優缺點和應用場景",
            "查詢蘋果(AAPL)的財務狀況和近期動態"
        ],
        inputs=query_input
    )
    
    # 頁腳說明
    gr.Markdown(
        """
        ---
        **使用說明：**
        1. 在輸入框中輸入您的研究問題
        2. 點擊「開始研究」按鈕
        3. 系統會自動規劃研究步驟並執行
        4. 您可以實時查看任務進度、研究筆記和最終報告
        5. 點擊「清除」按鈕可以重置所有內容
        """
    )


def _create_simple_chatbot_tab():
    """創建簡單聊天機器人標籤頁內容"""
    from .simple_chatbot_interface import chat_with_llm_streaming, get_llm_status
    from ..memory.chat_memory import save_conversation_summary, clear_chat_memory

    # 標題說明
    gr.Markdown(
        """
        ### 💬 Simple Chatbot - 純粹的對話體驗
        
        這是一個簡單的聊天機器人，不包含 RAG、Deep AI Agent 等複雜功能。
        只專注於自然對話，讓您與 AI 輕鬆交流。
        """
    )
    
    # LLM 狀態顯示
    llm_status = gr.Markdown(
        value=get_llm_status(),
        elem_classes=["warning-box"]
    )
    
    # Guardrails 與長期記憶開關
    with gr.Row():
        enable_guardrails_checkbox = gr.Checkbox(
            label="🛡️ 啟用 Guardrails 內容過濾",
            value=True,
            info="啟用後將檢查輸入和輸出內容，阻擋敏感話題"
        )
        enable_long_term_memory_checkbox = gr.Checkbox(
            label="🧠 啟用長期記憶（Chroma）",
            value=True,
            info="清除對話時會儲存摘要；下次提問會自動檢索相關記憶"
        )

    # 系統提示詞設定
    with gr.Accordion("⚙️ 進階設定", open=False):
        system_prompt = gr.Textbox(
            label="系統提示詞 (System Prompt)",
            value="你是一個有幫助的AI助手。請用繁體中文回答問題。",
            lines=3,
            placeholder="設定 AI 的角色和行為方式..."
        )
        
        gr.Markdown(
            """
            **提示詞範例：**
            - 專業助手：「你是一位專業的技術顧問，擅長解釋複雜的技術概念。」
            - 創意寫作：「你是一位富有創意的作家，擅長寫作故事和詩歌。」
            - 學習輔導：「你是一位耐心的老師，擅長用簡單的方式解釋複雜的概念。」
            """
        )
    
    # 聊天界面
    chatbot = gr.Chatbot(
        label="對話記錄",
        height=400,
        show_label=True
    )
    
    # 輸入區域
    msg = gr.Textbox(
        label="訊息",
        placeholder="在這裡輸入您的訊息...",
        lines=2,
        show_label=False
    )
    
    # 控制按鈕
    with gr.Row():
        submit_btn = gr.Button("📤 發送", variant="primary")
        clear_btn = gr.Button("🗑️ 清除對話", variant="secondary")
        clear_memory_btn = gr.Button("🧹 清空長期記憶", variant="secondary")
        refresh_status_btn = gr.Button("🔄 更新狀態", variant="secondary")
    memory_action_status = gr.Markdown(value="", visible=True)
    
    # 示例問題
    gr.Examples(
        examples=[
            "你好！請介紹一下你自己。",
            "請幫我解釋什麼是機器學習？",
            "能給我一些學習 Python 的建議嗎？",
            "請用簡單的方式解釋量子計算。",
            "寫一首關於春天的短詩。"
        ],
        inputs=msg,
        label="💡 快速試用範例"
    )
    
    # 事件綁定
    def save_then_clear(history, enable_long_term_memory):
        """清除對話前先將當前對話摘要寫入 Chroma 長期記憶，再清空畫面。"""
        if enable_long_term_memory and history and len(history) > 0:
            save_conversation_summary(history, user_id="default")
        return [], ""

    def do_clear_memory():
        """清空 Chroma 長期記憶並回傳狀態訊息"""
        clear_chat_memory()
        return "✅ 已清空長期記憶（Chroma 對話摘要已刪除）"

    def refresh_status():
        """更新 LLM 狀態"""
        return get_llm_status()

    _chat_inputs = [msg, chatbot, system_prompt, enable_guardrails_checkbox, enable_long_term_memory_checkbox]
    # 發送消息事件
    msg.submit(
        fn=chat_with_llm_streaming,
        inputs=_chat_inputs,
        outputs=[chatbot],
        queue=True
    ).then(
        fn=lambda: "",
        outputs=[msg],
        queue=False
    )

    submit_btn.click(
        fn=chat_with_llm_streaming,
        inputs=_chat_inputs,
        outputs=[chatbot],
        queue=True
    ).then(
        fn=lambda: "",
        outputs=[msg],
        queue=False
    )

    clear_btn.click(
        fn=save_then_clear,
        inputs=[chatbot, enable_long_term_memory_checkbox],
        outputs=[chatbot, msg],
        queue=False
    )

    clear_memory_btn.click(
        fn=do_clear_memory,
        outputs=[memory_action_status],
        queue=False
    )
    
    refresh_status_btn.click(
        fn=refresh_status,
        outputs=[llm_status],
        queue=False
    )



    
    