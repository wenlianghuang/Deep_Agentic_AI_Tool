"""
Gradio 界面模組
提供 Web UI 和流式更新功能
"""
import uuid
import re
import time
from typing import Iterator, Tuple
import gradio as gr
from langchain_core.messages import HumanMessage

# graph 和 rag_retriever 將從外部傳入，不在這裡導入
from ..utils.llm_utils import get_llm_type, is_using_local_llm


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
                    "research_agent": "🕵️",
                    "tools": "🔧",
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
            <h1>🚀 Deep Research Agent with RAG (Local MLX)</h1>
            <p><strong>功能特色：</strong></p>
            <p>📊 股票資訊查詢 | 🌐 網路搜尋 | 📚 PDF 知識庫查詢（Tree of Thoughts 論文）| 📧 智能郵件助手</p>
            <p><strong>智能規劃：</strong> 系統會根據問題類型自動選擇合適的研究工具</p>
            <p><strong>本地模型：</strong> 使用 MLX 本地模型，保護隱私，無需 API 金鑰</p>
            </div>
            """,
            elem_classes=["header"]
        )
        
        # 使用 Tabs 分離不同功能
        with gr.Tabs() as tabs:
            # Tab 1: Deep Research Agent
            with gr.Tab("🔍 Deep Research Agent"):
                _create_research_interface(graph)
            
            # Tab 2: Email Tool
            with gr.Tab("📧 Email Tool"):
                _create_email_interface()
    
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


def _create_email_interface():
    """創建 Email Tool 界面"""
    from ..agents.email_agent import generate_email_draft, send_email_draft
    from ..config import EMAIL_SENDER
    
    gr.Markdown(
        f"""
        ### 📧 智能郵件助手
        
        使用 AI 根據您的關鍵提示自動生成專業郵件草稿，您可以在發送前檢查和修改。
        
        **寄件者：** {EMAIL_SENDER}
        
        **使用方式：**
        1. 在下方輸入郵件提示（例如："寫一封感謝信"、"邀請參加會議"等）
        2. 輸入收件人郵箱地址
        3. 點擊「生成郵件草稿」按鈕
        4. 檢查並修改生成的郵件內容（特別是簽名部分）
        5. 確認無誤後點擊「發送郵件」按鈕
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # 郵件提示輸入
            email_prompt_input = gr.Textbox(
                label="📝 郵件提示",
                placeholder="例如：寫一封感謝信，感謝對方在項目中的幫助",
                lines=5,
                value="寫一封專業的郵件，介紹我們的 AI 產品"
            )
            
            # 收件人輸入
            recipient_input = gr.Textbox(
                label="📮 收件人郵箱",
                placeholder="recipient@example.com",
                lines=1
            )
            
            # 按鈕
            with gr.Row():
                generate_draft_btn = gr.Button("📝 生成郵件草稿", variant="primary", scale=1)
                clear_email_btn = gr.Button("🗑️ 清除", variant="secondary", scale=1)
            
            # 狀態顯示
            email_status_display = gr.Textbox(
                label="📊 狀態",
                value="等待操作...",
                interactive=False,
                lines=2
            )
        
        with gr.Column(scale=1):
            # 郵件主題（可編輯）
            email_subject_input = gr.Textbox(
                label="📌 郵件主題",
                placeholder="郵件主題將在這裡顯示，您可以編輯",
                lines=1,
                interactive=True
            )
            
            # 郵件正文（可編輯）
            email_body_input = gr.Textbox(
                label="📄 郵件正文（可編輯）",
                placeholder="郵件內容將在這裡顯示，您可以編輯",
                lines=15,
                interactive=True
            )
            
            # 發送按鈕
            send_draft_btn = gr.Button("📧 發送郵件", variant="primary", scale=1)
            
            # 發送結果顯示
            email_result_display = gr.Textbox(
                label="📊 發送結果",
                lines=5,
                interactive=False
            )
    
    # 事件處理函數
    def generate_draft(prompt, recipient):
        """生成郵件草稿"""
        if not prompt or not prompt.strip():
            return "❌ 請輸入郵件提示", "", "", "❌ 請輸入郵件提示"
        
        if not recipient or not recipient.strip():
            return "❌ 請輸入收件人郵箱", "", "", "❌ 請輸入收件人郵箱"
        
        # 驗證郵箱格式（簡單驗證）
        if "@" not in recipient or "." not in recipient.split("@")[1]:
            return "❌ 郵箱格式不正確", "", "", "❌ 郵箱格式不正確，請輸入有效的郵箱地址"
        
        try:
            status_msg = "🔄 正在生成郵件草稿..."
            
            # 生成郵件草稿
            subject, body, status = generate_email_draft(prompt, recipient.strip())
            
            if subject and body:
                return status, subject, body, ""
            else:
                return status, "", "", status
        except Exception as e:
            error_msg = f"❌ 發生錯誤：{str(e)}"
            print(f"Email Tool 錯誤：{e}")
            import traceback
            traceback.print_exc()
            return "❌ 發生錯誤", "", "", error_msg
    
    def send_draft(recipient, subject, body):
        """發送已編輯的郵件草稿"""
        if not recipient or not recipient.strip():
            return "❌ 請輸入收件人郵箱", "❌ 請輸入收件人郵箱"
        
        if not subject or not subject.strip():
            return "❌ 請輸入郵件主題", "❌ 請輸入郵件主題"
        
        if not body or not body.strip():
            return "❌ 請輸入郵件內容", "❌ 請輸入郵件內容"
        
        # 驗證郵箱格式
        if "@" not in recipient or "." not in recipient.split("@")[1]:
            return "❌ 郵箱格式不正確", "❌ 郵箱格式不正確，請輸入有效的郵箱地址"
        
        try:
            status_msg = "🔄 正在發送郵件..."
            
            # 發送郵件
            result = send_email_draft(recipient.strip(), subject.strip(), body.strip())
            
            return "✅ 郵件已發送", result
        except Exception as e:
            error_msg = f"❌ 發送郵件時發生錯誤：{str(e)}"
            print(f"Email Tool 錯誤：{e}")
            import traceback
            traceback.print_exc()
            return "❌ 發生錯誤", error_msg
    
    def clear_email():
        """清除郵件相關輸入和輸出"""
        return "", "", "等待操作...", "", "", ""
    
    # 綁定事件
    generate_draft_btn.click(
        fn=generate_draft,
        inputs=[email_prompt_input, recipient_input],
        outputs=[email_status_display, email_subject_input, email_body_input, email_result_display]
    )
    
    send_draft_btn.click(
        fn=send_draft,
        inputs=[recipient_input, email_subject_input, email_body_input],
        outputs=[email_status_display, email_result_display]
    )
    
    clear_email_btn.click(
        fn=clear_email,
        outputs=[email_prompt_input, recipient_input, email_status_display, email_subject_input, email_body_input, email_result_display]
    )
    
    # 示例
    gr.Examples(
        examples=[
            ["寫一封感謝信，感謝對方在項目中的幫助和支持", "example@example.com"],
            ["邀請參加下週的產品發布會", "colleague@company.com"],
            ["詢問項目進度並提供更新", "partner@partner.com"],
            ["發送會議記錄和後續行動項目", "team@company.com"]
        ],
        inputs=[email_prompt_input, recipient_input]
    )
    
    # 頁腳說明
    gr.Markdown(
        f"""
        ---
        **注意事項：**
        1. 使用 Gmail API 發送郵件，避免被歸類為垃圾郵件
        2. 首次使用需要在專案根目錄放置 `credentials.json`（從 Google Cloud Console 下載的 OAuth2 憑證）
        3. 首次運行時會自動開啟瀏覽器進行授權，授權後會生成 `token.json` 文件
        4. 郵件內容由 AI 自動生成，請在發送前檢查結果
        5. 寄件者固定為：{EMAIL_SENDER}
        
        **設置步驟：**
        - 前往 [Google Cloud Console](https://console.cloud.google.com/) 創建專案
        - 啟用 Gmail API
        - 創建 OAuth2 憑證並下載為 `credentials.json`
        - 將 `credentials.json` 放在專案根目錄
        """
    )

