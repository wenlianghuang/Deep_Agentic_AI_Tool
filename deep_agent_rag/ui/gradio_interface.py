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
            <p>📊 股票資訊查詢 | 🌐 網路搜尋 | 📚 PDF 知識庫查詢（Tree of Thoughts 論文）| 📧 智能郵件助手 | 📅 智能行事曆管理</p>
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
            
            # Tab 3: Calendar Tool
            with gr.Tab("📅 Calendar Tool"):
                _create_calendar_interface()
    
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
        2. 輸入收件人 Gmail 郵箱地址（僅支援 @gmail.com 或 @googlemail.com）
        3. 點擊「生成郵件草稿」按鈕
        4. 查看 AI 反思評估結果和改進建議（如有）
        5. 檢查並修改生成的郵件內容（特別是簽名部分）
        6. 確認無誤後點擊「發送郵件」按鈕
        
        **✨ 新功能：AI 迭代反思評估**
        - 系統會自動進行多輪反思評估（最多 3 輪）
        - 每輪評估後，如果有改進建議，會自動生成改進版本
        - 改進後的版本會再次評估，直到 AI 認為滿意為止
        - 您可以看到完整的反思過程和每輪的改進建議
        
        **注意：此工具僅支援 Gmail 郵箱，收件人必須使用 Gmail 郵箱地址。**
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
                label="📮 收件人郵箱（僅支援 Gmail）",
                placeholder="recipient@gmail.com",
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
            
            # 反思結果顯示
            email_reflection_display = gr.Textbox(
                label="🔍 AI 反思評估",
                value="等待生成郵件...",
                interactive=False,
                lines=8,
                visible=True
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
        """生成郵件草稿（包含反思功能）"""
        if not prompt or not prompt.strip():
            return "❌ 請輸入郵件提示", "", "", "❌ 請輸入郵件提示", "❌ 請輸入郵件提示"
        
        if not recipient or not recipient.strip():
            return "❌ 請輸入收件人郵箱", "", "", "❌ 請輸入收件人郵箱", "❌ 請輸入收件人郵箱"
        
        # 驗證郵箱格式和 Gmail 限制
        if "@" not in recipient or "." not in recipient.split("@")[1]:
            return "❌ 郵箱格式不正確", "", "", "❌ 郵箱格式不正確，請輸入有效的郵箱地址", "❌ 郵箱格式不正確，請輸入有效的郵箱地址"
        
        # 驗證是否為 Gmail 郵箱
        recipient_lower = recipient.strip().lower()
        if not (recipient_lower.endswith("@gmail.com") or recipient_lower.endswith("@googlemail.com")):
            return "❌ 僅支援 Gmail 郵箱", "", "", "❌ 此工具僅支援 Gmail 郵箱（@gmail.com 或 @googlemail.com），請輸入 Gmail 郵箱地址", "❌ 此工具僅支援 Gmail 郵箱（@gmail.com 或 @googlemail.com），請輸入 Gmail 郵箱地址"
        
        try:
            status_msg = "🔄 正在生成郵件草稿..."
            reflection_msg = "🔄 正在生成郵件草稿..."
            
            # 生成郵件草稿（包含反思功能，會自動改進）
            subject, body, status, reflection_result, was_improved = generate_email_draft(
                prompt, recipient.strip(), enable_reflection=True
            )
            
            if subject and body:
                # 格式化反思結果顯示
                if reflection_result:
                    # 計算反思輪數
                    reflection_count = reflection_result.count("【第") if "【第" in reflection_result else 0
                    
                    if was_improved:
                        if reflection_count > 1:
                            reflection_display = (
                                f"🔍 **AI 迭代反思評估結果**（共 {reflection_count} 輪）\n\n"
                                f"{reflection_result}\n\n"
                                f"✨ **已自動應用改進建議，經過 {reflection_count} 輪優化，當前顯示的是最終優化版本**"
                            )
                        else:
                            reflection_display = (
                                f"🔍 **AI 反思評估結果**\n\n"
                                f"{reflection_result}\n\n"
                                f"✨ **已自動應用改進建議，當前顯示的是優化後的版本**"
                            )
                    else:
                        reflection_display = (
                            f"🔍 **AI 反思評估結果**\n\n"
                            f"{reflection_result}\n\n"
                            f"✅ **郵件質量良好，無需改進**"
                        )
                else:
                    reflection_display = "⚠️ 反思功能未返回結果"
                
                return status, subject, body, "", reflection_display
            else:
                return status, "", "", status, "❌ 生成失敗，無法進行反思評估"
        except Exception as e:
            error_msg = f"❌ 發生錯誤：{str(e)}"
            print(f"Email Tool 錯誤：{e}")
            import traceback
            traceback.print_exc()
            return "❌ 發生錯誤", "", "", error_msg, f"❌ 發生錯誤：{str(e)}"
    
    def send_draft(recipient, subject, body):
        """發送已編輯的郵件草稿"""
        if not recipient or not recipient.strip():
            return "❌ 請輸入收件人郵箱", "❌ 請輸入收件人郵箱"
        
        if not subject or not subject.strip():
            return "❌ 請輸入郵件主題", "❌ 請輸入郵件主題"
        
        if not body or not body.strip():
            return "❌ 請輸入郵件內容", "❌ 請輸入郵件內容"
        
        # 驗證郵箱格式和 Gmail 限制
        if "@" not in recipient or "." not in recipient.split("@")[1]:
            return "❌ 郵箱格式不正確", "❌ 郵箱格式不正確，請輸入有效的郵箱地址"
        
        # 驗證是否為 Gmail 郵箱
        recipient_lower = recipient.strip().lower()
        if not (recipient_lower.endswith("@gmail.com") or recipient_lower.endswith("@googlemail.com")):
            return "❌ 僅支援 Gmail 郵箱", "❌ 此工具僅支援 Gmail 郵箱（@gmail.com 或 @googlemail.com），請輸入 Gmail 郵箱地址"
        
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
        return "", "", "等待操作...", "", "", "等待生成郵件..."
    
    # 綁定事件
    generate_draft_btn.click(
        fn=generate_draft,
        inputs=[email_prompt_input, recipient_input],
        outputs=[email_status_display, email_subject_input, email_body_input, email_result_display, email_reflection_display]
    )
    
    send_draft_btn.click(
        fn=send_draft,
        inputs=[recipient_input, email_subject_input, email_body_input],
        outputs=[email_status_display, email_result_display]
    )
    
    clear_email_btn.click(
        fn=clear_email,
        outputs=[email_prompt_input, recipient_input, email_status_display, email_subject_input, email_body_input, email_result_display, email_reflection_display]
    )
    
    # 示例
    gr.Examples(
        examples=[
            ["寫一封感謝信，感謝對方在項目中的幫助和支持", "example@gmail.com"],
            ["邀請參加下週的產品發布會", "colleague@gmail.com"],
            ["詢問項目進度並提供更新", "partner@gmail.com"],
            ["發送會議記錄和後續行動項目", "team@gmail.com"]
        ],
        inputs=[email_prompt_input, recipient_input]
    )
    
    # 頁腳說明
    gr.Markdown(
        f"""
        ---
        **注意事項：**
        1. 使用 Gmail API 發送郵件，避免被歸類為垃圾郵件
        2. **此工具僅支援 Gmail 郵箱，收件人必須使用 @gmail.com 或 @googlemail.com 結尾的郵箱地址**
        3. 首次使用需要在專案根目錄放置 `credentials.json`（從 Google Cloud Console 下載的 OAuth2 憑證）
        4. 首次運行時會自動開啟瀏覽器進行授權，授權後會生成 `token.json` 文件
        5. 郵件內容由 AI 自動生成，請在發送前檢查結果
        6. 寄件者固定為：{EMAIL_SENDER}
        
        **設置步驟：**
        - 前往 [Google Cloud Console](https://console.cloud.google.com/) 創建專案
        - 啟用 Gmail API
        - 創建 OAuth2 憑證並下載為 `credentials.json`
        - 將 `credentials.json` 放在專案根目錄
        """
    )


def _create_calendar_interface():
    """創建 Calendar Tool 界面"""
    from ..agents.calendar_agent import generate_calendar_draft, create_calendar_draft
    from datetime import datetime, timedelta
    
    gr.Markdown(
        """
        ### 📅 智能行事曆管理助手
        
        使用 AI 根據您的完整提示自動生成行事曆事件草稿，您可以在創建前檢查和修改。
        
        **使用方式：**
        1. 在下方輸入完整的事件提示，包含：事件、日期、時間、地點、參與者
           （例如："明天下午2點團隊會議，討論項目進度，地點在會議室A，參與者包括john@example.com"）
        2. 點擊「生成事件草稿」按鈕
        3. 如果有缺失的資訊（如時間），系統會顯示下拉選單讓您選擇
        4. 檢查並修改生成的事件內容
        5. 確認無誤後點擊「創建事件」按鈕
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # 單一 prompt 輸入
            calendar_prompt_input = gr.Textbox(
                label="📝 事件提示（包含事件、日期、時間、地點、參與者）",
                placeholder="例如：明天下午2點團隊會議，討論項目進度，地點在會議室A，參與者包括john@example.com和mary@example.com",
                lines=5,
                value=""
            )
            
            # 按鈕
            with gr.Row():
                generate_draft_btn = gr.Button("📝 生成事件草稿", variant="primary", scale=1)
                clear_calendar_btn = gr.Button("🗑️ 清除", variant="secondary", scale=1)
            
            # 狀態顯示
            calendar_status_display = gr.Textbox(
                label="📊 狀態",
                value="等待操作...",
                interactive=False,
                lines=2
            )
            
            # 缺失資訊的補充區域（動態顯示）
            missing_info_group = gr.Group(visible=False)
            with missing_info_group:
                gr.Markdown("**⚠️ 請補充以下缺失的資訊：**")
                
                # 日期選擇（如果缺失）
                missing_date_display = gr.Dropdown(
                    label="📆 選擇日期",
                    choices=[],
                    visible=False,
                    interactive=True
                )
                
                # 時間選擇（如果缺失）
                missing_time_display = gr.Dropdown(
                    label="🕐 選擇時間",
                    choices=[],
                    visible=False,
                    interactive=True
                )
                
                fill_missing_btn = gr.Button("✅ 確認補充資訊", variant="primary", visible=False)
            
            # 隱藏狀態變數，用於存儲 event_dict
            event_dict_storage = gr.State(value={})
        
        with gr.Column(scale=1):
            # 事件詳情顯示和編輯區域
            event_summary_display = gr.Textbox(
                label="📌 事件標題",
                placeholder="事件標題將在這裡顯示",
                lines=1,
                interactive=True
            )
            
            event_start_display = gr.Textbox(
                label="🕐 開始時間",
                placeholder="開始時間將在這裡顯示（格式: YYYY-MM-DDTHH:MM:SS+08:00）",
                lines=1,
                interactive=True
            )
            
            event_end_display = gr.Textbox(
                label="🕐 結束時間",
                placeholder="結束時間將在這裡顯示（格式: YYYY-MM-DDTHH:MM:SS+08:00）",
                lines=1,
                interactive=True
            )
            
            event_description_display = gr.Textbox(
                label="📄 事件描述（可編輯）",
                placeholder="事件描述將在這裡顯示，您可以編輯",
                lines=6,
                interactive=True
            )
            
            event_location_display = gr.Textbox(
                label="📍 地點（可編輯）",
                placeholder="事件地點將在這裡顯示，您可以編輯",
                lines=1,
                interactive=True
            )
            
            event_attendees_display = gr.Textbox(
                label="👥 參與者郵箱（可編輯，多個用逗號分隔）",
                placeholder="參與者郵箱將在這裡顯示，您可以編輯",
                lines=1,
                interactive=True
            )
            
            # 創建按鈕
            create_event_btn = gr.Button("✅ 創建事件", variant="primary", scale=1)
            
            # 操作結果顯示
            calendar_result_display = gr.Textbox(
                label="📊 操作結果",
                lines=8,
                interactive=False
            )
    
    # 生成時間選項（每30分鐘一個選項）
    def generate_time_options():
        """生成時間選項列表"""
        times = []
        for hour in range(24):
            for minute in [0, 30]:
                time_str = f"{hour:02d}:{minute:02d}"
                times.append(time_str)
        return times
    
    # 生成日期選項（今天、明天、後天，以及未來7天）
    def generate_date_options():
        """生成日期選項列表"""
        dates = []
        today = datetime.now()
        date_names = ["今天", "明天", "後天"]
        
        for i in range(3):
            date_obj = today + timedelta(days=i)
            date_str = date_obj.strftime('%Y-%m-%d')
            dates.append(f"{date_names[i]} ({date_str})")
        
        for i in range(3, 7):
            date_obj = today + timedelta(days=i)
            date_str = date_obj.strftime('%Y-%m-%d')
            dates.append(date_str)
        
        return dates
    
    # 事件處理函數
    def generate_draft(prompt):
        """生成行事曆事件草稿"""
        if not prompt or not prompt.strip():
            return (
                "❌ 請輸入事件提示",
                gr.update(visible=False),
                gr.update(visible=False, choices=[]),
                gr.update(visible=False, choices=[]),
                gr.update(visible=False),
                "", "", "", "", "", "", "",
                "❌ 請輸入事件提示"
            )
        
        try:
            status_msg = "🔄 正在生成事件草稿..."
            
            # 生成事件草稿
            event_dict, status, missing_info = generate_calendar_draft(prompt.strip())
            
            if not event_dict:
                return (
                    status,
                    gr.update(visible=False),
                    gr.update(visible=False, choices=[]),
                    gr.update(visible=False, choices=[]),
                    gr.update(visible=False),
                    "", "", "", "", "", "", "",
                    status
                )
            
            # 檢查是否有缺失資訊
            has_missing = bool(missing_info)
            
            if has_missing:
                # 顯示缺失資訊區域
                date_visible = missing_info.get("date", False)
                time_visible = missing_info.get("time", False)
                
                date_choices = generate_date_options() if date_visible else []
                time_choices = generate_time_options() if time_visible else []
                
                return (
                    status,
                    gr.update(visible=True),  # 顯示缺失資訊區域
                    gr.update(visible=date_visible, choices=date_choices, value=date_choices[0] if date_choices else None),
                    gr.update(visible=time_visible, choices=time_choices, value=time_choices[0] if time_choices else None),
                    gr.update(visible=True),  # 顯示確認按鈕
                    event_dict.get("summary", ""),
                    event_dict.get("start_datetime", ""),
                    event_dict.get("end_datetime", ""),
                    event_dict.get("description", ""),
                    event_dict.get("location", ""),
                    event_dict.get("attendees", ""),
                    event_dict,  # 傳遞完整的事件字典以便後續使用
                    ""
                )
            else:
                # 沒有缺失資訊，直接顯示結果
                return (
                    status,
                    gr.update(visible=False),
                    gr.update(visible=False, choices=[]),
                    gr.update(visible=False, choices=[]),
                    gr.update(visible=False),
                    event_dict.get("summary", ""),
                    event_dict.get("start_datetime", ""),
                    event_dict.get("end_datetime", ""),
                    event_dict.get("description", ""),
                    event_dict.get("location", ""),
                    event_dict.get("attendees", ""),
                    event_dict,
                    ""
                )
        except Exception as e:
            error_msg = f"❌ 發生錯誤：{str(e)}"
            print(f"Calendar Tool 錯誤：{e}")
            import traceback
            traceback.print_exc()
            return (
                "❌ 發生錯誤",
                gr.update(visible=False),
                gr.update(visible=False, choices=[]),
                gr.update(visible=False, choices=[]),
                gr.update(visible=False),
                "", "", "", "", "", "", {},
                error_msg
            )
    
    def fill_missing_info(event_dict_storage, selected_date, selected_time):
        """填充缺失的資訊"""
        if not event_dict_storage:
            return (
                "❌ 沒有事件資料",
                gr.update(visible=False),
                gr.update(visible=False, choices=[]),
                gr.update(visible=False, choices=[]),
                gr.update(visible=False),
                "", "", "", "", "", "",
                {}
            )
        
        # 更新日期和時間
        if selected_date:
            # 從選項中提取日期字串（例如："明天 (2026-01-25)" -> "2026-01-25"）
            if "(" in selected_date:
                date_str = selected_date.split("(")[1].split(")")[0]
            else:
                date_str = selected_date
        else:
            date_str = event_dict_storage.get("date", "今天")
        
        if selected_time:
            time_str = selected_time
        else:
            time_str = "09:00"  # 預設時間
        
        # 重新解析日期和時間
        from ..agents.calendar_agent import parse_datetime
        start_datetime, end_datetime = parse_datetime(date_str, time_str)
        
        # 更新事件字典
        event_dict_storage["start_datetime"] = start_datetime
        event_dict_storage["end_datetime"] = end_datetime
        
        return (
            "✅ 資訊已補充，請檢查並創建事件",
            gr.update(visible=False),  # 隱藏缺失資訊區域
            gr.update(visible=False, choices=[]),
            gr.update(visible=False, choices=[]),
            gr.update(visible=False),
            event_dict_storage.get("summary", ""),
            start_datetime,
            end_datetime,
            event_dict_storage.get("description", ""),
            event_dict_storage.get("location", ""),
            event_dict_storage.get("attendees", ""),
            event_dict_storage
        )
    
    def create_event(summary, start_datetime, end_datetime, description, location, attendees):
        """創建行事曆事件"""
        if not summary or not summary.strip():
            return "❌ 請輸入事件標題", "❌ 請輸入事件標題"
        
        if not start_datetime or not start_datetime.strip():
            return "❌ 請輸入開始時間", "❌ 請輸入開始時間"
        
        if not end_datetime or not end_datetime.strip():
            return "❌ 請輸入結束時間", "❌ 請輸入結束時間"
        
        try:
            status_msg = "🔄 正在創建事件..."
            
            # 構建事件字典
            event_dict = {
                "summary": summary.strip(),
                "start_datetime": start_datetime.strip(),
                "end_datetime": end_datetime.strip(),
                "description": description.strip() if description else "",
                "location": location.strip() if location else "",
                "attendees": attendees.strip() if attendees else "",
                "timezone": "Asia/Taipei"
            }
            
            # 創建事件
            result = create_calendar_draft(event_dict)
            
            return "✅ 事件已創建", result
        except Exception as e:
            error_msg = f"❌ 創建事件時發生錯誤：{str(e)}"
            print(f"Calendar Tool 錯誤：{e}")
            import traceback
            traceback.print_exc()
            return "❌ 發生錯誤", error_msg
    
    def clear_calendar():
        """清除行事曆相關輸入和輸出"""
        return (
            "",  # prompt
            "等待操作...",  # status
            gr.update(visible=False),  # missing_info_group
            gr.update(visible=False, choices=[]),  # missing_date
            gr.update(visible=False, choices=[]),  # missing_time
            gr.update(visible=False),  # fill_missing_btn
            "", "", "", "", "", "",  # event fields
            {},  # event_dict_storage
            ""  # result
        )
    
    # 綁定事件
    generate_draft_btn.click(
        fn=generate_draft,
        inputs=[calendar_prompt_input],
        outputs=[
            calendar_status_display,
            missing_info_group,
            missing_date_display,
            missing_time_display,
            fill_missing_btn,
            event_summary_display,
            event_start_display,
            event_end_display,
            event_description_display,
            event_location_display,
            event_attendees_display,
            event_dict_storage,
            calendar_result_display
        ]
    )
    
    fill_missing_btn.click(
        fn=fill_missing_info,
        inputs=[event_dict_storage, missing_date_display, missing_time_display],
        outputs=[
            calendar_status_display,
            missing_info_group,
            missing_date_display,
            missing_time_display,
            fill_missing_btn,
            event_summary_display,
            event_start_display,
            event_end_display,
            event_description_display,
            event_location_display,
            event_attendees_display,
            event_dict_storage
        ]
    )
    
    create_event_btn.click(
        fn=create_event,
        inputs=[
            event_summary_display,
            event_start_display,
            event_end_display,
            event_description_display,
            event_location_display,
            event_attendees_display
        ],
        outputs=[calendar_status_display, calendar_result_display]
    )
    
    clear_calendar_btn.click(
        fn=clear_calendar,
        outputs=[
            calendar_prompt_input,
            calendar_status_display,
            missing_info_group,
            missing_date_display,
            missing_time_display,
            fill_missing_btn,
            event_summary_display,
            event_start_display,
            event_end_display,
            event_description_display,
            event_location_display,
            event_attendees_display,
            event_dict_storage,
            calendar_result_display
        ]
    )
    
    # 示例
    gr.Examples(
        examples=[
            "明天下午2點團隊會議，討論項目進度，地點在會議室A，參與者包括john@example.com",
            "2026-01-25 上午9點產品發布會，介紹新功能和改進，地點在總部大樓",
            "後天下午3點客戶會議，討論合作細節，參與者包括客戶代表",
            "下週一上午10點技術分享會，分享最新的 AI 技術，地點在研發中心"
        ],
        inputs=[calendar_prompt_input]
    )
    
    # 頁腳說明
    gr.Markdown(
        """
        ---
        **注意事項：**
        1. 使用 Google Calendar API 管理行事曆事件
        2. 首次使用需要在專案根目錄放置 `credentials.json`（從 Google Cloud Console 下載的 OAuth2 憑證）
        3. 首次運行時會自動開啟瀏覽器進行授權，授權後會生成 `token.json` 文件
        4. 事件內容由 AI 自動生成，請在創建前檢查結果
        5. 在提示中包含所有資訊：事件、日期、時間、地點、參與者
        6. 如果缺少日期或時間，系統會顯示下拉選單讓您選擇
        7. 日期格式支援：YYYY-MM-DD（例如：2026-01-25）或相對日期（今天、明天、後天）
        8. 時間格式支援：24小時制（14:00）或12小時制（2:00 PM）
        
        **設置步驟：**
        - 前往 [Google Cloud Console](https://console.cloud.google.com/) 創建專案
        - 啟用 Google Calendar API
        - 創建 OAuth2 憑證並下載為 `credentials.json`
        - 將 `credentials.json` 放在專案根目錄
        - 確保授予 Calendar API 的完整存取權限
        """
    )

