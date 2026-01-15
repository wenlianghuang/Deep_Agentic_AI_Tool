# deep_agent_rag/ui/email_interface.py

import gradio as gr
import re
import json
import time

from ..agents.email_agent import generate_email_draft, send_email_draft
from ..config import EMAIL_SENDER
from ..utils.llm_utils import is_using_local_llm # Assuming this might be used for warnings/status

# Agent log path for debugging (if needed)
log_path = "/Users/matthuang/Desktop/Deep_Agentic_AI_Tool/.cursor/debug.log"

def _create_email_interface():
    """創建 Email Tool 界面"""
    gr.Markdown(
        f"""
        ### 📧 智能郵件助手
        
        使用 AI 根據您的關鍵提示自動生成專業郵件草稿，您可以在發送前檢查和修改。
        
        **預設寄件者：** {EMAIL_SENDER}（可在下方輸入框中修改）
        
        **使用方式：**
        1. 輸入發件人 Gmail 郵箱地址（可選，不填則使用預設：{EMAIL_SENDER}）
        2. 在下方輸入郵件提示（例如："寫一封感謝信"、"邀請參加會議"等）
        3. 輸入收件人郵箱地址（可以是單個或多個，多個收件人請用逗號分隔，例如："user1@example.com, user2@example.com"）
        4. 點擊「生成郵件草稿」按鈕
        5. 查看 AI 反思評估結果和改進建議（如有）
        6. 檢查並修改生成的郵件內容（特別是簽名部分）
        7. 確認無誤後點擊「發送郵件」按鈕
        
        **✨ 新功能：多使用者支援**
        - 每個使用者可以輸入自己的 Gmail 郵箱作為發件人
        - 系統會自動使用對應的 OAuth2 憑證和 token
        - 首次使用新帳號時會自動觸發授權流程
        
        **✨ 新功能：AI 迭代反思評估**
        - 系統會自動進行多輪反思評估（最多 3 輪）
        - 每輪評估後，如果有改進建議，會自動生成改進版本
        - 改進後的版本會再次評估，直到 AI 認為滿意為止
        - 您可以看到完整的反思過程和每輪的改進建議
        
        **注意：發件人必須是 Gmail 郵箱（因為使用 Gmail API），但收件人可以是任何郵箱地址。**
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # 發件人輸入
            sender_input = gr.Textbox(
                label="👤 發件人郵箱（可選，僅支援 Gmail）",
                placeholder=f"留空則使用預設：{EMAIL_SENDER}",
                value="",
                lines=1,
                info="輸入您的 Gmail 郵箱地址作為發件人。首次使用新帳號時會自動觸發授權流程。"
            )
            
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
                placeholder="user@example.com 或 user1@example.com, user2@example.com（多個收件人用逗號分隔）",
                lines=2,
                info="可以是單個或多個郵箱地址，多個收件人請用逗號分隔（例如：user1@example.com, user2@example.com）"
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
    def generate_draft(sender, prompt, recipient):
        """生成郵件草稿（包含反思功能）"""
        if not prompt or not prompt.strip():
            return "❌ 請輸入郵件提示", "", "", "❌ 請輸入郵件提示", "❌ 請輸入郵件提示"
        
        if not recipient or not recipient.strip():
            return "❌ 請輸入收件人郵箱", "", "", "❌ 請輸入收件人郵箱", "❌ 請輸入收件人郵箱"
        
        # 處理發件人（如果提供）
        actual_sender = sender.strip() if sender and sender.strip() else None
        
        # 如果提供了發件人，驗證發件人郵箱格式和 Gmail 限制
        if actual_sender:
            if "@" not in actual_sender or "." not in actual_sender.split("@")[1]:
                return "❌ 發件人郵箱格式不正確", "", "", "❌ 發件人郵箱格式不正確，請輸入有效的郵箱地址", "❌ 發件人郵箱格式不正確，請輸入有效的郵箱地址"
            
            sender_lower = actual_sender.strip().lower()
            if not (sender_lower.endswith("@gmail.com") or sender_lower.endswith("@googlemail.com")):
                return "❌ 發件人必須是 Gmail 郵箱", "", "", "❌ 發件人必須是 Gmail 郵箱（@gmail.com 或 @googlemail.com）", "❌ 發件人必須是 Gmail 郵箱（@gmail.com 或 @googlemail.com）"
        
        # 解析收件人（支援多個，用逗號分隔）
        recipients = [email.strip() for email in recipient.split(',') if email.strip()]
        
        if not recipients:
            return "❌ 請輸入至少一個收件人郵箱", "", "", "❌ 請輸入至少一個收件人郵箱", "❌ 請輸入至少一個收件人郵箱"
        
        # 驗證每個收件人郵箱格式
        invalid_emails = []
        for email in recipients:
            if "@" not in email or "." not in email.split("@")[1]:
                invalid_emails.append(email)
        
        if invalid_emails:
            return (
                "❌ 收件人郵箱格式不正確",
                "",
                "",
                f"❌ 以下收件人郵箱格式不正確：{', '.join(invalid_emails)}",
                f"❌ 以下收件人郵箱格式不正確：{', '.join(invalid_emails)}"
            )
        
        # 使用第一個收件人來生成郵件（郵件生成通常針對單一收件人）
        primary_recipient = recipients[0]
        
        try:
            status_msg = "🔄 正在生成郵件草稿..."
            reflection_msg = "🔄 正在生成郵件草稿..."
            
            # 生成郵件草稿（包含反思功能，會自動改進）
            # 使用第一個收件人來生成郵件內容
            subject, body, status, reflection_result, was_improved = generate_email_draft(
                prompt, primary_recipient, enable_reflection=True
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
    
    def send_draft(sender, recipient, subject, body):
        """發送已編輯的郵件草稿"""
        if not recipient or not recipient.strip():
            return "❌ 請輸入收件人郵箱", "❌ 請輸入收件人郵箱"
        
        if not subject or not subject.strip():
            return "❌ 請輸入郵件主題", "❌ 請輸入郵件主題"
        
        if not body or not body.strip():
            return "❌ 請輸入郵件內容", "❌ 請輸入郵件內容"
        
        # 處理發件人（如果提供）
        actual_sender = sender.strip() if sender and sender.strip() else None
        
        # 如果提供了發件人，驗證發件人郵箱格式和 Gmail 限制
        if actual_sender:
            if "@" not in actual_sender or "." not in actual_sender.split("@")[1]:
                return "❌ 發件人郵箱格式不正確", "❌ 發件人郵箱格式不正確，請輸入有效的郵箱地址"
            
            sender_lower = actual_sender.strip().lower()
            if not (sender_lower.endswith("@gmail.com") or sender_lower.endswith("@googlemail.com")):
                return "❌ 發件人必須是 Gmail 郵箱", "❌ 發件人必須是 Gmail 郵箱（@gmail.com 或 @googlemail.com）"
        
        # 解析收件人（支援多個，用逗號分隔）
        recipients = [email.strip() for email in recipient.split(',') if email.strip()]
        
        if not recipients:
            return "❌ 請輸入至少一個收件人郵箱", "❌ 請輸入至少一個收件人郵箱"
        
        # 驗證每個收件人郵箱格式
        invalid_emails = []
        for email in recipients:
            if "@" not in email or "." not in email.split("@")[1]:
                invalid_emails.append(email)
        
        if invalid_emails:
            return (
                "❌ 收件人郵箱格式不正確",
                f"❌ 以下收件人郵箱格式不正確：{', '.join(invalid_emails)}"
            )
        
        try:
            status_msg = "🔄 正在發送郵件..."
            
            # 發送郵件（傳遞發件人參數，底層工具會處理多個收件人）
            result = send_email_draft(recipient.strip(), subject.strip(), body.strip(), actual_sender)
            
            return "✅ 郵件已發送", result
        except Exception as e:
            error_msg = f"❌ 發送郵件時發生錯誤：{str(e)}"
            print(f"Email Tool 錯誤：{e}")
            import traceback
            traceback.print_exc()
            return "❌ 發生錯誤", error_msg
    
    def clear_email():
        """清除郵件相關輸入和輸出"""
        return "", "", "", "等待操作...", "", "", "等待生成郵件..."
    
    # 綁定事件
    generate_draft_btn.click(
        fn=generate_draft,
        inputs=[sender_input, email_prompt_input, recipient_input],
        outputs=[email_status_display, email_subject_input, email_body_input, email_result_display, email_reflection_display]
    )
    
    send_draft_btn.click(
        fn=send_draft,
        inputs=[sender_input, recipient_input, email_subject_input, email_body_input],
        outputs=[email_status_display, email_result_display]
    )
    
    clear_email_btn.click(
        fn=clear_email,
        outputs=[sender_input, email_prompt_input, recipient_input, email_status_display, email_subject_input, email_body_input, email_result_display, email_reflection_display]
    )
    
    # 示例
    gr.Examples(
        examples=[
            ["寫一封感謝信，感謝對方在項目中的幫助和支持", "example@company.com"],
            ["邀請參加下週的產品發布會", "colleague1@outlook.com, colleague2@outlook.com"],
            ["詢問項目進度並提供更新", "partner@yahoo.com"],
            ["發送會議記錄和後續行動項目", "team1@university.edu, team2@university.edu, team3@university.edu"]
        ],
        inputs=[email_prompt_input, recipient_input]
    )
    
    # 頁腳說明
    gr.Markdown(
        f"""
        ---
        **注意事項：**
        1. 使用 Gmail API 發送郵件，避免被歸類為垃圾郵件
        2. **發件人必須是 Gmail 郵箱（因為使用 Gmail API），但收件人可以是任何郵箱地址**
        3. 首次使用需要在專案根目錄放置 OAuth2 憑證文件（從 Google Cloud Console 下載）
        4. 如果使用預設發件人，需要 `credentials_matthuang.json` 和 `token.json`
        5. 如果使用其他發件人，系統會自動尋找 `credentials_{{username}}.json`，如果不存在則使用預設憑證文件
        6. 首次使用新帳號時會自動開啟瀏覽器進行授權，授權後會生成 `token_{{username}}.json` 文件
        7. 郵件內容由 AI 自動生成，請在發送前檢查結果
        8. 預設寄件者：{EMAIL_SENDER}（可在上方輸入框中修改）
        
        **設置步驟：**
        - 前往 [Google Cloud Console](https://console.cloud.google.com/) 創建專案
        - 啟用 Gmail API
        - 創建 OAuth2 憑證並下載
        - 將憑證文件放在專案根目錄（命名為 `credentials.json` 或 `credentials_{{username}}.json`）
        - 首次使用時會自動觸發授權流程
        """
    )
