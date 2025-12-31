"""
Email Agent
簡單的郵件生成和發送代理
"""
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..utils.llm_utils import get_llm, handle_groq_error
from ..tools.email_tool import send_email


def detect_language(text: str) -> str:
    """
    檢測文本的主要語言（中文或英文）
    
    Args:
        text: 輸入文本
    
    Returns:
        'zh' 或 'en'
    """
    # 簡單的語言檢測：檢查是否包含中文字符
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    if chinese_pattern.search(text):
        return 'zh'
    else:
        return 'en'


def generate_email_draft(prompt: str, recipient: str) -> tuple[str, str, str]:
    """
    根據用戶提示生成郵件草稿（不發送）
    
    Args:
        prompt: 用戶的關鍵提示（例如："寫一封感謝信"）
        recipient: 收件人郵箱地址
    
    Returns:
        (subject, body, status_message) 元組
    """
    try:
        # 檢測用戶輸入的語言
        user_language = detect_language(prompt)
        
        # 獲取 LLM
        llm = get_llm()
        
        # 根據語言選擇對應的 prompt 模板
        if user_language == 'zh':
            # 中文 prompt 模板
            email_prompt_template = (
                "你是一位專業的郵件撰寫助手。請根據以下用戶提示，撰寫一封專業、禮貌的郵件草稿。\n\n"
                "用戶提示：{prompt}\n\n"
                "收件人：{recipient}\n\n"
                "請生成完整的郵件內容，包括：\n"
                "1. 適當的問候語\n"
                "2. 清晰的主體內容（根據用戶提示）\n"
                "3. 適當的結尾\n"
                "4. 簽名部分請使用 [您的姓名] 和 [公司名稱] 作為佔位符，讓使用者自行填寫\n\n"
                "郵件應該專業、禮貌、簡潔明瞭。請使用中文撰寫。直接輸出郵件正文內容，不需要包含主題行。"
            )
            subject_prompt_template = (
                "請根據以下郵件內容，生成一個簡潔、專業的郵件主題（不超過50個字）：\n\n"
                "郵件內容：\n{email_body}\n\n"
                "只輸出主題，不要其他內容。請使用中文。"
            )
            default_subject = "郵件"
        else:
            # 英文 prompt 模板
            email_prompt_template = (
                "You are a professional email writing assistant. Please write a professional and polite email draft based on the following user prompt.\n\n"
                "User prompt: {prompt}\n\n"
                "Recipient: {recipient}\n\n"
                "Please generate a complete email content including:\n"
                "1. Appropriate greeting\n"
                "2. Clear main content (based on the user prompt)\n"
                "3. Appropriate closing\n"
                "4. For the signature section, use [Your Name] and [Company Name] as placeholders for the user to fill in\n\n"
                "The email should be professional, polite, and concise. Please write in English. Output only the email body content, do not include the subject line."
            )
            subject_prompt_template = (
                "Please generate a concise and professional email subject (no more than 50 characters) based on the following email content:\n\n"
                "Email content:\n{email_body}\n\n"
                "Output only the subject, nothing else. Please use English."
            )
            default_subject = "Email"
        
        # 創建郵件生成提示
        email_prompt = ChatPromptTemplate.from_template(email_prompt_template)
        
        # 生成郵件內容
        try:
            chain = email_prompt | llm | StrOutputParser()
            email_body = chain.invoke({
                "prompt": prompt,
                "recipient": recipient
            })
        except Exception as e:
            # 處理 Groq API 錯誤
            fallback_llm = handle_groq_error(e)
            if fallback_llm:
                print("   ⚠️ [EmailAgent] Groq API 額度已用完，已切換到本地 MLX 模型")
                chain = email_prompt | fallback_llm | StrOutputParser()
                email_body = chain.invoke({
                    "prompt": prompt,
                    "recipient": recipient
                })
            else:
                raise
        
        # 生成郵件主題
        subject_prompt = ChatPromptTemplate.from_template(subject_prompt_template)
        
        try:
            subject_chain = subject_prompt | llm | StrOutputParser()
            email_subject = subject_chain.invoke({"email_body": email_body})
        except Exception as e:
            # 如果生成主題失敗，使用預設主題
            fallback_llm = handle_groq_error(e)
            if fallback_llm:
                subject_chain = subject_prompt | fallback_llm | StrOutputParser()
                email_subject = subject_chain.invoke({"email_body": email_body})
            else:
                email_subject = default_subject
        
        # 清理主題（移除可能的引號或多餘空格）
        email_subject = email_subject.strip().strip('"').strip("'")
        if not email_subject:
            email_subject = default_subject
        
        status_message = "✅ 郵件草稿已生成，請檢查並修改後再發送"
        return email_subject, email_body, status_message
        
    except Exception as e:
        error_msg = f"❌ 生成郵件草稿時發生錯誤：{str(e)}"
        print(f"Email Agent 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return "", "", error_msg


def send_email_draft(recipient: str, subject: str, body: str) -> str:
    """
    發送已編輯的郵件草稿（僅支援 Gmail 郵箱）
    
    Args:
        recipient: 收件人郵箱地址（必須是 Gmail 郵箱）
        subject: 郵件主題
        body: 郵件正文內容
    
    Returns:
        發送結果消息
    """
    try:
        # 驗證收件人是否為 Gmail 郵箱
        recipient_lower = recipient.strip().lower()
        if not (recipient_lower.endswith("@gmail.com") or recipient_lower.endswith("@googlemail.com")):
            return (
                f"❌ 錯誤：此工具僅支援 Gmail 郵箱。\n"
                f"您輸入的郵箱：{recipient}\n"
                f"請使用 @gmail.com 或 @googlemail.com 結尾的郵箱地址。"
            )
        
        # 發送郵件
        result = send_email.invoke({
            "recipient": recipient,
            "subject": subject,
            "body": body
        })
        
        return f"📧 {result}\n\n郵件主題：{subject}\n\n郵件已成功發送！"
        
    except Exception as e:
        error_msg = f"❌ 發送郵件時發生錯誤：{str(e)}"
        print(f"Email Tool 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return error_msg


def generate_and_send_email(prompt: str, recipient: str) -> str:
    """
    根據用戶提示生成郵件內容並發送
    
    Args:
        prompt: 用戶的關鍵提示（例如："寫一封感謝信"）
        recipient: 收件人郵箱地址
    
    Returns:
        執行結果消息
    """
    try:
        # 檢測用戶輸入的語言
        user_language = detect_language(prompt)
        
        # 獲取 LLM
        llm = get_llm()
        
        # 根據語言選擇對應的 prompt 模板
        if user_language == 'zh':
            # 中文 prompt 模板
            email_prompt_template = (
                "你是一位專業的郵件撰寫助手。請根據以下用戶提示，撰寫一封專業、禮貌的郵件。\n\n"
                "用戶提示：{prompt}\n\n"
                "收件人：{recipient}\n\n"
                "請生成完整的郵件內容，包括：\n"
                "1. 適當的問候語\n"
                "2. 清晰的主體內容（根據用戶提示）\n"
                "3. 適當的結尾和簽名\n\n"
                "郵件應該專業、禮貌、簡潔明瞭。請使用中文撰寫。直接輸出郵件正文內容，不需要包含主題行。"
            )
            subject_prompt_template = (
                "請根據以下郵件內容，生成一個簡潔、專業的郵件主題（不超過50個字）：\n\n"
                "郵件內容：\n{email_body}\n\n"
                "只輸出主題，不要其他內容。請使用中文。"
            )
            default_subject = "郵件"
        else:
            # 英文 prompt 模板
            email_prompt_template = (
                "You are a professional email writing assistant. Please write a professional and polite email based on the following user prompt.\n\n"
                "User prompt: {prompt}\n\n"
                "Recipient: {recipient}\n\n"
                "Please generate a complete email content including:\n"
                "1. Appropriate greeting\n"
                "2. Clear main content (based on the user prompt)\n"
                "3. Appropriate closing and signature\n\n"
                "The email should be professional, polite, and concise. Please write in English. Output only the email body content, do not include the subject line."
            )
            subject_prompt_template = (
                "Please generate a concise and professional email subject (no more than 50 characters) based on the following email content:\n\n"
                "Email content:\n{email_body}\n\n"
                "Output only the subject, nothing else. Please use English."
            )
            default_subject = "Email"
        
        # 創建郵件生成提示
        email_prompt = ChatPromptTemplate.from_template(email_prompt_template)
        
        # 生成郵件內容
        try:
            chain = email_prompt | llm | StrOutputParser()
            email_body = chain.invoke({
                "prompt": prompt,
                "recipient": recipient
            })
        except Exception as e:
            # 處理 Groq API 錯誤
            fallback_llm = handle_groq_error(e)
            if fallback_llm:
                print("   ⚠️ [EmailAgent] Groq API 額度已用完，已切換到本地 MLX 模型")
                chain = email_prompt | fallback_llm | StrOutputParser()
                email_body = chain.invoke({
                    "prompt": prompt,
                    "recipient": recipient
                })
            else:
                raise
        
        # 生成郵件主題
        subject_prompt = ChatPromptTemplate.from_template(subject_prompt_template)
        
        try:
            subject_chain = subject_prompt | llm | StrOutputParser()
            email_subject = subject_chain.invoke({"email_body": email_body})
        except Exception as e:
            # 如果生成主題失敗，使用預設主題
            fallback_llm = handle_groq_error(e)
            if fallback_llm:
                subject_chain = subject_prompt | fallback_llm | StrOutputParser()
                email_subject = subject_chain.invoke({"email_body": email_body})
            else:
                email_subject = default_subject
        
        # 清理主題（移除可能的引號或多餘空格）
        email_subject = email_subject.strip().strip('"').strip("'")
        if not email_subject:
            email_subject = default_subject
        
        # 發送郵件
        result = send_email.invoke({
            "recipient": recipient,
            "subject": email_subject,
            "body": email_body
        })
        
        return f"📧 郵件生成和發送結果：\n\n{result}\n\n郵件主題：{email_subject}\n\n郵件內容預覽：\n{email_body[:200]}..."
        
    except Exception as e:
        error_msg = f"❌ 生成或發送郵件時發生錯誤：{str(e)}"
        print(f"Email Agent 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return error_msg

