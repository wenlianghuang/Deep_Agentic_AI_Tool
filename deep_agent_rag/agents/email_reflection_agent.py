"""
Email Reflection Agent
郵件反思代理：評估生成的郵件質量並提供改進建議
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..utils.llm_utils import get_llm, handle_groq_error


def detect_language(text: str) -> str:
    """
    檢測文本的主要語言（中文或英文）
    
    Args:
        text: 輸入文本
    
    Returns:
        'zh' 或 'en'
    """
    import re
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    if chinese_pattern.search(text):
        return 'zh'
    else:
        return 'en'


def reflect_on_email(
    prompt: str,
    recipient: str,
    subject: str,
    body: str
) -> tuple[str, str, bool]:
    """
    反思郵件質量，評估是否需要改進
    
    Args:
        prompt: 用戶的原始提示
        recipient: 收件人郵箱
        subject: 郵件主題
        body: 郵件正文
    
    Returns:
        (reflection_result, improvement_suggestions, needs_revision)
        - reflection_result: 反思結果（評估郵件質量）
        - improvement_suggestions: 改進建議（如果需要改進）
        - needs_revision: 是否需要重新生成（True/False）
    """
    try:
        # 檢測語言
        user_language = detect_language(prompt)
        
        # 獲取 LLM
        llm = get_llm()
        
        if user_language == 'zh':
            # 中文反思提示模板
            reflection_prompt_template = (
                "你是一位專業的郵件質量評估專家。請仔細評估以下生成的郵件，並提供詳細的反思和改進建議。\n\n"
                "【用戶原始提示】\n{prompt}\n\n"
                "【收件人】\n{recipient}\n\n"
                "【生成的郵件主題】\n{subject}\n\n"
                "【生成的郵件正文】\n{body}\n\n"
                "請從以下幾個方面進行評估：\n"
                "1. **內容完整性**：郵件是否完整回應了用戶的提示？是否遺漏重要信息？\n"
                "2. **專業性**：語氣是否專業、禮貌？是否符合商務郵件的標準？\n"
                "3. **清晰度**：郵件內容是否清晰易懂？結構是否合理？\n"
                "4. **適配性**：郵件內容是否適合收件人？問候語和結尾是否得當？\n"
                "5. **主題相關性**：郵件主題是否準確反映郵件內容？\n\n"
                "請按照以下格式輸出：\n"
                "【反思評估】\n"
                "(詳細評估郵件在各個方面的表現，指出優點和不足)\n\n"
                "【改進建議】\n"
                "(如果有需要改進的地方，請提供具體的改進建議；如果郵件質量很好，請說明為什麼)\n\n"
                "【是否需要重新生成】\n"
                "(回答：是/否，並簡要說明原因。只有在郵件有嚴重問題（如遺漏關鍵信息、語氣不當、內容不符合要求）時才回答「是」)"
            )
        else:
            # 英文反思提示模板
            reflection_prompt_template = (
                "You are a professional email quality assessment expert. Please carefully evaluate the following generated email and provide detailed reflection and improvement suggestions.\n\n"
                "【User's Original Prompt】\n{prompt}\n\n"
                "【Recipient】\n{recipient}\n\n"
                "【Generated Email Subject】\n{subject}\n\n"
                "【Generated Email Body】\n{body}\n\n"
                "Please evaluate from the following aspects:\n"
                "1. **Content Completeness**: Does the email fully address the user's prompt? Are there any missing important information?\n"
                "2. **Professionalism**: Is the tone professional and polite? Does it meet business email standards?\n"
                "3. **Clarity**: Is the email content clear and understandable? Is the structure reasonable?\n"
                "4. **Appropriateness**: Is the email content appropriate for the recipient? Are the greeting and closing appropriate?\n"
                "5. **Subject Relevance**: Does the email subject accurately reflect the email content?\n\n"
                "Please output in the following format:\n"
                "【Reflection Assessment】\n"
                "(Detailed assessment of the email's performance in various aspects, pointing out strengths and weaknesses)\n\n"
                "【Improvement Suggestions】\n"
                "(If there are areas that need improvement, provide specific suggestions; if the email quality is good, explain why)\n\n"
                "【Needs Revision】\n"
                "(Answer: Yes/No, and briefly explain the reason. Only answer 'Yes' if the email has serious issues such as missing key information, inappropriate tone, or content that doesn't meet requirements)"
            )
        
        # 創建反思提示
        reflection_prompt = ChatPromptTemplate.from_template(reflection_prompt_template)
        
        try:
            chain = reflection_prompt | llm | StrOutputParser()
            reflection_result = chain.invoke({
                "prompt": prompt,
                "recipient": recipient,
                "subject": subject,
                "body": body
            })
        except Exception as e:
            # 處理 Groq API 錯誤
            fallback_llm = handle_groq_error(e)
            if fallback_llm:
                print("   ⚠️ [EmailReflection] Groq API 額度已用完，已切換到本地 MLX 模型")
                chain = reflection_prompt | fallback_llm | StrOutputParser()
                reflection_result = chain.invoke({
                    "prompt": prompt,
                    "recipient": recipient,
                    "subject": subject,
                    "body": body
                })
            else:
                raise
        
        # 解析反思結果
        reflection_text = reflection_result
        improvement_suggestions = ""
        needs_revision = False
        
        # 提取改進建議部分
        if "【改進建議】" in reflection_text or "【Improvement Suggestions】" in reflection_text:
            parts = reflection_text.split("【改進建議】") if "【改進建議】" in reflection_text else reflection_text.split("【Improvement Suggestions】")
            if len(parts) > 1:
                improvement_part = parts[1].split("【是否需要重新生成】")[0] if "【是否需要重新生成】" in parts[1] else parts[1].split("【Needs Revision】")[0]
                improvement_suggestions = improvement_part.strip()
        
        # 檢查是否需要重新生成
        if "【是否需要重新生成】" in reflection_text or "【Needs Revision】" in reflection_text:
            revision_part = reflection_text.split("【是否需要重新生成】")[-1] if "【是否需要重新生成】" in reflection_text else reflection_text.split("【Needs Revision】")[-1]
            revision_text = revision_part.strip().lower()
            # 檢查是否包含「是」、「yes」等關鍵字
            needs_revision = any(keyword in revision_text for keyword in ["是", "yes", "需要", "need", "應該", "should"])
        
        return reflection_text, improvement_suggestions, needs_revision
        
    except Exception as e:
        error_msg = f"反思過程中發生錯誤：{str(e)}"
        print(f"Email Reflection Agent 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return error_msg, "", False


def generate_improved_email(
    prompt: str,
    recipient: str,
    original_subject: str,
    original_body: str,
    improvement_suggestions: str
) -> tuple[str, str]:
    """
    根據反思建議生成改進後的郵件
    
    Args:
        prompt: 用戶的原始提示
        recipient: 收件人郵箱
        original_subject: 原始郵件主題
        original_body: 原始郵件正文
        improvement_suggestions: 改進建議
    
    Returns:
        (improved_subject, improved_body)
    """
    try:
        # 檢測語言
        user_language = detect_language(prompt)
        
        # 獲取 LLM
        llm = get_llm()
        
        if user_language == 'zh':
            improvement_prompt_template = (
                "你是一位專業的郵件撰寫助手。請根據改進建議，重新生成一封更好的郵件。\n\n"
                "【用戶原始提示】\n{prompt}\n\n"
                "【收件人】\n{recipient}\n\n"
                "【原始郵件主題】\n{original_subject}\n\n"
                "【原始郵件正文】\n{original_body}\n\n"
                "【改進建議】\n{improvement_suggestions}\n\n"
                "請根據改進建議，生成改進後的郵件主題和正文。"
                "確保郵件專業、禮貌、完整，並充分回應用戶的原始提示。"
                "直接輸出改進後的郵件正文內容，不需要包含主題行。"
            )
            subject_prompt_template = (
                "請根據以下改進後的郵件內容，生成一個簡潔、專業的郵件主題（不超過50個字）：\n\n"
                "郵件內容：\n{email_body}\n\n"
                "只輸出主題，不要其他內容。請使用中文。"
            )
            default_subject = "郵件"
        else:
            improvement_prompt_template = (
                "You are a professional email writing assistant. Please regenerate a better email based on the improvement suggestions.\n\n"
                "【User's Original Prompt】\n{prompt}\n\n"
                "【Recipient】\n{recipient}\n\n"
                "【Original Email Subject】\n{original_subject}\n\n"
                "【Original Email Body】\n{original_body}\n\n"
                "【Improvement Suggestions】\n{improvement_suggestions}\n\n"
                "Please generate an improved email subject and body based on the improvement suggestions."
                "Ensure the email is professional, polite, complete, and fully addresses the user's original prompt."
                "Output only the improved email body content, do not include the subject line."
            )
            subject_prompt_template = (
                "Please generate a concise and professional email subject (no more than 50 characters) based on the following improved email content:\n\n"
                "Email content:\n{email_body}\n\n"
                "Output only the subject, nothing else. Please use English."
            )
            default_subject = "Email"
        
        # 生成改進後的郵件正文
        improvement_prompt = ChatPromptTemplate.from_template(improvement_prompt_template)
        
        try:
            chain = improvement_prompt | llm | StrOutputParser()
            improved_body = chain.invoke({
                "prompt": prompt,
                "recipient": recipient,
                "original_subject": original_subject,
                "original_body": original_body,
                "improvement_suggestions": improvement_suggestions
            })
        except Exception as e:
            fallback_llm = handle_groq_error(e)
            if fallback_llm:
                print("   ⚠️ [EmailReflection] Groq API 額度已用完，已切換到本地 MLX 模型")
                chain = improvement_prompt | fallback_llm | StrOutputParser()
                improved_body = chain.invoke({
                    "prompt": prompt,
                    "recipient": recipient,
                    "original_subject": original_subject,
                    "original_body": original_body,
                    "improvement_suggestions": improvement_suggestions
                })
            else:
                raise
        
        # 生成改進後的郵件主題
        subject_prompt = ChatPromptTemplate.from_template(subject_prompt_template)
        
        try:
            subject_chain = subject_prompt | llm | StrOutputParser()
            improved_subject = subject_chain.invoke({"email_body": improved_body})
        except Exception as e:
            fallback_llm = handle_groq_error(e)
            if fallback_llm:
                subject_chain = subject_prompt | fallback_llm | StrOutputParser()
                improved_subject = subject_chain.invoke({"email_body": improved_body})
            else:
                improved_subject = default_subject
        
        # 清理主題
        improved_subject = improved_subject.strip().strip('"').strip("'")
        if not improved_subject:
            improved_subject = default_subject
        
        return improved_subject, improved_body
        
    except Exception as e:
        error_msg = f"生成改進郵件時發生錯誤：{str(e)}"
        print(f"Email Reflection Agent 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return original_subject, original_body

