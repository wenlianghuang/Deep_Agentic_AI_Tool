"""
Calendar Reflection Agent
行事曆事件反思代理：評估生成的事件質量並提供改進建議
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..utils.llm_utils import get_llm, handle_groq_error
from ..guidelines import get_guideline


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


def reflect_on_calendar_event(
    prompt: str,
    event_dict: dict
) -> tuple[str, str, bool]:
    """
    反思行事曆事件質量，評估是否需要改進
    
    Args:
        prompt: 用戶的原始提示
        event_dict: 生成的事件字典，包含 summary, start_datetime, end_datetime, description, location, attendees
    
    Returns:
        (reflection_result, improvement_suggestions, needs_revision)
        - reflection_result: 反思結果（評估事件質量）
        - improvement_suggestions: 改進建議（如果需要改進）
        - needs_revision: 是否需要重新生成（True/False）
    """
    try:
        # 檢測語言
        user_language = detect_language(prompt)
        
        # 獲取 LLM
        llm = get_llm()
        
        # 【Parlant 整合】獲取事件反思評估指南
        event_reflection_guideline = get_guideline("calendar", "event_reflection")
        
        # 格式化事件資訊
        summary = event_dict.get("summary", "")
        start_datetime = event_dict.get("start_datetime", "")
        end_datetime = event_dict.get("end_datetime", "")
        description = event_dict.get("description", "")
        location = event_dict.get("location", "")
        attendees = event_dict.get("attendees", "")
        
        if user_language == 'zh':
            # 中文反思提示模板（整合指南）
            reflection_prompt_template = (
                "你是一位專業的行事曆事件質量評估專家。請仔細評估以下生成的行事曆事件，並提供詳細的反思和改進建議。\n\n"
                "【事件反思評估指南】\n{event_reflection_guideline}\n\n"
                "【用戶原始提示】\n{prompt}\n\n"
                "【生成的事件資訊】\n"
                "事件標題：{summary}\n"
                "開始時間：{start_datetime}\n"
                "結束時間：{end_datetime}\n"
                "事件描述：{description}\n"
                "事件地點：{location}\n"
                "參與者：{attendees}\n\n"
                "請嚴格遵循上述指南進行評估。\n\n"
                "請按照以下格式輸出：\n"
                "【反思評估】\n"
                "(詳細評估事件在各個方面的表現，指出優點和不足)\n\n"
                "【改進建議】\n"
                "(如果有需要改進的地方，請提供具體的改進建議；如果事件質量很好，請說明為什麼)\n\n"
                "【是否需要重新生成】\n"
                "(回答：是/否，並簡要說明原因。只有在事件有嚴重問題（如遺漏關鍵信息、時間不合理、描述不清楚）時才回答「是」)"
            )
        else:
            # 英文反思提示模板（整合指南）
            reflection_prompt_template = (
                "You are a professional calendar event quality assessment expert. Please carefully evaluate the following generated calendar event and provide detailed reflection and improvement suggestions.\n\n"
                "【Event Reflection Assessment Guidelines】\n{event_reflection_guideline}\n\n"
                "【User's Original Prompt】\n{prompt}\n\n"
                "【Generated Event Information】\n"
                "Event Title: {summary}\n"
                "Start Time: {start_datetime}\n"
                "End Time: {end_datetime}\n"
                "Event Description: {description}\n"
                "Event Location: {location}\n"
                "Attendees: {attendees}\n\n"
                "Please strictly follow the guidelines above for assessment.\n\n"
                "Please output in the following format:\n"
                "【Reflection Assessment】\n"
                "(Detailed assessment of the event's performance in various aspects, pointing out strengths and weaknesses)\n\n"
                "【Improvement Suggestions】\n"
                "(If there are areas that need improvement, provide specific suggestions; if the event quality is good, explain why)\n\n"
                "【Needs Revision】\n"
                "(Answer: Yes/No, and briefly explain the reason. Only answer 'Yes' if the event has serious issues such as missing key information, unreasonable time, unclear description)"
            )
        
        # 創建反思提示
        reflection_prompt = ChatPromptTemplate.from_template(reflection_prompt_template)
        
        try:
            chain = reflection_prompt | llm | StrOutputParser()
            reflection_result = chain.invoke({
                "prompt": prompt,
                "summary": summary,
                "start_datetime": start_datetime,
                "end_datetime": end_datetime,
                "description": description,
                "location": location,
                "attendees": attendees,
                "event_reflection_guideline": event_reflection_guideline
            })
        except Exception as e:
            # 處理 Groq API 錯誤
            fallback_llm = handle_groq_error(e)
            if fallback_llm:
                print("   ⚠️ [CalendarReflection] Groq API 額度已用完，已切換到本地 MLX 模型")
                chain = reflection_prompt | fallback_llm | StrOutputParser()
                reflection_result = chain.invoke({
                    "prompt": prompt,
                    "summary": summary,
                    "start_datetime": start_datetime,
                    "end_datetime": end_datetime,
                    "description": description,
                    "location": location,
                    "attendees": attendees,
                    "event_reflection_guideline": event_reflection_guideline
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
        print(f"Calendar Reflection Agent 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return error_msg, "", False


def generate_improved_calendar_event(
    prompt: str,
    original_event_dict: dict,
    improvement_suggestions: str
) -> dict:
    """
    根據反思建議生成改進後的行事曆事件
    
    Args:
        prompt: 用戶的原始提示
        original_event_dict: 原始事件字典
        improvement_suggestions: 改進建議
    
    Returns:
        改進後的事件字典
    """
    try:
        # 檢測語言
        user_language = detect_language(prompt)
        
        # 獲取 LLM
        llm = get_llm()
        
        # 【Parlant 整合】獲取事件改進生成指南
        event_improvement_guideline = get_guideline("calendar", "event_improvement")
        event_creation_guideline = get_guideline("calendar", "event_creation")
        time_parsing_guideline = get_guideline("calendar", "time_parsing")
        
        # 格式化原始事件資訊
        original_summary = original_event_dict.get("summary", "")
        original_start = original_event_dict.get("start_datetime", "")
        original_end = original_event_dict.get("end_datetime", "")
        original_description = original_event_dict.get("description", "")
        original_location = original_event_dict.get("location", "")
        original_attendees = original_event_dict.get("attendees", "")
        
        if user_language == 'zh':
            improvement_prompt_template = (
                "你是一位專業的行事曆事件解析助手。請根據改進建議，重新生成一個更好的行事曆事件。\n\n"
                "【事件改進生成指南】\n{event_improvement_guideline}\n\n"
                "【事件創建指南】\n{event_creation_guideline}\n\n"
                "【時間解析指南】\n{time_parsing_guideline}\n\n"
                "【用戶原始提示】\n{prompt}\n\n"
                "【原始事件資訊】\n"
                "事件標題：{original_summary}\n"
                "開始時間：{original_start}\n"
                "結束時間：{original_end}\n"
                "事件描述：{original_description}\n"
                "事件地點：{original_location}\n"
                "參與者：{original_attendees}\n\n"
                "【改進建議】\n{improvement_suggestions}\n\n"
                "請嚴格遵循上述指南，根據改進建議重新提取和生成事件資訊。"
                "確保事件完整、準確、清晰，並充分回應用戶的原始提示。"
                "請以 JSON 格式輸出，格式如下：\n"
                "{{\n"
                '  "summary": "事件標題",\n'
                '  "start_datetime": "ISO 8601 格式的開始時間（例如：2026-01-25T14:00:00+08:00），如果無法確定則為空字符串",\n'
                '  "end_datetime": "ISO 8601 格式的結束時間（例如：2026-01-25T15:00:00+08:00），如果無法確定則為空字符串",\n'
                '  "description": "事件描述",\n'
                '  "location": "事件地點（如果沒有則為空字符串）",\n'
                '  "attendees": "參與者郵箱，多個用逗號分隔（只包含有效的郵箱地址，格式：user@domain.com，如果沒有則為空字符串）",\n'
                '  "date": "原始日期字符串（用於 UI 顯示，例如：明天、2026-01-25）",\n'
                '  "time": "原始時間字符串（用於 UI 顯示，例如：14:00、下午2點）"\n'
                "}}\n\n"
                "重要：start_datetime 和 end_datetime 必須是完整的 ISO 8601 格式。\n"
                "只輸出 JSON，不要其他內容。請使用中文。"
            )
        else:
            improvement_prompt_template = (
                "You are a professional calendar event parsing assistant. Please regenerate a better calendar event based on the improvement suggestions.\n\n"
                "【Event Improvement Generation Guidelines】\n{event_improvement_guideline}\n\n"
                "【Event Creation Guidelines】\n{event_creation_guideline}\n\n"
                "【Time Parsing Guidelines】\n{time_parsing_guideline}\n\n"
                "【User's Original Prompt】\n{prompt}\n\n"
                "【Original Event Information】\n"
                "Event Title: {original_summary}\n"
                "Start Time: {original_start}\n"
                "End Time: {original_end}\n"
                "Event Description: {original_description}\n"
                "Event Location: {original_location}\n"
                "Attendees: {original_attendees}\n\n"
                "【Improvement Suggestions】\n{improvement_suggestions}\n\n"
                "Please strictly follow the guidelines above and regenerate the event information based on the improvement suggestions."
                "Ensure the event is complete, accurate, clear, and fully addresses the user's original prompt."
                "Please output in JSON format as follows:\n"
                "{{\n"
                '  "summary": "Event title",\n'
                '  "start_datetime": "ISO 8601 formatted start time (e.g., 2026-01-25T14:00:00+08:00), empty string if cannot determine",\n'
                '  "end_datetime": "ISO 8601 formatted end time (e.g., 2026-01-25T15:00:00+08:00), empty string if cannot determine",\n'
                '  "description": "Event description",\n'
                '  "location": "Event location (empty string if not mentioned)",\n'
                '  "attendees": "Attendee emails, comma-separated (only valid email addresses in format: user@domain.com, empty string if not mentioned)",\n'
                '  "date": "Original date string (for UI display, e.g., tomorrow, 2026-01-25)",\n'
                '  "time": "Original time string (for UI display, e.g., 14:00, 2:00 PM)"\n'
                "}}\n\n"
                "Important: start_datetime and end_datetime must be complete ISO 8601 format.\n"
                "Output only JSON, nothing else. Please use English."
            )
        
        # 生成改進後的事件資訊
        improvement_prompt = ChatPromptTemplate.from_template(improvement_prompt_template)
        
        try:
            chain = improvement_prompt | llm | StrOutputParser()
            improved_content = chain.invoke({
                "prompt": prompt,
                "original_summary": original_summary,
                "original_start": original_start,
                "original_end": original_end,
                "original_description": original_description,
                "original_location": original_location,
                "original_attendees": original_attendees,
                "improvement_suggestions": improvement_suggestions,
                "event_improvement_guideline": event_improvement_guideline,
                "event_creation_guideline": event_creation_guideline,
                "time_parsing_guideline": time_parsing_guideline
            })
        except Exception as e:
            fallback_llm = handle_groq_error(e)
            if fallback_llm:
                print("   ⚠️ [CalendarReflection] Groq API 額度已用完，已切換到本地 MLX 模型")
                chain = improvement_prompt | fallback_llm | StrOutputParser()
                improved_content = chain.invoke({
                    "prompt": prompt,
                    "original_summary": original_summary,
                    "original_start": original_start,
                    "original_end": original_end,
                    "original_description": original_description,
                    "original_location": original_location,
                    "original_attendees": original_attendees,
                    "improvement_suggestions": improvement_suggestions,
                    "event_improvement_guideline": event_improvement_guideline,
                    "event_creation_guideline": event_creation_guideline,
                    "time_parsing_guideline": time_parsing_guideline
                })
            else:
                raise
        
        # 解析 JSON 響應
        import json
        try:
            # 清理可能的 markdown 代碼塊
            improved_content = improved_content.strip()
            if improved_content.startswith('```'):
                lines = improved_content.split('\n')
                improved_content = '\n'.join(lines[1:-1])
            elif improved_content.startswith('```json'):
                lines = improved_content.split('\n')
                improved_content = '\n'.join(lines[1:-1])
            
            improved_data = json.loads(improved_content)
        except json.JSONDecodeError:
            # 如果 JSON 解析失敗，使用原始事件
            print("   ⚠️ [CalendarReflection] JSON 解析失敗，使用原始事件")
            return original_event_dict
        
        # 優先使用 LLM 直接輸出的 ISO 8601 格式日期時間
        start_datetime = improved_data.get("start_datetime", "").strip()
        end_datetime = improved_data.get("end_datetime", "").strip()
        
        # 驗證 ISO 8601 格式
        def validate_iso8601(dt_str: str) -> bool:
            """驗證 ISO 8601 格式"""
            if not dt_str:
                return False
            try:
                from datetime import datetime as dt
                dt_str_clean = dt_str.replace('+08:00', '+08:00')
                dt.fromisoformat(dt_str_clean)
                return True
            except:
                return False
        
        # 如果 LLM 沒有輸出格式化的日期時間，或格式不正確，使用後備解析
        if not validate_iso8601(start_datetime) or not validate_iso8601(end_datetime):
            # 後備：使用 Python 解析邏輯
            from .calendar_agent import parse_datetime
            date_str = improved_data.get("date", "").strip() or original_event_dict.get("date", "今天")
            time_str = improved_data.get("time", "").strip() or original_event_dict.get("time", None)
            print("   ⚠️ [CalendarReflection] 改進版本未輸出有效的 ISO 8601 格式，使用後備解析邏輯")
            start_datetime, end_datetime = parse_datetime(date_str, time_str)
        else:
            # LLM 已輸出有效格式，提取原始日期時間字符串用於 UI
            date_str = improved_data.get("date", "").strip() or original_event_dict.get("date", "")
            time_str = improved_data.get("time", "").strip() or original_event_dict.get("time", "")
        
        # 構建改進後的事件字典
        improved_event_dict = {
            "summary": improved_data.get("summary", original_summary),
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "description": improved_data.get("description", original_description),
            "location": improved_data.get("location", original_location),
            "attendees": improved_data.get("attendees", original_attendees),
            "timezone": original_event_dict.get("timezone", "Asia/Taipei"),
            "date": date_str,
            "time": time_str if time_str else ""
        }
        
        return improved_event_dict
        
    except Exception as e:
        error_msg = f"生成改進事件時發生錯誤：{str(e)}"
        print(f"Calendar Reflection Agent 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return original_event_dict

