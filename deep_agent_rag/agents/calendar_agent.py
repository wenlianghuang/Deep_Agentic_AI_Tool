"""
Calendar Agent
行事曆事件生成和管理代理
"""
import re
from datetime import datetime, timedelta
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..utils.llm_utils import get_llm, handle_groq_error
from ..tools.calendar_tool import create_calendar_event, update_calendar_event, delete_calendar_event


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


def parse_datetime(date_str: str, time_str: str = None) -> tuple[str, str]:
    """
    解析日期和時間，生成 ISO 8601 格式的開始和結束時間
    
    Args:
        date_str: 日期字符串（例如: "2026-01-25" 或 "明天"）
        time_str: 時間字符串（例如: "09:00" 或 "9:00 AM"），可選
    
    Returns:
        (start_datetime, end_datetime) 元組，格式為 ISO 8601
    """
    try:
        # 處理相對日期（今天、明天等）
        today = datetime.now()
        if '今天' in date_str or 'today' in date_str.lower():
            target_date = today
        elif '明天' in date_str or 'tomorrow' in date_str.lower():
            target_date = today + timedelta(days=1)
        elif '後天' in date_str or 'day after tomorrow' in date_str.lower():
            target_date = today + timedelta(days=2)
        else:
            # 嘗試解析日期格式
            try:
                target_date = datetime.strptime(date_str, '%Y-%m-%d')
            except:
                # 如果無法解析，使用今天
                target_date = today
        
        # 處理時間
        if time_str:
            # 嘗試解析時間
            time_formats = ['%H:%M', '%I:%M %p', '%I:%M%p']
            parsed_time = None
            for fmt in time_formats:
                try:
                    parsed_time = datetime.strptime(time_str.strip(), fmt).time()
                    break
                except:
                    continue
            
            if parsed_time:
                start_datetime = datetime.combine(target_date.date(), parsed_time)
            else:
                # 預設時間：上午 9:00
                start_datetime = datetime.combine(target_date.date(), datetime.min.time().replace(hour=9))
        else:
            # 預設時間：上午 9:00
            start_datetime = datetime.combine(target_date.date(), datetime.min.time().replace(hour=9))
        
        # 預設持續時間：1 小時
        end_datetime = start_datetime + timedelta(hours=1)
        
        # 轉換為 ISO 8601 格式（帶時區）
        timezone_offset = "+08:00"  # 台灣時區
        start_iso = start_datetime.strftime('%Y-%m-%dT%H:%M:%S') + timezone_offset
        end_iso = end_datetime.strftime('%Y-%m-%dT%H:%M:%S') + timezone_offset
        
        return start_iso, end_iso
        
    except Exception as e:
        # 如果解析失敗，使用今天上午 9:00
        today = datetime.now()
        start_datetime = datetime.combine(today.date(), datetime.min.time().replace(hour=9))
        end_datetime = start_datetime + timedelta(hours=1)
        timezone_offset = "+08:00"
        start_iso = start_datetime.strftime('%Y-%m-%dT%H:%M:%S') + timezone_offset
        end_iso = end_datetime.strftime('%Y-%m-%dT%H:%M:%S') + timezone_offset
        return start_iso, end_iso


def generate_calendar_draft(
    prompt: str
) -> tuple[dict, str, dict]:
    """
    根據用戶提示生成行事曆事件草稿（不創建）
    從單一 prompt 中提取所有資訊：事件、日期、時間、地點、參與者
    
    Args:
        prompt: 完整的用戶提示（例如："明天下午2點團隊會議，討論項目進度，地點在會議室A，參與者包括john@example.com和mary@example.com"）
    
    Returns:
        (event_dict, status_message, missing_info) 元組
        event_dict 包含: summary, start_datetime, end_datetime, description, location, attendees
        missing_info 包含缺失的資訊標記，用於 UI 顯示下拉選單
    """
    try:
        # 檢測用戶輸入的語言
        user_language = detect_language(prompt)
        
        # 獲取 LLM
        llm = get_llm()
        
        # 根據語言選擇對應的 prompt 模板
        if user_language == 'zh':
            # 中文 prompt 模板 - 從單一 prompt 中提取所有資訊
            calendar_prompt_template = (
                "你是一位專業的行事曆事件解析助手。請從以下用戶提示中提取所有行事曆事件資訊。\n\n"
                "用戶提示：{prompt}\n\n"
                "請從提示中提取以下資訊：\n"
                "1. 事件標題（summary）\n"
                "2. 日期（date，例如：2026-01-25、明天、今天、後天）\n"
                "3. 時間（time，例如：14:00、下午2點、9:00 AM）\n"
                "4. 事件描述（description，詳細說明事件的內容、目的、議程等）\n"
                "5. 事件地點（location，如果提示中有提到）\n"
                "6. 參與者郵箱（attendees，如果提示中有提到，多個用逗號分隔）\n\n"
                "請以 JSON 格式輸出，格式如下：\n"
                "{{\n"
                '  "summary": "事件標題",\n'
                '  "date": "日期（如果無法確定則為空字符串）",\n'
                '  "time": "時間（如果無法確定則為空字符串）",\n'
                '  "description": "事件描述",\n'
                '  "location": "事件地點（如果沒有則為空字符串）",\n'
                '  "attendees": "參與者郵箱，多個用逗號分隔（只包含有效的郵箱地址，格式：user@domain.com，如果沒有則為空字符串）"\n'
                "}}\n\n"
                "重要：attendees 欄位必須只包含有效的郵箱地址（格式：user@domain.com），如果提示中只有名字沒有郵箱，則留空。\n"
                "只輸出 JSON，不要其他內容。請使用中文。"
            )
        else:
            # 英文 prompt 模板
            calendar_prompt_template = (
                "You are a professional calendar event parsing assistant. Please extract all calendar event information from the following user prompt.\n\n"
                "User prompt: {prompt}\n\n"
                "Please extract the following information:\n"
                "1. Event title (summary)\n"
                "2. Date (e.g., 2026-01-25, tomorrow, today, day after tomorrow)\n"
                "3. Time (e.g., 14:00, 2:00 PM, 9:00 AM)\n"
                "4. Event description (detailed explanation of the event content, purpose, agenda, etc.)\n"
                "5. Event location (if mentioned in the prompt)\n"
                "6. Attendee emails (if mentioned in the prompt, comma-separated)\n\n"
                "Please output in JSON format as follows:\n"
                "{{\n"
                '  "summary": "Event title",\n'
                '  "date": "Date (empty string if cannot determine)",\n'
                '  "time": "Time (empty string if cannot determine)",\n'
                '  "description": "Event description",\n'
                '  "location": "Event location (empty string if not mentioned)",\n'
                '  "attendees": "Attendee emails, comma-separated (only valid email addresses in format: user@domain.com, empty string if not mentioned)"\n'
                "}}\n\n"
                "Important: The attendees field must only contain valid email addresses (format: user@domain.com). If the prompt only mentions names without emails, leave it empty.\n"
                "Output only JSON, nothing else. Please use English."
            )
        
        # 創建事件生成提示
        calendar_prompt = ChatPromptTemplate.from_template(calendar_prompt_template)
        
        # 生成事件內容
        try:
            chain = calendar_prompt | llm | StrOutputParser()
            event_content = chain.invoke({"prompt": prompt})
        except Exception as e:
            # 處理 Groq API 錯誤
            fallback_llm = handle_groq_error(e)
            if fallback_llm:
                print("   ⚠️ [CalendarAgent] Groq API 額度已用完，已切換到本地 MLX 模型")
                chain = calendar_prompt | fallback_llm | StrOutputParser()
                event_content = chain.invoke({"prompt": prompt})
            else:
                raise
        
        # 解析 JSON 響應
        try:
            import json
            # 清理可能的 markdown 代碼塊
            event_content = event_content.strip()
            if event_content.startswith('```'):
                # 移除 markdown 代碼塊標記
                lines = event_content.split('\n')
                event_content = '\n'.join(lines[1:-1])
            elif event_content.startswith('```json'):
                lines = event_content.split('\n')
                event_content = '\n'.join(lines[1:-1])
            
            event_data = json.loads(event_content)
        except json.JSONDecodeError:
            # 如果 JSON 解析失敗，使用預設值
            event_data = {
                "summary": "新事件",
                "date": "",
                "time": "",
                "description": prompt if prompt else "由 AI 生成的行事曆事件",
                "location": "",
                "attendees": ""
            }
        
        # 檢查缺失的資訊
        missing_info = {}
        if not event_data.get("date") or not event_data.get("date").strip():
            missing_info["date"] = True
        if not event_data.get("time") or not event_data.get("time").strip():
            missing_info["time"] = True
        
        # 解析日期和時間
        date_str = event_data.get("date", "").strip()
        time_str = event_data.get("time", "").strip()
        
        # 如果日期或時間缺失，使用預設值但標記為缺失
        if not date_str:
            date_str = "今天"  # 預設使用今天
        if not time_str:
            time_str = None  # 時間缺失，將在下拉選單中選擇
        
        start_datetime, end_datetime = parse_datetime(date_str, time_str)
        
        # 構建事件字典
        event_dict = {
            "summary": event_data.get("summary", "新事件"),
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "description": event_data.get("description", ""),
            "location": event_data.get("location", ""),
            "attendees": event_data.get("attendees", ""),
            "timezone": "Asia/Taipei",
            "date": date_str,  # 保留原始日期字串
            "time": time_str if time_str else ""  # 保留原始時間字串
        }
        
        status_message = "✅ 行事曆事件草稿已生成"
        if missing_info:
            missing_items = []
            if missing_info.get("date"):
                missing_items.append("日期")
            if missing_info.get("time"):
                missing_items.append("時間")
            status_message += f"，請補充以下資訊：{', '.join(missing_items)}"
        else:
            status_message += "，請檢查並修改後再創建"
        
        return event_dict, status_message, missing_info
        
    except Exception as e:
        error_msg = f"❌ 生成行事曆事件草稿時發生錯誤：{str(e)}"
        print(f"Calendar Agent 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return {}, error_msg, {}


def create_calendar_draft(event_dict: dict) -> str:
    """
    創建已編輯的行事曆事件草稿
    
    Args:
        event_dict: 包含事件資訊的字典
    
    Returns:
        創建結果消息
    """
    try:
        # 創建事件
        result = create_calendar_event.invoke({
            "summary": event_dict.get("summary", ""),
            "start_datetime": event_dict.get("start_datetime", ""),
            "end_datetime": event_dict.get("end_datetime", ""),
            "description": event_dict.get("description", ""),
            "location": event_dict.get("location", ""),
            "attendees": event_dict.get("attendees", ""),
            "timezone": event_dict.get("timezone", "Asia/Taipei")
        })
        
        return f"📅 {result}\n\n事件已成功創建！"
        
    except Exception as e:
        error_msg = f"❌ 創建行事曆事件時發生錯誤：{str(e)}"
        print(f"Calendar Tool 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return error_msg


def update_calendar_draft(event_id: str, event_dict: dict) -> str:
    """
    更新已編輯的行事曆事件
    
    Args:
        event_id: 要更新的事件 ID
        event_dict: 包含更新資訊的字典
    
    Returns:
        更新結果消息
    """
    try:
        # 更新事件
        result = update_calendar_event.invoke({
            "event_id": event_id,
            "summary": event_dict.get("summary"),
            "start_datetime": event_dict.get("start_datetime"),
            "end_datetime": event_dict.get("end_datetime"),
            "description": event_dict.get("description"),
            "location": event_dict.get("location"),
            "attendees": event_dict.get("attendees"),
            "timezone": event_dict.get("timezone", "Asia/Taipei")
        })
        
        return f"📅 {result}\n\n事件已成功更新！"
        
    except Exception as e:
        error_msg = f"❌ 更新行事曆事件時發生錯誤：{str(e)}"
        print(f"Calendar Tool 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return error_msg


def delete_calendar_draft(event_id: str) -> str:
    """
    刪除行事曆事件
    
    Args:
        event_id: 要刪除的事件 ID
    
    Returns:
        刪除結果消息
    """
    try:
        # 刪除事件
        result = delete_calendar_event.invoke({
            "event_id": event_id
        })
        
        return f"📅 {result}\n\n事件已成功刪除！"
        
    except Exception as e:
        error_msg = f"❌ 刪除行事曆事件時發生錯誤：{str(e)}"
        print(f"Calendar Tool 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return error_msg

