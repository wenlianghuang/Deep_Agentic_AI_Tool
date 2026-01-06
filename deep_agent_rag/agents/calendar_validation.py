"""
Calendar Validation Utilities
行事曆驗證和修正工具函數
提供日期時間驗證和 LLM 修正機制
"""
import re
from datetime import datetime, timedelta
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..utils.llm_utils import get_llm, handle_groq_error
from ..guidelines import get_guideline


def parse_datetime(date_str: str, time_str: str = None) -> tuple[str, str]:
    """
    解析日期和時間，生成 ISO 8601 格式的開始和結束時間
    增強版：支援下週X格式
    
    ⚠️ 注意：此函數現在僅作為「最後的安全網」使用。
    優先使用 LLM 根據 Parlant 指南進行計算和修正。
    只有在所有 LLM 修正嘗試都失敗時才會調用此函數。
    
    Args:
        date_str: 日期字符串（例如: "2026-01-25"、"明天"、"下週三"）
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
        # 處理下週X格式（中文）
        elif '下週' in date_str or '下星期' in date_str:
            # 星期對應：週一=0, 週二=1, 週三=2, 週四=3, 週五=4, 週六=5, 週日=6
            weekdays_cn = {
                '一': 0, '二': 1, '三': 2, '四': 3, 
                '五': 4, '六': 5, '日': 6, '天': 6
            }
            matched = False
            for day_char, day_num in weekdays_cn.items():
                if day_char in date_str:
                    # 計算下一個指定的星期幾
                    # 如果今天是週三，說"下週三"是指下一個週三（7天後），不是今天
                    days_ahead = day_num - today.weekday()
                    if days_ahead <= 0:  # 如果這個星期幾已經過了，就找下週的
                        days_ahead += 7
                    target_date = today + timedelta(days=days_ahead)
                    matched = True
                    break
            
            if not matched:
                # 如果沒有匹配到，預設為下週一
                days_ahead = (0 - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                target_date = today + timedelta(days=days_ahead)
        # 處理 next Monday/Tuesday 等格式（英文）
        elif 'next' in date_str.lower():
            weekdays_en = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            date_lower = date_str.lower()
            matched = False
            for day_name, day_num in weekdays_en.items():
                if day_name in date_lower:
                    # 計算下一個指定的星期幾
                    days_ahead = day_num - today.weekday()
                    if days_ahead <= 0:
                        days_ahead += 7
                    target_date = today + timedelta(days=days_ahead)
                    matched = True
                    break
            
            if not matched:
                # 如果沒有匹配到，預設為下週一
                days_ahead = (0 - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                target_date = today + timedelta(days=days_ahead)
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


def validate_iso8601(dt_str: str) -> bool:
    """
    驗證 ISO 8601 格式（純格式驗證，不重新計算）
    
    Args:
        dt_str: 日期時間字符串
    
    Returns:
        是否為有效的 ISO 8601 格式
    """
    if not dt_str:
        return False
    try:
        from datetime import datetime as dt
        # 處理時區格式
        dt_str_clean = dt_str.replace('+08:00', '+08:00')
        dt.fromisoformat(dt_str_clean)
        return True
    except:
        return False


def is_datetime_reasonable(start_datetime: str, end_datetime: str) -> bool:
    """
    驗證日期時間的合理性（邏輯驗證，不重新計算）
    
    Args:
        start_datetime: 開始時間（ISO 8601 格式）
        end_datetime: 結束時間（ISO 8601 格式）
    
    Returns:
        是否合理（結束時間晚於開始時間）
    """
    try:
        from datetime import datetime as dt
        start_dt = dt.fromisoformat(start_datetime.replace('+08:00', ''))
        end_dt = dt.fromisoformat(end_datetime.replace('+08:00', ''))
        return end_dt > start_dt
    except:
        return False


def build_validation_error_message(
    start_datetime: str,
    end_datetime: str,
    current_datetime: datetime,
    date_str: str = "",
    time_str: str = ""
) -> str:
    """
    構建驗證錯誤訊息，幫助 LLM 理解問題
    
    Args:
        start_datetime: 原始開始時間字符串
        end_datetime: 原始結束時間字符串
        current_datetime: 當前日期時間
        date_str: 原始日期字符串
        time_str: 原始時間字符串
    
    Returns:
        錯誤訊息
    """
    errors = []
    
    if not validate_iso8601(start_datetime):
        errors.append(f"開始時間格式無效：'{start_datetime}'（應為 ISO 8601 格式，例如：2026-01-25T14:00:00+08:00）")
    if not validate_iso8601(end_datetime):
        errors.append(f"結束時間格式無效：'{end_datetime}'（應為 ISO 8601 格式，例如：2026-01-25T15:00:00+08:00）")
    
    if validate_iso8601(start_datetime) and validate_iso8601(end_datetime):
        if not is_datetime_reasonable(start_datetime, end_datetime):
            errors.append(f"結束時間必須晚於開始時間（開始：{start_datetime}，結束：{end_datetime}）")
    
    if date_str:
        errors.append(f"原始日期字符串：'{date_str}'")
    if time_str:
        errors.append(f"原始時間字符串：'{time_str}'")
    
    current_date_str = current_datetime.strftime('%Y年%m月%d日')
    current_weekday_cn = ['週一', '週二', '週三', '週四', '週五', '週六', '週日'][current_datetime.weekday()]
    errors.append(f"今天是：{current_date_str} ({current_weekday_cn})")
    
    return "\n".join(errors)


def detect_language(text: str) -> str:
    """
    檢測文本的主要語言（中文或英文）
    
    Args:
        text: 輸入文本
    
    Returns:
        'zh' 或 'en'
    """
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    if chinese_pattern.search(text):
        return 'zh'
    else:
        return 'en'


def request_llm_correction(
    prompt: str,
    original_output: dict,
    error_message: str,
    current_datetime: datetime,
    user_language: str = 'zh'
) -> dict:
    """
    請求 LLM 修正日期時間格式錯誤（二輪修正機制）
    
    Args:
        prompt: 用戶原始提示
        original_output: LLM 的原始輸出
        error_message: 驗證錯誤訊息
        current_datetime: 當前日期時間
        user_language: 用戶語言
    
    Returns:
        修正後的事件數據字典
    """
    llm = get_llm()
    
    # 獲取時間解析指南
    time_parsing_guideline = get_guideline("calendar", "time_parsing")
    
    # 格式化當前日期上下文
    current_date_str = current_datetime.strftime('%Y年%m月%d日')
    current_weekday_cn = ['週一', '週二', '週三', '週四', '週五', '週六', '週日'][current_datetime.weekday()]
    current_date_iso = current_datetime.strftime('%Y-%m-%d')
    current_weekday_en = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][current_datetime.weekday()]
    
    if user_language == 'zh':
        correction_prompt_template = (
            "你剛才輸出的日期時間格式有誤。請根據「時間解析指南」重新計算並輸出正確的 ISO 8601 格式。\n\n"
            "【當前日期上下文】\n"
            "今天是：{current_date_str} ({current_weekday_cn})\n"
            "Today is: {current_date_iso} ({current_weekday_en})\n\n"
            "【時間解析指南】\n{time_parsing_guideline}\n\n"
            "【用戶原始提示】\n{prompt}\n\n"
            "【你剛才的輸出】\n"
            "開始時間：{original_start}\n"
            "結束時間：{original_end}\n"
            "日期字符串：{original_date}\n"
            "時間字符串：{original_time}\n\n"
            "【驗證錯誤訊息】\n{error_message}\n\n"
            "請仔細閱讀時間解析指南，特別是「下週X」的計算規則，然後重新計算並輸出正確的 ISO 8601 格式。\n\n"
            "請以 JSON 格式輸出，格式如下：\n"
            "{{\n"
            '  "start_datetime": "ISO 8601 格式的開始時間（例如：2026-01-25T14:00:00+08:00）",\n'
            '  "end_datetime": "ISO 8601 格式的結束時間（例如：2026-01-25T15:00:00+08:00）",\n'
            '  "date": "原始日期字符串（用於 UI 顯示）",\n'
            '  "time": "原始時間字符串（用於 UI 顯示）"\n'
            "}}\n\n"
            "重要：必須直接輸出 ISO 8601 格式，不要輸出原始字符串。只輸出 JSON，不要其他內容。"
        )
    else:
        correction_prompt_template = (
            "The datetime format you just output is incorrect. Please recalculate and output the correct ISO 8601 format according to the 'Time Parsing Guidelines'.\n\n"
            "【Current Date Context】\n"
            "Today is: {current_date_iso} ({current_weekday_en})\n"
            "今天是：{current_date_str} ({current_weekday_cn})\n\n"
            "【Time Parsing Guidelines】\n{time_parsing_guideline}\n\n"
            "【User's Original Prompt】\n{prompt}\n\n"
            "【Your Previous Output】\n"
            "Start Time: {original_start}\n"
            "End Time: {original_end}\n"
            "Date String: {original_date}\n"
            "Time String: {original_time}\n\n"
            "【Validation Error Message】\n{error_message}\n\n"
            "Please carefully read the time parsing guidelines, especially the calculation rules for 'next X', then recalculate and output the correct ISO 8601 format.\n\n"
            "Please output in JSON format as follows:\n"
            "{{\n"
            '  "start_datetime": "ISO 8601 formatted start time (e.g., 2026-01-25T14:00:00+08:00)",\n'
            '  "end_datetime": "ISO 8601 formatted end time (e.g., 2026-01-25T15:00:00+08:00)",\n'
            '  "date": "Original date string (for UI display)",\n'
            '  "time": "Original time string (for UI display)"\n'
            "}}\n\n"
            "Important: You must directly output ISO 8601 format, not raw strings. Output only JSON, nothing else."
        )
    
    correction_prompt = ChatPromptTemplate.from_template(correction_prompt_template)
    
    try:
        chain = correction_prompt | llm | StrOutputParser()
        corrected_content = chain.invoke({
            "prompt": prompt,
            "current_date_str": current_date_str,
            "current_date_iso": current_date_iso,
            "current_weekday_cn": current_weekday_cn,
            "current_weekday_en": current_weekday_en,
            "time_parsing_guideline": time_parsing_guideline,
            "original_start": original_output.get("start_datetime", ""),
            "original_end": original_output.get("end_datetime", ""),
            "original_date": original_output.get("date", ""),
            "original_time": original_output.get("time", ""),
            "error_message": error_message
        })
    except Exception as e:
        fallback_llm = handle_groq_error(e)
        if fallback_llm:
            print("   ⚠️ [CalendarValidation] Groq API 額度已用完，已切換到本地 MLX 模型（修正階段）")
            chain = correction_prompt | fallback_llm | StrOutputParser()
            corrected_content = chain.invoke({
                "prompt": prompt,
                "current_date_str": current_date_str,
                "current_date_iso": current_date_iso,
                "current_weekday_cn": current_weekday_cn,
                "current_weekday_en": current_weekday_en,
                "time_parsing_guideline": time_parsing_guideline,
                "original_start": original_output.get("start_datetime", ""),
                "original_end": original_output.get("end_datetime", ""),
                "original_date": original_output.get("date", ""),
                "original_time": original_output.get("time", ""),
                "error_message": error_message
            })
        else:
            raise
    
    # 解析 JSON 響應
    import json
    try:
        corrected_content = corrected_content.strip()
        if corrected_content.startswith('```'):
            lines = corrected_content.split('\n')
            corrected_content = '\n'.join(lines[1:-1])
        elif corrected_content.startswith('```json'):
            lines = corrected_content.split('\n')
            corrected_content = '\n'.join(lines[1:-1])
        
        corrected_data = json.loads(corrected_content)
        return corrected_data
    except json.JSONDecodeError:
        # 如果 JSON 解析失敗，返回原始輸出
        print("   ⚠️ [CalendarValidation] 修正階段的 JSON 解析失敗")
        return original_output


def validate_and_correct_datetime(
    llm_output: dict,
    current_datetime: datetime,
    prompt: str,
    user_language: str = 'zh',
    max_retries: int = 2,
    parse_datetime_fallback=None
) -> tuple[str, str, str, str]:
    """
    驗證並修正 LLM 輸出的日期時間（使用 LLM 修正，而非 Python 計算）
    
    Args:
        llm_output: LLM 的原始輸出字典
        current_datetime: 當前日期時間
        prompt: 用戶原始提示
        user_language: 用戶語言
        max_retries: 最大重試次數
        parse_datetime_fallback: 最後的安全網函數（可選，用於 fallback）
    
    Returns:
        (start_datetime, end_datetime, date_str, time_str) 元組
    """
    start_datetime = llm_output.get("start_datetime", "").strip()
    end_datetime = llm_output.get("end_datetime", "").strip()
    date_str = llm_output.get("date", "").strip()
    time_str = llm_output.get("time", "").strip()
    
    # 第一層：格式驗證（不計算，只檢查格式）
    if validate_iso8601(start_datetime) and validate_iso8601(end_datetime):
        # 第二層：合理性驗證（檢查邏輯，不重新計算）
        if is_datetime_reasonable(start_datetime, end_datetime):
            return start_datetime, end_datetime, date_str, time_str
    
    # 如果驗證失敗，使用 LLM 修正（而非 Python fallback）
    print(f"   🔄 [CalendarValidation] 檢測到日期時間格式錯誤，開始 LLM 修正流程（最多 {max_retries} 次嘗試）...")
    
    for attempt in range(max_retries):
        error_msg = build_validation_error_message(
            start_datetime, end_datetime, current_datetime, date_str, time_str
        )
        
        print(f"   🔄 [CalendarValidation] 第 {attempt + 1} 次修正嘗試...")
        corrected = request_llm_correction(
            prompt=prompt,
            original_output=llm_output,
            error_message=error_msg,
            current_datetime=current_datetime,
            user_language=user_language
        )
        
        corrected_start = corrected.get("start_datetime", "").strip()
        corrected_end = corrected.get("end_datetime", "").strip()
        
        if validate_iso8601(corrected_start) and validate_iso8601(corrected_end):
            if is_datetime_reasonable(corrected_start, corrected_end):
                print(f"   ✅ [CalendarValidation] 第 {attempt + 1} 次修正成功！")
                return (
                    corrected_start,
                    corrected_end,
                    corrected.get("date", date_str).strip(),
                    corrected.get("time", time_str).strip()
                )
        
        # 更新為修正後的版本，準備下一輪
        start_datetime = corrected_start
        end_datetime = corrected_end
        date_str = corrected.get("date", date_str).strip()
        time_str = corrected.get("time", time_str).strip()
        llm_output = corrected
    
    # 最後的安全網：只有在所有 LLM 修正都失敗時才使用 Python
    # 但應該記錄警告，因為這表示指南可能有問題或 LLM 無法理解
    if parse_datetime_fallback:
        print("   ⚠️ [CalendarValidation] 所有 LLM 修正嘗試失敗，使用最後的安全網（Python 解析）")
        print("   ⚠️ [CalendarValidation] 這可能表示時間解析指南需要改進，或 LLM 無法正確理解日期計算規則")
        
        if not date_str:
            date_str = "今天"
        if not time_str:
            time_str = None
        
        fallback_start, fallback_end = parse_datetime_fallback(date_str, time_str)
        return fallback_start, fallback_end, date_str, time_str if time_str else ""
    else:
        # 如果沒有提供 fallback，使用內部的 parse_datetime
        print("   ⚠️ [CalendarValidation] 所有 LLM 修正嘗試失敗，使用內部的 parse_datetime 作為安全網")
        
        if not date_str:
            date_str = "今天"
        if not time_str:
            time_str = None
        
        fallback_start, fallback_end = parse_datetime(date_str, time_str)
        return fallback_start, fallback_end, date_str, time_str if time_str else ""


def is_valid_attendees_format(attendees_str: str) -> bool:
    """
    驗證參與者郵箱格式（簡單驗證，不重新提取）
    
    Args:
        attendees_str: 參與者郵箱字符串，多個用逗號分隔
    
    Returns:
        是否包含至少一個有效的郵箱格式
    """
    if not attendees_str or not attendees_str.strip():
        return True  # 空字符串視為有效（表示沒有參與者）
    
    # 郵箱正則表達式
    email_pattern = re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )
    
    # 檢查是否包含至少一個郵箱格式
    emails = re.split(r'[,;\s]+', attendees_str.strip())
    for email in emails:
        email = email.strip().strip('"').strip("'")
        if email and email_pattern.search(email):
            return True
    
    return False


def build_attendees_validation_error_message(
    attendees_str: str,
    prompt: str
) -> str:
    """
    構建參與者郵箱驗證錯誤訊息，幫助 LLM 理解問題
    
    Args:
        attendees_str: 原始參與者字符串
        prompt: 用戶原始提示
    
    Returns:
        錯誤訊息
    """
    errors = []
    
    errors.append(f"參與者郵箱格式無效：'{attendees_str}'")
    errors.append("要求：")
    errors.append("- 只包含有效的郵箱地址（格式：user@domain.com）")
    errors.append("- 多個郵箱用逗號分隔")
    errors.append("- 如果提示中只有名字沒有郵箱，則留空")
    errors.append("- 可以從 'John <john@example.com>' 格式中提取郵箱")
    errors.append(f"\n用戶原始提示：{prompt}")
    
    return "\n".join(errors)


def request_llm_attendees_correction(
    prompt: str,
    original_output: dict,
    error_message: str,
    user_language: str = 'zh'
) -> dict:
    """
    請求 LLM 修正參與者郵箱格式錯誤（二輪修正機制）
    
    Args:
        prompt: 用戶原始提示
        original_output: LLM 的原始輸出
        error_message: 驗證錯誤訊息
        user_language: 用戶語言
    
    Returns:
        修正後的事件數據字典
    """
    llm = get_llm()
    
    # 獲取事件創建指南（包含參與者處理規則）
    event_creation_guideline = get_guideline("calendar", "event_creation")
    
    if user_language == 'zh':
        correction_prompt_template = (
            "你剛才輸出的參與者郵箱格式有誤。請根據「事件創建指南」重新提取和驗證參與者郵箱。\n\n"
            "【事件創建指南】\n{event_creation_guideline}\n\n"
            "【用戶原始提示】\n{prompt}\n\n"
            "【你剛才的輸出】\n"
            "參與者：{original_attendees}\n\n"
            "【驗證錯誤訊息】\n{error_message}\n\n"
            "請仔細閱讀事件創建指南中關於「參與者處理」的部分，然後重新提取和輸出正確的郵箱地址。\n\n"
            "請以 JSON 格式輸出，格式如下：\n"
            "{{\n"
            '  "attendees": "參與者郵箱，多個用逗號分隔（只包含有效的郵箱地址，格式：user@domain.com，如果沒有則為空字符串）"\n'
            "}}\n\n"
            "重要要求：\n"
            "- 只提取有效的郵箱地址（格式：user@domain.com）\n"
            "- 如果提示中只有名字沒有郵箱，則留空\n"
            "- 可以從 'John <john@example.com>' 格式中提取郵箱\n"
            "- 多個郵箱用逗號分隔\n"
            "只輸出 JSON，不要其他內容。"
        )
    else:
        correction_prompt_template = (
            "The attendees email format you just output is incorrect. Please re-extract and validate attendee emails according to the 'Event Creation Guidelines'.\n\n"
            "【Event Creation Guidelines】\n{event_creation_guideline}\n\n"
            "【User's Original Prompt】\n{prompt}\n\n"
            "【Your Previous Output】\n"
            "Attendees: {original_attendees}\n\n"
            "【Validation Error Message】\n{error_message}\n\n"
            "Please carefully read the 'Attendee Handling' section in the Event Creation Guidelines, then re-extract and output the correct email addresses.\n\n"
            "Please output in JSON format as follows:\n"
            "{{\n"
            '  "attendees": "Attendee emails, comma-separated (only valid email addresses in format: user@domain.com, empty string if not mentioned)"\n'
            "}}\n\n"
            "Important requirements:\n"
            "- Only extract valid email addresses (format: user@domain.com)\n"
            "- If the prompt only mentions names without emails, leave it empty\n"
            "- Can extract emails from formats like 'John <john@example.com>'\n"
            "- Multiple emails separated by commas\n"
            "Output only JSON, nothing else."
        )
    
    correction_prompt = ChatPromptTemplate.from_template(correction_prompt_template)
    
    try:
        chain = correction_prompt | llm | StrOutputParser()
        corrected_content = chain.invoke({
            "prompt": prompt,
            "event_creation_guideline": event_creation_guideline,
            "original_attendees": original_output.get("attendees", ""),
            "error_message": error_message
        })
    except Exception as e:
        fallback_llm = handle_groq_error(e)
        if fallback_llm:
            print("   ⚠️ [CalendarValidation] Groq API 額度已用完，已切換到本地 MLX 模型（參與者修正階段）")
            chain = correction_prompt | fallback_llm | StrOutputParser()
            corrected_content = chain.invoke({
                "prompt": prompt,
                "event_creation_guideline": event_creation_guideline,
                "original_attendees": original_output.get("attendees", ""),
                "error_message": error_message
            })
        else:
            raise
    
    # 解析 JSON 響應
    import json
    try:
        corrected_content = corrected_content.strip()
        if corrected_content.startswith('```'):
            lines = corrected_content.split('\n')
            corrected_content = '\n'.join(lines[1:-1])
        elif corrected_content.startswith('```json'):
            lines = corrected_content.split('\n')
            corrected_content = '\n'.join(lines[1:-1])
        
        corrected_data = json.loads(corrected_content)
        return corrected_data
    except json.JSONDecodeError:
        # 如果 JSON 解析失敗，返回原始輸出
        print("   ⚠️ [CalendarValidation] 參與者修正階段的 JSON 解析失敗")
        return original_output


def validate_and_correct_attendees(
    llm_output: dict,
    prompt: str,
    user_language: str = 'zh',
    max_retries: int = 2,
    validate_and_clean_emails_fallback=None
) -> str:
    """
    驗證並修正 LLM 輸出的參與者郵箱（使用 LLM 修正，而非 Python 計算）
    
    Args:
        llm_output: LLM 的原始輸出字典
        prompt: 用戶原始提示
        user_language: 用戶語言
        max_retries: 最大重試次數
        validate_and_clean_emails_fallback: 最後的安全網函數（可選，用於 fallback）
    
    Returns:
        修正後的參與者郵箱字符串
    """
    attendees_str = llm_output.get("attendees", "").strip()
    
    # 第一層：格式驗證（檢查是否包含有效的郵箱格式）
    if is_valid_attendees_format(attendees_str):
        return attendees_str
    
    # 如果驗證失敗，使用 LLM 修正（而非 Python fallback）
    print(f"   🔄 [CalendarValidation] 檢測到參與者郵箱格式錯誤，開始 LLM 修正流程（最多 {max_retries} 次嘗試）...")
    
    for attempt in range(max_retries):
        error_msg = build_attendees_validation_error_message(attendees_str, prompt)
        
        print(f"   🔄 [CalendarValidation] 第 {attempt + 1} 次修正嘗試（參與者郵箱）...")
        corrected = request_llm_attendees_correction(
            prompt=prompt,
            original_output=llm_output,
            error_message=error_msg,
            user_language=user_language
        )
        
        corrected_attendees = corrected.get("attendees", "").strip()
        
        if is_valid_attendees_format(corrected_attendees):
            print(f"   ✅ [CalendarValidation] 第 {attempt + 1} 次修正成功（參與者郵箱）！")
            return corrected_attendees
        
        # 更新為修正後的版本，準備下一輪
        attendees_str = corrected_attendees
        llm_output = corrected
    
    # 最後的安全網：只有在所有 LLM 修正都失敗時才使用 Python
    if validate_and_clean_emails_fallback:
        print("   ⚠️ [CalendarValidation] 所有 LLM 修正嘗試失敗（參與者郵箱），使用最後的安全網（Python 正則）")
        print("   ⚠️ [CalendarValidation] 這可能表示參與者處理指南需要改進，或 LLM 無法正確理解郵箱提取規則")
        
        # 使用 Python fallback 清理郵箱
        valid_emails = validate_and_clean_emails_fallback(attendees_str)
        if valid_emails:
            return ", ".join(valid_emails)
        else:
            return ""  # 如果所有郵箱都無效，返回空字符串
    else:
        # 如果沒有提供 fallback，返回空字符串（讓上層處理）
        print("   ⚠️ [CalendarValidation] 所有 LLM 修正嘗試失敗（參與者郵箱），且未提供 fallback 函數")
        return ""

