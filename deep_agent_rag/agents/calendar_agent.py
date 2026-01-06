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
from .calendar_reflection_agent import reflect_on_calendar_event, generate_improved_calendar_event
from ..config import MAX_REFLECTION_ITERATION
from ..tools.googlemaps_tool import enrich_location_info
from ..guidelines import get_guideline
from .calendar_validation import (
    validate_iso8601,
    is_datetime_reasonable,
    build_validation_error_message,
    request_llm_correction,
    validate_and_correct_datetime,
    validate_and_correct_attendees,
    detect_language,
    parse_datetime
)
from ..tools.calendar_tool import validate_and_clean_emails


def generate_calendar_draft(
    prompt: str,
    enable_reflection: bool = True
) -> tuple[dict, str, dict, str, bool]:
    """
    根據用戶提示生成行事曆事件草稿（不創建），並進行迭代反思評估
    從單一 prompt 中提取所有資訊：事件、日期、時間、地點、參與者
    
    Args:
        prompt: 完整的用戶提示（例如："明天下午2點團隊會議，討論項目進度，地點在會議室A，參與者包括john@example.com和mary@example.com"）
        enable_reflection: 是否啟用反思功能（默認 True）
    
    Returns:
        (event_dict, status_message, missing_info, reflection_result, was_improved) 元組
        event_dict 包含: summary, start_datetime, end_datetime, description, location, attendees
        missing_info 包含缺失的資訊標記，用於 UI 顯示下拉選單
        reflection_result: 反思結果（如果啟用反思）
        was_improved: 是否經過改進（如果啟用反思）
    """
    try:
        # 檢測用戶輸入的語言
        user_language = detect_language(prompt)
        
        # 獲取 LLM
        llm = get_llm()
        
        # 【Parlant 整合】獲取行事曆指南
        event_creation_guideline = get_guideline("calendar", "event_creation")
        time_parsing_guideline = get_guideline("calendar", "time_parsing")
        location_handling_guideline = get_guideline("calendar", "location_handling")
        
        # 獲取當前日期時間作為上下文數據（不是規則，是必要的上下文信息）
        current_datetime = datetime.now()
        current_date_str = current_datetime.strftime('%Y年%m月%d日')
        current_weekday_cn = ['週一', '週二', '週三', '週四', '週五', '週六', '週日'][current_datetime.weekday()]
        current_date_iso = current_datetime.strftime('%Y-%m-%d')
        current_weekday_en = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][current_datetime.weekday()]
        
        # 根據語言選擇對應的 prompt 模板
        if user_language == 'zh':
            # 中文 prompt 模板 - 整合指南並要求直接輸出 ISO 8601 格式
            calendar_prompt_template = (
                "你是一位專業的行事曆事件解析助手。請從以下用戶提示中提取所有行事曆事件資訊。\n\n"
                "【當前日期上下文】\n"
                "今天是：{current_date_str} ({current_weekday_cn})\n"
                "Today is: {current_date_iso} ({current_weekday_en})\n\n"
                "【事件創建指南】\n{event_creation_guideline}\n\n"
                "【時間解析指南】\n{time_parsing_guideline}\n\n"
                "【地點處理指南】\n{location_handling_guideline}\n\n"
                "用戶提示：{prompt}\n\n"
                "請嚴格遵循上述指南，直接輸出 ISO 8601 格式的日期時間。\n\n"
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
                "重要要求：\n"
                "- start_datetime 和 end_datetime 必須是完整的 ISO 8601 格式（YYYY-MM-DDTHH:MM:SS+08:00）\n"
                "- 如果無法確定日期或時間，start_datetime 和 end_datetime 可以為空字符串\n"
                "- date 和 time 欄位保留原始輸入，用於 UI 顯示和編輯\n"
                "- 預設時區為 Asia/Taipei (+08:00)\n"
                "- 預設持續時間為 1 小時\n"
                "- attendees 欄位必須只包含有效的郵箱地址（格式：user@domain.com），如果提示中只有名字沒有郵箱，則留空\n"
                "只輸出 JSON，不要其他內容。請使用中文。"
            )
        else:
            # 英文 prompt 模板 - 整合指南並要求直接輸出 ISO 8601 格式
            calendar_prompt_template = (
                "You are a professional calendar event parsing assistant. Please extract all calendar event information from the following user prompt.\n\n"
                "【Current Date Context】\n"
                "Today is: {current_date_iso} ({current_weekday_en})\n"
                "今天是：{current_date_str} ({current_weekday_cn})\n\n"
                "【Event Creation Guidelines】\n{event_creation_guideline}\n\n"
                "【Time Parsing Guidelines】\n{time_parsing_guideline}\n\n"
                "【Location Handling Guidelines】\n{location_handling_guideline}\n\n"
                "User prompt: {prompt}\n\n"
                "Please strictly follow the guidelines above and directly output ISO 8601 formatted datetime.\n\n"
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
                "Important requirements:\n"
                "- start_datetime and end_datetime must be complete ISO 8601 format (YYYY-MM-DDTHH:MM:SS+08:00)\n"
                "- If date or time cannot be determined, start_datetime and end_datetime can be empty strings\n"
                "- date and time fields preserve original input for UI display and editing\n"
                "- Default timezone is Asia/Taipei (+08:00)\n"
                "- Default duration is 1 hour\n"
                "- The attendees field must only contain valid email addresses (format: user@domain.com). If the prompt only mentions names without emails, leave it empty\n"
                "Output only JSON, nothing else. Please use English."
            )
        
        # 創建事件生成提示
        calendar_prompt = ChatPromptTemplate.from_template(calendar_prompt_template)
        
        # 生成事件內容
        try:
            chain = calendar_prompt | llm | StrOutputParser()
            event_content = chain.invoke({
                "prompt": prompt,
                "current_date_str": current_date_str,
                "current_date_iso": current_date_iso,
                "current_weekday_cn": current_weekday_cn,
                "current_weekday_en": current_weekday_en,
                "event_creation_guideline": event_creation_guideline,
                "time_parsing_guideline": time_parsing_guideline,
                "location_handling_guideline": location_handling_guideline
            })
        except Exception as e:
            # 處理 Groq API 錯誤
            fallback_llm = handle_groq_error(e)
            if fallback_llm:
                print("   ⚠️ [CalendarAgent] Groq API 額度已用完，已切換到本地 MLX 模型")
                chain = calendar_prompt | fallback_llm | StrOutputParser()
                event_content = chain.invoke({
                    "prompt": prompt,
                    "current_date_str": current_date_str,
                    "current_date_iso": current_date_iso,
                    "current_weekday_cn": current_weekday_cn,
                    "current_weekday_en": current_weekday_en,
                    "event_creation_guideline": event_creation_guideline,
                    "time_parsing_guideline": time_parsing_guideline,
                    "location_handling_guideline": location_handling_guideline
                })
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
        
        # 【二輪修正機制】驗證並修正 LLM 輸出的日期時間
        # 優先使用 LLM 直接輸出的 ISO 8601 格式，如果無效則請求 LLM 修正（而非直接 fallback 到 Python）
        start_datetime, end_datetime, date_str, time_str = validate_and_correct_datetime(
            llm_output=event_data,
            current_datetime=current_datetime,
            prompt=prompt,
            user_language=user_language,
            max_retries=2,
            parse_datetime_fallback=parse_datetime
        )
        
        # 檢查缺失的資訊（用於 UI 顯示）
        missing_info = {}
        if not date_str or not date_str.strip():
            missing_info["date"] = True
        if not time_str or not time_str.strip():
            missing_info["time"] = True
        
        # 【Google Maps 整合】驗證並豐富地點資訊
        location = event_data.get("location", "").strip()
        location_info = None
        location_suggestion = ""
        
        if location:
            try:
                # 將 start_datetime 轉換為 datetime 對象用於計算交通時間
                from datetime import datetime as dt
                try:
                    event_dt = dt.fromisoformat(start_datetime.replace('+08:00', ''))
                except:
                    event_dt = None
                
                # 豐富地點資訊（驗證地址、計算交通時間）
                location_info = enrich_location_info(location, event_dt)
                
                # 如果地址驗證成功，使用標準化地址
                if location_info.get("validated"):
                    location = location_info.get("standardized_address", location)
                    location_suggestion = location_info.get("suggestion", "")
                    print(f"   🗺️ [GoogleMaps] 地點已驗證並標準化：{location}")
                    if location_info.get("travel_time_info"):
                        travel_info = location_info["travel_time_info"]
                        print(f"   🗺️ [GoogleMaps] 交通時間：{travel_info.get('duration_text', 'N/A')}")
                else:
                    # 地址驗證失敗，保留原始地址但記錄警告
                    location_suggestion = location_info.get("suggestion", "")
                    print(f"   ⚠️ [GoogleMaps] 地點驗證失敗：{location_suggestion}")
            except Exception as e:
                # Google Maps API 調用失敗，不影響事件創建，只記錄警告
                print(f"   ⚠️ [GoogleMaps] 地點資訊豐富化失敗：{e}，將使用原始地址")
                location_suggestion = f"⚠️ 無法驗證地址（{str(e)}），將使用原始地址"
        
        # 【二輪修正機制】驗證並修正 LLM 輸出的參與者郵箱
        # 優先使用 LLM 根據指南提取和驗證，而非直接使用 Python 正則
        attendees = validate_and_correct_attendees(
            llm_output=event_data,
            prompt=prompt,
            user_language=user_language,
            max_retries=2,
            validate_and_clean_emails_fallback=validate_and_clean_emails
        )
        
        # 構建事件字典
        event_dict = {
            "summary": event_data.get("summary", "新事件"),
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "description": event_data.get("description", ""),
            "location": location,  # 使用標準化後的地址（如果驗證成功）
            "attendees": attendees,  # 使用驗證和修正後的參與者郵箱
            "timezone": "Asia/Taipei",
            "date": date_str,  # 保留原始日期字串
            "time": time_str if time_str else "",  # 保留原始時間字串
            "location_info": location_info,  # 保存完整的地點資訊（用於 UI 顯示）
            "location_suggestion": location_suggestion  # 保存地點建議訊息
        }
        
        # 【迭代反思功能】不斷反思直到滿意為止
        reflection_result = ""
        was_improved = False
        all_reflections = []  # 記錄所有反思結果
        
        if enable_reflection:
            try:
                current_event_dict = event_dict.copy()
                current_iteration = 0
                
                # 迭代反思循環：最多進行 MAX_REFLECTION_ITERATION 輪
                while current_iteration < MAX_REFLECTION_ITERATION:
                    try:
                        print(f"   🔍 [CalendarReflection] 第 {current_iteration + 1} 輪反思評估...")
                        reflection_text, improvement_suggestions, needs_revision = reflect_on_calendar_event(
                            prompt, current_event_dict
                        )
                        
                        # 記錄本輪反思結果
                        all_reflections.append({
                            "iteration": current_iteration + 1,
                            "reflection": reflection_text,
                            "suggestions": improvement_suggestions,
                            "needs_revision": needs_revision
                        })
                        
                        # 檢查是否有改進建議
                        has_meaningful_suggestions = (
                            improvement_suggestions and 
                            improvement_suggestions.strip() and 
                            len(improvement_suggestions.strip()) > 20  # 至少要有一定長度的建議
                        )
                        
                        if has_meaningful_suggestions:
                            print(f"   🔄 [CalendarReflection] 第 {current_iteration + 1} 輪：檢測到改進建議，正在生成改進版本...")
                            try:
                                improved_event_dict = generate_improved_calendar_event(
                                    prompt, current_event_dict, improvement_suggestions
                                )
                                
                                # 對改進後的版本再次進行反思評估
                                if current_iteration < MAX_REFLECTION_ITERATION - 1:  # 如果不是最後一輪
                                    print(f"   🔍 [CalendarReflection] 評估改進後的版本...")
                                    next_reflection_text, next_suggestions, next_needs_revision = reflect_on_calendar_event(
                                        prompt, improved_event_dict
                                    )
                                    
                                    # 檢查改進後的版本是否滿意
                                    has_next_suggestions = (
                                        next_suggestions and 
                                        next_suggestions.strip() and 
                                        len(next_suggestions.strip()) > 20
                                    )
                                    
                                    if not has_next_suggestions:
                                        # 改進後的版本沒有新的改進建議，說明已經滿意
                                        print(f"   ✅ [CalendarReflection] 第 {current_iteration + 1} 輪改進後，AI 認為質量已達標")
                                        current_event_dict = improved_event_dict
                                        was_improved = True
                                        all_reflections.append({
                                            "iteration": current_iteration + 1,
                                            "reflection": next_reflection_text,
                                            "suggestions": "無，質量已達標",
                                            "needs_revision": False
                                        })
                                        break  # 滿意了，退出循環
                                    else:
                                        # 還有改進空間，繼續下一輪
                                        print(f"   🔄 [CalendarReflection] 第 {current_iteration + 1} 輪改進後仍有改進空間，繼續反思...")
                                        current_event_dict = improved_event_dict
                                        was_improved = True
                                        current_iteration += 1
                                        continue
                                else:
                                    # 最後一輪，直接使用改進版本
                                    print(f"   ✅ [CalendarReflection] 已達最大反思次數，使用最終改進版本")
                                    current_event_dict = improved_event_dict
                                    was_improved = True
                                    break
                                    
                            except Exception as e:
                                print(f"   ⚠️ [CalendarReflection] 生成改進版本失敗: {e}")
                                break
                        else:
                            # 沒有改進建議，說明已經滿意
                            print(f"   ✅ [CalendarReflection] 第 {current_iteration + 1} 輪：事件質量已達標，無需改進")
                            break
                            
                    except Exception as e:
                        print(f"   ⚠️ [CalendarReflection] 第 {current_iteration + 1} 輪反思過程發生錯誤: {e}")
                        break
                
                # 使用最終版本
                event_dict = current_event_dict
                
                # 重新檢查缺失的資訊（因為改進後可能改變了日期時間）
                missing_info = {}
                if not event_dict.get("date") or not event_dict.get("date").strip():
                    missing_info["date"] = True
                if not event_dict.get("time") or not event_dict.get("time").strip():
                    missing_info["time"] = True
                
                # 合併所有反思結果
                if all_reflections:
                    reflection_parts = []
                    for r in all_reflections:
                        iteration_num = r['iteration']
                        reflection_parts.append(f"【第 {iteration_num} 輪反思評估】\n{r['reflection']}")
                        if r.get('suggestions') and r['suggestions'] != "無，質量已達標":
                            reflection_parts.append(f"\n【改進建議】\n{r['suggestions']}")
                    
                    reflection_result = "\n\n".join(reflection_parts)
                else:
                    reflection_result = "反思過程未產生結果"
                
                # 生成狀態消息
                if missing_info:
                    missing_items = []
                    if missing_info.get("date"):
                        missing_items.append("日期")
                    if missing_info.get("time"):
                        missing_items.append("時間")
                    if was_improved:
                        total_iterations = len([r for r in all_reflections if r.get('suggestions') and r['suggestions'] != "無，質量已達標"])
                        status_message = f"✅ 行事曆事件草稿已生成並經過 {total_iterations} 輪 AI 反思優化，請補充以下資訊：{', '.join(missing_items)}"
                    else:
                        status_message = f"✅ 行事曆事件草稿已生成（AI 反思評估：質量良好），請補充以下資訊：{', '.join(missing_items)}"
                else:
                    if was_improved:
                        total_iterations = len([r for r in all_reflections if r.get('suggestions') and r['suggestions'] != "無，質量已達標"])
                        status_message = f"✅ 行事曆事件草稿已生成並經過 {total_iterations} 輪 AI 反思優化，請檢查並修改後再創建"
                    else:
                        status_message = "✅ 行事曆事件草稿已生成（AI 反思評估：質量良好），請檢查並修改後再創建"
                    
            except Exception as e:
                print(f"   ⚠️ [CalendarReflection] 反思過程發生錯誤: {e}")
                reflection_result = f"反思過程發生錯誤：{str(e)}"
                # 使用原始狀態消息
        if missing_info:
            missing_items = []
            if missing_info.get("date"):
                missing_items.append("日期")
            if missing_info.get("time"):
                missing_items.append("時間")
                status_message = f"✅ 行事曆事件草稿已生成，請補充以下資訊：{', '.join(missing_items)}"
            else:
                status_message = "✅ 行事曆事件草稿已生成，請檢查並修改後再創建"
        else:
            # 未啟用反思功能，使用原始邏輯
            if missing_info:
                missing_items = []
                if missing_info.get("date"):
                    missing_items.append("日期")
                if missing_info.get("time"):
                    missing_items.append("時間")
                status_message = f"✅ 行事曆事件草稿已生成，請補充以下資訊：{', '.join(missing_items)}"
            else:
                status_message = "✅ 行事曆事件草稿已生成，請檢查並修改後再創建"
        
        return event_dict, status_message, missing_info, reflection_result, was_improved
        
    except Exception as e:
        error_msg = f"❌ 生成行事曆事件草稿時發生錯誤：{str(e)}"
        print(f"Calendar Agent 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return {}, error_msg, {}, "", False


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

