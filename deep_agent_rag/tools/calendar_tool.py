"""
Calendar 工具
提供行事曆事件管理功能（使用 Google Calendar API）
"""
import os
import re
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
from langchain_core.tools import tool

from ..config import (
    CALENDAR_CREDENTIALS_FILE,
    CALENDAR_TOKEN_FILE,
    CALENDAR_SCOPES,
    GMAIL_SCOPES
)


def validate_and_clean_emails(attendees_str: str) -> list[str]:
    """
    驗證和清理參與者郵箱列表
    自動將 Gmail 用戶名補全為完整郵箱格式
    
    Args:
        attendees_str: 參與者郵箱字符串，多個用逗號分隔
    
    Returns:
        有效的郵箱列表
    """
    if not attendees_str or not attendees_str.strip():
        return []
    
    # 郵箱正則表達式（基本驗證）
    email_pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    # Gmail 用戶名模式（只包含字母、數字、點、下劃線、加號、減號，且沒有 @ 符號）
    gmail_username_pattern = re.compile(
        r'^[a-zA-Z0-9._+-]+$'
    )
    
    valid_emails = []
    # 分割郵箱（支援逗號、分號、空格分隔）
    emails = re.split(r'[,;\s]+', attendees_str.strip())
    
    for email in emails:
        email = email.strip()
        if not email:
            continue
        
        # 移除可能的引號
        email = email.strip('"').strip("'").strip()
        
        # 驗證郵箱格式
        if email_pattern.match(email):
            valid_emails.append(email)
        else:
            # 如果格式不正確，嘗試提取郵箱（例如從 "John <john@example.com>" 中提取）
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', email)
            if email_match:
                valid_emails.append(email_match.group())
            # 如果看起來像 Gmail 用戶名（沒有 @ 符號，且符合用戶名格式），自動補全 @gmail.com
            elif '@' not in email and gmail_username_pattern.match(email):
                # 自動補全為 Gmail 郵箱
                gmail_email = f"{email}@gmail.com"
                valid_emails.append(gmail_email)
                print(f"ℹ️ 自動將用戶名 '{email}' 補全為 '{gmail_email}'")
    
    return valid_emails


def get_calendar_service():
    """
    獲取 Google Calendar API 服務實例
    
    Returns:
        Calendar API 服務對象
    """
    creds = None
    
    # 合併 Gmail 和 Calendar 的 scopes（因為共用同一個 token.json）
    # 使用 set 去重，確保 scopes 唯一
    combined_scopes = list(set(CALENDAR_SCOPES + GMAIL_SCOPES))
    
    # 檢查是否存在 token.json（儲存使用者的存取令牌）
    if os.path.exists(CALENDAR_TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(CALENDAR_TOKEN_FILE, combined_scopes)
        except Exception as e:
            print(f"⚠️ 讀取 token.json 時發生錯誤：{e}")
            creds = None
    
    # 如果沒有憑證或憑證過期，則進行登入
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # 嘗試刷新令牌
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"⚠️ 刷新令牌時發生錯誤：{e}")
                creds = None
        
        # 如果仍然沒有有效憑證，需要重新授權
        if not creds:
            if not os.path.exists(CALENDAR_CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"❌ 找不到憑證文件 {CALENDAR_CREDENTIALS_FILE}。\n"
                    "請從 Google Cloud Console 下載 OAuth2 憑證文件並命名為 credentials.json。"
                )
            
            # 使用合併的 scopes 進行授權，這樣 token.json 會包含兩個權限
            print(f"🔐 [Calendar] 正在請求授權，權限範圍：{combined_scopes}")
            flow = InstalledAppFlow.from_client_secrets_file(CALENDAR_CREDENTIALS_FILE, combined_scopes)
            creds = flow.run_local_server(port=0)
        
        # 儲存憑證以供下次使用
        try:
            with open(CALENDAR_TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
            print(f"✅ [Calendar] 憑證已保存，包含的權限：{creds.scopes if hasattr(creds, 'scopes') else 'N/A'}")
        except Exception as e:
            print(f"⚠️ 儲存 token.json 時發生錯誤：{e}")
    
    return build('calendar', 'v3', credentials=creds)


@tool
def create_calendar_event(
    summary: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
    location: str = "",
    attendees: str = "",
    timezone: str = "Asia/Taipei"
) -> str:
    """
    創建行事曆事件
    
    Args:
        summary: 事件標題
        start_datetime: 開始時間 (格式: YYYY-MM-DDTHH:MM:SS，例如: 2026-01-25T09:00:00)
        end_datetime: 結束時間 (格式: YYYY-MM-DDTHH:MM:SS，例如: 2026-01-25T10:00:00)
        description: 事件描述（可選）
        location: 事件地點（可選）
        attendees: 參與者郵箱，多個用逗號分隔（可選）
        timezone: 時區（預設: Asia/Taipei）
    
    Returns:
        創建結果消息，包含事件連結
    """
    try:
        service = get_calendar_service()
        
        # 構建事件對象
        event = {
            'summary': summary,
            'start': {
                'dateTime': start_datetime,
                'timeZone': timezone,
            },
            'end': {
                'dateTime': end_datetime,
                'timeZone': timezone,
            },
        }
        
        if description:
            event['description'] = description
        
        if location:
            event['location'] = location
        
        if attendees:
            # 驗證和清理參與者郵箱
            attendee_list = validate_and_clean_emails(attendees)
            if attendee_list:
                event['attendees'] = [{'email': email} for email in attendee_list]
            else:
                # 如果所有郵箱都無效，記錄警告但不添加 attendees
                print(f"⚠️ 警告：未找到有效的參與者郵箱，已跳過參與者設定。原始輸入：{attendees}")
        
        # 設置提醒
        event['reminders'] = {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},  # 24小時前郵件通知
                {'method': 'popup', 'minutes': 10},       # 10分鐘前視窗通知
            ],
        }
        
        # 創建事件
        event_result = service.events().insert(calendarId='primary', body=event, sendUpdates='all').execute()
        
        event_link = event_result.get('htmlLink', '')
        event_id = event_result.get('id', '')
        
        # 構建返回消息
        result_msg = (
            f"✅ 行事曆事件已成功創建！\n"
            f"標題：{summary}\n"
            f"開始時間：{start_datetime}\n"
            f"結束時間：{end_datetime}\n"
        )
        
        # 如果有參與者，顯示參與者資訊
        if event.get('attendees'):
            attendee_count = len(event['attendees'])
            result_msg += f"參與者：{attendee_count} 人\n"
        
        result_msg += f"事件連結：{event_link}\n"
        result_msg += f"事件 ID：{event_id}"
        
        return result_msg
        
    except FileNotFoundError as e:
        return str(e)
    except HttpError as error:
        error_msg = f"❌ 創建行事曆事件時發生錯誤：{error}"
        print(f"Calendar Tool 錯誤：{error}")
        return error_msg
    except Exception as e:
        error_msg = f"❌ 創建行事曆事件時發生錯誤：{str(e)}"
        print(f"Calendar Tool 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return error_msg


@tool
def update_calendar_event(
    event_id: str,
    summary: str = None,
    start_datetime: str = None,
    end_datetime: str = None,
    description: str = None,
    location: str = None,
    attendees: str = None,
    timezone: str = "Asia/Taipei"
) -> str:
    """
    更新現有行事曆事件
    
    Args:
        event_id: 要更新的事件 ID
        summary: 事件標題（可選，不提供則不更新）
        start_datetime: 開始時間（可選，格式: YYYY-MM-DDTHH:MM:SS）
        end_datetime: 結束時間（可選，格式: YYYY-MM-DDTHH:MM:SS）
        description: 事件描述（可選）
        location: 事件地點（可選）
        attendees: 參與者郵箱，多個用逗號分隔（可選）
        timezone: 時區（預設: Asia/Taipei）
    
    Returns:
        更新結果消息
    """
    try:
        service = get_calendar_service()
        
        # 獲取現有事件
        event = service.events().get(calendarId='primary', eventId=event_id).execute()
        
        # 更新提供的欄位
        if summary is not None:
            event['summary'] = summary
        if start_datetime is not None:
            event['start'] = {
                'dateTime': start_datetime,
                'timeZone': timezone,
            }
        if end_datetime is not None:
            event['end'] = {
                'dateTime': end_datetime,
                'timeZone': timezone,
            }
        if description is not None:
            event['description'] = description
        if location is not None:
            event['location'] = location
        if attendees is not None:
            if attendees:
                # 驗證和清理參與者郵箱
                attendee_list = validate_and_clean_emails(attendees)
                if attendee_list:
                    event['attendees'] = [{'email': email} for email in attendee_list]
                else:
                    # 如果所有郵箱都無效，記錄警告但不添加 attendees
                    print(f"⚠️ 警告：未找到有效的參與者郵箱，已跳過參與者設定。原始輸入：{attendees}")
                    event['attendees'] = []
            else:
                event['attendees'] = []
        
        # 更新事件
        updated_event = service.events().update(
            calendarId='primary',
            eventId=event_id,
            body=event,
            sendUpdates='all'
        ).execute()
        
        event_link = updated_event.get('htmlLink', '')
        
        return (
            f"✅ 行事曆事件已成功更新！\n"
            f"標題：{updated_event.get('summary', 'N/A')}\n"
            f"事件連結：{event_link}\n"
            f"事件 ID：{event_id}"
        )
        
    except HttpError as error:
        if error.resp.status == 404:
            return f"❌ 找不到事件 ID：{event_id}，請確認事件是否存在"
        error_msg = f"❌ 更新行事曆事件時發生錯誤：{error}"
        print(f"Calendar Tool 錯誤：{error}")
        return error_msg
    except Exception as e:
        error_msg = f"❌ 更新行事曆事件時發生錯誤：{str(e)}"
        print(f"Calendar Tool 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return error_msg


@tool
def delete_calendar_event(event_id: str) -> str:
    """
    刪除行事曆事件
    
    Args:
        event_id: 要刪除的事件 ID
    
    Returns:
        刪除結果消息
    """
    try:
        service = get_calendar_service()
        
        # 刪除事件
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        
        return f"✅ 行事曆事件已成功刪除！\n事件 ID：{event_id}"
        
    except HttpError as error:
        if error.resp.status == 404:
            return f"❌ 找不到事件 ID：{event_id}，請確認事件是否存在"
        error_msg = f"❌ 刪除行事曆事件時發生錯誤：{error}"
        print(f"Calendar Tool 錯誤：{error}")
        return error_msg
    except Exception as e:
        error_msg = f"❌ 刪除行事曆事件時發生錯誤：{str(e)}"
        print(f"Calendar Tool 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return error_msg


@tool
def list_calendar_events(
    max_results: int = 10,
    time_min: str = None,
    time_max: str = None
) -> str:
    """
    列出行事曆事件
    
    Args:
        max_results: 最大返回結果數（預設: 10）
        time_min: 開始時間過濾（可選，格式: YYYY-MM-DDTHH:MM:SS）
        time_max: 結束時間過濾（可選，格式: YYYY-MM-DDTHH:MM:SS）
    
    Returns:
        事件列表（包含事件 ID、標題、時間等）
    """
    try:
        service = get_calendar_service()
        
        # 構建查詢參數
        now = datetime.utcnow().isoformat() + 'Z'  # 'Z' 表示 UTC 時間
        time_min_param = time_min if time_min else now
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min_param,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        if not events:
            return "📅 目前沒有找到任何行事曆事件"
        
        result_lines = [f"📅 找到 {len(events)} 個事件：\n"]
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            summary = event.get('summary', '無標題')
            event_id = event.get('id', 'N/A')
            result_lines.append(
                f"- **{summary}**\n"
                f"  時間：{start}\n"
                f"  ID：{event_id}\n"
            )
        
        return "\n".join(result_lines)
        
    except Exception as e:
        error_msg = f"❌ 列出行事曆事件時發生錯誤：{str(e)}"
        print(f"Calendar Tool 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return error_msg

