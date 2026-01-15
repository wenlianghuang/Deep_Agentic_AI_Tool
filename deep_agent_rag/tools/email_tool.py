"""
Email 工具
提供郵件發送功能（使用 Gmail API）
"""
import os
import base64
from email.message import EmailMessage
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from langchain_core.tools import tool

from ..config import (
    EMAIL_SENDER,
    GMAIL_CREDENTIALS_FILE,
    GMAIL_TOKEN_FILE,
    GMAIL_SCOPES,
    CALENDAR_SCOPES,
)


def is_gmail_address(email: str) -> bool:
    """
    驗證郵箱是否為 Gmail 郵箱
    
    Args:
        email: 郵箱地址
    
    Returns:
        如果是 Gmail 郵箱則返回 True，否則返回 False
    """
    if not email or not email.strip():
        return False
    
    email = email.strip().lower()
    
    # 檢查是否為 Gmail 郵箱（@gmail.com 或 @googlemail.com）
    gmail_domains = ['@gmail.com', '@googlemail.com']
    return any(email.endswith(domain) for domain in gmail_domains)


def is_valid_email(email: str) -> bool:
    """
    驗證郵箱格式是否有效
    
    Args:
        email: 郵箱地址
    
    Returns:
        如果郵箱格式有效則返回 True，否則返回 False
    """
    if not email or not email.strip():
        return False
    
    email = email.strip()
    
    # 簡單的郵箱格式驗證：必須包含 @ 符號，且 @ 前後都有內容
    if '@' not in email:
        return False
    
    parts = email.split('@')
    if len(parts) != 2:
        return False
    
    local_part, domain = parts
    if not local_part or not domain:
        return False
    
    # 檢查域名部分是否包含點（基本驗證）
    if '.' not in domain:
        return False
    
    return True


def parse_recipients(recipient_string: str) -> list[str]:
    """
    解析收件人字符串，支援逗號分隔的多個郵箱
    
    Args:
        recipient_string: 收件人字符串，可以是單個郵箱或多個用逗號分隔的郵箱
    
    Returns:
        收件人郵箱列表
    """
    if not recipient_string or not recipient_string.strip():
        return []
    
    # 按逗號分割，並清理每個郵箱地址
    recipients = [email.strip() for email in recipient_string.split(',')]
    # 過濾掉空字符串
    recipients = [email for email in recipients if email]
    return recipients


def get_credentials_for_email(email: str) -> str:
    """
    根據郵箱地址獲取對應的憑證文件路徑
    
    Args:
        email: 郵箱地址
    
    Returns:
        憑證文件路徑
    """
    if not email:
        return GMAIL_CREDENTIALS_FILE
    
    # 從郵箱地址提取用戶名部分（例如：user@gmail.com -> user）
    email_username = email.split("@")[0].lower()
    
    # 構建憑證文件路徑：credentials_{username}.json
    credentials_file = f"credentials_{email_username}.json"
    
    # 如果對應的憑證文件存在，使用它；否則使用預設憑證文件
    if os.path.exists(credentials_file):
        return credentials_file
    else:
        # 如果找不到對應的憑證文件，返回預設憑證文件
        # 這允許使用者共用同一個 OAuth2 應用程式但使用不同的 token
        return GMAIL_CREDENTIALS_FILE


def get_token_for_email(email: str) -> str:
    """
    根據郵箱地址獲取對應的 token 文件路徑
    
    Args:
        email: 郵箱地址
    
    Returns:
        token 文件路徑
    """
    if not email:
        return GMAIL_TOKEN_FILE
    
    # 從郵箱地址提取用戶名部分
    email_username = email.split("@")[0].lower()
    
    # 構建 token 文件路徑：token_{username}.json
    token_file = f"token_{email_username}.json"
    
    return token_file


def validate_recipients(recipients: list[str]) -> tuple[bool, str]:
    """
    驗證多個收件人郵箱格式
    
    Args:
        recipients: 收件人郵箱列表
    
    Returns:
        (是否有效, 錯誤訊息)
    """
    if not recipients:
        return False, "❌ 請至少輸入一個收件人郵箱地址"
    
    invalid_emails = []
    for email in recipients:
        if not is_valid_email(email):
            invalid_emails.append(email)
    
    if invalid_emails:
        return False, (
            f"❌ 以下收件人郵箱格式無效：\n"
            f"{', '.join(invalid_emails)}\n"
            f"請輸入有效的郵箱地址（例如：user@example.com）"
        )
    
    return True, ""


def get_gmail_service(sender_email: str = None):
    """
    獲取 Gmail API 服務實例
    
    Args:
        sender_email: 發件人郵箱地址（可選），用於選擇對應的憑證和 token 文件
    
    Returns:
        Gmail API 服務對象
    """
    creds = None
    
    # 根據發件人郵箱選擇對應的憑證和 token 文件
    if sender_email:
        credentials_file = get_credentials_for_email(sender_email)
        token_file = get_token_for_email(sender_email)
        print(f"🔐 [Gmail] 使用發件人：{sender_email}")
        print(f"   📁 憑證文件：{credentials_file}")
        print(f"   📁 Token 文件：{token_file}")
    else:
        # 使用預設配置（向後兼容）
        credentials_file = GMAIL_CREDENTIALS_FILE
        token_file = GMAIL_TOKEN_FILE
    
    # 合併 Gmail 和 Calendar 的 scopes（因為共用同一個 token.json）
    # 使用 set 去重，確保 scopes 唯一
    combined_scopes = list(set(GMAIL_SCOPES + CALENDAR_SCOPES))
    
    # 檢查是否存在 token 文件（儲存使用者的存取令牌）
    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, combined_scopes)
        except Exception as e:
            print(f"⚠️ 讀取 {token_file} 時發生錯誤：{e}")
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
            if not os.path.exists(credentials_file):
                raise FileNotFoundError(
                    f"❌ 找不到憑證文件 {credentials_file}。\n"
                    f"請從 Google Cloud Console 下載 OAuth2 憑證文件。\n"
                    f"如果這是新使用者，請將憑證文件命名為 {credentials_file} 或使用預設的 {GMAIL_CREDENTIALS_FILE}。"
                )
            
            # 使用合併的 scopes 進行授權，這樣 token 文件會包含兩個權限
            print(f"🔐 [Gmail] 正在請求授權，權限範圍：{combined_scopes}")
            if sender_email:
                print(f"   👤 請選擇帳號：{sender_email}")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, combined_scopes)
            creds = flow.run_local_server(port=0)
        
        # 儲存憑證以供下次使用
        try:
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
            print(f"✅ [Gmail] 憑證已保存到 {token_file}，包含的權限：{creds.scopes if hasattr(creds, 'scopes') else 'N/A'}")
        except Exception as e:
            print(f"⚠️ 儲存 {token_file} 時發生錯誤：{e}")
    
    return build('gmail', 'v1', credentials=creds)


@tool
def send_email(recipient: str, subject: str, body: str, sender: str = None) -> str:
    """
    使用 Gmail API 發送郵件（支援多個收件人，用逗號分隔）
    
    Args:
        recipient: 收件人郵箱地址（可以是單個或多個用逗號分隔的郵箱，例如："user1@example.com, user2@example.com"）
        subject: 郵件主題
        body: 郵件正文內容
        sender: 發件人郵箱地址（可選，必須是 Gmail），如果不提供則使用預設發件人
    
    Returns:
        發送結果消息
    """
    try:
        # 解析收件人（支援多個，用逗號分隔）
        recipients = parse_recipients(recipient)
        
        # 驗證所有收件人
        is_valid, error_msg = validate_recipients(recipients)
        if not is_valid:
            return error_msg
        
        # 確定發件人地址
        actual_sender = sender.strip() if sender and sender.strip() else EMAIL_SENDER
        
        # 驗證發件人是否為 Gmail 郵箱（發件人必須是 Gmail，因為使用 Gmail API）
        if not is_gmail_address(actual_sender):
            return (
                f"❌ 錯誤：發件人必須是 Gmail 郵箱。\n"
                f"您輸入的發件人：{actual_sender}\n"
                f"請使用 @gmail.com 或 @googlemail.com 結尾的郵箱地址。\n"
                f"（收件人可以是任何郵箱地址）"
            )
        
        # 獲取 Gmail API 服務（使用對應的發件人憑證）
        service = get_gmail_service(actual_sender)
        
        # 創建郵件消息
        message = EmailMessage()
        message.set_content(body)
        # 使用逗號分隔的多個收件人
        message['To'] = ', '.join(recipients)
        message['From'] = actual_sender
        message['Subject'] = subject
        
        # 必須將郵件編碼為 base64url 格式
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        create_message = {
            'raw': encoded_message
        }
        
        # 執行發送
        try:
            send_result = service.users().messages().send(
                userId="me",
                body=create_message
            ).execute()
            
            message_id = send_result.get('id', '未知')
            recipient_count = len(recipients)
            if recipient_count == 1:
                return f"✅ 郵件已成功發送到 {recipients[0]}\n主題：{subject}\nMessage ID: {message_id}"
            else:
                return (
                    f"✅ 郵件已成功發送到 {recipient_count} 個收件人：\n"
                    f"{', '.join(recipients)}\n\n"
                    f"主題：{subject}\n"
                    f"Message ID: {message_id}"
                )
        except Exception as e:
            error_msg = str(e)
            if 'insufficient authentication scopes' in error_msg.lower():
                return (
                    "❌ 錯誤：認證權限不足。\n"
                    "請刪除 token.json 文件並重新授權，確保授予 Gmail 發送郵件的權限。"
                )
            elif 'invalid_grant' in error_msg.lower():
                return (
                    "❌ 錯誤：令牌已過期或無效。\n"
                    "請刪除 token.json 文件並重新授權。"
                )
            else:
                return f"❌ 發送郵件時發生錯誤：{error_msg}"
                
    except FileNotFoundError as e:
        return str(e)
    except Exception as e:
        error_msg = str(e)
        print(f"Email Tool 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return f"❌ 創建或發送郵件時發生錯誤：{error_msg}"

