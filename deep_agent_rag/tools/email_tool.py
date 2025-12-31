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
    GMAIL_SCOPES
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


def get_gmail_service():
    """
    獲取 Gmail API 服務實例
    
    Returns:
        Gmail API 服務對象
    """
    creds = None
    
    # 檢查是否存在 token.json（儲存使用者的存取令牌）
    if os.path.exists(GMAIL_TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, GMAIL_SCOPES)
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
            if not os.path.exists(GMAIL_CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"❌ 找不到憑證文件 {GMAIL_CREDENTIALS_FILE}。\n"
                    "請從 Google Cloud Console 下載 OAuth2 憑證文件並命名為 credentials.json。"
                )
            
            flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS_FILE, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        
        # 儲存憑證以供下次使用
        try:
            with open(GMAIL_TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        except Exception as e:
            print(f"⚠️ 儲存 token.json 時發生錯誤：{e}")
    
    return build('gmail', 'v1', credentials=creds)


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """
    使用 Gmail API 發送郵件（僅支援 Gmail 郵箱）
    
    Args:
        recipient: 收件人郵箱地址（必須是 Gmail 郵箱）
        subject: 郵件主題
        body: 郵件正文內容
    
    Returns:
        發送結果消息
    """
    try:
        # 驗證收件人是否為 Gmail 郵箱
        if not is_gmail_address(recipient):
            return (
                f"❌ 錯誤：此工具僅支援 Gmail 郵箱。\n"
                f"您輸入的郵箱：{recipient}\n"
                f"請使用 @gmail.com 或 @googlemail.com 結尾的郵箱地址。"
            )
        
        # 獲取 Gmail API 服務
        service = get_gmail_service()
        
        # 創建郵件消息
        message = EmailMessage()
        message.set_content(body)
        message['To'] = recipient
        message['From'] = EMAIL_SENDER
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
            return f"✅ 郵件已成功發送到 {recipient}\n主題：{subject}\nMessage ID: {message_id}"
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

