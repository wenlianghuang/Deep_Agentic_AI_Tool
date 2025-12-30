# Gmail API 設置指南

本系統使用 Gmail API 發送郵件，相比傳統 SMTP 方式，可以避免被歸類為垃圾郵件的風險。

## 📋 設置步驟

### 1. 創建 Google Cloud 專案

1. 前往 [Google Cloud Console](https://console.cloud.google.com/)
2. 點擊頂部的專案選擇器，然後點擊「新建專案」
3. 輸入專案名稱（例如：`Deep Agent Email`）
4. 點擊「創建」

### 2. 啟用 Gmail API

1. 在 Google Cloud Console 中，進入「API 和服務」>「資料庫」
2. 搜尋「Gmail API」
3. 點擊「Gmail API」並點擊「啟用」

### 3. 創建 OAuth2 憑證

1. 進入「API 和服務」>「憑證」
2. 點擊「建立憑證」>「OAuth 用戶端 ID」
3. 如果這是第一次，需要先設定 OAuth 同意畫面：
   - 選擇「外部」用戶類型
   - 填寫應用程式名稱（例如：`Deep Agent Email Tool`）
   - 填寫支援電子郵件
   - 點擊「儲存並繼續」
   - 在「範圍」頁面，點擊「儲存並繼續」
   - 在「測試使用者」頁面，添加您的 Gmail 地址
   - 點擊「儲存並繼續」完成設定
4. 回到「憑證」頁面，點擊「建立憑證」>「OAuth 用戶端 ID」
5. 選擇應用程式類型為「桌面應用程式」
6. 輸入名稱（例如：`Deep Agent Email Client`）
7. 點擊「建立」
8. 下載憑證 JSON 文件

### 4. 配置憑證文件

1. 將下載的 JSON 文件重新命名為 `credentials.json`
2. 將 `credentials.json` 放在專案根目錄（與 `Deep_Agent_Gradio_RAG_localLLM_main.py` 同一層級）

### 5. 首次授權

1. 運行應用程式：
   ```bash
   python Deep_Agent_Gradio_RAG_localLLM_main.py
   ```

2. 當首次使用 Email Tool 功能時，系統會自動：
   - 開啟瀏覽器進行 OAuth2 授權
   - 選擇您的 Gmail 帳號
   - 授予「發送郵件」權限
   - 自動生成 `token.json` 文件

3. 授權完成後，`token.json` 會自動儲存在專案根目錄，之後就不需要再次授權了。

## 🔧 配置選項

可以在 `.env` 文件中自定義以下配置（可選）：

```env
# Gmail API 配置（可選，有預設值）
GMAIL_CREDENTIALS_FILE=credentials.json
GMAIL_TOKEN_FILE=token.json
```

## ⚠️ 注意事項

1. **憑證文件安全**：
   - `credentials.json` 包含 OAuth2 客戶端資訊，請勿分享或上傳到公開儲存庫
   - 建議將 `credentials.json` 和 `token.json` 添加到 `.gitignore`

2. **令牌刷新**：
   - `token.json` 中的令牌會自動刷新，無需手動處理
   - 如果令牌過期，系統會自動重新授權

3. **權限範圍**：
   - 目前只請求 `gmail.send` 權限，僅用於發送郵件
   - 如果需要其他功能（如讀取郵件），需要修改 `GMAIL_SCOPES` 配置

4. **測試使用者限制**：
   - 在開發階段，OAuth 同意畫面設為「測試」模式時，只有添加到測試使用者的帳號才能使用
   - 如果要讓其他使用者使用，需要將應用程式發布或將他們添加為測試使用者

## 🔄 重新授權

如果需要重新授權（例如更換 Gmail 帳號）：

1. 刪除專案根目錄中的 `token.json` 文件
2. 重新運行應用程式
3. 使用 Email Tool 時會自動觸發新的授權流程

## ✅ 驗證設置

設置完成後，可以通過以下方式驗證：

1. 在 Gradio UI 中切換到「📧 Email Tool」標籤
2. 輸入郵件提示（例如：「寫一封測試郵件」）
3. 輸入收件人郵箱地址
4. 點擊「📧 生成並發送郵件」
5. 如果設置正確，郵件會成功發送並顯示 Message ID

## 🆚 Gmail API vs SMTP

使用 Gmail API 的優勢：

- ✅ **避免垃圾郵件**：通過官方 API 發送，信譽更高
- ✅ **更安全**：使用 OAuth2 認證，無需儲存密碼
- ✅ **更可靠**：API 有更好的錯誤處理和重試機制
- ✅ **更靈活**：可以訪問更多 Gmail 功能（未來擴展）

