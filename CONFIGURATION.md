# 配置指南

本文檔詳細說明 Deep Agentic AI Tool 的所有配置選項。

## 📋 配置方式

系統配置有兩種方式：

1. **環境變數**：在 `.env` 文件中設置（推薦）
2. **代碼配置**：在 `deep_agent_rag/config.py` 中直接修改

## 🔧 環境變數配置

在專案根目錄創建 `.env` 文件：

```env
# ============================================
# LLM 配置
# ============================================

# Groq API（可選，用於快速推理）
GROQ_API_KEY=your_groq_api_key_here

# ============================================
# 搜尋 API 配置
# ============================================

# Tavily API（可選，用於網路搜尋）
TAVILY_API_KEY=your_tavily_api_key_here

# ============================================
# Google API 配置
# ============================================

# Gmail API 憑證文件（可選，預設：credentials.json）
GMAIL_CREDENTIALS_FILE=credentials.json
GMAIL_TOKEN_FILE=token.json

# Calendar API 憑證文件（可選，可與 Gmail 共用）
CALENDAR_CREDENTIALS_FILE=credentials.json
CALENDAR_TOKEN_FILE=token.json

# Google Maps API（可選，用於行事曆地點驗證）
NORMAL_GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# 用戶常用位置（可選，用於計算交通時間）
USER_HOME_ADDRESS=台北市信義區信義路五段7號
USER_OFFICE_ADDRESS=台北市大安區敦化南路二段216號
```

## ⚙️ 代碼配置

所有配置選項在 `deep_agent_rag/config.py` 中定義。

### MLX 模型配置

```python
# MLX 模型 ID
MLX_MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

# 最大生成 tokens
MLX_MAX_TOKENS = 2048

# 溫度參數（控制隨機性，0.0-1.0）
MLX_TEMPERATURE = 0.7
```

**說明：**
- `MLX_MODEL_ID`：HuggingFace 模型 ID，必須是 MLX 兼容格式
- `MLX_MAX_TOKENS`：生成的最大 token 數，較大值會使用更多記憶體
- `MLX_TEMPERATURE`：較低值（0.1-0.3）更確定性，較高值（0.7-1.0）更創造性

### Groq API 配置

```python
# Groq API 金鑰（從環境變數讀取）
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Groq 模型名稱
GROQ_MODEL = "llama-3.3-70b-versatile"

# 最大生成 tokens
GROQ_MAX_TOKENS = 2048

# 溫度參數
GROQ_TEMPERATURE = 0.7

# 是否優先使用 Groq API
USE_GROQ_FIRST = True
```

**說明：**
- `GROQ_API_KEY`：從環境變數讀取，如果未設置則為空字串
- `GROQ_MODEL`：Groq 支援的模型，可選其他模型
- `USE_GROQ_FIRST`：設為 `True` 時優先使用 Groq，否則優先使用本地模型

### RAG 配置

```python
# PDF 文件路徑
PDF_PATH = "./data/Tree_of_Thoughts.pdf"

# 嵌入模型
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"

# 文本分塊大小
CHUNK_SIZE = 1000

# 分塊重疊大小
CHUNK_OVERLAP = 200

# 檢索結果數量
RETRIEVER_K = 3
```

**說明：**
- `PDF_PATH`：PDF 文件的相對或絕對路徑
- `EMBEDDING_MODEL`：用於向量化的嵌入模型
- `CHUNK_SIZE`：每個文本塊的大小（字符數）
- `CHUNK_OVERLAP`：相鄰塊之間的重疊大小，有助於保持上下文
- `RETRIEVER_K`：每次檢索返回的結果數量

**調整建議：**
- 較大的 `CHUNK_SIZE`：適合長文檔，但可能包含無關資訊
- 較小的 `CHUNK_SIZE`：更精確，但可能丟失上下文
- 較大的 `CHUNK_OVERLAP`：更好的上下文連續性，但增加存儲
- 較大的 `RETRIEVER_K`：更多相關結果，但可能包含無關內容

### Agent 配置

```python
# 最大迭代次數（整體）
MAX_ITERATIONS = 5

# 最大研究迭代次數
MAX_RESEARCH_ITERATIONS = 20

# 反思的最大迭代次數（通用）
MAX_REFLECTION_ITERATION = 0
```

**說明：**
- `MAX_ITERATIONS`：整個代理流程的最大迭代次數
- `MAX_RESEARCH_ITERATIONS`：研究代理節點的最大迭代次數
- `MAX_REFLECTION_ITERATION`：反思代理的最大迭代次數（0 表示不進行反思）

**調整建議：**
- 較大的值：允許更深入的研究，但可能耗時更長
- 較小的值：更快完成，但可能不夠深入

### Email 配置

```python
# 發件人郵箱地址
EMAIL_SENDER = "matthuang46@gmail.com"

# Gmail API 憑證文件
GMAIL_CREDENTIALS_FILE = os.getenv("GMAIL_CREDENTIALS_FILE", "credentials_matthuang.json")

# Gmail API Token 文件
GMAIL_TOKEN_FILE = os.getenv("GMAIL_TOKEN_FILE", "token.json")

# Gmail API 權限範圍
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.send']
```

**說明：**
- `EMAIL_SENDER`：發送郵件的 Gmail 地址
- `GMAIL_CREDENTIALS_FILE`：OAuth2 憑證文件路徑
- `GMAIL_TOKEN_FILE`：Token 存儲文件路徑
- `GMAIL_SCOPES`：API 權限範圍，目前只請求發送郵件權限

**擴展權限：**
如果需要讀取郵件等功能，可以添加更多 scope：
```python
GMAIL_SCOPES = [
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.readonly',
]
```

### Calendar 配置

```python
# Calendar API 憑證文件（可與 Gmail 共用）
CALENDAR_CREDENTIALS_FILE = os.getenv("CALENDAR_CREDENTIALS_FILE", "credentials_matthuang.json")

# Calendar API Token 文件（可與 Gmail 共用）
CALENDAR_TOKEN_FILE = os.getenv("CALENDAR_TOKEN_FILE", "token.json")

# Calendar API 權限範圍
CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar']
```

**說明：**
- Calendar API 可以與 Gmail API 共用相同的 OAuth2 憑證
- `CALENDAR_SCOPES`：目前請求完整的 Calendar 權限

### Google Maps 配置

```python
# Google Maps API 金鑰
NORMAL_GOOGLE_MAPS_API_KEY = os.getenv("NORMAL_GOOGLE_MAPS_API_KEY", "")

# 用戶常用位置（用於計算交通時間）
USER_HOME_ADDRESS = os.getenv("USER_HOME_ADDRESS", "")
USER_OFFICE_ADDRESS = os.getenv("USER_OFFICE_ADDRESS", "")
```

**說明：**
- `NORMAL_GOOGLE_MAPS_API_KEY`：Google Maps API 金鑰（必需，用於地點驗證）
- `USER_HOME_ADDRESS`：家庭地址（可選，用於計算交通時間）
- `USER_OFFICE_ADDRESS`：辦公室地址（可選，優先使用）

**地址格式：**
- 支援多種格式：完整地址、地標名稱、座標等
- 例如：`"台北市信義區信義路五段7號"`、`"台北101"`、`"25.0330,121.5654"`

### 外部 SSD 配置

```python
# 外接 SSD 路徑
EXTERNAL_SSD_PATH = "/Volumes/T7_SSD"

# HuggingFace 緩存目錄
HF_CACHE_DIR = os.path.join(EXTERNAL_SSD_PATH, "huggingface_cache")
```

**說明：**
- 如果配置了外部 SSD，模型會緩存到外部 SSD
- 節省本地磁碟空間
- 如果找不到外部 SSD，會使用預設緩存目錄

**修改路徑：**
根據您的實際 SSD 掛載路徑修改 `EXTERNAL_SSD_PATH`。

## 🔍 配置驗證

### 檢查配置是否正確

1. **檢查環境變數**

```bash
# 在專案根目錄
cat .env
```

2. **檢查 Python 配置**

```python
# 在 Python 中
from deep_agent_rag.config import *
print(f"Groq API Key: {'已設置' if GROQ_API_KEY else '未設置'}")
print(f"MLX Model: {MLX_MODEL_ID}")
print(f"PDF Path: {PDF_PATH}")
```

3. **測試 API 連接**

- **Groq API**：運行研究代理，查看是否使用 Groq
- **Gmail API**：嘗試發送測試郵件
- **Calendar API**：嘗試創建測試事件
- **Google Maps API**：創建包含地點的事件，查看是否驗證成功

## 🎯 配置最佳實踐

### 開發環境

```env
# 使用本地 MLX 模型（無需 API 金鑰）
# 不設置 GROQ_API_KEY

# 可選：設置 Tavily API 用於網路搜尋
TAVILY_API_KEY=your_key_here
```

### 生產環境

```env
# 使用 Groq API（更快）
GROQ_API_KEY=your_key_here

# 設置所有必要的 API 金鑰
TAVILY_API_KEY=your_key_here
NORMAL_GOOGLE_MAPS_API_KEY=your_key_here

# 設置常用位置
USER_OFFICE_ADDRESS=your_office_address
```

### 隱私敏感環境

```env
# 不使用任何外部 API
# 只使用本地 MLX 模型

# 不設置任何 API 金鑰
```

## ⚠️ 常見配置問題

### 1. API 金鑰未設置

**問題**：功能無法使用（例如：網路搜尋、地點驗證）

**解決**：
- 檢查 `.env` 文件是否存在
- 檢查環境變數名稱是否正確
- 確認 API 金鑰是否有效

### 2. 憑證文件找不到

**問題**：Gmail/Calendar API 無法使用

**解決**：
- 確認 `credentials.json` 在專案根目錄
- 檢查 `GMAIL_CREDENTIALS_FILE` 路徑是否正確
- 參考 [Gmail API 設置指南](GMAIL_API_SETUP.md)

### 3. 模型緩存問題

**問題**：模型載入失敗或緩存損壞

**解決**：
- 刪除緩存目錄重新下載
- 檢查磁碟空間是否足夠
- 如果使用外部 SSD，確認 SSD 已正確掛載

### 4. PDF 文件找不到

**問題**：RAG 系統無法載入 PDF

**解決**：
- 確認 PDF 文件存在於指定路徑
- 檢查 `PDF_PATH` 是否為相對路徑（相對於專案根目錄）
- 或使用絕對路徑

## 📝 配置備份

建議將配置備份：

1. **環境變數備份**：保存 `.env` 文件（但不要提交敏感資訊）
2. **代碼配置備份**：`config.py` 可以提交到版本控制
3. **憑證備份**：安全存儲 `credentials.json`（不要提交到版本控制）

## 🔐 安全建議

1. **不要提交敏感資訊**
   - 將 `.env` 添加到 `.gitignore`
   - 將 `credentials.json` 和 `token.json` 添加到 `.gitignore`

2. **使用環境變數**
   - 優先使用環境變數而非硬編碼
   - 在生產環境使用密鑰管理服務

3. **限制 API 權限**
   - 在 Google Cloud Console 中限制 API Key 使用
   - 只請求必要的 OAuth scope

4. **定期輪換金鑰**
   - 定期更新 API 金鑰
   - 監控 API 使用情況

