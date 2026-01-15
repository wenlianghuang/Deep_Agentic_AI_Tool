"""
配置模組
包含系統配置、路徑設定和常量
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 設定 tokenizers 並行性環境變數（解決 fork 警告）
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 設定 HuggingFace 模型緩存目錄到外接 SSD
EXTERNAL_SSD_PATH = "/Volumes/T7_SSD"
HF_CACHE_DIR = os.path.join(EXTERNAL_SSD_PATH, "huggingface_cache")

# 檢查外接 SSD 是否存在
if os.path.exists(EXTERNAL_SSD_PATH):
    # 創建緩存目錄（如果不存在）
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    # 設置 HuggingFace 環境變數（必須在導入 HuggingFace 相關庫之前設置）
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
    os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
    print(f"💾 模型緩存目錄：{HF_CACHE_DIR}")
else:
    print(f"⚠️ 警告：找不到外接 SSD {EXTERNAL_SSD_PATH}，將使用預設緩存目錄")

# MLX 模型配置
MLX_MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
MLX_MAX_TOKENS = 2048
MLX_TEMPERATURE = 0.7

# RAG 配置
PDF_PATH = "./data/Tree_of_Thoughts.pdf"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 3

# Agent 配置
MAX_ITERATIONS = 5
MAX_RESEARCH_ITERATIONS = 20

# Reflection 配置（通用反思迭代次數）
MAX_REFLECTION_ITERATION = 0  # 反思的最大迭代次數

# Groq API 配置
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"  # 或其他 Groq 支持的模型
GROQ_MAX_TOKENS = 2048
GROQ_TEMPERATURE = 0.7
USE_GROQ_FIRST = True  # 是否优先使用 Groq API

# Ollama 配置
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # Llama 3.2 3B
OLLAMA_MAX_TOKENS = 2048
OLLAMA_TEMPERATURE = 0.7
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"  # 是否啟用 Ollama

# Email 配置 - 使用 Gmail API
EMAIL_SENDER = "matthuang46@gmail.com"  # 預設發件人（向後兼容）
# Gmail API 配置
GMAIL_CREDENTIALS_FILE = os.getenv("GMAIL_CREDENTIALS_FILE", "credentials_matthuang.json")  # OAuth2 憑證文件（預設）
GMAIL_TOKEN_FILE = os.getenv("GMAIL_TOKEN_FILE", "token.json")  # 儲存存取令牌的文件（預設）
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.send']  # Gmail API 權限範圍

# Calendar 配置 - 使用 Google Calendar API
CALENDAR_CREDENTIALS_FILE = os.getenv("CALENDAR_CREDENTIALS_FILE", "credentials_matthuang.json")  # OAuth2 憑證文件（可與 Gmail 共用）
CALENDAR_TOKEN_FILE = os.getenv("CALENDAR_TOKEN_FILE", "token.json")  # 儲存存取令牌的文件（可與 Gmail 共用）
CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar']  # Calendar API 權限範圍

# Google Maps API 配置
NORMAL_GOOGLE_MAPS_API_KEY = os.getenv("NORMAL_GOOGLE_MAPS_API_KEY", "")

# 多模態圖片分析 API 配置
# 優先順序：OpenAI GPT-4 Vision > Google Gemini > Anthropic Claude > Ollama LLaVA

# OpenAI GPT-4 Vision 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")  # gpt-4o, gpt-4o-mini, gpt-4-turbo
USE_OPENAI_VISION_FIRST = os.getenv("USE_OPENAI_VISION_FIRST", "false").lower() == "true"

# Google Gemini API 配置
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY", "")
GOOGLE_GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-1.5-flash")  # gemini-1.5-flash, gemini-1.5-pro
USE_GEMINI_FIRST = os.getenv("USE_GEMINI_FIRST", "true").lower() == "true"  # 默認優先使用 Gemini（免費額度較高）

# Anthropic Claude 配置（支持多模態）
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_VISION_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", "claude-3-5-sonnet-20241022")  # claude-3-5-sonnet, claude-3-opus
USE_ANTHROPIC_VISION = os.getenv("USE_ANTHROPIC_VISION", "false").lower() == "true"

# Ollama LLaVA 配置（本地多模態模型，完全免費）
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")  # llava, llava:13b, llava:34b
USE_OLLAMA_VISION = os.getenv("USE_OLLAMA_VISION", "true").lower() == "true"  # 默認啟用作為備援

# 用戶常用位置配置（用於計算交通時間）
# 設置您的家庭地址或辦公室地址，系統會自動計算從這些位置到事件地點的交通時間
USER_HOME_ADDRESS = os.getenv("USER_HOME_ADDRESS", "")  # 例如："台北市信義區信義路五段7號"
USER_OFFICE_ADDRESS = os.getenv("USER_OFFICE_ADDRESS", "")  # 例如："台北市大安區敦化南路二段216號"

