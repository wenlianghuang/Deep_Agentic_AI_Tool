"""
配置模組
包含系統配置、路徑設定和常量
"""
import os
from dotenv import load_dotenv

load_dotenv()

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

# Groq API 配置
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"  # 或其他 Groq 支持的模型
GROQ_MAX_TOKENS = 2048
GROQ_TEMPERATURE = 0.7
USE_GROQ_FIRST = True  # 是否优先使用 Groq API

