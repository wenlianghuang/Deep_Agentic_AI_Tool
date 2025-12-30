"""
RAG 系統初始化
處理 PDF 載入、向量化和檢索
"""
import os
import shutil
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import (
    EXTERNAL_SSD_PATH, HF_CACHE_DIR, PDF_PATH,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K
)


def get_device():
    """自動檢測可用的設備（優先使用 Apple Silicon GPU）"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def init_rag_system():
    """初始化 RAG 系統（PDF 向量資料庫）"""
    retriever = None
    
    if not os.path.exists(PDF_PATH):
        print(f"⚠️ 警告：找不到 {PDF_PATH}，RAG 功能將無法使用。")
        return retriever
    
    print("🚀 [RAG] 正在初始化 PDF 向量資料庫（使用 Jina Embeddings v3）...")
    
    try:
        # 載入 PDF
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(docs)
        print(f"   ✓ PDF 載入完成，共 {len(splits)} 個文字塊")
        
        # 初始化 Jina Embeddings
        device = get_device()
        device_name = "Apple Silicon GPU (MPS)" if device == "mps" else ("NVIDIA GPU (CUDA)" if device == "cuda" else "CPU")
        print(f"   📦 正在載入 Jina Embeddings 模型（使用設備：{device_name}）...")
        
        # 設定緩存目錄
        cache_folder = None
        if os.path.exists(EXTERNAL_SSD_PATH):
            cache_folder = os.path.join(HF_CACHE_DIR, "transformers")
            os.makedirs(cache_folder, exist_ok=True)
        
        # 準備 model_kwargs
        model_kwargs = {
            "device": device,
            "trust_remote_code": True
        }
        
        # 建立 embeddings
        embeddings_kwargs = {
            "model_name": EMBEDDING_MODEL,
            "model_kwargs": model_kwargs,
            "encode_kwargs": {
                "normalize_embeddings": True,
                "batch_size": 4,
            },
            "show_progress": True
        }
        
        if cache_folder:
            embeddings_kwargs["cache_folder"] = cache_folder
        
        # 嘗試載入模型
        try:
            embeddings = HuggingFaceEmbeddings(**embeddings_kwargs)
            print("   ✅ Jina Embeddings 載入完成")
        except (FileNotFoundError, OSError, Exception) as e:
            error_msg = str(e)
            if "No such file or directory" in error_msg or "cache" in error_msg.lower() or "transformers_modules" in error_msg:
                print("   ⚠️ 檢測到模型緩存不完整，正在清理並重新下載...")
                cache_paths_to_clean = [
                    os.path.join(HF_CACHE_DIR, "modules", "transformers_modules", "jinaai"),
                    os.path.join(HF_CACHE_DIR, "modules", "transformers_modules", "jinaai", "jina_hyphen_embeddings_hyphen_v3"),
                ]
                
                for cache_path in cache_paths_to_clean:
                    if os.path.exists(cache_path):
                        try:
                            shutil.rmtree(cache_path)
                        except Exception:
                            pass
                
                print("   正在重新下載模型（這可能需要幾分鐘）...")
                embeddings = HuggingFaceEmbeddings(**embeddings_kwargs)
                print("   ✅ Jina Embeddings 載入完成（已重新下載）")
            else:
                print(f"   ❌ 載入模型時發生錯誤：{error_msg}")
                return None
        
        # 建立向量資料庫
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
        print("   ✅ RAG 系統初始化完成")
        
    except Exception as e:
        print(f"   ❌ RAG 系統初始化失敗：{e}")
        return None
    
    return retriever

