"""
RAG 系統初始化
使用 Private File RAG 系統，支持多文件、進階 RAG 方法
"""
import os
import glob
from typing import Optional

from .private_file_rag import PrivateFileRAG


def init_rag_system() -> Optional[PrivateFileRAG]:
    """
    初始化 RAG 系統（使用 Private File RAG）
    自動載入 data 目錄中的所有 PDF 文件
    
    Returns:
        PrivateFileRAG 實例，如果初始化失敗則返回 None
    """
    # 查找 data 目錄中的所有 PDF 文件
    data_dir = "./data"
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ 警告：在 {data_dir} 目錄中找不到 PDF 文件，RAG 功能將無法使用。")
        return None
    
    print(f"🚀 [RAG] 正在使用 Private File RAG 初始化系統...")
    print(f"   找到 {len(pdf_files)} 個 PDF 文件：")
    for pdf_file in pdf_files:
        print(f"      - {os.path.basename(pdf_file)}")
    
    try:
        # 創建 Private File RAG 實例
        # 啟用自適應選擇和進階 RAG 方法
        # 優化參數以提高檢索準確性
        private_rag = PrivateFileRAG(
            use_semantic_chunking=False,  # 可以根據需要改為 True
            chunk_size=800,  # 增加 chunk 大小以保留更多上下文
            chunk_overlap=150,  # 增加重疊以保持連貫性
            enable_adaptive_selection=True,  # 啟用自適應選擇最佳 RAG 方法
            selected_rag_method=None  # None 表示自動選擇
        )
        
        # 處理所有 PDF 文件
        documents, status_msg = private_rag.process_files(pdf_files)
        
        if not documents:
            print(f"   ❌ 處理文件失敗：{status_msg}")
            return None
        
        print(f"   ✅ {status_msg}")
        return private_rag
        
    except Exception as e:
        print(f"   ❌ Private File RAG 初始化失敗：{e}")
        import traceback
        traceback.print_exc()
        return None
