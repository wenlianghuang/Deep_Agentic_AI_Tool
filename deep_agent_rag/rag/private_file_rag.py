"""
私有文件 RAG 系統
集成 Learn_RAG 的功能，支持上傳私有文件（PDF、DOCX、TXT）並使用 RAG 回答問題

LLM 使用策略：
- 優先使用 Groq API（如果配置了 API 金鑰）
- 其次使用 Ollama（如果服務正在運行）
- 最後使用 MLX 本地模型（作為備選方案）
"""
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import tempfile
import shutil

# 導入 Deep_Agentic_AI_Tool 的 LLM 工具
# 這樣可以使用統一的 LLM 優先順序策略（Groq -> Ollama -> MLX）
from ..utils.llm_utils import get_llm
from langchain_core.messages import HumanMessage

# 導入 LLM 適配器和智能選擇器
from .llm_adapter import LangChainLLMAdapter
from .adaptive_rag_selector import AdaptiveRAGSelector, RAGMethod

# 添加 Learn_RAG 到 Python 路徑
# 計算 Learn_RAG 的路徑（與 Deep_Agentic_AI_Tool 在同一目錄下）
current_file = Path(__file__).resolve()
# 從 deep_agent_rag/rag/private_file_rag.py 向上找到 Deep_Agentic_AI_Tool 根目錄
deep_agent_root = current_file.parent.parent.parent.parent
learn_rag_path = deep_agent_root.parent / "Learn_RAG"

# 如果 Learn_RAG 不在預期位置，嘗試其他可能的位置
if not learn_rag_path.exists():
    # 嘗試當前工作目錄的父目錄
    cwd = Path.cwd()
    learn_rag_path = cwd.parent / "Learn_RAG"
    
    if not learn_rag_path.exists():
        # 嘗試直接使用絕對路徑
        learn_rag_path = Path("/Users/matthuang/Desktop/Learn_RAG")

# 將 Learn_RAG 目錄添加到 Python 路徑（這樣可以導入 src 模組）
# 注意：需要將 Learn_RAG 目錄本身添加到路徑，因為 src 模組在 Learn_RAG/src/ 下
if learn_rag_path.exists() and learn_rag_path.is_dir():
    if str(learn_rag_path) not in sys.path:
        sys.path.insert(0, str(learn_rag_path))
    print(f"✓ 找到 Learn_RAG 項目: {learn_rag_path}")
    print(f"  Python 路徑已添加: {learn_rag_path}")
else:
    print(f"⚠️ 無法找到 Learn_RAG 項目")
    print(f"   嘗試的路徑: {learn_rag_path}")
    print(f"   請確保 Learn_RAG 項目在: {deep_agent_root.parent / 'Learn_RAG'}")

# 嘗試導入 Learn_RAG 模組
# 注意：document_processor.py 在頂層導入了 arxiv，所以需要先安裝依賴
try:
    # 先檢查必要的依賴是否已安裝
    import importlib
    
    required_deps = {
        "arxiv": "arxiv",
        "langchain_community": "langchain-community",
        "langchain_text_splitters": "langchain-text-splitters",
        "chromadb": "chromadb",
        "sentence_transformers": "sentence-transformers",
        "rank_bm25": "rank-bm25",
        "pypdf": "pypdf",
    }
    
    missing_deps = []
    for module_name, package_name in required_deps.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_deps.append(package_name)
    
    if missing_deps:
        print(f"⚠️ 缺少以下依賴包: {', '.join(missing_deps)}")
        print(f"\n💡 請安裝 Learn_RAG 項目的依賴:")
        print(f"   方法 1: 使用 pip")
        print(f"   pip install {' '.join(missing_deps)}")
        print(f"\n   方法 2: 使用 uv (推薦，如果 Learn_RAG 使用 uv)")
        print(f"   cd {learn_rag_path}")
        print(f"   uv sync")
        print(f"\n   方法 3: 安裝所有 Learn_RAG 依賴")
        print(f"   pip install arxiv langchain-community langchain-text-splitters chromadb sentence-transformers rank-bm25 pypdf docx2txt langchain-experimental")
        LEARN_RAG_AVAILABLE = False
    else:
        # 所有依賴都已安裝，嘗試導入模組
        from src.document_processor import DocumentProcessor
        from src.retrievers.bm25_retriever import BM25Retriever
        from src.retrievers.vector_retriever import VectorRetriever
        from src.retrievers.hybrid_search import HybridSearch
        from src.retrievers.reranker import Reranker, RAGPipeline
        from src.prompt_formatter import PromptFormatter
        # 導入進階 RAG 方法
        from src.subquery_rag import SubQueryDecompositionRAG
        from src.hyde_rag import HyDERAG
        from src.step_back_rag import StepBackRAG
        from src.hybrid_subquery_hyde_rag import HybridSubqueryHyDERAG
        from src.triple_hybrid_rag import TripleHybridRAG
        # 不再需要導入 OllamaLLM，因為我們使用 Deep_Agentic_AI_Tool 的統一 LLM 系統（get_llm()）
        # from src.llm_integration import OllamaLLM
        LEARN_RAG_AVAILABLE = True
        print("✓ 成功導入 Learn_RAG 模組（包含進階 RAG 方法）")
        
except ImportError as e:
    error_msg = str(e)
    print(f"⚠️ 無法導入 Learn_RAG 模組: {error_msg}")
    print(f"\n💡 請安裝 Learn_RAG 項目的依賴:")
    print(f"   pip install arxiv langchain-community langchain-text-splitters chromadb sentence-transformers rank-bm25 pypdf docx2txt langchain-experimental")
    print(f"\n   或者:")
    print(f"   cd {learn_rag_path}")
    print(f"   uv sync")
    LEARN_RAG_AVAILABLE = False
except Exception as e:
    error_msg = str(e)
    print(f"⚠️ 導入 Learn_RAG 模組時發生錯誤: {error_msg}")
    print(f"   當前 Python 路徑: {sys.path[:3]}")
    print(f"   Learn_RAG 路徑: {learn_rag_path}")
    LEARN_RAG_AVAILABLE = False


class PrivateFileRAG:
    """
    私有文件 RAG 系統管理器
    
    這個類負責管理私有文件的 RAG（檢索增強生成）系統，包括：
    - 文件處理和分塊（支持字符分塊和語義分塊）
    - 檢索系統初始化（BM25 + 向量檢索 + 混合搜尋）
    - RAG 查詢和回答生成
    
    LLM 使用策略：
    - 使用 Deep_Agentic_AI_Tool 的統一 LLM 系統（get_llm()）
    - 自動遵循優先順序：Groq API > Ollama > MLX 本地模型
    - 無需手動指定 LLM 類型，系統會根據配置和可用性自動選擇最合適的 LLM
    - 如果 Groq API 額度用完或服務不可用，會自動切換到備選 LLM
    """
    
    def __init__(
        self,
        use_semantic_chunking: bool = False,
        chunk_size: int = 500,  # 預設改為 500，提供更細的粒度
        chunk_overlap: int = 100,  # 預設改為 100，保持 20% 的重疊比例
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db_private",
        # 語義分塊參數
        semantic_threshold: float = 1.0,  # 預設改為 1.0，提供更細的粒度（原為 1.5）
        semantic_min_chunk_size: int = 100,  # 語義分塊的最小 chunk 大小（字符數）
        # 進階 RAG 方法參數
        enable_adaptive_selection: bool = True,  # 是否啟用自動選擇最佳 RAG 方法
        selected_rag_method: Optional[str] = None  # 手動指定方法（可選，如果設置則覆蓋自動選擇）
    ):
        """
        初始化私有文件 RAG 系統
        
        Args:
            use_semantic_chunking: 是否使用語義分塊
                                  True: 使用語義分塊（保持語義完整性，不會在句子中間切斷，但處理時間較長）
                                  False: 使用字符分塊（快速，但可能在句子中間切斷，預設）
            chunk_size: 字符分塊大小（僅用於字符分塊模式）
                       每個 chunk 的字符數，預設 500（較細的粒度）
                       建議值：
                       - 300-500：細粒度，適合精確檢索，但可能遺漏上下文
                       - 500-800：中等粒度，平衡精確度和上下文（推薦）
                       - 800-1200：粗粒度，包含更多上下文，但可能包含不相關內容
                       較大的值可以包含更多上下文，但可能包含不相關內容
                       較小的值更精確，但可能遺漏重要信息
            chunk_overlap: 字符分塊重疊大小（僅用於字符分塊模式）
                          相鄰 chunks 之間的重疊字符數，預設 100（約為 chunk_size 的 20%）
                          建議值：chunk_size 的 15-25%
                          重疊可以幫助保持上下文連貫性，避免在重要信息邊界處切斷
                          較大的重疊可以更好地保持上下文，但會增加 chunks 數量
            embedding_model: Embedding 模型名稱
                            用於向量檢索和語義分塊的 embedding 模型
                            預設使用 "sentence-transformers/all-MiniLM-L6-v2"（輕量級、快速、效果好）
                            如果需要更好的效果，可以使用更大的模型，但會增加計算時間和內存使用
            persist_directory: 向量資料庫持久化目錄
                               ChromaDB 會將向量資料庫保存在此目錄，下次使用時可以直接載入
                               預設為 "./chroma_db_private"
                               如果目錄已存在，會自動載入已有的向量資料庫
            semantic_threshold: 語義分塊的敏感度閾值（標準差倍數）
                               數值越小，分塊越多、越細（chunks 越小）
                               數值越大，分塊越少、越粗（chunks 越大）
                               建議範圍：
                               - 0.8-1.2：細粒度，適合需要精確檢索的場景（推薦）
                               - 1.2-1.8：中等粒度，平衡精確度和上下文
                               - 1.8-2.5：粗粒度，包含更多上下文，但可能包含不相關內容
                               預設值：1.0（已優化為更細的粒度，原為 1.5）
            semantic_min_chunk_size: 語義分塊的最小 chunk 大小（字符數）
                                    小於此大小的 chunks 會被合併到相鄰的 chunks
                                    預設值：100 字符
                                    建議值：50-200，根據文檔類型調整
                                    較小的值可以保留更多細節，但可能產生過多的小 chunks
        """
        if not LEARN_RAG_AVAILABLE:
            raise ImportError("Learn_RAG 模組不可用，請檢查安裝")
        
        self.use_semantic_chunking = use_semantic_chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        # 語義分塊參數
        self.semantic_threshold = semantic_threshold
        self.semantic_min_chunk_size = semantic_min_chunk_size
        # 進階 RAG 方法參數
        self.enable_adaptive_selection = enable_adaptive_selection
        self.selected_rag_method = selected_rag_method
        
        # 組件
        self.processor = None
        self.bm25_retriever = None
        self.vector_retriever = None
        self.hybrid_search = None
        self.reranker = None
        self.rag_pipeline = None
        self.formatter = None
        self.shared_embeddings = None
        
        # 進階 RAG 方法組件
        self.llm_adapter = None  # LLM 適配器
        self.rag_selector = AdaptiveRAGSelector()  # 智能選擇器
        self.subquery_rag = None
        self.hyde_rag = None
        self.step_back_rag = None
        self.hybrid_subquery_hyde_rag = None
        self.triple_hybrid_rag = None
        
        # 當前載入的文件
        self.current_files = []
        self.is_initialized = False
    
    def _init_embeddings(self):
        """
        初始化共用的 Embedding 模型（用於語義分塊）
        
        這個方法會創建一個 HuggingFace Embeddings 模型，用於：
        - 語義分塊：計算文本的語義相似度，在語義邊界處切分
        - 向量檢索：將文檔轉換為向量，用於語義搜尋
        
        如果初始化失敗，會自動回退到字符分塊模式。
        
        Returns:
            HuggingFaceEmbeddings 實例，如果失敗則返回 None
        """
        # 如果不需要語義分塊，直接返回 None
        if not self.use_semantic_chunking:
            return None
        
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from src.retrievers.vector_retriever import get_device
            
            # 獲取 Hugging Face 模型緩存目錄（如果設置了環境變數）
            # 這對於使用外接硬碟存儲模型很有用
            hf_cache_dir = os.getenv("HF_CACHE_DIR", None)
            
            # 自動檢測可用的設備（MPS/CUDA/CPU）
            # MPS: macOS GPU, CUDA: NVIDIA GPU, CPU: 備選
            device = get_device()
            
            # 構建模型參數
            model_kwargs = {'device': device}
            if hf_cache_dir:
                model_kwargs['cache_dir'] = hf_cache_dir
            
            # 創建 HuggingFace Embeddings 模型
            # normalize_embeddings=True 會將向量正規化，有助於提升檢索效果
            self.shared_embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': True}
            )
            return self.shared_embeddings
        except Exception as e:
            # 如果初始化失敗，記錄錯誤並回退到字符分塊模式
            print(f"⚠️ 初始化 Embedding 模型失敗: {e}")
            print("   將回退到字符分塊模式")
            self.use_semantic_chunking = False
            return None
    
    def process_files(self, file_paths: List[str]) -> Tuple[List[Dict], str]:
        """
        處理上傳的文件
        
        Args:
            file_paths: 文件路徑列表（可以是字符串路徑或 Gradio 文件對象）
            
        Returns:
            (documents, status_message) 元組
        """
        if not file_paths:
            return [], "❌ 未提供文件路徑"
        
        try:
            # 處理文件路徑（可能是 Gradio 文件對象）
            actual_paths = []
            for file_path in file_paths:
                if hasattr(file_path, 'name'):
                    # Gradio 文件對象
                    actual_path = file_path.name
                else:
                    # 字符串路徑
                    actual_path = file_path
                
                if os.path.exists(actual_path):
                    actual_paths.append(actual_path)
                else:
                    print(f"⚠️ 文件不存在: {actual_path}")
            
            if not actual_paths:
                return [], "❌ 沒有有效的文件路徑"
            
            # 初始化 Embedding（如果需要語義分塊）
            if self.use_semantic_chunking:
                self._init_embeddings()
            
            # 初始化文檔處理器
            if self.use_semantic_chunking and self.shared_embeddings:
                # 使用可調整的語義分塊參數
                self.processor = DocumentProcessor(
                    embeddings=self.shared_embeddings,
                    use_semantic_chunking=True,
                    breakpoint_threshold_amount=self.semantic_threshold,  # 使用可調整的閾值
                    min_chunk_size=self.semantic_min_chunk_size  # 使用可調整的最小 chunk 大小
                )
                print(f"📏 使用語義分塊：threshold={self.semantic_threshold}, min_chunk_size={self.semantic_min_chunk_size}")
            else:
                self.processor = DocumentProcessor(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            
            # 處理所有文件
            all_documents = []
            for file_path in actual_paths:
                print(f"處理文件: {file_path}")
                documents = self.processor.process_file(file_path)
                all_documents.extend(documents)
                print(f"  ✓ 創建了 {len(documents)} 個 chunks")
            
            if not all_documents:
                return [], "❌ 處理後沒有文檔內容"
            
            self.current_files = actual_paths
            
            # 初始化檢索系統
            status_msg = self._init_retrievers(all_documents)
            
            return all_documents, status_msg
            
        except Exception as e:
            error_msg = f"❌ 處理文件失敗: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return [], error_msg
    
    def _init_retrievers(self, documents: List[Dict]) -> str:
        """
        初始化檢索器
        
        Args:
            documents: 文檔列表
            
        Returns:
            狀態訊息
        """
        try:
            # 初始化 BM25 檢索器
            print("  - 初始化 BM25 檢索器...")
            self.bm25_retriever = BM25Retriever(documents)
            
            # 初始化向量檢索器
            print("  - 初始化向量檢索器...")
            self.vector_retriever = VectorRetriever(
                documents,
                embedding_model=self.embedding_model,
                persist_directory=self.persist_directory,
                embeddings=self.shared_embeddings
            )
            
            # 初始化混合搜尋
            print("  - 初始化混合搜尋...")
            self.hybrid_search = HybridSearch(
                sparse_retriever=self.bm25_retriever,
                dense_retriever=self.vector_retriever,
                fusion_method="rrf",
                rrf_k=60
            )
            
            # 嘗試初始化重排序器（可選）
            try:
                print("  - 初始化重排序器...")
                self.reranker = Reranker(
                    model_name="BAAI/bge-reranker-base",
                    batch_size=16
                )
                
                # 初始化 RAG 管線
                print("  - 初始化 RAG 管線...")
                self.rag_pipeline = RAGPipeline(
                    hybrid_search=self.hybrid_search,
                    reranker=self.reranker,
                    recall_k=20,
                    adaptive_recall=True
                )
                
                # 初始化 Prompt 格式化器
                self.formatter = PromptFormatter(format_style="detailed")
                
                # 初始化進階 RAG 方法（延遲初始化，只在需要時創建）
                self._init_advanced_rag_methods()
                
                self.is_initialized = True
                return f"✅ 成功處理 {len(self.current_files)} 個文件，創建了 {len(documents)} 個 chunks，RAG 系統已初始化（包含重排序）"
                
            except Exception as e:
                print(f"  ⚠️ 重排序器初始化失敗: {e}")
                print("   將使用混合搜尋（不進行重排序）")
                self.formatter = PromptFormatter(format_style="detailed")
                
                # 即使重排序失敗，也初始化進階 RAG 方法
                self._init_advanced_rag_methods()
                
                self.is_initialized = True
                return f"✅ 成功處理 {len(self.current_files)} 個文件，創建了 {len(documents)} 個 chunks，RAG 系統已初始化（無重排序）"
                
        except Exception as e:
            error_msg = f"❌ 檢索系統初始化失敗: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
    
    def _init_advanced_rag_methods(self):
        """
        初始化所有進階 RAG 方法
        
        這個方法會創建 LLM 適配器並初始化所有 5 種進階 RAG 方法實例
        使用延遲初始化策略，只在需要時創建
        """
        try:
            # 創建 LLM 適配器（將 LangChain ChatModel 包裝成 OllamaLLM 接口）
            if self.llm_adapter is None:
                print("  - 創建 LLM 適配器...")
                langchain_llm = get_llm()
                self.llm_adapter = LangChainLLMAdapter(langchain_llm)
                print("    ✓ LLM 適配器創建完成")
            
            # 確保有必要的組件
            if not self.rag_pipeline:
                print("  ⚠️ RAG Pipeline 未初始化，無法創建進階 RAG 方法")
                return
            
            if not self.vector_retriever:
                print("  ⚠️ Vector Retriever 未初始化，無法創建進階 RAG 方法")
                return
            
            # 初始化 SubQuery RAG
            if self.subquery_rag is None:
                try:
                    print("  - 初始化 SubQuery RAG...")
                    self.subquery_rag = SubQueryDecompositionRAG(
                        rag_pipeline=self.rag_pipeline,
                        llm=self.llm_adapter,
                        max_sub_queries=3,
                        top_k_per_subquery=5,
                        enable_parallel=True
                    )
                    print("    ✓ SubQuery RAG 初始化完成")
                except Exception as e:
                    print(f"    ⚠️ SubQuery RAG 初始化失敗: {e}")
            
            # 初始化 HyDE RAG
            if self.hyde_rag is None:
                try:
                    print("  - 初始化 HyDE RAG...")
                    self.hyde_rag = HyDERAG(
                        rag_pipeline=self.rag_pipeline,
                        vector_retriever=self.vector_retriever,
                        llm=self.llm_adapter,
                        hypothetical_length=200,
                        temperature=0.7
                    )
                    print("    ✓ HyDE RAG 初始化完成")
                except Exception as e:
                    print(f"    ⚠️ HyDE RAG 初始化失敗: {e}")
            
            # 初始化 Step-back RAG
            if self.step_back_rag is None:
                try:
                    print("  - 初始化 Step-back RAG...")
                    self.step_back_rag = StepBackRAG(
                        rag_pipeline=self.rag_pipeline,
                        vector_retriever=self.vector_retriever,
                        llm=self.llm_adapter,
                        step_back_temperature=0.3,
                        answer_temperature=0.7,
                        enable_parallel=True
                    )
                    print("    ✓ Step-back RAG 初始化完成")
                except Exception as e:
                    print(f"    ⚠️ Step-back RAG 初始化失敗: {e}")
            
            # 初始化 Hybrid Subquery+HyDE RAG
            if self.hybrid_subquery_hyde_rag is None:
                try:
                    print("  - 初始化 Hybrid Subquery+HyDE RAG...")
                    self.hybrid_subquery_hyde_rag = HybridSubqueryHyDERAG(
                        rag_pipeline=self.rag_pipeline,
                        vector_retriever=self.vector_retriever,
                        llm=self.llm_adapter,
                        max_sub_queries=3,
                        top_k_per_subquery=5,
                        hypothetical_length=200,
                        temperature_subquery=0.3,
                        temperature_hyde=0.7,
                        enable_parallel=True
                    )
                    print("    ✓ Hybrid Subquery+HyDE RAG 初始化完成")
                except Exception as e:
                    print(f"    ⚠️ Hybrid Subquery+HyDE RAG 初始化失敗: {e}")
            
            # 初始化 Triple Hybrid RAG
            if self.triple_hybrid_rag is None:
                try:
                    print("  - 初始化 Triple Hybrid RAG...")
                    self.triple_hybrid_rag = TripleHybridRAG(
                        rag_pipeline=self.rag_pipeline,
                        vector_retriever=self.vector_retriever,
                        llm=self.llm_adapter,
                        max_sub_queries=3,
                        top_k_per_subquery=5,
                        hypothetical_length=200,
                        temperature_subquery=0.3,
                        temperature_hyde=0.7,
                        temperature_stepback=0.3,
                        answer_temperature=0.7,
                        enable_parallel=True
                    )
                    print("    ✓ Triple Hybrid RAG 初始化完成")
                except Exception as e:
                    print(f"    ⚠️ Triple Hybrid RAG 初始化失敗: {e}")
            
            print("  ✅ 所有進階 RAG 方法初始化完成")
            
        except Exception as e:
            print(f"  ⚠️ 初始化進階 RAG 方法時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    def query(
        self,
        query: str,
        top_k: int = 3,
        use_llm: bool = True,
        llm_model: Optional[str] = None,
        conversation_history: Optional[List[Tuple[str, str]]] = None
    ) -> Dict:
        """
        查詢 RAG 系統並生成回答
        
        這個方法會執行完整的 RAG 流程：
        1. 使用混合搜尋（BM25 + 向量檢索）檢索相關文檔片段
        2. 可選：使用重排序器進一步優化結果（如果已初始化）
        3. 格式化檢索結果為 LLM 可讀的上下文
        4. 使用 LLM 生成回答（如果啟用）
        
        LLM 選擇策略：
        - 自動使用 get_llm() 獲取 LLM 實例
        - 優先順序：Groq API > Ollama > MLX 本地模型
        - 無需手動指定 LLM 類型，系統會根據配置和可用性自動選擇最合適的
        - 如果當前 LLM 失敗（如 API 額度用完），會自動切換到備選 LLM
        
        Args:
            query: 查詢問題（用戶想要問的問題）
                  例如："這份文檔的主要內容是什麼？"
            top_k: 返回的結果數量（檢索到的文檔片段數量）
                  建議值：3-5（太少可能遺漏重要信息，太多可能包含不相關內容）
                  預設值：3
            use_llm: 是否使用 LLM 生成回答
                    True: 使用 LLM 基於檢索結果生成完整、連貫的回答（推薦）
                    False: 只返回檢索到的文檔片段，不生成回答（適合快速查看相關內容）
            llm_model: LLM 模型名稱（已廢棄，不再使用）
                      現在統一使用 get_llm() 自動選擇 LLM，遵循 Groq -> Ollama -> MLX 的優先順序
                      保留此參數僅為向後兼容，實際不會被使用
            
        Returns:
            包含以下內容的字典：
            - success: 是否成功（bool）
            - query: 原始查詢問題（str）
            - answer: LLM 生成的回答（str，如果 use_llm=True 且成功）
            - results: 檢索到的文檔片段列表（List[Dict]），每個片段包含：
              * content: 文檔內容（str）
              * metadata: 元數據（Dict），包含標題、文件路徑等
              * score: 相關性分數（float）
            - formatted_context: 格式化後的上下文（str），用於 LLM 生成回答
            - stats: 檢索統計信息（Dict），包含：
              * total_time: 總耗時（float，秒）
              * recall_time: 召回階段耗時（float，秒）
              * rerank_time: 重排序階段耗時（float，秒，如果有重排序）
            - document_type: 檢測到的文檔類型（str）
                            "paper": 學術論文
                            "cv": 簡歷/履歷
                            "general": 通用文檔（預設）
            - error: 錯誤訊息（str，如果失敗）
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "RAG 系統尚未初始化，請先上傳文件"
            }
        
        try:
            # 選擇 RAG 方法
            selected_method = None
            method_reason = ""
            
            if self.enable_adaptive_selection and self.selected_rag_method is None:
                # 自動選擇最佳方法
                query_features = self.rag_selector.analyze_query(query)
                file_features = self.rag_selector.analyze_files(self.current_files, None)
                selected_method = self.rag_selector.select_best_method(
                    query_features, 
                    file_features,
                    enable_advanced=True
                )
                method_reason = self.rag_selector.get_method_reason(
                    selected_method, 
                    query_features, 
                    file_features
                )
                print(f"🔍 自動選擇 RAG 方法: {selected_method.value}")
                print(f"   理由: {method_reason}")
            elif self.selected_rag_method:
                # 手動指定方法
                try:
                    selected_method = RAGMethod(self.selected_rag_method)
                    method_reason = f"手動選擇: {selected_method.value}"
                    print(f"🔍 使用手動指定的 RAG 方法: {selected_method.value}")
                except ValueError:
                    print(f"⚠️ 無效的 RAG 方法: {self.selected_rag_method}，使用基礎方法")
                    selected_method = RAGMethod.BASIC
                    method_reason = "無效方法，回退到基礎方法"
            else:
                # 使用基礎方法
                selected_method = RAGMethod.BASIC
                method_reason = "使用基礎 RAG 方法"
            
            # 根據選擇的方法執行查詢
            if selected_method == RAGMethod.BASIC:
                # 使用基礎 RAG 方法（原有邏輯）
                return self._query_basic(query, top_k, use_llm, conversation_history)
            else:
                # 使用進階 RAG 方法
                return self._query_advanced(query, top_k, use_llm, selected_method, method_reason, conversation_history)
                
        except Exception as e:
            error_msg = f"❌ 查詢時發生錯誤: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_msg,
                "query": query
            }
    
    def query_stream(
        self,
        query: str,
        top_k: int = 3,
        conversation_history: Optional[List[Tuple[str, str]]] = None
    ):
        """
        流式查詢 RAG 系統並逐步生成回答（逐字輸出）
        
        這個方法會執行完整的 RAG 流程，但使用流式 LLM 輸出：
        1. 使用混合搜尋（BM25 + 向量檢索）檢索相關文檔片段
        2. 可選：使用重排序器進一步優化結果（如果已初始化）
        3. 格式化檢索結果為 LLM 可讀的上下文
        4. 使用 LLM 流式生成回答（逐字輸出）
        
        Args:
            query: 查詢問題（用戶想要問的問題）
            top_k: 返回的結果數量（檢索到的文檔片段數量）
            conversation_history: 可選的對話歷史，格式為 List[Tuple[str, str]]
        
        Yields:
            包含以下內容的字典：
            - success: 是否成功（bool）
            - answer: 當前累積的回答（str，逐步更新）
            - query: 原始查詢問題（str）
            - results: 檢索到的文檔片段列表（List[Dict]）
            - formatted_context: 格式化後的上下文（str）
            - stats: 檢索統計信息（Dict）
            - document_type: 檢測到的文檔類型（str）
            - rag_method: 使用的 RAG 方法（str）
            - method_reason: 方法選擇理由（str）
            - error: 錯誤訊息（str，如果失敗）
        """
        if not self.is_initialized:
            yield {
                "success": False,
                "error": "RAG 系統尚未初始化，請先上傳文件",
                "answer": ""
            }
            return
        
        try:
            # 選擇 RAG 方法（與 query 方法相同的邏輯）
            selected_method = None
            method_reason = ""
            
            if self.enable_adaptive_selection and self.selected_rag_method is None:
                # 自動選擇最佳方法
                query_features = self.rag_selector.analyze_query(query)
                file_features = self.rag_selector.analyze_files(self.current_files, None)
                selected_method = self.rag_selector.select_best_method(
                    query_features, 
                    file_features,
                    enable_advanced=True
                )
                method_reason = self.rag_selector.get_method_reason(
                    selected_method, 
                    query_features, 
                    file_features
                )
                print(f"🔍 自動選擇 RAG 方法: {selected_method.value}")
                print(f"   理由: {method_reason}")
            elif self.selected_rag_method:
                # 手動指定方法
                try:
                    selected_method = RAGMethod(self.selected_rag_method)
                    method_reason = f"手動選擇: {selected_method.value}"
                    print(f"🔍 使用手動指定的 RAG 方法: {selected_method.value}")
                except ValueError:
                    print(f"⚠️ 無效的 RAG 方法: {self.selected_rag_method}，使用基礎方法")
                    selected_method = RAGMethod.BASIC
                    method_reason = "無效方法，回退到基礎方法"
            else:
                # 使用基礎方法
                selected_method = RAGMethod.BASIC
                method_reason = "使用基礎 RAG 方法"
            
            # 目前只支持基礎方法的流式輸出
            if selected_method != RAGMethod.BASIC:
                # 對於進階方法，回退到非流式查詢
                result = self._query_advanced(query, top_k, True, selected_method, method_reason, conversation_history)
                if result.get("success"):
                    answer = result.get("answer", "")
                    # 逐字輸出
                    accumulated = ""
                    for char in answer:
                        accumulated += char
                        yield {
                            "success": True,
                            "answer": accumulated,
                            "query": query,
                            "results": result.get("results", []),
                            "formatted_context": result.get("formatted_context", ""),
                            "stats": result.get("stats", {}),
                            "document_type": result.get("document_type", "general"),
                            "rag_method": result.get("rag_method", "basic"),
                            "method_reason": method_reason
                        }
                        time.sleep(0.01)  # 每字符延遲 10 毫秒
                else:
                    yield result
                return
            
            # 使用基礎 RAG 方法的流式輸出
            # 🔧 在檢索之前擴展查詢，從對話歷史中提取關鍵信息
            expanded_query = self._expand_query_with_history(query, conversation_history)
            
            # 檢索相關文檔（使用擴展後的查詢）
            if self.rag_pipeline:
                # 使用完整的 RAG 管線（包含重排序）
                results, stats = self.rag_pipeline.query(
                    text=expanded_query,  # 使用擴展後的查詢
                    top_k=top_k,
                    enable_rerank=True,
                    return_stats=True
                )
            else:
                # 僅使用混合搜尋
                results = self.hybrid_search.retrieve(expanded_query, top_k=top_k)  # 使用擴展後的查詢
                stats = {"total_time": 0, "recall_time": 0, "rerank_time": 0}
            
            if not results:
                yield {
                    "success": False,
                    "error": "未找到相關文檔片段",
                    "answer": "",
                    "results": [],
                    "rag_method": "basic",
                    "method_reason": "基礎 RAG 方法"
                }
                return
            
            # 格式化上下文
            formatted_context = self.formatter.format_context(
                results,
                format_style="detailed"
            )
            
            # 檢測文檔類型
            document_type = self._detect_document_type(results)
            
            # 使用流式 LLM 生成回答
            try:
                # 使用 Deep_Agentic_AI_Tool 的統一 LLM 系統
                llm = get_llm()
                
                # 構建包含對話歷史的 prompt
                prompt = self._build_prompt_with_history(
                    query,
                    formatted_context,
                    document_type,
                    conversation_history
                )
                
                messages = [HumanMessage(content=prompt)]
                
                # 嘗試使用流式輸出
                accumulated_answer = ""
                try:
                    # 檢查 LLM 是否支持 stream 方法
                    if hasattr(llm, 'stream'):
                        # 使用流式輸出
                        for chunk in llm.stream(messages):
                            if hasattr(chunk, 'content'):
                                chunk_text = chunk.content
                            elif isinstance(chunk, str):
                                chunk_text = chunk
                            else:
                                chunk_text = str(chunk)
                            
                            if chunk_text:
                                accumulated_answer += chunk_text
                                yield {
                                    "success": True,
                                    "answer": accumulated_answer,
                                    "query": query,
                                    "results": results,
                                    "formatted_context": formatted_context,
                                    "stats": stats,
                                    "document_type": document_type,
                                    "rag_method": "basic",
                                    "method_reason": "基礎 RAG 方法"
                                }
                    else:
                        # 如果不支持流式輸出，使用 invoke 然後逐字輸出
                        response = llm.invoke(messages)
                        answer = response.content if hasattr(response, 'content') else str(response)
                        
                        # 逐字輸出
                        for char in answer:
                            accumulated_answer += char
                            yield {
                                "success": True,
                                "answer": accumulated_answer,
                                "query": query,
                                "results": results,
                                "formatted_context": formatted_context,
                                "stats": stats,
                                "document_type": document_type,
                                "rag_method": "basic",
                                "method_reason": "基礎 RAG 方法"
                            }
                            time.sleep(0.01)  # 每字符延遲 10 毫秒
                except Exception as stream_error:
                    # 如果流式輸出失敗，回退到非流式
                    print(f"⚠️ 流式輸出失敗，使用非流式: {stream_error}")
                    response = llm.invoke(messages)
                    answer = response.content if hasattr(response, 'content') else str(response)
                    
                    # 逐字輸出
                    for char in answer:
                        accumulated_answer += char
                        yield {
                            "success": True,
                            "answer": accumulated_answer,
                            "query": query,
                            "results": results,
                            "formatted_context": formatted_context,
                            "stats": stats,
                            "document_type": document_type,
                            "rag_method": "basic",
                            "method_reason": "基礎 RAG 方法"
                        }
                        time.sleep(0.01)  # 每字符延遲 10 毫秒
                
            except Exception as e:
                print(f"⚠️ LLM 生成回答失敗: {e}")
                import traceback
                traceback.print_exc()
                yield {
                    "success": False,
                    "error": f"LLM 生成回答失敗: {str(e)}",
                    "answer": "",
                    "query": query,
                    "rag_method": "basic"
                }
                
        except Exception as e:
            error_msg = f"❌ 查詢時發生錯誤: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield {
                "success": False,
                "error": error_msg,
                "answer": "",
                "query": query
            }
    
    def _query_basic(self, query: str, top_k: int, use_llm: bool, conversation_history: Optional[List[Tuple[str, str]]] = None) -> Dict:
        """
        使用基礎 RAG 方法查詢（原有邏輯）
        
        Args:
            query: 查詢問題
            top_k: 返回結果數量
            use_llm: 是否使用 LLM 生成回答
            conversation_history: 可選的對話歷史，格式為 List[Tuple[str, str]]，每個元組為 (用戶問題, AI回答)
        """
        try:
            # 🔧 在檢索之前擴展查詢，從對話歷史中提取關鍵信息
            expanded_query = self._expand_query_with_history(query, conversation_history)
            
            # 檢索相關文檔（使用擴展後的查詢）
            if self.rag_pipeline:
                # 使用完整的 RAG 管線（包含重排序）
                results, stats = self.rag_pipeline.query(
                    text=expanded_query,  # 使用擴展後的查詢
                    top_k=top_k,
                    enable_rerank=True,
                    return_stats=True
                )
            else:
                # 僅使用混合搜尋
                results = self.hybrid_search.retrieve(expanded_query, top_k=top_k)  # 使用擴展後的查詢
                stats = {"total_time": 0, "recall_time": 0, "rerank_time": 0}
            
            if not results:
                return {
                    "success": False,
                    "error": "未找到相關文檔片段",
                    "results": [],
                    "rag_method": "basic",
                    "method_reason": "基礎 RAG 方法"
                }
            
            # 格式化上下文
            formatted_context = self.formatter.format_context(
                results,
                format_style="detailed"
            )
            
            # 檢測文檔類型
            document_type = self._detect_document_type(results)
            
            # 生成回答（如果啟用）
            answer = None
            if use_llm:
                try:
                    # 使用 Deep_Agentic_AI_Tool 的統一 LLM 系統
                    llm = get_llm()
                    
                    # 構建包含對話歷史的 prompt
                    prompt = self._build_prompt_with_history(
                        query,
                        formatted_context,
                        document_type,
                        conversation_history
                    )
                    
                    messages = [HumanMessage(content=prompt)]
                    response = llm.invoke(messages)
                    answer = response.content
                    
                except Exception as e:
                    print(f"⚠️ LLM 生成回答失敗: {e}")
                    import traceback
                    traceback.print_exc()
                    answer = None
            
            return {
                "success": True,
                "query": query,
                "answer": answer,
                "results": results,
                "formatted_context": formatted_context,
                "stats": stats,
                "document_type": document_type,
                "rag_method": "basic",
                "method_reason": "基礎 RAG 方法"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"基礎 RAG 查詢失敗: {str(e)}",
                "query": query,
                "rag_method": "basic"
            }
    
    def _query_advanced(
        self, 
        query: str, 
        top_k: int, 
        use_llm: bool, 
        method: RAGMethod,
        method_reason: str,
        conversation_history: Optional[List[Tuple[str, str]]] = None
    ) -> Dict:
        """
        使用進階 RAG 方法查詢
        
        Args:
            query: 查詢問題
            top_k: 返回結果數量
            use_llm: 是否使用 LLM 生成回答
            method: RAG 方法
            method_reason: 方法選擇理由
            conversation_history: 可選的對話歷史
        """
        try:
            # 🔧 在檢索之前擴展查詢，從對話歷史中提取關鍵信息
            expanded_query = self._expand_query_with_history(query, conversation_history)
            
            # 確保進階方法已初始化
            if not self.llm_adapter:
                print("⚠️ LLM 適配器未初始化，回退到基礎方法")
                return self._query_basic(query, top_k, use_llm, conversation_history)
            
            # 根據方法選擇對應的 RAG 實例
            rag_instance = None
            method_name = method.value
            
            if method == RAGMethod.SUBQUERY:
                rag_instance = self.subquery_rag
            elif method == RAGMethod.HYDE:
                rag_instance = self.hyde_rag
            elif method == RAGMethod.STEP_BACK:
                rag_instance = self.step_back_rag
            elif method == RAGMethod.HYBRID_SUBQUERY_HYDE:
                rag_instance = self.hybrid_subquery_hyde_rag
            elif method == RAGMethod.TRIPLE_HYBRID:
                rag_instance = self.triple_hybrid_rag
            
            # 如果方法未初始化，回退到基礎方法
            if rag_instance is None:
                print(f"⚠️ {method_name} 方法未初始化，回退到基礎方法")
                return self._query_basic(query, top_k, use_llm, conversation_history)
            
            # 使用進階方法生成回答（使用擴展後的查詢）
            if use_llm:
                try:
                    result = rag_instance.generate_answer(
                        question=expanded_query,  # 使用擴展後的查詢
                        formatter=self.formatter,
                        top_k=top_k,
                        document_type=self._detect_document_type([])  # 暫時使用空列表，實際會在方法內部檢索
                    )
                    
                    # 統一返回格式
                    return {
                        "success": True,
                        "query": query,
                        "answer": result.get("answer", ""),
                        "results": result.get("results", []),
                        "formatted_context": result.get("formatted_context", ""),
                        "stats": {
                            "total_time": result.get("elapsed_time", 0),
                            "recall_time": 0,
                            "rerank_time": 0
                        },
                        "document_type": result.get("document_type", "general"),
                        "rag_method": method_name,
                        "method_reason": method_reason,
                        "advanced_details": result  # 保留進階方法的額外信息
                    }
                except Exception as e:
                    print(f"⚠️ 進階 RAG 方法執行失敗: {e}")
                    import traceback
                    traceback.print_exc()
                    # 回退到基礎方法
                    print("   回退到基礎 RAG 方法...")
                    return self._query_basic(query, top_k, use_llm, conversation_history)
            else:
                # 不使用 LLM，只檢索（使用擴展後的查詢）
                # 不同方法有不同的 query 接口，這裡統一處理
                if hasattr(rag_instance, 'query'):
                    result = rag_instance.query(expanded_query, top_k=top_k)  # 使用擴展後的查詢
                    return {
                        "success": True,
                        "query": query,
                        "answer": None,
                        "results": result.get("results", []),
                        "formatted_context": "",
                        "stats": result.get("stats", {}),
                        "document_type": "general",
                        "rag_method": method_name,
                        "method_reason": method_reason
                    }
                else:
                    # 如果方法不支持 query，回退到基礎方法
                    return self._query_basic(query, top_k, use_llm, conversation_history)
                    
        except Exception as e:
            print(f"⚠️ 進階 RAG 查詢失敗: {e}")
            import traceback
            traceback.print_exc()
            # 回退到基礎方法
            return self._query_basic(query, top_k, use_llm, conversation_history)
    
    def _expand_query_with_history(
        self,
        query: str,
        conversation_history: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """
        使用對話歷史擴展查詢，提取關鍵信息（如產品名稱）
        
        這個方法會從對話歷史中提取關鍵信息（如產品名稱、技術術語等），
        並將其添加到當前查詢中，以改進檢索效果。
        
        Args:
            query: 當前查詢問題
            conversation_history: 可選的對話歷史，格式為 List[Tuple[str, str]]，每個元組為 (用戶問題, AI回答)
            
        Returns:
            擴展後的查詢字符串
        """
        if not conversation_history or len(conversation_history) == 0:
            return query
        
        # 已知的產品名稱列表（可以從文檔中動態提取，這裡先硬編碼）
        product_names = [
            "Lumina-Grid", "Gaia-7", "Nebula-X", "Deep-Void", "Synapse-Link",
            "Lumina Grid", "Gaia 7", "Nebula X", "Deep Void", "Synapse Link"
        ]
        
        # 檢查當前查詢中是否已經包含產品名稱
        query_lower = query.lower()
        has_product_in_query = any(
            name.lower() in query_lower for name in product_names
        )
        
        # 如果當前查詢已經包含產品名稱，不需要擴展
        if has_product_in_query:
            return query
        
        # 從對話歷史中提取產品名稱
        # 檢查最近幾輪對話（最多檢查最近 5 輪）
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        
        # 收集所有歷史文本（包括用戶問題和 AI 回答）
        history_texts = []
        for user_q, ai_a in recent_history:
            if user_q:
                history_texts.append(user_q)
            if ai_a:
                history_texts.append(ai_a)
        
        # 合併所有歷史文本
        all_history_text = " ".join(history_texts)
        
        # 查找歷史中提到的產品名稱
        found_products = []
        for name in product_names:
            if name in all_history_text:
                found_products.append(name)
        
        # 如果找到產品名稱，添加到查詢中
        if found_products:
            # 使用最近提到的產品（最後一個）
            most_recent_product = found_products[-1]
            expanded_query = f"{most_recent_product} {query}"
            print(f"🔍 [查詢擴展] 從對話歷史中提取產品名稱 '{most_recent_product}'，擴展查詢為：{expanded_query}")
            return expanded_query
        
        # 如果沒有找到產品名稱，嘗試使用 LLM 來提取關鍵信息
        # 這是一個更智能的方法，但需要額外的 LLM 調用
        try:
            from ..utils.llm_utils import get_llm
            llm = get_llm()
            
            # 構建提取關鍵信息的 prompt
            history_summary = "\n".join([
                f"用戶: {user_q}\nAI: {ai_a[:200]}..." if ai_a else f"用戶: {user_q}"
                for user_q, ai_a in recent_history[-3:]  # 只使用最近 3 輪
            ])
            
            extract_prompt = f"""根據以下對話歷史，提取與當前查詢相關的關鍵信息（如產品名稱、技術術語等）。

對話歷史：
{history_summary}

當前查詢：{query}

請提取對話歷史中與當前查詢相關的關鍵信息（特別是產品名稱），如果有的話，請以簡潔的方式返回（只返回關鍵詞，不要解釋）。
如果沒有相關信息，請返回 "無"。
"""
            
            messages = [HumanMessage(content=extract_prompt)]
            response = llm.invoke(messages)
            extracted_info = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # 如果提取到有效信息且不是"無"，則添加到查詢中
            if extracted_info and extracted_info.lower() not in ["無", "无", "none", "no", ""]:
                # 檢查提取的信息是否包含產品名稱
                has_product = any(name.lower() in extracted_info.lower() for name in product_names)
                if has_product:
                    expanded_query = f"{extracted_info} {query}"
                    print(f"🔍 [查詢擴展] 使用 LLM 提取關鍵信息 '{extracted_info}'，擴展查詢為：{expanded_query}")
                    return expanded_query
        except Exception as e:
            # 如果 LLM 提取失敗，忽略錯誤，返回原始查詢
            print(f"⚠️ [查詢擴展] LLM 提取關鍵信息失敗: {e}，使用原始查詢")
        
        # 如果沒有找到任何關鍵信息，返回原始查詢
        return query
    
    def _build_prompt_with_history(
        self,
        query: str,
        formatted_context: str,
        document_type: str,
        conversation_history: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """
        構建包含對話歷史的 prompt
        
        Args:
            query: 當前查詢問題
            formatted_context: 格式化後的上下文
            document_type: 文檔類型
            conversation_history: 可選的對話歷史，格式為 List[Tuple[str, str]]，每個元組為 (用戶問題, AI回答)
            
        Returns:
            完整的 prompt 字符串
        """
        # 獲取基礎 prompt（不包含歷史）
        base_prompt = self.formatter.create_prompt(
            query,
            formatted_context,
            document_type=document_type
        )
        
        # 如果沒有對話歷史，直接返回基礎 prompt
        if not conversation_history or len(conversation_history) == 0:
            return base_prompt
        
        # 限制歷史長度，只保留最近 10 輪對話（避免上下文過長）
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        
        # 構建歷史對話文本
        history_text = ""
        for i, (user_q, ai_a) in enumerate(recent_history, 1):
            if ai_a:  # 如果有 AI 回答
                history_text += f"**對話 {i}:**\n"
                history_text += f"用戶: {user_q}\n"
                history_text += f"AI: {ai_a}\n\n"
            else:  # 如果只有用戶問題（不完整對話）
                history_text += f"**對話 {i}:**\n"
                history_text += f"用戶: {user_q}\n\n"
        
        # 檢測語言
        detected_language = self.formatter.detect_language(query) if self.formatter.auto_detect_language else "zh"
        
        # 根據語言構建包含歷史的 prompt
        if detected_language == "zh":
            history_section = f"""## 之前的對話歷史：

{history_text}---

"""
        else:
            history_section = f"""## Previous Conversation History:

{history_text}---

"""
        
        # 將歷史插入到系統提示詞和文檔片段之間
        # 找到 "## 相關文檔片段：" 或 "## Relevant Document Excerpts:" 的位置
        if detected_language == "zh":
            marker = "## 相關文檔片段："
        else:
            marker = "## Relevant Document Excerpts:"
        
        # 在 marker 之前插入歷史
        if marker in base_prompt:
            parts = base_prompt.split(marker, 1)
            prompt_with_history = parts[0] + history_section + marker + parts[1]
        else:
            # 如果找不到 marker，在開頭添加歷史
            prompt_with_history = history_section + base_prompt
        
        return prompt_with_history
    
    def _detect_document_type(self, results: List[Dict]) -> str:
        """檢測文檔類型"""
        if not results:
            return "general"
        
        # 檢查 metadata
        for result in results:
            metadata = result.get("metadata", {})
            file_path = str(metadata.get("file_path", "")).lower()
            
            if any(keyword in file_path for keyword in ["cv", "resume", "履歷", "簡歷"]):
                return "cv"
            elif any(keyword in file_path for keyword in ["arxiv", "paper", "論文"]):
                return "paper"
        
        return "general"
    
    def clear(self):
        """清除當前載入的文件和 RAG 系統"""
        self.current_files = []
        self.is_initialized = False
        self.processor = None
        self.bm25_retriever = None
        self.vector_retriever = None
        self.hybrid_search = None
        self.reranker = None
        self.rag_pipeline = None
        self.formatter = None
        self.shared_embeddings = None


# 全局實例（用於 UI）
_private_rag_instance: Optional[PrivateFileRAG] = None


def get_private_rag_instance() -> PrivateFileRAG:
    """獲取全局私有文件 RAG 實例"""
    global _private_rag_instance
    if _private_rag_instance is None:
        _private_rag_instance = PrivateFileRAG()
    return _private_rag_instance


def reset_private_rag_instance():
    """重置全局實例"""
    global _private_rag_instance
    _private_rag_instance = None

