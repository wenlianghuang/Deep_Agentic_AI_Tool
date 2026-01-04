# deep_agent_rag/ui/private_file_rag_interface.py

import gradio as gr
import re
import json
import os
import time

from ..rag.private_file_rag import get_private_rag_instance, reset_private_rag_instance
# Assuming is_using_local_llm might be used for warnings/status, similar to email_interface
# from ..utils.llm_utils import is_using_local_llm 

# Agent log path for debugging (if needed)
log_path = "/Users/matthuang/Desktop/Deep_Agentic_AI_Tool/.cursor/debug.log"

def _create_private_file_rag_interface():
    """創建私有文件 RAG 界面（對話式 Chatbot）"""
    gr.Markdown(
        """
        ### 📚 私有文件 RAG 對話系統
        
        上傳您的私有文件（PDF、DOCX、TXT），系統會自動建立 RAG 知識庫，讓 AI 可以回答關於這些文件的問題。
        支持多輪對話，AI 會記住之前的對話內容，提供更連貫的回答。
        
        **使用方式：**
        1. 上傳一個或多個文件（PDF、DOCX、TXT）
        2. 點擊「處理文件」按鈕，系統會自動處理文件並建立 RAG 系統
        3. 在對話框中輸入您的問題，按 Enter 或點擊「發送」按鈕
        4. AI 會基於上傳的文件回答問題，支持多輪對話
        
        **功能特色：**
        - 💬 **對話式界面** ：類似 Gemini 的對話體驗，支持多輪對話
        - 📄 支持多種文件格式：PDF、DOCX、TXT
        - 🔍 使用混合搜尋（BM25 + 向量檢索）提升檢索準確度
        - 🎯 可選重排序功能，進一步優化結果
        - 🧠 支持語義分塊，保持語義完整性
        - 🌐 自動檢測文檔類型並調整回答風格
        
        **LLM 使用策略：**
        - 🥇 **優先使用 Groq API** ：如果配置了 API 金鑰，優先使用 Groq（速度快、質量高）
        - 🥈 **其次使用 Ollama** ：如果 Groq 不可用，自動切換到 Ollama 本地模型
        - 🥉 **最後使用 MLX** ：如果前兩者都不可用，使用 MLX 本地模型作為備選
        - 💡 **自動切換** ：系統會根據 API 額度、服務狀態等自動選擇最合適的 LLM
        
        **注意：** 此功能需要 Learn_RAG 項目在正確的位置
        """
    )
    
    # 對話歷史狀態
    chat_history = gr.State(value=[])
    
    with gr.Row():
        # 左側：文件上傳和設置
        with gr.Column(scale=1):
            # 文件上傳區域
            file_upload = gr.File(
                label="📁 上傳文件（PDF、DOCX、TXT）",
                file_count="multiple",
                file_types=[ ".pdf", ".docx", ".doc", ".txt"]
            )
            
            # 處理按鈕
            with gr.Row():
                process_btn = gr.Button("📝 處理文件", variant="primary", scale=1)
                clear_files_btn = gr.Button("🗑️ 清除所有", variant="secondary", scale=1)
            
            # 處理狀態
            process_status = gr.Textbox(
                label="📊 處理狀態",
                value="等待上傳文件...",
                interactive=False,
                lines=2
            )
            
            # 設置區域（使用 Accordion 摺疊）
            with gr.Accordion("⚙️ 進階設置", open=False):
                # 處理選項
                use_semantic_chunking = gr.Checkbox(
                    label="使用語義分塊（推薦）",
                    value=False,
                    info="語義分塊能保持語義完整性，但處理時間較長"
                )
                
                # 分塊參數調整（字符分塊模式）
                gr.Markdown("**📏 字符分塊參數（僅在未使用語義分塊時有效）**")
                chunk_size_slider = gr.Slider(
                    minimum=200,
                    maximum=1500,
                    value=500,
                    step=50,
                    label="分塊大小（字符數）",
                    info="建議：300-800"
                )
                chunk_overlap_slider = gr.Slider(
                    minimum=0,
                    maximum=300,
                    value=100,
                    step=25,
                    label="分塊重疊（字符數）",
                    info="建議：chunk_size 的 15-25%"
                )
                
                # 語義分塊參數調整（僅在使用語義分塊時有效）
                gr.Markdown("**🔬 語義分塊參數（僅在使用語義分塊時有效）**")
                semantic_threshold_slider = gr.Slider(
                    minimum=0.5,
                    maximum=2.5,
                    value=1.0,
                    step=0.1,
                    label="語義分塊閾值（敏感度）",
                    info="建議：0.8-1.2（細粒度）"
                )
                semantic_min_chunk_slider = gr.Slider(
                    minimum=50,
                    maximum=300,
                    value=100,
                    step=25,
                    label="最小分塊大小（字符數）",
                    info="建議：50-200"
                )
                
                # RAG 方法選擇
                gr.Markdown("**🎯 RAG 方法選擇**")
                enable_adaptive_selection = gr.Checkbox(
                    label="自動選擇最佳 RAG 方法（推薦）",
                    value=True,
                    info="系統會根據查詢和文件特征自動選擇最合適的 RAG 方法"
                )
                manual_rag_method = gr.Dropdown(
                    choices=[
                        "basic",
                        "subquery",
                        "hyde",
                        "step_back",
                        "hybrid_subquery_hyde",
                        "triple_hybrid"
                    ],
                    value="basic",
                    label="手動選擇 RAG 方法",
                    info="僅在自動選擇關閉時生效",
                    visible=False
                )
                
                # 查詢選項
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="返回結果數量"
                )
                use_llm_checkbox = gr.Checkbox(
                    label="使用 LLM 生成回答",
                    value=True
                )
        
        # 右側：對話界面
        with gr.Column(scale=2):
            # Chatbot 組件
            # #region agent log
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A",
                        "location": "private_file_rag_interface.py:1409", # Adjusted line number
                        "message": "Before Chatbot creation",
                        "data": {
                            "gradio_version": gr.__version__ if hasattr(gr, '__version__') else "unknown"
                        },
                        "timestamp": int(time.time() * 1000)
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except:
                pass
            # #endregion
            
            # 創建 Chatbot（移除不支持的參數：show_copy_button 和 avatar_images）
            # #region agent log
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "A",
                        "location": "private_file_rag_interface.py:1430", # Adjusted line number
                        "message": "Creating Chatbot with minimal params",
                        "data": {"params": ["label", "height"]},
                        "timestamp": int(time.time() * 1000)
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except:
                pass
            # #endregion
            
            try:
                chatbot = gr.Chatbot(
                    label="💬 對話",
                    height=500
                )
                # #region agent log
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        log_entry = {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "A",
                            "location": "private_file_rag_interface.py:1448", # Adjusted line number
                            "message": "Chatbot created successfully",
                            "data": {"success": True},
                            "timestamp": int(time.time() * 1000)
                        }
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                except:
                    pass
                # #endregion
            except Exception as e:
                # #region agent log
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        log_entry = {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "A",
                            "location": "private_file_rag_interface.py:1460", # Adjusted line number
                            "message": "Chatbot creation failed",
                            "data": {
                                "error_type": type(e).__name__,
                                "error_message": str(e)
                            },
                            "timestamp": int(time.time() * 1000)
                        }
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                except:
                    pass
                # #endregion
                raise
            
            # 輸入框
            msg = gr.Textbox(
                label="輸入問題",
                placeholder="輸入您的問題，按 Enter 發送...",
                lines=2,
                scale=4
            )
            
            # 按鈕區域
            with gr.Row():
                submit_btn = gr.Button("📤 發送", variant="primary", scale=1)
                clear_chat_btn = gr.Button("🗑️ 清除對話", variant="secondary", scale=1)
            
            # 查詢狀態
            query_status = gr.Textbox(
                label="📊 狀態",
                value="等待查詢...",
                interactive=False,
                lines=1
            )
    
    # 輔助函數：轉換 Gradio 歷史格式（dict）和 RAG 歷史格式（tuple）
    def history_dict_to_tuple(history_dict):
        """
        將 Gradio 歷史格式（List[Dict]）轉換為 RAG 歷史格式（List[Tuple[str, str]]）
        
        Args:
            history_dict: Gradio 格式的歷史，每個元素為 {"role": "user"/"assistant", "content": "..."}
        
        Returns:
            RAG 格式的歷史，每個元素為 (user_message, assistant_message)
        """
        if not history_dict:
            return []
        
        conversation_history = []
        current_user_msg = None
        
        for msg in history_dict:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "user":
                    current_user_msg = content
                elif role == "assistant" and current_user_msg is not None:
                    conversation_history.append((current_user_msg, content))
                    current_user_msg = None
            elif isinstance(msg, tuple) and len(msg) == 2:
                # 如果已經是 tuple 格式，直接使用（向後兼容）
                conversation_history.append(msg)
        
        return conversation_history
    
    def history_tuple_to_dict(history_tuple):
        """
        將 RAG 歷史格式（List[Tuple[str, str]]）轉換為 Gradio 歷史格式（List[Dict]）
        
        Args:
            history_tuple: RAG 格式的歷史，每個元素為 (user_message, assistant_message)
        
        Returns:
            Gradio 格式的歷史，每個元素為 {"role": "user"/"assistant", "content": "..."}
        """
        if not history_tuple:
            return []
        
        history_dict = []
        for msg in history_tuple:
            if isinstance(msg, dict):
                # 如果已經是 dict 格式，直接使用
                history_dict.append(msg)
            elif isinstance(msg, tuple) and len(msg) == 2:
                # 轉換 tuple 為 dict 格式
                user_msg, assistant_msg = msg
                history_dict.append({"role": "user", "content": user_msg})
                history_dict.append({"role": "assistant", "content": assistant_msg})
        
        return history_dict
    
    def ensure_dict_format(history):
        """
        確保歷史是 Gradio dict 格式
        
        Args:
            history: 歷史列表（可能是 dict 或 tuple 格式，也可能是 None）
        
        Returns:
            Gradio 格式的歷史（List[Dict]）
        """
        if not history:
            return []
        
        # 檢查第一個元素的類型來判斷格式
        try:
            if isinstance(history[0], dict):
                return history
            elif isinstance(history[0], tuple):
                return history_tuple_to_dict(history)
            else:
                # 未知格式，返回空列表
                return []
        except (IndexError, TypeError):
            # 如果 history 為空或無法索引，返回空列表
            return []
    
    # 事件處理函數
    def process_files(files, use_semantic, chunk_size, chunk_overlap, semantic_threshold, semantic_min_chunk):
        """
        處理上傳的文件
        
        Args:
            files: 上傳的文件列表
            use_semantic: 是否使用語義分塊
            chunk_size: 字符分塊大小（僅用於字符分塊模式）
            chunk_overlap: 字符分塊重疊大小（僅用於字符分塊模式）
            semantic_threshold: 語義分塊閾值（僅用於語義分塊模式）
            semantic_min_chunk: 語義分塊最小 chunk 大小（僅用於語義分塊模式）
        """
        if not files:
            return "❌ 請先上傳文件", "等待上傳文件..."
        
        try:
            # 獲取 RAG 實例
            rag = get_private_rag_instance()
            
            # 更新配置
            rag.use_semantic_chunking = use_semantic
            
            # 更新分塊參數（根據分塊模式選擇）
            if not use_semantic:
                # 字符分塊模式：更新字符分塊參數
                rag.chunk_size = int(chunk_size)
                rag.chunk_overlap = int(chunk_overlap)
                print(f"📏 使用字符分塊：chunk_size={rag.chunk_size}, chunk_overlap={rag.chunk_overlap}")
            else:
                # 語義分塊模式：更新語義分塊參數
                rag.semantic_threshold = float(semantic_threshold)
                rag.semantic_min_chunk_size = int(semantic_min_chunk)
                print(f"📏 使用語義分塊：threshold={rag.semantic_threshold}, min_chunk_size={rag.semantic_min_chunk_size}")
            
            # 處理上傳的文件（Gradio 會自動保存到臨時目錄）
            # Gradio 6.x 返回的是文件路徑字符串列表
            file_paths = []
            
            for file in files:
                # Gradio 6.x 返回字符串路徑，舊版本可能返回文件對象
                if isinstance(file, str):
                    file_path = file
                elif hasattr(file, 'name'):
                    # 舊版本 Gradio 文件對象
                    file_path = file.name
                else:
                    # 嘗試轉換為字符串
                    file_path = str(file)
                
                if os.path.exists(file_path):
                    file_paths.append(file_path)
                else:
                    return f"❌ 文件不存在: {file_path}", "處理失敗"
            
            if not file_paths:
                return "❌ 沒有有效的文件路徑", "處理失敗"
            
            # 處理文件
            documents, status_msg = rag.process_files(file_paths)
            
            if documents:
                return status_msg, "✅ 文件處理完成，可以開始查詢"
            else:
                return status_msg, "❌ 處理失敗"
                
        except Exception as e:
            error_msg = f"❌ 處理文件時發生錯誤: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, "❌ 處理失敗"
    
    def query_rag_stream(message, history, top_k, use_llm, enable_adaptive, manual_method):
        """
        查詢 RAG 系統（對話式，流式輸出）
        
        Args:
            message: 當前用戶消息
            history: 對話歷史（Gradio 格式：List[Dict] 或 List[Tuple[str, str]]）
            top_k: 返回結果數量
            use_llm: 是否使用 LLM 生成回答
            enable_adaptive: 是否啟用自動選擇
            manual_method: 手動選擇的方法（僅在自動選擇關閉時生效）
        
        Yields:
            Tuple[history, status_msg]: 逐步更新的對話歷史和狀態訊息
        """
        if not message or not message.strip():
            yield history, "❌ 請輸入問題"
            return
        
        try:
            # 獲取 RAG 實例
            rag = get_private_rag_instance()
            
            if not rag.is_initialized:
                error_msg = "❌ RAG 系統尚未初始化，請先處理文件"
                # 確保 history 是 dict 格式
                history = ensure_dict_format(history)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": error_msg})
                yield history, error_msg
                return
            
            # 設置 RAG 方法選擇參數
            rag.enable_adaptive_selection = enable_adaptive
            if not enable_adaptive:
                rag.selected_rag_method = manual_method
            else:
                rag.selected_rag_method = None
            
            # 準備對話歷史：轉換為 RAG 需要的 tuple 格式
            conversation_history = history_dict_to_tuple(history) if history else []
            
            # 確保 history 是 dict 格式並添加用戶消息
            history = ensure_dict_format(history)
            history.append({"role": "user", "content": message})
            
            # 執行查詢（傳入對話歷史，使用流式輸出）
            if use_llm:
                # 使用流式查詢
                answer_generator = rag.query_stream(
                    query=message,
                    top_k=int(top_k),
                    conversation_history=conversation_history
                )
                
                # 初始化回答
                accumulated_answer = ""
                history_with_user = history.copy()
                final_result = {}
                
                # 逐步接收流式回答
                for chunk in answer_generator:
                    if chunk.get("success") is False:
                        error = chunk.get("error", "未知錯誤")
                        error_msg = f"❌ 查詢失敗: {error}"
                        history_with_user.append({"role": "assistant", "content": error_msg})
                        yield history_with_user, error_msg
                        return
                    
                    # 保存最後一個 chunk 作為最終結果
                    final_result = chunk
                    
                    # 獲取新的回答片段
                    new_answer = chunk.get("answer", "")
                    if new_answer:
                        # 累積回答
                        accumulated_answer = new_answer
                        # 更新歷史
                        history_with_answer = history_with_user.copy()
                        history_with_answer.append({"role": "assistant", "content": accumulated_answer})
                        yield history_with_answer, "🔄 正在生成回答..."
                
                # 獲取最終結果（包含統計信息）
                rag_method = final_result.get("rag_method", "basic")
                stats = final_result.get("stats", {})
                status_msg = f"✅ 查詢完成（方法: {rag_method.upper()}）"
                if stats:
                    total_time = stats.get("total_time", 0)
                    if total_time > 0:
                        status_msg += f" | 耗時: {total_time:.2f}秒"
                
                # 確保最終回答完整
                if accumulated_answer:
                    history_with_answer = history_with_user.copy()
                    history_with_answer.append({"role": "assistant", "content": accumulated_answer})
                    yield history_with_answer, status_msg
                else:
                    error_msg = "⚠️ LLM 未生成回答（可能 LLM 服務未啟動）"
                    history_with_answer = history_with_user.copy()
                    history_with_answer.append({"role": "assistant", "content": error_msg})
                    yield history_with_answer, status_msg
            else:
                # 不使用 LLM，直接返回檢索結果
                result = rag.query(
                    query=message,
                    top_k=int(top_k),
                    use_llm=False,
                    conversation_history=conversation_history
                )
                
                if not result.get("success"):
                    error = result.get("error", "未知錯誤")
                    error_msg = f"❌ 查詢失敗: {error}"
                    history.append({"role": "assistant", "content": error_msg})
                    yield history, error_msg
                    return
                
                # 格式化檢索結果
                formatted_context = result.get("formatted_context", "")
                answer = f"📄 檢索到的相關內容：\n\n{formatted_context}"
                
                # 獲取 RAG 方法信息
                rag_method = result.get("rag_method", "basic")
                stats = result.get("stats", {})
                status_msg = f"✅ 查詢完成（方法: {rag_method.upper()}）"
                if stats:
                    total_time = stats.get("total_time", 0)
                    if total_time > 0:
                        status_msg += f" | 耗時: {total_time:.2f}秒"
                
                history.append({"role": "assistant", "content": answer})
                yield history, status_msg
            
        except Exception as e:
            error_msg = f"❌ 查詢時發生錯誤: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            # 確保 history 是 dict 格式
            history = ensure_dict_format(history)
            if not any(msg.get("role") == "user" and msg.get("content") == message for msg in history):
                history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            yield history, error_msg
    
    def clear_chat():
        """清除對話歷史（不重置 RAG 系統）"""
        return [], "對話已清除"
    
    def clear_all():
        """清除所有內容（包括 RAG 系統）"""
        reset_private_rag_instance()
        empty_history = []
        return (
            None,  # file_upload
            False,  # use_semantic_chunking
            500,  # chunk_size_slider
            100,  # chunk_overlap_slider
            1.0,  # semantic_threshold_slider
            100,  # semantic_min_chunk_slider
            True,  # enable_adaptive_selection
            "basic",  # manual_rag_method
            "等待上傳文件...",  # process_status
            empty_history,  # chatbot (對話歷史)
            empty_history,  # chat_history (狀態)
            "等待查詢...",  # query_status
        )
    
    # 綁定事件
    process_btn.click(
        fn=process_files,
        inputs=[
            file_upload, 
            use_semantic_chunking, 
            chunk_size_slider, 
            chunk_overlap_slider,
            semantic_threshold_slider,
            semantic_min_chunk_slider
        ],
        outputs=[process_status, query_status]
    )
    
    # 自動選擇開關時顯示/隱藏手動選擇下拉菜單
    def toggle_manual_method(enable_adaptive):
        return gr.update(visible=not enable_adaptive)
    
    enable_adaptive_selection.change(
        fn=toggle_manual_method,
        inputs=[enable_adaptive_selection],
        outputs=[manual_rag_method]
    )
    
    # 提交消息（按鈕點擊或 Enter 鍵）
    def submit_message(message, history, top_k, use_llm, enable_adaptive, manual_method):
        """提交消息並更新對話歷史（流式輸出）"""
        if not message or not message.strip():
            # 確保 history 是 dict 格式
            history = ensure_dict_format(history)
            return history, history, "", "等待查詢..."
        # 清空輸入框並執行流式查詢
        for new_history, status in query_rag_stream(message, history, top_k, use_llm, enable_adaptive, manual_method):
            yield new_history, new_history, "", status
    
    # 綁定提交按鈕和 Enter 鍵
    submit_btn.click(
        fn=submit_message,
        inputs=[msg, chat_history, top_k_slider, use_llm_checkbox, enable_adaptive_selection, manual_rag_method],
        outputs=[chatbot, chat_history, msg, query_status]
    )
    
    msg.submit(
        fn=submit_message,
        inputs=[msg, chat_history, top_k_slider, use_llm_checkbox, enable_adaptive_selection, manual_rag_method],
        outputs=[chatbot, chat_history, msg, query_status]
    )
    
    # 清除對話按鈕（需要更新 chat_history 狀態）
    def clear_chat_with_state():
        """清除對話歷史並更新狀態"""
        empty_history = []
        return empty_history, empty_history, "對話已清除"
    
    clear_chat_btn.click(
        fn=clear_chat_with_state,
        outputs=[chatbot, chat_history, query_status]
    )
    
    # 清除所有按鈕
    clear_files_btn.click(
        fn=clear_all,
        outputs=[
            file_upload,
            use_semantic_chunking,
            chunk_size_slider,
            chunk_overlap_slider,
            semantic_threshold_slider,
            semantic_min_chunk_slider,
            enable_adaptive_selection,
            manual_rag_method,
            process_status,
            chatbot,  # 更新 chatbot 顯示
            chat_history,  # 更新 chat_history 狀態
            query_status
        ]
    )
