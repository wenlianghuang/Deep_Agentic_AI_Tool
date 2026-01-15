"""
圖片分析界面模組
提供獨立的圖片分析 UI
"""
import gradio as gr
import os
import tempfile
from typing import Tuple, Optional

from ..tools.image_analysis_tool import get_multimodal_llm
from ..graph.image_analysis_graph import build_image_analysis_graph
from ..config import (
    OPENAI_API_KEY,
    GOOGLE_GEMINI_API_KEY,
    ANTHROPIC_API_KEY,
    USE_OLLAMA_VISION,
    OLLAMA_VISION_MODEL,
    MAX_REFLECTION_ITERATION,
)


def get_available_providers() -> str:
    """
    獲取當前可用的多模態 API 提供商信息
    
    Returns:
        提供商信息字符串
    """
    providers = []
    
    if OPENAI_API_KEY:
        providers.append("✅ OpenAI GPT-4 Vision")
    else:
        providers.append("❌ OpenAI GPT-4 Vision (未配置)")
    
    if GOOGLE_GEMINI_API_KEY:
        providers.append("✅ Google Gemini")
    else:
        providers.append("❌ Google Gemini (未配置)")
    
    if ANTHROPIC_API_KEY:
        providers.append("✅ Anthropic Claude")
    else:
        providers.append("❌ Anthropic Claude (未配置)")
    
    if USE_OLLAMA_VISION:
        providers.append("✅ Ollama LLaVA (本地)")
    else:
        providers.append("❌ Ollama LLaVA (未啟用)")
    
    return "\n".join(providers)


def analyze_image_ui(
    image: Optional[gr.File],
    question: str,
    progress: Optional[gr.Progress] = None
) -> Tuple[str, str, str]:
    """
    分析圖片的 UI 處理函數（使用 LangGraph 工作流，包含反思）
    
    Args:
        image: 上傳的圖片文件
        question: 可選的問題
        progress: Gradio 進度條（可選）
    
    Returns:
        (分析結果, 反思結果, 狀態訊息)
    """
    if image is None:
        return "", "", "❌ 請先上傳一張圖片"
    
    try:
        # 獲取圖片文件路徑
        if isinstance(image, str):
            image_path = image
        elif hasattr(image, 'name'):
            image_path = image.name
        elif isinstance(image, dict) and 'name' in image:
            image_path = image['name']
        else:
            return "", "", "❌ 無法讀取圖片文件，請重新上傳"
        
        # 檢查文件是否存在
        if not image_path or not os.path.exists(image_path):
            return "", "", f"❌ 圖片文件不存在：{image_path}"
        
        # 顯示當前使用的 API 提供商
        llm, provider = get_multimodal_llm()
        if llm is None:
            return "", "", provider  # 返回錯誤訊息
        
        status_msg = f"🔄 正在使用 {provider.upper()} 分析圖片（包含 AI 反思）..."
        if progress is not None:
            progress(0.1, desc=status_msg)
        
        # 構建問題（如果提供）
        question_text = question.strip() if question else None
        
        # 構建 LangGraph 工作流
        graph = build_image_analysis_graph()
        
        # 初始化狀態
        initial_state = {
            "question": question_text,
            "image_path": image_path,
            "analysis_result": "",
            "reflection_result": "",
            "improvement_suggestions": "",
            "needs_revision": False,
            "iteration": 0,
            "messages": []
        }
        
        # 執行工作流
        config = {"configurable": {"thread_id": f"image-analysis-{os.path.basename(image_path)}"}}
        
        final_analysis = ""
        final_reflection = ""
        current_status = status_msg
        final_iteration = 0
        
        if progress is not None:
            progress(0.2, desc="開始分析圖片...")
        
        # 流式執行工作流
        for event in graph.stream(initial_state, config, stream_mode="updates"):
            for node_name, node_state in event.items():
                iteration = node_state.get("iteration", 0)
                final_iteration = max(final_iteration, iteration)
                
                if progress is not None:
                    max_iter = MAX_REFLECTION_ITERATION + 1  # +1 因為初始分析也算一輪
                    progress_val = min(0.2 + (iteration / max_iter) * 0.7, 0.9)
                    
                if node_name == "analyze":
                    current_status = f"🔄 第 {iteration} 輪：正在分析圖片..."
                    if progress is not None:
                        progress(progress_val, desc=current_status)
                
                elif node_name == "reflection":
                    current_status = f"🔍 第 {iteration} 輪：正在反思分析結果..."
                    if "reflection_result" in node_state and node_state["reflection_result"]:
                        final_reflection = node_state["reflection_result"]
                    if progress is not None:
                        progress(progress_val, desc=current_status)
                
                elif node_name == "improvement":
                    current_status = f"✨ 第 {iteration} 輪：正在生成改進版本..."
                    if progress is not None:
                        progress(progress_val, desc=current_status)
                
                # 更新最終結果
                if "analysis_result" in node_state:
                    final_analysis = node_state["analysis_result"]
                if "reflection_result" in node_state and node_state["reflection_result"]:
                    final_reflection = node_state["reflection_result"]
        
        if progress is not None:
            progress(1.0, desc="分析完成！")
        
        # 檢查結果是否為錯誤訊息
        if final_analysis.startswith("❌"):
            return "", "", final_analysis
        
        # 構建最終狀態訊息
        if final_iteration > 1:
            final_status = f"✅ 分析完成！使用 {provider.upper()} API（經過 {final_iteration} 輪分析）"
        else:
            final_status = f"✅ 分析完成！使用 {provider.upper()} API"
        
        return final_analysis, final_reflection, final_status
        
    except Exception as e:
        error_msg = f"❌ 分析圖片時發生錯誤：{str(e)}"
        print(f"圖片分析錯誤：{e}")
        import traceback
        traceback.print_exc()
        return "", "", error_msg




def _create_image_analysis_interface():
    """創建圖片分析界面"""
    gr.Markdown(
        """
        ### 🖼️ 智能圖片分析工具
        
        使用多模態 AI 模型分析圖片內容，支持多個 API 提供商自動切換。
        
        **支持的功能：**
        - 📸 圖片內容識別和描述
        - 🔍 回答關於圖片的特定問題
        - 🎨 分析圖片風格、構圖、色彩等視覺特徵
        - 📝 識別圖片中的文字（OCR）
        
        **支持的 API 提供商：**
        - OpenAI GPT-4 Vision
        - Google Gemini（推薦，免費額度較高）
        - Anthropic Claude
        - Ollama LLaVA（本地，完全免費）
        
        **使用方式：**
        1. 上傳一張圖片（支持 jpg, png, gif, webp 等格式）
        2. （可選）輸入特定問題，例如："這張圖片中有什麼？"、"描述圖片中的場景"
        3. 點擊「分析圖片」按鈕
        4. 查看 AI 的分析結果和反思評估
        
        **✨ 新功能：AI 反思評估**
        - 系統會自動評估分析結果的質量
        - 如果分析有改進空間，會自動生成改進版本
        - 最多進行 {MAX_REFLECTION_ITERATION} 輪改進（避免免費 API 額度快速用完）
        
        **提示：** 如果不輸入問題，系統會進行通用的圖片分析。
        """
    )
    
    # 顯示當前可用的 API 提供商
    with gr.Accordion("📊 當前可用的 API 提供商", open=False):
        providers_display = gr.Markdown(
            value=get_available_providers(),
            elem_classes=["provider-info"]
        )
        
        refresh_providers_btn = gr.Button("🔄 刷新提供商狀態", variant="secondary", size="sm")
        
        def refresh_providers():
            return get_available_providers()
        
        refresh_providers_btn.click(
            fn=refresh_providers,
            outputs=[providers_display]
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            # 圖片上傳區域
            image_input = gr.File(
                label="📸 上傳圖片",
                file_types=["image"],
                type="filepath"
            )
            
            # 圖片預覽
            image_preview = gr.Image(
                label="圖片預覽",
                type="filepath",
                height=300,
                visible=True
            )
            
            # 問題輸入（可選）
            question_input = gr.Textbox(
                label="❓ 問題（可選）",
                placeholder="例如：這張圖片中有什麼？描述圖片中的場景。如果不填，將進行通用分析。",
                lines=3,
                value=""
            )
            
            # 按鈕
            with gr.Row():
                analyze_btn = gr.Button("🔍 分析圖片", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ 清除", variant="secondary", scale=1)
            
            # 狀態顯示
            status_display = gr.Textbox(
                label="📊 狀態",
                value="等待上傳圖片...",
                interactive=False,
                lines=2
            )
        
        with gr.Column(scale=1):
            # 分析結果顯示
            result_display = gr.Textbox(
                label="📄 分析結果",
                placeholder="分析結果將顯示在這裡...",
                lines=12,
                interactive=True  # 設為 True 以便用戶可以複製內容
            )
            
            # 反思結果顯示
            reflection_display = gr.Textbox(
                label="🔍 AI 反思評估",
                placeholder="AI 反思評估結果將顯示在這裡...",
                lines=8,
                interactive=False,
                visible=True
            )
    
    # 更新圖片預覽
    def update_image_preview(image):
        """更新圖片預覽"""
        if image is None:
            return None
        
        # 處理不同類型的輸入
        if hasattr(image, 'name'):
            file_path = image.name
        elif isinstance(image, str):
            file_path = image
        elif isinstance(image, dict) and 'name' in image:
            file_path = image['name']
        else:
            return None
        
        # 檢查文件是否存在
        if file_path and os.path.exists(file_path):
            return file_path
        return None
    
    image_input.change(
        fn=update_image_preview,
        inputs=[image_input],
        outputs=[image_preview]
    )
    
    # 分析按鈕事件
    analyze_btn.click(
        fn=analyze_image_ui,
        inputs=[image_input, question_input],
        outputs=[result_display, reflection_display, status_display],
        show_progress="full"
    )
    
    # 清除按鈕事件
    def clear_image_analysis():
        """清除所有輸入和輸出"""
        return None, "", "", "", "等待上傳圖片..."
    
    clear_btn.click(
        fn=clear_image_analysis,
        outputs=[image_input, question_input, result_display, reflection_display, status_display]
    )
    
    # 示例問題
    gr.Examples(
        examples=[
            "這張圖片中有什麼？",
            "描述圖片中的場景和人物",
            "圖片中的文字是什麼？",
            "分析這張圖片的風格和構圖",
            "這張圖片表達了什麼情感或意義？"
        ],
        inputs=question_input,
        label="💡 快速問題範例"
    )
    
    # 使用說明
    gr.Markdown(
        """
        ---
        **💡 使用技巧：**
        
        1. **通用分析**：不上傳問題，讓 AI 自動分析圖片的所有方面
        2. **特定問題**：輸入具體問題，獲得針對性的回答
        3. **多張圖片**：目前一次只能分析一張圖片，如需分析多張，請分別上傳
        4. **圖片格式**：支持常見圖片格式（jpg, png, gif, webp, bmp）
        5. **API 切換**：系統會自動選擇可用的 API 提供商，無需手動配置
        
        **⚠️ 注意事項：**
        - 圖片大小建議不超過 10MB
        - 如果使用本地 Ollama LLaVA，首次使用可能需要下載模型
        - 不同 API 提供商的免費額度不同，建議配置多個作為備援
        """
    )
