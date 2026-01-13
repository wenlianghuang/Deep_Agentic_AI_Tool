"""
Simple Chatbot Interface
簡單的聊天機器人界面，不包含 RAG 和 Deep AI Agent 功能
純粹的對話式聊天機器人
包含內容過濾 Guardrails 功能（混合式：關鍵字 + 語義過濾）
"""
import gradio as gr
from typing import List, Dict, Any
import time
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
import jieba

from ..utils.llm_utils import get_llm_type, is_using_local_llm, get_llm
from ..guardrails.nemo_manager import get_guardrail_manager


# ==================== Guardrails 配置 ====================
# 敏感關鍵字名單
BLOCKED_KEYWORDS = [
    "伊斯蘭教",
    "阿拉",
    "回教徒",
    "默罕默德",
    "Islam",
    "Allah",
    "Muslim",
    "Muhammad"
]

# 攔截門檻：5% 的詞彙密度
KEYWORD_DENSITY_THRESHOLD = 0.05

# 預設攔截訊息
DEFAULT_BLOCKED_MESSAGE = "抱歉，您的問題包含敏感內容，無法回答。請換個話題或重新表述您的問題。"

# 初始化 jieba 自定義詞典（確保準確識別敏感詞）
def _init_jieba_custom_dict():
    """初始化 jieba 自定義詞典，添加敏感關鍵字以提高識別準確度"""
    for keyword in BLOCKED_KEYWORDS:
        jieba.add_word(keyword, freq=10000, tag='sensitive')

# 在模組載入時初始化
_init_jieba_custom_dict()


def check_content_guardrails(text: str) -> tuple[bool, float]:
    """
    檢查文本是否包含敏感內容
    
    使用 jieba 進行斷詞，計算敏感詞彙密度
    支持中文和英文（不區分大小寫）
    
    Args:
        text: 要檢查的文本
    
    Returns:
        tuple[bool, float]: (是否應該攔截, 關鍵字密度)
    """
    if not text or not text.strip():
        return False, 0.0
    
    # 使用 jieba 進行斷詞
    words = list(jieba.cut(text))
    total_words = len(words)
    
    if total_words == 0:
        return False, 0.0
    
    # 建立小寫敏感詞集合以便快速比對
    blocked_keywords_lower = {k.lower() for k in BLOCKED_KEYWORDS}
    
    # 計算敏感詞數量
    sensitive_word_count = 0
    for word in words:
        # 移除空白並轉為小寫進行比對
        clean_word = word.strip().lower()
        if clean_word and clean_word in blocked_keywords_lower:
            sensitive_word_count += 1
    
    # 計算關鍵字密度
    keyword_density = sensitive_word_count / total_words
    
    # 判斷是否超過門檻
    should_block = keyword_density >= KEYWORD_DENSITY_THRESHOLD
    
    return should_block, keyword_density


def guardrail_filter(response: str) -> str:
    """
    Guardrail 過濾函數 - 用於 LangChain RunnableLambda
    
    Args:
        response: LLM 的回應文本
    
    Returns:
        str: 過濾後的文本（如果超過門檻則返回預設訊息）
    """
    should_block, density = check_content_guardrails(response)
    
    if should_block:
        print(f"🚫 Guardrails 攔截：關鍵字密度 {density:.2%} 超過門檻 {KEYWORD_DENSITY_THRESHOLD:.2%}")
        return DEFAULT_BLOCKED_MESSAGE
    else:
        print(f"🟢 Guardrails 通過：關鍵字密度 {density:.2%} 低於門檻 {KEYWORD_DENSITY_THRESHOLD:.2%}")
        
    return response


# 創建 LangChain RunnableLambda（用於串接在 Chain 末端）
guardrail_runnable = RunnableLambda(guardrail_filter)


def chat_with_llm_streaming(
    message: str,
    history: List[Dict[str, str]],
    system_prompt: str = "你是一個有幫助的AI助手。請用繁體中文回答問題。",
    enable_guardrails: bool = True
):
    """
    與 LLM 進行流式對話（逐字顯示）
    整合混合式 Guardrails（關鍵字 + 語義過濾）
    
    Args:
        message: 用戶輸入的消息
        history: 對話歷史 (字典格式：[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...])
        system_prompt: 系統提示詞
        enable_guardrails: 是否啟用 Guardrails 內容過濾
    
    Yields:
        List[Dict[str, str]]: 更新中的歷史記錄
    """
    if not message or not message.strip():
        yield history
        return
    
    # ==================== 立即顯示用戶消息 ====================
    # 先將用戶消息添加到歷史並立即顯示
    new_history = history + [{"role": "user", "content": message}]
    yield new_history
    
    try:
        # ==================== 輸入過濾檢查 ====================
        # 根據 checkbox 狀態決定是否使用 Guardrails
        if enable_guardrails:
            guardrail_mgr = get_guardrail_manager()
            should_block_input, blocked_message = guardrail_mgr.check_input(message)
            
            if should_block_input:
                # 輸入被阻擋，逐字顯示阻擋訊息
                print(f"🚫 輸入被阻擋")
                new_history.append({"role": "assistant", "content": ""})
                
                # 按字符逐步顯示阻擋訊息（創造打字效果）
                for i in range(len(blocked_message)):
                    new_history[-1] = {"role": "assistant", "content": blocked_message[:i+1]}
                    yield new_history
                    time.sleep(0.01)  # 10ms 延遲
                
                # 確保完整顯示
                new_history[-1] = {"role": "assistant", "content": blocked_message}
                yield new_history
                return
        
        # 獲取 LLM
        llm = get_llm()
        
        # 構建消息列表
        messages = [SystemMessage(content=system_prompt)]
        
        # 添加對話歷史
        for msg in history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # 添加當前用戶消息
        messages.append(HumanMessage(content=message))
        
        # 添加空的助手回應（將逐步填充）
        new_history.append({"role": "assistant", "content": ""})
        full_response = ""
        
        # 使用流式調用獲取回應
        for chunk in llm.stream(messages):
            # 獲取內容 (chunk 可能是 BaseMessageChunk)
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            
            # 按字符平滑顯示內容（增加一點打字感）
            for char in content:
                full_response += char
                new_history[-1] = {"role": "assistant", "content": full_response}
                yield new_history
                # 如果是高速模型，稍微延遲一點讓視覺更平滑
                time.sleep(0.005)
            
            # ==================== 輸出過濾即時檢查 (快速層) ====================
            if enable_guardrails:
                # 進行快速的關鍵字密度檢查，避免等到生成完才發現
                should_block_fast, _ = check_content_guardrails(full_response)
                if should_block_fast:
                    print(f"🚫 輸出因關鍵字密度被即時阻擋")
                    # 清空當前內容，準備逐字顯示阻擋訊息
                    new_history[-1] = {"role": "assistant", "content": ""}
                    yield new_history
                    
                    # 逐字顯示阻擋訊息
                    for i in range(len(DEFAULT_BLOCKED_MESSAGE)):
                        new_history[-1] = {"role": "assistant", "content": DEFAULT_BLOCKED_MESSAGE[:i+1]}
                        yield new_history
                        time.sleep(0.01)
                    
                    # 確保完整顯示
                    new_history[-1] = {"role": "assistant", "content": DEFAULT_BLOCKED_MESSAGE}
                    yield new_history
                    return
        
        # ==================== 最終輸出過濾檢查 (含語義) ====================
        # 根據 checkbox 狀態決定是否使用 Guardrails
        if enable_guardrails:
            guardrail_mgr = get_guardrail_manager()
            # 進行完整的檢查（包含可能較慢的語義過濾）
            should_block_output, filtered_response = guardrail_mgr.check_output(full_response)
            
            if should_block_output:
                print(f"🚫 輸出被最終語義過濾阻擋")
                # 清空當前內容，準備逐字顯示過濾後的訊息
                new_history[-1] = {"role": "assistant", "content": ""}
                yield new_history
                
                # 逐字顯示過濾後的訊息（例如自訂的主題攔截訊息）
                for i in range(len(filtered_response)):
                    new_history[-1] = {"role": "assistant", "content": filtered_response[:i+1]}
                    yield new_history
                    time.sleep(0.01)
                
                # 確保完整顯示
                new_history[-1] = {"role": "assistant", "content": filtered_response}
                yield new_history
            else:
                # 確保最終顯示的是完整的回應
                new_history[-1] = {"role": "assistant", "content": full_response}
                yield new_history
        else:
            # 確保完整顯示
            new_history[-1] = {"role": "assistant", "content": full_response}
            yield new_history
    
    except Exception as e:
        error_msg = f"❌ 發生錯誤: {str(e)}"
        print(f"聊天錯誤: {e}")
        import traceback
        traceback.print_exc()
        
        # 添加錯誤回應到已顯示用戶消息的歷史
        new_history.append({"role": "assistant", "content": error_msg})
        yield new_history


def get_llm_status() -> str:
    """獲取當前 LLM 狀態信息"""
    if is_using_local_llm():
        return "⚠️ **當前使用：本地 MLX 模型 (Qwen2.5)**\n\n本地模型處理速度可能較慢，請耐心等待。"
    else:
        llm_type = get_llm_type()
        if llm_type == "groq":
            return "✅ **當前使用：Groq API (高速雲端模型)**"
        else:
            return "ℹ️ **當前使用：本地 MLX 模型 (Qwen2.5)**"


def get_guardrails_status() -> str:
    """獲取當前 Guardrails 狀態信息"""
    try:
        guardrail_mgr = get_guardrail_manager()
        status = guardrail_mgr.get_status()
        topics = guardrail_mgr.get_topics_info()
        
        enabled = status.get("enabled", {})
        keyword_filter = status.get("keyword_filter", {})
        semantic_filter = status.get("semantic_filter", {})
        
        status_text = "# 🛡️ Guardrails 狀態\n\n"
        status_text += "## 混合過濾策略\n\n"
        
        # 關鍵字過濾狀態
        if enabled.get("keyword_filter", False):
            status_text += f"✅ **關鍵字過濾**：已啟用\n"
            status_text += f"   - 密度門檻：{keyword_filter.get('threshold', 0.05):.1%}\n"
            status_text += f"   - 關鍵字數量：{keyword_filter.get('keywords_count', 0)} 個\n\n"
        else:
            status_text += "❌ **關鍵字過濾**：已停用\n\n"
        
        # 語義過濾狀態
        if enabled.get("semantic_filter", False):
            if semantic_filter.get("initialized", False):
                status_text += f"✅ **語義主題過濾**：已啟用\n"
                status_text += f"   - 相似度門檻：{semantic_filter.get('threshold', 0.75):.1%}\n"
                status_text += f"   - 主題數量：{semantic_filter.get('topics_count', 0)} 個\n\n"
                
                if topics:
                    status_text += "   **主題列表**：\n"
                    for topic in topics:
                        status_text += f"   - {topic['display_name']} ({topic['examples_count']} 個範例)\n"
            else:
                status_text += "⚠️ **語義主題過濾**：啟用中（模型未初始化）\n\n"
        else:
            status_text += "❌ **語義主題過濾**：已停用\n\n"
        
        # 防護方向
        status_text += "\n## 防護方向\n\n"
        if enabled.get("input_rails", False):
            status_text += "✅ **輸入過濾**：已啟用（阻擋敏感問題）\n"
        else:
            status_text += "❌ **輸入過濾**：已停用\n"
        
        if enabled.get("output_rails", False):
            status_text += "✅ **輸出過濾**：已啟用（過濾回應內容）\n"
        else:
            status_text += "❌ **輸出過濾**：已停用\n"
        
        return status_text
    
    except Exception as e:
        return f"⚠️ 無法獲取 Guardrails 狀態：{str(e)}"


def create_simple_chatbot_interface():
    """
    創建簡單聊天機器人界面
    純粹的對話式聊天，不包含 RAG 或其他複雜功能
    """
    with gr.Blocks(title="Simple Chatbot") as demo:
        # 標題
        gr.Markdown(
            """
            <div style="text-align: center; padding: 20px;">
                <h1>💬 Simple Chatbot</h1>
                <p style="font-size: 16px; color: #666;">
                    簡單的對話式聊天機器人 - 純粹的 AI 對話體驗
                </p>
                <p style="font-size: 14px; color: #888;">
                    不包含 RAG、Deep AI Agent 等複雜功能，只專注於自然對話
                </p>
            </div>
            """
        )
        
        # LLM 狀態顯示
        llm_status = gr.Markdown(
            value=get_llm_status(),
            elem_classes=["warning-box"]
        )
        
        # Guardrails 啟用開關
        with gr.Row():
            enable_guardrails_checkbox = gr.Checkbox(
                label="🛡️ 啟用 Guardrails 內容過濾",
                value=True,
                info="啟用後將檢查輸入和輸出內容，阻擋敏感話題"
            )
        
        # 系統提示詞設定（可選）
        with gr.Accordion("⚙️ 進階設定", open=False):
            system_prompt = gr.Textbox(
                label="系統提示詞 (System Prompt)",
                value="你是一個有幫助的AI助手。請用繁體中文回答問題。",
                lines=3,
                placeholder="設定 AI 的角色和行為方式..."
            )
            
            gr.Markdown(
                """
                **提示詞範例：**
                - 專業助手：「你是一位專業的技術顧問，擅長解釋複雜的技術概念。」
                - 創意寫作：「你是一位富有創意的作家，擅長寫作故事和詩歌。」
                - 學習輔導：「你是一位耐心的老師，擅長用簡單的方式解釋複雜的概念。」
                """
            )
        
        # Guardrails 設定顯示
        with gr.Accordion("🛡️ 內容過濾 Guardrails（混合策略）", open=False):
            guardrails_status_md = gr.Markdown(
                value=get_guardrails_status()
            )
            
            with gr.Row():
                refresh_guardrails_btn = gr.Button("🔄 更新 Guardrails 狀態", variant="secondary", size="sm")
            
            gr.Markdown(
                """
                ---
                
                ## 混合策略說明
                
                本系統採用**雙層過濾**策略，受 NeMo Guardrails 啟發：
                
                ### 第一層：關鍵字密度檢查（快速層）
                - ⚡ 速度：< 1ms
                - 🔍 使用 `jieba` 進行中英文斷詞
                - 📊 計算敏感詞密度（敏感詞數 / 總詞數）
                - 🎯 適用於：明確的關鍵字匹配
                
                ### 第二層：語義主題過濾（深度層）
                - 🤖 使用 Sentence Transformers 語義理解
                - 🎭 可偵測改寫、隱喻等複雜表達
                - 📝 基於主題範例進行相似度匹配
                - 🎯 適用於：主題層級的內容控制
                
                ### 雙向防護
                - 🔒 **輸入過濾**：阻擋敏感問題
                - 🛡️ **輸出過濾**：確保回應安全
                
                ℹ️ 配置文件位於：`deep_agent_rag/guardrails/config/`
                """
            )
        
        # 聊天界面（Gradio 5.x+ 默認使用字典格式）
        chatbot = gr.Chatbot(
            label="對話記錄",
            height=500,
            show_label=True
        )
        
        # 輸入區域
        msg = gr.Textbox(
            label="訊息",
            placeholder="在這裡輸入您的訊息...",
            lines=2,
            show_label=False
        )
        
        # 控制按鈕
        with gr.Row():
            submit_btn = gr.Button("📤 發送", variant="primary")
            clear_btn = gr.Button("🗑️ 清除對話", variant="secondary")
            refresh_status_btn = gr.Button("🔄 更新狀態", variant="secondary")
        
        # 示例問題
        gr.Examples(
            examples=[
                "你好！請介紹一下你自己。",
                "請幫我解釋什麼是機器學習？",
                "能給我一些學習 Python 的建議嗎？",
                "請用簡單的方式解釋量子計算。",
                "寫一首關於春天的短詩。"
            ],
            inputs=msg,
            label="💡 快速試用範例"
        )
        
        # 事件綁定
        def clear_chat():
            """清除對話"""
            return [], ""
        
        def refresh_status():
            """更新 LLM 狀態"""
            return get_llm_status()
        
        def refresh_guardrails_status():
            """更新 Guardrails 狀態"""
            return get_guardrails_status()
        
        # 發送消息事件
        msg.submit(
            fn=chat_with_llm_streaming,
            inputs=[msg, chatbot, system_prompt, enable_guardrails_checkbox],
            outputs=[chatbot],
            queue=True
        ).then(
            fn=lambda: "",
            outputs=[msg],
            queue=False
        )
        
        submit_btn.click(
            fn=chat_with_llm_streaming,
            inputs=[msg, chatbot, system_prompt, enable_guardrails_checkbox],
            outputs=[chatbot],
            queue=True
        ).then(
            fn=lambda: "",
            outputs=[msg],
            queue=False
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, msg],
            queue=False
        )
        
        refresh_status_btn.click(
            fn=refresh_status,
            outputs=[llm_status],
            queue=False
        )
        
        refresh_guardrails_btn.click(
            fn=refresh_guardrails_status,
            outputs=[guardrails_status_md],
            queue=False
        )
        
        # 頁腳
        gr.Markdown(
            """
            ---
            ### 📝 使用說明
            
            1. **開始對話**：在輸入框中輸入訊息，點擊「發送」或按 Enter
            2. **系統提示詞**：展開「進階設定」可自訂 AI 的角色和行為
            3. **清除對話**：點擊「清除對話」可重新開始
            4. **快速試用**：點擊下方的範例問題快速測試
            
            **特色功能：**
            - 🎯 純粹的對話體驗，無複雜功能
            - 💫 流式輸出，逐字顯示回應
            - 🔧 可自訂系統提示詞
            - 📝 保留完整對話歷史
            - 🚀 支持本地模型和雲端 API
            - 🛡️ 混合式 Guardrails 內容過濾（關鍵字 + 語義雙層防護）
            - 🔒 雙向過濾（輸入阻擋 + 輸出過濾）
            """
        )
    
    return demo


# 如果直接執行此文件，啟動界面
if __name__ == "__main__":
    demo = create_simple_chatbot_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )
