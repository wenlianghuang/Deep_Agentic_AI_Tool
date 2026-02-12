"""
Image Reflection Agent
圖片分析反思代理：評估分析結果質量並提供改進建議
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..utils.llm_utils import get_llm, handle_groq_error


def detect_language(text: str) -> str:
    """
    檢測文本的主要語言（中文或英文）
    
    Args:
        text: 輸入文本
    
    Returns:
        'zh' 或 'en'
    """
    import re
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    if chinese_pattern.search(text):
        return 'zh'
    else:
        return 'en'


def reflect_on_image_analysis(
    question: str,
    image_path: str,
    analysis_result: str
) -> tuple[str, str, bool]:
    """
    反思圖片分析結果質量，評估是否需要改進
    
    Args:
        question: 用戶的問題（如果有的話）
        image_path: 圖片路徑
        analysis_result: 初始分析結果
    
    Returns:
        (reflection_result, improvement_suggestions, needs_revision)
        - reflection_result: 反思結果（評估分析質量）
        - improvement_suggestions: 改進建議（如果需要改進）
        - needs_revision: 是否需要重新生成（True/False）
    """
    try:
        # 檢測語言
        user_language = detect_language(question if question else analysis_result)
        
        # 獲取 LLM
        llm = get_llm()
        
        if user_language == 'zh':
            # 中文反思提示模板
            reflection_prompt_template = (
                "你是一位專業的圖片分析質量評估專家。請仔細評估以下圖片分析結果，並提供詳細的反思和改進建議。\n\n"
                "【用戶問題】\n{question}\n\n"
                "【圖片路徑】\n{image_path}\n\n"
                "【分析結果】\n{analysis_result}\n\n"
                "請從以下幾個方面進行評估：\n"
                "1. **完整性**：分析是否完整回答了用戶的問題？是否遺漏重要信息？\n"
                "2. **準確性**：分析是否準確描述了圖片內容？是否有錯誤或誤解？\n"
                "3. **詳細度**：分析是否足夠詳細？是否需要補充更多細節？\n"
                "4. **結構性**：分析是否結構清晰、邏輯合理？\n"
                "5. **相關性**：分析是否緊扣用戶的問題？如果沒有特定問題，是否全面分析了圖片？\n\n"
                "請按照以下格式輸出：\n"
                "【反思評估】\n"
                "(詳細評估分析結果在各個方面的表現，指出優點和不足)\n\n"
                "【改進建議】\n"
                "(如果有需要改進的地方，請提供具體的改進建議；如果分析質量很好，請說明為什麼)\n\n"
                "【是否需要重新生成】\n"
                "(回答：是/否，並簡要說明原因。只有在分析有嚴重問題（如遺漏關鍵信息、明顯錯誤、結構混亂、未回答用戶問題）時才回答「是」)"
            )
        else:
            # 英文反思提示模板
            reflection_prompt_template = (
                "You are a professional image analysis quality assessment expert. "
                "Please carefully evaluate the following image analysis result and provide detailed reflection and improvement suggestions.\n\n"
                "【User Question】\n{question}\n\n"
                "【Image Path】\n{image_path}\n\n"
                "【Analysis Result】\n{analysis_result}\n\n"
                "Please evaluate from the following aspects:\n"
                "1. **Completeness**: Does the analysis fully answer the user's question? Are there any missing important information?\n"
                "2. **Accuracy**: Does the analysis accurately describe the image content? Are there any errors or misunderstandings?\n"
                "3. **Detail**: Is the analysis detailed enough? Does it need more details?\n"
                "4. **Structure**: Is the analysis well-structured and logically organized?\n"
                "5. **Relevance**: Does the analysis address the user's question? If no specific question, is it comprehensive?\n\n"
                "Please output in the following format:\n"
                "【Reflection Assessment】\n"
                "(Detailed assessment of the analysis result's performance in various aspects, pointing out strengths and weaknesses)\n\n"
                "【Improvement Suggestions】\n"
                "(If there are areas that need improvement, provide specific suggestions; if the analysis quality is good, explain why)\n\n"
                "【Needs Revision】\n"
                "(Answer: Yes/No, and briefly explain the reason. Only answer 'Yes' if the analysis has serious issues such as missing key information, obvious errors, poor structure, or not answering the user's question)"
            )
        
        # 創建反思提示
        reflection_prompt = ChatPromptTemplate.from_template(reflection_prompt_template)
        
        try:
            chain = reflection_prompt | llm | StrOutputParser()
            reflection_result = chain.invoke({
                "question": question or "通用圖片分析",
                "image_path": image_path,
                "analysis_result": analysis_result
            })
        except Exception as e:
            # 處理 Groq API 錯誤
            fallback_llm = handle_groq_error(e)
            if fallback_llm:
                print("   ⚠️ [ImageReflection] Groq API 額度已用完，已切換到本地 MLX 模型")
                chain = reflection_prompt | fallback_llm | StrOutputParser()
                reflection_result = chain.invoke({
                    "question": question or "通用圖片分析",
                    "image_path": image_path,
                    "analysis_result": analysis_result
                })
            else:
                raise
        
        # 解析反思結果
        reflection_text = reflection_result
        improvement_suggestions = ""
        needs_revision = False
        
        # 提取改進建議部分
        if "【改進建議】" in reflection_text or "【Improvement Suggestions】" in reflection_text:
            parts = reflection_text.split("【改進建議】") if "【改進建議】" in reflection_text else reflection_text.split("【Improvement Suggestions】")
            if len(parts) > 1:
                improvement_part = parts[1].split("【是否需要重新生成】")[0] if "【是否需要重新生成】" in parts[1] else parts[1].split("【Needs Revision】")[0]
                improvement_suggestions = improvement_part.strip()
        
        # 檢查是否需要重新生成
        if "【是否需要重新生成】" in reflection_text or "【Needs Revision】" in reflection_text:
            revision_part = reflection_text.split("【是否需要重新生成】")[-1] if "【是否需要重新生成】" in reflection_text else reflection_text.split("【Needs Revision】")[-1]
            revision_text = revision_part.strip().lower()
            # 檢查是否包含「是」、「yes」等關鍵字
            needs_revision = any(keyword in revision_text for keyword in ["是", "yes", "需要", "need", "應該", "should"])
        
        return reflection_text, improvement_suggestions, needs_revision
        
    except Exception as e:
        error_msg = f"反思過程中發生錯誤：{str(e)}"
        print(f"Image Reflection Agent 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return error_msg, "", False


def generate_improved_analysis(
    question: str,
    image_path: str,
    original_analysis: str,
    improvement_suggestions: str
) -> str:
    """
    根據改進建議生成改進後的圖片分析
    重要：重新調用多模態 LLM 分析圖片，而不是只基於文本生成
    
    Args:
        question: 用戶的問題（如果有的話）
        image_path: 圖片路徑
        original_analysis: 原始分析結果
        improvement_suggestions: 改進建議
    
    Returns:
        改進後的分析結果
    """
    try:
        # 導入圖片分析函數（使用多模態 LLM）
        from ..tools.image_analysis_mcp_client import analyze_image_result

        # 檢測語言
        user_language = detect_language(question if question else original_analysis)
        
        # 根據改進建議構建改進後的問題/提示
        # 將改進建議整合到問題中，引導 LLM 關注需要改進的方面
        if user_language == 'zh':
            if improvement_suggestions:
                improved_question = (
                    f"{question or '請詳細分析這張圖片'}\n\n"
                    f"【重要改進建議】\n{improvement_suggestions}\n\n"
                    f"請特別注意改進建議中提到的問題，確保分析準確、完整、詳細。"
                    f"如果原始分析有錯誤，請重新仔細觀察圖片並提供正確的分析。"
                )
            else:
                # 如果沒有改進建議，使用原始問題
                improved_question = question or "請詳細分析這張圖片，確保分析準確、完整、詳細。"
        else:
            if improvement_suggestions:
                improved_question = (
                    f"{question or 'Please analyze this image in detail'}\n\n"
                    f"【Important Improvement Suggestions】\n{improvement_suggestions}\n\n"
                    f"Please pay special attention to the issues mentioned in the improvement suggestions to ensure the analysis is accurate, complete, and detailed."
                    f"If the original analysis had errors, please carefully re-examine the image and provide a correct analysis."
                )
            else:
                improved_question = question or "Please analyze this image in detail, ensuring the analysis is accurate, complete, and detailed."
        
        # 重新調用圖片分析（使用多模態 LLM，真正查看圖片）
        print(f"   🔄 [ImageImprovement] 重新分析圖片（結合改進建議）...")
        improved_analysis = analyze_image_result(image_path, question=improved_question)
        
        # 檢查結果是否為錯誤訊息
        if improved_analysis.startswith("❌"):
            print(f"   ⚠️ [ImageImprovement] 重新分析失敗，返回原始分析")
            return original_analysis
        
        return improved_analysis.strip()
        
    except Exception as e:
        error_msg = f"生成改進分析時發生錯誤：{str(e)}"
        print(f"Image Reflection Agent 錯誤：{e}")
        import traceback
        traceback.print_exc()
        return original_analysis  # 返回原始分析作為備援
