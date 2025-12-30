"""
筆記節點
將研究結果轉化為筆記，存入 research_notes 緩存
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .state import DeepAgentState
from ..utils.llm_utils import get_llm


def note_taking_node(state: DeepAgentState, llm=None):
    """紀錄節點：將研究結果轉化為筆記，存入 research_notes 緩存"""
    if llm is None:
        llm = get_llm()
    
    try:
        last_msg = state["messages"][-1]
        completed_count = len(state.get("completed_tasks", []))
        tasks = state.get("tasks", [])
        
        if completed_count >= len(tasks):
            return {}
        
        current_task = tasks[completed_count]
        
        # 使用 LLM 摘要研究結果，提取關鍵資訊
        try:
            summary_prompt = ChatPromptTemplate.from_template(
                "請將以下研究結果摘要為3-5個關鍵要點：\n\n{content}\n\n"
                "請以簡潔的條列式呈現。"
            )
            chain = summary_prompt | llm | StrOutputParser()
            summary = chain.invoke({"content": last_msg.content})
        except:
            # 如果摘要失敗，直接使用原始內容
            summary = last_msg.content[:500] + "..." if len(last_msg.content) > 500 else last_msg.content
        
        note = f"【任務 {completed_count + 1}: {current_task}】\n{summary}\n"
        print(f"   📌 [NoteTaker] 已紀錄任務 {completed_count + 1} 的研究筆記。")
        
        # 注意：由於使用了 operator.add，這裡返回的列表會被追加到現有列表
        return {
            "research_notes": [note], 
            "completed_tasks": [current_task]
        }
    except Exception as e:
        print(f"   ⚠️ [NoteTaker] 記錄失敗: {e}")
        return {}

