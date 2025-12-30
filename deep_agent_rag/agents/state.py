"""
Deep Agent 狀態定義
"""
from typing import List, TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage


class DeepAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    tasks: List[str]            # 待執行的子任務清單
    completed_tasks: Annotated[List[str], operator.add]  # 已完成的任務（使用 operator.add 追加）
    research_notes: Annotated[List[str], operator.add]   # 儲存每一輪搜尋到的深度內容（使用 operator.add 追加）
    iteration: int              # 追蹤迭代次數，防止無限循環
    query: str                  # 原始問題

