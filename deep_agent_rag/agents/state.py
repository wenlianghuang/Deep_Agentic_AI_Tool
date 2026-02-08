"""
Deep Agent 狀態定義
"""
from typing import List, TypedDict, Annotated, Optional

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

import operator
from langchain_core.messages import BaseMessage


class DeepAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    tasks: List[str]            # 待執行的子任務清單
    completed_tasks: Annotated[List[str], operator.add]  # 已完成的任務（使用 operator.add 追加）
    research_notes: Annotated[List[str], operator.add]   # 儲存每一輪搜尋到的深度內容（使用 operator.add 追加）
    iteration: int              # 追蹤迭代次數，防止無限循環
    query: str                  # 原始問題
    # 圖級重試（做法 B）：planner / research_agent 失敗時由條件邊決定是否重試
    planner_retry_count: NotRequired[int]
    planner_succeeded: NotRequired[bool]
    planner_error: NotRequired[Optional[str]]
    research_agent_retry_count: NotRequired[int]
    research_agent_succeeded: NotRequired[bool]
    research_agent_error: NotRequired[Optional[str]]

