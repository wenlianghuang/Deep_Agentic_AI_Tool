# Parlant SDK 指南系統

## 概述

本模組使用官方 **Parlant SDK** 實現指南驅動（Guideline-Driven）架構，使用自然語言定義代理行為規則，取代硬編碼的邏輯，提高系統的可控性、一致性和可維護性。

## 功能特點

1. **自然語言指南**：使用自然語言定義行為規則，易於理解和修改
2. **集中管理**：所有指南集中在一個模組中，便於維護
3. **類型安全**：提供清晰的函數接口，避免錯誤
4. **客戶旅程定義**：明確定義每個代理的交互流程

## 使用方式

### 基本使用

```python
from deep_agent_rag.guidelines import get_guideline, get_customer_journey

# 獲取研究代理的工具選擇指南
tool_guideline = get_guideline("research", "tool_selection")

# 獲取研究代理的任務規劃指南
task_guideline = get_guideline("research", "task_planning")

# 獲取客戶旅程定義
journey = get_customer_journey("research")
print(journey["steps"])  # 查看步驟
print(journey["checkpoints"])  # 查看檢查點
```

### 在代理中使用

```python
from ..guidelines import get_guideline

def my_agent_node(state, llm=None):
    # 獲取指南
    tool_guideline = get_guideline("research", "tool_selection")
    behavior_guideline = get_guideline("research", "research_behavior")
    
    # 在系統提示中使用指南
    system_msg = SystemMessage(content=(
        f"你是一位研究員。當前任務：{current_task}\n\n"
        f"【工具選擇指南】\n{tool_guideline}\n\n"
        f"【行為指南】\n{behavior_guideline}"
    ))
    
    # ... 其餘代碼
```

## 可用的指南類型

### 研究代理 (research)

- `tool_selection`: 工具選擇指南
- `task_planning`: 任務規劃指南
- `research_behavior`: 研究行為指南

### 郵件代理 (email)

- `email_writing`: 郵件撰寫指南
- `reflection_criteria`: 反思評估標準

### 行事曆代理 (calendar)

- `event_creation`: 事件創建指南
- `reflection_criteria`: 反思評估標準

## 修改指南

指南定義在 `parlant_manager.py` 文件的 `_load_guidelines_to_cache()` 函數中。您可以：

1. 直接編輯指南文本（使用自然語言）
2. 添加新的指南類型
3. 為新代理添加指南

例如，修改工具選擇指南：

```python
_guideline_cache["research"]["tool_selection"] = """
您的新指南內容...
"""
```

或者，如果您想使用 Parlant SDK 的完整功能，可以在 `_initialize_parlant()` 函數中使用 `create_guideline()` API。

## 優勢

1. **易於維護**：指南集中在一個文件中，修改方便
2. **清晰明確**：使用自然語言，非技術人員也能理解
3. **一致性**：統一的指南確保代理行為一致
4. **可擴展**：容易添加新的指南類型或代理類型

## 測試

運行測試腳本驗證系統：

```bash
python3 test_parlant_integration.py
```

## 整合狀態

✅ **已完成**：
- Parlant SDK 整合
- 指南管理系統（使用 Parlant SDK）
- 研究代理指南定義
- 郵件代理指南定義
- 行事曆代理指南定義
- 客戶旅程定義
- `researcher.py` 整合
- `planner.py` 整合
- 應用啟動時自動初始化

🔄 **待完成**（可選）：
- 運行完整的 Parlant Server
- 使用 Parlant 的動態指南匹配
- 整合 Parlant 的工具系統
- 使用 Parlant 的客戶旅程引擎

## 相關文件

- `parlant_manager.py`: Parlant SDK 管理器（核心文件）
- `__init__.py`: 模組導出
- `../agents/researcher.py`: 研究代理（已整合）
- `../agents/planner.py`: 規劃代理（已整合）
- `PARLANT_SDK_INTEGRATION.md`: 整合文檔

