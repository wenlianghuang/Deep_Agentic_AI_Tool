"""
測試 Parlant SDK 整合
驗證指南系統是否正常工作
"""

from deep_agent_rag.guidelines import (
    get_guideline,
    get_customer_journey,
    initialize_parlant_sync
)


def test_guidelines():
    """測試指南獲取功能"""
    print("=" * 60)
    print("測試指南系統")
    print("=" * 60)
    
    # 測試研究代理指南
    print("\n1. 測試研究代理的工具選擇指南...")
    tool_guideline = get_guideline("research", "tool_selection")
    assert tool_guideline, "❌ 工具選擇指南不應為空"
    assert "query_pdf_knowledge" in tool_guideline, "❌ 應包含 PDF 工具說明"
    assert "get_company_deep_info" in tool_guideline, "❌ 應包含股票工具說明"
    assert "search_web" in tool_guideline, "❌ 應包含網路搜尋工具說明"
    print("   ✅ 工具選擇指南獲取成功")
    print(f"   📄 指南長度: {len(tool_guideline)} 字符")
    
    print("\n2. 測試研究代理的任務規劃指南...")
    task_guideline = get_guideline("research", "task_planning")
    assert task_guideline, "❌ 任務規劃指南不應為空"
    assert "學術理論問題" in task_guideline, "❌ 應包含學術問題說明"
    assert "股票相關問題" in task_guideline, "❌ 應包含股票問題說明"
    print("   ✅ 任務規劃指南獲取成功")
    
    print("\n3. 測試研究代理的研究行為指南...")
    behavior_guideline = get_guideline("research", "research_behavior")
    assert behavior_guideline, "❌ 研究行為指南不應為空"
    print("   ✅ 研究行為指南獲取成功")
    
    # 測試郵件代理指南
    print("\n4. 測試郵件代理的撰寫指南...")
    email_guideline = get_guideline("email", "email_writing")
    assert email_guideline, "❌ 郵件撰寫指南不應為空"
    print("   ✅ 郵件撰寫指南獲取成功")
    
    # 測試行事曆代理指南
    print("\n5. 測試行事曆代理的創建指南...")
    calendar_guideline = get_guideline("calendar", "event_creation")
    assert calendar_guideline, "❌ 事件創建指南不應為空"
    print("   ✅ 事件創建指南獲取成功")
    
    # 測試不存在的指南
    print("\n6. 測試錯誤處理（不存在的指南）...")
    invalid_guideline = get_guideline("research", "nonexistent")
    assert invalid_guideline == "", "❌ 不存在的指南應返回空字符串"
    print("   ✅ 錯誤處理正常")


def test_customer_journey():
    """測試客戶旅程獲取功能"""
    print("\n" + "=" * 60)
    print("測試客戶旅程系統")
    print("=" * 60)
    
    print("\n1. 測試研究代理的客戶旅程...")
    research_journey = get_customer_journey("research")
    assert research_journey, "❌ 研究代理客戶旅程不應為空"
    assert "steps" in research_journey, "❌ 應包含步驟定義"
    assert "checkpoints" in research_journey, "❌ 應包含檢查點"
    print("   ✅ 研究代理客戶旅程獲取成功")
    print(f"   📋 步驟: {research_journey['steps'][0]}")
    print(f"   🔍 檢查點數量: {len(research_journey['checkpoints'])}")
    
    print("\n2. 測試郵件代理的客戶旅程...")
    email_journey = get_customer_journey("email")
    assert email_journey, "❌ 郵件代理客戶旅程不應為空"
    print("   ✅ 郵件代理客戶旅程獲取成功")
    
    print("\n3. 測試行事曆代理的客戶旅程...")
    calendar_journey = get_customer_journey("calendar")
    assert calendar_journey, "❌ 行事曆代理客戶旅程不應為空"
    print("   ✅ 行事曆代理客戶旅程獲取成功")
    
    # 測試不存在的客戶旅程
    print("\n4. 測試錯誤處理（不存在的客戶旅程）...")
    invalid_journey = get_customer_journey("nonexistent")
    assert invalid_journey == {}, "❌ 不存在的客戶旅程應返回空字典"
    print("   ✅ 錯誤處理正常")


def test_guideline_structure():
    """測試指南結構完整性"""
    print("\n" + "=" * 60)
    print("測試指南結構完整性")
    print("=" * 60)
    
    print("\n1. 檢查研究代理指南...")
    tool_guideline = get_guideline("research", "tool_selection")
    task_guideline = get_guideline("research", "task_planning")
    behavior_guideline = get_guideline("research", "research_behavior")
    assert tool_guideline, "❌ 缺少工具選擇指南"
    assert task_guideline, "❌ 缺少任務規劃指南"
    assert behavior_guideline, "❌ 缺少研究行為指南"
    print("   ✅ 研究代理指南結構完整")
    
    print("\n2. 檢查郵件代理指南...")
    email_guideline = get_guideline("email", "email_writing")
    assert email_guideline, "❌ 缺少郵件撰寫指南"
    print("   ✅ 郵件代理指南結構完整")
    
    print("\n3. 檢查行事曆代理指南...")
    calendar_guideline = get_guideline("calendar", "event_creation")
    assert calendar_guideline, "❌ 缺少事件創建指南"
    print("   ✅ 行事曆代理指南結構完整")


def main():
    """運行所有測試"""
    print("\n" + "🚀 " * 20)
    print("開始測試 Parlant SDK 整合")
    print("🚀 " * 20 + "\n")
    
    try:
        # 初始化 Parlant SDK
        print("初始化 Parlant SDK...")
        initialize_parlant_sync()
        print()
        
        test_guidelines()
        test_customer_journey()
        
        print("\n" + "=" * 60)
        print("✅ 所有測試通過！")
        print("=" * 60)
        print("\nParlant SDK 指南系統已成功整合，可以開始使用了！")
        
    except AssertionError as e:
        print(f"\n❌ 測試失敗: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

