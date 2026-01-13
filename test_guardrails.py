"""
測試 Guardrails 內容過濾功能
Test script for content guardrails
"""
import jieba
from deep_agent_rag.ui.simple_chatbot_interface import (
    check_content_guardrails,
    guardrail_filter,
    BLOCKED_KEYWORDS,
    KEYWORD_DENSITY_THRESHOLD,
    _init_jieba_custom_dict
)

# 確保 jieba 自定義詞典已初始化
_init_jieba_custom_dict()


def test_guardrails():
    """測試 Guardrails 功能"""
    
    print("=" * 80)
    print("🛡️ Guardrails 內容過濾測試")
    print("=" * 80)
    print()
    
    print(f"📋 敏感關鍵字列表：{BLOCKED_KEYWORDS}")
    print(f"🎯 攔截門檻：{KEYWORD_DENSITY_THRESHOLD:.1%} (關鍵字密度)")
    print()
    print("=" * 80)
    print()
    
    # 測試案例
    test_cases = [
        {
            "name": "正常內容 - 不應該被攔截",
            "text": "今天天氣很好，我們一起去公園散步吧。這是一個美好的日子。"
        },
        {
            "name": "包含少量敏感詞 - 低於門檻",
            "text": "伊斯蘭教是世界主要宗教之一，有著悠久的歷史和豐富的文化傳統。許多信徒在世界各地實踐他們的信仰，並為社會做出貢獻。"
        },
        {
            "name": "包含多個敏感詞 - 超過門檻",
            "text": "伊斯蘭教的先知默罕默德教導信徒向阿拉禱告。"
        },
        {
            "name": "高密度敏感詞 - 明顯超過門檻",
            "text": "阿拉默罕默德伊斯蘭教"
        },
        {
            "name": "技術討論 - 正常內容",
            "text": "機器學習是人工智能的一個分支，它使用統計技術讓計算機系統能夠從數據中學習。深度學習是機器學習的一個子集。"
        }
    ]
    
    # 執行測試
    for i, test_case in enumerate(test_cases, 1):
        print(f"測試案例 {i}: {test_case['name']}")
        print("-" * 80)
        
        text = test_case['text']
        print(f"📝 原文本：{text}")
        print()
        
        # 使用 jieba 分詞
        words = list(jieba.cut(text))
        print(f"🔤 分詞結果：{' / '.join(words)}")
        print(f"📊 總詞數：{len(words)}")
        print()
        
        # 檢查敏感詞
        sensitive_words_found = [w for w in words if w in BLOCKED_KEYWORDS]
        print(f"⚠️  發現敏感詞：{sensitive_words_found if sensitive_words_found else '無'}")
        print(f"🔢 敏感詞數量：{len(sensitive_words_found)}")
        print()
        
        # 執行 Guardrails 檢查
        should_block, density = check_content_guardrails(text)
        print(f"📈 關鍵字密度：{density:.2%} (門檻：{KEYWORD_DENSITY_THRESHOLD:.2%})")
        print(f"🚦 判定結果：{'🚫 攔截' if should_block else '✅ 通過'}")
        print()
        
        # 應用過濾器
        filtered = guardrail_filter(text)
        if filtered != text:
            print(f"🛡️ 過濾後輸出：{filtered}")
        else:
            print(f"✅ 原文通過，無需過濾")
        
        print()
        print("=" * 80)
        print()


def test_edge_cases():
    """測試邊界情況"""
    
    print("🔬 邊界測試")
    print("=" * 80)
    print()
    
    edge_cases = [
        ("空字符串", ""),
        ("純空格", "   "),
        ("單個敏感詞", "伊斯蘭教"),
        ("重複敏感詞", "阿拉阿拉阿拉"),
        ("長文本混合", "今天我們要討論世界宗教的歷史。" * 10 + "伊斯蘭教是其中之一。"),
    ]
    
    for name, text in edge_cases:
        should_block, density = check_content_guardrails(text)
        print(f"{name}：")
        print(f"  文本長度：{len(text)}")
        print(f"  關鍵字密度：{density:.2%}")
        print(f"  結果：{'🚫 攔截' if should_block else '✅ 通過'}")
        print()


if __name__ == "__main__":
    try:
        # 執行主要測試
        test_guardrails()
        
        # 執行邊界測試
        test_edge_cases()
        
        print("✅ 所有測試完成！")
        
    except Exception as e:
        print(f"❌ 測試失敗：{e}")
        import traceback
        traceback.print_exc()
