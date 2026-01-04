# deep_agent_rag/ui/calendar_interface.py

import gradio as gr
from datetime import datetime, timedelta
import re
import json
import time

from ..agents.calendar_agent import generate_calendar_draft, create_calendar_draft
# Assuming is_using_local_llm might be used for warnings/status, similar to email_interface
# from ..utils.llm_utils import is_using_local_llm 

# Agent log path for debugging (if needed)
log_path = "/Users/matthuang/Desktop/Deep_Agentic_AI_Tool/.cursor/debug.log"

def _create_calendar_interface():
    """創建 Calendar Tool 界面"""
    gr.Markdown(
        """
        ### 📅 智能行事曆管理助手
        
        使用 AI 根據您的完整提示自動生成行事曆事件草稿，您可以在創建前檢查和修改。
        
        **使用方式：**
        1. **快速選擇**：點擊下方常見事件按鈕，自動生成草稿
        2. **自定義輸入**：在下方輸入完整的事件提示，包含：事件、日期、時間、地點、參與者
        3. 查看 AI 反思評估結果和改進建議（如有）
        4. 如果有缺失的資訊（如時間），系統會顯示下拉選單讓您選擇
        5. 檢查並修改生成的事件內容
        6. 確認無誤後點擊「創建事件」按鈕
        
        **✨ 新功能：AI 迭代反思評估 + Google Maps 地點驗證**
        - 系統會自動進行多輪反思評估（最多 3 輪）
        - 自動驗證並標準化地址，計算交通時間
        - 每輪評估後，如果有改進建議，會自動生成改進版本
        - 改進後的版本會再次評估，直到 AI 認為滿意為止
        """
    )
    
    # 快速選擇按鈕區域
    gr.Markdown("### 🚀 快速選擇常見事件")
    with gr.Row():
        quick_meeting_btn = gr.Button("📋 團隊會議", variant="secondary", scale=1)
        quick_client_btn = gr.Button("🤝 客戶拜訪", variant="secondary", scale=1)
        quick_lunch_btn = gr.Button("🍽️ 午餐會議", variant="secondary", scale=1)
        quick_oneonone_btn = gr.Button("💬 一對一會議", variant="secondary", scale=1)
    with gr.Row():
        quick_project_btn = gr.Button("📊 項目討論", variant="secondary", scale=1)
        quick_training_btn = gr.Button("🎓 培訓/學習", variant="secondary", scale=1)
        quick_social_btn = gr.Button("🎉 社交活動", variant="secondary", scale=1)
        quick_custom_btn = gr.Button("✏️ 自定義輸入", variant="secondary", scale=1)
    
    with gr.Row():
        with gr.Column(scale=1):
            # 單一 prompt 輸入
            calendar_prompt_input = gr.Textbox(
                label="📝 事件提示（包含事件、日期、時間、地點、參與者）",
                placeholder="例如：明天下午2點團隊會議，討論項目進度，地點在會議室A，參與者包括john@example.com和mary@example.com",
                lines=5,
                value=""
            )
            
            # 按鈕
            with gr.Row():
                generate_draft_btn = gr.Button("📝 生成事件草稿", variant="primary", scale=1)
                clear_calendar_btn = gr.Button("🗑️ 清除", variant="secondary", scale=1)
            
            # 狀態顯示
            calendar_status_display = gr.Textbox(
                label="📊 狀態",
                value="等待操作...",
                interactive=False,
                lines=2
            )
            
            # 反思結果顯示
            calendar_reflection_display = gr.Textbox(
                label="🔍 AI 反思評估",
                value="等待生成事件...",
                interactive=False,
                lines=8,
                visible=True
            )
            
            # 缺失資訊的補充區域（動態顯示）
            missing_info_group = gr.Group(visible=False)
            with missing_info_group:
                gr.Markdown("**⚠️ 請補充以下缺失的資訊：**")
                
                # 日期選擇（如果缺失）
                missing_date_display = gr.Dropdown(
                    label="📆 選擇日期",
                    choices=[],
                    visible=False,
                    interactive=True
                )
                
                # 時間選擇（如果缺失）
                missing_time_display = gr.Dropdown(
                    label="🕐 選擇時間",
                    choices=[],
                    visible=False,
                    interactive=True
                )
                
                fill_missing_btn = gr.Button("✅ 確認補充資訊", variant="primary", visible=False)
            
            # 隱藏狀態變數，用於存儲 event_dict
            event_dict_storage = gr.State(value={})
        
        with gr.Column(scale=1):
            # 事件詳情顯示和編輯區域
            event_summary_display = gr.Textbox(
                label="📌 事件標題",
                placeholder="事件標題將在這裡顯示",
                lines=1,
                interactive=True
            )
            
            event_start_display = gr.Textbox(
                label="🕐 開始時間",
                placeholder="開始時間將在這裡顯示（格式: YYYY-MM-DDTHH:MM:SS+08:00）",
                lines=1,
                interactive=True
            )
            
            event_end_display = gr.Textbox(
                label="🕐 結束時間",
                placeholder="結束時間將在這裡顯示（格式: YYYY-MM-DDTHH:MM:SS+08:00）",
                lines=1,
                interactive=True
            )
            
            event_description_display = gr.Textbox(
                label="📄 事件描述（可編輯）",
                placeholder="事件描述將在這裡顯示，您可以編輯",
                lines=6,
                interactive=True
            )
            
            event_location_display = gr.Textbox(
                label="📍 地點（可編輯，已自動驗證並標準化）",
                placeholder="事件地點將在這裡顯示，您可以編輯",
                lines=2,
                interactive=True
            )
            
            event_attendees_display = gr.Textbox(
                label="👥 參與者郵箱（可編輯，多個用逗號分隔）",
                placeholder="參與者郵箱將在這裡顯示，您可以編輯",
                lines=1,
                interactive=True
            )
            
            # 創建按鈕
            create_event_btn = gr.Button("✅ 創建事件", variant="primary", scale=1)
            
            # 操作結果顯示
            calendar_result_display = gr.Textbox(
                label="📊 操作結果",
                lines=8,
                interactive=False
            )
    
    # 生成時間選項（每30分鐘一個選項）
    def generate_time_options():
        """生成時間選項列表"""
        times = []
        for hour in range(24):
            for minute in [0, 30]:
                time_str = f"{hour:02d}:{minute:02d}"
                times.append(time_str)
        return times
    
    # 生成日期選項（今天、明天、後天，以及未來7天）
    def generate_date_options():
        """生成日期選項列表"""
        dates = []
        today = datetime.now()
        date_names = ["今天", "明天", "後天"]
        
        for i in range(3):
            date_obj = today + timedelta(days=i)
            date_str = date_obj.strftime('%Y-%m-%d')
            dates.append(f"{date_names[i]} ({date_str})")
        
        for i in range(3, 7):
            date_obj = today + timedelta(days=i)
            date_str = date_obj.strftime('%Y-%m-%d')
            dates.append(date_str)
        
        return dates
    
    # 快速選擇事件模板生成函數
    def generate_quick_prompt(event_type: str) -> str:
        """根據事件類型生成預設提示"""
        from datetime import datetime, timedelta
        
        # 獲取明天的日期
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")
        
        templates = {
            "meeting": f"明天下午2點團隊會議，討論項目進度和下週計劃，地點在會議室，參與者包括團隊成員",
            "client": f"明天上午10點客戶拜訪，討論合作方案和需求，地點在客戶公司或會議室",
            "lunch": f"明天中午12點午餐會議，與合作夥伴討論業務合作，地點在附近的餐廳",
            "oneonone": f"明天下午3點一對一會議，討論工作進展和職業發展，地點在會議室或咖啡廳",
            "project": f"明天上午9點項目討論會議，審查項目進度和解決問題，地點在項目室，參與者包括項目團隊",
            "training": f"明天下午2點培訓課程，學習新技能和最佳實踐，地點在培訓室或線上",
            "social": f"明天晚上6點團隊聚餐，慶祝項目完成，地點在餐廳，參與者包括團隊成員",
            "custom": ""  # 自定義，返回空讓用戶輸入
        }
        
        return templates.get(event_type, "")
    
    # 快速選擇按鈕處理函數（自動生成草稿）
    def quick_select_and_generate(event_type: str):
        """快速選擇事件類型並自動生成草稿"""
        prompt = generate_quick_prompt(event_type)
        if not prompt:
            # 如果是自定義，只返回空提示，不自動生成
            return (
                prompt,  # calendar_prompt_input
                "請在下方輸入框中輸入事件提示，然後點擊「生成事件草稿」",  # calendar_status_display
                "等待輸入...",  # calendar_reflection_display
                gr.update(visible=False),  # missing_info_group
                gr.update(visible=False, choices=[]),  # missing_date_display
                gr.update(visible=False, choices=[]),  # missing_time_display
                gr.update(visible=False),  # fill_missing_btn
                "", "", "", "", "", "",  # event fields
                {},
                ""  # calendar_result_display
            )
        
        # 自動生成草稿（調用 generate_draft 並返回所有輸出）
        draft_result = generate_draft(prompt)
        # generate_draft 返回的格式是：(status, reflection_display, missing_info_group, ...)
        # 但我們需要返回 (prompt, status, reflection_display, ...)
        # draft_result 是一個元組，我們需要將 prompt 添加到開頭
        return (prompt,) + draft_result
    
    def quick_select_meeting():
        """快速選擇：團隊會議"""
        return quick_select_and_generate("meeting")
    
    def quick_select_client():
        """快速選擇：客戶拜訪"""
        return quick_select_and_generate("client")
    
    def quick_select_lunch():
        """快速選擇：午餐會議"""
        return quick_select_and_generate("lunch")
    
    def quick_select_oneonone():
        """快速選擇：一對一會議"""
        return quick_select_and_generate("oneonone")
    
    def quick_select_project():
        """快速選擇：項目討論"""
        return quick_select_and_generate("project")
    
    def quick_select_training():
        """快速選擇：培訓/學習"""
        return quick_select_and_generate("training")
    
    def quick_select_social():
        """快速選擇：社交活動"""
        return quick_select_and_generate("social")
    
    def quick_select_custom():
        """快速選擇：自定義輸入（只清空，不自動生成）"""
        return (
            "",  # calendar_prompt_input
            "請在下方輸入框中輸入事件提示，然後點擊「生成事件草稿」",  # calendar_status_display
            "等待輸入...",  # calendar_reflection_display
            gr.update(visible=False),  # missing_info_group
            gr.update(visible=False, choices=[]),  # missing_date_display
            gr.update(visible=False, choices=[]),  # missing_time_display
            gr.update(visible=False),  # fill_missing_btn
            "", "", "", "", "", "",  # event fields
            {},
            ""  # calendar_result_display
        )
    
    # 事件處理函數
    def generate_draft(prompt):
        """生成行事曆事件草稿（包含反思功能）"""
        if not prompt or not prompt.strip():
            return (
                "❌ 請輸入事件提示",
                "❌ 請輸入事件提示",
                gr.update(visible=False),
                gr.update(visible=False, choices=[]),
                gr.update(visible=False, choices=[]),
                gr.update(visible=False),
                "", "", "", "", "", "", "",
                "❌ 請輸入事件提示"
            )
        
        try:
            status_msg = "🔄 正在生成事件草稿..."
            
            # 生成事件草稿（包含反思功能）
            event_dict, status, missing_info, reflection_result, was_improved = generate_calendar_draft(
                prompt.strip(), enable_reflection=True
            )
            
            if not event_dict:
                return (
                    status,
                    gr.update(visible=False),
                    gr.update(visible=False, choices=[]),
                    gr.update(visible=False, choices=[]),
                    gr.update(visible=False),
                    "", "", "", "", "", "", "",
                    status
                )
            
            # 格式化反思結果顯示
            if reflection_result:
                # 計算反思輪數
                reflection_count = reflection_result.count("【第") if "【第" in reflection_result else 0
                
                if was_improved:
                    if reflection_count > 1:
                        reflection_display = (
                            f"🔍 **AI 迭代反思評估結果**（共 {reflection_count} 輪）\n\n"
                            f"{reflection_result}\n\n"
                            f"✨ **已自動應用改進建議，經過 {reflection_count} 輪優化，當前顯示的是最終優化版本**"
                        )
                    else:
                        reflection_display = (
                            f"🔍 **AI 反思評估結果**\n\n"
                            f"{reflection_result}\n\n"
                            f"✨ **已自動應用改進建議，當前顯示的是優化後的版本**"
                        )
                else:
                    reflection_display = (
                        f"🔍 **AI 反思評估結果**\n\n"
                        f"{reflection_result}\n\n"
                        f"✅ **事件質量良好，無需改進**"
                    )
            else:
                reflection_display = "⚠️ 反思功能未返回結果"
            
            # 【Google Maps 整合】添加地點建議訊息
            location_suggestion = event_dict.get("location_suggestion", "")
            if location_suggestion:
                # 將地點建議添加到狀態訊息中
                if status:
                    status = f"{status}\n\n🗺️ **地點資訊：**\n{location_suggestion}"
                else:
                    status = f"🗺️ **地點資訊：**\n{location_suggestion}"
            
            # 檢查是否有缺失資訊
            has_missing = bool(missing_info)
            
            if has_missing:
                # 顯示缺失資訊區域
                date_visible = missing_info.get("date", False)
                time_visible = missing_info.get("time", False)
                
                date_choices = generate_date_options() if date_visible else []
                time_choices = generate_time_options() if time_visible else []
                
                return (
                    status,
                    reflection_display,
                    gr.update(visible=True),  # 顯示缺失資訊區域
                    gr.update(visible=date_visible, choices=date_choices, value=date_choices[0] if date_choices else None),
                    gr.update(visible=time_visible, choices=time_choices, value=time_choices[0] if time_choices else None),
                    gr.update(visible=True),  # 顯示確認按鈕
                    event_dict.get("summary", ""),
                    event_dict.get("start_datetime", ""),
                    event_dict.get("end_datetime", ""),
                    event_dict.get("description", ""),
                    event_dict.get("location", ""),
                    event_dict.get("attendees", ""),
                    event_dict,  # 傳遞完整的事件字典以便後續使用
                    ""
                )
            else:
                # 沒有缺失資訊，直接顯示結果
                return (
                    status,
                    reflection_display,
                    gr.update(visible=False),
                    gr.update(visible=False, choices=[]),
                    gr.update(visible=False, choices=[]),
                    gr.update(visible=False),
                    event_dict.get("summary", ""),
                    event_dict.get("start_datetime", ""),
                    event_dict.get("end_datetime", ""),
                    event_dict.get("description", ""),
                    event_dict.get("location", ""),
                    event_dict.get("attendees", ""),
                    event_dict,
                    ""
                )
        except Exception as e:
            error_msg = f"❌ 發生錯誤：{str(e)}"
            print(f"Calendar Tool 錯誤：{e}")
            import traceback
            traceback.print_exc()
            return (
                "❌ 發生錯誤",
                f"❌ 發生錯誤：{str(e)}",
                gr.update(visible=False),
                gr.update(visible=False, choices=[]),
                gr.update(visible=False, choices=[]),
                gr.update(visible=False),
                "", "", "", "", "", "", {},
                error_msg
            )
    
    def fill_missing_info(event_dict_storage, selected_date, selected_time):
        """填充缺失的資訊"""
        if not event_dict_storage:
            return (
                "❌ 沒有事件資料",
                gr.update(visible=False),
                gr.update(visible=False, choices=[]),
                gr.update(visible=False, choices=[]),
                gr.update(visible=False),
                "", "", "", "", "", "",
                {}
            )
        
        # 更新日期和時間
        if selected_date:
            # 從選項中提取日期字串（例如："明天 (2026-01-25)" -> "2026-01-25"）
            if "(" in selected_date:
                date_str = selected_date.split("(")[1].split(")")[0]
            else:
                date_str = selected_date
        else:
            date_str = event_dict_storage.get("date", "今天")
        
        if selected_time:
            time_str = selected_time
        else:
            time_str = "09:00"  # 預設時間
        
        # 重新解析日期和時間
        from ..agents.calendar_agent import parse_datetime # Import here to avoid circular dependency or unnecessary global import
        start_datetime, end_datetime = parse_datetime(date_str, time_str)
        
        # 更新事件字典
        event_dict_storage["start_datetime"] = start_datetime
        event_dict_storage["end_datetime"] = end_datetime
        
        return (
            "✅ 資訊已補充，請檢查並創建事件",
            gr.update(visible=False),  # 隱藏缺失資訊區域
            gr.update(visible=False, choices=[]),
            gr.update(visible=False, choices=[]),
            gr.update(visible=False),
            event_dict_storage.get("summary", ""),
            start_datetime,
            end_datetime,
            event_dict_storage.get("description", ""),
            event_dict_storage.get("location", ""),
            event_dict_storage.get("attendees", ""),
            event_dict_storage
        )
    
    def create_event(summary, start_datetime, end_datetime, description, location, attendees):
        """創建行事曆事件"""
        if not summary or not summary.strip():
            return "❌ 請輸入事件標題", "❌ 請輸入事件標題"
        
        if not start_datetime or not start_datetime.strip():
            return "❌ 請輸入開始時間", "❌ 請輸入開始時間"
        
        if not end_datetime or not end_datetime.strip():
            return "❌ 請輸入結束時間", "❌ 請輸入結束時間"
        
        try:
            status_msg = "🔄 正在創建事件..."
            
            # 構建事件字典
            event_dict = {
                "summary": summary.strip(),
                "start_datetime": start_datetime.strip(),
                "end_datetime": end_datetime.strip(),
                "description": description.strip() if description else "",
                "location": location.strip() if location else "",
                "attendees": attendees.strip() if attendees else "",
                "timezone": "Asia/Taipei"
            }
            
            # 創建事件
            result = create_calendar_draft(event_dict)
            
            return "✅ 事件已創建", result
        except Exception as e:
            error_msg = f"❌ 創建事件時發生錯誤：{str(e)}"
            print(f"Calendar Tool 錯誤：{e}")
            import traceback
            traceback.print_exc()
            return "❌ 發生錯誤", error_msg
    
    def clear_calendar():
        """清除行事曆相關輸入和輸出"""
        return (
            "",  # prompt
            "等待操作...",  # status
            "等待生成事件...",  # reflection_display
            gr.update(visible=False),  # missing_info_group
            gr.update(visible=False, choices=[]),  # missing_date
            gr.update(visible=False, choices=[]),  # missing_time
            gr.update(visible=False),  # fill_missing_btn
            "", "", "", "", "", "",  # event fields
            {},
            ""  # result
        )
    
    # 綁定事件
    generate_draft_btn.click(
        fn=generate_draft,
        inputs=[calendar_prompt_input],
        outputs=[
            calendar_status_display,
            calendar_reflection_display,
            missing_info_group,
            missing_date_display,
            missing_time_display,
            fill_missing_btn,
            event_summary_display,
            event_start_display,
            event_end_display,
            event_description_display,
            event_location_display,
            event_attendees_display,
            event_dict_storage,
            calendar_result_display
        ]
    )
    
    # 綁定快速選擇按鈕（自動填充提示並生成草稿）
    quick_outputs = [
        calendar_prompt_input,  # 更新提示輸入框
        calendar_status_display,
        calendar_reflection_display,
        missing_info_group,
        missing_date_display,
        missing_time_display,
        fill_missing_btn,
        event_summary_display,
        event_start_display,
        event_end_display,
        event_description_display,
        event_location_display,
        event_attendees_display,
        event_dict_storage,
        calendar_result_display
    ]
    
    quick_meeting_btn.click(fn=quick_select_meeting, outputs=quick_outputs)
    quick_client_btn.click(fn=quick_select_client, outputs=quick_outputs)
    quick_lunch_btn.click(fn=quick_select_lunch, outputs=quick_outputs)
    quick_oneonone_btn.click(fn=quick_select_oneonone, outputs=quick_outputs)
    quick_project_btn.click(fn=quick_select_project, outputs=quick_outputs)
    quick_training_btn.click(fn=quick_select_training, outputs=quick_outputs)
    quick_social_btn.click(fn=quick_select_social, outputs=quick_outputs)
    quick_custom_btn.click(fn=quick_select_custom, outputs=quick_outputs)
    
    fill_missing_btn.click(
        fn=fill_missing_info,
        inputs=[event_dict_storage, missing_date_display, missing_time_display],
        outputs=[
            calendar_status_display,
            missing_info_group,
            missing_date_display,
            missing_time_display,
            fill_missing_btn,
            event_summary_display,
            event_start_display,
            event_end_display,
            event_description_display,
            event_location_display,
            event_attendees_display,
            event_dict_storage
        ]
    )
    
    create_event_btn.click(
        fn=create_event,
        inputs=[
            event_summary_display,
            event_start_display,
            event_end_display,
            event_description_display,
            event_location_display,
            event_attendees_display
        ],
        outputs=[calendar_status_display, calendar_result_display]
    )
    
    clear_calendar_btn.click(
        fn=clear_calendar,
        outputs=[
            calendar_prompt_input,
            calendar_status_display,
            calendar_reflection_display,
            missing_info_group,
            missing_date_display,
            missing_time_display,
            fill_missing_btn,
            event_summary_display,
            event_start_display,
            event_end_display,
            event_description_display,
            event_location_display,
            event_attendees_display,
            event_dict_storage,
            calendar_result_display
        ]
    )
    
    # 示例
    gr.Examples(
        examples=[
            "明天下午2點團隊會議，討論項目進度，地點在會議室A，參與者包括john@example.com",
            "2026-01-25 上午9點產品發布會，介紹新功能和改進，地點在總部大樓",
            "後天下午3點客戶會議，討論合作細節，參與者包括客戶代表",
            "下週一上午10點技術分享會，分享最新的 AI 技術，地點在研發中心"
        ],
        inputs=[calendar_prompt_input]
    )
    
    # 頁腳說明
    gr.Markdown(
        """
        ---
        **注意事項：**
        1. 使用 Google Calendar API 管理行事曆事件
        2. 首次使用需要在專案根目錄放置 `credentials.json`（從 Google Cloud Console 下載的 OAuth2 憑證）
        3. 首次運行時會自動開啟瀏覽器進行授權，授權後會生成 `token.json` 文件
        4. 事件內容由 AI 自動生成，請在創建前檢查結果
        5. 在提示中包含所有資訊：事件、日期、時間、地點、參與者
        6. 如果缺少日期或時間，系統會顯示下拉選單讓您選擇
        7. 日期格式支援：YYYY-MM-DD（例如：2026-01-25）或相對日期（今天、明天、後天）
        8. 時間格式支援：24小時制（14:00）或12小時制（2:00 PM）
        
        **設置步驟：**
        - 前往 [Google Cloud Console](https://console.cloud.google.com/) 創建專案
        - 啟用 Google Calendar API
        - 創建 OAuth2 憑證並下載為 `credentials.json`
        - 將 `credentials.json` 放在專案根目錄
        - 確保授予 Calendar API 的完整存取權限
        """
    )
