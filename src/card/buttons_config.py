"""Button configuration for card builders.

Button text values are sourced from UI_TEXT (src/card/ui_text.py) where a
corresponding key exists, ensuring a single source of truth across both the
static card builders and the CardSession pipeline.
"""

from src.card.ui_text import UI_TEXT

# Button Configuration
BUTTON_CONFIG = {
    # Deep Engine Buttons
    "pause": {"text": "⏸️ 暂停", "type": "default"},
    "stop": {"text": "⏹️ 停止", "type": "danger", "confirm": {"title": "确认停止", "text": "将中止当前运行的引擎任务，已产出内容将保留。"}},
    "resume": {"text": "▶️ 继续", "type": "primary"},
    "expand": {"text": "🔽 展开日志", "type": "default"},
    "collapse": {"text": "🔼 收起日志", "type": "default"},
    "expand_ac": {"text": "🔽 展开验收标准", "type": "default"},
    "collapse_ac": {"text": "🔼 收起验收标准", "type": "default"},
    "mode_full": {"text": UI_TEXT["card_btn_mode_full"], "type": "default"},
    "mode_compact": {"text": UI_TEXT["card_btn_mode_compact"], "type": "default"},
    "history": {"text": "📜 历史", "type": "default"},
    # Mode Switch & Project Buttons
    "exit_claude": {"text": "🚪 退出 Claude 模式", "type": "default"},
    "exit_coco": {"text": "🚪 退出 Coco 模式", "type": "default"},
    "exit_gemini": {"text": "🚪 退出 Gemini 模式", "type": "default"},
    "exit_traex": {"text": "🚪 退出 Traex 模式", "type": "default"},
    "exit_grok": {"text": "🚪 退出 Grok 模式", "type": "default"},
    "exit_dsh": {"text": "🚪 退出 DSH 模式", "type": "default"},
    "enter_coco": {"text": "🤖 进入 Coco 模式", "type": "default"},
    "enter_claude": {"text": "🔮 进入 Claude 模式", "type": "default"},
    "enter_gemini": {"text": "✨ 进入 Gemini 模式", "type": "default"},
    "enter_traex": {"text": "🚀 进入 Traex 模式", "type": "default"},
    "enter_grok": {"text": "🌌 进入 Grok 模式", "type": "default"},
    "enter_dsh": {"text": "🧭 进入 DSH 模式", "type": "default"},
    "switch_project": {"text": "🔄 切换项目", "type": "default"},
    # Reducer escalation button — rendered to user by DeepBuilder in escalation scenarios.
    # Also used in confirm dialogs when user initiates force-stop.
    "stop_danger": {"text": UI_TEXT["card_btn_force_stop"], "type": "danger"},
}
