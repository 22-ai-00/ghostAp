"""UI text constants for card rendering.

Key naming conventions:
- `system_*` — general system/handler messages (non-card pipeline).
- `<engine>_*` (deep_*, spec_*) — engine-specific card content.
"""

import types

__all__ = ["UI_TEXT"]

# UI Text Constants
UI_TEXT = {
    # Core Card UI
    "log_truncated_warning": "\n> ⚠️ **日志内容过长，已被截断**\n> 🔍 完整日志请查看服务器本地文件\n> (仅显示末尾内容)…\n",
    "claude_mode_title": "🔮 Claude 编程模式",
    "gemini_mode_title": "✨ Gemini 编程模式",
    "coco_mode_title": "编程模式",
    "smart_mode_title": "智能模式",
    "project_dir_label": "📂 项目目录: `{path}`",
    "image_alt_text": "图片 {index}",
    "time_just_now": "刚刚",
    "time_secs_ago": "{seconds} 秒前",
    "time_mins_ago": "{minutes} 分钟前",
    "time_mins_secs_ago": "{minutes} 分钟 {seconds} 秒前",
    "time_hours_ago": "{hours} 小时前",
    "time_hours_mins_ago": "{hours} 小时 {minutes} 分钟前",
    "time_days_ago": "{days} 天前",
    # Deep Engine
    "deep_cmd_help_usage": "📝 请提供需求描述\n\n用法: `/deep <你的需求描述>`\n\n例如: `/deep 帮我写一个 Python 爬虫，爬取豆瓣电影 Top250`",
    "deep_cmd_update_usage": "📝 请提供上下文信息\n\n用法: `/deep_update <上下文描述>`\n\n例如: `/deep_update 数据库改用 PostgreSQL 而不是 SQLite`",
    "deep_cmd_unknown": "❓ 未知的 Deep 命令\n\n可用命令:\n• `/deep <需求>` - 启动 Deep Agent\n• `/deep_update <上下文>` - 注入执行上下文\n• `/deep_status` - 查看当前项目进度\n• `/deep_status --all` - 查看所有项目 Deep Agent 看板\n• `/stop_deep` - 停止当前项目任务\n• `/stop_deep --all` - 停止所有项目任务",
    "deep_task_exists": "⚠️ 当前项目已有 Deep Agent 任务在执行中\n\n发送 `/deep_status` 查看进度\n发送 `/stop_deep` 停止任务",
    "deep_no_task_running": "⚠️ 当前没有正在运行的 Deep Agent 任务\n\n请先使用 `/deep <需求>` 启动任务，或使用 `/deep_status all` 查看所有项目任务",
    "deep_board_empty": "📊 当前没有 Deep Agent 任务\n\n发送 `/deep <需求>` 开始一个复杂任务",
    "deep_status_empty": "📊 当前没有进行中的任务\n\n发送 `/deep <需求>` 开始一个复杂任务",
    "deep_stop_all_success": "⏹ 已发送停止信号：{count} 个 Deep Agent 任务将在当前步骤完成后停止",
    "deep_no_active_tasks": "📊 当前没有正在执行的 Deep Agent 任务",
    # Spec Engine
    "spec_status_empty": "📊 当前没有 Spec 任务\n\n发送 `/spec 你的需求` 开始结构化开发闭环",
    "spec_cmd_guide_usage": "📝 请提供引导信息\n\n用法: `/spec_guide <引导描述>`\n\n例如: `/spec_guide 优先考虑性能优化`",
    "spec_cmd_help_usage": (
        "📋 **Spec 模式：结构化开发闭环**\n\n"
        "用法：`/spec <你的需求描述>`\n"
        "示例：`/spec 实现用户登录注册功能，支持邮箱和手机号`\n\n"
        "**Spec vs Deep**\n"
        "- Spec：按 `Spec→Plan→Task→Build→Review` 产出结构化产物并迭代收敛\n"
        "- Deep：一次性深度拆解并执行一个复杂任务（更偏单次冲刺）\n\n"
        "**最小示例（推荐命令组合）**\n"
        "- Web：`/spec 做一个登录页+登录接口` → `/spec_status` → `/spec_guide 优先补测试与错误提示`\n"
        "- API：`/spec 新增 /v1/users 查询接口` → `/spec_status`\n"
        "- 脚本：`/spec 写一个批量重命名脚本，支持dry-run` → `/spec_status`\n\n"
        "**可用命令**\n"
        "- `/spec <需求>`：启动\n"
        "- `/spec_guide <引导>`：补充约束/偏好（下轮生效）\n"
        "- `/spec_status`：查看进度\n"
        "- `/spec_history`：查看 spec 文件与循环历史\n"
        "- `/spec_metrics`：查看目标达成度与指标变化\n"
        "- `/spec_config`：查看 Spec 长程配置（阈值/保留策略）\n"
        "- `/spec_export`：导出当前 Spec/Plan 报告\n"
        "- `/spec_save`：立即落盘保存状态（用于断点续传）\n"
        "- `/stop_spec`：停止\n"
    ),
    # Generic Engine Lifecycle
    "engine_no_active_task": "📊 当前没有正在执行的 {engine_prefix} 任务",
    "engine_multi_stop_conflict": "⚠️ 有多个项目正在执行 {engine_prefix} 任务，请先切换项目再停止",
    "engine_stop_no_active": "📊 当前没有正在执行的 {engine_prefix} 任务",
    # System Commands
    "system_help_deep_prompt": "💭 启动 Deep Engine\n\n请发送: `/deep <你的需求>`\n\n例如: `/deep 帮我重构 src/feishu 模块`",
    "system_help_project_section": "\n\n📋 **项目管理命令**\n• `/projects` - 查看项目看板\n• `/new 名称 路径` - 创建新项目\n• `/switch 名称` - 切换项目\n• `/status` - 查看所有引擎任务状态（Deep/Spec）\n• `/status <task_id>` - 查看指定任务详情\n• `/diff` - 查看最近两次版本变更（Diff 报告）",
    "system_autonomous_manager_retired": (
        "⚠️ 独立 Autonomous Manager 命令面已退役，本次命令未执行。"
        "请使用 `/hire <名字>` 创建数字员工，并通过员工自己的会话继续任务。"
    ),
    "system_autonomous_manager_retired_goal": (
        "⚠️ 独立 Autonomous Manager 命令面已退役；原命令未执行。"
        "请先使用 `/hire <名字>` 创建数字员工，再在员工会话中提交任务：{task}"
    ),
    "system_new_project_usage": "用法: `/new 项目名 [路径]`",
    "system_close_project_usage": "用法: `/close <项目名>`\n例如: `/close myApp`",
    "system_setadmin_bootstrap_success": "✅ 已完成管理员初始化。当前发送者已写入 `.env`，并成为唯一 Bot 管理员。",
    "system_setadmin_update_success": "✅ 已更新 Bot 管理员。当前唯一管理员为 `{admin_id}`。",
    "system_setadmin_denied": "⚠️ Bot 管理员已初始化，只有现有管理员可以修改。",
    "system_setadmin_missing_sender": "无法识别发送者身份，不能设置管理员。",
    "system_setadmin_invalid_target": "管理员 ID 无效。用法：`/setadmin` 或由现有管理员发送 `/setadmin <open_id>`。",
    "system_setadmin_requires_p2p": "⚠️ 首次管理员初始化必须在私聊中进行，请通过私聊发送 `/setadmin`。",
    "system_setadmin_rate_limited": "⚠️ 操作过于频繁，请 60 秒后再试。",
    "system_current_project": "当前项目: **{name}**\n\n{help}{project_help}",
    "system_help_default": "{help}{project_help}",
    "system_mode_smart": "💭 智能模式",
    "system_mode_coco": "🤖 Coco 编程模式",
    "system_mode_claude": "🔮 Claude 编程模式",
    "system_mode_aiden": "🎯 Aiden 编程模式",
    "system_mode_codex": "💻 Codex 编程模式",
    "system_mode_gemini": "✨ Gemini 编程模式",
    "system_mode_traex": "🚀 Traex 编程模式",
    "system_mode_grok": "🌌 Grok 编程模式",
    "system_mode_dsh": "🧭 DSH 编程模式",
    # Mode name labels (without emoji prefix, used in card title building)
    "mode_name_claude": "Claude 编程模式",
    "mode_name_coco": "编程模式",
    "mode_name_gemini": "Gemini 编程模式",
    "mode_name_traex": "Traex 编程模式",
    "mode_name_grok": "Grok 编程模式",
    "mode_name_dsh": "DSH 编程模式",
    "mode_name_smart": "智能模式",
    # Quick action button labels (used in shared.py build_quick_actions)
    "qa_btn_confirm": "✅ 确认",
    "qa_btn_retry": "🔄 重试",
    "qa_btn_cancel": "❌ 取消",
    "qa_btn_continue": "▶️ 继续",
    "qa_btn_next": "⏭️ 下一步",
    "qa_btn_stop": "⏹️ 停止",
    "qa_btn_new_project": "➕ 创建项目",
    "qa_btn_list_projects": "📋 项目列表",
    "system_switching_to": "🔄 正在切换到 {tool} / {model}…",
    "system_already_in_mode": "💭 当前已经在智能模式中",
    "system_not_set": "未设置",
    "system_default": "默认",
    "system_no_active_project": "⚠️ 当前没有活跃项目，请先创建或切换到一个项目。\n\n发送 `/projects` 查看项目看板",
    "system_auto": "自动",
    "system_on": "开启",
    "system_off": "关闭",
    "system_no_project": "无",
    "system_no_project_display": "无项目",
    "system_unknown_error": "未知错误，请发送 /help 获取帮助",
    "system_unknown_unit": "未知单元",
    "system_unknown_execution_error": "未知执行异常，请重新发送命令重试",
    "system_mode_label": " 模式",
    "system_theme_color_label": "主题色: ",
    "system_history_record_title": " · 历史记录",
    "system_btn_prev_page": "⬅️ 上一页",
    "system_btn_next_page": "➡️ 下一页",
    "system_btn_back_status": "📊 返回状态",
    "system_status_completed": "已完成",
    "system_status_executing": "执行中",
    "system_status_partial_failed": "部分失败",
    "system_status_ready": "就绪",
    "system_status_preparing": "准备中",
    "system_label_current_tool": "🔧 **当前工具**",
    "system_label_current_model": "🤖 **当前模型**",
    "system_status_available": "✅ 可用",
    "system_status_unavailable": "❌ 不可用",
    "system_never_used": "从未使用",
    "system_unknown": "未知",
    "acp_result_text_header": "📝 输出文本",
    "acp_result_plan_header": "📋 计划",
    "acp_result_tools_header": "🛠️ 工具调用",
    "acp_result_tool_results_header": "📦 工具结果(本地记录)",
    "acp_result_files_header": "🗂️ 改动文件",
    "system_acp_tool_desc_coco": "字节跳动 AI",
    "system_acp_tool_coco_display": "字节跳动 AI 编程",
    "system_acp_tool_desc_claude": "Anthropic AI",
    "system_acp_tool_desc_aiden": "Aiden CLI",
    "system_acp_tool_aiden_display": "AI 编程助手",
    "system_acp_tool_desc_codex": "OpenAI Codex",
    "system_acp_tool_desc_gemini": "Google Gemini CLI",
    "system_acp_tool_desc_traex": "TRAE CLI",
    "system_acp_tool_desc_grok": "xAI Grok Build",
    "system_acp_tool_desc_dsh": "DeepSeek Harness",
    "system_arg_error": "参数错误",
    "system_internal_error": "系统错误",
    "system_slash_parse_missing": "命令解析结果缺失（系统侧），请重新发送该命令重试",
    "system_unknown_slash_command": "未知命令 `{command}`。发送 `/help` 查看可用命令。",
    "system_btw_usage": "请在 `/btw` 后补充想顺便了解的问题。",
    "system_btw_no_active_session": "当前没有活跃编程会话。请先发送 `/coco`、`/codex` 等进入编程模式。",
    "system_btw_observing": "💬 已启动只读旁路观察；主任务不会被中断，回答稍后发到本消息下。",
    "system_btw_already_running": "已有一个旁路问题正在处理中；主任务仍在继续，请稍后再试。",
    "system_btw_answer": "💬 **旁路回答**（未中断主任务）\n\n{answer}",
    "system_btw_fallback": "💬 旁路 Agent 暂时不可用，但主任务未被中断。\n\n{progress}",
    "system_acp_unsupported_tool": "不支持的 ACP 工具: {tool_name}",
    "system_acp_unsupported_1m_model": "Claude 模型 `{model}` 不支持 1M 上下文，请选择标准版本。",
    "system_acp_no_available_tools": "未检测到可用 ACP 工具",
    "system_acp_select_tool_prompt": "请选择 ACP 工具",
    "system_acp_select_model_prompt": "请选择 ACP 模型",
    "system_acp_model_select_title": "{tool} 模型选择",
    "system_acp_model_select_intro": "请选择本次编程会话使用的模型配置。显式选择完成后才会初始化会话。",
    "system_acp_model_group_label": "**模型族**",
    "system_acp_model_profile_label": "**Profile**",
    "system_acp_model_effort_label": "**Effort**",
    "system_acp_model_group_placeholder": "选择模型",
    "system_acp_model_profile_placeholder": "选择 Profile",
    "system_acp_model_effort_placeholder": "选择 Effort",
    "system_acp_default_model_option": "默认模型（推荐）",
    "system_acp_confirm_model": "确认此模型：{model}",
    "system_acp_refresh_models": "刷新模型",
    "system_acp_no_models_hint": "当前后端未报告可选模型，仍可使用后端默认模型。",
    "system_acp_model_loading_title": "{tool} 模型加载中",
    "system_acp_model_loading_body": "正在读取 **{tool}** 官方 ACP 模型能力，请稍候。",
    "system_acp_model_error_title": "{tool} 模型加载失败",
    "system_acp_model_error_body": "未能读取 **{tool}** 的模型能力。可刷新后重试。",
    "system_acp_programming_initializing_title": "{tool} 编程模式正在初始化",
    "system_acp_programming_initializing_body": "已选择 `{model}`，正在启动 ACP 会话。无需重复操作。",
    "system_acp_programming_ready_title": "{tool} 编程模式已就绪",
    "system_acp_programming_ready_body": "ACP 会话已使用 `{model}` 启动，现在可以直接发送需求。",
    "system_acp_programming_failed_title": "{tool} 编程模式初始化失败",
    "system_acp_programming_failed_body": "模型 `{model}` 未能启动。\n\n{reason}",
    "system_acp_switch_model_btn": "切换模型",
    "system_acp_retry_activation_btn": "重试初始化",
    "system_acp_back_to_models_btn": "返回模型选择",
    "system_acp_activation_failed_safe": "ACP 会话启动失败，请重试",
    "system_acp_config_summary": "**ACP 配置**\n当前工具：`{current}`\n可用工具：{available}\n用法：`/acp <tool>`",
    "system_acp_config_saved": "ACP 配置已保存：`{tool}` / `{model}`。后续任务将自动采用。",
    "system_model_config_summary": "**模型配置** · `{tool}`\n当前模型：`{current}`\n可用模型：{available}\n用法：`/model <model>` 或 `/model default`",
    "system_model_config_saved": "模型配置已保存：`{tool}` / `{model}`。后续任务将自动采用。",
    "system_acp_config_save_failed": "ACP 配置保存失败，原配置保持不变。",
    "system_acp_unknown_model": "当前工具未报告模型 `{model}`。请发送 `/model` 查看可用值。",
    "system_dir_changed_title": "目录已切换",
    "system_error_title": "操作失败",
    "system_coco_status_title": "**🤖 Coco 状态**\n",
    "system_coco_current_model": "当前模型: `{model}`",
    "system_coco_available_models": "\n**可用模型:**",
    "system_error_prompt_title": "⚠️ 错误提示",
    "system_ref_note_prefix": "🔗 关联：",
    "system_rate_limit_title": "⏸️ 限速等待",
    "system_rate_limit_content": "🔄 API 限速触发，自动等待 {wait_seconds} 秒后恢复…\n\n无需操作，任务将自动继续。",
    "system_shell_success_title": "✅ 命令执行成功",
    "system_shell_failed_title": "❌ 命令执行失败",
    "system_shell_no_output": "✅ 命令执行成功（无输出）",
    "system_shell_return_code": "返回码: `{code}`",
    "system_shell_stderr_label": "⚠️ **错误输出**:",
    "system_menu_title": "📱 快捷菜单",
    "system_menu_header": "**📱 常用指令菜单**",
    "system_menu_btn_new_project": "新项目",
    "system_menu_btn_switch_project": "切项目",
    "system_menu_btn_deep_task": "Deep",
    "system_menu_btn_workflow": "Workflow",
    "system_menu_btn_status": "状态",
    "system_menu_btn_help": "📖 帮助",
    "system_help_title": "📖 GhostAP 使用帮助",
    "system_help_status_header": "**当前状态** · {mode} · 项目: {project}\n<font color='grey'>{cwd}</font>",
    "system_help_section_modes": "🔄 编程模式切换",
    "system_help_section_modes_body": (
        "`/coco` · 进入 Coco 编程模式（字节跳动 AI）\n"
        "`/claude` · 进入 Claude 编程模式（Anthropic AI）\n"
        "`/aiden` · 进入 Aiden 编程模式\n"
        "`/codex` · 进入 Codex 编程模式\n"
        "`/gemini` · 进入 Gemini 编程模式\n"
        "`/traex` · 进入 Traex 编程模式\n"
        "`/grok` · 进入 Grok 编程模式（xAI Grok Build）\n"
        "`/dsh` · 进入 DSH 编程模式（DeepSeek Harness）\n"
        "`/exit` · 退出当前编程模式\n"
        "`/coco_info` · `/claude_info` · `/aiden_info` · `/codex_info` · `/gemini_info` · `/traex_info` · `/grok_info` · `/dsh_info` · 查看会话/模型信息"
    ),
    "system_help_section_deep": "💭 Deep Engine · 复杂任务一次交付",
    "system_help_section_deep_body": (
        "`/deep <需求>` · 启动 Deep Engine\n"
        "`/deep_status` · 查看任务进度\n"
        "`/stop_deep` · 停止任务"
    ),
    "system_help_section_spec": "📋 Spec Engine · 结构化开发闭环",
    "system_help_section_spec_body": (
        "`/spec <需求>` · 启动  ·  `/spec_status` · 查看进度\n"
        "`/spec_guide <引导>` · 补充约束/偏好\n"
        "`/spec_history` · 历史  ·  `/spec_metrics` · 目标达成度\n"
        "`/spec_config` · 配置  ·  `/spec_save` · 立即保存\n"
        "`/spec_export` · 导出报告  ·  `/stop_spec` · 停止任务\n"
        "`/stop_spec` · 停止"
    ),
    "system_help_section_project": "📂 项目管理",
    "system_help_section_project_body": (
        "`/projects` · 查看所有项目\n"
        "`/new <名称> [路径]` · 创建新项目\n"
        "`/new-chat <名称> [后缀] [路径]` · 创建/绑定项目群\n"
        "`/switch <名称>` · 切换项目  ·  `/close <名称>` · 关闭项目\n"
        "`/status` · 查看所有引擎任务状态\n"
        "`/diff` · 查看最近两次版本变更"
    ),
    "system_help_section_workflow": "⚡ Workflow · 多 Agent 编排执行",
    "system_help_section_workflow_body": (
        "`/wf <需求>` · 选择 1–8 个 Agent 组成 Agent Pool\n"
        "`/wf_status` · 查看当前 Workflow 进度\n"
        "`/wf_help`（`/workflow_help`） · Workflow 命令帮助\n"
        "`/stop_wf` · 停止正在运行的 Workflow\n"
        "别名：`/workflow` · `/workflow_status` · `/stop_workflow`\n"
        "流程：发起者点击“使用此池开始编排”完成单次确认 → "
        "确认后自动生成、验证并执行脚本 → 汇总结果"
    ),
    "system_help_section_hire": "👤 数字员工 · 雇佣与管理",
    "system_help_section_hire_body": (
        "`/hire <名字>` · 自动选择推荐可用工具和后端默认模型\n"
        "`/h <名字>` · `/hire` 的简写\n"
        "`/hire <名字> --tool codex --model <模型>` · 受控覆盖工具或模型\n"
        "`/employees`（`/roster`） · 查看在职员工\n"
        "`/employee-role <员工名> <职责>` · 设置员工职责\n"
        "`/fire <名字或ID> [--drain]` · 退役员工，可等待当前任务排空\n"
        "`/history <名字或ID>` · 查看员工执行历史\n"
        "`/employee-memory <名字或ID>` · 查看员工记忆\n"
        "以上管理命令由配置管理员在主 Bot 私聊使用"
    ),
    "system_help_tips": (
        "**💡 使用提示**\n"
        "1. 发送 `/menu` 打开完整快捷菜单\n"
        "2. 发送 `/help` 随时查看帮助\n"
        "3. ⏰ 长时间无更新超过 {timeout_display} 才会关闭/回收；运行中的任务会自动顺延，回收前 {warn_display} 会提醒续期\n"
        "<font color='grey'>调整: CARD_SESSION_IDLE_TIMEOUT</font>"
    ),
    "system_tools_list_title": "🛠️ 工具选择",
    "system_tools_list_header": "**🔧 可用工具列表**",
    "system_tools_list_footer": "可用工具: {available}/{total} • 点击按钮进入对应模式 • 灰色按钮表示工具不可用",
    "system_tools_status_title": "📋 工具状态",
    "system_tools_status_header": "**📊 工具状态详情**",
    "system_tools_status_item": "{emoji} **{name}**\n   状态: {status}\n   最后使用: {last_used}{active_info}",
    "system_tools_status_active_session": "\n   🔴 活跃会话: {chat_id}",
    "system_tools_status_btn_enter": "进入 {name}",
    # Project Commands
    "project_create_success": "✅ 项目 **{name}** 创建成功\n\n📁 路径: `{path}`",
    "project_create_error": "❌ 创建项目失败: {error}",
    "project_not_found_title": "未找到项目",
    "project_switch_success": "✅ 已切换到项目: **{name}**\n\n📁 路径: `{path}`",
    "project_switch_error": "❌ 切换项目失败: {error}",
    "project_not_found": "❌ 未找到项目: {name}",
    "project_not_found_hint": "❌ 未找到项目: {name}\n\n发送 `/projects` 查看所有项目",
    "project_close_success": "✅ 项目 **{name}** 已关闭",
    "project_close_error": "❌ 关闭项目失败: {error}",
    "project_dir_info_title": "目录信息",
    "project_dir_info": "📂 **项目目录**: `{root}`\n📁 **工作目录**: `{cwd}`",
    "project_dir_info_cwd": "• 📁 工作目录: `{cwd}`",
    "project_dir_switched": "✅ 已切换到: `{path}`",
    "project_dir_switched_banner": "目录已切换到: {path}",
    "project_dir_switched_detail": "当前工作目录已成功更新为 `{path}`。",
    "project_dir_switch_failed_banner": "切换目录失败: {path}",
    "project_dir_switch_failed_detail": "无法切换到目标目录: `{path}`",
    "project_dir_not_exist": "目录不存在: {path}",
    "project_current_header": "📁 **当前项目: {name}**",
    "project_id_label": "• 项目 ID: `{id}`",
    "project_status_label": "• 状态: {emoji} {status}",
    "project_coco_status": "• Coco 模式: {status}",
    "project_claude_status": "• Claude 模式: {status}",
    "project_last_active_label": "• 最后活跃: {time_ago}",
    "project_coco_session_header": "\n\n🤖 **Coco 会话**",
    "project_claude_session_header": "\n\n🔮 **Claude 会话**",
    "project_session_id_label": "• 会话 ID: `{id}`",
    "project_session_count_label": "• 对话数: {count}",
    "project_similar_header": "\n\n**相似项目：**",
    "project_restore_info": "\n\n📋 已恢复上下文: {count} 条记录",
    "project_restore_last_mode": ", 上次模式: {mode}",
    "project_switched_content": "已切换到项目 **{name}**\n\n📂 项目目录: `{root}`{context_info}",
    "project_board_title": "📋 项目看板",
    "project_board_empty": "当前没有活动项目\n\n发送 `/new 项目名 [路径]` 创建新项目",
    "project_board_active_section": "**🟢 当前活动项目**",
    "project_board_other_section": "**⚪ 其他项目**",
    "project_info_card_title": "当前项目",
    "project_status_card_title": "项目状态",
    "project_switch_title": "🔄 项目已切换",
    "project_btn_open_group": "进入项目群",
    "project_status_group_label": "• 项目群: {name}",
    "project_status_no_group": "• 项目群: 未绑定",
    "diag_current_mode": "**当前模式**: {mode}",
    "diag_no_active_tasks": "暂无正在进行的任务",
    "diag_no_tasks": "暂无任务",
    "diag_project_header": "### {name} {id_info}",
    "diag_task_line": "- {emoji} `{run_id}` {name} ({task_type}){pct}{msg}",
    "diag_unified_status_header": "**引擎任务 ({count})**\n",
    "diag_engine_line": "- {emoji} **{mode}** · {name}{tid} · {info}",
    "diag_status_all_hint": "\n_发送 `/status all` 查看包括已完成的任务_",
    "diag_no_engine_tasks": "当前没有 Deep/Spec 引擎任务\n\n",
    "diag_engine_launch_hints": "启动任务:\n• `/deep <需求>` — 单次深度执行\n• `/spec <需求>` — 结构化开发",
    "diag_task_board_title": "📋 任务看板",
    "diag_no_active_project_tasks": "当前没有活跃项目，无法按项目查看任务。\n\n发送 /projects 查看项目看板",
    "diag_status_iteration": "迭代{iteration}",
    "diag_status_criteria": "标准{criteria}",
    "diag_status_cycle": "循环{cycle}",
    "diag_unified_status_title": "📊 统一状态",
    "diag_task_detail_deep_title": "📊 Deep 任务详情",
    "diag_task_detail_spec_title": "📊 Spec 任务详情",
    "diag_diff_no_active_project": "当前没有活跃项目，无法生成 Diff 报告。\n\n发送 /projects 选择项目",
    "diag_diff_report_title": "🧾 Diff 报告",
    "diag_diff_usage_footer": "用法：/diff（最近两版） • /diff current（到当前） • /diff N • /diff A..B",
    "diag_trace_not_found": "未找到关联信息：{key}",
    "diag_trace_replies_header": "### 📨 回复消息 ({count})",
    "diag_trace_runs_header": "### 🧵 任务 run_id ({count})",
    "diag_trace_usage_footer": "提示：/trace <id> 支持 origin/reply/run_id/request_id",
    "diag_trace_title": "🔎 关联查询",
    "diag_step_parsing": "解析参数",
    "diag_step_generating": "生成报告",
    "diag_step_completed": "完成",
    "diag_engine_deep": "Deep",
    "diag_engine_spec": "Spec",
    "diag_label_origin_id": "**Origin ID**",
    "diag_label_request_id": "**Request ID**",
    "diag_label_project_id": "**Project ID**",
    "diag_project_id_fmt": "(`{id}`)",
    "diag_task_pct_fmt": " {pct:.0f}%",
    "diag_task_msg_fmt": " — {msg}",
    "diag_task_detail_title": "📊 任务详情",
    "diag_task_name_label": "**任务**: {name}",
    "diag_task_type_label": "**类型**: {type}",
    "diag_task_status_label": "**状态**: {emoji} {status}",
    "diag_task_run_id_label": "**run_id**: `{id}`",
    "diag_task_id_label": "**task_id**: `{id}`",
    "diag_task_progress_label": "**进度**: {msg}",
    "diag_task_not_found": "未找到 task_id: `{id}`\n\n发送 `/status` 查看所有任务",
    "diag_diff_title": "## 🧾 Diff 报告",
    "diag_diff_project_label": "**项目**: {name} (`{id}`)",
    "diag_diff_range_label": "**范围**: v{from_v} → v{to_v}",
    "diag_diff_range_current_label": "**范围**: v{from_v} → 当前",
    "diag_diff_start_reason_label": "**起点原因**: {reason}",
    "diag_diff_end_reason_label": "**终点原因**: {reason}",
    "diag_diff_added_entries_label": "**新增条目**: {count}",
    "diag_diff_no_entries": "✅ 本范围内没有新增上下文条目",
    "diag_diff_file_changes_header": "### 📝 文件变更 ({count})",
    "diag_diff_deep_results_header": "### 💭 Deep 结果 ({count})",
    "diag_diff_deep_item": "- `{name}`：已完成 {done}/{total} 个任务",
    "diag_diff_mode_changes_header": "### 🔄 模式切换 ({count})",
    "diag_diff_summaries_header": "### 📌 AI 摘要 ({count})",
    "diag_diff_conversations_header": "### 💬 对话片段 ({count})",
    "diag_diff_others_header": "### 📎 其他 ({count})",
    "diag_diff_truncated": "\n…（内容过长已截断，使用 /shell cat <path> 查看完整输出）",
    "diag_diff_generating_banner": "🧾 正在生成 Diff 报告…",
    "diag_diff_started_banner": "🧾 已开始生成 Diff 报告…",
    "diag_diff_generating_progress": "🧾 {step}中（{pct}%）…",
    "diag_diff_failed": "生成 Diff 报告失败",
    "diag_diff_exception": "Diff 报告生成异常: {error}",
    "diag_diff_no_record": "该项目暂无上下文记录，无法生成 Diff 报告。",
    "diag_diff_no_bookmarks": "该项目尚无版本书签。\n\n提示：版本书签会在模式切换、项目切换、Deep 完成等关键节点自动创建。",
    "diag_diff_no_current": "该项目尚无版本书签，无法计算 `current` diff。",
    "diag_diff_usage_error": "用法错误：`/diff <A>..<B>`，例如 `/diff 3..5`。",
    "diag_diff_usage_hint": "用法：`/diff [last|current|N|A..B]`，例如 `/diff current` 或 `/diff 2..3`。",
    "diag_diff_version_not_found": "找不到版本 v{vnum}，当前共有 {total} 个版本。",
    "project_board_empty_content": "暂无项目\n\n发送 `/new 项目名 路径` 创建新项目",
    "project_board_total_projects": "共 **{total}** 个项目",
    "project_board_current_marker": " (当前)",
    "project_board_claude_info": " | 🔮 Claude 模式中 (消息数: {count})",
    "project_board_coco_info": " | 🤖 Coco 模式中 (消息数: {count})",
    "project_board_busy_info": " | ⏳ {task_type}",
    "project_board_page_info": "第 {page}/{total} 页",
    "project_board_btn_new": "➕ 新建项目",
    "project_board_btn_switch": "切换到此项目",
    "project_board_btn_continue": "继续开发",
    "project_board_btn_detail": "查看详情",
    "project_board_btn_prev": "⬅️ 上一页",
    "project_board_btn_next": "下一页 ➡️",
    "project_board_btn_refresh": "🔄 刷新",
    "project_created_title": "🎉 新项目已创建",
    "project_btn_start_coco": "🤖 开始 Coco",
    "project_btn_start_claude": "🔮 开始 Claude",
    "project_notif_suggestion_header": "💡 **建议下一步:**",
    # ── Programming Mode Prompts (centralized) ──
    "mode_enter_msg": "{emoji} 已进入{name}编程模式\n\n现在可以用自然语言描述你的需求\n\n说「退出模式」或发送 `/exit` 退出",
    "mode_enter_no_project_msg": "{emoji} 已进入 {name} 编程模式\n\n现在可以用自然语言描述你的需求\n\n说「退出模式」或发送 `/exit` 退出",
    "mode_already_in_msg": "已经在{name}编程模式中\n\n{info}\n\n说「退出模式」或发送 /exit 退出",
    "mode_exit_msg": "👋 已退出{name}编程模式\n\n会话已保存，下次可以恢复\n\n当前为 💭 智能模式",
    "mode_exit_pending_msg": "👋 已退出{name}编程模式\n\n当前为 💭 智能模式",
    "mode_not_in_msg": "当前不在 {name} 模式中",
    "mode_thinking_msg": "{emoji} {name} 正在思考…",
    "mode_resume_msg": "🔄 已恢复 {name} 会话\n\n• 会话 ID: `{session_id}`\n• 历史对话: {query_count} 条\n\n{hint}",
    "mode_resume_hint_default": "继续之前的对话吧！",
    "mode_session_fail_msg": "{name} 会话启动失败，请重新发送 /{cmd} 开始",
    # ── Deep Engine Card Prompts ──
    "deep_error_no_detail": "任务异常中断，发送 {engine_cmd} 重试",
    "deep_error_fallback_no_prefix": "任务异常中断，发送 /deep 重试或 /help 查看可用命令",
    "deep_error_expand_hint": '\n> (展开日志查看更多)…',
    "deep_executing_placeholder": "正在执行…",
    "deep_content_expand_hint": '\n> (展开日志查看更多)',
    "deep_folded_lines_hint": "…(已折叠 {count} 行)…\n",
    "deep_ac_expand_hint": '\n> (展开验收标准查看更多)…',
    "deep_status_failed_zh": "失败",
    "deep_status_completed_zh": "完成",
    "deep_status_finished_zh": "结束",
    "deep_status_planning_zh": "规划",
    "deep_status_analyzing_zh": "分析",
    # ── Shell Truncation ──
    "shell_truncated": "\n…(已截断，使用 /shell cat <path> 查看完整输出)…",
    # ── Streaming Card Collapsible / Continuation ──
    "continuation_stale_stub": "▼ 请查看下方最新卡片",
    # ── ws_client Prompts ──
    "ws_session_fail_msg": "⚠️ {name} 会话启动失败，已退回智能模式，请重新发送 /{cmd} 重试",
    "ws_topic_hint_msg": "💡 当前对话已在编程模式中，直接发送需求即可\n\n如需切换工具，请先发送 `/exit` 回到智能模式，再发送 `/codex`、`/coco` 等命令",
    "ws_topic_project_unavailable": "⚠️ 当前引擎话题绑定的项目不可用，本次消息未执行。请先通过 `/projects` 恢复项目关联。",
    "topic_engine_switch_blocked": "当前话题已绑定 {current} 模式，不能直接切换到 {requested}。\n\n请先停止当前话题任务，或在主对话中新开一个话题启动 {requested}。",
    "topic_engine_exit_msg": "👋 已退出当前话题的 {engine} 模式\n\n这个话题后续消息将回到智能模式。",
    "ws_exit_deferred_msg": "✅ 已收到 /exit，将在当前任务完成后退出（不中断执行）",
    "ws_active_topic_msg": "💡 你有一个活跃的 {name} 编程话题正在进行中\n\n请在话题中回复继续对话\n如需新建编程环境，请先发送对应的编程模式命令（如 /coco）",
    "ws_backpressure_spec": "⚠️ 系统繁忙，当前 Spec cycle 仍会自动重试，请稍候",
    "ws_backpressure_generic": "⚠️ 系统繁忙，请稍后再试",
    "ws_message_timeout": "⏳ 处理消息超时，请稍后重试",
    "ws_message_internal_error": "❌ 处理消息时发生内部错误，请稍后重试",
    "ws_message_prestart_terminal": (
        "⚠️ 消息任务在开始执行前因服务中断而终止，GhostAP 主 Bot 未执行。"
        "请重新发送该命令。"
    ),
    "ws_employee_handoff_unconfirmed": (
        "⚠️ 尚未确认目标员工已接收该命令；GhostAP 主 Bot 未执行。"
        "请查看员工回复后再决定是否重试。"
    ),
    "ws_employee_handoff_unknown": (
        "⚠️ 无法确认目标员工是否已接收该命令；命令状态未知，请勿重试。"
        "请先等待目标员工回复，或联系管理员核查。"
    ),
    "ws_main_bot_identity_unavailable": (
        "⚠️ 暂时无法确认该命令的处理方；GhostAP 主 Bot 未执行。"
        "命令状态未知，请勿立即重试，先等待目标 Bot 回复。"
    ),
    "ws_unsupported_msg_type": "⚠️ 目前仅支持文本、图片和富文本消息",
    "ws_image_only_prefix": "请查看并理解以下图片",
    "ws_system_cmd_gate_blocked": "⏳ 系统指令处理中，按钮暂不可用，请稍后重试",
    "ws_card_action_ack": "已收到操作，正在处理…",
    "ws_project_eviction_notify": "该项目「{name}」暂时与当前群断开连接——因为同时使用的群聊数量已满。你可以随时发送 /project 重新连接",
    "ws_fallback_admin_name": "Bot 管理员",
    # Engine error prefixes
    "engine_error_timeout": "超时",
    "engine_error_exception": "异常",
    # ── Retry / Signature ──
    "retry_command_sig_mismatch": "⚠️ 此按钮已失效，请重新发送命令",
    "retry_command_sig_upgrade_expired": "此按钮已过期，请重新发送命令",
    "retry_project_unavailable": "⚠️ 原项目不可用，请重新操作。",
    # ── LRU Eviction Notification ──
    "eviction_notify_title": "📤 项目已自动解绑",
    "eviction_notify_body": "该项目「{name}」暂时与当前群断开连接——因为同时使用的群聊数量已达 {max} 个上限。\n你可以随时发送 /project 重新连接。",
    "eviction_notify_btn_rebind": "🔗 重新绑定",
    # ── Dispatcher: Intent / Multi-task ──
    "intent_unknown_msg": "🤔 无法理解你的意图，请尝试 /help 查看可用命令",
    "project_close_name_required": "❌ 请指定要关闭的项目名称",
    "multi_task_step_failed": "⚠️ 步骤 {step} 执行失败，后续步骤已取消",
    "multi_task_step_success": "✅ 步骤 {step}: {desc}",
    "multi_task_step_error": "❌ 步骤 {step}: {error}",
    "task_desc_unknown": "未知操作",
    # ── Programming Mode Card / Banner ──
    "mode_card_enter_title": "{emoji} {name}编程模式已开启",
    "mode_card_programming_title": "{emoji} {name}编程模式",
    "mode_enter_banner": "{name} 编程模式已开启",
    "mode_exit_banner": "已退出 {name} 编程模式",
    "mode_exit_card_title": "模式已退出",
    "mode_resume_banner": "{name} 会话已恢复",
    "mode_session_info_title": "{name} 会话信息",
    "mode_model_switched_banner": "已切换 {name} 模型为: {model}",
    "mode_model_switched_title": "{name} 模型已切换",
    "mode_model_switch_context_kept": "对话上下文已保留，可以继续当前任务。",
    "mode_model_switch_restarted": "已重启会话，可以开始新任务。",
    "mode_model_switch_error": "切换 {name} 模型失败: {error}",
    "mode_startup_timeout_title": "启动 {name} 会话超时",
    "mode_startup_fail_title": "启动 {name} 会话失败",
    "mode_invalid_project_path": "⚠️ {msg}\n\n请切换到有效目录后重试",
    "mode_project_dir_label": "📂 项目目录: {path}",
    "mode_exec_complete": "✅ 执行完成",
    "mode_exec_cancelled_title": "执行已取消",
    "mode_exec_cancelled_msg": "⏹️ 任务已取消：{reason}",
    "mode_exec_incomplete_title": "执行未完成",
    "mode_exec_incomplete_msg": "⚠️ 任务未完成：{reason}",
    "mode_exec_finalization_incomplete_msg": (
        "执行窗口已耗尽；安全收尾后{reason}\n"
        "可查看脱敏诊断后重新发起任务。"
    ),
    "mode_exec_finalization_paused_goal_msg": (
        "执行窗口已耗尽；安全收尾后{reason}\n"
        "Codex Goal 已确认暂停，未在收尾阶段继续扩张；"
        "可查看脱敏诊断后重新发起或从会话历史恢复任务。"
    ),
    "mode_exec_window_limit_incomplete_msg": (
        "已自动执行 {windows} 个窗口并达到配置的安全上限；{reason}\n"
        "现有成果已安全保存，可查看脱敏诊断后继续处理。"
    ),
    "mode_exec_timeout_title": "执行超时",
    "mode_exec_exception_title": "执行异常",
    "mode_exec_timeout_msg": "⏳ 执行超时: {error}",
    "mode_exec_exception_msg": "❌ 执行异常: {error}",
    "mode_stream_fallback_running": (
        "⚠️ 流式编程卡片当前不可用，已切换为普通文本；任务仍在执行，"
        "完成或失败后会在这里明确回复。"
    ),
    "mode_stream_fallback_permission_hint": (
        "如尚未开通卡片权限，可点击下方链接申请并发布；"
        "不开通也不影响本次任务继续使用普通文本："
    ),
    "mode_working_dir_label": "📁 工作目录: `{path}`",
    # --- Card lifecycle strings ---
    "card_lifecycle_thinking": "💭 正在思考…",
    "card_lifecycle_reasoning": "💭 正在分析…",
    "card_lifecycle_error_fallback": "❌ 执行异常，请重新发送 {engine_cmd} 启动新任务",
    "card_lifecycle_error_fallback_spec": "❌ 自动恢复已耗尽，任务已明确失败；可重新发送 {engine_cmd} 启动新任务",
    "card_lifecycle_retry_failed": "🔄 重新执行失败工具",
    "card_lifecycle_restart": "🔄 重新开始",
    "card_lifecycle_degraded_primary": "继续使用 {mode}",
    "card_lifecycle_degraded_primary_unknown": "重新发送原命令",
    "card_lifecycle_continue_degraded_ack": "✅ 已继续使用 {mode}。你可以直接继续发送消息；原能力恢复由系统自动处理。",
    "card_lifecycle_continue_degraded_unknown_ack": "当前暂未确定可继续模式，请重新发送原命令或查看诊断。",
    "card_lifecycle_show_details": "查看详情",
    "card_lifecycle_details_collapsed": "🔒 详情已收起：点击“查看详情”后将返回已脱敏、截断的诊断信息。",
    "mode_exec_child_history_reconciled": (
        "⚠️ 状态对账提醒：检测到 {count} 条历史子代理执行记录未闭合；"
        "最后一次权威 Agent 状态已确认全部终态，不影响本次结果。"
    ),
    "card_lifecycle_cancelled_block": "⚠️ 已取消",
    "card_lifecycle_cancelled_status": "已取消",
    "card_lifecycle_paused": "⏸ 已暂停",
    "card_lifecycle_stopping": "🛑 正在停止…",
    "card_lifecycle_resume": "▶️ 恢复执行",
    "card_lifecycle_blocked": "🚫 任务阻塞",
    "card_lifecycle_archived": "📦 已归档",
    "card_lifecycle_archived_hint": "请查看下方最新卡片获取当前进度。",
    "card_archived_navigate_btn": "📍 查看最新卡片",
    "card_archived_scroll_hint": "💡 请在对话底部查看最新卡片",
    "card_lifecycle_mode_title": "{emoji} {mode_name} 编程模式",
    "card_btn_stop": "⏹️ 停止",
    "card_btn_stopping": "正在停止…",
    "card_btn_stop_confirm": "是否停止？已完成的部分不受影响。",
    "card_btn_restart": "🔄 重新开始",
    "card_tool_running": "🔧 执行中: {tool_name}",
    "card_retry_waiting": "⏳ 等待 {delay_sec} 秒后重试（{attempt}/{max_attempts}）",
    "card_retry_executing": "🔄 重试中 ({attempt}/{max_attempts})",
    "card_retry_exhausted": "❌ 重试已耗尽 ({max_attempts}/{max_attempts})，请重新发送命令启动新任务",
    "card_retry_skip": "ℹ️ 无需重试",
    # --- Card session strings ---
    "card_session_recovery_banner": "进度显示曾中断，当前已恢复",
    "card_session_max_failures_banner": "状态更新暂停，任务仍在运行。最后更新于 {timestamp}。请等待自动恢复，或发送 /status 查看",
    "card_session_max_failures_banner_no_ts": "状态更新暂停，任务仍在运行。请等待自动恢复，或发送 /status 查看最新状态",
    "card_session_toast_closed": "{engine_name}任务已结束，发送 {engine_cmd} 启动新任务",
    "card_session_toast_ttl_closed": "{engine_name}已超时关闭，发送 {engine_cmd} 可重新启动",
    "card_session_toast_failed": "{engine_name}任务已失败，请重新发送 {engine_cmd} 重试",
    "card_session_toast_unknown_action": "该按钮已失效，请在对话中重新发送 {engine_cmd} 或 /help 查看可用命令",
    "card_session_toast_factory_error": "操作异常，请在对话中重新输入命令重试",
    "card_session_toast_dispatch_error": "系统暂时繁忙，请稍等片刻后重新发送命令",
    "card_session_toast_action_ack": "操作已提交，正在处理",
    "card_session_toast_dedup": "操作正在处理中，请稍候",
    "workflow_selection_toast_processing": "正在应用选择，卡片将自动更新",
    "card_session_terminal_retry_failed": "⚠️ 任务已结束（卡片未能刷新最终状态）。可发送状态命令查看结果；开始新任务请发送 {engine_cmd} <需求描述>",
    # TTL expiry messages — mutually exclusive triggers:
    # All engine-specific messages follow a consistent structure:
    # TTL expired messages — unified two-line structure: status line + action line
    "card_session_ttl_expired": "⏰ 已超时关闭 · 系统回收资源\n发送 {expired_commands} 重新启动",
    "card_session_ttl_expired_spec": "⏰ {engine_name} 已超时关闭 · 失败终态与进度已保存\n可重新发送 {engine_cmd} 启动新任务",
    "card_session_ttl_expired_deep": "⏰ {engine_name} 已超时关闭 · 执行结果已保存\n发送 /deep_status 查看状态 / {engine_cmd} 重新启动",
    "card_session_ttl_expired_commands": "",  # dynamically generated below from ENGINE_CMD_MAP
    "card_session_ttl_prewarning": "{engine_name} 会话长时间无更新，将在约 {minutes} 分钟后关闭；执行中的任务会自动顺延。点击下方「保持连接」按钮或发送任意消息即可续期",
    "card_session_ttl_force_close_notice": "⏰ {engine_name} 已因空闲超时关闭（系统回收资源）· 发送 {engine_cmd} 重新启动",
    "card_session_terminal_fallback_notice": "⚠️ 任务已结束，但卡片未能及时更新。可发送状态命令查看结果；开始新任务请发送 {engine_cmd} <需求描述>，无需点击之前卡片中的按钮",
    "card_session_warning_render_fail": "⚠️ 状态更新异常，任务仍在运行，可发送 {engine_cmd} 查看最新状态",
    "card_session_ttl_lock_contention": "⚠️ 系统繁忙，请稍后重试或发送 {engine_cmd} 重新启动",
    "card_session_toast_completed": "任务已完成，无需额外操作",
    "card_session_rejected_throttled": "当前使用人数较多，将在 {delay} 秒后自动重试（第 {attempt} 次，最多 {max} 次）",
    "card_footer_last_updated": "最后更新于 {timestamp}",
    "ttl_keep_alive_btn": "保持连接（剩余约 {minutes} 分钟）",
    "ttl_keep_alive_toast": "已续期，会话将继续保持活跃",
    "card_btn_disabled_tips": "该按钮当前不可用",
    "card_btn_show_status": "📋 查看状态",
    "card_btn_mode_full": "📋 切换完整模式",
    "card_btn_mode_compact": "📄 切换精简模式",
    "card_btn_force_stop": "⛔ 强制停止",
    "card_btn_confirm_dialog_title": "确认",
    "card_btn_confirm_stop_title": "确定停止当前任务？",  # DEPRECATED: use _normal or _danger variant
    "card_btn_confirm_stop_title_normal": "确定停止当前任务？",
    "card_btn_confirm_stop_title_danger": "强制停止当前任务",
    "card_btn_confirm_stop_body": "进行中的执行将中断，已完成的部分不受影响",
    "card_btn_confirm_stop_body_normal": "正在进行的步骤将中断，已保存的产物不受影响",
    "card_btn_confirm_stop_danger_body": "⚠️ 强制停止将中断一切进度（普通停止仅中断当前步骤，已完成部分保留）。\n已提交的代码不受影响，发送 {engine_cmd} 可重新开始。",
    "card_btn_confirm_retry_title": "重新执行任务",
    "card_btn_confirm_retry_body": "确认重新执行？上次执行的结果将被替换",
    "card_btn_confirm_default_title": "确认操作",
    "btn_confirm_template": "确认「{text}」？",
    "card_stop_escalation_hint": "任务未响应停止信号时将出现强制停止选项",
    "card_stop_escalation_countdown": "约半分钟后可强制停止",
    "card_session_toast_cancelled": "{engine_name}任务已取消，发送 {engine_cmd} 重新启动",
    "card_session_rejected_notice": "⚠️ 当前系统繁忙，请稍等片刻后重试，或重新发送 {engine_cmd}",
    "card_session_rejected_shutdown": "⚠️ 系统维护中，请稍后再试",
    "card_session_rejected_banner": "系统繁忙，请稍后重试",
    "card_session_toast_btn_completed": "{engine_name}任务已完成，按钮不再可用。发送 {engine_cmd} 可开始新任务",
    "card_session_toast_btn_ttl_expired": "{engine_name}已超时关闭，按钮不再可用。重新发送 {engine_cmd} 可开始新任务",
    # --- Footer ---
    "card_footer_duration_fmt": "耗时 {duration}",
    "card_footer_elapsed_total_fmt": "已执行 {duration}",
    "card_footer_system_notice": "（系统提示）",
    "card_footer_idle_timeout_hint": "长时间无更新超过 {timeout_display} 后才会回收；执行中任务会自动顺延",
    "duration_sub_second": "< 1 秒",
    "duration_secs": "{seconds} 秒",
    "duration_mins_secs": "{minutes} 分钟 {seconds} 秒",
    "duration_hours_mins_secs": "{hours} 小时 {minutes} 分钟 {seconds} 秒",
    # --- Progress semantic label ---
    "card_progress_criteria_label": "验收",
    "card_progress_tool_label": "工具进度",
    # --- Criteria panel ---
    "criteria_panel_header": "📋 **验收标准**",
    "criteria_panel_header_with_count": "📋 **验收标准** ({satisfied}/{total})",
    # --- Card session retry ---
    "card_session_retry_pending_banner": "更新中，请稍候",
    # --- Lock undo window ---
    "lock_undo_window_duration": "{minutes} 分钟",
    "lock_undo_window_closed": "撤销窗口已关闭",
    # --- Content error ---
    "card_content_load_error": "部分内容暂时无法渲染，请稍后重试或联系管理员。",
    "card_content_load_error_running": "部分内容暂时无法渲染，任务仍在运行中，后续更新将自动恢复。",
    # --- New keys for review feedback ---
    "card_lifecycle_blocked_reason_fmt": "🚫 **任务阻塞** — {reason}",
    "card_completed_cta_fmt": "发送 /{engine_cmd} 启动新任务",
    "card_project_failed": "项目未能成功完成",
    # --- Tool panel labels ---
    "tool_label_command": "**命令**",
    "tool_label_result": "**结果**",
    "tool_label_input": "**输入**",
    "tool_label_progress": "**过程**",
    "tool_label_output": "**输出**",
    # --- Phase display names ---
    "phase_planning": "规划中",
    "phase_coding": "编码中",
    "phase_review": "审查中",
    "phase_spec_review": "规格审查中",
    "phase_spec": "规格定义中",
    "phase_plan": "方案规划中",
    "phase_task": "任务分解中",
    "phase_build": "执行构建中",
    "spec_review_phase_done": "多角色审查完成",
    "phase_testing": "测试中",
    "phase_building": "构建中",
    "phase_deploying": "部署中",
    "phase_analyzing": "分析中",
    "phase_refactoring": "重构中",
    "phase_debugging": "调试中",
    "phase_default": "进行中",
    # --- Reasoning panel ---
    "reasoning_panel_thinking": "💭 **正在分析…**",
    "reasoning_panel_done": "💭 **过程摘要** · {char_count}字",
    # --- Programming execution flow ---
    "execution_current_analysis": "⏳ **正在分析** · {summary}",
    "execution_current_tool": "⏳ **正在调用 {tool_name}**{suffix}",
    "execution_history_title": "📋 **执行记录** · {count} 步",
    "execution_history_folded": "… 较早 {count} 步已折叠",
    # --- Truncation (extracted from payload_truncator.py) ---
    "truncation_suffix": "…(已截断)",
    "truncation_warning_with_cmd": "⚠️ 内容过长已自动截断。请发送 {cmd_hint} 查看完整内容",
    "truncation_warning_generic": "⚠️ 内容过长已自动截断。可重新发送命令获取最新状态",
    "truncation_fallback_prefix": "内容过长无法完整展示。",
    "truncation_fallback_suffix": "\n\n…(内容过长已截断)",
    "truncation_card_header": "⚠️ 卡片过大",
    # --- Session inline text (extracted from session.py) ---
    "card_session_paused_notice": "状态更新暂停，任务仍在运行",
    "card_session_fallback_cmd": "/help 查看可用命令",
    "card_session_fallback_engine_name": "任务",
    # --- Hook failure notice (extracted from hooks.py) ---
    "hook_persistence_failed_notice": "⚠️ 执行结果保存失败，您可重新运行 {engine_cmd} 查看最新结果",
    # --- Session rotation truncation ---
    "card_session_rotation_truncated": "⚠️ 这是最后一张卡片，内容将精简显示。后续输出合并于此，超出部分可能被截断。{cmd_hint}",
    # --- Task orchestrator ---
    "orch_plan_archived": "✅ 规划完成 — 共 {task_count} 个任务\n{task_list}",
    "orch_continuation_link": "↓ 第 {seq} 篇 · [{link_text} →](lark://message/{msg_id})",
    "orch_continuation_link_fallback": "↓ 第 {seq} 篇 · 请查看下方最新卡片",
    "orch_continuation_hint": "↩ 第 {page} 篇 · 承接前篇",
    "orch_task_continuation_nav": "\n\n---\n↓ 任务「{task_name}」第 {page} 篇 · [{link_text} →](lark://message/{msg_id})",
    "orch_task_continuation_nav_fallback": "\n\n---\n↓ 任务「{task_name}」第 {page} 篇 · 请查看下方最新卡片",
    "orch_fallback_warning": "⚠️ 任务已合并展示（可能由网络波动引起，不影响执行结果；如持续出现请联系管理员）",
    "orch_flood_merged": "📎 任务「{task_name}」已合并展示于此",
    "orch_archived_navigate_fallback": "↓ 内容已续至下方卡片，请查看下方最新卡片",
    "orch_plan_transition_hint": "以下每个任务将在独立卡片中展示 · [查看首个任务 →](lark://message/{first_task_msg_id})",
    "orch_plan_transition_hint_no_link": "以下每个任务将在独立卡片中展示 ↓",
    "orch_plan_transition_hint_overflow": "以下 {independent_count} 个任务将在独立卡片中展示，其余 {merged_count} 个合并至最后一张 ↓",
    "orch_back_link": "↑ [前篇](lark://message/{msg_id})",
    "orch_back_link_with_task": "↑ [「{task_name}」前篇](lark://message/{msg_id})",
    "orch_nav_link_text": "查看最新卡片",
    "orch_overflow_separator": "---\n📎 {status_emoji} **{task_name}**\n",
    "orch_overflow_separator_first": "---\n📌 以下任务合并展示\n\n---\n📎 {status_emoji} **{task_name}**\n",
    "orch_overflow_collapsed": "---\n📌 …及 {count} 项更多任务合并展示\n",
    "orch_rotation_failed_notice": "⚠️ 内容过长无法创建新卡片，后续内容将继续在本卡片追加",
    "orch_final_summary_task_name": "任务总结",
    # --- Task status prefixes ---
    "orch_task_status_pending": "⏳",
    "orch_task_status_running": "🔄",
    "orch_task_status_done": "✅",
    "orch_task_status_failed": "❌",
    # --- Renderer inline text ---
    "deep_exec_start": "🚀 ACP Deep 执行开始\n\n📂 **{project_name}**\n🔗 路径: `{root_path}`",
    "deep_phase_executing": "🔄 执行中",
    "deep_phase_planning": "🧠 分析/规划中",
    "deep_main_task": "分析与执行主流程",
    "deep_spec_style_subtitle_analyzing": "Deep · 分析",
    "deep_spec_style_subtitle_build": "Deep · 执行",
    "deep_spec_style_analyzing_done": "分析与任务拆解已完成。",
    "deep_spec_style_build_done": "执行阶段已完成。",
    "deep_exec_completed": "✅ 执行完成 · 工具调用: {tool_calls_count}",
    "deep_exec_incomplete": "⚠️ 执行未完成（已完成 {completed}/{total} 步）\n自动恢复次数已耗尽，任务已明确失败。",
    "deep_progress_executing": "执行中",
    "deep_progress_done": "已完成",
    "spec_cycle_label": "第 {cycle_num} 轮",
    "spec_build_progress": "🔨 {tool_count} 次工具调用",
    "spec_build_progress_files": " · {file_count} 文件",
    "spec_build_heartbeat_thinking": " · 思考中 ({seconds}s)",
    "spec_build_heartbeat_tool_running": " · 执行中 ({seconds}s)",
    "spec_build_heartbeat_idle": "🔨 正在编码 · 思考中 ({seconds}s)",
    "spec_build_done": "🔨 **构建完成**  {summary}",
    "spec_build_done_plain": "🔨 **构建完成**",
    # Workflow header colors by phase
    "workflow_header_colors": {
        "agent_select": "blue",      # Orchestrator agent selection phase
        "tool_select": "turquoise",   # Initial tool selection phase
        "generating": "blue",         # Script generation in progress
        "confirm": "turquoise",       # Confirmation awaiting user
        "running": "blue",            # Execution in progress
        "error": "red",               # Error state
        "completed": "green",         # Successfully completed
    },
    # --- Workflow confirm dialogs ---
    "workflow_btn_confirm_stop_title": "确认停止 Workflow？",
    "workflow_btn_confirm_stop_body": "正在执行的步骤将中断，已完成的部分不受影响。",
    # --- Workflow error cards ---
    "workflow_error_session_expired_title": "会话已过期",
    "workflow_error_session_expired_body": "当前 Workflow 会话已超时失效。\n\n💡 恢复方式：发送 `/wf <需求>` 重新启动",
    "workflow_error_invalid_state_title": "状态不匹配",
    "workflow_error_invalid_state_body": "当前操作与 Workflow 状态不一致。\n\n💡 恢复方式：发送 `/wf_status` 查看最新状态后重试",
    "workflow_error_invalid_argument_title": "参数错误",
    "workflow_error_invalid_argument_body": "请求参数校验失败：{detail}\n\n💡 请检查输入后重试",
    "workflow_error_forbidden_title": "无操作权限",
    "workflow_error_forbidden_body": "只有 Workflow 发起者或管理员可以执行此操作。\n\n💡 如需操作请联系发起者或管理员",
    "workflow_error_runtime_timeout_title": "Workflow 执行超时",
    "workflow_error_runtime_timeout_body": "Workflow 已达到执行时间上限。\n\n🔎 细节：{detail}\n\n💡 恢复方式：缩小任务范围、减少并行 Agent 数量，或重新发起 `/wf` 让编排器生成更短超时和 fallback 的动态流程",
    "workflow_error_review_failed_title": "独立评审失败",
    "workflow_error_review_failed_body": "一个或多个独立评审 Agent 未正常结束，因此 Workflow 按安全策略未发布成功结果。\n\n🔎 细节：{detail}\n\n💡 可重新发起 `/wf`，或改用当前可用的评审 Agent",
    "workflow_error_internal_error_title": "Workflow 未完成",
    "workflow_error_internal_error_body": "Workflow 已明确结束，但未产出可验证结果。\n\n💡 直接重新发起 `/wf <需求>`；系统会使用当前可用的池内 Agent 自动编排并执行。",
}

# Merge spec-engine UI text from the shared utils layer (avoids card→engine reverse dep)
from ..utils.ui_text import SPEC_UI_TEXT  # noqa: E402

_spec_overlap = UI_TEXT.keys() & SPEC_UI_TEXT.keys()
if _spec_overlap:
    raise RuntimeError(f"UI_TEXT key conflict with SPEC_UI_TEXT: {_spec_overlap}")
UI_TEXT.update(SPEC_UI_TEXT)

# Merge lock-related UI text from dedicated module
from .styles_lock import LOCK_UI_TEXT  # noqa: E402

_lock_overlap = UI_TEXT.keys() & LOCK_UI_TEXT.keys()
if _lock_overlap:
    raise RuntimeError(f"UI_TEXT key conflict with LOCK_UI_TEXT: {_lock_overlap}")
UI_TEXT.update(LOCK_UI_TEXT)

# Dynamically generate TTL expired commands list from engine metadata (single source of truth)
from .engine_meta import ENGINE_CMD_MAP  # noqa: E402

_cmds = list(ENGINE_CMD_MAP.values())
UI_TEXT["card_session_ttl_expired_commands"] = (
    "、".join(_cmds[:-1]) + " 或 " + _cmds[-1] if len(_cmds) > 1 else _cmds[0]
)

# Keep mutable reference for testing (mock.patch.dict targets this)
_MUTABLE_UI_TEXT = UI_TEXT

# Freeze UI_TEXT as read-only to prevent accidental runtime mutation
UI_TEXT = types.MappingProxyType(UI_TEXT)  # type: ignore[assignment]
