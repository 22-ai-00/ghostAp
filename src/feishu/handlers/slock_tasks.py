"""Slock task management, memory, discussion, and council handlers.

Extracted from slock.py to reduce handler size. Contains /task, /memory,
/discuss, /council commands and their callback handlers.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Optional

from ...card.shared import build_responsive_layout
from ...slock_engine.slash_commands import is_slock_command
from ...utils.errors import safe_error_message
from ...utils.redact import redact_sensitive
from ..user_cache import resolve_display_name_nonblocking
from .base import CardActionContext

if TYPE_CHECKING:
    from ...project import ProjectContext

logger = logging.getLogger(__name__)


class SlockTaskMixin:
    """Mixin providing task/memory/discussion/council handlers."""

    def list_tasks(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """List all tasks in the current slock session."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("📋 任务列表", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        tasks = engine.tasks
        if not tasks:
            from ...slock_engine.card_templates.common import build_callback_button, build_empty_state_card
            guide_btn = build_callback_button(
                "➕ 分配任务", "slock_assign_task_hint",
                channel_id=chat_id, button_type="primary",
            )
            card = build_empty_state_card(
                "📋 任务列表",
                "当前没有任务。\n\n发送 `/task assign <任务描述>` 或点击下方按钮分配任务。",
                guide_buttons=[guide_btn],
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates import build_task_board_card

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        agents = engine.registry.list_agents(channel_id=channel_id)
        team_name = engine.channel.team_name if engine.channel else ""
        card = build_task_board_card(engine.tasks, agents, team_name=team_name, channel_id=channel_id)
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    def assign_task(
        self,
        message_id: str,
        chat_id: str,
        content: str = "",
        role_name: str = "",
        project: Optional["ProjectContext"] = None,
    ):
        """Assign a task to a specific role and execute it.

        If role_name is provided: claim → execute → complete/rollback.
        If no role_name: create task only (unassigned).
        """
        if not content:
            self.reply_text(message_id, "请提供任务内容\n\n用法: `/task assign <任务内容> <角色名称>`")
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self.reply_text(message_id, "请先激活 Slock 模式: `/slock`")
            return

        # Rate-limit check for non-admin/non-owner users
        if not self._check_assign_rate_limit(engine, message_id, chat_id):
            return

        task = engine.add_task(content)
        if task is None:
            self.reply_text(
                message_id,
                "❌ 任务创建失败（任务队列已满或内部错误），请稍后重试",
            )
            return

        if role_name:
            channel_id = engine.channel.channel_id if engine.channel else chat_id
            agent = engine.registry.find_by_name(role_name, channel_id=channel_id)
            if not agent:
                self.reply_text(
                    message_id,
                    f"⚠️ 任务已创建但未找到角色 **{redact_sensitive(role_name)}**\n"
                    f"• 任务 ID: `{task.task_id[:8]}`\n"
                    f"• 发送 `/role list` 查看可用角色",
                )
                return

            # Claim the task
            if not engine.claim_task(task.task_id, agent.agent_id):
                self.reply_text(message_id, f"❌ 任务 claim 失败，{agent.name} 可能正在执行其他任务")
                return

            # Submit async execution — send placeholder, update on completion
            self._submit_task_execution(
                engine, task, agent, message_id, chat_id, content, project
            )
            return

        # No role specified — use skill-based automatic assignment.
        channel_id = engine.channel.channel_id if engine.channel else chat_id

        # Auto-detect multi-role collaboration: if a chain template matches,
        # create a collaboration plan instead of single-agent execution.
        chain = engine._chain_manager.find_chain_for_task(content)
        if chain and len(chain.roles) > 1:
            plan = engine._collaboration_orchestrator.create_plan(task, channel_id)
            if plan:
                from ...slock_engine.card_templates.progress import build_collaboration_plan_card
                agents = list(engine.registry.list_agents(channel_id=channel_id))
                card = build_collaboration_plan_card(plan, agents, channel_id=channel_id)
                card_msg_id = self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
                # Register overview message_id for in-place card updates
                if card_msg_id and hasattr(engine, '_progress_tracker'):
                    engine._progress_tracker.set_overview_message_id(plan.plan_id, card_msg_id)
                # Confirm delivery — reset auto-start timer from this point
                engine._collaboration_orchestrator.confirm_plan_delivery(plan.plan_id)
                return

        agents = list(engine.registry.list_agents(channel_id=channel_id))
        if not agents:
            self.reply_text(
                message_id,
                f"✅ 任务已创建（等待分配）\n"
                f"• ID: `{task.task_id[:8]}`\n"
                f"• 内容: {redact_sensitive(content[:80])}\n"
                f"• 当前没有可用角色，发送 `/new-role <名称>` 创建角色",
            )
            return

        rank_agents = getattr(engine.router, "rank_agents_for_claim", None)
        ranked_agents = rank_agents(content, agents) if callable(rank_agents) else []
        if not isinstance(ranked_agents, list):
            selected = engine.router.route_message(content, agents)
            ranked_agents = [selected] if selected else []
        if not ranked_agents:
            self.reply_text(
                message_id,
                f"✅ 任务已创建（等待分配）\n"
                f"• ID: `{task.task_id[:8]}`\n"
                f"• 内容: {redact_sensitive(content[:80])}\n"
                f"• 暂无匹配角色，发送 `/role list` 查看可用角色",
            )
            return

        claimed_agent = None
        for candidate in ranked_agents:
            if engine.claim_task(task.task_id, candidate.agent_id):
                claimed_agent = candidate
                break

        if claimed_agent is None:
            self.reply_text(message_id, "❌ 任务 claim 失败，所有匹配角色可能都在执行其他任务")
            return

        self._submit_task_execution(
            engine, task, claimed_agent, message_id, chat_id, content, project, auto_routed=True
        )

    def _submit_task_execution(
        self,
        engine,
        task,
        agent,
        message_id: str,
        chat_id: str,
        content: str,
        project: Optional["ProjectContext"],
        *,
        auto_routed: bool = False,
    ) -> None:
        """Submit task execution to the engine's thread pool asynchronously.

        Sends a placeholder card immediately, then updates it with the result
        or error when execution completes.
        """
        prefix = "自动分配给" if auto_routed else "已分配给"

        def _execute():
            callbacks = self._create_callbacks(message_id, chat_id, project, engine.engine_name, engine.root_path)
            return engine.execute_task(task.task_id, agent.agent_id, callbacks)

        def _result_card(result: str, duration: float) -> str:
            try:
                engine.memory.write_agent_reasoning_snapshot(
                    agent.agent_id,
                    task.task_id,
                    prompt_summary=content[:1000],
                    result_summary=result[:2000],
                    tool_name=agent.agent_type,
                    model_name=agent.model_name,
                )
            except Exception:
                logger.warning(
                    "Failed to persist Slock reasoning snapshot for task %s agent %s",
                    task.task_id,
                    agent.agent_id,
                    exc_info=True,
                )
            card_data = engine._mouthpiece.format_card(
                agent,
                result,
                model_info=agent.agent_type,
                duration_s=duration,
                channel_id=chat_id,
                task_id=task.task_id,
            )
            return json.dumps(card_data, ensure_ascii=False)

        def _error_card(exc: Exception) -> str:
            return json.dumps({
                "schema": "2.0",
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "❌ 任务执行失败"}, "template": "red"},
                "body": {"elements": [{"tag": "markdown", "content": (
                    f"Agent: {agent.emoji} {agent.name}\n"
                    f"任务: {content[:60]}\n"
                    f"错误: {safe_error_message(exc)}\n\n"
                    "任务已回退为 TODO，可重新分配"
                )}]},
            }, ensure_ascii=False)

        def _empty_card() -> str:
            return json.dumps({
                "schema": "2.0",
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "⚠️ 任务完成无输出"}, "template": "orange"},
                "body": {"elements": [{"tag": "markdown", "content": (
                    f"Agent: {agent.emoji} {agent.name}\n"
                    f"任务: {content[:60]}\n\n"
                    "任务已回退为 TODO，可重新分配"
                )}]},
            }, ensure_ascii=False)

        def _busy_card() -> str:
            return json.dumps({
                "schema": "2.0",
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "⚠️ 团队繁忙"}, "template": "orange"},
                "body": {"elements": [{"tag": "markdown", "content": "当前所有角色均在忙碌中，请稍后重试。"}]},
            }, ensure_ascii=False)

        placeholder_card = json.dumps({
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": f"⏳ 任务{prefix} {agent.emoji} {agent.name}"},
                "template": "indigo",
            },
            "body": {"elements": [{"tag": "markdown", "content": (
                f"**任务**: {content[:80]}\n"
                f"**ID**: `{task.task_id[:8]}`\n\n"
                "Agent 正在执行中..."
            )}]},
        }, ensure_ascii=False)

        self._execute_async(
            engine=engine,
            execute_fn=_execute,
            placeholder_card_json=placeholder_card,
            result_card_fn=_result_card,
            error_card_fn=_error_card,
            empty_card_fn=_empty_card,
            busy_card_fn=_busy_card,
            message_id=message_id,
            chat_id=chat_id,
        )

    def show_task_status(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Show task board with status summary."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("📊 任务状态", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates import build_task_board_card

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        agents = engine.registry.list_agents(channel_id=channel_id)
        team_name = engine.channel.team_name if engine.channel else ""
        card = build_task_board_card(engine.tasks, agents, team_name=team_name, channel_id=channel_id)
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def show_memory_group(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Show L2 shared group memory for the current slock team."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("🧠 群组记忆", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        group_memory = engine.memory.read_group_memory(channel_id)
        if not group_memory:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card(
                "🧠 群组共享记忆",
                "当前群组暂无共享记忆。\n\nAgent 协作过程中会自动积累群组知识。",
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates.memory import build_memory_group_card

        # Parse group memory text into structured memory_items
        lines = [line.strip() for line in group_memory.strip().split("\n") if line.strip()]
        memory_items: list[dict] = []
        current_category = "context"
        for line in lines[:50]:  # NFR-6: limit to 50 items per read
            if line.startswith("## ") or line.startswith("# "):
                current_category = line.lstrip("#").strip().lower() or "context"
            else:
                memory_items.append({
                    "category": current_category,
                    "content": line[:200],
                    "timestamp": None,
                })

        total_lines = len(lines)
        team_name = engine.channel.team_name if engine.channel else "群组"

        card = build_memory_group_card(
            agent_name=f"{team_name} (共享)",
            agent_emoji="🧠",
            memory_items=memory_items,
            channel_id=channel_id,
        )

        # Inject meta-info row at the top of elements (after header)
        truncated = total_lines > 50
        meta_text = f"📊 共 {total_lines} 条记录"
        if truncated:
            meta_text += " | ⚠️ 内容已截断，仅显示前 50 条"
        if card.get("body", {}).get("elements"):
            card["body"]["elements"].insert(0, {"tag": "markdown", "content": meta_text})

        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Agent Memory (per-agent L1 memory)
    # ------------------------------------------------------------------

    def show_agent_memory(self, message_id: str, chat_id: str, agent_name: str = "", project: Optional["ProjectContext"] = None):
        """Show L1 memory for a specific agent."""
        if not agent_name:
            from ...slock_engine.card_templates.common import build_usage_hint_card
            card = build_usage_hint_card(
                "/memory <agent_name>",
                ["/memory coder", "/memory list"],
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("🧠 角色记忆", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        agent = engine.registry.find_by_name(agent_name, channel_id=channel_id)
        if not agent:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card("🧠 角色记忆", f"未找到角色 `{agent_name}`")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates.memory import build_memory_group_card

        memory = engine.memory.read_agent_memory(agent.agent_id)
        memory_items: list[dict] = []

        # Add key_knowledge entries
        if memory.key_knowledge:
            for line in memory.key_knowledge.strip().split("\n"):
                line = line.strip()
                if line:
                    memory_items.append({"category": "key_knowledge", "content": line[:200], "timestamp": None})

        # Add active_context entries
        if memory.active_context:
            for line in memory.active_context.strip().split("\n"):
                line = line.strip()
                if line:
                    memory_items.append({"category": "context", "content": line[:200], "timestamp": None})

        # Add role definition
        if memory.role:
            memory_items.append({"category": "role", "content": memory.role[:200], "timestamp": None})

        if not memory_items:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card("🧠 角色记忆", f"{agent.display_name} 暂无记忆数据")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        card = build_memory_group_card(
            agent_name=agent.name,
            agent_emoji=agent.emoji,
            memory_items=memory_items,
            channel_id=channel_id,
            agent_id=agent.agent_id,
        )
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    def show_memory_list(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Show memory summary for all agents in the channel."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("🧠 记忆列表", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        agents = engine.registry.list_agents(channel_id=channel_id)

        if not agents:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card("🧠 记忆列表", "当前团队暂无角色")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates.common import build_card_wrapper

        rows: list[str] = []
        for agent in agents:
            memory = engine.memory.read_agent_memory(agent.agent_id)
            item_count = 0
            if memory.key_knowledge:
                item_count += len([line for line in memory.key_knowledge.split("\n") if line.strip()])
            if memory.active_context:
                item_count += len([line for line in memory.active_context.split("\n") if line.strip()])
            rows.append(f"| {agent.emoji} | {agent.name} | {item_count} 条 |")

        table_md = "| 头像 | 角色 | 记忆条数 |\n| --- | --- | --- |\n" + "\n".join(rows)
        elements: list[dict] = [{"tag": "markdown", "content": table_md}]

        card = build_card_wrapper(
            header_title="🧠 团队记忆概览",
            header_template="turquoise",
            elements=elements,
        )
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Discussion management (stop/history/list)
    # ------------------------------------------------------------------

    def stop_discussion(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Stop the active discussion in the current channel."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("💬 停止讨论", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        channel_id = engine.channel.channel_id if engine.channel else chat_id

        # Find active discussion threads
        active_thread = None
        with engine._discussions_lock:
            threads = engine._active_discussions.get(channel_id, [])
            if threads:
                active_thread = threads[0]

        if not active_thread:
            self.reply_text(message_id, "💬 当前无进行中的讨论")
            return

        # Stop the discussion via DiscussionManager

        if engine._discussion_manager is not None:
            stopped_thread = engine._discussion_manager.stop_discussion(active_thread)
            engine._remove_discussion(channel_id, stopped_thread.thread_id)

            from ...slock_engine.card_templates.discussion import build_discussion_conclusion_card
            card = build_discussion_conclusion_card(
                thread_id=stopped_thread.thread_id,
                participants=getattr(stopped_thread, "participants", []),
                conclusion=getattr(stopped_thread, "conclusion", "") or "讨论已手动停止",
                total_rounds=getattr(stopped_thread, "current_round", 0),
                total_tokens=getattr(stopped_thread, "total_tokens", 0),
                status="manually_stopped",
                channel_id=channel_id,
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
        else:
            self.reply_text(message_id, "💬 讨论管理器未初始化，无法停止讨论")

    def show_discussion_history(self, message_id: str, chat_id: str, thread_id: str = "", project: Optional["ProjectContext"] = None):
        """Show discussion history for a specific thread."""
        if not thread_id:
            from ...slock_engine.card_templates.common import build_usage_hint_card
            card = build_usage_hint_card(
                "/discussion history <thread_id>",
                ["/discussion list", "/discussion history abc123"],
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("💬 讨论历史", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        channel_id = engine.channel.channel_id if engine.channel else chat_id

        # Try to find in active discussions first
        thread = engine.find_active_discussion(channel_id, thread_id)

        # Try loading from persisted discussions if not active
        if not thread and engine._discussion_manager is not None:
            persisted = engine._discussion_manager.load_discussions(channel_id)
            for t in persisted:
                if getattr(t, "thread_id", "") == thread_id:
                    thread = t
                    break

        if not thread:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card("💬 讨论历史", f"未找到讨论 `{thread_id[:12]}`")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates import build_discussion_history_card

        # Convert thread to history format expected by the card builder
        messages = getattr(thread, "messages", [])
        history_entry = {
            "topic_hash": getattr(thread, "thread_id", "")[:8],
            "title": getattr(thread, "topic", "") or "讨论",
            "participants": getattr(thread, "participants", []),
            "time": getattr(thread, "created_at", 0),
            "conclusion": getattr(thread, "conclusion", "") or "",
            "messages": [
                {"speaker": getattr(m, "speaker", ""), "content": getattr(m, "content", "")[:200]}
                for m in messages[-20:]
            ],
        }
        card = build_discussion_history_card(history=[history_entry], channel_id=channel_id)
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    def list_discussions(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """List all discussions (active + recent persisted) in the channel."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("💬 讨论列表", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        channel_id = engine.channel.channel_id if engine.channel else chat_id

        # Collect active threads
        all_threads = []
        with engine._discussions_lock:
            active = engine._active_discussions.get(channel_id, [])
            all_threads.extend(active)

        # Load persisted threads
        if engine._discussion_manager is not None:
            persisted = engine._discussion_manager.load_discussions(channel_id)
            existing_ids = {getattr(t, "thread_id", "") for t in all_threads}
            for t in persisted:
                if getattr(t, "thread_id", "") not in existing_ids:
                    all_threads.append(t)

        if not all_threads:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card("💬 讨论列表", "当前暂无讨论记录")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        import time as _time

        from ...slock_engine.card_templates.common import build_card_wrapper

        rows: list[str] = []
        for thread in all_threads[:20]:
            tid = getattr(thread, "thread_id", "?")[:8]
            status = getattr(thread, "status", "unknown")
            status_label = status.value if hasattr(status, "value") else str(status)
            participants = getattr(thread, "participants", [])
            p_names = ", ".join(participants[:3]) if participants else "-"
            created = getattr(thread, "created_at", 0)
            ts = _time.strftime("%m-%d %H:%M", _time.localtime(created)) if created else "-"
            rows.append(f"| `{tid}` | {status_label} | {p_names} | {ts} |")

        table_md = "| ID | 状态 | 参与者 | 创建时间 |\n| --- | --- | --- | --- |\n" + "\n".join(rows)
        elements: list[dict] = [{"tag": "markdown", "content": table_md}]

        card = build_card_wrapper(
            header_title="💬 讨论列表",
            header_template="blue",
            elements=elements,
        )
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Plan management
    # ------------------------------------------------------------------

    def list_plans(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """List all collaboration plans in the current slock session."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("📋 协作计划", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates.progress import build_progress_overview_card

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        plans = engine.collaboration_orchestrator.list_active_plans(channel_id=channel_id)
        agents = engine.registry.list_agents(channel_id=channel_id)
        if not plans:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card(
                "📋 协作计划",
                "当前没有协作计划。\n\n任务创建后会自动触发协作规划。",
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        card = build_progress_overview_card(
            plans=plans,
            agents=agents,
            team_name=engine.channel.team_name if engine.channel else "",
            channel_id=channel_id,
        )
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    def show_plan_detail(self, message_id: str, chat_id: str, plan_id: str = "", project: Optional["ProjectContext"] = None):
        """Show detailed view of a specific collaboration plan."""
        if not plan_id:
            from ...slock_engine.card_templates.common import build_usage_hint_card
            card = build_usage_hint_card(
                "/plan <plan_id>",
                ["/plan list", "/plan abc123"],
            )
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            from ...slock_engine.card_templates.common import build_error_state_card
            card = build_error_state_card("📋 协作计划", "当前没有活跃的 Slock 团队，请先 /slock activate")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        from ...slock_engine.card_templates.progress import build_task_overview_card
        from ...slock_engine.models import AgentStatus as AgentStatusEnum

        channel_id = engine.channel.channel_id if engine.channel else chat_id
        plan = engine.collaboration_orchestrator.get_plan(plan_id)
        if not plan:
            from ...slock_engine.card_templates.common import build_empty_state_card
            card = build_empty_state_card("📋 协作计划", f"未找到计划 `{plan_id[:12]}`，可能已结束或不存在")
            self.reply_card(message_id, json.dumps(card, ensure_ascii=False))
            return

        # Build agents with status for the task overview card
        agents: list[tuple] = []
        for agent in engine.registry.list_agents(channel_id=channel_id):
            status = engine.get_agent_status(agent.agent_id) or AgentStatusEnum.IDLE
            agents.append((agent, status))

        # Extract latest output from the last completed step
        latest_output_summary = ""
        if plan.steps:
            completed_steps = [s for s in plan.steps if s.get("status") == "DONE"]
            if completed_steps:
                last_step = completed_steps[-1]
                latest_output_summary = last_step.get("output_summary", "")[:200]

        # Gather discussion entries (last 5) from active discussions
        discussion_entries: list[dict] = []
        try:
            with engine._discussions_lock:
                active_threads = engine._active_discussions.get(channel_id, [])
            for thread in active_threads[:5]:
                for msg in getattr(thread, "messages", [])[-5:]:
                    discussion_entries.append({
                        "speaker": getattr(msg, "speaker", ""),
                        "content": getattr(msg, "content", "")[:100],
                        "timestamp": getattr(msg, "timestamp", 0),
                    })
        except Exception:
            pass
        discussion_entries = discussion_entries[-5:]

        # Gather timeline events (last 10) from plan step history
        timeline_events: list[dict] = []
        if plan.steps:
            for step in plan.steps:
                if step.get("started_at"):
                    timeline_events.append({
                        "event_type": "step_started",
                        "agent_id": step.get("agent_id", ""),
                        "timestamp": step.get("started_at", 0),
                        "detail": step.get("name", ""),
                    })
                if step.get("completed_at"):
                    timeline_events.append({
                        "event_type": "step_completed",
                        "agent_id": step.get("agent_id", ""),
                        "timestamp": step.get("completed_at", 0),
                        "detail": step.get("name", ""),
                    })
        timeline_events.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        timeline_events = timeline_events[:10]

        card = build_task_overview_card(
            plan=plan,
            agents=agents,
            channel_id=channel_id,
            latest_output_summary=latest_output_summary,
            discussion_entries=discussion_entries or None,
            timeline_events=timeline_events or None,
        )
        self.reply_card(message_id, json.dumps(card, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop_slock_engine(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Stop the slock engine, unregister managed chat, and clean up."""
        manager = self._get_engine_manager()

        # Deactivate the engine if it exists
        engine = manager.get_activated_engine(chat_id)
        if engine:
            if not self._check_slock_permission(engine, message_id, chat_id):
                return
            engine.deactivate()

        # Unregister managed chat so dispatcher stops routing to this engine
        manager.unregister_managed_chat(chat_id)

        self._safe_lifecycle_action(
            lambda: self._stop_engine_generic(message_id, chat_id, project),
            "stop", chat_id, message_id, project,
        )

    def _stop_single_agent(self, message_id: str, chat_id: str, value: dict):
        """Stop a single agent by agent_id from card action value.

        Falls back to full engine stop if agent_id is not provided (backwards compat).
        """
        agent_id = value.get("agent_id", "")

        if not agent_id:
            # Fallback: no agent_id means legacy card — stop the whole engine
            self.stop_slock_engine(message_id, chat_id)
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self.send_text_to_chat(chat_id, "⚠️ 当前群组未激活 Slock 模式。")
            return

        if not self._check_slock_permission(engine, message_id, chat_id):
            return

        stopped = engine.stop_agent(agent_id)
        if stopped:
            self.send_text_to_chat(chat_id, "⏹ Agent 已停止，状态已重置为 IDLE。")
        else:
            self.send_text_to_chat(chat_id, "⚠️ 未找到该 Agent 或其已处于空闲状态。")

    def _expand_discussion(self, chat_id: str, value: dict):
        """Show full discussion thread content in response to expand button."""
        thread_id = value.get("thread_id", "")
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self.send_text_to_chat(chat_id, "⚠️ 当前群组未激活 Slock 模式。")
            return

        thread = self._find_discussion_thread(engine, chat_id, thread_id)
        if not thread:
            self.send_text_to_chat(chat_id, "ℹ️ 讨论线程已结束或不存在。")
            return

        # Format all messages for display
        lines: list[str] = [f"💬 **讨论详情** (thread: `{thread_id[:12]}...`)\n"]
        for msg in thread.messages:
            lines.append(f"**{msg.sender_agent_id}** (R{msg.round_num}):\n{msg.content}\n")
        self.send_text_to_chat(chat_id, "\n".join(lines)[:4000])

    def _stop_discussion(self, chat_id: str, value: dict):
        """Manually stop an active discussion thread."""
        thread_id = value.get("thread_id", "")
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self.send_text_to_chat(chat_id, "⚠️ 当前群组未激活 Slock 模式。")
            return

        dm = getattr(engine, "_discussion_manager", None)
        if dm and thread_id:
            thread_obj = self._find_discussion_thread(engine, chat_id, thread_id)
            if thread_obj:
                dm.stop_discussion(thread_obj)
                self.send_text_to_chat(chat_id, "⏹ 讨论已手动终止。")
                return
        self.send_text_to_chat(chat_id, "ℹ️ 讨论线程已结束或不存在。")

    @staticmethod
    def _find_discussion_thread(engine, chat_id: str, thread_id: str):
        """Resolve an active discussion thread across real engine and old test-double shapes."""
        finder = getattr(engine, "find_active_discussion", None)
        if callable(finder):
            try:
                candidate = finder(chat_id, thread_id)
            except Exception:
                candidate = None
            if getattr(candidate, "thread_id", None) == thread_id:
                return candidate

        active_discussions = getattr(engine, "_active_discussions", {})
        candidates = active_discussions.get(chat_id, [])
        if not isinstance(candidates, list):
            candidates = [candidates]
        for candidate in candidates:
            if getattr(candidate, "thread_id", None) == thread_id:
                return candidate
        return None

    # ------------------------------------------------------------------
    # Card action handler
    # ------------------------------------------------------------------

    def _refresh_status_card(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Rebuild and update the status panel card in-place."""
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self.send_text_to_chat(chat_id, "⚠️ 当前群组未激活 Slock 模式，无法刷新状态。")
            return
        team_name = engine.channel.team_name if engine.channel else ""
        status_card = engine.get_status_card(team_name=team_name)
        card_content = json.dumps(status_card, ensure_ascii=False)
        self.update_card(message_id, card_content)

    def _refresh_task_board_card(self, message_id: str, chat_id: str, project: Optional["ProjectContext"] = None):
        """Rebuild and update the task board card in-place."""
        from ...slock_engine.card_templates import build_task_board_card

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            self.send_text_to_chat(chat_id, "⚠️ 当前群组未激活 Slock 模式，无法刷新任务看板。")
            return
        channel_id = engine.channel.channel_id if engine.channel else chat_id
        agents = engine.registry.list_agents(channel_id=channel_id)
        team_name = engine.channel.team_name if engine.channel else ""
        card = build_task_board_card(engine.tasks, agents, team_name=team_name, channel_id=channel_id)
        self.update_card(message_id, json.dumps(card, ensure_ascii=False))

    def _has_slock_permission(self, engine) -> bool:
        """Pure boolean check: is current operator admin or channel owner? No side effects.

        When ADMIN_USER_IDS is empty (bootstrap state), all users are permitted —
        consistent with the one-way /setadmin bootstrap contract.
        """
        from ...config import get_settings
        from ...thread.manager import get_current_sender_id

        operator_id = get_current_sender_id() or ""
        settings = get_settings()
        admin_ids = settings.admin_user_ids if hasattr(settings, "admin_user_ids") else frozenset()

        # Bootstrap state: no admin configured → permissive
        if not admin_ids:
            return True

        channel_owner_id = ""
        if engine.channel:
            channel_owner_id = getattr(engine.channel, "owner_id", "") or ""

        return bool(
            (operator_id and operator_id in admin_ids)
            or (operator_id and channel_owner_id and operator_id == channel_owner_id)
        )

    def _check_slock_permission(self, engine, message_id: str, chat_id: str) -> bool:
        """Check if current operator is admin or channel owner. Returns True if authorized."""
        if not self._has_slock_permission(engine):
            perm_msg = "⚠️ 权限不足，仅管理员或团队创建者可执行此操作。"
            if not self.reply_text(message_id, perm_msg):
                self.send_text_to_chat(chat_id, perm_msg)
            return False
        return True

    def _require_slock_permission(self, engine, action_type: str, *, allow_assignee: bool = False, task_id: str = "") -> dict | None:
        """Check permission and return rejection card if denied, None if allowed.

        Authorized: admin_user_ids | channel_owner | (task claimed_by if allow_assignee).
        When ADMIN_USER_IDS is empty (bootstrap state), all users are permitted.
        """
        from ...config import get_settings
        from ...thread.manager import get_current_sender_id

        settings = get_settings()
        sender_id = get_current_sender_id() or ""

        # Check admin
        admin_ids = settings.admin_user_ids if hasattr(settings, "admin_user_ids") else frozenset()

        # Bootstrap state: no admin configured → permissive
        if not admin_ids:
            return None  # allowed

        if sender_id and sender_id in admin_ids:
            return None  # allowed

        # Check channel owner
        if engine and engine.channel:
            if sender_id and sender_id == (getattr(engine.channel, "owner_id", "") or ""):
                return None  # allowed

        # Check task assignee
        if allow_assignee and task_id and engine:
            for task in engine.tasks:
                if task.task_id == task_id and task.claimed_by == sender_id:
                    return None  # allowed

        # Denied - return rejection toast
        return {"toast": {"type": "error", "content": "\u26d4 权限不足：仅管理员、群主或任务负责人可执行此操作"}}

    def _check_assign_rate_limit(self, engine, message_id: str, chat_id: str) -> bool:
        """Check rate-limit for task assignment. Admin/owner bypass. Returns True if allowed."""
        import time as _time

        from ...config import get_settings
        from ...thread.manager import get_current_sender_id

        operator_id = get_current_sender_id() or ""
        settings = get_settings()
        admin_ids = settings.admin_user_ids if hasattr(settings, "admin_user_ids") else frozenset()
        channel_owner_id = ""
        if engine.channel:
            channel_owner_id = getattr(engine.channel, "owner_id", "") or ""

        # Admin and owner bypass rate-limit
        is_privileged = (
            (operator_id and operator_id in admin_ids)
            or (operator_id and channel_owner_id and operator_id == channel_owner_id)
        )
        if is_privileged:
            return True

        # Rate-limit for regular users: sliding window of 60s
        rate_limit = settings.slock_assign_rate_limit
        tracker_key = f"{chat_id}:{operator_id}"
        now = _time.time()
        window = 60.0

        self._prune_assign_rate_limit_tracker(now, window)
        timestamps = self._rate_limit_tracker.get(tracker_key, [])

        if len(timestamps) >= rate_limit:
            self.reply_text(
                message_id,
                f"⚠️ 任务提交频率超限（每分钟最多 {rate_limit} 次），请稍后重试。",
            )
            self._rate_limit_tracker[tracker_key] = timestamps
            return False

        timestamps.append(now)
        self._rate_limit_tracker[tracker_key] = timestamps
        return True

    def _prune_assign_rate_limit_tracker(self, now: float, window: float) -> None:
        """Remove expired assign-rate-limit entries for inactive chat/sender pairs."""
        _GLOBAL_TTL = 3600.0  # Remove keys inactive for > 1 hour
        for key, timestamps in list(self._rate_limit_tracker.items()):
            if not timestamps or now - max(timestamps) > _GLOBAL_TTL:
                # Global TTL: no activity for over 1 hour, remove entirely
                self._rate_limit_tracker.pop(key, None)
                continue
            active = [ts for ts in timestamps if now - ts < window]
            if active:
                self._rate_limit_tracker[key] = active
            else:
                self._rate_limit_tracker.pop(key, None)

    def _check_discussion_permission(self, message_id: str, chat_id: str) -> bool:
        """Check permission for triggering a discussion.

        Admin/owner: always allowed (no rate-limit).
        Regular members: rate-limited to 3 discussion triggers per 5 minutes.
        Returns True if allowed, False if denied (with user notification).
        """
        import time as _time

        from ...config import get_settings
        from ...thread.manager import get_current_sender_id

        operator_id = get_current_sender_id() or ""
        settings = get_settings()
        admin_ids = settings.admin_user_ids if hasattr(settings, "admin_user_ids") else frozenset()

        # Try to get channel owner from engine
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        channel_owner_id = ""
        if engine and engine.channel:
            channel_owner_id = getattr(engine.channel, "owner_id", "") or ""

        # Admin and owner bypass rate-limit
        is_privileged = (
            (operator_id and operator_id in admin_ids)
            or (operator_id and channel_owner_id and operator_id == channel_owner_id)
        )
        if is_privileged:
            return True

        # Rate-limit for regular users: sliding window of 5 minutes, max 3 triggers
        DISCUSSION_WINDOW = 300.0  # 5 minutes
        DISCUSSION_MAX = 3
        tracker_key = f"disc:{chat_id}:{operator_id}"
        now = _time.time()

        timestamps = self._rate_limit_tracker.get(tracker_key, [])
        timestamps = [t for t in timestamps if now - t < DISCUSSION_WINDOW]

        if len(timestamps) >= DISCUSSION_MAX:
            self.reply_text(
                message_id,
                f"⚠️ 讨论触发频率超限（每 5 分钟最多 {DISCUSSION_MAX} 次），请稍后重试。",
            )
            self._rate_limit_tracker[tracker_key] = timestamps
            return False

        timestamps.append(now)
        self._rate_limit_tracker[tracker_key] = timestamps
        return True

    def _check_queue_wait_timeout(self, future, start_time: float, card_message_id: str, chat_id: str) -> bool:
        """Check if task waited too long in queue. Returns True if timed out (caller should abort).

        If future is None (e.g. synchronous executor in tests where the work runs
        before submit() returns), skip timeout detection entirely — the task is
        already executing, so queue-wait is irrelevant.
        """
        import json as _json
        import time

        from ...config import get_settings as _get_settings

        if future is None:
            return False

        _settings = _get_settings()
        _enqueue_time = getattr(future, "enqueue_time", None)
        if _enqueue_time is None:
            _enqueue_time = start_time
        enqueue_elapsed = time.time() - _enqueue_time
        if enqueue_elapsed > _settings.slock_queue_wait_timeout:
            logger.warning(
                "Slock queue wait timeout (%.1fs > %ds) for chat %s",
                enqueue_elapsed, _settings.slock_queue_wait_timeout, chat_id,
            )
            timeout_card = _json.dumps({
                "schema": "2.0",
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "⏱️ 排队超时"}, "template": "orange"},
                "body": {"elements": [{"tag": "markdown", "content": "消息在队列中等待过久，已自动取消。请稍后重试。"}]},
            }, ensure_ascii=False)
            if card_message_id:
                self.update_card(card_message_id, timeout_card)
            return True
        return False

    def _resolve_escalation(self, message_id: str, chat_id: str, value: dict):
        """Handle admin clicking Retry/Skip/Abort on an escalation card.

        Enhanced flow:
        (a) Get operator via thread-local
        (b) Permission check via _check_slock_permission
        (c) Resolution whitelist validation
        (d) Resolve via engine
        (e) Update card to resolved state
        (f) Fallback to text if card update fails
        (g) Handle duplicate clicks gracefully
        """
        from ...slock_engine.card_templates import build_resolved_escalation_card
        from ...thread.manager import get_current_sender_id

        escalation_id = value.get("escalation_id", "")
        resolution = value.get("resolution", "")

        if not escalation_id or not resolution:
            logger.warning("Escalation resolve missing params: escalation_id=%s resolution=%s", escalation_id, resolution)
            self.send_text_to_chat(chat_id, "⚠️ 升级请求处理参数缺失，请重试或联系管理员。")
            return

        # (a) Get operator identity
        operator_id = get_current_sender_id() or ""
        operator_display = (
            resolve_display_name_nonblocking(operator_id, self.ctx.api_client_factory)
            if operator_id
            else ""
        )

        # (b) Permission check — admin or team owner (unified)
        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(chat_id)
        if not engine:
            logger.warning("No active engine for escalation resolve in chat %s", chat_id)
            self.send_text_to_chat(chat_id, "⚠️ 当前群组未激活 Slock 模式，无法处理升级请求。")
            return

        if not self._check_slock_permission(engine, message_id, chat_id):
            return

        # (c) Resolution whitelist validation
        escalation = engine.get_escalation(escalation_id)
        if not escalation:
            self.send_text_to_chat(chat_id, f"⚠️ Escalation `{escalation_id}` 不存在。")
            return

        # Handle duplicate clicks — already resolved
        if escalation.resolved:
            self.send_text_to_chat(chat_id, f"ℹ️ 此升级请求已处理（{escalation.resolution}），无需重复操作。")
            return

        allowed_options = escalation.options or ["重试", "跳过", "中止"]
        resolution_stripped = resolution.strip()
        if resolution_stripped not in allowed_options:
            logger.warning(
                "Escalation resolve invalid resolution: '%s' not in %s",
                resolution, allowed_options,
            )
            self.send_text_to_chat(
                chat_id,
                f"⚠️ 无效的解决选项 `{resolution}`，允许的选项: {', '.join(allowed_options)}",
            )
            return

        # (d) Execute resolve
        resolved = engine.resolve_escalation(escalation_id, resolution_stripped)
        if not resolved:
            # Race condition: resolved between our check and this call
            self.send_text_to_chat(chat_id, "ℹ️ 此升级请求已处理，无需重复操作。")
            return

        # (e) Build resolved card and update in-place
        resolved_card = build_resolved_escalation_card(
            escalation,
            resolved_by=operator_display or operator_id,
            resolution=resolution_stripped,
            resolved_at=resolved.resolved_at,
            channel_id=chat_id,
        )
        card_json = json.dumps(resolved_card, ensure_ascii=False)
        card_updated = self.update_card(message_id, card_json)

        # (f) Fallback to text confirmation if card update fails
        if not card_updated:
            logger.error(
                "Failed to update escalation card: message_id=%s chat_id=%s",
                message_id, chat_id,
            )
            confirm_text = (
                f"✅ Escalation resolved: **{resolved.agent_name}** — {resolution_stripped}\n"
                f"Reason: {redact_sensitive(resolved.reason)}"
            )
            self.send_text_to_chat(chat_id, confirm_text)

        # (g) Trigger agent recovery based on resolution
        engine.resume_after_escalation(resolved)

    def _handle_clarify_confirm(self, open_message_id: str, open_chat_id: str, value: dict) -> None:
        """Handle "是，这是任务" button click from clarification card.

        Triggers task enqueue logic similar to auto-activate path.
        Validates that the action performer is the original message sender.
        """
        import json as _json

        from ...slock_engine.card_templates.queue_feedback import build_clarification_confirmed_card
        from ...thread.manager import get_current_sender_id

        message_preview = str(value.get("message_preview") or "")
        original_message_id = str(value.get("message_id") or "")
        original_sender_id = str(value.get("sender_id") or "")
        current_sender_id = get_current_sender_id() or ""

        # Sender verification: only original sender can confirm
        if original_sender_id and current_sender_id and original_sender_id != current_sender_id:
            logger.warning(
                "Clarify confirm rejected: sender mismatch (original=%s, current=%s)",
                original_sender_id, current_sender_id,
            )
            self.reply_text(open_message_id, "⚠️ 仅消息发送者可确认此操作。")
            return

        manager = self._get_engine_manager()
        engine = manager.get_activated_engine(open_chat_id)

        # Build confirmation card first (will be shown regardless of path)
        confirm_card = build_clarification_confirmed_card(message_preview=message_preview)
        confirm_card_json = _json.dumps(confirm_card, ensure_ascii=False)

        if engine:
            # Engine already active — route the message to the engine
            logger.info(
                "Clarify confirm: engine already active, routing message chat=%s preview=%s",
                open_chat_id, message_preview[:50],
            )
            # Update card first to show confirmation
            self.update_card(open_message_id, confirm_card_json)
            # Then route the message to the engine
            project_id = value.get("project_id", "")
            project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
            # Use original message_id if available, otherwise use the card message_id
            target_message_id = original_message_id or open_message_id
            self.handle_message(target_message_id, open_chat_id, message_preview, project)
        else:
            # Engine not active — trigger auto-activate path
            logger.info(
                "Clarify confirm: engine not active, triggering auto-activate chat=%s preview=%s",
                open_chat_id, message_preview[:50],
            )
            # Activate slock with the message as requirement
            project_id = value.get("project_id", "")
            project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
            success = self.activate_slock(
                message_id=open_message_id,
                chat_id=open_chat_id,
                requirement=message_preview,
                project=project,
            )
            if not success:
                # If activation failed, at least show the confirmation card
                self.update_card(open_message_id, confirm_card_json)

    def _handle_clarify_ignore(self, open_message_id: str, open_chat_id: str, value: dict) -> None:
        """Handle "不是，只是聊天" button click from clarification card.

        Updates the card to "已忽略" state without creating any task.
        Validates that the action performer is the original message sender.
        """
        import json as _json

        from ...slock_engine.card_templates.queue_feedback import build_clarification_ignored_card
        from ...thread.manager import get_current_sender_id

        message_preview = str(value.get("message_preview") or "")
        original_sender_id = str(value.get("sender_id") or "")
        current_sender_id = get_current_sender_id() or ""

        # Sender verification: only original sender can ignore
        if original_sender_id and current_sender_id and original_sender_id != current_sender_id:
            logger.warning(
                "Clarify ignore rejected: sender mismatch (original=%s, current=%s)",
                original_sender_id, current_sender_id,
            )
            self.reply_text(open_message_id, "⚠️ 仅消息发送者可忽略此操作。")
            return

        logger.info(
            "Clarify ignore: user marked message as chat chat=%s preview=%s",
            open_chat_id, message_preview[:50],
        )

        # Build and send ignored card
        ignored_card = build_clarification_ignored_card(message_preview=message_preview)
        ignored_card_json = _json.dumps(ignored_card, ensure_ascii=False)
        self.update_card(open_message_id, ignored_card_json)

    def handle_card_action(self, open_message_id: str, open_chat_id: str, action_type: str, value: dict):
        """Handle slock_* card actions."""
        # Escalation resolve needs extra params from value — handle before standard dispatch
        if action_type == "slock_escalation_resolve":
            self._resolve_escalation(open_message_id, open_chat_id, value)
            return

        # Per-agent stop needs agent_id from value — handle before standard dispatch
        if action_type == "slock_stop_agent":
            self._stop_single_agent(open_message_id, open_chat_id, value)
            return

        if action_type == "slock_agent_follow_up":
            agent_name = value.get("agent_name") or value.get("agent_id") or "Agent"
            self.send_text_to_chat(open_chat_id, f"可以直接在群里 @ {agent_name} 继续追问。")
            return

        if action_type == "slock_agent_show_reasoning":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id) or manager.get_active_engine(open_chat_id)
            agent_id = str(value.get("agent_id") or "")
            task_id = str(value.get("task_id") or "")
            rejection = self._require_slock_permission(engine, action_type, allow_assignee=True, task_id=task_id)
            if rejection:
                return rejection
            snapshot = engine.memory.read_agent_reasoning_snapshot(agent_id, task_id) if engine and agent_id and task_id else {}
            if snapshot:
                tool_label = snapshot.get("tool_name") or "unknown"
                model_label = snapshot.get("model_name") or "default"
                prompt_summary = redact_sensitive(str(snapshot.get("prompt_summary") or ""))[:1200]
                result_summary = redact_sensitive(str(snapshot.get("result_summary") or ""))[:1800]
                self.send_text_to_chat(
                    open_chat_id,
                    "🧠 **执行摘要**\n\n"
                    f"• Agent: `{agent_id[:8]}`\n"
                    f"• 工具/模型: `{tool_label}` / `{model_label}`\n\n"
                    f"**输入摘要**\n{prompt_summary or '(空)'}\n\n"
                    f"**输出摘要**\n{result_summary or '(空)'}",
                )
                return
            self.send_text_to_chat(open_chat_id, "当前 Agent 回复未保存可展示的执行摘要；可查看任务执行卡和消息归档。")
            return

        if action_type == "slock_agent_mark_done":
            task_id = value.get("task_id", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            rejection = self._require_slock_permission(engine, action_type, allow_assignee=True, task_id=task_id)
            if rejection:
                if isinstance(rejection, dict):
                    content = rejection.get("toast", {}).get("content", "权限不足")
                    self.send_text_to_chat(open_chat_id, content)
                    return rejection
                if not self._has_slock_permission(engine):
                    self.send_text_to_chat(open_chat_id, "⚠️ 权限不足，仅管理员或团队创建者可执行此操作。")
                    return rejection
            if not self._has_slock_permission(engine) and isinstance(rejection, dict):
                return rejection
            if task_id and engine:
                engine._force_complete_task(task_id)
            self.send_text_to_chat(open_chat_id, "✅ 已标记完成。")
            return

        # --- Task 20: Show agent memory (L1 snapshot) ---
        if action_type == "slock_agent_show_memory":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            agent_id = str(value.get("agent_id") or "")
            rejection = self._require_slock_permission(engine, action_type)
            if rejection:
                return rejection
            if engine and agent_id:
                memory = engine.memory.read_agent_memory(agent_id)
                if memory:
                    import json as _json

                    from src.slock_engine.card_templates import build_memory_display_card
                    agent = engine.registry.get(agent_id)
                    agent_name = agent.display_name if agent else agent_id[:8]
                    card = build_memory_display_card(memory, agent_name=agent_name)
                    self.send_card_to_chat(open_chat_id, _json.dumps(card, ensure_ascii=False))
                    return
            self.send_text_to_chat(open_chat_id, "该 Agent 暂无记忆记录。")
            return

        # --- Task 21: Show role switch card ---
        if action_type == "slock_agent_switch_role":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            agent_id = str(value.get("agent_id") or "")
            rejection = self._require_slock_permission(engine, action_type)
            if rejection:
                if isinstance(rejection, dict):
                    content = rejection.get("toast", {}).get("content", "权限不足")
                    self.send_text_to_chat(open_chat_id, content)
                    return rejection
                if not self._has_slock_permission(engine):
                    self.send_text_to_chat(open_chat_id, "⚠️ 权限不足，仅管理员或团队创建者可执行此操作。")
                    return rejection
            if not self._has_slock_permission(engine) and isinstance(rejection, dict):
                return rejection
            if engine and agent_id:
                channel_id = engine.channel.channel_id if engine.channel else ""
                agents = engine.registry.list_agents(channel_id=channel_id)
                available_roles = sorted({a.role for a in agents if a.role})
                # Add standard roles if not already present
                for std_role in ("coder", "reviewer", "writer", "tester", "planner", "architect"):
                    if std_role not in available_roles:
                        available_roles.append(std_role)
                import json as _json

                from src.slock_engine.card_templates import build_role_switch_card
                project_id = value.get("project_id", "")
                card = build_role_switch_card(
                    roles=available_roles,
                    agent_id=agent_id,
                    channel_id=channel_id,
                    project_id=project_id,
                )
                self.send_card_to_chat(open_chat_id, _json.dumps(card, ensure_ascii=False))
            else:
                self.send_text_to_chat(open_chat_id, "⚠️ 未找到活跃引擎或 Agent。")
            return

        # --- Task 22: Confirm role switch ---
        if action_type == "slock_confirm_switch_role":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            agent_id = str(value.get("agent_id") or "")
            target_role = str(value.get("target_role") or "")
            rejection = self._require_slock_permission(engine, action_type)
            if rejection:
                return rejection
            if engine and agent_id and target_role:
                agent = engine.registry.get(agent_id)
                if agent:
                    old_role = agent.role
                    agent.role = target_role
                    engine.registry.update(agent)
                    # Update memory role field
                    memory = engine.memory.read_agent_memory(agent_id)
                    if memory:
                        memory.role = f"{target_role}: {agent.system_prompt[:200]}" if agent.system_prompt else target_role
                        engine.memory.write_agent_memory(agent_id, memory)
                    self.send_text_to_chat(
                        open_chat_id,
                        f"🎭 **{agent.display_name}** 角色已切换: `{old_role}` → `{target_role}`",
                    )
                    return
            self.send_text_to_chat(open_chat_id, "⚠️ 角色切换失败，请重试。")
            return

        if action_type == "slock_confirm_discussion":
            thread_id = str(value.get("thread_id") or "")
            trust_type = str(value.get("trust_type") or "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if not engine:
                self.send_text_to_chat(open_chat_id, "⚠️ 未找到活跃 Slock 引擎。")
                return
            engine.confirm_discussion(thread_id, trust_type=trust_type)
            self.send_text_to_chat(open_chat_id, "✅ 讨论已启动。")
            return

        if action_type == "slock_cancel_discussion":
            thread_id = str(value.get("thread_id") or "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if not engine:
                self.send_text_to_chat(open_chat_id, "⚠️ 未找到活跃 Slock 引擎。")
                return
            engine.cancel_discussion(thread_id)
            self.send_text_to_chat(open_chat_id, "✅ 已取消讨论。")
            return

        # --- Tasks 26-28: Dissolve confirmation & undo ---
        if action_type == "slock_confirm_dissolve":
            team_name = str(value.get("team_name") or "")
            manager = self._get_engine_manager()
            engine = manager.find_team(team_name) if team_name else manager.get_activated_engine(open_chat_id)
            if engine and engine.channel:
                if not self._check_slock_permission(engine, open_message_id, open_chat_id):
                    return
                # Save snapshot for undo (30s TTL)
                from src.slock_engine.models import TeamSnapshot
                snapshot = TeamSnapshot(
                    channel_id=engine.channel.channel_id,
                    team_name=engine.channel.team_name or engine.channel.name or "",
                    owner_id=engine.channel.owner_id or "",
                    channel=engine.channel,
                    agent_ids=[a.agent_id for a in engine.registry.list_agents(channel_id=engine.channel.channel_id)],
                )
                if not hasattr(self, "_dissolve_snapshots"):
                    self._dissolve_snapshots: dict[str, TeamSnapshot] = {}
                self._dissolve_snapshots[snapshot.channel_id] = snapshot

                target_chat_id = engine.channel.channel_id
                engine.deactivate()
                manager.unregister_managed_chat(target_chat_id)
                manager.remove(target_chat_id, engine.root_path)
                self.send_text_to_chat(
                    open_chat_id,
                    f"✅ 团队 **{snapshot.team_name}** 已解散。30 秒内可点击撤销恢复。",
                )
            else:
                self.send_text_to_chat(open_chat_id, "⚠️ 未找到目标团队。")
            return

        if action_type == "slock_undo_dissolve":
            channel_id = str(value.get("channel_id") or "")
            snapshots = getattr(self, "_dissolve_snapshots", {})
            snapshot = snapshots.pop(channel_id, None) if channel_id else None
            if snapshot and (time.time() - snapshot.created_at) <= 30:
                self.send_text_to_chat(
                    open_chat_id,
                    f"↩️ 团队 **{snapshot.team_name}** 解散已撤销（本地状态恢复）。如飞书群已删除需手动重建。",
                )
            else:
                self.send_text_to_chat(open_chat_id, "⚠️ 撤销已过期或快照不存在。")
            return

        # --- Task 24: Form submissions from command panel ---
        if action_type == "slock_form_new_team":
            team_name = str(value.get("team_name") or "").strip()
            if not team_name:
                self.send_text_to_chat(open_chat_id, "⚠️ 请输入团队名称。")
                return
            project_id = value.get("project_id", "")
            project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
            self.create_team(open_message_id, open_chat_id, team_name, project)
            return

        if action_type == "slock_form_new_role":
            role_name = str(value.get("role_name") or "").strip()
            role_type = str(value.get("role_type") or "coder").strip()
            agent_type = str(value.get("agent_type") or "coco").strip()
            if not role_name:
                self.send_text_to_chat(open_chat_id, "⚠️ 请输入角色名称。")
                return
            project_id = value.get("project_id", "")
            project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
            # Build params string for create_role
            params_str = f"{role_name} --type={role_type} --agent={agent_type}"
            self.create_role(open_message_id, open_chat_id, params_str, project)
            return

        if action_type == "slock_form_council":
            topic = str(value.get("topic") or "").strip()
            if not topic:
                self.send_text_to_chat(open_chat_id, "⚠️ 请输入评审议题。")
                return
            project_id = value.get("project_id", "")
            project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
            self.run_council(open_message_id, open_chat_id, topic, project)
            return

        if action_type == "slock_form_discuss":
            topic = str(value.get("topic") or "").strip()
            if not topic:
                self.send_text_to_chat(open_chat_id, "⚠️ 请输入讨论主题。")
                return
            project_id = value.get("project_id", "")
            project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
            self._trigger_nli_discussion(open_message_id, open_chat_id, topic, {}, project)
            return

        if action_type == "slock_discussion_expand":
            self._expand_discussion(open_chat_id, value)
            return

        if action_type == "slock_discussion_stop":
            self._stop_discussion(open_chat_id, value)
            return

        # --- Tasks 19+20: Collaboration plan actions & user intervention ---
        if action_type == "slock_plan_approve":
            plan_id = value.get("plan_id", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine and plan_id:
                if not self._check_slock_permission(engine, open_message_id, open_chat_id):
                    return
                success = engine.collaboration_orchestrator.approve_plan(plan_id)
                if success:
                    import json as _json

                    from src.slock_engine.card_templates.progress import build_collaboration_plan_card
                    plan = engine.collaboration_orchestrator.get_plan(plan_id)
                    if plan:
                        channel_id = engine.channel.channel_id if engine.channel else ""
                        agents = engine.registry.list_agents(channel_id=channel_id)
                        card = build_collaboration_plan_card(plan, agents, channel_id=channel_id)
                        self.update_card(open_message_id, _json.dumps(card, ensure_ascii=False))
            return

        if action_type == "slock_plan_cancel":
            plan_id = value.get("plan_id", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine and plan_id:
                if not self._check_slock_permission(engine, open_message_id, open_chat_id):
                    return
                engine.collaboration_orchestrator.cancel_plan(plan_id)
            return

        if action_type == "slock_progress_refresh":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine:
                import json as _json

                from src.slock_engine.card_templates.progress import build_progress_overview_card
                channel_id = engine.channel.channel_id if engine.channel else ""
                plans = engine.collaboration_orchestrator.list_active_plans(channel_id)
                agents = engine.registry.list_agents(channel_id=channel_id)
                card = build_progress_overview_card(plans, agents, channel_id=channel_id)
                self.send_card_to_chat(open_chat_id, _json.dumps(card, ensure_ascii=False))
            return

        if action_type == "slock_user_intervention":
            plan_id = value.get("plan_id", "")
            message = value.get("message", "") or value.get("supplement_content", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if not engine or not plan_id:
                self.send_text_to_chat(open_chat_id, "⚠️ 引擎未激活或缺少 plan_id。")
                return
            if not self._check_slock_permission(engine, open_message_id, open_chat_id):
                return

            plan = engine.collaboration_orchestrator.get_plan(plan_id)
            if not plan:
                self.send_text_to_chat(open_chat_id, "⚠️ 未找到该计划。")
                return

            # If no message provided, pause the plan and prompt user for input
            if not message:
                engine.collaboration_orchestrator.pause_plan(plan_id)
                self.send_text_to_chat(
                    open_chat_id,
                    "⏸ 计划已暂停。请在讨论面板中输入补充信息后点击发送，或使用 ▶️ 恢复 按钮继续。",
                )
                return

            # Inject intervention message into current step's agent context
            current = plan.current_step
            if current and current.agent_id:
                engine.memory.update_agent_context(
                    current.agent_id,
                    f"[用户干预] {message[:200]}",
                )
                self.send_text_to_chat(
                    open_chat_id,
                    f"✅ 已将干预信息注入 Agent `{current.agent_id[:8]}` 上下文。",
                )
                # Resume if paused
                from ...slock_engine.models import CollaborationPlanStatus
                if plan.status == CollaborationPlanStatus.PAUSED:
                    engine.collaboration_orchestrator.resume_plan(plan_id)
                    self.send_text_to_chat(open_chat_id, "▶️ 计划已恢复执行。")
            else:
                self.send_text_to_chat(open_chat_id, "⚠️ 当前步骤无执行中的 Agent，无法注入干预信息。")
            return

        if action_type == "slock_plan_supplement":
            plan_id = value.get("plan_id", "")
            # 兼容读取: 优先 _form_value 子字典，回退顶层
            form_value = value.get("_form_value", {})
            content = (form_value.get("supplement_content") or value.get("supplement_content", "")).strip()
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if not engine or not plan_id:
                return
            if not content:
                return {"toast": {"type": "error", "content": "请输入补充信息内容"}}
            # Permission check
            rejection = self._require_slock_permission(engine, action_type)
            if rejection:
                return rejection
            # Text length check
            if len(content) > 2000:
                return {"toast": {"type": "error", "content": "补充内容过长，请限制在 2000 字符以内"}}
            # Sensitive content filter
            import re as _re
            if _re.search(r'(token|key|secret|password|credential)[=:]\s*\S{8,}', content, _re.I):
                return {"toast": {"type": "error", "content": "检测到疑似敏感凭据信息，请移除后重试"}}
            plan = engine.collaboration_orchestrator.get_plan(plan_id)
            if not plan:
                return
            # Inject to all collaborating agents in this plan
            agent_ids = {step.agent_id for step in plan.steps if step.agent_id}
            if agent_ids:
                supplement_text = f"[用户补充] {content[:500]}"
                for agent_id in agent_ids:
                    engine.memory.update_agent_context(agent_id, supplement_text)
                self.send_text_to_chat(
                    open_chat_id,
                    f"✅ 补充信息已注入 {len(agent_ids)} 个协作 Agent 上下文。",
                )
            else:
                self.send_text_to_chat(open_chat_id, "⚠️ 计划中暂无已分配的 Agent。")
            return

        # --- Collaboration plan detail / pause / resume ---
        if action_type == "slock_show_plan_detail":
            plan_id = value.get("plan_id", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine and plan_id:
                from ...slock_engine.card_templates.progress import build_collaboration_plan_card
                plan = engine.collaboration_orchestrator.get_plan(plan_id)
                if plan:
                    channel_id = engine.channel.channel_id if engine.channel else ""
                    agents = engine.registry.list_agents(channel_id=channel_id)
                    card = build_collaboration_plan_card(plan, agents, channel_id=channel_id)
                    self.send_card_to_chat(open_chat_id, json.dumps(card, ensure_ascii=False))
                    return
            self.send_text_to_chat(open_chat_id, "⚠️ 未找到该计划。")
            return

        if action_type == "slock_pause_plan":
            plan_id = value.get("plan_id", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine and plan_id:
                if not self._check_slock_permission(engine, open_message_id, open_chat_id):
                    return
                success = engine.collaboration_orchestrator.pause_plan(plan_id)
                if success:
                    self.send_text_to_chat(open_chat_id, f"⏸ 计划 `{plan_id[:8]}` 已暂停。")
                    return
            self.send_text_to_chat(open_chat_id, "⚠️ 暂停失败，计划不存在或状态不允许。")
            return

        if action_type == "slock_resume_plan":
            plan_id = value.get("plan_id", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine and plan_id:
                if not self._check_slock_permission(engine, open_message_id, open_chat_id):
                    return
                success = engine.collaboration_orchestrator.resume_plan(plan_id)
                if success:
                    self.send_text_to_chat(open_chat_id, f"▶️ 计划 `{plan_id[:8]}` 已恢复执行。")
                    return
            self.send_text_to_chat(open_chat_id, "⚠️ 恢复失败，计划不存在或状态不允许。")
            return

        # --- Role info from card button ---
        if action_type == "slock_role_info":
            agent_id = value.get("agent_id", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine and agent_id:
                agent = engine.registry.get(agent_id)
                if agent:
                    self.show_role_info(open_message_id, open_chat_id, agent.name)
                    return
            self.send_text_to_chat(open_chat_id, "⚠️ 未找到该角色。")
            return

        # --- Task board actions ---
        if action_type == "slock_new_task":
            content = value.get("content", "")
            if not content:
                self.send_text_to_chat(open_chat_id, "⚠️ 请输入任务内容。")
                return
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine:
                self.assign_task(open_message_id, open_chat_id, content)
            return

        if action_type == "slock_dispatch_tasks":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine:
                engine.dispatch_pending_tasks()
                self.send_text_to_chat(open_chat_id, "✅ 已派发待处理任务。")
            return

        if action_type == "slock_show_task_board":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if engine:
                from ...slock_engine.card_templates import build_task_board_card
                channel_id = engine.channel.channel_id if engine.channel else ""
                agents = engine.registry.list_agents(channel_id=channel_id)
                card = build_task_board_card(
                    tasks=engine.tasks,
                    agents=agents,
                    channel_id=channel_id,
                )
                self.send_card_to_chat(open_chat_id, json.dumps(card, ensure_ascii=False))
            return

        if action_type == "slock_assign_task_to_agent":
            task_id = value.get("task_id", "")
            agent_id = value.get("agent_id", "")
            task_content = value.get("task_content", "")
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(open_chat_id)
            if not engine or not agent_id:
                self.send_text_to_chat(open_chat_id, "⚠️ 引擎未激活或缺少 agent_id。")
                return

            # Case 1: Form submission with task_content → create + assign
            if task_content and not task_id:
                new_task = engine.create_and_assign_task(task_content.strip(), agent_id)
                if new_task:
                    self.send_text_to_chat(
                        open_chat_id,
                        f"✅ 任务已创建并分配给 `{agent_id[:8]}`。",
                    )
                else:
                    self.send_text_to_chat(open_chat_id, "⚠️ 任务创建失败。")
                return

            # Case 2: Existing task_id → reassign
            if task_id:
                success = engine.assign_task_to_agent(task_id, agent_id)
                if success:
                    self.send_text_to_chat(open_chat_id, f"✅ 任务已分配给 `{agent_id[:8]}`。")
                else:
                    self.send_text_to_chat(open_chat_id, "⚠️ 任务分配失败。")
                return

            self.send_text_to_chat(open_chat_id, "⚠️ 缺少任务内容或任务 ID。")
            return

        # --- Task 9: Clarification card button handlers ---
        if action_type == "slock_clarify_confirm":
            self._handle_clarify_confirm(open_message_id, open_chat_id, value)
            return

        if action_type == "slock_clarify_ignore":
            self._handle_clarify_ignore(open_message_id, open_chat_id, value)
            return

        # --- Hub card button routing (slock_hub_cmd) ---
        if action_type == "slock_hub_cmd":
            cmd_text = str(value.get("cmd") or "").strip()
            if not cmd_text:
                self.send_text_to_chat(open_chat_id, "⚠️ 无效的命令。")
                return
            project_id = value.get("project_id", "")
            project = self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
            self.handle_slock_command(open_message_id, open_chat_id, cmd_text, project)
            return

        # --- Command panel button routing (slock_cmd_* prefix) ---
        if action_type.startswith("slock_cmd_"):
            self._dispatch_cmd_panel_action(open_message_id, open_chat_id, action_type, value)
            return

        project_id = value.get("project_id", "")
        target_project = (
            self.project_manager.get_project_for_chat(project_id, open_chat_id) if project_id else None
        )

        slock_actions = {
            "slock_stop": self.stop_slock_engine,
            "slock_refresh_status": self._refresh_status_card,
            "slock_refresh_task_board": self._refresh_task_board_card,
        }

        if action_type not in slock_actions:
            logger.warning("Unhandled slock card action: %s", action_type)

        self._dispatch_standard_card_action(CardActionContext(
            open_message_id=open_message_id,
            open_chat_id=open_chat_id,
            action_type=action_type,
            value=value,
            prefix="slock",
            action_map=slock_actions,
            toggle_log_method=self._toggle_log,
            switch_mode_method=self._switch_card_mode,
            toggle_ac_method=self._toggle_ac,
            project=target_project,
        ))

    # ------------------------------------------------------------------
    # Command panel button dispatch
    # ------------------------------------------------------------------

    def _dispatch_cmd_panel_action(
        self, message_id: str, chat_id: str, action_type: str, value: dict
    ) -> None:
        """Route slock_cmd_* button actions to existing handler methods."""
        project = None
        project_id = value.get("project_id", "")
        if project_id:
            project = self.project_manager.get_project_for_chat(project_id, chat_id)

        # --- Task 25: Permission check for destructive actions ---
        _PERM_REQUIRED = {"slock_cmd_dissolve_team", "slock_cmd_stop"}
        if action_type in _PERM_REQUIRED:
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(chat_id)
            if engine and not self._check_slock_permission(engine, message_id, chat_id):
                return

        # --- Task 26: Dissolve with confirmation card ---
        if action_type == "slock_cmd_dissolve_team":
            manager = self._get_engine_manager()
            engine = manager.get_activated_engine(chat_id)
            if not engine or not engine.channel:
                self.send_text_to_chat(chat_id, "⚠️ 当前没有活跃团队可解散。")
                return
            team_name = engine.channel.team_name or engine.channel.name or "当前团队"
            channel_id = engine.channel.channel_id
            import json as _json
            buttons = [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "确认解散"},
                    "type": "danger",
                    "behaviors": [{"type": "callback", "value": {
                        "action": "slock_confirm_dissolve",
                        "team_name": team_name,
                        "channel_id": channel_id,
                        "project_id": project_id,
                    }}],
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "取消"},
                    "type": "default",
                    "behaviors": [{"type": "callback", "value": {
                        "action": "slock_noop", "channel_id": channel_id,
                    }}],
                },
            ]
            confirm_card = {
                "schema": "2.0",
                "config": {"wide_screen_mode": True},
                "header": {"title": {"tag": "plain_text", "content": "⚠️ 确认解散团队"}, "template": "red"},
                "body": {"elements": [
                    {"tag": "markdown", "content": f"即将解散团队 **{team_name}**，此操作将：\n- 停止所有 Agent\n- 删除飞书群\n- 清除运行时状态\n\n确认继续？"},
                    *build_responsive_layout(buttons),
                ]},
            }
            sent_message_id = self.send_card_to_chat(
                chat_id, _json.dumps(confirm_card, ensure_ascii=False)
            )
            if not sent_message_id:
                self.send_text_to_chat(
                    chat_id,
                    "⚠️ 解散确认卡发送失败，未执行解散。请重新打开 `/slock` 控制台后再试。",
                )
            return

        # --- Task 23: Discuss routing ---
        if action_type == "slock_cmd_discuss":
            topic = str(value.get("topic") or "").strip()
            if not topic:
                # Task 29: send hint for empty params
                self.send_text_to_chat(chat_id, "💡 请输入讨论主题，例如: `/slock discuss 方案对比`")
                return
            self._trigger_nli_discussion(message_id, chat_id, topic, {}, project)
            return

        # --- Task 23: Council routing (empty topic → hint) ---
        if action_type == "slock_cmd_council":
            topic = str(value.get("topic") or "").strip()
            if not topic:
                self.send_text_to_chat(chat_id, "💡 请输入评审议题，例如: `/council 方案是否可行`")
                return
            self.run_council(message_id, chat_id, topic, project)
            return

        # --- Task 22: Extended panel routing ---
        if action_type == "slock_cmd_panel_extended":
            import json as _json

            from ...slock_engine.card_templates import build_command_panel_extended_card
            extended_card = build_command_panel_extended_card(channel_id=chat_id, project_id=project_id)
            self.send_card_to_chat(chat_id, _json.dumps(extended_card, ensure_ascii=False))
            return

        # --- Task 21: Memory routing (empty target → hint) ---
        if action_type == "slock_cmd_memory":
            target = str(value.get("target") or "").strip()
            if not target:
                self.send_text_to_chat(chat_id, "💡 请指定角色，例如: `/memory @coder` 或 `/memory coder`")
                return
            # Delegate to handle_slock_command with /memory syntax
            self.handle_slock_command(message_id, chat_id, f"/memory {target}", project)
            return

        # --- Task 20: Parameter-required actions with hints ---
        if action_type == "slock_cmd_task_assign":
            self.send_text_to_chat(chat_id, "💡 请使用命令分配任务，例如: `/task assign 修复登录bug @coder`")
            return
        if action_type == "slock_cmd_role_info":
            self.send_text_to_chat(chat_id, "💡 请指定角色，例如: `/role info coder`")
            return
        if action_type == "slock_cmd_role_remove":
            self.send_text_to_chat(chat_id, "💡 请指定要移除的角色，例如: `/role remove coder`")
            return
        if action_type == "slock_cmd_team_status":
            self.send_text_to_chat(chat_id, "💡 请指定团队，例如: `/team status 前端团队` 或直接使用 `/slock status`")
            return

        # --- Command fix suggestion: execute the suggested command ---
        if action_type == "slock_cmd_fix":
            fix_command = str(value.get("fix_command") or "").strip()
            if fix_command:
                self.handle_slock_command(message_id, chat_id, fix_command, project)
            else:
                self.send_text_to_chat(chat_id, "⚠️ 修正命令为空，请手动输入命令")
            return

        _CMD_DISPATCH: dict[str, callable] = {
            "slock_cmd_team_list": lambda: self.list_teams(message_id, chat_id, project),
            "slock_cmd_new_team": lambda: self.create_team(message_id, chat_id, "", project),
            "slock_cmd_role_list": lambda: self.list_roles(message_id, chat_id, project),
            "slock_cmd_new_role": lambda: self.create_role(message_id, chat_id, "", project),
            "slock_cmd_task_list": lambda: self.list_tasks(message_id, chat_id, project),
            "slock_cmd_task_status": lambda: self.show_task_status(message_id, chat_id, project),
            "slock_cmd_status": lambda: self._refresh_status_card(message_id, chat_id, value),
        }

        handler = _CMD_DISPATCH.get(action_type)
        if handler:
            handler()
        else:
            self.send_text_to_chat(chat_id, f"未知的面板操作: {action_type}")

    # ------------------------------------------------------------------
    # Static command detection
    # ------------------------------------------------------------------

    @staticmethod
    def is_slock_command(text: str, chat_id: str | None = None, manager=None) -> bool:
        """Check if text is any slock-related command."""
        return is_slock_command(text, chat_id=chat_id, manager=manager)
