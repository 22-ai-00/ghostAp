"""Main-Bot control surface for the Journal-backed Employee Department."""

from __future__ import annotations

import logging
import re
import shlex
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from ...acp.helper import list_acp_tools
from ...autonomous.data.ports import HistoryQuerySpec, MemoryQuerySpec
from ...autonomous.data.query import (
    AuditFailedError,
    AuthenticatedDataRequest,
    QueryDeniedError,
)
from ...autonomous.provisioning.fire_service import (
    EmployeeFireRequest,
    FireServiceError,
)
from ...autonomous.provisioning.fire_state import (
    FireCleanupMode,
    FireEffectState,
    FirePhase,
)
from ...autonomous.provisioning.hire_port import (
    EmployeeHireRequest,
    EmployeeRoleUpdateRequest,
)
from ...autonomous.provisioning.hire_service import HireAdmissionError
from ...card.builders.system import SystemBuilder
from ...config import get_settings
from ...thread import (
    get_current_is_p2p,
    get_current_sender_id,
    get_current_sender_union_id,
    get_current_tenant_key,
    get_current_thread_id,
)
from ...utils.redact import redact_sensitive
from .base import BaseHandler

if TYPE_CHECKING:
    from ...project import ProjectContext


logger = logging.getLogger(__name__)

_ROSTER_UNAVAILABLE = "数字员工目录暂不可用，请稍后重试。"
_ROLE_USAGE = "用法：`/employee-role <员工名> <职责>`"
_ROLE_UNAVAILABLE = "数字员工职责更新暂不可用，请稍后重试。"
_HIRE_USAGE = (
    "用法：`/hire <名字> [--tool <工具>] [--model <模型>] "
    "[--role <职责>] [--profile <档位>] [--effort <强度>] "
    "[--app-id <AppID>]`"
)
_FIRE_USAGE = (
    "用法：`/fire <员工名称或ID> [--drain] "
    "[--confirm-app-disposed <AppID|NO_APP_FOUND>]`"
)
_HISTORY_USAGE = "用法：`/history <员工名称或ID>`"
_MEMORY_USAGE = "用法：`/employee-memory <员工名称或ID>`"
_ACP_RECOMMENDATION_ORDER = (
    "traex",
    "claude",
    "codex",
    "grok",
    "aiden",
    "gemini",
    "coco",
)
_HIRE_VALUE_OPTIONS = frozenset(
    {"--tool", "--model", "--role", "--profile", "--effort", "--app-id"}
)
_APP_ID_PATTERN = re.compile(r"cli_[A-Za-z0-9_-]{3,128}\Z")
_MAX_DATA_REPLY_CHARS = 18_000
_POSIX_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:[^\s/]+/)+[^\s,;]+"
)
_WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9_])[A-Z]:\\(?:[^\s\\]+\\)+[^\s,;]+"
)
_QUOTED_SECRET_VALUE = re.compile(
    r'''(?ix)
    (
        ["']?[A-Z0-9_.-]{0,64}
        (?:token|secret|password|passwd|credential|api[_-]?key|private[_-]?key)
        [A-Z0-9_.-]{0,64}["']?\s*[:=]\s*
    )
    (["'])[^"'\n]*(?:\2|(?=\n|\Z))
    '''
)


class EmployeeHandler(BaseHandler):
    """Expose tenant-scoped employee controls without owning domain state."""

    def _configured_admin_ids(self) -> frozenset[str]:
        raw = getattr(getattr(self.ctx, "settings", None), "admin_user_ids", None)
        if isinstance(raw, str):
            return frozenset(item.strip() for item in raw.split(",") if item.strip())
        if isinstance(raw, (set, frozenset, tuple, list)):
            return frozenset(
                item
                for item in raw
                if isinstance(item, str) and item and item.strip() == item
            )
        return self._admin_ids()

    def _admin_request_context(
        self,
        message_id: str,
        *,
        require_union_id: bool = False,
    ) -> tuple[str, str, str] | None:
        """Return authenticated main-Bot admin coordinates or fail closed."""

        sender_id = get_current_sender_id() or ""
        if (
            not get_current_is_p2p()
            or not sender_id
            or sender_id not in self._configured_admin_ids()
        ):
            self.reply_text(
                message_id,
                "⛔ 该命令仅允许配置管理员在主 Bot 私聊中使用。",
            )
            return None
        resolver = getattr(self.ctx, "tenant_key_resolver", None)
        if not callable(resolver):
            self.reply_text(message_id, "租户身份不可用；未执行任何员工操作。")
            return None
        try:
            tenant_key = resolver()
        except Exception:
            tenant_key = ""
        transport_tenant = get_current_tenant_key() or ""
        if (
            not isinstance(tenant_key, str)
            or not tenant_key
            or tenant_key != tenant_key.strip()
            or transport_tenant != tenant_key
        ):
            self.reply_text(message_id, "租户身份不可用；未执行任何员工操作。")
            return None
        union_id = get_current_sender_union_id() or ""
        if require_union_id and (
            not isinstance(union_id, str)
            or not union_id
            or union_id != union_id.strip()
        ):
            self.reply_text(message_id, "用户跨应用身份不可用；未发起员工雇佣。")
            return None
        return sender_id, union_id, tenant_key

    @staticmethod
    def _parse_options(
        args: str,
        *,
        value_options: frozenset[str],
        boolean_options: frozenset[str] = frozenset(),
    ) -> tuple[str, dict[str, str | bool]] | None:
        try:
            tokens = shlex.split(args or "", posix=True)
        except ValueError:
            return None
        positional: list[str] = []
        options: dict[str, str | bool] = {}
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token in boolean_options:
                if token in options:
                    return None
                options[token] = True
                index += 1
                continue
            if token.startswith("--"):
                if token not in value_options or token in options:
                    return None
                if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                    return None
                value = tokens[index + 1]
                if not value or value != value.strip():
                    return None
                options[token] = value
                index += 2
                continue
            positional.append(token)
            index += 1
        if len(positional) != 1:
            return None
        return positional[0], options

    @staticmethod
    def _bounded_plaintext(value: Any, *, limit: int = _MAX_DATA_REPLY_CHARS) -> str:
        text = str(value or "")
        text = "".join(
            character
            for character in text
            if character in {"\n", "\t"} or ord(character) >= 32
        )
        # Redact the complete quoted value before the generic scanner can
        # partially replace a whitespace-delimited prefix and expose its tail.
        text = _QUOTED_SECRET_VALUE.sub(r"\1\2<redacted>\2", text)
        text = redact_sensitive(text)
        text = _POSIX_ABSOLUTE_PATH.sub("<redacted:path>", text)
        text = _WINDOWS_ABSOLUTE_PATH.sub("<redacted:path>", text)
        if len(text) <= limit:
            return text
        return text[:limit] + "\n\n[内容已截断]"

    @staticmethod
    def _safe_fragment(value: Any, *, limit: int) -> str:
        return " ".join(str(value or "").split())[:limit]

    def _available_tool_names(self) -> tuple[str, ...]:
        try:
            discovered = list_acp_tools()
        except Exception:
            logger.warning("employee ACP discovery failed", exc_info=True)
            return ()
        names: list[str] = []
        for option in discovered:
            name = getattr(option, "name", "")
            if isinstance(name, str) and name and name == name.strip():
                names.append(name.casefold())
        return tuple(dict.fromkeys(names))

    def _select_hire_tool(self, requested: str) -> str:
        available = self._available_tool_names()
        if requested:
            normalized = requested.strip().casefold()
            return normalized if normalized in available else ""
        configured = str(
            getattr(getattr(self.ctx, "settings", None), "default_acp_tool", "")
            or ""
        ).strip().casefold()
        if configured and configured in available:
            return configured
        return next(
            (name for name in _ACP_RECOMMENDATION_ORDER if name in available),
            "",
        )

    def hire_employee(
        self,
        message_id: str,
        chat_id: str,
        args: str,
    ) -> None:
        """Anchor one controlled hire admission, then let provisioning continue."""

        identity = self._admin_request_context(message_id, require_union_id=True)
        if identity is None:
            return
        parsed = self._parse_options(args, value_options=_HIRE_VALUE_OPTIONS)
        if parsed is None:
            self.reply_text(message_id, _HIRE_USAGE)
            return
        employee_name, options = parsed
        if "--prompt" in options:
            self.reply_text(message_id, _HIRE_USAGE)
            return
        app_id = str(options.get("--app-id", ""))
        if app_id and _APP_ID_PATTERN.fullmatch(app_id) is None:
            self.reply_text(message_id, _HIRE_USAGE)
            return
        requested_tool = str(options.get("--tool", ""))
        tool = self._select_hire_tool(requested_tool)
        if not tool:
            self.reply_text(
                message_id,
                "没有可严格确认可用的编程工具；雇佣请求未写入 Journal。",
            )
            return
        readiness_provider = getattr(self.ctx, "employee_hire_readiness", None)
        service = getattr(self.ctx, "employee_hire_service", None)
        start_hire = getattr(service, "start_hire", None)
        if not callable(readiness_provider) or not callable(start_hire):
            self.reply_text(message_id, "员工雇佣服务未就绪；请求未写入 Journal。")
            return
        try:
            readiness = readiness_provider()
        except Exception:
            readiness = None
        if getattr(readiness, "ready", None) is not True:
            self.reply_text(
                message_id,
                "员工雇佣安全前置条件尚未满足；请求未写入 Journal。",
            )
            return
        sender_id, union_id, tenant_key = identity
        request = EmployeeHireRequest(
            employee_name=employee_name,
            tool=tool,
            model=str(options.get("--model", "")),
            effort=str(options.get("--effort", "default")),
            profile=str(options.get("--profile", "standard")),
            role=str(options.get("--role", "")),
            chat_id=chat_id,
            message_id=message_id,
            requester_principal_id=sender_id,
            requester_union_id=union_id,
            tenant_key=tenant_key,
            existing_app_id=app_id,
        )
        try:
            state = start_hire(request)
        except HireAdmissionError as exc:
            logger.warning("employee hire admission rejected: %s", type(exc).__name__)
            reason = str(exc)
            if reason == "provisioning submission failed after durable admission":
                visible = (
                    "⚠️ 雇佣准入已持久化，但本次自动调度失败。请勿重复发起雇佣；"
                    "系统会在运行时恢复流程中继续该记录，请检查服务健康状态。"
                )
            elif "capacity" in reason or "visible_employee_limit" in reason:
                visible = "员工容量已满；请求未受理。"
            elif "conflict" in reason:
                visible = "员工名称、应用或请求幂等信息冲突；请求未受理。"
            elif "authorized" in reason:
                visible = "员工雇佣授权失败；请求未受理。"
            else:
                visible = "员工雇佣被安全门禁拒绝；请求未受理。"
            self.reply_text(message_id, visible)
            return
        except Exception as exc:
            logger.error("employee hire admission failed: %s", type(exc).__name__)
            self.reply_text(message_id, "员工雇佣暂不可用；请查看审计日志后重试。")
            return
        display_name = self._safe_fragment(
            getattr(state, "employee_name", "") or employee_name,
            limit=80,
        )
        self.reply_text(
            message_id,
            f"✅ 员工 `{display_name}` 的雇佣请求已受理并持久化，正在自动完成配置。",
        )

    def fire_employee(self, message_id: str, chat_id: str, args: str) -> None:
        """Retire one employee through the current one-way cleanup service."""

        identity = self._admin_request_context(message_id)
        if identity is None:
            return
        parsed = self._parse_options(
            args,
            value_options=frozenset({"--confirm-app-disposed"}),
            boolean_options=frozenset({"--drain"}),
        )
        if parsed is None:
            self.reply_text(message_id, _FIRE_USAGE)
            return
        employee, options = parsed
        confirmation_ref = str(options.get("--confirm-app-disposed", ""))
        if confirmation_ref and (
            confirmation_ref != "NO_APP_FOUND"
            and _APP_ID_PATTERN.fullmatch(confirmation_ref) is None
        ):
            self.reply_text(message_id, _FIRE_USAGE)
            return
        service = getattr(self.ctx, "employee_fire_service", None)
        start_fire = getattr(service, "start_fire", None)
        confirm = getattr(service, "confirm_external_disposition", None)
        if not callable(start_fire) or not callable(confirm):
            self.reply_text(message_id, "员工退役服务未就绪；未执行任何清理操作。")
            return
        sender_id, _union_id, tenant_key = identity
        try:
            request = EmployeeFireRequest(
                employee=employee,
                tenant_key=tenant_key,
                message_id=message_id,
                chat_id=chat_id,
                requester_principal_id=sender_id,
                drain=options.get("--drain") is True,
            )
            state = (
                confirm(request, confirmation_ref)
                if confirmation_ref
                else start_fire(request)
            )
        except (FireServiceError, ValueError) as exc:
            logger.warning("employee fire rejected: %s", type(exc).__name__)
            reason = str(exc)
            if reason == "employee already archived":
                visible = "该员工已经归档，无需重复退役。"
            elif "uniquely resolved" in reason:
                visible = "未找到唯一的在职员工；请核对名称或员工 ID。"
            elif "reference mismatch" in reason:
                visible = "外部应用处置确认与待处理记录不匹配。"
            else:
                visible = "员工退役被安全门禁拒绝；未执行或误报任何清理操作。"
            self.reply_text(message_id, visible)
            return
        except Exception as exc:
            logger.error("employee fire failed closed: %s", type(exc).__name__)
            self.reply_text(message_id, "员工退役未能安全推进；请查看审计日志后处理。")
            return
        self._reply_fire_state(message_id, employee, state)

    def _reply_fire_state(self, message_id: str, employee: str, state: Any) -> None:
        if getattr(state, "phase", None) is FirePhase.ARCHIVED:
            if getattr(state, "external_disposition_confirmed", False):
                text = (
                    "✅ 已记录管理员的开放平台处置确认，员工现已完成归档；"
                    "GhostAP 仅记录该确认，未声称代为删除应用。"
                )
            elif getattr(state, "cleanup_mode", None) is FireCleanupMode.SAFE_ABORT:
                text = (
                    "✅ 员工创建已在外部注册执行前安全中止并完成归档；"
                    "GhostAP 未修改或删除任何已有应用。"
                )
            elif getattr(state, "cleanup_mode", None) is FireCleanupMode.EXTERNAL_UNKNOWN:
                text = (
                    "⚠️ 退役记录缺少可验证的外部处置事实；请检查安全审计，"
                    "GhostAP 未声称应用已删除。"
                )
            else:
                text = (
                    "✅ 员工已停止托管、清理本地凭据并归档。开放平台应用仍需"
                    "管理员按实际状态处置，GhostAP 未声称已删除该应用。"
                )
            self.reply_text(message_id, text)
            return
        effects = dict(getattr(state, "effects", ()) or ())
        if getattr(state, "error_code", "") == "external_cleanup_authority_unavailable":
            disposition = getattr(state, "app_id", "") or "NO_APP_FOUND"
            self.reply_text(
                message_id,
                "⚠️ 员工已关闭新任务入口，但外部应用状态无法验证。请在开放平台"
                "完成核对或处置后执行 "
                f"`/fire {employee} --confirm-app-disposed {disposition}`；"
                "确认前名称和容量不会被释放。",
            )
        elif effects.get("slash_cleanup") is FireEffectState.ACTION_REQUIRED:
            self.reply_text(
                message_id,
                "⚠️ Slash 命令清理结果未确认；请核对开放平台后再次执行 `/fire`。"
                "系统不会盲目重复删除或误报完成。",
            )
        else:
            self.reply_text(
                message_id,
                "⚠️ 员工已关闭新任务入口，但退役仍需人工核对；"
                "凭据和归档不会被误报为完成。",
            )

    def _resolve_employee(self, tenant_key: str, target: str) -> Any | None:
        service = getattr(self.ctx, "employee_hire_service", None)
        list_roster = getattr(service, "list_employee_roster", None)
        if not callable(list_roster):
            return None
        employees = list_roster(tenant_key, include_archived=False)
        folded = target.strip().casefold()
        matches = tuple(
            employee
            for employee in employees
            if folded
            in {
                str(getattr(employee, "agent_id", "")).casefold(),
                str(getattr(employee, "name", "")).casefold(),
            }
        )
        return matches[0] if len(matches) == 1 else None

    def _employee_data_request(
        self,
        *,
        sender_id: str,
        tenant_key: str,
        employee: Any,
        chat_id: str,
    ) -> AuthenticatedDataRequest:
        return AuthenticatedDataRequest(
            principal_id=sender_id,
            tenant_key=tenant_key,
            receiving_bot_app_id=str(
                getattr(getattr(self.ctx, "settings", None), "app_id", "") or ""
            ),
            chat_id=chat_id,
            chat_type="p2p",
            thread_root_id=get_current_thread_id() or "",
            requested_agent_id=str(getattr(employee, "agent_id", "")),
        )

    def show_employee_history(
        self,
        message_id: str,
        chat_id: str,
        target: str,
    ) -> None:
        identity = self._admin_request_context(message_id)
        if identity is None:
            return
        if not isinstance(target, str) or not target.strip():
            self.reply_text(message_id, _HISTORY_USAGE)
            return
        sender_id, _union_id, tenant_key = identity
        try:
            employee = self._resolve_employee(tenant_key, target)
        except Exception as exc:
            logger.error(
                "employee history target resolution failed: %s",
                type(exc).__name__,
            )
            employee = None
        if employee is None:
            self.reply_text(message_id, "未找到唯一的在职员工；请核对名称或员工 ID。")
            return
        data = getattr(self.ctx, "employee_data_composition", None)
        query = getattr(getattr(data, "query", None), "query", None)
        if not callable(query):
            self.reply_text(message_id, "员工历史暂不可用，请稍后重试。")
            return
        try:
            timezone_name = str(getattr(data.service, "shard_timezone", "UTC"))
            end = datetime.now(ZoneInfo(timezone_name)).date()
            result = query(
                self._employee_data_request(
                    sender_id=sender_id,
                    tenant_key=tenant_key,
                    employee=employee,
                    chat_id=chat_id,
                ),
                HistoryQuerySpec(
                    start_day=(end - timedelta(days=6)).isoformat(),
                    end_day=end.isoformat(),
                    page_size=50,
                ),
            )
        except QueryDeniedError:
            self.reply_text(message_id, "权限不足，无法读取该员工历史。")
            return
        except AuditFailedError:
            logger.error("employee history audit failed closed")
            self.reply_text(message_id, "员工历史审计暂不可用；未返回任何数据。")
            return
        except Exception as exc:
            logger.error(
                "employee history query failed closed: %s",
                type(exc).__name__,
            )
            self.reply_text(message_id, "员工历史暂不可用，请稍后重试。")
            return
        records = tuple(getattr(result, "records", ()) or ())
        if not records:
            content = "最近 7 天暂无执行记录。"
        else:
            rows = []
            for record in records[:50]:
                ended = self._safe_fragment(getattr(record, "ended_at", ""), limit=19)
                status = self._safe_fragment(getattr(record, "status", ""), limit=32)
                tool = self._safe_fragment(getattr(record, "tool", ""), limit=32)
                model = self._safe_fragment(getattr(record, "model", ""), limit=80)
                effort = self._safe_fragment(getattr(record, "effort", ""), limit=24)
                summary = self._safe_fragment(
                    getattr(record, "safe_summary_text", ""),
                    limit=220,
                )
                tokens = getattr(record, "total_tokens", 0)
                token_text = str(tokens) if type(tokens) is int and tokens >= 0 else "?"
                rows.append(
                    f"{ended} · {status} · {tool}/{model or 'default'}/{effort} · "
                    f"tokens={token_text} · {summary}"
                )
            content = "\n".join(rows)
        name = self._safe_fragment(getattr(employee, "name", target), limit=80)
        self.reply_text(
            message_id,
            self._bounded_plaintext(f"{name} · 最近 7 天执行历史\n\n{content}"),
        )

    def show_employee_memory(
        self,
        message_id: str,
        chat_id: str,
        target: str,
    ) -> None:
        identity = self._admin_request_context(message_id)
        if identity is None:
            return
        if not isinstance(target, str) or not target.strip():
            self.reply_text(message_id, _MEMORY_USAGE)
            return
        sender_id, _union_id, tenant_key = identity
        try:
            employee = self._resolve_employee(tenant_key, target)
        except Exception as exc:
            logger.error(
                "employee memory target resolution failed: %s",
                type(exc).__name__,
            )
            employee = None
        if employee is None:
            self.reply_text(message_id, "未找到唯一的在职员工；请核对名称或员工 ID。")
            return
        data = getattr(self.ctx, "employee_data_composition", None)
        query = getattr(getattr(data, "memory_query", None), "query", None)
        if not callable(query):
            self.reply_text(message_id, "员工记忆暂不可用，请稍后重试。")
            return
        agent_id = str(getattr(employee, "agent_id", ""))
        try:
            result = query(
                self._employee_data_request(
                    sender_id=sender_id,
                    tenant_key=tenant_key,
                    employee=employee,
                    chat_id=chat_id,
                ),
                MemoryQuerySpec(agent_id=agent_id, full_l1=True),
            )
        except QueryDeniedError:
            self.reply_text(message_id, "权限不足，无法读取该员工完整记忆。")
            return
        except AuditFailedError:
            logger.error("employee memory audit failed closed")
            self.reply_text(message_id, "员工记忆审计暂不可用；未返回任何数据。")
            return
        except Exception as exc:
            logger.error(
                "employee memory query failed closed: %s",
                type(exc).__name__,
            )
            self.reply_text(message_id, "员工记忆暂不可用，请稍后重试。")
            return
        name = self._safe_fragment(getattr(employee, "name", target), limit=80)
        content = getattr(result, "content", "") or "当前员工暂无 L1 记忆。"
        self.reply_text(
            message_id,
            self._bounded_plaintext(f"{name} · 权威 L1 记忆\n\n{content}"),
        )

    @staticmethod
    def _display_entry(employee: Any) -> dict[str, object]:
        state = getattr(employee, "state", "")
        state_value = getattr(state, "value", state)
        groups = getattr(employee, "member_groups", ())
        if not isinstance(groups, (tuple, list, set, frozenset)):
            groups = ()
        return {
            "agent_id": getattr(employee, "agent_id", ""),
            "name": getattr(employee, "name", ""),
            "emoji": getattr(employee, "emoji", "🤖"),
            "state": state_value,
            "role": getattr(employee, "role", ""),
            "tool": getattr(employee, "tool", ""),
            "model": getattr(employee, "model", ""),
            "profile": getattr(employee, "profile", ""),
            "effort": getattr(employee, "effort", ""),
            "group_count": len(groups),
            "created_at": getattr(employee, "created_at", 0.0),
        }

    def list_employees_roster(
        self,
        message_id: str,
        chat_id: str,
        project: "ProjectContext | None" = None,
    ) -> None:
        """Render every visible employee from the caller's tenant projection."""

        del chat_id, project
        resolver = self.ctx.tenant_key_resolver
        service = self.ctx.employee_hire_service
        list_roster = getattr(service, "list_employee_roster", None)
        if not callable(resolver) or not callable(list_roster):
            self.reply_text(message_id, _ROSTER_UNAVAILABLE)
            return

        try:
            tenant_key = resolver()
            if not isinstance(tenant_key, str) or not tenant_key.strip():
                raise RuntimeError("employee roster tenant is unavailable")
            employees = list_roster(tenant_key.strip(), include_archived=False)
            entries = [self._display_entry(employee) for employee in employees]
            _msg_type, card_content = SystemBuilder.build_employee_roster_card(entries)
        except Exception as exc:
            logger.error(
                "employee roster projection unavailable: %s",
                type(exc).__name__,
            )
            self.reply_text(message_id, _ROSTER_UNAVAILABLE)
            return

        self.reply_card(message_id, card_content)

    @staticmethod
    def _admin_ids() -> frozenset[str]:
        try:
            raw = getattr(get_settings(), "admin_user_ids", frozenset())
        except Exception:
            return frozenset()
        if isinstance(raw, str):
            return frozenset(item.strip() for item in raw.split(",") if item.strip())
        if isinstance(raw, (set, frozenset, tuple, list)):
            return frozenset(item for item in raw if isinstance(item, str) and item)
        return frozenset()

    def update_employee_role(
        self,
        message_id: str,
        chat_id: str,
        args: str,
    ) -> None:
        """Update one tenant employee role through the Journal-backed service."""

        del chat_id
        sender_id = get_current_sender_id() or ""
        if not sender_id or sender_id not in self._admin_ids():
            self.reply_text(message_id, "仅 GhostAP 管理员可以设置数字员工职责。")
            return

        parts = (args or "").strip().split(maxsplit=1)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            self.reply_text(message_id, _ROLE_USAGE)
            return
        employee_name, role = (part.strip() for part in parts)

        resolver = self.ctx.tenant_key_resolver
        service = self.ctx.employee_hire_service
        update_role = getattr(service, "update_employee_role", None)
        if not callable(resolver) or not callable(update_role):
            self.reply_text(message_id, _ROLE_UNAVAILABLE)
            return
        try:
            tenant_key = resolver()
        except Exception as exc:
            logger.error(
                "employee role tenant resolution failed: %s",
                type(exc).__name__,
            )
            self.reply_text(message_id, _ROLE_UNAVAILABLE)
            return
        if not isinstance(tenant_key, str) or not tenant_key.strip():
            self.reply_text(message_id, _ROLE_UNAVAILABLE)
            return
        try:
            updated = update_role(
                EmployeeRoleUpdateRequest(
                    tenant_key=tenant_key.strip(),
                    employee=employee_name,
                    role=role,
                    requester_principal_id=sender_id,
                    message_id=message_id,
                )
            )
        except Exception as exc:
            logger.error(
                "employee role update failed: %s",
                type(exc).__name__,
            )
            self.reply_text(
                message_id,
                "职责更新失败：请确认员工存在、职责有效且当前状态允许修改。",
            )
            return

        display_name = (
            getattr(updated, "name", "")
            or getattr(updated, "employee_name", "")
            or employee_name
        )
        display_role = getattr(updated, "role", "") or role
        self.reply_text(
            message_id,
            f"✅ 已将数字员工 **{display_name}** 的职责更新为：{display_role}",
        )


__all__ = ["EmployeeHandler"]
