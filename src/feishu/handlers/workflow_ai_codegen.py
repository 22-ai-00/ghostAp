"""Workflow AI script codegen and output extraction helpers."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import threading

    from .workflow import _WorkflowGenerationActivityBuffer, _WorkflowLifecycleOwner

logger = logging.getLogger(__name__)


def _workflow_module():
    """Return the workflow handler module for runtime-patchable symbols."""
    from . import workflow as _wf_mod

    return _wf_mod


class WorkflowAICodegenMixin:
    """Mixin providing AI-driven workflow script generation and extraction."""

    def _generate_script_via_ai(
        self,
        requirement: str,
        root_path: str,
        selected_tools: list[str] | None = None,
        engine: Any = None,
        progress_callback: Any = None,
        *,
        output_path: str,
        cancel_event: threading.Event | None = None,
        artifact_lock: Any = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate through the confirmed pool with strict attempt fencing."""
        del selected_tools
        from ...agent_session import create_engine_session
        from ...workflow_engine.constants import SCRIPT_GEN_TIMEOUT_S
        from ...workflow_engine.models import WorkflowStatus
        from ...workflow_engine.script_gen import (
            build_script_gen_prompt,
            extract_meta_from_script,
            generate_simple_script,
            validate_generated_script,
        )
        from ...workflow_engine.tool_registry import get_available_tools

        if engine is None or engine.project is None or engine.project.pending is None:
            raise RuntimeError("Workflow generation session is missing")
        pending = engine.project.pending
        pool = tuple(pending.agent_pool or ())
        if not pool:
            raise RuntimeError("Workflow generation requires a confirmed Agent Pool")
        orchestrator_agent_id = pending.orchestrator_agent_id or pool[0].agent_id
        orchestrator = next(
            (item for item in pool if item.agent_id == orchestrator_agent_id),
            None,
        )
        if orchestrator is None:
            raise RuntimeError("Workflow orchestrator is outside the confirmed Agent Pool")

        registered_tools = get_available_tools(require_available=True)
        unavailable = sorted({item.tool_name for item in pool} - set(registered_tools))
        if unavailable:
            raise RuntimeError("Confirmed Workflow tool is unavailable: " + ", ".join(unavailable))
        prompt_tools = {
            item.tool_name: str(registered_tools[item.tool_name])
            for item in pool
        }
        base_prompt = build_script_gen_prompt(
            requirement=requirement,
            available_tools=prompt_tools,
            orchestrator_agent=orchestrator.tool_name,
            orchestrator_binding=orchestrator.to_dict(),
            review_agents=[],
            auto_reviewer=True,
            agent_pool=pool,
            orchestrator_agent_id=orchestrator_agent_id,
        )

        script_dir = _workflow_module().workflow_scripts_dir(root_path)
        os.makedirs(script_dir, exist_ok=True)
        if (
            os.path.realpath(os.path.dirname(output_path)) != os.path.realpath(script_dir)
            or not os.path.basename(output_path).endswith(".js")
        ):
            raise ValueError("Workflow generation output must stay in the script staging directory")

        captured_session_key = pending.engine_session_key
        started_at = _workflow_module().time.monotonic()
        hard_deadline = started_at + float(SCRIPT_GEN_TIMEOUT_S)
        configured_attempt_timeout = float(
            getattr(
                self.settings,
                "workflow_script_gen_timeout_s",
                SCRIPT_GEN_TIMEOUT_S,
            )
            or SCRIPT_GEN_TIMEOUT_S
        )
        if configured_attempt_timeout <= 0:
            configured_attempt_timeout = float(SCRIPT_GEN_TIMEOUT_S)
        ordered_pool = (
            orchestrator,
            *(item for item in pool if item.agent_id != orchestrator.agent_id),
        )

        def ensure_current() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise _workflow_module()._WorkflowGenerationCancelled("Workflow generation cancelled")
            with engine._lock:
                current_project = engine.project
                current_pending = current_project.pending if current_project else None
                if (
                    current_project is None
                    or current_project.status != WorkflowStatus.GENERATING_SCRIPT
                    or current_pending is not pending
                    or current_pending.engine_session_key != captured_session_key
                ):
                    raise RuntimeError("stale Workflow generation session was superseded")

        def _owner_is_current_locked(owner: _WorkflowLifecycleOwner) -> bool:
            current_project = engine.project
            current_pending = current_project.pending if current_project else None
            return bool(
                vars(engine).get("_script_generation_owner") is owner
                and not owner.stop_event.is_set()
                and (cancel_event is None or not cancel_event.is_set())
                and owner.session_key == captured_session_key
                and current_project is not None
                and current_project.status == WorkflowStatus.GENERATING_SCRIPT
                and current_pending is pending
                and current_pending.engine_session_key == captured_session_key
            )

        def current_generation_owner() -> _WorkflowLifecycleOwner | None:
            with engine._lock:
                owner = vars(engine).get("_script_generation_owner")
                if (
                    isinstance(owner, _workflow_module()._WorkflowLifecycleOwner)
                    and _owner_is_current_locked(owner)
                ):
                    return owner
                return None

        def bind_active_session(
            owner: _WorkflowLifecycleOwner,
            session: Any,
        ) -> bool:
            with engine._lock:
                if not _owner_is_current_locked(owner):
                    return False
                active_session = owner.active_generation_session
                if active_session is not None and active_session is not session:
                    return False
                object.__setattr__(owner, "active_generation_session", session)
                return True

        def active_session_is_current(
            owner: _WorkflowLifecycleOwner,
            session: Any,
        ) -> bool:
            with engine._lock:
                return bool(
                    _owner_is_current_locked(owner)
                    and owner.active_generation_session is session
                )

        def anchor_session_for_cleanup(
            owner: _WorkflowLifecycleOwner,
            session: Any,
        ) -> None:
            with engine._lock:
                self._retire_workflow_owner(engine, owner)
                active_session = owner.active_generation_session
                if active_session is None or active_session is session:
                    object.__setattr__(owner, "active_generation_session", session)

        def unbind_active_session(
            owner: _WorkflowLifecycleOwner,
            session: Any,
        ) -> None:
            with engine._lock:
                if owner.active_generation_session is session:
                    object.__setattr__(owner, "active_generation_session", None)

        session_owner: _WorkflowLifecycleOwner | None = None

        def close_strict(
            session: Any,
            *,
            cancel: bool,
            quarantine_event: threading.Event,
            provider_cancel_event: threading.Event,
        ) -> None:
            quarantine_event.set()
            cancel_confirmed = True
            close_confirmed = True
            details: list[str] = []
            if cancel:
                provider_cancel_event.set()
                try:
                    cancelled = session.cancel(wait=True, timeout=2.0)
                    if cancelled is False:
                        cancel_confirmed = False
                        details.append("generation session cancellation was not confirmed")
                except Exception as exc:
                    cancel_confirmed = False
                    details.append(f"generation session cancellation failed: {exc}")
            try:
                session.close()
            except Exception as exc:
                close_confirmed = False
                details.append(f"generation session close uncertain: {exc}")
            if not (cancel_confirmed and close_confirmed):
                logger.warning(
                    "Workflow generation session cleanup was not confirmed; "
                    "continuing with the next frozen-pool member: %s",
                    "; ".join(details),
                )
            if session_owner is not None:
                unbind_active_session(session_owner, session)
                release_owner = getattr(engine, "release_lifecycle_owner", None)
                if callable(release_owner):
                    release_owner(session_owner)

        def transport_failure(exc: BaseException) -> bool:
            text = f"{type(exc).__name__}: {exc}".lower()
            return isinstance(exc, (TimeoutError, ConnectionError, OSError)) or any(
                token in text
                for token in (
                    "timeout",
                    "timed out",
                    "transport",
                    "disconnect",
                    "connection",
                    "rate limit",
                    "ratelimit",
                    "too many requests",
                    "429",
                )
            )

        def provider_rejection(exc: BaseException | str) -> bool:
            text = str(exc).lower()
            return any(
                token in text
                for token in ("permission", "forbidden", "unauthorized")
            )

        def validate_and_write(script_content: str) -> tuple[str, dict[str, Any]]:
            is_valid, errors = validate_generated_script(
                script_content,
                review_agents=[],
                agent_pool=pool,
            )
            if not is_valid:
                raise ValueError(
                    "; ".join(errors[:3]) or "script validation failed"
                )
            meta = extract_meta_from_script(script_content) or {}
            unsupported = sorted(
                set(meta.get("tools", [])) - {item.tool_name for item in pool}
            )
            if unsupported:
                raise ValueError(
                    "脚本引用未确认工具: " + ", ".join(unsupported)
                )
            ensure_current()
            lock_context = artifact_lock if artifact_lock is not None else nullcontext()
            with lock_context:
                ensure_current()
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(script_content)
            return output_path, meta

        last_error = "模型未返回有效脚本"
        for binding_index, binding in enumerate(ordered_pool):
            ensure_current()
            now = _workflow_module().time.monotonic()
            remaining_hard_budget = hard_deadline - now
            if remaining_hard_budget <= 0:
                last_error = "Workflow generation total timeout exceeded"
                break
            remaining_bindings = len(ordered_pool) - binding_index
            member_deadline = now + (
                remaining_hard_budget / max(1, remaining_bindings)
            )
            for repair_attempt in range(1, _workflow_module()._SCRIPT_GENERATION_MAX_ATTEMPTS + 1):
                ensure_current()
                now = _workflow_module().time.monotonic()
                remaining = hard_deadline - now
                member_remaining = member_deadline - now
                if remaining <= 0:
                    last_error = "Workflow generation total timeout exceeded"
                    break
                if member_remaining <= 0:
                    last_error = f"{binding.agent_id} generation slice exhausted"
                    break
                if progress_callback:
                    progress_callback(
                        f"{binding.agent_id} 正在生成编排脚本（修复 {repair_attempt}/"
                        f"{_workflow_module()._SCRIPT_GENERATION_MAX_ATTEMPTS}）"
                    )

                retry_note = ""
                if repair_attempt > 1:
                    retry_note = (
                        "\n\nBounded repair attempt "
                        f"{repair_attempt}: the previous output failed validation: {last_error}. "
                        "Return one complete corrected JavaScript module."
                    )
                session = None
                should_fallback = False
                activity_buffer = _workflow_module()._WorkflowGenerationActivityBuffer()
                attempt_cancel_event = _workflow_module().threading.Event()
                attempt_quarantined = _workflow_module().threading.Event()
                try:
                    session_owner = current_generation_owner()
                    if session_owner is None:
                        raise RuntimeError(
                            "stale Workflow generation owner before session creation"
                        )
                    session = create_engine_session(
                        agent_type=binding.tool_name,
                        cwd=root_path,
                        thread_id=(
                            f"workflow_script_gen_{captured_session_key}_{binding.agent_id}_"
                            f"{repair_attempt}"
                        ),
                        model_name=binding.model_name,
                        cancel_event=attempt_cancel_event,
                        # Member timeout/startup failure is an expected input
                        # to frozen-pool fallback, not an unhandled service
                        # failure. This handler reports the bounded fallback.
                        startup_log_failures=False,
                    )
                    if session is None:
                        last_error = f"{binding.agent_id}: unable to create generation session"
                        should_fallback = True
                        break
                    if not bind_active_session(session_owner, session):
                        anchor_session_for_cleanup(session_owner, session)
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        raise RuntimeError(
                            "stale Workflow generation owner before session bind"
                        )
                    if not active_session_is_current(session_owner, session):
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        raise RuntimeError(
                            "stale Workflow generation owner before prompt send"
                        )

                    def on_event(
                        event: Any,
                        *,
                        _session: Any = session,
                        _session_owner: _WorkflowLifecycleOwner = session_owner,
                        _quarantine_event: threading.Event = attempt_quarantined,
                        _activity_buffer: _WorkflowGenerationActivityBuffer = activity_buffer,
                        _binding_agent_id: str = binding.agent_id,
                        _progress_callback: Any = progress_callback,
                    ) -> None:
                        if (
                            _quarantine_event.is_set()
                            or not active_session_is_current(
                                _session_owner,
                                _session,
                            )
                        ):
                            return
                        text = str(getattr(event, "text", "") or "")
                        event_type = getattr(event, "event_type", None)
                        event_name = str(getattr(event_type, "value", event_type) or "").lower()
                        meaningful = self._workflow_generation_event_is_meaningful(event)
                        if not meaningful or not _progress_callback:
                            return
                        summaries = _activity_buffer.feed(text) if text else ()
                        if not text:
                            summaries = (
                                *_activity_buffer.flush(),
                                event_name.replace("_", " "),
                            )
                        for summary in summaries:
                            _progress_callback(
                                f"{_binding_agent_id} last activity: {summary[:180]}"
                            )

                    try:
                        attempt_timeout = max(
                            0.001,
                            min(
                                configured_attempt_timeout,
                                remaining,
                                member_remaining,
                            ),
                        )
                        result = session.send_prompt(
                            base_prompt + retry_note,
                            timeout=attempt_timeout,
                            idle_timeout=max(0.001, min(120.0, attempt_timeout)),
                            on_event=on_event,
                            activity_predicate=self._workflow_generation_event_is_meaningful,
                        )
                    finally:
                        if progress_callback:
                            for summary in activity_buffer.flush():
                                progress_callback(
                                    f"{binding.agent_id} last activity: {summary[:180]}"
                                )
                    try:
                        ensure_current()
                    except Exception:
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        raise

                    stop_reason = str(getattr(result, "stop_reason", "") or "").strip().lower()
                    if stop_reason in {"cancelled", "canceled"}:
                        last_error = (
                            f"{binding.agent_id} terminal stop_reason={stop_reason}"
                        )
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        should_fallback = True
                        break
                    if stop_reason in {"timeout", "timed_out", "error", "failed", "failure"}:
                        last_error = f"{binding.agent_id} terminal stop_reason={stop_reason}"
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        should_fallback = True
                        break
                    if stop_reason not in {
                        "end_turn",
                        "completed",
                        "complete",
                        "success",
                        "done",
                    }:
                        last_error = f"{binding.agent_id} terminal stop_reason={stop_reason}"
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        should_fallback = True
                        break
                    text = str(getattr(result, "text", "") or "").strip()
                    if not text:
                        last_error = f"{binding.agent_id}: model returned an empty script"
                        close_strict(
                            session,
                            cancel=False,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        continue
                    script_content = self._strip_markdown_fences(text)
                    is_valid, errors = validate_generated_script(
                        script_content,
                        review_agents=[],
                        agent_pool=pool,
                    )
                    if not is_valid:
                        last_error = "; ".join(errors[:3]) or "script validation failed"
                        close_strict(
                            session,
                            cancel=False,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        continue
                    meta = extract_meta_from_script(script_content) or {}
                    unsupported = sorted(
                        set(meta.get("tools", [])) - {item.tool_name for item in pool}
                    )
                    if unsupported:
                        last_error = "脚本引用未确认工具: " + ", ".join(unsupported)
                        close_strict(
                            session,
                            cancel=False,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        continue

                    close_strict(
                        session,
                        cancel=False,
                        quarantine_event=attempt_quarantined,
                        provider_cancel_event=attempt_cancel_event,
                    )
                    session = None
                    return validate_and_write(script_content)
                except _workflow_module()._WorkflowGenerationCancelled:
                    if session is not None:
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                    raise
                except Exception as exc:
                    if session is not None:
                        close_strict(
                            session,
                            cancel=True,
                            quarantine_event=attempt_quarantined,
                            provider_cancel_event=attempt_cancel_event,
                        )
                        session = None
                        if transport_failure(exc) or provider_rejection(exc):
                            session = None
                            last_error = f"{type(exc).__name__}: {exc}"
                            should_fallback = True
                            break
                    if "stale" in str(exc).lower() or "superseded" in str(exc).lower():
                        raise
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "Workflow generation %s repair %s failed: %s",
                        binding.agent_id,
                        repair_attempt,
                        exc,
                    )
                    continue
                if should_fallback:
                    break

        ensure_current()
        if progress_callback:
            progress_callback(
                "模型编排未完成，正在切换池内确定性编排并继续执行"
            )
        try:
            fallback_script = generate_simple_script(
                requirement,
                agent_pool=pool,
                orchestrator_agent_id=orchestrator_agent_id,
            )
            ensure_current()
            return validate_and_write(fallback_script)
        except _workflow_module()._WorkflowGenerationCancelled:
            raise
        except Exception as exc:
            if "stale" in str(exc).lower() or "superseded" in str(exc).lower():
                raise
            logger.error(
                "Workflow deterministic pool fallback failed after model generation: %s; "
                "last model error: %s",
                exc,
                last_error,
                exc_info=True,
            )
            raise _workflow_module()._WorkflowGenerationExhausted(
                "Workflow 自动编排恢复未完成，请重新发起 `/wf`。"
            ) from exc

    @staticmethod
    def _workflow_generation_event_is_meaningful(event: Any) -> bool:
        """Return True if an ACP generation event carries user-visible activity."""
        text = str(getattr(event, "text", "") or "").strip()
        if text:
            return True
        event_type = getattr(event, "event_type", None)
        event_name = str(getattr(event_type, "value", event_type) or "").lower()
        return any(
            token in event_name
            for token in (
                "tool_call_start",
                "tool_call_update",
                "tool_call_done",
                "plan_update",
                "image_chunk",
            )
        )

    @staticmethod
    def _strip_markdown_fences(content: str) -> str:
        """Remove markdown code fences and natural language preamble from AI output.

        AI models sometimes prefix their code output with explanatory text like
        "Let me analyze..." or "Here's the workflow script:". This method
        extracts the actual JavaScript code by:
        1. Attempting to extract code from markdown fences (even if preceded by text)
        2. Stripping any natural language preamble before the actual JS code
        """
        import re

        # Strategy 1: Find markdown code fence containing the actual code.
        # This handles cases like: "Here's the script:\n```javascript\n...code...\n```"
        fence_match = re.search(r"```\s*(?:javascript|js|)\s*\n", content, re.IGNORECASE)
        if fence_match:
            after_fence = content[fence_match.end() :]
            # Find the closing fence (last occurrence to handle nested fences in strings)
            close_idx = after_fence.rfind("```")
            if close_idx >= 0:
                content = after_fence[:close_idx].rstrip()
            else:
                content = after_fence.rstrip()
            # After extracting from fences, if it looks like valid JS, return it
            stripped = content.lstrip()
            if stripped and re.match(
                r"^(export|/[/*]|const |let |var |\"use strict\"|'use strict')",
                stripped,
            ):
                return content.strip()

        # Strategy 2: Original logic — content starts directly with a fence
        elif content.startswith("```"):
            lines = content.split("\n", 1)
            content = lines[1] if len(lines) > 1 else content
            if content.rstrip().endswith("```"):
                content = content.rstrip()[:-3].rstrip()

        # Strategy 3: Detect and strip natural language preamble.
        # If content doesn't start with valid JS syntax, find the actual code start.
        stripped = content.lstrip()
        if stripped and not re.match(
            r"^(export|/[/*]|const |let |var |\"use strict\"|'use strict'|/\*\*)",
            stripped,
        ):
            # Look for the start of the actual export statement (multiline search)
            export_match = re.search(
                r"^(export\s+const\s+meta\s*=|export\s+default\s)",
                content,
                re.MULTILINE,
            )
            if export_match:
                start_idx = export_match.start()
                # Include preceding JSDoc/comment lines that are part of the code
                preceding = content[:start_idx]
                if preceding.rstrip():
                    lines_before = preceding.rstrip().split("\n")
                    # Walk backwards to include leading comment block
                    comment_start = start_idx
                    for line in reversed(lines_before):
                        ls = line.strip()
                        if ls.startswith("//") or ls.startswith("*") or ls.startswith("/*") or ls.endswith("*/"):
                            # This line is a comment, include it
                            idx = content.rfind(line, 0, comment_start)
                            if idx >= 0:
                                comment_start = idx
                        else:
                            break
                    start_idx = comment_start
                content = content[start_idx:]

        return content.strip()
