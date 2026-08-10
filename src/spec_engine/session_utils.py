"""Session-related utility functions for Spec Engine."""

import logging
from typing import Callable, Optional

from ..acp import ACPEvent
from ..agent_session import create_engine_session
from ..coco_model import get_coco_model_manager
from ..engine_base import EngineRunState
from ..utils.errors import get_error_detail
from ..utils.retry import RetryPolicy

logger = logging.getLogger(__name__)


def send_prompt_with_retry(
    session,
    prompt: str,
    *,
    on_event: Optional[Callable[[ACPEvent], None]] = None,
    timeout: Optional[int] = None,
    retry_policy: Optional[RetryPolicy] = None,
    before_retry: Optional[Callable[[int, Exception], None]] = None,
    total_timeout: Optional[float] = None,
):
    if not session:
        raise RuntimeError("Spec session is not initialized")

    sender = getattr(session, "send_prompt_with_retry", None)
    if callable(sender):
        return sender(
            prompt,
            on_event=on_event,
            timeout=timeout,
            retry_policy=retry_policy,
            before_retry=before_retry,
            total_timeout=total_timeout,
        )

    fallback_sender = getattr(session, "send_prompt", None)
    if not callable(fallback_sender):
        raise AttributeError("session has neither send_prompt_with_retry nor send_prompt")

    return fallback_sender(prompt, on_event=on_event, timeout=timeout)


def try_switch_model(
    *,
    agent_type: str,
    run_state: EngineRunState,
    models_tried: list[str],
    current_model: Optional[str],
    root_path: str,
    model_name: Optional[str],
    on_rate_limit: Optional[Callable[[int], None]],
    close_session_fn: Callable,
    callbacks,
) -> tuple[bool, Optional[str], Optional[str], list[str], object]:
    if run_state != EngineRunState.RUNNING:
        return False, current_model, None, models_tried, None

    agent_type = str(agent_type or "").strip().lower()

    if agent_type == "claude":
        return False, current_model, None, models_tried, None

    model_manager = get_coco_model_manager()
    result = model_manager.get_models()
    all_models = [m.name for m in result.models]
    apply_switch = model_manager.set_model

    available = [m for m in all_models if m not in models_tried]
    if not available:
        return False, current_model, None, models_tried, None

    old_model = current_model or "(unknown)"
    new_model = available[0]

    if not apply_switch(new_model):
        return False, current_model, None, models_tried, None

    models_tried.append(new_model)
    new_current_model = new_model

    close_session_fn()
    new_session = create_engine_session(
        agent_type=agent_type,
        cwd=root_path,
        on_rate_limit=on_rate_limit,
        model_name=new_current_model or model_name,
    )

    logger.info("[Spec] 模型切换: %s -> %s", old_model, new_model)
    if callbacks.on_model_switch:
        callbacks.on_model_switch(old_model, new_model)

    return True, new_current_model, new_model, models_tried, new_session


def recreate_session_best_effort(
    *,
    agent_type: str,
    root_path: str,
    on_rate_limit: Optional[Callable[[int], None]],
    current_model: Optional[str],
    model_name: Optional[str],
    close_session_fn: Callable,
) -> object:
    logger.info("[Spec] 正在重建 ACP Session...")
    close_session_fn()
    try:
        new_session = create_engine_session(
            agent_type=agent_type,
            cwd=root_path,
            on_rate_limit=on_rate_limit,
            model_name=current_model or model_name,
        )
        logger.info("[Spec] ACP Session 重建成功")
        return new_session
    except Exception as e:
        logger.warning("[Spec] ACP Session 重建失败: %s", get_error_detail(e))
        return None


def initialize_model_context(agent_type: str) -> tuple[Optional[str], list[str]]:
    agent_type = str(agent_type or "").strip().lower()

    if agent_type == "claude":
        return None, []

    current_model = get_coco_model_manager().get_current_model()
    return current_model, [current_model] if current_model else []


def build_runtime_context(
    agent_type: str,
    engine_name: str,
    model_name: Optional[str],
    current_model: Optional[str],
    models_tried: list[str],
    infer_engine_name_fn: Callable,
) -> dict:
    return {
        "agent_type": str(agent_type or "").strip().lower() or "coco",
        "engine_name": engine_name or infer_engine_name_fn(agent_type),
        "model_name": model_name,
        "current_model": current_model,
        "models_tried": list(models_tried),
    }


def restore_runtime_context(
    runtime_context: Optional[dict],
    *,
    agent_type: str,
    engine_name: str,
    model_name: Optional[str],
    current_model: Optional[str],
    models_tried: list[str],
    infer_engine_name_fn: Callable,
    initialize_model_context_fn: Callable,
    on_rate_limit: Optional[Callable[[int], None]] = None,
) -> dict:
    runtime = dict(runtime_context or {})
    new_agent_type = str(runtime.get("agent_type") or agent_type or "coco").strip().lower() or "coco"
    new_engine_name = str(runtime.get("engine_name") or engine_name or infer_engine_name_fn(new_agent_type)).strip() or infer_engine_name_fn(new_agent_type)

    restored_models = [
        str(m).strip() for m in (runtime.get("models_tried") or []) if str(m).strip()
    ]
    restored_current_model = str(runtime.get("current_model") or "").strip() or None
    restored_model_name = str(runtime.get("model_name") or "").strip() or None

    new_model_name = restored_model_name or restored_current_model or model_name
    new_current_model = restored_current_model or current_model
    new_models_tried = restored_models

    if new_current_model and new_current_model not in new_models_tried:
        new_models_tried.append(new_current_model)
    if not new_current_model and new_models_tried:
        new_current_model = new_models_tried[-1]
    if not new_model_name:
        new_model_name = new_current_model

    if not new_models_tried and new_agent_type != "claude":
        new_current_model, new_models_tried = initialize_model_context_fn()

    return {
        "agent_type": new_agent_type,
        "engine_name": new_engine_name,
        "model_name": new_model_name,
        "current_model": new_current_model,
        "models_tried": new_models_tried,
        "on_rate_limit": on_rate_limit,
    }
