"""ACP Client implementation — handles agent callbacks.

GhostAPClient implements the ACP Client interface and converts raw ACP session
updates into ACPEvent objects, forwarding them to the registered event handler.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import mimetypes
import os
import re
import shlex
import stat
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import unquote, urlparse

from acp.interfaces import Agent, Client
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    BlobResourceContents,
    ContentToolCallContent,
    CreateTerminalResponse,
    DeniedOutcome,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    ResourceContentBlock,
    SessionInfoUpdate,
    TerminalExitStatus,
    TerminalOutputResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)

try:
    from acp.schema import KillTerminalCommandResponse
except ImportError:
    from acp.schema import KillTerminalResponse as KillTerminalCommandResponse

from ..sandbox.executor import DangerousPatternCheckStrategy, SandboxExecutor
from ..utils.errors import get_error_detail
from ..utils.text import sanitize_single_line_label
from .models import (
    ACPEvent,
    ACPEventType,
    ACPGoalInfo,
    ACPImageInfo,
    ACPSessionInfo,
    PlanEntryInfo,
    PlanInfo,
    ToolCallInfo,
)

logger = logging.getLogger(__name__)


# Default fallback; prefer Settings.acp_max_file_chars at runtime.
_MAX_FILE_CHARS = 200_000
_ACP_PERMISSION_DANGEROUS_CHECK = DangerousPatternCheckStrategy()
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHELL_CONTROL_TOKENS = frozenset({";", "&", "&&", "|", "||", "<", ">", "(", ")", "`"})
_PERMISSION_KIND_TOOL_NAMES = {
    "read": "file_read",
    "edit": "file_write",
    "delete": "file_write",
    "move": "file_write",
    "search": "search",
    "fetch": "fetch",
    "think": "think",
    "switch_mode": "switch_mode",
    "other": "other",
}
MAX_ACP_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_BASE64_IMAGE_CHARS = ((MAX_ACP_IMAGE_BYTES + 2) // 3) * 4
SUPPORTED_ACP_IMAGE_MIME_TYPES = frozenset({
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/bmp",
})
_IMAGE_MIME_ALIASES = {
    "image/jpg": "image/jpeg",
    "image/x-png": "image/png",
}
_IMAGE_FILE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_IMAGE_SCAN_IGNORED_DIRS = frozenset({
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
})
_MAX_IMAGE_SCAN_ENTRIES = 50_000
_MAX_DISCOVERED_IMAGES_PER_PROMPT = 20
_MAX_DISCOVERED_IMAGE_BYTES_PER_PROMPT = 50 * 1024 * 1024
_ImageArtifactSignature = tuple[int, int, int]
_QUOTED_IMAGE_PATH_RE = re.compile(
    r"""[`'"](?P<path>[^`'"]+\.(?:png|jpe?g|gif|webp|bmp))[`'"]""",
    re.IGNORECASE,
)
_BARE_IMAGE_PATH_RE = re.compile(
    r"""(?P<path>(?:file://[^\s`"'<>|]+|(?:/|\.\.?/)?[^\s`"'<>|]+\.(?:png|jpe?g|gif|webp|bmp)))""",
    re.IGNORECASE,
)
_MARKDOWN_IMAGE_TARGET_RE = re.compile(
    r"""!\[[^\]\r\n]*\]\((?P<target>[^)\r\n]+)\)""",
    re.IGNORECASE,
)


@dataclass(eq=False)
class LocalImageArtifactSnapshot:
    """One prompt's image baseline and same-root overlap state."""

    root_key: str
    files: dict[str, _ImageArtifactSignature]
    complete: bool = True
    conflicted: bool = False
    active: bool = True


@dataclass
class LocalImageEmissionBudget:
    """Prompt-wide limits and deduplication for changed local images."""

    max_images: int = _MAX_DISCOVERED_IMAGES_PER_PROMPT
    max_bytes: int = _MAX_DISCOVERED_IMAGE_BYTES_PER_PROMPT
    seen_image_ids: set[str] = field(default_factory=set)
    emitted_count: int = 0
    emitted_bytes: int = 0


_IMAGE_SNAPSHOT_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
_ACTIVE_IMAGE_SNAPSHOTS: dict[str, list[LocalImageArtifactSnapshot]] = {}


def _image_roots_overlap(first: str, second: str) -> bool:
    """Return whether canonical roots share an ancestor/descendant tree."""
    first_path = Path(first)
    second_path = Path(second)
    try:
        first_path.relative_to(second_path)
        return True
    except ValueError:
        pass
    try:
        second_path.relative_to(first_path)
        return True
    except ValueError:
        return False


def _permission_execute_tool_name(command: str) -> str:
    """Classify a plain Git invocation without treating shell wrappers as Git."""
    if "\n" in command or "\r" in command:
        return "shell"
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()`")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except (TypeError, ValueError):
        return "shell"
    if not tokens or os.path.basename(tokens[0]).casefold() != "git":
        return "shell"
    if any(token in _SHELL_CONTROL_TOKENS for token in tokens[1:]):
        return "shell"
    return "git"


def _permission_tool_request(
    tool_call: Any,
    *,
    root_dir: str,
) -> tuple[str, dict, Optional[str]]:
    """Map every ACP permission kind to a stable tool-filter request."""

    raw_input = getattr(tool_call, "raw_input", None)
    if isinstance(raw_input, Mapping):
        tool_args = dict(raw_input)
    elif raw_input is None:
        tool_args = {}
    else:
        tool_args = {"raw_input": raw_input}

    raw_kind = getattr(tool_call, "kind", None)
    kind = raw_kind.strip().casefold() if isinstance(raw_kind, str) else ""
    if kind != "execute":
        return _PERMISSION_KIND_TOOL_NAMES.get(kind, "other"), tool_args, None

    command_value: Any = None
    if isinstance(raw_input, Mapping):
        command_value = (
            raw_input.get("command")
            or raw_input.get("cmd")
            or raw_input.get("shell_command")
        )
    elif isinstance(raw_input, str):
        command_value = raw_input
    if command_value and not isinstance(command_value, str):
        raise ValueError("execute command must be a string")
    command = command_value if isinstance(command_value, str) and command_value else None
    if command is None:
        return "execute", tool_args, None

    tool_args["command"] = command
    tool_args.setdefault("cwd", root_dir)
    return _permission_execute_tool_name(command), tool_args, command


def _get_max_file_chars() -> int:
    """Read acp_max_file_chars from Settings (lazy, best-effort)."""
    try:
        from ..config import get_settings
        return getattr(get_settings(), "acp_max_file_chars", _MAX_FILE_CHARS)
    except Exception:
        return _MAX_FILE_CHARS


def _safe_session_filename(session_id: str) -> str:
    """Make a safe filename from session_id."""
    sid = (session_id or "").strip() or "unknown"
    sid = re.sub(r"[^a-zA-Z0-9._-]+", "_", sid)
    return sid[:120]


class ACPHistoryStore:
    """Local persistence for ACP session history (jsonl).

    Stores command execution results and file operations so that GhostAP can
    display/recover historical info even if the agent-side session store is
    unavailable.
    """

    def __init__(self, base_dir: Optional[str] = None):
        from ..config import get_settings
        settings_dir = get_settings().acp_history_dir.strip()
        root = base_dir or settings_dir
        if not root:
            root = str(Path.home() / ".ghostap" / "acp_history")
        self._base = Path(root).expanduser()

    def _path_for(self, session_id: str) -> Path:
        name = _safe_session_filename(session_id)
        return self._base / f"{name}.jsonl"

    def append(self, session_id: str, entry: dict) -> None:
        if not session_id:
            return
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            p = self._path_for(session_id)
            payload = dict(entry or {})
            payload.setdefault("ts", time.time())
            payload.setdefault("session_id", session_id)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("[ACP] history append failed: %s", get_error_detail(e))

    def load(self, session_id: str, limit: int = 200) -> list[dict]:
        if not session_id:
            return []
        p = self._path_for(session_id)
        if not p.exists():
            return []

        try:
            raw = p.read_text(encoding="utf-8")
        except Exception as e:
            logger.info("[ACP] history read failed: %s", get_error_detail(e))
            return []

        raw_strip = raw.lstrip()
        # Backward compatibility: accept a single JSON array file.
        if raw_strip.startswith("["):
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    items = [x for x in data if isinstance(x, dict)]
                    return items[-limit:] if limit > 0 else items
            except (ValueError, json.JSONDecodeError):
                # Fall through to jsonl parsing
                pass

        items: list[dict] = []
        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    items.append(obj)
            except (ValueError, json.JSONDecodeError):
                # Corrupt line: skip
                continue

        return items[-limit:] if limit > 0 else items


def _safe_resolve_path(root_dir: str, user_path: str) -> Path:
    """Resolve a user-supplied path inside root_dir.

    Supports both absolute and relative paths, but always enforces:
    resolved_path must be within root_dir.
    """
    root = Path(root_dir).expanduser().resolve()
    p = Path(user_path).expanduser()
    if not p.is_absolute():
        p = root / p
    resolved = p.resolve()
    try:
        resolved.relative_to(root)
    except Exception as e:
        raise PermissionError(f"path escapes root_dir: {user_path}") from e
    return resolved


def _image_name(uri: str | None) -> str:
    if not uri:
        return "任务图片"
    parsed = urlparse(uri)
    name = Path(unquote(parsed.path or "")).name.strip()
    return name[:120] or "任务图片"


def _normalize_acp_image(
    *,
    data: str,
    mime_type: str,
    uri: str | None = None,
    name: str | None = None,
) -> ACPImageInfo | None:
    """Validate and normalize an ACP base64 raster payload."""
    compact = "".join(str(data or "").split())
    if not compact or len(compact) > _MAX_BASE64_IMAGE_CHARS:
        return None
    try:
        decoded = base64.b64decode(compact, validate=True)
    except (ValueError, TypeError):
        return None
    if not decoded or len(decoded) > MAX_ACP_IMAGE_BYTES:
        return None
    detected_mime = detect_acp_image_mime(decoded)
    if detected_mime is None:
        return None
    declared_mime = str(mime_type or "").strip().lower()
    normalized_mime = _IMAGE_MIME_ALIASES.get(declared_mime, declared_mime)
    if normalized_mime:
        if (
            normalized_mime not in SUPPORTED_ACP_IMAGE_MIME_TYPES
            or normalized_mime != detected_mime
        ):
            return None
    else:
        normalized_mime = detected_mime
    canonical_data = base64.b64encode(decoded).decode("ascii")
    return ACPImageInfo(
        image_id=f"sha256:{hashlib.sha256(decoded).hexdigest()}",
        mime_type=normalized_mime,
        data=canonical_data,
        name=sanitize_single_line_label(
            name or _image_name(uri),
            fallback="任务图片",
            max_chars=120,
        ),
        source_uri=uri or None,
    )


def detect_acp_image_mime(payload: bytes) -> str | None:
    """Identify supported raster bytes independently from declared metadata."""
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if (
        len(payload) >= 4
        and payload.startswith(b"\xff\xd8\xff")
        and payload.endswith(b"\xff\xd9")
    ):
        return "image/jpeg"
    if (
        len(payload) >= 14
        and payload[:6] in {b"GIF87a", b"GIF89a"}
        and payload.endswith(b";")
    ):
        return "image/gif"
    if (
        len(payload) >= 12
        and payload.startswith(b"RIFF")
        and payload[8:12] == b"WEBP"
    ):
        return "image/webp"
    if len(payload) >= 14 and payload.startswith(b"BM"):
        return "image/bmp"
    return None


def _read_bounded_file_inside_root(
    root_dir: str,
    user_path: str,
) -> tuple[bytes, Path, _ImageArtifactSignature] | None:
    """Open one canonical in-root file without following raced symlinks."""
    try:
        root = Path(root_dir).expanduser().resolve()
        resolved = _safe_resolve_path(root_dir, user_path)
        relative = resolved.relative_to(root)
    except (OSError, PermissionError, RuntimeError, ValueError):
        return None
    if not relative.parts:
        return None

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow | cloexec
    file_flags = os.O_RDONLY | nofollow | cloexec
    opened_fds: list[int] = []
    try:
        directory_fd = os.open(str(root), directory_flags)
        opened_fds.append(directory_fd)
        for part in relative.parts[:-1]:
            directory_fd = os.open(
                part,
                directory_flags,
                dir_fd=directory_fd,
            )
            opened_fds.append(directory_fd)
        file_fd = os.open(
            relative.parts[-1],
            file_flags,
            dir_fd=directory_fd,
        )
        opened_fds.append(file_fd)
        file_stat = os.fstat(file_fd)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_size > MAX_ACP_IMAGE_BYTES
        ):
            return None

        chunks: list[bytes] = []
        remaining = MAX_ACP_IMAGE_BYTES + 1
        while remaining > 0:
            chunk = os.read(file_fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if not payload or len(payload) > MAX_ACP_IMAGE_BYTES:
            return None
        signature = (
            int(file_stat.st_mtime_ns),
            int(file_stat.st_ctime_ns),
            int(file_stat.st_size),
        )
        return payload, resolved, signature
    except OSError:
        return None
    finally:
        for fd in reversed(opened_fds):
            try:
                os.close(fd)
            except OSError:
                pass


def _read_local_image_resource(
    *,
    root_dir: str,
    uri: str,
    mime_type: str | None,
    name: str | None,
    image_snapshot: LocalImageArtifactSnapshot | None = None,
    require_changed: bool = False,
) -> ACPImageInfo | None:
    """Read an ACP local image resource without allowing remote fetches."""
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        return None
    if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
        return None
    if Path(unquote(parsed.path or uri)).suffix.casefold() not in _IMAGE_FILE_SUFFIXES:
        return None
    raw_path = unquote(parsed.path) if parsed.scheme == "file" else uri
    loaded = _read_bounded_file_inside_root(root_dir, raw_path)
    if loaded is None:
        return None
    payload, path, signature = loaded
    if require_changed and not _snapshot_allows_local_image(
        root_dir=root_dir,
        path=path,
        signature=signature,
        image_snapshot=image_snapshot,
    ):
        return None
    detected_mime = mime_type or mimetypes.guess_type(path.name)[0] or ""
    return _normalize_acp_image(
        data=base64.b64encode(payload).decode("ascii"),
        mime_type=detected_mime,
        uri=str(path),
        name=name or path.name,
    )


def _local_image_candidates(value: Any) -> list[str]:
    """Extract bounded, path-shaped image references from structured tool output."""
    candidates: list[str] = []
    pending = [value]
    visited = 0
    while pending and visited < 200:
        current = pending.pop()
        visited += 1
        if isinstance(current, dict):
            pending.extend(current.values())
            continue
        if isinstance(current, (list, tuple)):
            pending.extend(current)
            continue
        if not isinstance(current, str):
            continue
        text = current[:12_000]
        for match in _MARKDOWN_IMAGE_TARGET_RE.finditer(text):
            target = match.group("target").strip()
            if target.startswith("<") and ">" in target:
                target = target[1 : target.index(">")].strip()
            elif any(char.isspace() for char in target):
                target = target.split(maxsplit=1)[0]
            if Path(unquote(urlparse(target).path or target)).suffix.casefold() in (
                _IMAGE_FILE_SUFFIXES
            ):
                candidates.append(target)
        candidates.extend(
            match.group("path").strip()
            for match in _QUOTED_IMAGE_PATH_RE.finditer(text)
        )
        candidates.extend(
            match.group("path").rstrip(".,;:!?)]}")
            for match in _BARE_IMAGE_PATH_RE.finditer(text)
        )
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _iter_local_image_files(
    root_dir: str,
    scan_complete: list[bool] | None = None,
) -> list[Path]:
    """Enumerate local raster candidates without following links or huge trees."""
    def _mark_incomplete() -> None:
        if scan_complete is not None:
            scan_complete[0] = False

    try:
        root = Path(root_dir).expanduser().resolve()
    except (OSError, RuntimeError):
        _mark_incomplete()
        return []
    if not root.is_dir():
        _mark_incomplete()
        return []

    found: list[Path] = []
    stack = [root]
    inspected = 0
    while stack:
        directory = stack.pop()
        try:
            entries = os.scandir(directory)
        except OSError:
            _mark_incomplete()
            continue
        with entries:
            for entry in entries:
                if inspected >= _MAX_IMAGE_SCAN_ENTRIES:
                    _mark_incomplete()
                    return found
                inspected += 1
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name not in _IMAGE_SCAN_IGNORED_DIRS:
                            stack.append(Path(entry.path))
                    elif (
                        entry.is_file(follow_symlinks=False)
                        and Path(entry.name).suffix.casefold() in _IMAGE_FILE_SUFFIXES
                    ):
                        found.append(Path(entry.path))
                except OSError:
                    _mark_incomplete()
                    continue
    return found


def _resolve_image_scan_root(root_dir: str) -> Path | None:
    try:
        root = Path(root_dir).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    return root if root.is_dir() else None


def _capture_local_image_artifacts(
    root_dir: str,
    scan_complete: list[bool] | None = None,
) -> dict[str, _ImageArtifactSignature]:
    snapshot: dict[str, _ImageArtifactSignature] = {}
    for path in _iter_local_image_files(root_dir, scan_complete):
        try:
            stat = path.stat()
            snapshot[str(path)] = (
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
                int(stat.st_size),
            )
        except OSError:
            if scan_complete is not None:
                scan_complete[0] = False
            continue
    return snapshot


def snapshot_local_image_artifacts(
    root_dir: str,
) -> LocalImageArtifactSnapshot:
    """Capture a prompt baseline and mark overlapping same-root scans unsafe."""
    root = _resolve_image_scan_root(root_dir)
    if root is None:
        return LocalImageArtifactSnapshot(
            root_key="",
            files={},
            complete=False,
            active=False,
        )

    snapshot = LocalImageArtifactSnapshot(root_key=str(root), files={})
    with _IMAGE_SNAPSHOT_LOCK:
        overlapping = [
            other
            for root_key, snapshots in _ACTIVE_IMAGE_SNAPSHOTS.items()
            if _image_roots_overlap(snapshot.root_key, root_key)
            for other in snapshots
        ]
        if overlapping:
            snapshot.conflicted = True
            for other in overlapping:
                other.conflicted = True
        active = _ACTIVE_IMAGE_SNAPSHOTS.setdefault(snapshot.root_key, [])
        active.append(snapshot)
    try:
        scan_complete = [True]
        snapshot.files = _capture_local_image_artifacts(
            snapshot.root_key,
            scan_complete,
        )
        snapshot.complete = scan_complete[0]
    except BaseException:
        release_local_image_artifact_snapshot(snapshot)
        raise
    return snapshot


def release_local_image_artifact_snapshot(
    snapshot: LocalImageArtifactSnapshot | Mapping[str, _ImageArtifactSignature],
) -> None:
    """Release an image baseline lease; safe to call more than once."""
    if not isinstance(snapshot, LocalImageArtifactSnapshot):
        return
    with _IMAGE_SNAPSHOT_LOCK:
        if not snapshot.active:
            return
        snapshot.active = False
        active = _ACTIVE_IMAGE_SNAPSHOTS.get(snapshot.root_key)
        if active is None:
            return
        try:
            active.remove(snapshot)
        except ValueError:
            pass
        if not active:
            _ACTIVE_IMAGE_SNAPSHOTS.pop(snapshot.root_key, None)


def _snapshot_allows_local_image(
    *,
    root_dir: str,
    path: Path,
    signature: _ImageArtifactSignature,
    image_snapshot: LocalImageArtifactSnapshot | None,
) -> bool:
    """Prove an explicitly referenced local image changed in this prompt."""
    if image_snapshot is None:
        return False
    resolved_root = _resolve_image_scan_root(root_dir)
    if resolved_root is None:
        return False
    try:
        resolved_path = path.resolve()
        relative_path = resolved_path.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return False
    if any(
        component in _IMAGE_SCAN_IGNORED_DIRS
        for component in relative_path.parts[:-1]
    ):
        return False
    with _IMAGE_SNAPSHOT_LOCK:
        if (
            not image_snapshot.complete
            or image_snapshot.conflicted
            or not image_snapshot.active
            or image_snapshot.root_key != str(resolved_root)
        ):
            return False
        return image_snapshot.files.get(str(resolved_path)) != signature


def emit_referenced_changed_local_image_events(
    root_dir: str,
    before: LocalImageArtifactSnapshot | Mapping[str, _ImageArtifactSignature],
    references: Any,
    on_event: Callable[[ACPEvent], None],
    *,
    budget: LocalImageEmissionBudget | None = None,
    release_snapshot: bool = True,
) -> int:
    """Publish only explicitly referenced local images changed since baseline."""
    snapshot = before if isinstance(before, LocalImageArtifactSnapshot) else None
    active_budget = budget or LocalImageEmissionBudget()
    emitted = 0
    try:
        for path in _local_image_candidates(references):
            if active_budget.emitted_count >= active_budget.max_images:
                break
            image = _read_local_image_resource(
                root_dir=root_dir,
                uri=path,
                mime_type=None,
                name=None,
                image_snapshot=snapshot,
                require_changed=True,
            )
            if image is None or image.image_id in active_budget.seen_image_ids:
                continue
            image_bytes = (len(image.data) * 3) // 4
            if active_budget.emitted_bytes + image_bytes > active_budget.max_bytes:
                break
            active_budget.seen_image_ids.add(image.image_id)
            active_budget.emitted_count += 1
            active_budget.emitted_bytes += image_bytes
            on_event(
                ACPEvent(
                    event_type=ACPEventType.IMAGE_CHUNK,
                    image=image,
                )
            )
            emitted += 1
    finally:
        if release_snapshot:
            release_local_image_artifact_snapshot(before)
    return emitted


def _parse_acp_image_content(
    content: Any,
    *,
    root_dir: str,
    image_snapshot: LocalImageArtifactSnapshot | None = None,
) -> ACPImageInfo | None:
    """Extract supported image representations from ACP content blocks."""
    if isinstance(content, ImageContentBlock):
        return _normalize_acp_image(
            data=content.data,
            mime_type=content.mime_type,
            uri=content.uri,
        )
    if isinstance(content, EmbeddedResourceContentBlock):
        resource = content.resource
        if isinstance(resource, BlobResourceContents):
            return _normalize_acp_image(
                data=resource.blob,
                mime_type=resource.mime_type or "",
                uri=resource.uri,
            )
        return None
    if isinstance(content, ResourceContentBlock):
        return _read_local_image_resource(
            root_dir=root_dir,
            uri=content.uri,
            mime_type=content.mime_type,
            name=content.title or content.name,
            image_snapshot=image_snapshot,
            require_changed=True,
        )
    return None


def _tool_call_images(
    update: ToolCallStart | ToolCallProgress,
    *,
    root_dir: str,
    image_snapshot: LocalImageArtifactSnapshot | None = None,
) -> list[ACPImageInfo]:
    status = str(update.status or "").strip().lower()
    kind = str(update.kind or "other").strip().lower() or "other"
    title = str(update.title or "").casefold()
    is_output_capable = kind not in {"read", "search", "fetch", "delete"} and not any(
        marker in title for marker in ("read image", "view image", "读取图片", "查看图片")
    )
    # Tool content is often its input/reference material on START/PROGRESS.
    # Only a completed output-capable tool may publish image artifacts.
    if status != "completed" or not is_output_capable:
        return []

    images: list[ACPImageInfo] = []
    path_candidates: list[str] = []
    for item in update.content or []:
        if not isinstance(item, ContentToolCallContent):
            continue
        image = _parse_acp_image_content(
            item.content,
            root_dir=root_dir,
            image_snapshot=image_snapshot,
        )
        if image is not None:
            images.append(image)
        elif isinstance(item.content, TextContentBlock):
            path_candidates.extend(_local_image_candidates(item.content.text))

    path_candidates.extend(
        str(location.path)
        for location in (update.locations or [])
        if getattr(location, "path", None)
    )
    path_candidates.extend(_local_image_candidates(update.raw_output))

    for path in dict.fromkeys(path_candidates):
        image = _read_local_image_resource(
            root_dir=root_dir,
            uri=path,
            mime_type=None,
            name=None,
            image_snapshot=image_snapshot,
            require_changed=True,
        )
        if image is not None:
            images.append(image)

    unique: dict[str, ACPImageInfo] = {}
    for image in images:
        unique.setdefault(image.image_id, image)
    return list(unique.values())


def _validated_env_overrides(env: Optional[list[Any]]) -> dict[str, str]:
    """Convert ACP terminal environment values without leaking them into logs."""
    overrides: dict[str, str] = {}
    for item in env or []:
        name = str(getattr(item, "name", "") or "")
        if not _ENVIRONMENT_NAME_RE.fullmatch(name):
            raise ValueError("终端环境变量名称无效")
        overrides[name] = str(getattr(item, "value", "") or "")
    return overrides


def _truncate_utf8_tail(text: str, byte_limit: Optional[int]) -> tuple[str, bool]:
    """Retain the ACP-required final byte window at a UTF-8 character boundary."""
    if byte_limit is None:
        return text, False
    limit = max(0, int(byte_limit))
    payload = (text or "").encode("utf-8")
    if len(payload) <= limit:
        return text, False
    if limit == 0:
        return "", True
    return payload[-limit:].decode("utf-8", errors="ignore"), True


_TERMINAL_TTL = 3600  # 1 hour — expired terminals are cleaned up lazily


@dataclass
class _TerminalRecord:
    output: str
    exit_code: int
    truncated: bool
    cursor: int = 0
    created_at: float = 0.0


def _format_todo_content(raw_input: Any) -> str:
    """Format TodoWrite raw_input into a readable checklist."""
    if not isinstance(raw_input, dict):
        return ""
    todos = raw_input.get("todos")
    if not isinstance(todos, list):
        return ""

    _icons = {"completed": "✅", "in_progress": "🔄", "pending": "⏳"}
    lines: list[str] = []
    for item in todos:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or ""
        status = item.get("status", "pending")
        # For in_progress items, prefer activeForm for better readability
        if status == "in_progress":
            content = item.get("activeForm") or content
        if not content:
            continue
        icon = _icons.get(status, "⬜")
        lines.append(f"{icon} {content}")

    return "\n".join(lines)


def _is_todo_tool(title: str, raw_input: Any) -> bool:
    """Check if this tool call is a TodoWrite."""
    if "todo" in (title or "").lower():
        return True
    if isinstance(raw_input, dict) and "todos" in raw_input:
        return True
    return False


def _codex_namespaced_metadata(update: Any) -> tuple[dict, dict]:
    """Return normalized Codex subagent/collaboration metadata containers."""
    meta = getattr(update, "field_meta", None) or getattr(update, "_meta", None)
    if not isinstance(meta, Mapping):
        return {}, {}
    codex = meta.get("codex")
    if not isinstance(codex, Mapping):
        return {}, {}
    subagent = codex.get("subagent")
    collaboration = codex.get("collaboration")
    return (
        dict(subagent) if isinstance(subagent, Mapping) else {},
        dict(collaboration) if isinstance(collaboration, Mapping) else {},
    )


def _parse_subagent_metadata(update: Any) -> tuple[str, str, str]:
    """Extract the stable child thread, display path, and lifecycle activity."""
    subagent, _ = _codex_namespaced_metadata(update)
    if not subagent:
        return "", "", ""
    source_id = str(subagent.get("threadId") or "").strip()
    path = str(subagent.get("path") or "").strip()
    activity = str(subagent.get("activity") or "").strip().lower()
    return source_id, path, activity


def _parse_collaboration_metadata(
    update: Any,
    raw_input: Any,
) -> tuple[str, tuple[str, ...], str, tuple[dict, ...]]:
    """Normalize Codex collaboration state without exposing provider ids."""
    _, collaboration = _codex_namespaced_metadata(update)
    if not collaboration:
        return "", (), "", ()

    tool = str(collaboration.get("tool") or "").strip()
    raw_receivers = collaboration.get("receiverThreadIds")
    if not isinstance(raw_receivers, (list, tuple)) and isinstance(raw_input, Mapping):
        raw_receivers = raw_input.get("receiverThreadIds")
    if not isinstance(raw_receivers, (list, tuple)):
        raw_receivers = ()
    receivers = tuple(
        value
        for item in (raw_receivers or ())
        if (value := str(item or "").strip())
    )

    model = ""
    raw_states: Any = None
    if isinstance(raw_input, Mapping):
        model = str(raw_input.get("model") or "").strip()
        raw_states = raw_input.get("agentsStates")

    states: list[dict] = []
    if isinstance(raw_states, Mapping):
        normalized_states = {str(key): value for key, value in raw_states.items()}
        ordered_ids = list(receivers)
        ordered_ids.extend(
            source_id
            for source_id in normalized_states
            if source_id not in ordered_ids
        )
        for source_id in ordered_ids:
            raw_state = normalized_states.get(source_id)
            if not isinstance(raw_state, Mapping):
                continue
            states.append(
                {
                    "source_id": source_id,
                    "status": str(raw_state.get("status") or "").strip(),
                    "message": str(raw_state.get("message") or "").strip(),
                }
            )
    return tool, receivers, model, tuple(states)


def _parse_tool_call(update: ToolCallStart | ToolCallProgress) -> ToolCallInfo:
    """Extract ToolCallInfo from a ToolCallStart or ToolCallProgress."""
    locations: list[str] = []
    if update.locations:
        locations = [loc.path for loc in update.locations]

    title = update.title or ""
    raw_input = getattr(update, "raw_input", None)
    raw_output = getattr(update, "raw_output", None)
    status = (update.status or "in_progress").strip() or "in_progress"
    subagent_source_id, subagent_path, subagent_activity = (
        _parse_subagent_metadata(update)
    )
    (
        collaboration_tool,
        collaboration_receivers,
        collaboration_model,
        subagent_states,
    ) = _parse_collaboration_metadata(update, raw_input)

    def _json_dump(obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            return str(obj)

    def _truncate(s: str, max_chars: int) -> str:
        s = s or ""
        if len(s) <= max_chars:
            return s
        return s[:max_chars] + "\n... (truncated)"

    # Prefer tool kind for rendering decisions; fall back to title heuristics.
    kind = (update.kind or "other").strip() or "other"
    title_lower = title.lower()
    is_execute = (kind == "execute") or ("bash" in title_lower)
    is_agent_task = (
        kind == "agent"
        or title_lower == "agent"
        or title_lower == "task"
        or bool(subagent_source_id)
        or bool(collaboration_tool)
        or (isinstance(raw_input, dict) and any(k in raw_input for k in ("subagent_type", "description", "prompt")))
    )

    # Decide which side to render into ToolCallInfo.content.
    # NOTE: tests rely on conservative rendering: only TodoWrite and execute-like
    # tools populate content on the input side; read/list/etc keep it empty.
    use_output = status in ("completed", "failed")

    content = ""

    # TodoWrite: always extract checklist from raw_input, regardless of status.
    if raw_input and _is_todo_tool(title, raw_input):
        content = _format_todo_content(raw_input)
        use_output = False

    if not use_output and not content:
        # ---- input side ----
        if is_execute:
            if isinstance(raw_input, dict):
                content = str(
                    raw_input.get("command")
                    or raw_input.get("cmd")
                    or raw_input.get("shell_command")
                    or ""
                )
            elif isinstance(raw_input, str):
                content = raw_input
            elif raw_input is not None:
                content = _json_dump(raw_input)

        elif is_agent_task:
            if isinstance(raw_input, dict):
                description = str(raw_input.get("description") or "").strip()
                prompt = str(raw_input.get("prompt") or "").strip()
                subagent_type = str(raw_input.get("subagent_type") or "").strip()
                parts = []
                if description:
                    parts.append(description)
                if subagent_type:
                    parts.append(f"子代理：{subagent_type}")
                if prompt:
                    prompt_first_line = prompt.splitlines()[0].strip()
                    if prompt_first_line and prompt_first_line != description:
                        parts.append(prompt_first_line)
                content = "\n".join(parts)
            elif isinstance(raw_input, str):
                content = raw_input

        # For other non-execute tools, keep content empty to reduce noise.

        content = _truncate((content or "").strip("\n"), 4000)
    else:
        # ---- output side ----
        if is_execute:
            if isinstance(raw_output, dict):
                # Best-effort normalize common shapes
                out = raw_output.get("output")
                if isinstance(out, str) and out.strip():
                    content = out
                else:
                    stdout = raw_output.get("stdout") or ""
                    stderr = raw_output.get("stderr") or ""
                    parts = []
                    if isinstance(stdout, str) and stdout:
                        parts.append(stdout)
                    if isinstance(stderr, str) and stderr:
                        parts.append(stderr)
                    content = "\n".join(parts).strip("\n")
                    if not content:
                        content = _json_dump(raw_output)
            elif isinstance(raw_output, str):
                content = raw_output
            elif raw_output is not None:
                content = _json_dump(raw_output)
        else:
            if isinstance(raw_output, str):
                content = raw_output
            elif raw_output is not None:
                content = _json_dump(raw_output)

        content = _truncate((content or "").strip("\n"), 12000)

    # Collaboration output is normalized separately into per-child states.
    # Keep the input prompt as the stable user-facing child label instead of
    # rendering provider result JSON. Failed calls retain raw output so the
    # ordinary error path can expose a sanitized failure reason.
    if (
        collaboration_tool
        and status != "failed"
        and isinstance(raw_input, Mapping)
    ):
        prompt = str(raw_input.get("prompt") or "").strip()
        if prompt:
            content = _truncate(prompt.splitlines()[0].strip(), 4000)

    return ToolCallInfo(
        id=update.tool_call_id,
        title=title,
        kind=kind,
        status=status,
        content=content,
        locations=locations,
        subagent_source_id=subagent_source_id or None,
        subagent_path=subagent_path or None,
        subagent_activity=subagent_activity or None,
        collaboration_tool=collaboration_tool or None,
        collaboration_receivers=collaboration_receivers,
        collaboration_model=collaboration_model or None,
        subagent_states=subagent_states,
    )


def _parse_plan(update: AgentPlanUpdate) -> PlanInfo:
    """Extract PlanInfo from an AgentPlanUpdate."""
    entries: list[PlanEntryInfo] = []
    for entry in update.entries:
        try:
            raw_content = getattr(entry, "content", None)
            content = ("" if raw_content is None else str(raw_content)).strip()
        except Exception:
            content = ""
        # Some agents may emit placeholder entries with empty content; skip them to
        # avoid rendering "✅" lines without text.
        if not content:
            continue

        entries.append(
            PlanEntryInfo(
                content=content,
                priority=getattr(entry, "priority", None) or "medium",
                status=getattr(entry, "status", None) or "pending",
            )
        )
    return PlanInfo(entries=entries)


def _extract_update_source_id(update: Any) -> str | None:
    """Best-effort source identifier for streaming chunks.

    ACP does not expose a first-class source field for text chunks today, but
    some providers attach agent/task identity through dynamic attributes or
    ``_meta``. Keep that identity so concurrent streams do not share one card
    text block.
    """
    subagent_source_id, _, _ = _parse_subagent_metadata(update)
    if subagent_source_id:
        return subagent_source_id

    # 注意：不要包含"id"，因为这通常是每个块的唯一标识符，而不是源标识符
    # 我们需要保留源标识符，以便将来自同一源的连续块合并到同一个文本块中
    candidates = ("source_id", "source", "agent_id", "task_id", "tool_call_id")

    def _from_obj(obj: Any) -> str | None:
        for key in candidates:
            value = getattr(obj, key, None)
            if value:
                return str(value)
        meta = getattr(obj, "_meta", None) or getattr(obj, "field_meta", None)
        if isinstance(meta, dict):
            # 明确排除"id"字段，确保不会将块ID误用作源ID
            if "id" in meta:
                logger.debug(f"Ignoring generic chunk id as source: {meta['id']}")
            for key in candidates:
                value = meta.get(key)
                if value:
                    return str(value)
        return None

    return _from_obj(update) or _from_obj(getattr(update, "content", None))


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _parse_codex_goal(value: object) -> ACPGoalInfo | None:
    if not isinstance(value, Mapping):
        return None
    objective = value.get("objective")
    status = value.get("status")
    if not isinstance(objective, str) or not objective.strip():
        return None
    if not isinstance(status, str) or not status.strip():
        return None

    token_budget = None
    if "tokenBudget" in value and value.get("tokenBudget") is not None:
        token_number = _finite_number(value.get("tokenBudget"))
        if token_number is None or not token_number.is_integer():
            return None
        token_budget = int(token_number)

    time_used_seconds = None
    if "timeUsedSeconds" in value and value.get("timeUsedSeconds") is not None:
        time_number = _finite_number(value.get("timeUsedSeconds"))
        if time_number is None:
            return None
        time_used_seconds = time_number

    control_method = value.get("controlMethod", "")
    created_at: str | float | None = value.get("createdAt")
    if not isinstance(control_method, str):
        return None
    if created_at is not None and not isinstance(created_at, str):
        created_number = _finite_number(created_at)
        if created_number is None or created_number < 0:
            return None
        created_at = created_number
    return ACPGoalInfo(
        objective=objective.strip(),
        status=status.strip(),
        control_method=control_method,
        token_budget=token_budget,
        time_used_seconds=time_used_seconds,
        created_at=created_at,
    )


def _parse_session_info_update(update: SessionInfoUpdate) -> ACPSessionInfo:
    meta = update.field_meta
    if not isinstance(meta, Mapping):
        return ACPSessionInfo()
    codex = meta.get("codex")
    if not isinstance(codex, Mapping):
        return ACPSessionInfo()

    goal_known = False
    goal = None
    if "goal" in codex:
        raw_goal = codex.get("goal")
        if raw_goal is None:
            goal_known = True
        else:
            goal = _parse_codex_goal(raw_goal)
            goal_known = goal is not None

    thread_status_observed = "threadStatus" in codex
    thread_status_known = False
    thread_status = ""
    raw_thread_status = codex.get("threadStatus")
    if isinstance(raw_thread_status, Mapping):
        raw_type = raw_thread_status.get("type")
        if isinstance(raw_type, str) and raw_type.strip() in {"active", "idle"}:
            thread_status_known = True
            thread_status = raw_type.strip()

    return ACPSessionInfo(
        goal_known=goal_known,
        goal=goal,
        thread_status_observed=thread_status_observed,
        thread_status_known=thread_status_known,
        thread_status=thread_status,
    )


class GhostAPClient(Client):
    """ACP Client implementation — processes agent session updates."""

    def __init__(
        self,
        on_event: Callable[[ACPEvent], None],
        on_session_info: Callable[[str, ACPSessionInfo], None] | None = None,
        auto_approve: bool = True,
        root_dir: str = ".",
        sandbox: Optional[SandboxExecutor] = None,
        history_store: Optional[ACPHistoryStore] = None,
    ):
        self._on_event = on_event
        self._on_session_info = on_session_info
        self._auto_approve = auto_approve
        self._root_dir = os.path.abspath(os.path.expanduser(root_dir or "."))
        self._sandbox = sandbox or SandboxExecutor()
        self._terminals: dict[str, _TerminalRecord] = {}
        self._terminals_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._history = history_store or ACPHistoryStore()
        self._tool_filter: Optional[Callable[[str, dict | None], bool]] = None
        self._image_snapshot_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._active_image_snapshot: LocalImageArtifactSnapshot | None = None

    def set_tool_filter(self, filter_fn: Optional[Callable[[str, dict | None], bool]]) -> None:
        """Install or clear a per-session tool filter."""
        self._tool_filter = filter_fn

    def get_tool_filter(self) -> Optional[Callable[[str, dict | None], bool]]:
        """Return the current per-session tool filter, if any."""
        return self._tool_filter

    def snapshot_local_images(self) -> LocalImageArtifactSnapshot:
        """Capture bounded image file metadata before a prompt starts."""
        snapshot = snapshot_local_image_artifacts(self._root_dir)
        with self._image_snapshot_lock:
            self._active_image_snapshot = snapshot
        return snapshot

    def release_local_image_snapshot(
        self,
        before: LocalImageArtifactSnapshot | Mapping[str, _ImageArtifactSignature],
    ) -> None:
        """Release a prompt snapshot and clear its local attribution context."""
        with self._image_snapshot_lock:
            if self._active_image_snapshot is before:
                self._active_image_snapshot = None
        release_local_image_artifact_snapshot(before)

    def _current_image_snapshot(self) -> LocalImageArtifactSnapshot | None:
        with self._image_snapshot_lock:
            return self._active_image_snapshot

    def release_active_local_image_snapshot(self) -> None:
        """Release any prompt attribution lease still owned during shutdown."""
        with self._image_snapshot_lock:
            snapshot = self._active_image_snapshot
            self._active_image_snapshot = None
        if snapshot is not None:
            release_local_image_artifact_snapshot(snapshot)

    def _is_tool_allowed(self, tool_name: str, args: dict | None = None) -> bool:
        filter_fn = self._tool_filter
        if not callable(filter_fn):
            return True
        try:
            return bool(filter_fn(tool_name, args or {}))
        except Exception as exc:
            logger.warning("[ACP] tool filter failed closed: tool=%s err=%s", tool_name, get_error_detail(exc))
            return False

    def _record(self, session_id: str, kind: str, data: dict) -> None:
        try:
            payload = {"kind": kind, "data": data or {}}
            self._history.append(session_id, payload)
        except Exception:
            logger.debug("[ACP] history record failed", exc_info=True)

    # ------------------------------------------------------------------
    # Core callback: session_update
    # ------------------------------------------------------------------
    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        """Receive agent's streaming updates — the core event handler."""
        try:
            if isinstance(update, SessionInfoUpdate):
                info = _parse_session_info_update(update)
                if (
                    self._on_session_info is not None
                    and (
                        info.goal_known
                        or info.thread_status_known
                    )
                ):
                    self._on_session_info(session_id, info)
            elif isinstance(update, AgentMessageChunk):
                self._handle_message_chunk(update)
            elif isinstance(update, AgentThoughtChunk):
                self._handle_thought_chunk(update)
            elif isinstance(update, ToolCallStart):
                self._handle_tool_call_start(update)
            elif isinstance(update, ToolCallProgress):
                self._handle_tool_call_progress(update)
            elif isinstance(update, AgentPlanUpdate):
                self._handle_plan_update(update)
            # Other update types (UsageUpdate, etc.) are silently ignored
        except Exception as e:
            logger.debug("Error processing ACP session_update: %s", get_error_detail(e))

    def _handle_message_chunk(self, update: AgentMessageChunk) -> None:
        content = update.content
        if isinstance(content, TextContentBlock):
            self._on_event(
                ACPEvent(
                    event_type=ACPEventType.TEXT_CHUNK,
                    text=content.text,
                    source_id=_extract_update_source_id(update),
                )
            )
        elif image := _parse_acp_image_content(
            content,
            root_dir=self._root_dir,
            image_snapshot=self._current_image_snapshot(),
        ):
            self._on_event(
                ACPEvent(
                    event_type=ACPEventType.IMAGE_CHUNK,
                    image=image,
                    source_id=_extract_update_source_id(update),
                )
            )
        else:
            logger.debug(f"Unhandled content type in message chunk: {type(content)}")

    def _handle_thought_chunk(self, update: AgentThoughtChunk) -> None:
        content = update.content
        if isinstance(content, TextContentBlock):
            self._on_event(
                ACPEvent(
                    event_type=ACPEventType.THOUGHT_CHUNK,
                    text=content.text,
                    source_id=_extract_update_source_id(update),
                )
            )
        elif image := _parse_acp_image_content(
            content,
            root_dir=self._root_dir,
            image_snapshot=self._current_image_snapshot(),
        ):
            self._on_event(
                ACPEvent(
                    event_type=ACPEventType.IMAGE_CHUNK,
                    image=image,
                    source_id=_extract_update_source_id(update),
                )
            )
        else:
            logger.debug(f"Unhandled content type in thought chunk: {type(content)}")

    def _handle_tool_call_start(self, update: ToolCallStart) -> None:
        tool_info = _parse_tool_call(update)
        self._on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_START,
                tool_call=tool_info,
                source_id=tool_info.subagent_source_id,
            )
        )
        self._emit_tool_images(update)

    def _handle_tool_call_progress(self, update: ToolCallProgress) -> None:
        tool_info = _parse_tool_call(update)
        status = tool_info.status
        if status in ("completed", "failed"):
            event_type = ACPEventType.TOOL_CALL_DONE
            # Publish artifacts while the tool/task card is still open. The
            # terminal event closes that target and fences later mutations.
            self._emit_tool_images(update)
        else:
            event_type = ACPEventType.TOOL_CALL_UPDATE
        self._on_event(
            ACPEvent(
                event_type=event_type,
                tool_call=tool_info,
                source_id=tool_info.subagent_source_id,
            )
        )

    def _emit_tool_images(self, update: ToolCallStart | ToolCallProgress) -> None:
        source_id = _extract_update_source_id(update) or update.tool_call_id
        for image in _tool_call_images(
            update,
            root_dir=self._root_dir,
            image_snapshot=self._current_image_snapshot(),
        ):
            self._on_event(
                ACPEvent(
                    event_type=ACPEventType.IMAGE_CHUNK,
                    image=image,
                    source_id=source_id,
                )
            )

    def _handle_plan_update(self, update: AgentPlanUpdate) -> None:
        plan = _parse_plan(update)
        self._on_event(
            ACPEvent(
                event_type=ACPEventType.PLAN_UPDATE,
                plan=plan,
            )
        )

    # ------------------------------------------------------------------
    # Permission handling
    # ------------------------------------------------------------------
    async def request_permission(
        self, session_id: str, tool_call, options, **kwargs: Any
    ) -> RequestPermissionResponse:
        """Handle permission requests from agent."""
        if not self._auto_approve:
            self._record(session_id, "permission", {"outcome": "cancelled", "reason": "auto_approve_disabled"})
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

        if not options:
            # Defensive: some agents might ask with empty options.
            self._record(session_id, "permission", {"outcome": "cancelled", "reason": "empty_options"})
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

        # Permission filters apply to every ACP kind before auto-approval.
        try:
            permission_tool, tool_args, command = _permission_tool_request(
                tool_call,
                root_dir=self._root_dir,
            )
            if not self._is_tool_allowed(permission_tool, tool_args):
                denied_data = {
                    "outcome": "cancelled",
                    "reason": "tool_filter_denied",
                    "tool": permission_tool,
                }
                if command is not None:
                    denied_data["command"] = command
                self._record(session_id, "permission", denied_data)
                return RequestPermissionResponse(
                    outcome=DeniedOutcome(outcome="cancelled")
                )

            # Keep the existing hard and sandbox checks for executable commands.
            if command is not None:
                hard_ok, hard_reason = _ACP_PERMISSION_DANGEROUS_CHECK.check(command, None)
                if not hard_ok:
                    logger.info("[ACP] Reject dangerous auto-approved command: %s (%s)", command, hard_reason)
                    self._record(
                        session_id,
                        "permission",
                        {
                            "outcome": "cancelled",
                            "reason": "dangerous_execute",
                            "command": command,
                            "detail": hard_reason,
                        },
                    )
                    return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
                ok, reason = self._sandbox.is_command_safe(command)
                if not ok:
                    logger.info("[ACP] Reject unsafe command: %s (%s)", command, reason)
                    self._record(
                        session_id,
                        "permission",
                        {
                            "outcome": "cancelled",
                            "reason": "unsafe_execute",
                            "command": command,
                            "detail": reason,
                        },
                    )
                    return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        except Exception as e:
            logger.warning("[ACP] Permission safety check failed closed: %s", type(e).__name__)
            self._record(
                session_id,
                "permission",
                {"outcome": "cancelled", "reason": "permission_safety_check_failed"},
            )
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

        # Find an "allow_once" option, or use the first option.
        allow_option_id = ""
        for opt in options:
            if getattr(opt, "kind", None) == "allow_once":
                allow_option_id = opt.option_id
                break
        if not allow_option_id and options:
            allow_option_id = options[0].option_id

        return RequestPermissionResponse(outcome=AllowedOutcome(option_id=allow_option_id, outcome="selected"))

    # ------------------------------------------------------------------
    # File operations (delegate to agent's own filesystem)
    # ------------------------------------------------------------------
    async def read_text_file(
        self,
        session_id: str,
        path: str,
        line: Optional[int] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> ReadTextFileResponse:
        tool_args = {"path": path, "line": line, "limit": limit, **kwargs}
        if not self._is_tool_allowed("file_read", tool_args):
            self._record(session_id, "read_file", {"path": path, "blocked": True, "reason": "tool_filter_denied"})
            return ReadTextFileResponse(content="", field_meta={"blocked": True, "reason": "tool_filter_denied"})
        try:
            resolved = _safe_resolve_path(self._root_dir, path)
            content = resolved.read_text(encoding="utf-8")
            if line is not None or limit is not None:
                start = max(0, int(line or 1) - 1)
                end = start + max(0, int(limit)) if limit is not None else None
                content = "".join(content.splitlines(keepends=True)[start:end])
            max_chars = _get_max_file_chars()
            if len(content) > max_chars:
                content = content[:max_chars]
                self._record(
                    session_id, "read_file", {"path": str(resolved), "truncated": True, "max_chars": max_chars}
                )
                return ReadTextFileResponse(
                    content=content,
                    field_meta={"truncated": True, "path": str(resolved), "max_chars": max_chars},
                )
            self._record(session_id, "read_file", {"path": str(resolved), "truncated": False, "chars": len(content)})
            return ReadTextFileResponse(content=content)
        except Exception as e:
            logger.info("[ACP] read_text_file failed: path=%s err=%s", path, get_error_detail(e))
            self._record(session_id, "read_file", {"path": path, "error": get_error_detail(e)})
            return ReadTextFileResponse(content="", field_meta={"error": get_error_detail(e), "path": path})

    async def write_text_file(
        self, session_id: str, path: str, content: str, **kwargs: Any
    ) -> Optional[WriteTextFileResponse]:
        if not self._is_tool_allowed("file_write", {"path": path, **kwargs}):
            self._record(session_id, "write_file", {"path": path, "blocked": True, "reason": "tool_filter_denied"})
            return WriteTextFileResponse(field_meta={"blocked": True, "reason": "tool_filter_denied", "path": path})
        try:
            resolved = _safe_resolve_path(self._root_dir, path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content or "", encoding="utf-8")
            self._record(session_id, "write_file", {"path": str(resolved), "chars": len(content or "")})
            return WriteTextFileResponse()
        except Exception as e:
            logger.info("[ACP] write_text_file failed: path=%s err=%s", path, get_error_detail(e))
            self._record(session_id, "write_file", {"path": path, "error": get_error_detail(e)})
            return WriteTextFileResponse(field_meta={"error": get_error_detail(e), "path": path})

    # ------------------------------------------------------------------
    # Terminal operations (stub — agent manages its own terminals)
    # ------------------------------------------------------------------
    async def create_terminal(
        self,
        session_id: str,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[list[Any]] = None,
        cwd: Optional[str] = None,
        output_byte_limit: Optional[int] = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        # Lazy cleanup of expired terminals to prevent unbounded growth
        self._cleanup_expired_terminals()

        shell_command = shlex.join([command, *(args or [])]) if args else command
        try:
            execution_cwd = str(_safe_resolve_path(self._root_dir, cwd)) if cwd else self._root_dir
        except Exception:
            return self._create_failed_terminal(
                session_id=session_id,
                command=shell_command,
                message="❌ 工作目录超出项目范围",
                reason="cwd_outside_project_root",
            )
        try:
            env_overrides = _validated_env_overrides(env)
        except ValueError as e:
            return self._create_failed_terminal(
                session_id=session_id,
                command=shell_command,
                message=f"❌ {get_error_detail(e)}",
                reason="invalid_environment_variable",
            )

        tool_args = {
            "command": shell_command,
            "args": list(args or []),
            "cwd": execution_cwd,
            "output_byte_limit": output_byte_limit,
            **kwargs,
        }
        if not self._is_tool_allowed("shell", tool_args):
            return self._create_failed_terminal(
                session_id=session_id,
                command=shell_command,
                message="❌ 工具权限检查未通过: tool_filter_denied",
                reason="tool_filter_denied",
            )

        ok, reason = self._sandbox.is_command_safe(shell_command)
        if not ok:
            return self._create_failed_terminal(
                session_id=session_id,
                command=shell_command,
                message=f"❌ 安全检查未通过: {reason}",
                reason=reason or "sandbox_rejected",
            )

        result = self._sandbox.execute(
            shell_command,
            cwd=execution_cwd,
            interactive=False,
            env_overrides=env_overrides or None,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output_parts: list[str] = []
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(stderr)
        output = "\n".join(output_parts).strip("\n")
        output, byte_truncated = _truncate_utf8_tail(output, output_byte_limit)
        truncated = byte_truncated or ("输出被截断" in output) or ("错误输出被截断" in output)

        # Persist a capped copy of output for recovery/debug UI.
        cap = 8000
        output_cap = output if len(output) <= cap else (output[:cap] + "\n... (truncated)")
        self._record(
            session_id,
            "execute",
            {
                "command": shell_command,
                "cwd": execution_cwd,
                "exit_code": result.return_code,
                "truncated": truncated,
                "output": output_cap,
            },
        )

        term_id = f"term_{uuid.uuid4().hex[:8]}"
        with self._terminals_lock:
            self._terminals[term_id] = _TerminalRecord(
                output=output,
                exit_code=result.return_code,
                truncated=truncated,
                created_at=time.time(),
            )
        return CreateTerminalResponse(terminal_id=term_id)

    def _create_failed_terminal(
        self,
        *,
        session_id: str,
        command: str,
        message: str,
        reason: str,
    ) -> CreateTerminalResponse:
        """Create a virtual terminal for a denied ACP operation without executing it."""
        term_id = f"term_{uuid.uuid4().hex[:8]}"
        with self._terminals_lock:
            self._terminals[term_id] = _TerminalRecord(
                output=message,
                exit_code=1,
                truncated=False,
                created_at=time.time(),
            )
        self._record(
            session_id,
            "execute",
            {"command": command, "blocked": True, "reason": reason},
        )
        return CreateTerminalResponse(
            terminal_id=term_id,
            field_meta={"blocked": True, "reason": reason},
        )

    def _cleanup_expired_terminals(self) -> None:
        """Remove terminal records older than _TERMINAL_TTL."""
        with self._terminals_lock:
            if not self._terminals:
                return
            now = time.time()
            expired = [tid for tid, rec in self._terminals.items() if now - rec.created_at > _TERMINAL_TTL]
            for tid in expired:
                del self._terminals[tid]

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> TerminalOutputResponse:
        with self._terminals_lock:
            rec = self._terminals.get(terminal_id)
        if not rec:
            return TerminalOutputResponse(output="", truncated=False, field_meta={"error": "unknown_terminal"})
        chunk = rec.output[rec.cursor :]
        rec.cursor = len(rec.output)
        return TerminalOutputResponse(
            output=chunk,
            truncated=rec.truncated,
            exit_status=TerminalExitStatus(exit_code=rec.exit_code),
        )

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        with self._terminals_lock:
            rec = self._terminals.get(terminal_id)
        if not rec:
            return WaitForTerminalExitResponse(exit_code=None, signal=None, field_meta={"error": "unknown_terminal"})
        return WaitForTerminalExitResponse(exit_code=rec.exit_code, signal=None)

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> Optional[KillTerminalCommandResponse]:
        with self._terminals_lock:
            self._terminals.pop(terminal_id, None)
        return KillTerminalCommandResponse()

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> Optional[ReleaseTerminalResponse]:
        with self._terminals_lock:
            self._terminals.pop(terminal_id, None)
        return ReleaseTerminalResponse()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_connect(self, conn: Agent) -> None:
        # NOTE: ACP SDK calls `on_connect()` synchronously.
        # Keep this hook sync to avoid "coroutine was never awaited" warnings.
        return None
