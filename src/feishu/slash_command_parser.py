"""Slash command parsing and normalization.

This module is the single source of truth for how GhostAP interprets
user-entered slash commands (e.g. ``/deep <goal>``).

Design goals:
- Case-insensitive command matching (only the command token is normalized).
- Whitespace tolerant: supports spaces, tabs, newlines as separators.
- No "startswith + slicing" in handlers: handlers should consume parsed args.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .product_catalog import resolve_command


@dataclass(frozen=True, slots=True)
class CommandMatch:
    """Parsed slash command (DTO).

    本类型用于在 Feishu 处理链路中跨模块传递 slash 解析结果。
    约束：handler/gate/manager **必须**消费该结构化结果，而不是再对原始
    ``text`` 做 ``startswith()+切片`` 或用 ``raw_text`` 推断参数。

    跨模块稳定字段（最小契约）：
    - raw_text: 原始输入文本（仅用于日志/回显；不要用它做业务解析）
    - normalized_text: strip 后的文本（保留内部空白，用于一致性回显）
    - raw_command: 用户输入的命令 token（lower 后，例如 "/deep"）
    - command: 规范化后的 canonical 命令（例如 "/deep"）
    - args: 命令 token 之后的参数串（strip 后，保留内部空白）
    - has_args: args 是否非空

    注意：下游逻辑应以 ``command``/``args``/``has_args`` 为单一事实源。
    """

    raw_text: str
    normalized_text: str
    raw_command: str
    command: str
    args: str
    has_args: bool


class SlashCommandParser:
    """Parse and match slash commands."""

    @staticmethod
    def parse(text: str) -> Optional[CommandMatch]:
        """Parse *text* as a slash command.

        Returns None if the input is empty or does not start with '/'.
        """
        raw = text or ""
        normalized = raw.strip()
        if not normalized or not normalized.startswith("/"):
            return None

        # Split on any whitespace: supports space/tab/newline.
        parts = normalized.split(None, 1)
        token = parts[0]
        args = parts[1].strip() if len(parts) > 1 else ""

        raw_cmd = token.lower()
        resolved = resolve_command(raw_cmd, args)
        canonical = (
            resolved.rewritten_command
            if resolved and resolved.rewritten_command
            else resolved.action.command
            if resolved
            else raw_cmd
        )

        return CommandMatch(
            raw_text=raw,
            normalized_text=normalized,
            raw_command=raw_cmd,
            command=canonical,
            args=args,
            has_args=bool(args),
        )

    @staticmethod
    def match(text: str, allowed_commands: Iterable[str]) -> Optional[CommandMatch]:
        """Parse *text* and return a match when command is in *allowed_commands*.

        *allowed_commands* should contain canonical commands (lowercase).
        """
        m = SlashCommandParser.parse(text)
        if not m:
            return None
        allowed = set((c or "").strip().lower() for c in allowed_commands)
        if m.command in allowed:
            return m
        return None
