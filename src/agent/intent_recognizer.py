import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from ..config import get_settings

logger = logging.getLogger(__name__)


class IntentType(Enum):
    ENTER_COCO = "enter_coco"
    EXIT_COCO = "exit_coco"
    ENTER_CLAUDE = "enter_claude"
    EXIT_CLAUDE = "exit_claude"
    ENTER_AIDEN = "enter_aiden"
    EXIT_AIDEN = "exit_aiden"
    ENTER_CODEX = "enter_codex"
    EXIT_CODEX = "exit_codex"
    ENTER_GEMINI = "enter_gemini"
    EXIT_GEMINI = "exit_gemini"
    ENTER_TRAEX = "enter_traex"
    EXIT_TRAEX = "exit_traex"
    EXIT_MODE = "exit_mode"
    CHANGE_DIR = "change_dir"
    SHELL_COMMAND = "shell"
    COCO_MESSAGE = "coco_message"
    CLAUDE_MESSAGE = "claude_message"
    AIDEN_MESSAGE = "aiden_message"
    CODEX_MESSAGE = "codex_message"
    GEMINI_MESSAGE = "gemini_message"
    TRAEX_MESSAGE = "traex_message"
    CREATE_PROJECT = "create_project"
    SWITCH_PROJECT = "switch_project"
    LIST_PROJECTS = "list_projects"
    CLOSE_PROJECT = "close_project"
    PROJECT_STATUS = "project_status"
    NEW_CHAT_PROJECT = "new_chat_project"
    ENTER_DEEP = "enter_deep"
    DEEP_STATUS = "deep_status"
    STOP_DEEP = "stop_deep"
    DEEP_UPDATE = "deep_update"
    ENTER_SPEC = "enter_spec"
    SPEC_STATUS = "spec_status"
    STOP_SPEC = "stop_spec"
    SPEC_GUIDE = "spec_guide"
    SHOW_HELP = "show_help"
    SHOW_TOOLS = "show_tools"
    TOOLS_STATUS = "tools_status"
    UNKNOWN = "unknown"


@dataclass
class TaskStep:
    intent: IntentType
    description: str
    data: dict = field(default_factory=dict)


@dataclass
class IntentResult:
    tasks: list[TaskStep] = field(default_factory=list)
    confidence: float = 0.0
    original_text: str = ""
    reasoning: str = ""

    @property
    def is_multi_task(self) -> bool:
        return len(self.tasks) > 1

    @property
    def primary_intent(self) -> IntentType:
        return self.tasks[0].intent if self.tasks else IntentType.UNKNOWN

    @property
    def primary_data(self) -> dict:
        return self.tasks[0].data if self.tasks else {}

    @classmethod
    def single(
        cls,
        intent: IntentType,
        confidence: float = 0.0,
        data: dict = None,
        original_text: str = "",
        reasoning: str = "",
        description: str = "",
    ) -> "IntentResult":
        return cls(
            tasks=[TaskStep(intent=intent, description=description, data=data or {})],
            confidence=confidence,
            original_text=original_text,
            reasoning=reasoning,
        )


QuickMatchRule = Callable[["IntentRecognizer", str, str], Optional[IntentResult]]


class IntentMatcher:
    """Ordered registry for quick intent match rules."""

    def __init__(self) -> None:
        self._rules: list[tuple[str, QuickMatchRule]] = []

    def register(self, name: str, rule: QuickMatchRule) -> "IntentMatcher":
        self._rules.append((name, rule))
        return self

    @property
    def rule_names(self) -> tuple[str, ...]:
        return tuple(name for name, _rule in self._rules)

    def match(self, recognizer: "IntentRecognizer", text: str, current_mode: str = "smart") -> Optional[IntentResult]:
        for _name, rule in self._rules:
            result = rule(recognizer, text, current_mode)
            if result is not None:
                return result
        return None


class IntentRecognizer:
    INTENT_MAP = {
        "enter_coco": IntentType.ENTER_COCO,
        "exit_coco": IntentType.EXIT_COCO,
        "enter_claude": IntentType.ENTER_CLAUDE,
        "exit_claude": IntentType.EXIT_CLAUDE,
        "enter_aiden": IntentType.ENTER_AIDEN,
        "exit_aiden": IntentType.EXIT_AIDEN,
        "enter_codex": IntentType.ENTER_CODEX,
        "exit_codex": IntentType.EXIT_CODEX,
        "enter_gemini": IntentType.ENTER_GEMINI,
        "exit_gemini": IntentType.EXIT_GEMINI,
        "enter_traex": IntentType.ENTER_TRAEX,
        "exit_traex": IntentType.EXIT_TRAEX,
        "exit_mode": IntentType.EXIT_MODE,
        "coco_message": IntentType.COCO_MESSAGE,
        "claude_message": IntentType.CLAUDE_MESSAGE,
        "aiden_message": IntentType.AIDEN_MESSAGE,
        "codex_message": IntentType.CODEX_MESSAGE,
        "gemini_message": IntentType.GEMINI_MESSAGE,
        "traex_message": IntentType.TRAEX_MESSAGE,
        "change_dir": IntentType.CHANGE_DIR,
        "shell": IntentType.SHELL_COMMAND,
        "create_project": IntentType.CREATE_PROJECT,
        "switch_project": IntentType.SWITCH_PROJECT,
        "list_projects": IntentType.LIST_PROJECTS,
        "close_project": IntentType.CLOSE_PROJECT,
        "project_status": IntentType.PROJECT_STATUS,
        "new_chat_project": IntentType.NEW_CHAT_PROJECT,
        "enter_deep": IntentType.ENTER_DEEP,
        "deep_status": IntentType.DEEP_STATUS,
        "stop_deep": IntentType.STOP_DEEP,
        "deep_update": IntentType.DEEP_UPDATE,
        "enter_spec": IntentType.ENTER_SPEC,
        "spec_status": IntentType.SPEC_STATUS,
        "stop_spec": IntentType.STOP_SPEC,
        "spec_guide": IntentType.SPEC_GUIDE,
        "show_help": IntentType.SHOW_HELP,
        "show_tools": IntentType.SHOW_TOOLS,
        "tools_status": IntentType.TOOLS_STATUS,
        "unknown": IntentType.UNKNOWN,
    }

    EXACT_COMMANDS = {
        "/coco": (IntentType.ENTER_COCO, "进入 Coco 编程模式"),
        "/enter_coco": (IntentType.ENTER_COCO, "进入 Coco 编程模式"),
        "/end_coco": (IntentType.EXIT_COCO, "退出 Coco 编程模式"),
        "/exit_coco": (IntentType.EXIT_COCO, "退出 Coco 编程模式"),
        "/claude": (IntentType.ENTER_CLAUDE, "进入 Claude 编程模式"),
        "/enter_claude": (IntentType.ENTER_CLAUDE, "进入 Claude 编程模式"),
        "/end_claude": (IntentType.EXIT_CLAUDE, "退出 Claude 编程模式"),
        "/exit_claude": (IntentType.EXIT_CLAUDE, "退出 Claude 编程模式"),
        "/aiden": (IntentType.ENTER_AIDEN, "进入 Aiden 编程模式"),
        "/enter_aiden": (IntentType.ENTER_AIDEN, "进入 Aiden 编程模式"),
        "/end_aiden": (IntentType.EXIT_AIDEN, "退出 Aiden 编程模式"),
        "/exit_aiden": (IntentType.EXIT_AIDEN, "退出 Aiden 编程模式"),
        "/codex": (IntentType.ENTER_CODEX, "进入 Codex 编程模式"),
        "/enter_codex": (IntentType.ENTER_CODEX, "进入 Codex 编程模式"),
        "/end_codex": (IntentType.EXIT_CODEX, "退出 Codex 编程模式"),
        "/exit_codex": (IntentType.EXIT_CODEX, "退出 Codex 编程模式"),
        "/gemini": (IntentType.ENTER_GEMINI, "进入 Gemini 编程模式"),
        "/enter_gemini": (IntentType.ENTER_GEMINI, "进入 Gemini 编程模式"),
        "/end_gemini": (IntentType.EXIT_GEMINI, "退出 Gemini 编程模式"),
        "/exit_gemini": (IntentType.EXIT_GEMINI, "退出 Gemini 编程模式"),
        "/traex": (IntentType.ENTER_TRAEX, "进入 Traex 编程模式"),
        "/enter_traex": (IntentType.ENTER_TRAEX, "进入 Traex 编程模式"),
        "/end_traex": (IntentType.EXIT_TRAEX, "退出 Traex 编程模式"),
        "/exit_traex": (IntentType.EXIT_TRAEX, "退出 Traex 编程模式"),
        "/exit": (IntentType.EXIT_MODE, "退出当前模式"),
        "/quit": (IntentType.EXIT_MODE, "退出当前模式"),
        "/projects": (IntentType.LIST_PROJECTS, "查看项目列表"),
        "/switch": (IntentType.SWITCH_PROJECT, "切换项目（打开项目看板）"),
        "/project": (IntentType.PROJECT_STATUS, "查看当前项目"),
        "/status": (IntentType.PROJECT_STATUS, "查看项目状态"),
        "/deep": (IntentType.ENTER_DEEP, "进入 Deep 模式"),
        "/deep_status": (IntentType.DEEP_STATUS, "查看 Deep 任务状态"),
        "/deep_update": (IntentType.DEEP_UPDATE, "更新 Deep 任务上下文"),
        "/stop_deep": (IntentType.STOP_DEEP, "停止 Deep 任务"),
        "/spec": (IntentType.ENTER_SPEC, "进入 Spec 模式"),
        "/spec_status": (IntentType.SPEC_STATUS, "查看 Spec 任务状态"),
        "/stop_spec": (IntentType.STOP_SPEC, "停止 Spec 任务"),
        "/help": (IntentType.SHOW_HELP, "显示帮助信息"),
        "/帮助": (IntentType.SHOW_HELP, "显示帮助信息"),
        "/tools": (IntentType.SHOW_TOOLS, "查看所有可用工具"),
        "/tools_status": (IntentType.TOOLS_STATUS, "查看工具状态"),
    }

    SHELL_COMMANDS = {
        "ls",
        "pwd",
        "cd",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "echo",
        "mkdir",
        "touch",
        "rm",
        "cp",
        "mv",
        "chmod",
        "chown",
        "ln",
        "git",
        "npm",
        "yarn",
        "pnpm",
        "python",
        "pip",
        "uv",
        "node",
        "docker",
        "kubectl",
        "curl",
        "wget",
        "ssh",
        "scp",
        "rsync",
        "ps",
        "top",
        "kill",
        "df",
        "du",
        "free",
        "whoami",
        "date",
        "uname",
        "tar",
        "zip",
        "unzip",
        "gzip",
        "gunzip",
        "xz",
        "vim",
        "nano",
        "less",
        "more",
        "wc",
        "sort",
        "uniq",
        "awk",
        "sed",
        "brew",
        "apt",
        "yum",
        "pacman",
        "make",
        "cmake",
        "cargo",
        "go",
        "which",
        "whereis",
        "man",
        "env",
        "export",
        "source",
        "sh",
        "bash",
        "zsh",
        "alias",
        "ping",
        "netstat",
        "ifconfig",
        "ip",
        "nc",
        "telnet",
        "nslookup",
    }

    COMMON_WORDS = {
        "ok",
        "yes",
        "no",
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank",
        "good",
        "nice",
        "great",
        "cool",
        "fine",
        "sure",
        "please",
        "sorry",
        "wow",
        "test",
        "testing",
        "done",
        "ready",
        "start",
        "stop",
        "wait",
        "check",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "need",
        "want",
        "like",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "my",
        "your",
        "our",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
    }

    def __init__(self):
        self.settings = get_settings()
        self.intent_matcher = self._build_intent_matcher()

    EXIT_KEYWORDS = {"退出", "结束", "exit", "quit", "不用了", "算了", "停止"}
    PROJECT_SWITCH_KEYWORDS = {"切换项目", "换项目", "换到项目", "去项目", "打开项目"}
    PROJECT_LIST_KEYWORDS = {"项目列表", "所有项目", "有哪些项目", "看看项目"}
    DIR_KEYWORDS = {"切换目录", "去目录", "进入目录", "当前目录", "上级目录", "cd "}

    ENTER_COCO_KEYWORDS = {"进入编程模式", "编程模式", "开始编程", "进入coco", "coco模式"}
    ENTER_CLAUDE_KEYWORDS = {"进入claude模式", "claude模式", "进入claude", "使用claude"}
    ENTER_AIDEN_KEYWORDS = {"进入aiden模式", "aiden模式", "进入aiden", "使用aiden"}
    ENTER_CODEX_KEYWORDS = {"进入codex模式", "codex模式", "进入codex", "使用codex"}
    ENTER_GEMINI_KEYWORDS = {"进入gemini模式", "gemini模式", "进入gemini", "使用gemini"}
    ENTER_TRAEX_KEYWORDS = {"进入traex模式", "traex模式", "进入traex", "使用traex"}
    EXIT_MODE_KEYWORDS = {"退出模式", "退出编程模式"}

    DEEP_MODE_KEYWORDS = {"deep模式", "深度模式", "deep agent", "复杂任务", "大任务"}

    HELP_KEYWORDS = {"帮助", "help", "使用说明", "怎么用", "如何使用"}

    COMMAND_TYPO_MAP = {
        "/calude": "/claude",
        "/cluade": "/claude",
        "/cluad": "/claude",
        "/calud": "/claude",
        "/claud": "/claude",
        "/cooc": "/coco",
        "/coc": "/coco",
        "/cocoo": "/coco",
        "/exti": "/exit",
        "/eixt": "/exit",
        "/exut": "/exit",
        "/hlep": "/help",
        "/hepl": "/help",
        "/helo": "/help",
    }

    def _build_intent_matcher(self) -> IntentMatcher:
        matcher = IntentMatcher()
        for name, rule in (
            ("command_typo", IntentRecognizer._match_command_typo),
            ("exact_command", IntentRecognizer._match_exact_command),
            ("coco_info", IntentRecognizer._match_coco_info),
            ("new_chat", IntentRecognizer._match_new_chat_command),
            ("new_project", IntentRecognizer._match_new_project_command),
            ("switch_project", IntentRecognizer._match_switch_project_command),
            ("close_project", IntentRecognizer._match_close_project_command),
            ("deep_update", IntentRecognizer._match_deep_update_command),
            ("deep_status", IntentRecognizer._match_deep_status_command),
            ("stop_deep", IntentRecognizer._match_stop_deep_command),
            ("deep", IntentRecognizer._match_deep_command),
            ("spec_guide", IntentRecognizer._match_spec_guide_command),
            ("spec", IntentRecognizer._match_spec_command),
            ("exit_mode_keyword", IntentRecognizer._match_exit_mode_keyword),
            ("enter_coco_keyword", IntentRecognizer._match_enter_coco_keyword),
            ("enter_claude_keyword", IntentRecognizer._match_enter_claude_keyword),
            ("info_commands", IntentRecognizer._match_info_commands),
            ("enter_aiden_keyword", IntentRecognizer._match_enter_aiden_keyword),
            ("enter_codex_keyword", IntentRecognizer._match_enter_codex_keyword),
            ("enter_gemini_keyword", IntentRecognizer._match_enter_gemini_keyword),
            ("enter_traex_keyword", IntentRecognizer._match_enter_traex_keyword),
            ("programming_exit_keyword", IntentRecognizer._match_programming_exit_keyword),
            ("project_list_keyword", IntentRecognizer._match_project_list_keyword),
            ("cd_command", IntentRecognizer._match_cd_command),
            ("shell_command", IntentRecognizer._match_shell_command),
            ("shell_heuristic", IntentRecognizer._match_shell_heuristic),
        ):
            matcher.register(name, rule)
        return matcher

    @staticmethod
    def _lower(text: str) -> str:
        return text.lower().strip()

    def _match_command_typo(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = text.lower().strip()
        if text_lower in self.COMMAND_TYPO_MAP:
            corrected = self.COMMAND_TYPO_MAP[text_lower]
            if corrected in self.EXACT_COMMANDS:
                intent, desc = self.EXACT_COMMANDS[corrected]
                return IntentResult.single(
                    intent=intent,
                    confidence=0.95,
                    original_text=text,
                    reasoning=f"纠正拼写错误: {text_lower} -> {corrected}",
                    description=f"{desc}（已纠正拼写）",
                )
        return None

    def _match_exact_command(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if text_lower in self.EXACT_COMMANDS:
            intent, desc = self.EXACT_COMMANDS[text_lower]
            return IntentResult.single(
                intent=intent,
                confidence=1.0,
                original_text=text,
                reasoning=f"精确匹配命令: {text_lower}",
                description=desc,
            )
        return None

    def _match_coco_info(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if text_lower == "/coco_info":
            return IntentResult.single(
                intent=IntentType.COCO_MESSAGE,
                confidence=1.0,
                data={"command": "info"},
                original_text=text,
                reasoning="精确匹配: /coco_info",
                description="查看 Coco 会话信息",
            )
        return None

    def _match_new_chat_command(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if text_lower == "/new-chat" or text_lower.startswith("/new-chat "):
            parts = text.split()
            data = {}
            if len(parts) >= 2:
                data["name"] = parts[1]
            if len(parts) >= 3:
                data["suffix"] = parts[2]
            if len(parts) >= 4:
                data["path"] = parts[3]
            return IntentResult.single(
                intent=IntentType.NEW_CHAT_PROJECT,
                confidence=1.0,
                data=data,
                original_text=text,
                reasoning="精确匹配: /new-chat 命令",
                description="创建项目专属群",
            )
        return None

    def _match_new_project_command(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if text_lower.startswith("/new "):
            parts = text.split(None, 2)
            name = parts[1] if len(parts) >= 2 else "project"
            path = parts[2] if len(parts) >= 3 else ""
            return IntentResult.single(
                intent=IntentType.CREATE_PROJECT,
                confidence=1.0,
                data={"name": name, "path": path},
                original_text=text,
                reasoning="精确匹配: /new 命令",
                description=f"创建项目: {name}",
            )
        return None

    def _match_switch_project_command(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if text_lower.startswith("/switch "):
            name = text[8:].strip()
            return IntentResult.single(
                intent=IntentType.SWITCH_PROJECT,
                confidence=1.0,
                data={"name": name},
                original_text=text,
                reasoning="精确匹配: /switch 命令",
                description=f"切换到项目: {name}",
            )
        return None

    def _match_close_project_command(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if text_lower.startswith("/close "):
            name = text[7:].strip()
            return IntentResult.single(
                intent=IntentType.CLOSE_PROJECT,
                confidence=1.0,
                data={"name": name},
                original_text=text,
                reasoning="精确匹配: /close 命令",
                description=f"关闭项目: {name}",
            )
        return None

    def _match_deep_update_command(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if text_lower.startswith("/deep_update "):
            update_message = text[len("/deep_update ") :].strip()
            return IntentResult.single(
                intent=IntentType.DEEP_UPDATE,
                confidence=1.0,
                data={"message": update_message},
                original_text=text,
                reasoning="精确匹配: /deep_update 命令",
                description="更新 Deep Engine 上下文",
            )
        return None

    def _match_deep_status_command(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if text_lower.startswith("/deep_status "):
            # e.g. /deep_status all
            return IntentResult.single(
                intent=IntentType.DEEP_STATUS,
                confidence=1.0,
                data={"arg": text[len("/deep_status ") :].strip()},
                original_text=text,
                reasoning="前缀匹配: /deep_status 命令",
                description="查看 Deep Agent 任务状态",
            )
        return None

    def _match_stop_deep_command(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if text_lower.startswith("/stop_deep "):
            # e.g. /stop_deep all
            return IntentResult.single(
                intent=IntentType.STOP_DEEP,
                confidence=1.0,
                data={"arg": text[len("/stop_deep ") :].strip()},
                original_text=text,
                reasoning="前缀匹配: /stop_deep 命令",
                description="停止 Deep Agent 任务",
            )
        return None

    def _match_deep_command(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if text_lower.startswith("/deep "):
            requirement = text[6:].strip()
            return IntentResult.single(
                intent=IntentType.ENTER_DEEP,
                confidence=1.0,
                data={"requirement": requirement},
                original_text=text,
                reasoning="精确匹配: /deep 命令",
                description="启动 Deep Engine",
            )
        return None

    def _match_spec_guide_command(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if text_lower.startswith("/spec_guide "):
            guide_message = text[len("/spec_guide ") :].strip()
            return IntentResult.single(
                intent=IntentType.SPEC_GUIDE,
                confidence=1.0,
                data={"message": guide_message},
                original_text=text,
                reasoning="精确匹配: /spec_guide 命令",
                description="注入 Spec 引导信息",
            )
        return None

    def _match_spec_command(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if text_lower.startswith("/spec "):
            requirement = text[6:].strip()
            return IntentResult.single(
                intent=IntentType.ENTER_SPEC,
                confidence=1.0,
                data={"requirement": requirement},
                original_text=text,
                reasoning="精确匹配: /spec 命令",
                description="启动 Spec Engine",
            )
        return None

    def _match_exit_mode_keyword(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if any(kw in text_lower for kw in self.EXIT_MODE_KEYWORDS):
            return IntentResult.single(
                intent=IntentType.EXIT_MODE,
                confidence=0.95,
                original_text=text,
                reasoning="检测到退出模式关键词",
                description="退出当前模式",
            )
        return None

    def _match_enter_coco_keyword(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if any(kw in text_lower for kw in self.ENTER_COCO_KEYWORDS):
            return IntentResult.single(
                intent=IntentType.ENTER_COCO,
                confidence=0.95,
                original_text=text,
                reasoning="检测到进入 Coco 编程模式关键词",
                description="进入 Coco 编程模式",
            )
        return None

    def _match_enter_claude_keyword(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if any(kw in text_lower for kw in self.ENTER_CLAUDE_KEYWORDS):
            return IntentResult.single(
                intent=IntentType.ENTER_CLAUDE,
                confidence=0.95,
                original_text=text,
                reasoning="检测到进入 Claude 编程模式关键词",
                description="进入 Claude 编程模式",
            )
        return None

    def _match_info_commands(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        info_commands = {
            "/claude_info": (IntentType.CLAUDE_MESSAGE, "Claude"),
            "/aiden_info": (IntentType.AIDEN_MESSAGE, "Aiden"),
            "/codex_info": (IntentType.CODEX_MESSAGE, "Codex"),
            "/gemini_info": (IntentType.GEMINI_MESSAGE, "Gemini"),
            "/traex_info": (IntentType.TRAEX_MESSAGE, "Traex"),
        }
        if text_lower in info_commands:
            intent, name = info_commands[text_lower]
            return IntentResult.single(
                intent=intent,
                confidence=1.0,
                data={"command": "info"},
                original_text=text,
                reasoning=f"精确匹配: {text_lower}",
                description=f"查看 {name} 会话信息",
            )
        return None

    def _match_enter_aiden_keyword(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if any(kw in text_lower for kw in self.ENTER_AIDEN_KEYWORDS):
            return IntentResult.single(
                intent=IntentType.ENTER_AIDEN,
                confidence=0.95,
                original_text=text,
                reasoning="检测到进入 Aiden 编程模式关键词",
                description="进入 Aiden 编程模式",
            )
        return None

    def _match_enter_codex_keyword(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if any(kw in text_lower for kw in self.ENTER_CODEX_KEYWORDS):
            return IntentResult.single(
                intent=IntentType.ENTER_CODEX,
                confidence=0.95,
                original_text=text,
                reasoning="检测到进入 Codex 编程模式关键词",
                description="进入 Codex 编程模式",
            )
        return None

    def _match_enter_gemini_keyword(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if any(kw in text_lower for kw in self.ENTER_GEMINI_KEYWORDS):
            return IntentResult.single(
                intent=IntentType.ENTER_GEMINI,
                confidence=0.95,
                original_text=text,
                reasoning="检测到进入 Gemini 编程模式关键词",
                description="进入 Gemini 编程模式",
            )
        return None

    def _match_enter_traex_keyword(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if any(kw in text_lower for kw in self.ENTER_TRAEX_KEYWORDS):
            return IntentResult.single(
                intent=IntentType.ENTER_TRAEX,
                confidence=0.95,
                original_text=text,
                reasoning="检测到进入 Traex 编程模式关键词",
                description="进入 Traex 编程模式",
            )
        return None

    def _match_programming_exit_keyword(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        is_programming = current_mode in ("coco", "claude", "aiden", "codex", "gemini", "traex")
        if is_programming and len(text) < 20:
            if any(kw in text_lower for kw in self.EXIT_KEYWORDS):
                return IntentResult.single(
                    intent=IntentType.EXIT_MODE,
                    confidence=0.95,
                    original_text=text,
                    reasoning=f"{current_mode}模式下检测到退出关键词",
                    description="退出当前编程模式",
                )
        return None

    def _match_project_list_keyword(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        if any(kw in text_lower for kw in self.PROJECT_LIST_KEYWORDS):
            return IntentResult.single(
                intent=IntentType.LIST_PROJECTS,
                confidence=0.9,
                original_text=text,
                reasoning="检测到项目列表关键词",
                description="查看项目列表",
            )
        return None

    def _match_cd_command(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        first_word = text_lower.split()[0] if text_lower else ""
        if first_word == "cd":
            parts = text.strip().split(maxsplit=1)
            path = parts[1] if len(parts) > 1 else ""
            return IntentResult.single(
                intent=IntentType.CHANGE_DIR,
                confidence=1.0,
                data={"path": path},
                original_text=text,
                reasoning="cd 命令匹配为目录切换",
                description=f"切换目录: {path}" if path else "查看当前目录",
            )
        return None

    def _match_shell_command(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        first_word = text_lower.split()[0] if text_lower else ""
        if first_word in self.SHELL_COMMANDS:
            return IntentResult.single(
                intent=IntentType.SHELL_COMMAND,
                confidence=0.95,
                original_text=text,
                reasoning=f"Shell命令白名单匹配: {first_word}",
                description=f"执行命令: {text}",
            )
        return None

    def _match_shell_heuristic(self, text: str, current_mode: str) -> Optional[IntentResult]:
        text_lower = self._lower(text)
        first_word = text_lower.split()[0] if text_lower else ""
        if first_word in self.COMMON_WORDS:
            return None
        if self._looks_like_shell_token(first_word, text_lower):
            return IntentResult.single(
                intent=IntentType.SHELL_COMMAND,
                confidence=0.7,
                original_text=text,
                reasoning=f"可能是命令: {first_word}",
                description=f"执行命令: {text}",
            )

        return None

    def _quick_match(self, text: str, current_mode: str = "smart") -> Optional[IntentResult]:
        return self.intent_matcher.match(self, text, current_mode)

    def _normalize_path(self, path: str) -> str:
        if not path:
            return ""
        path = path.strip()
        if path.startswith("~"):
            path = os.path.expanduser(path)
        return path

    @staticmethod
    def _looks_like_shell_token(first_word: str, text_lower: str) -> bool:
        """Shared heuristic for command-like tokens.

        Returns True when the first whitespace-delimited token resembles a
        shell command name (lowercase identifier 2-15 chars) AND the text
        does not contain natural-language Chinese hint words that usually
        signal a programming request rather than a shell command.
        """
        if not first_word:
            return False
        if IntentRecognizer._looks_like_local_executable_path(first_word):
            return True
        if not re.match(r"^[a-z][a-z0-9_.-]*$", first_word):
            return False
        if not (2 <= len(first_word) <= 15):
            return False
        if any(kw in text_lower for kw in ("帮", "请", "能", "可以", "想", "吗", "呢")):
            return False
        return True

    @staticmethod
    def _looks_like_local_executable_path(first_word: str) -> bool:
        """Return True for direct local executable invocations like ./restart.sh."""
        token = (first_word or "").strip()
        if not token:
            return False
        if token.startswith(("./", "../", "~/")):
            tail = token.rsplit("/", 1)[-1]
            return bool(tail and tail not in {".", ".."})
        return False

    @classmethod
    def looks_like_shell(cls, text: str) -> bool:
        """Public: does *text* look like a shell command invocation?

        True when the first token is in the SHELL_COMMANDS whitelist, is
        literally ``cd``, or matches the command-token heuristic used by
        ``_quick_match`` (but not when it looks like a natural-language
        programming request). Empty text returns False.

        Callers (e.g. project-chat auto-Coco routing) use this to decide
        whether to fall through to shell execution instead of forwarding
        the text to a programming agent as a new requirement.
        """
        if not text:
            return False
        text_lower = text.lower().strip()
        if not text_lower:
            return False
        first_word = text_lower.split()[0]
        if first_word == "cd":
            return True
        if first_word in cls.SHELL_COMMANDS:
            return True
        if first_word in cls.COMMON_WORDS:
            return False
        return cls._looks_like_shell_token(first_word, text_lower)

    def _get_fallback_intent(self, current_mode: str) -> IntentType:
        if current_mode == "coco":
            return IntentType.COCO_MESSAGE
        elif current_mode == "claude":
            return IntentType.CLAUDE_MESSAGE
        elif current_mode == "aiden":
            return IntentType.AIDEN_MESSAGE
        elif current_mode == "codex":
            return IntentType.CODEX_MESSAGE
        elif current_mode == "gemini":
            return IntentType.GEMINI_MESSAGE
        elif current_mode == "traex":
            return IntentType.TRAEX_MESSAGE
        else:
            return IntentType.SHELL_COMMAND

    def recognize(self, text: str, current_mode: str = "smart") -> IntentResult:
        quick_result = self._quick_match(text, current_mode)
        if quick_result:
            return quick_result

        # Default ACP tool: forward unmatched SMART mode messages to configured tool
        default_tool = self.settings.default_acp_tool
        if default_tool and current_mode == "smart":
            _TOOL_INTENT_MAP = {
                "coco": IntentType.ENTER_COCO,
                "claude": IntentType.ENTER_CLAUDE,
                "aiden": IntentType.ENTER_AIDEN,
                "codex": IntentType.ENTER_CODEX,
                "gemini": IntentType.ENTER_GEMINI,
            }
            intent = _TOOL_INTENT_MAP.get(default_tool.lower())
            if intent:
                return IntentResult.single(
                    intent=intent,
                    confidence=0.6,
                    data={"auto_forward": True},
                    original_text=text,
                    reasoning=f"默认工具转发: {default_tool}",
                    description=f"转发到 {default_tool}",
                )
            else:
                logger.warning("无效的 DEFAULT_ACP_TOOL: %s，回退 shell", default_tool)

        # Fallback: in programming mode → forward to current mode; otherwise → shell
        fallback = self._get_fallback_intent(current_mode)
        return IntentResult.single(
            intent=fallback,
            confidence=0.5,
            original_text=text,
            reasoning="规则未匹配，使用默认意图",
            description=f"执行: {text}",
        )
