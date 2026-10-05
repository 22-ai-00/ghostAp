"""Verify unattended instructions at the actual ACP prompt transport boundary."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

from src.acp.execution_instructions import AUTONOMOUS_EXECUTION_PROMPT
from src.acp.session import ACPSession


def test_acp_prompt_keeps_original_request_on_initial_and_resumed_turns(
    tmp_path: Path,
) -> None:
    class Connection:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def prompt(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(stop_reason="end_turn")

        async def load_session(self, **kwargs):
            self.loaded = kwargs

    session = ACPSession(agent_cmd="test", agent_args=[], cwd=str(tmp_path))
    connection = Connection()
    session._conn = connection
    session._session_id = "initial-session"
    original = "只分析这个问题；不要修改文件。\n保留  exact spaces  和换行。"

    async def exercise() -> None:
        await session.prompt(original)
        await session.load_session("resumed-session")
        await session.prompt(original)

    asyncio.run(exercise())

    assert connection.loaded == {
        "cwd": str(tmp_path),
        "session_id": "resumed-session",
    }
    assert [call["session_id"] for call in connection.calls] == [
        "initial-session",
        "resumed-session",
    ]
    for call in connection.calls:
        assert [block.text for block in call["prompt"]] == [
            AUTONOMOUS_EXECUTION_PROMPT,
            original,
        ]
    assert "明确限定只做分析" in AUTONOMOUS_EXECUTION_PROMPT
    assert "实际调用工具并核实结果" in AUTONOMOUS_EXECUTION_PROMPT
