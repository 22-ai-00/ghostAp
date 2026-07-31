"""Record the direct session-manager remote-call topology baseline.

This benchmark intentionally has no wall-clock gate.  It detects extra remote
or LLM hops by recording the factory/prompt sequence for every run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from src.acp.manager import ACPSessionManager
from tests.helpers.session_call_recorder import SessionCallRecorder


def run_once(run_number: int) -> tuple[str, ...]:
    recorder = SessionCallRecorder()
    manager = ACPSessionManager("codex", session_starter=recorder.session_factory)
    chat_id = f"benchmark-chat-{run_number}"
    project_id = "benchmark-project"
    session = manager.ensure_session(
        chat_id,
        cwd="/tmp/ghostap-direct-lane-benchmark",
        project_id=project_id,
        model_name="benchmark-model",
    )
    recorder.observe_manager_session_key(
        session,
        chat_id=chat_id,
        project_id=project_id,
        thread_id=None,
        session_key=manager._session_key(chat_id, project_id),
    )
    session.send_prompt("record direct topology")
    topology = recorder.remote_call_topology()
    expected = ("factory:codex", "prompt:codex")
    if topology != expected:
        raise RuntimeError(
            f"direct lane topology changed on run {run_number}: {topology!r}; expected {expected!r}"
        )
    return topology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20, help="number of topology samples")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    samples = [run_once(index) for index in range(1, args.runs + 1)]
    topology_distribution = Counter(" -> ".join(sample) for sample in samples)
    print(
        json.dumps(
            {
                "runs": args.runs,
                "topology_distribution": dict(sorted(topology_distribution.items())),
                "wall_clock_threshold": None,
                "admission": "all samples contain exactly one target factory and one target prompt",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
