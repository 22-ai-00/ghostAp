from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urldefrag

from src.workflow_engine.constants import NODE_MIN_VERSION

ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = [
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    ROOT / "AGENTS.md",
    ROOT / "CLAUDE.md",
    *sorted((ROOT / "docs").glob("*.md")),
]

ARCHIVED_DOC_PATHS = [
    ROOT / ".plans",
    ROOT / "docs" / "superpowers",
]

REMOVED_ARTIFACT_REFERENCES = {
    "2025-04-25-" + "multi-chat-isolation-design",
    "2026-04-29-" + "new-chat-project-design",
    "2026-04-30-" + "card-refactor-design",
    "2026-04-30-" + "card-refactor-plan",
    "acp_" + "architecture.md",
    "card-migration-" + "faq.md",
    "docs/" + "plan.md",
    "docs/" + "superpowers",
    ".plans",
    "card-pipeline-review-fixes",
    "card-migration-tasks",
    "card-session-migration-tasks",
    "card-cleanup-tasks",
    "topic-scoped-engine-sessions",
    "adaptive-spec-review-roles",
    "unified_card_" + "v1",
    "unified_card_" + "v2",
    "check_shim_" + "deadline",
}


def test_archived_doc_noise_paths_are_removed() -> None:
    violations = [path.relative_to(ROOT).as_posix() for path in ARCHIVED_DOC_PATHS if path.exists()]

    assert violations == []


def test_retained_docs_do_not_reference_removed_cleanup_artifacts() -> None:
    violations: list[str] = []
    for path in DOC_PATHS:
        text = path.read_text(encoding="utf-8")
        for needle in REMOVED_ARTIFACT_REFERENCES:
            if needle in text:
                violations.append(f"{path.relative_to(ROOT)} references {needle}")

    assert violations == []


def test_local_markdown_links_in_retained_docs_resolve() -> None:
    violations: list[str] = []
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

    for path in DOC_PATHS:
        text = path.read_text(encoding="utf-8")
        for match in link_pattern.finditer(text):
            raw_target = match.group(1).strip()
            target, _fragment = urldefrag(raw_target)
            if not target or "://" in target or target.startswith("mailto:"):
                continue

            candidate = (path.parent / target).resolve()
            if not candidate.exists():
                violations.append(f"{path.relative_to(ROOT)} -> {raw_target}")

    assert violations == []


def test_readme_card_tree_documents_current_pipeline_directories() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    for directory in (
        "actions/",
        "delivery/",
        "events/",
        "render/",
        "session/",
        "state/",
        "timers/",
    ):
        assert f"│   │   ├── {directory}" in text

    old_card_summary = "卡片构建器（schema 2.0）" + "+ 流式更新 + 统一布局"
    assert old_card_summary not in text


def test_public_positioning_matches_product_contract() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    contract = (ROOT / "docs" / "product-contract.md").read_text(
        encoding="utf-8"
    )

    assert "主 Bot" in readme
    assert "provider/transport" in readme
    assert (
        'description = "飞书原生多工具 AI 研发控制面与执行网关"'
        in pyproject
    )
    for boundary in (
        "Main Bot: unified control plane",
        "Host Shell: direct host execution",
        "single-host and file-backed",
        "Direct programming keeps the selected Agent",
    ):
        assert boundary in contract


def test_public_docs_do_not_reference_retired_employee_department() -> None:
    for path in (
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "docs" / "product-contract.md",
        ROOT / "pyproject.toml",
    ):
        text = path.read_text(encoding="utf-8")
        assert "Agent Department" not in text, path.name
        assert "src/autonomous" not in text, path.name
        assert "/hire" not in text, path.name


def test_readme_discovers_all_programming_session_info_commands() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    modes_section = readme.split("### 模式与模型", maxsplit=1)[1].split(
        "### 项目",
        maxsplit=1,
    )[0]

    for command in (
        "/coco_info",
        "/claude_info",
        "/aiden_info",
        "/codex_info",
        "/gemini_info",
        "/traex_info",
        "/grok_info",
        "/dsh_info",
        "/agy_info",
    ):
        assert f"`{command}`" in modes_section


def test_agents_declares_linux_macos_compatibility_contract() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "至少支持 Linux 与 macOS" in agents
    assert "Linux-only 假设" in agents
    assert "平台分支或能力探测" in agents


def test_workflow_node_minimum_is_consistent_across_runtime_and_readme() -> None:
    runtime_package = json.loads(
        (ROOT / "src" / "workflow_engine" / "runtime" / "package.json").read_text(
            encoding="utf-8"
        )
    )
    minimum = ".".join(str(part) for part in NODE_MIN_VERSION)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert runtime_package["engines"]["node"] == f">={minimum}"
    assert NODE_MIN_VERSION[1:] == (0, 0), "README's major+ notation would be too broad"
    assert f"Node.js {NODE_MIN_VERSION[0]}+" in readme


def test_readme_documents_workflow_auto_bound_agent_pool() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "`/workflow_help`" in readme
    assert "1–8 个 `tool+model` Agent" in readme
    assert "自动绑定" in readme
    assert "使用此池开始编排" not in readme


def test_product_contract_documents_workflow_auto_bound_agent_pool() -> None:
    contract = (ROOT / "docs" / "product-contract.md").read_text(
        encoding="utf-8"
    )

    assert "auto-binds an available recommended Agent Pool" in contract
    assert "1-8 `tool+model` Agents" in contract
    assert "owner-confirmed Agent Pool" not in contract


def test_agents_md_stays_concise_and_links_project_skills() -> None:
    agents_path = ROOT / "AGENTS.md"
    agents_text = agents_path.read_text(encoding="utf-8")
    assert len(agents_text.splitlines()) <= 110

    skills_root = ROOT / ".agents" / "skills"
    skill_files = sorted(skills_root.glob("*/SKILL.md"))
    assert skill_files, "Expected at least one project skill under .agents/skills/"

    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    frontmatter_re = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    for skill_file in skill_files:
        rel = skill_file.relative_to(ROOT).as_posix()
        assert rel in agents_text, f"AGENTS.md must index skill {rel}"

        content = skill_file.read_text(encoding="utf-8")
        fm = frontmatter_re.match(content)
        assert fm is not None, f"Missing YAML frontmatter in {rel}"
        header = fm.group(1)
        assert re.search(rf"^name:\s*{re.escape(skill_file.parent.name)}\s*$", header, re.MULTILINE), rel
        assert re.search(r"^description:\s*\S+", header, re.MULTILINE), rel

        for match in link_pattern.finditer(content):
            raw_target = match.group(1).strip()
            target, _fragment = urldefrag(raw_target)
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            assert (skill_file.parent / target).resolve().exists(), f"{rel} -> {raw_target}"


def test_memory_abstract_stays_rolling_and_links_resolve() -> None:
    memory_dir = ROOT / ".Memory"
    abstract = memory_dir / "Abstract.md"
    text = abstract.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 80

    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for match in link_pattern.finditer(text):
        raw_target = match.group(1).strip()
        target, _fragment = urldefrag(raw_target)
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        assert (memory_dir / target).resolve().exists(), f".Memory/Abstract.md -> {raw_target}"
