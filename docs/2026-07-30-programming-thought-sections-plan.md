# Programming Thought Sections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render ordinary programming prose as distinct Feishu Card 2.0 thought sections and label subagent sections with a safe task brief without breaking streaming, pagination, or the one-main-card contract.

**Architecture:** Preserve ACP/TextBlock boundaries as the semantic source of sections; do not split text heuristically on punctuation or blank lines. Carry non-opaque source attribution through `TEXT_STARTED`, then wrap only ordinary programming text atoms in legal, initially expanded `collapsible_panel` sections. Keep the body markdown as the existing CardKit streaming target and make page signatures recurse into nested elements while excluding active `element_id` content.

**Tech Stack:** Python 3.13, frozen dataclasses, pure card reducers/renderers, Feishu Interactive Card schema 2.0, pytest, uv.

## Global Constraints

- Use `uv` only.
- Keep one Feishu main card per ordinary programming task; only capacity pagination may create continuation cards.
- Do not expose ACP `source_id`, tool-call IDs, structured JSON, terminal controls, or secrets in section headings.
- Do not infer section boundaries from punctuation or `\n\n`; use existing TextBlock/source/tool boundaries.
- Do not add a fake callback to `interactive_container`; read-only bordered sections use official `collapsible_panel.background_color` and `border`.
- Preserve nested CardKit `element_id` streaming and the original monotonic delivery sequence.
- Deep, Spec, Worktree, Workflow, Autonomous, and generic/static cards keep their existing text rendering.
- Preserve the existing parallel-subtask and execution-history fold components byte-for-byte in visual fields.
- Keep actual rendered pages below the 200-element and 30 KiB Feishu limits.
- Preserve all pre-existing dirty worktree files and do not include them in task changes.

---

### Task 1: Reviewable UX Contract

**Files:**
- Create: `ux/programming-thought-sections.html`

**Interfaces:**
- Consumes: Existing programming header, subtask summary, final-answer-first, and folded execution-history visual contracts.
- Produces: A light-theme Before/After contract plus a 360 px mobile specimen.

- [x] **Step 1: Create the preview before production edits**

  Show one continuous prose stream before the change and, after the change, blue main-Agent progress, grey subagent output with `子代理 · {任务简述}`, and a green final answer.

- [x] **Step 2: Render and inspect the preview**

  Open the local HTML at desktop and 360 px widths. Confirm full-width single-column blocks, 8–12 px spacing/radius, readable Markdown/code, no animation, and no more than three normal-state colors.

### Task 2: Lock the Source-Attribution Contract with Failing Tests

**Files:**
- Modify: `tests/test_programming_card_session.py`
- Modify: `tests/test_card_reducers.py`
- Create: `tests/test_programming_text_sections.py`

**Interfaces:**
- Consumes: `ACPEvent.source_id`, `_agent_summaries[*].label`, and existing TextBlock boundaries.
- Produces: A `TextBlock` source contract with main/subagent kind, safe visible sequence/label, and a non-rendered identity token for coalescing.

- [x] **Step 1: Add reducer and adapter failures**

  Assert that `TEXT_STARTED` attribution survives DELTA/DONE; main text defaults safely; an agent TOOL_START followed by same-source TEXT_CHUNK records the task brief; unrelated sources remain distinct.

- [x] **Step 2: Add renderer failures**

  Assert that two ordinary programming TextBlocks become two schema-2.0 bordered sections, the last completed main block is titled `最终答复`, and a subagent section title uses the safe task brief without its opaque source ID.

- [x] **Step 3: Add isolation and sanitization failures**

  Assert engine/static cards retain the old renderer, unsafe Markdown/control/secret material is absent from headings, and a one-character subagent fragment cannot coalesce into another source.

- [x] **Step 4: Run RED**

  Run:

  ```bash
  uv run python -m pytest \
    tests/test_programming_text_sections.py \
    tests/test_programming_card_session.py \
    tests/test_card_reducers.py -q
  ```

  Expected: the new attribution/section assertions fail because TextBlock has no source metadata and text still renders as bare Markdown.

### Task 3: Implement Source-Aware Static Sections

**Files:**
- Modify: `src/card/state/models.py`
- Modify: `src/card/events/payloads.py`
- Modify: `src/card/events/factories.py`
- Modify: `src/card/state/reducers/text.py`
- Modify: `src/card/programming_adapter.py`
- Create: `src/card/render/programming_sections.py`
- Modify: `src/card/render/atoms.py`
- Modify: `src/card/render/renderer.py`

**Interfaces:**
- Consumes: `CardEvent.text_started(block_id, source_kind, source_sequence, source_label, source_ref)`.
- Produces: `TextBlock.source_kind`, `.source_sequence`, `.source_label`, `.source_ref`; `render_programming_text_section(...)`.

- [x] **Step 1: Extend the immutable text contract**

  Add optional attribution fields with safe defaults:

  ```python
  source_kind: Literal["main", "subagent"] = "main"
  source_sequence: str | None = None
  source_label: str | None = None
  source_ref: str = "main"
  ```

- [x] **Step 2: Capture attribution at the adapter boundary**

  Main text uses `source_kind="main"`. Only a source already registered in `_agent_summaries` is a subagent; other provider source IDs remain main prose while retaining a bounded hashed `source_ref` for stream isolation.

- [x] **Step 3: Render legal Card 2.0 sections**

  Use an initially expanded `collapsible_panel` with:

  ```python
  {
      "tag": "collapsible_panel",
      "expanded": True,
      "background_color": "blue-50",
      "border": {"color": "blue-100", "corner_radius": "8px"},
      "padding": "8px 12px",
      "vertical_spacing": "4px",
      "header": {"title": {"tag": "markdown", "content": "**主 Agent · 当前进展**"}},
      "elements": [body_markdown],
  }
  ```

  Main progress is blue, subagent output grey, and the terminal last main answer green. System error/cancellation/archive text remains on its established renderer and is not wrapped as thought prose.

- [x] **Step 4: Scope the feature explicitly**

  Add a metadata opt-in set only by `build_programming_metadata()` so generic `engine_type is None` states and all engine cards remain unchanged.

- [x] **Step 5: Preserve source-aware coalescing**

  Merge a pathological one-character TextBlock only when both atoms resolve to the same `source_ref`.

### Task 4: Preserve Streaming and Capacity

**Files:**
- Modify: `src/card/render/renderer.py`
- Modify: `src/card/render/atoms.py`
- Modify: `src/card/render/pagination.py`
- Modify: `tests/test_programming_text_sections.py`
- Modify: `tests/test_card_pagination.py`
- Modify: `tests/test_card_budget_regression.py`

**Interfaces:**
- Consumes: Nested markdown with an active `element_id`, styled text atom node/byte estimates.
- Produces: Stable structure signatures across active text deltas and conservative pagination.

- [x] **Step 1: Add nested-streaming RED tests**

  Assert an active section keeps its nested `element_id`; changing only active content leaves `structure_signature` unchanged; changing the element ID or static heading changes it.

- [x] **Step 2: Make page signatures recursive**

  Traverse nested Card 2.0 components. For the selected streaming markdown include only its `element_id`; hash content for every other markdown, including concurrently active sources. This keeps the selected element incremental without silently dropping a second source.

- [x] **Step 3: Account for the wrapper**

  Styled text atoms count the panel, title markdown, and body markdown nodes plus bounded JSON overhead. Split atoms preserve that accounting and the original block ID.

- [x] **Step 4: Add budget and split regressions**

  Verify many short thought sections paginate under the official node/byte caps and long Markdown/code fences remain complete and independently parseable.

### Task 5: Verification, Documentation, and Review

**Files:**
- Modify: `.Memory/2026-07-30.md`
- Modify: `.Memory/Abstract.md`

**Interfaces:**
- Consumes: Completed implementation and test evidence.
- Produces: Project memory describing the UI contract, reason, verification, and remaining real-tenant/mobile risk.

- [x] **Step 1: Run focused tests**

  ```bash
  uv run python -m pytest \
    tests/test_programming_text_sections.py \
    tests/test_programming_card_session.py \
    tests/test_card_reducers.py \
    tests/test_card_renderer.py \
    tests/test_card_pagination.py \
    tests/test_card_budget_regression.py -q
  ```

- [x] **Step 2: Run adjacent card/delivery tests**

  ```bash
  uv run python -m pytest \
    tests/test_card_e2e.py \
    tests/test_card_delivery_page_mutator.py \
    tests/test_card_execution_flow.py \
    tests/test_programming_completion_guards.py -q
  ```

- [x] **Step 3: Run quality gates**

  ```bash
  uv run ruff check src/card/ tests/test_programming_text_sections.py
  uv run python -m src.main --validate
  git diff --check
  ```

- [x] **Step 4: Expand according to shared-card risk**

  Run the non-slow suite because renderer signatures, pagination, and CardState are shared:

  ```bash
  uv run python -m pytest tests/ -q -m "not slow"
  ```

- [x] **Step 5: Update memory and perform final review**

  Record exact test counts and the remaining requirement for real Feishu desktop/mobile visual acceptance. Re-read the user request and verify: visually distinct prose blocks, subagent task brief heading, unchanged parallel-subtask/execution-history folds, one main card, official schema-2.0 fields, streaming, and capacity.
