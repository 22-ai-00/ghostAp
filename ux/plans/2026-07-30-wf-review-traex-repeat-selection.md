# WF Review Traex Repeat Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the Traex model cascade open after adding a review Agent so a second Traex reviewer can be selected and the review selection can then be confirmed.

**Architecture:** Treat `SelectionFlowController.set_step()` as an idempotent state transition: changing steps clears transient tool/model picker state, while reaffirming the current step preserves it. The existing review handler may continue normalizing callbacks to step 2 without destroying the cascade state needed for multi-selection.

**Tech Stack:** Python 3, pytest, Feishu CardKit card dictionaries, `uv`

## Global Constraints

- Use `uv` for every Python and test command.
- Add a targeted regression test before modifying production behavior.
- Keep the fix local to Workflow selection state; do not refactor unrelated card or handler code.
- Preserve existing behavior when actually moving between steps.

---

### Task 1: Reproduce the repeated Traex review selection state loss

**Files:**
- Modify: `tests/test_workflow_selection_controller.py`

**Interfaces:**
- Consumes: `SelectionFlowController.set_step(step: int) -> None`, `select_tool(...)`, `set_model_group(...)`, and `build_review_combined_card(...)`
- Produces: A regression contract that same-step callback normalization preserves the active tool and model cascade

- [x] **Step 1: Write the failing test**

```python
def test_set_step_same_review_step_preserves_traex_picker_for_second_selection():
    ctrl = SelectionFlowController(step=2)
    ctrl.select_tool("traex", is_review=True)
    ctrl.set_model_group("traex", "DeepSeek-V4-Flash", is_review=True)

    ctrl.set_step(2)

    assert ctrl.pending_tool_name == "traex"
    assert ctrl.pending_model_group == "DeepSeek-V4-Flash"
    card = ctrl.build_review_combined_card(
        available_tools=[{"tool_name": "traex", "display_name": "Traex"}],
        available_models=[
            {
                "name": "DeepSeek-V4-Flash/standard/high",
                "display_name": "DeepSeek-V4-Flash/standard/high",
            }
        ],
    )
    assert "workflow_review_select_model" in str(card)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `uv run python -m pytest tests/test_workflow_selection_controller.py::TestStepNavigation::test_set_step_same_review_step_preserves_traex_picker_for_second_selection -q`

Expected: FAIL because `pending_tool_name` is reset to `None`.

### Task 2: Make same-step transitions idempotent

**Files:**
- Modify: `src/workflow_engine/selection_flow.py`
- Test: `tests/test_workflow_selection_controller.py`

**Interfaces:**
- Consumes: The regression contract from Task 1
- Produces: `set_step()` that clears transient picker state only when `step != self.step`

- [x] **Step 1: Write the minimal implementation**

```python
def set_step(self, step: int) -> None:
    if step not in (1, 2, 3):
        raise ValueError(f"step must be 1, 2, or 3, got {step!r}")
    if step == self.step:
        return
    self.step = step
    self.pending_tool_name = None
    self.model_page = 0
    self._reset_model_cascade()
```

- [x] **Step 2: Run the focused regression test**

Run: `uv run python -m pytest tests/test_workflow_selection_controller.py::TestStepNavigation::test_set_step_same_review_step_preserves_traex_picker_for_second_selection -q`

Expected: PASS.

- [x] **Step 3: Run adjacent Workflow selection tests**

Run: `uv run python -m pytest tests/test_workflow_selection_controller.py tests/test_workflow_orchestrator_select.py -q`

Expected: PASS with no failures or warnings attributable to the change.

### Task 3: Verify repository gates and record the decision

**Files:**
- Modify: `.Memory/2026-07-30.md`
- Modify: `.Memory/Abstract.md`

**Interfaces:**
- Consumes: Passing targeted Workflow tests
- Produces: Persistent project record with root cause, fix, verification, and residual risk

- [x] **Step 1: Run the Workflow regression suite**

Run: `uv run python -m pytest tests/test_workflow*.py -q -m "not slow"`

Expected: PASS.

- [x] **Step 2: Run static and diff checks**

Run: `uv run ruff check src/workflow_engine/selection_flow.py tests/test_workflow_selection_controller.py`

Expected: PASS.

Run: `git diff --check`

Expected: no output and exit code 0.

- [x] **Step 3: Update project memory**

Record that repeated review callbacks called non-idempotent `set_step(2)`, which erased the active Traex picker despite `keep_panel_open=True`; document the idempotent transition fix and exact passing commands.
