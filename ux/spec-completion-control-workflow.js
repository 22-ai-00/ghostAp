export const meta = {
  name: "spec-completion-control-analysis",
  description: "Analyze spec mode goal completion control mechanisms and propose approaches",
  phases: [
    { title: "Code Exploration", detail: "Explore current spec engine architecture and goal tracking" },
    { title: "Problem Diagnosis", detail: "Identify root causes of completion uncertainty and edge cases" },
    { title: "Solution Design", detail: "Generate multiple approaches for goal completion verification" },
    { title: "Adversarial Review", detail: "Independent verification of proposed approaches by reviewers" },
    { title: "Synthesis", detail: "Synthesize final recommendation with trade-off analysis" },
  ],
  maxConcurrent: 6,
  tools: ["traex", "coco"],
  patterns: ["fanout", "verify"],
};

export default async function() {
  const task = "现在的spec模式的目标完成度上不是太可控，可能需要增加一个目标完成度把控的逻辑，你帮分析下如何操作比较合适，先不要动手改代码";

  // Phase 1: Parallel code exploration of spec engine
  phase("Code Exploration");
  log("Exploring spec engine codebase in parallel...");

  const [specCore, specPersistence, specHandlers, relatedEngines] = await parallel([
    () => agent(
      `Read and summarize the core spec engine architecture. Focus on:
      1. src/spec_engine/ directory structure and key modules
      2. How spec tasks are defined, decomposed, and tracked
      3. What constitutes "completion" in current spec flow
      4. State machine and lifecycle of a spec task
      5. How progress is reported to cards/UI

      Use rg and Read to explore src/spec_engine/ thoroughly. Read key files fully.
      Report: module map, data flow, completion signals, and any existing checkpoints.`,
      { tool: "traex", role: "code-explorer", label: "explore-spec-core", phase: "Code Exploration", timeout: 240 }
    ),
    () => agent(
      `Read and analyze spec result persistence and context management:
      1. src/spec_engine/ - find SpecManager.persistResult and how results are stored
      2. How does spec know when it's "done"? Search for completion/done/finish signals
      3. Check hooks and callbacks - ContextPersistenceHook and any spec-related hooks
      4. Look at how spec interacts with project/thread/chat context
      5. Check src/mode/ for spec mode state transitions

      Use rg to find keywords: "spec", "persist", "complete", "done", "finish", "goal".
      Report: persistence flow, completion detection logic, state transitions.`,
      { tool: "traex", role: "code-explorer", label: "explore-spec-persistence", phase: "Code Exploration", timeout: 240 }
    ),
    () => agent(
      `Analyze spec command handlers and user interaction flow:
      1. src/feishu/handlers/ - find the spec command handler (likely /spec or similar)
      2. How is a spec task initiated? What parameters does it accept?
      3. How is progress shown to users? Card updates, message streams?
      4. What user controls exist (stop, cancel, review, approve)?
      5. How does spec mode remain isolated from other execution strategies (deep, workflow)?

      Search for spec-related handlers, card builders for spec (src/card/), and any spec-specific UI.
      Report: handler flow, UI/card rendering, user controls, interaction protocol.`,
      { tool: "traex", role: "code-explorer", label: "explore-spec-handlers", phase: "Code Exploration", timeout: 240 }
    ),
    () => agent(
      `Analyze other engines for completion control patterns to learn from:
      1. src/deep_engine/ - how does deep mode handle completion? Look at ContextPersistenceHook
      2. src/workflow_engine/ - how does workflow track phase completion and verify task success?
      3. src/autonomous/manager/cards.py + src/autonomous/coordinator.py - team/employee workflow state and card orchestration patterns for intent confidence
      4. Any verification/validation loops in other engines (e.g., verify patterns, checkpoints)

      Compare: which engines have the strongest completion guarantees? What patterns do they use?
      Report: patterns from other engines that could be adapted for spec mode.`,
      { tool: "traex", role: "code-explorer", label: "explore-related-engines", phase: "Code Exploration", timeout: 240 }
    ),
  ]);

  // Phase 2: Problem diagnosis from multiple angles
  phase("Problem Diagnosis");
  log("Diagnosing root causes of completion uncertainty...");

  const [gapAnalysis, failureModes, stateModel] = await parallel([
    () => agent(
      `Based on this code exploration, analyze why spec mode completion is "not controllable":

      === Spec Core Architecture ===
      ${specCore}

      === Persistence & Completion Signals ===
      ${specPersistence}

      === Handler & UI Flow ===
      ${specHandlers}

      Specifically identify:
      1. Where does spec silently stop without verifying all goals are met?
      2. Are sub-goals/decomposed tasks tracked individually?
      3. Is there a definition of "done" per requirement vs just stopping?
      4. What feedback loops exist between AI output and goal verification?
      5. Can users see per-goal completion status?
      6. Are there race conditions where AI thinks it's done but output is incomplete?

      Be specific: cite file paths, function names, and code patterns that cause the problem.
      Focus on concrete failure modes, not vague complaints.`,
      { tool: "traex", role: "gap-analyst", label: "gap-analysis", phase: "Problem Diagnosis", timeout: 240 }
    ),
    () => agent(
      `Analyze failure modes in spec mode completion based on the codebase:

      === Patterns from other engines ===
      ${relatedEngines}

      === Spec Architecture ===
      ${specCore}

      Classify failure modes into categories:
      1. Premature termination: AI stops early, thinking work is complete
      2. Scope creep: AI drifts from original goals but reports completion
      3. Partial delivery: some goals met, others silently dropped
      4. Ambiguous success: no objective criteria to verify completion
      5. State corruption: internal state marks done but deliverables missing
      6. User expectation mismatch: AI's "done" differs from user's "done"

      For each failure mode, identify: frequency (high/med/low), detectability, and which code path allows it.
      Reference how other engines (deep, workflow) prevent or detect similar failures.`,
      { tool: "traex", role: "failure-analyst", label: "failure-modes", phase: "Problem Diagnosis", timeout: 240 }
    ),
    () => agent(
      `Build a formal model of spec task state and goal tracking:

      === Spec Core ===
      ${specCore}

      === Persistence ===
      ${specPersistence}

      Produce:
      1. Current state model: What states does a spec task pass through? What transitions exist?
      2. Goal tracking model: How are individual goals represented? Are they explicit?
      3. Completion predicate: What boolean condition gates "done"? Is it checkable?
      4. Gap in model: What state is missing to properly track per-goal completion?
      5. Ideal state model: What additions would make completion objectively verifiable?

      Draw the state machine (ASCII) for both current and ideal models.
      Be precise about data structures needed, not just vague ideas.`,
      { tool: "coco", role: "state-modeler", label: "state-model", phase: "Problem Diagnosis", timeout: 240 }
    ),
  ]);

  // Phase 3: Generate solution approaches
  phase("Solution Design");
  log("Generating solution approaches...");

  const approaches = await fanout(
    `The problem: Spec mode goal completion is not controllable. AI agents prematurely mark tasks as done, partially deliver, or drift from original goals without detection.

    === Gap Analysis ===
    ${gapAnalysis}

    === Failure Modes ===
    ${failureModes}

    === Current vs Ideal State Model ===
    ${stateModel}

    Design a solution approach for spec completion control. Your approach should address:
    1. How to make goals explicit and trackable (goal decomposition schema)
    2. How to verify each goal is actually met before marking complete
    3. How to prevent premature termination (checkpoint/verification gates)
    4. How to surface completion status to users (per-goal progress)
    5. How to integrate with existing spec engine without major rewrites
    6. What existing patterns from other engines (workflow verify, deep hooks) can be reused

    Be specific about:
    - Data structures to add
    - Code modules to modify/add
    - State machine changes
    - Verification heuristics vs LLM-based checks
    - Integration points with existing spec flow
    - Backward compatibility considerations

    Think about this from YOUR assigned perspective, producing a distinct approach.`,
    [
      {
        prompt: "${input}\n\nYou are a PRAGMATIC engineer. Propose the MINIMUM viable change that adds meaningful completion control. Focus on checkpoint hooks and explicit goal checklists with LLM verification at each gate. Minimal new modules, maximum reuse of existing patterns. Keep it simple, avoid over-engineering.",
        tool: "traex",
        role: "pragmatic-architect",
      },
      {
        prompt: "${input}\n\nYou are a ROBUSTNESS-focused engineer. Propose a STRONG verification approach inspired by adversarial patterns. Consider: pre/post conditions for each goal, independent verification agent, self-reflection loops, test-driven spec validation. Model after workflow_engine's verify() pattern. Error on the side of catching incomplete work even if it means more iterations.",
        tool: "traex",
        role: "robustness-architect",
      },
      {
        prompt: "${input}\n\nYou are a UX-CENTRIC architect. Design a solution that makes completion transparent to users. Consider: per-goal checklists in cards, user approval gates, inline completion indicators, the ability to drill into partial deliverables. Look at how card rendering works for workflow engine progress and adapt similar per-goal granularity for spec mode. Users should SEE and CONTROL completion, not just trust the AI.",
        tool: "coco",
        role: "ux-architect",
      },
    ],
    {
      synthesizerTool: "coco",
      synthesizerRole: "lead-architect",
      synthesisPrompt: `You have three architectural approaches for spec completion control:

      APPROACH 1 (Pragmatic/Minimal):
      {{worker0}}

      APPROACH 2 (Robust/Adversarial):
      {{worker1}}

      APPROACH 3 (UX-Centric/Transparent):
      {{worker2}}

      Synthesize these into a coherent analysis. For each approach produce:
      1. Core idea summary (2-3 sentences)
      2. Key components/changes needed
      3. Strengths (what it handles well)
      4. Weaknesses (what it misses or overdoes)
      5. Estimated complexity (low/med/high) and files touched
      6. Risk assessment

      Then propose a RECOMMENDED HYBRID that takes the best from each:
      - What minimum verification gates are essential?
      - What UX elements are needed for transparency?
      - What can be deferred to v2?
      - What is the incremental implementation order?

      Output a structured comparison table and a phased implementation roadmap (v1 minimum viable → v2 enhanced → v3 full).
      Do NOT write code. This is an analysis and recommendation document.`,
    }
  );

  // Phase 4: Adversarial verification by reviewers
  phase("Adversarial Review");
  log("Running independent adversarial reviews...");

  const { accepted, output: reviewed, feedback } = await verify(approaches, {
    criteria: `The analysis must:
      1. Be grounded in the actual codebase (cite specific files and functions)
      2. Address all 6 failure modes identified in diagnosis
      3. Be implementable without rewriting the entire spec engine
      4. Include concrete data structures, not just abstract ideas
      5. Consider backward compatibility with existing spec usage
      6. Propose a clear incremental path (not a big-bang rewrite)
      7. Correctly identify what patterns from other engines are reusable
      8. Not over-engineer the solution (proportional to the problem)
      9. Address lock ordering and thread safety if adding state
      10. Consider impact on card rendering pipeline (import boundaries!)`,
    verifiers: [
      {
        tool: "coco",
        role: "skeptical-reviewer",
        focus: `Challenge these proposals aggressively:
        - Are the proposed approaches actually addressing the root cause, or just symptoms?
        - Will adding verification gates slow down spec mode unacceptably?
        - Is there a simpler approach that was missed?
        - Are the recommended integrations consistent with GhostAP's architecture (import boundaries, lock order, handler->session->render layering)?
        - What new failure modes do the proposed solutions introduce?
        - Be specific about gaps and weaknesses in the recommendation.`,
      },
      {
        tool: "traex",
        role: "correctness-reviewer",
        focus: `Verify correctness and completeness of the analysis:
        - Does the state model accurately reflect how spec engine actually works (check against code)?
        - Are all code references accurate (read files to verify)?
        - Do the proposed changes respect card pipeline import boundaries (handler -> session -> render, render doesn't import delivery)?
        - Are lock ordering implications considered (chat_lock, repo_lock, utils/lock_order)?
        - Is the phased implementation realistic given code dependencies?
        - Check for any incorrect assumptions about how other engines work.`,
      },
      {
        tool: "traex",
        role: "simplicity-reviewer",
        focus: `Review for over-engineering and unnecessary complexity:
        - Which parts of the proposal are YAGNI (You Ain't Gonna Need It)?
        - Can the v1 be even smaller than proposed while still solving the core problem?
        - Are there existing hooks/utilities that the proposal duplicates?
        - Is the adversarial verification loop itself creating new reliability problems?
        - What is the SIMPLEST thing that could meaningfully improve completion control?
        - Apply the "three similar lines is better than premature abstraction" principle.`,
      },
    ],
    maxRounds: 2,
    reviseTool: "coco",
    reviseRole: "lead-architect",
  });

  // Phase 5: Final synthesis
  phase("Synthesis");
  log("Producing final recommendation...");

  const finalSynthesis = await agent(
    `Produce the final analysis document for spec mode goal completion control.

    === Reviewed Approaches (with adversarial feedback incorporated) ===
    ${reviewed}

    === Reviewer Feedback ===
    ${JSON.stringify(feedback, null, 2)}

    Structure your output as:

    ## Problem Summary
    Brief statement of why spec completion is uncontrollable, citing specific code paths.

    ## Root Cause Analysis
    The fundamental gaps in current spec engine (per failure modes from diagnosis).

    ## Current Architecture Recap
    Brief summary of how spec works today (state model, completion signals, gaps).

    ## Approaches Considered
    For each approach (pragmatic, robust, ux-centric): summary, strengths, weaknesses, complexity.

    ## Recommended Approach
    The hybrid recommendation with clear rationale for what was included/excluded.

    ## Proposed Design
    Concrete design:
    - Data structures (goal schema, checkpoint state)
    - Module changes (which files, what modifications)
    - State machine changes (new states, transitions, verification gates)
    - Card/UI changes (per-goal progress display)
    - Integration points with existing spec flow

    ## Implementation Phases
    - Phase 1 (MVP): Minimum changes for meaningful completion control
    - Phase 2 (Enhanced): Stronger verification and UX
    - Phase 3 (Full): Advanced patterns (adversarial loops, user gates)

    ## Risks & Open Questions
    What could go wrong, what needs user decision before implementation.

    Do NOT write code. This is a design analysis document only.
    Be specific, cite files, and make actionable recommendations.`,
    { tool: "coco", role: "technical-writer", label: "final-synthesis", phase: "Synthesis", timeout: 180 }
  );

  return finalSynthesis;
}
