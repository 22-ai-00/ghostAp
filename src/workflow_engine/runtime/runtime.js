/**
 * GhostAP Workflow Runtime Harness
 *
 * Communicates with the Python host via JSON-RPC 2.0 over stdin/stdout (NDJSON).
 * Exposes orchestration primitives (agent, parallel, pipeline, phase, log)
 * as globals to user workflow scripts.
 */

import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { readFileSync } from 'node:fs';
import { AsyncLocalStorage } from 'node:async_hooks';
import vm from 'node:vm';

// ---------------------------------------------------------------------------
// Path sanitization (strip absolute paths and stack traces from messages)
// ---------------------------------------------------------------------------

function sanitizePath(msg) {
  if (typeof msg !== 'string') return msg || '';
  // Replace absolute paths, keeping only the last filename component
  let cleaned = msg.replace(/(?:\/[\w.\-]+){2,}\/([\w.\-]+)/g, '<path>/$1');
  // Remove Node.js stack trace lines
  cleaned = cleaned.replace(/\n\s+at .+/g, '');
  return cleaned.trim();
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let requestId = 0;
const pendingRequests = new Map(); // id -> { resolve, reject }
let cancelled = false;
const DEFAULT_MAX_TOTAL_AGENTS = 200;
let maxTotalAgents = DEFAULT_MAX_TOTAL_AGENTS;
let totalAgentCalls = 0;

// Concurrency cap for parallel(). Matches the Python ThreadPoolExecutor
// size so both sides of the bridge agree on the upper bound. A falsy
// or non-positive value disables the cap (unbounded).
let maxConcurrent = 0;

// Host-provided workflow deadline. Python remains authoritative; JS uses this
// to avoid starting work that cannot finish before the bridge hard timeout.
let workflowStartedMs = 0;
let workflowDeadlineMs = 0;
let workflowTotalTimeoutMs = 0;
// Host-provided per-agent timeout floor (ms). The Python executor treats the
// configured workflow_agent_call_timeout_s as the authoritative floor and only
// lets the script's baked timeout *raise* it. The JS-side agent_call watchdog
// must agree, otherwise a small script timeout (e.g. 180s) fires here and
// aborts a legitimately long-running agent() call before the host does. A
// value of 0 means "unlimited" per-agent — the watchdog then relies only on
// the total workflow deadline (or never fires if that is also unlimited).
let workflowAgentCallTimeoutMs = 0;
const REQUEST_TIMEOUT_GRACE_MS = 10000;
// Large finite backstop mirroring AGENT_UNLIMITED_BACKSTOP_S on the Python
// side: even in "unlimited" mode a bounded timer eventually fires so a wedged
// host cannot hang the runtime forever. 30 days in ms.
const AGENT_UNLIMITED_BACKSTOP_MS = 30 * 24 * 3600 * 1000;
// Node's setTimeout silently clamps any delay > 2^31-1 ms (~24.8 days) down to
// 1ms, which would make an "unlimited" watchdog fire *immediately*. Clamp every
// timer delay to this safe maximum so a huge/omitted timeout degrades to a very
// long (but real) watchdog instead of an instant abort.
const MAX_SAFE_TIMER_MS = 2147483647;

// Lexically-scoped request collectors. Descendant async work inherits the
// current lineage while concurrent siblings remain isolated. A global stack
// cannot provide that guarantee because overlapping race()/pipeline() calls
// would register each sibling's request IDs in every active collector.
const requestCollectorContext = new AsyncLocalStorage();
const raceContext = new AsyncLocalStorage();

function withRequestCollector(collector, callback) {
  const parentLineage = requestCollectorContext.getStore() || [];
  return requestCollectorContext.run([...parentLineage, collector], callback);
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

class CancelledError extends Error {
  constructor() {
    super('Workflow cancelled by host');
    this.name = 'CancelledError';
  }
}

class JsonRpcError extends Error {
  constructor({ code, message, data } = {}) {
    super(message || 'JSON-RPC error');
    this.name = 'JsonRpcError';
    this.code = code;
    this.data = data;
  }

  toJSON() {
    return { name: this.name, message: this.message, code: this.code, data: this.data };
  }
}

// ---------------------------------------------------------------------------
// Transport (NDJSON over stdin/stdout)
// ---------------------------------------------------------------------------

function send(obj) {
  const line = JSON.stringify(obj) + '\n';
  process.stdout.write(line);
}

function sendNotification(method, params = {}) {
  send({ jsonrpc: '2.0', method, params });
}

// Exit only AFTER stdout/stderr have drained to the host pipe.
//
// process.exit() truncates any output still buffered in an async pipe. During a
// real workflow the runtime emits a large backlog of log/phase frames, so the
// terminal `done`/`error` frame written just before exit is frequently still in
// the buffer — process.exit() then discards it and the Python bridge sees a
// bare stdout EOF, surfacing the misleading "Runtime process closed stdout
// unexpectedly" error instead of the true result/error. Draining first
// guarantees the final frame reaches the host. A bounded fallback timer still
// forces exit if a wedged pipe never drains, so we can never hang the runtime.
function flushAndExit(code) {
  let exited = false;
  const doExit = () => {
    if (exited) return;
    exited = true;
    process.exit(code);
  };
  // Stop reading stdin so nothing keeps the event loop alive past the flush.
  try { process.stdin.pause(); } catch { /* stdin may already be closed */ }

  let pending = 0;
  const onStreamDrained = () => {
    pending -= 1;
    if (pending <= 0) doExit();
  };
  // A zero-length write's callback fires once the stream's buffer has flushed
  // to the OS pipe — the portable way to await drain for both empty and
  // backlogged buffers.
  for (const stream of [process.stdout, process.stderr]) {
    try {
      pending += 1;
      stream.write('', onStreamDrained);
    } catch {
      pending -= 1; // stream already destroyed — ignore
    }
  }
  if (pending <= 0) {
    doExit();
    return;
  }
  // Safety net: never let an un-drainable pipe hang the process.
  const timer = setTimeout(doExit, 2000);
  if (typeof timer.unref === 'function') timer.unref();
}

function remainingWorkflowMs() {
  if (!workflowDeadlineMs) return Infinity;
  return workflowDeadlineMs - Date.now();
}

function workflowDeadlineError(label) {
  return new JsonRpcError({
    code: -32002,
    message: `Workflow deadline exceeded before ${label || 'request'}`,
  });
}

// Resolve the effective per-agent timeout in seconds, mirroring the Python
// executor: the host floor (workflowAgentCallTimeoutMs) is authoritative and a
// script-provided value may only RAISE it, never lower it. A floor of 0 means
// unlimited → a large finite backstop is used so a timer still eventually
// fires. The result is NOT yet capped by the remaining workflow deadline; that
// happens in capTimeoutToDeadline / requestTimeoutMs where a live deadline
// exists.
function effectiveAgentTimeoutSeconds(requestedSeconds) {
  const floorMs = workflowAgentCallTimeoutMs > 0
    ? workflowAgentCallTimeoutMs
    : AGENT_UNLIMITED_BACKSTOP_MS;
  const floorSeconds = Math.max(1, Math.floor(floorMs / 1000));
  const requested = Number(requestedSeconds) > 0 ? Math.floor(Number(requestedSeconds)) : 0;
  return requested > 0 ? Math.max(floorSeconds, requested) : floorSeconds;
}

function capTimeoutToDeadline(timeoutSeconds) {
  // Host floor is authoritative; script value may only raise it. This replaces
  // the previous behavior of honoring the (often tiny) script timeout, which
  // killed long-running agent() calls before the host's own deadline.
  const effective = effectiveAgentTimeoutSeconds(timeoutSeconds);
  const remainingMs = remainingWorkflowMs();
  if (!Number.isFinite(remainingMs)) return Math.max(1, effective);

  const usableMs = remainingMs - REQUEST_TIMEOUT_GRACE_MS;
  if (usableMs <= 0) {
    throw workflowDeadlineError('agent_call');
  }
  const usableSeconds = Math.max(1, Math.floor(usableMs / 1000));
  return Math.max(1, Math.min(effective, usableSeconds));
}

function requestTimeoutMs(method, params) {
  const remainingMs = remainingWorkflowMs();
  let timeoutMs = 0;
  if (method === 'agent_call') {
    // params.timeout is already the floor-aware effective value set by agent()
    // via capTimeoutToDeadline; fall back to the host floor if it is missing.
    const effectiveSeconds = effectiveAgentTimeoutSeconds(params && params.timeout);
    timeoutMs = Math.max(1, Math.ceil(effectiveSeconds * 1000) + REQUEST_TIMEOUT_GRACE_MS);
  } else if (Number.isFinite(remainingMs)) {
    timeoutMs = Math.max(1, Math.floor(remainingMs));
  }

  if (Number.isFinite(remainingMs)) {
    timeoutMs = timeoutMs > 0
      ? Math.min(timeoutMs, Math.max(1, Math.floor(remainingMs)))
      : Math.max(1, Math.floor(remainingMs));
  }
  // Clamp to Node's safe setTimeout ceiling: a delay > 2^31-1 ms is silently
  // truncated to 1ms by Node, which would make an "unlimited" watchdog fire
  // instantly. Clamping keeps it a very-long (but real) timer instead.
  if (timeoutMs > MAX_SAFE_TIMER_MS) timeoutMs = MAX_SAFE_TIMER_MS;
  return timeoutMs > 0 ? timeoutMs : 0;
}

function clearPendingTimer(entry) {
  if (entry && entry.timer) {
    clearTimeout(entry.timer);
    entry.timer = null;
  }
}

function sendRequest(method, params = {}) {
  if (cancelled) {
    return Promise.reject(new CancelledError());
  }

  const id = ++requestId;

  // Notify collectors in this request's lexical async lineage only.
  const collectorLineage = requestCollectorContext.getStore() || [];
  if (method === 'agent_call' && collectorLineage.length > 0) {
    for (const collector of collectorLineage) {
      collector.ids.add(id);
      if (params && params.label) {
        collector.labels.set(id, params.label);
      }
    }
  }

  return new Promise((resolve, reject) => {
    const entry = { resolve, reject, aborted: false, timer: null };
    const timeoutMs = requestTimeoutMs(method, params);
    if (timeoutMs > 0) {
      entry.timer = setTimeout(() => {
        const current = pendingRequests.get(id);
        if (!current || current.aborted) return;
        current.aborted = true;
        pendingRequests.delete(id);
        if (method === 'agent_call') {
          sendNotification('abort_request', { request_id: id });
        }
        reject(new JsonRpcError({
          code: -32002,
          message: `${method} timed out waiting for host response`,
        }));
      }, timeoutMs);
    }
    pendingRequests.set(id, entry);
    send({ jsonrpc: '2.0', id, method, params });
  });
}

// ---------------------------------------------------------------------------
// Incoming message dispatcher
// ---------------------------------------------------------------------------

function handleMessage(line) {
  if (!line.trim()) return;

  let msg;
  try {
    msg = JSON.parse(line);
  } catch {
    debugLog(`[runtime] Failed to parse incoming message: ${line}`);
    return;
  }

  // Response to a request we sent
  if (msg.id != null && pendingRequests.has(msg.id)) {
    const entry = pendingRequests.get(msg.id);
    pendingRequests.delete(msg.id);
    clearPendingTimer(entry);

    // If the request was already aborted, ignore the late response
    if (entry.aborted) return;

    if (msg.error) {
      // Preserve JSON-RPC error code/data so the JS orchestration layer
      // Runtime primitives such as agent() can respond to structured failure
      // payloads such as `{ kind: "missing_tools", ... }` produced by the
      // Python bridge. Falling back to `new Error(msg.error.message)`
      // would silently drop these attachments.
      entry.reject(
        new JsonRpcError({
          code: msg.error.code,
          message: msg.error.message,
          data: msg.error.data,
        }),
      );
    } else {
      entry.resolve(msg.result);
    }
    return;
  }

  // Notification from Python host (no id)
  if (msg.method && msg.id == null) {
    switch (msg.method) {
      case 'cancel':
        cancelled = true;
        // Reject all pending requests
        for (const [id, entry] of pendingRequests) {
          clearPendingTimer(entry);
          if (!entry.aborted) {
            entry.aborted = true;
            entry.reject(new CancelledError());
          }
        }
        pendingRequests.clear();
        break;

      case 'init':
        // Handled separately in main() via initPromise
        if (initResolve) {
          initResolve(msg.params);
          initResolve = null;
        }
        break;

      default:
        debugLog(`[runtime] Unknown notification: ${msg.method}`);
    }
    return;
  }
}

// ---------------------------------------------------------------------------
// Init synchronisation
// ---------------------------------------------------------------------------

let initResolve = null;

function waitForInit() {
  return new Promise((resolve) => {
    initResolve = resolve;
  });
}

// ---------------------------------------------------------------------------
// Debug logging (stderr only)
// ---------------------------------------------------------------------------

function debugLog(msg) {
  process.stderr.write(`${msg}\n`);
}

// ---------------------------------------------------------------------------
// Meta validation
// ---------------------------------------------------------------------------

function validateMeta(meta) {
  if (!meta || typeof meta !== 'object') {
    throw new Error('Script must export `const meta` object');
  }
  if (typeof meta.name !== 'string' || !meta.name) {
    throw new Error('meta.name must be a non-empty string');
  }
  if (typeof meta.description !== 'string' || !meta.description) {
    throw new Error('meta.description must be a non-empty string');
  }
  if (!Array.isArray(meta.phases) || meta.phases.length === 0) {
    throw new Error('meta.phases must be a non-empty array');
  }
  for (let i = 0; i < meta.phases.length; i++) {
    const p = meta.phases[i];
    if (!p || typeof p.title !== 'string' || typeof p.detail !== 'string') {
      throw new Error(`meta.phases[${i}] must have string "title" and "detail" fields`);
    }
  }
  if (meta.maxConcurrent != null && typeof meta.maxConcurrent !== 'number') {
    throw new Error('meta.maxConcurrent must be a number if provided');
  }
  if (meta.tools != null && !Array.isArray(meta.tools)) {
    throw new Error('meta.tools must be an array if provided');
  }
}

// ---------------------------------------------------------------------------
// Schema validation (compact recursive shape check)
// ---------------------------------------------------------------------------

function matchesSchema(value, schema) {
  if (typeof schema === 'string') {
    const aliases = {
      str: 'string',
      list: 'array',
      dict: 'object',
      float: 'number',
      int: 'integer',
      bool: 'boolean',
      none: 'null',
      '*': 'any',
    };
    const knownTypes = new Set([
      'string', 'array', 'object', 'number', 'integer', 'boolean', 'null', 'any',
    ]);
    let expected = aliases[schema.trim().toLowerCase()] || schema.trim().toLowerCase();
    if (!knownTypes.has(expected)) expected = 'string';
    if (expected === 'any') return true;
    if (expected === 'array') return Array.isArray(value);
    if (expected === 'object') {
      return typeof value === 'object' && value !== null && !Array.isArray(value);
    }
    if (expected === 'number') return typeof value === 'number' && Number.isFinite(value);
    if (expected === 'integer') return Number.isInteger(value);
    if (expected === 'null') return value === null;
    return typeof value === expected;
  }
  if (schema === null) return value === null;
  if (typeof schema === 'boolean') return typeof value === 'boolean';
  if (typeof schema === 'number') return typeof value === 'number' && Number.isFinite(value);
  if (Array.isArray(schema)) {
    if (!Array.isArray(value)) return false;
    if (schema.length === 0) return true;
    return value.every((item) => matchesSchema(item, schema[0]));
  }
  if (typeof schema === 'object') {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
    return Object.keys(schema).every(
      (key) => key in value && matchesSchema(value[key], schema[key]),
    );
  }
  return false;
}

function isFailedResult(value) {
  return Boolean(
    value
    && typeof value === 'object'
    && typeof value.error === 'string'
    && value.error.trim(),
  );
}

function partitionResults(results) {
  const successes = [];
  const failures = [];
  results.forEach((value, index) => {
    if (isFailedResult(value)) {
      failures.push({
        index,
        error: sanitizePath(value.error),
        stage: value.stage || 'agent_call',
      });
    } else {
      successes.push({ index, value });
    }
  });
  return { successes, failures };
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

async function agent(promptOrOpts, opts = {}) {
  if (cancelled) throw new CancelledError();
  const activeRace = raceContext.getStore();
  if (activeRace && activeRace.cancelled) {
    return { error: activeRace.timeoutMessage, stage: 'race_timeout' };
  }

  // Support both agent("prompt", {opts}) and agent({prompt, ...opts})
  let prompt;
  if (typeof promptOrOpts === 'object' && promptOrOpts !== null) {
    const { prompt: p, ...rest } = promptOrOpts;
    prompt = p;
    opts = rest;
  } else {
    prompt = promptOrOpts;
  }

  const agentId = typeof opts.agentId === 'string' ? opts.agentId.trim() : '';

  if (totalAgentCalls >= maxTotalAgents) {
    return {
      error: `Agent call limit exceeded (${maxTotalAgents})`,
      stage: 'agent_limit',
    };
  }
  totalAgentCalls += 1;

  let effectiveTimeout;
  try {
    effectiveTimeout = capTimeoutToDeadline(opts.timeout);
  } catch (err) {
    return { error: err && err.message ? err.message : String(err), stage: 'deadline' };
  }

  const params = {
    prompt,
    ...(agentId && { agent_id: agentId }),
    ...(opts.tool && { tool: opts.tool }),
    ...(opts.model && { model: opts.model }),
    ...(opts.role && { role: opts.role }),
    ...(opts.schema && { schema: opts.schema }),
    ...(opts.label && { label: opts.label }),
    ...(opts.phase && { phase: opts.phase }),
    timeout: effectiveTimeout,
  };

  // Backpressure retry: -32000 errors from the bridge (queue full, pool
  // exhausted) are transient — retry with exponential backoff + jitter.
  const MAX_BACKPRESSURE_RETRIES = 3;
  const BACKPRESSURE_BASE_DELAY_MS = 500;

  async function callWithBackpressureRetry() {
    for (let attempt = 0; attempt <= MAX_BACKPRESSURE_RETRIES; attempt++) {
      if (cancelled) throw new CancelledError();
      try {
        return await sendRequest('agent_call', params);
      } catch (err) {
        if (err instanceof CancelledError) throw err;
        const isBackpressure = err instanceof JsonRpcError && err.code === -32000;
        if (isBackpressure && attempt < MAX_BACKPRESSURE_RETRIES) {
          const delay = BACKPRESSURE_BASE_DELAY_MS * (2 ** attempt) * (0.5 + Math.random() * 0.5);
          debugLog(`[runtime] agent_call backpressure (attempt ${attempt + 1}/${MAX_BACKPRESSURE_RETRIES}), retrying in ${Math.round(delay)}ms`);
          await new Promise((resolve) => setTimeout(resolve, delay));
          continue;
        }
        throw err;
      }
    }
    // Unreachable — loop either returns or throws
    throw new Error('agent_call: backpressure retries exhausted');
  }

  let result;
  try {
    result = await callWithBackpressureRetry();
  } catch (err) {
    if (err instanceof CancelledError) throw err;
    return { error: sanitizePath(err.message), stage: 'agent_call' };
  }

  // AgentExecutor owns structured-output retries. Repeating agent_call here
  // would multiply its attempts (formerly up to 3×3 calls). The runtime only
  // propagates the authoritative error and performs one defense-in-depth check.
  if (typeof result === 'object' && result !== null && result.error) {
    return { error: sanitizePath(String(result.error)), stage: result.stage || 'agent_call' };
  }
  const payload = typeof result === 'object' && result !== null && 'data' in result
    ? result.data
    : result;
  if (opts.schema && !matchesSchema(payload, opts.schema)) {
    return { error: 'Structured output schema validation failed at runtime boundary', stage: 'schema' };
  }
  return payload;
}

// NOTE: parallel() runs N agent-call descriptors truly concurrently via
// Promise.all when `maxConcurrent` is unset/zero. When a cap is supplied, a
// lightweight in-flight semaphore keeps the JS side aligned with the Python
// ThreadPoolExecutor size so that neither side starves the other. The cap is
// controlled by the host via the init message (`params.max_concurrent`),
// which itself is bounded by HARD_MAX_CONCURRENT on the Python side. pipeline()
// reuses this scheduler for item-level parallelism while keeping stages within
// each item sequential.
async function parallel(items) {
  if (!Array.isArray(items)) {
    throw new TypeError('parallel() expects an array');
  }

  // When a concurrency cap is configured, bound the number of in-flight
  // promises with a simple semaphore so both sides of the bridge agree
  // on the parallelism upper bound. Without this cap, Promise.all would
  // start N HTTP/subprocess trips at once and starve downstream limits.
  const cap = typeof maxConcurrent === 'number' && maxConcurrent > 0
    ? Math.min(maxConcurrent, items.length || 1)
    : 0;

  if (!cap) {
    return Promise.all(items.map((item) => {
      if (typeof item === 'function') return item();
      if (typeof item === 'object' && item !== null && item.prompt) {
        return agent(item);
      }
      throw new TypeError('parallel() items must be functions or agent-call descriptors');
    }));
  }

  let inFlight = 0;
  let cursor = 0;
  const results = new Array(items.length);
  let rejector = null;
  let settled = false;

  return new Promise((resolve, reject) => {
    rejector = reject;

    function launch(index) {
      if (settled) return;
      if (inFlight >= cap) return;
      if (index >= items.length) return;
      const item = items[index];
      let p;
      if (typeof item === 'function') {
        try {
          p = item();
        } catch (err) {
          settled = true;
          rejector(err);
          return;
        }
      } else if (typeof item === 'object' && item !== null && item.prompt) {
        p = agent(item);
      } else {
        settled = true;
        rejector(new TypeError('parallel() items must be functions or agent-call descriptors'));
        return;
      }
      inFlight += 1;
      Promise.resolve(p).then((value) => {
        if (settled) return;
        results[index] = value;
        inFlight -= 1;
        if (cursor < items.length) launch(cursor++);
        else if (inFlight === 0) { settled = true; resolve(results); }
      }, (err) => {
        if (settled) return;
        settled = true;
        rejector(err);
      });
    }

    // Seed up to `cap` in-flight tasks.
    const initial = Math.min(cap, items.length);
    for (let i = 0; i < initial; i++) {
      cursor = i + 1;
      launch(i);
    }
    // All items smaller than cap; in-flight completion will resolve.
    if (!settled && cursor >= items.length && inFlight === 0) resolve(results);
  });
}

async function pipeline(items, ...args) {
  // pipeline(items, ...stages) — bounded parallel/map: items run concurrently
  // through parallel(); each item flows through stages sequentially.
  // NOTE: this is NOT an outer-level concurrent primitive for stage fan-out —
  // use parallel() if you need heterogeneous tasks fanned out concurrently.
  // For strictly sequential item processing use sequence(...) / serialPipeline(...).
  if (!Array.isArray(items)) {
    throw new TypeError('pipeline() expects an array of items as first argument');
  }
  if (cancelled) throw new CancelledError();

  // Extract options from last argument if it's an object (not a function)
  let stages = args;
  let options = {};
  if (args.length > 0 && typeof args[args.length - 1] === 'object' && args[args.length - 1] !== null) {
    options = args[args.length - 1];
    stages = args.slice(0, -1);
  }

  const continueOnFailure = options.continueOnFailure || options.continue_on_failure || false;

  // When continueOnFailure is false, track in-flight agent_call requests so we
  // can abort other items' pending calls on first failure and avoid wasting
  // Python-side thread pool resources.
  let pipelineRequestIds = null;
  let pipelineRequestLabels = null;
  let pipelineInterceptor = null;

  if (!continueOnFailure) {
    pipelineInterceptor = { ids: new Set(), labels: new Map() };
    pipelineRequestIds = pipelineInterceptor.ids;
    pipelineRequestLabels = pipelineInterceptor.labels;
  }

  function abortInFlight(firstError) {
    if (!pipelineRequestIds) return;
    for (const rid of pipelineRequestIds) {
      const entry = pendingRequests.get(rid);
      if (entry && !entry.aborted) {
        const label = (pipelineRequestLabels && pipelineRequestLabels.get(rid)) || `pipeline-item-${rid}`;
        abortRequest(rid);
        sendNotification('agent_aborted', {
          label,
          reason: `pipeline failure: ${firstError && firstError.message ? firstError.message : 'unknown'}`,
          request_id: rid,
        });
      }
    }
  }

  let failed = false;

  const itemTasks = items.map((item) => async () => {
      let current = item;
      for (let i = 0; i < stages.length; i++) {
        const stage = stages[i];
        try {
          current = await stage(current);
        } catch (err) {
          if (continueOnFailure) {
            debugLog(`[runtime] Pipeline stage ${i} failed for item, continuing: ${err.message}`);
            return {
              error: sanitizePath(err.message),
              failedAtStage: i,
              partialResult: current,
            };
          }
          // Aborted items are not the root cause — just re-throw so
          // Promise.all rejects with the real first error.
          if (err instanceof CancelledError) {
            throw err;
          }
          if (!failed) {
            failed = true;
            abortInFlight(err);
          }
          throw err;
        }
      }
      return current;
    });

  const runItems = () => parallel(itemTasks);
  const results = pipelineInterceptor
    ? await withRequestCollector(pipelineInterceptor, runItems)
    : await runItems();

  return results;
}

// ---------------------------------------------------------------------------
// Dynamic Workflow Patterns — Higher-order orchestration primitives
// ---------------------------------------------------------------------------

/**
 * classify(input, categories, opts) — Classify-and-Act pattern.
 *
 * Uses a classifier agent to categorize input, then routes to the appropriate
 * handler. Each category maps to a handler function that receives the input
 * and classification result.
 *
 * @param {string} input - The input to classify
 * @param {Object} categories - Map of category name → { description, handler }
 * @param {Object} opts - Options: { classifierTool, classifierModel, classifierPrompt }
 * @returns {*} Result from the matched handler
 */
async function classify(input, categories, opts = {}) {
  if (cancelled) throw new CancelledError();
  if (!input || typeof input !== 'string') {
    throw new TypeError('classify() expects a string input as first argument');
  }
  if (!categories || typeof categories !== 'object') {
    throw new TypeError('classify() expects a categories object as second argument');
  }

  const categoryNames = Object.keys(categories);
  if (categoryNames.length < 2) {
    throw new Error('classify() requires at least 2 categories');
  }

  const categoryDescriptions = categoryNames
    .map(name => `- "${name}": ${categories[name].description || name}`)
    .join('\n');

  const classifierPrompt = opts.classifierPrompt ||
    `Classify the input into exactly one category.\n\n${categoryDescriptions}\n\nInput:\n${input}\n\nReturn JSON only: { "category": "one exact category name" }.`;

  const classification = await agent(classifierPrompt, {
    agentId: opts.classifierAgentId || opts.agentId,
    tool: opts.classifierTool || opts.tool,
    model: opts.classifierModel || opts.model,
    role: 'classifier',
    label: opts.label ? `${opts.label}-classify` : 'classify',
    schema: opts.schema || { category: '' },
    timeout: opts.classifierTimeout || opts.timeout,
  });

  const rawCategory = typeof classification === 'string'
    ? classification
    : classification && classification.category;
  const normalizedCategory = String(rawCategory || '').trim().toLowerCase();
  const matched = categoryNames.find(name => name.toLowerCase() === normalizedCategory);
  const fallbackCategory = categoryNames.includes(opts.defaultCategory)
    ? opts.defaultCategory
    : null;
  const selectedCategory = matched || fallbackCategory;
  if (!selectedCategory) {
    return {
      error: isFailedResult(classification)
        ? classification.error
        : 'Classifier returned no exact category',
      stage: 'classify',
      categories: categoryNames,
    };
  }
  const handler = categories[selectedCategory].handler;

  if (typeof handler === 'function') {
    return handler(input, selectedCategory, classification);
  }
  if (typeof handler === 'object' && handler !== null && handler.prompt) {
    return agent({
      ...handler,
      prompt: handler.prompt.replace('${input}', input),
      timeout: handler.timeout || opts.handlerTimeout || opts.timeout,
    });
  }

  return { category: selectedCategory, input, classification };
}

/**
 * fanout(input, workers, opts) — Fan-out-and-Synthesize pattern.
 *
 * Splits work across multiple specialized agents running in parallel,
 * then synthesizes their outputs into a unified result.
 *
 * @param {string|Object} input - The shared input/context
 * @param {Array} workers - Array of { prompt, tool, role, label, ... } agent descriptors
 * @param {Object} opts - Options: { synthesizerTool, synthesizerPrompt, synthesizerRole }
 * @returns {*} Synthesized result
 */
async function fanout(input, workers, opts = {}) {
  if (cancelled) throw new CancelledError();
  if (!Array.isArray(workers) || workers.length === 0) {
    throw new TypeError('fanout() expects a non-empty array of workers');
  }

  const inputStr = typeof input === 'string' ? input : JSON.stringify(input);

  const tasks = workers.map((worker, idx) => {
    const prompt = typeof worker === 'string'
      ? worker
      : (worker.prompt || '').replace('${input}', inputStr);
    return {
      prompt,
      agentId: worker.agentId || opts.workerAgentId || opts.agentId,
      tool: worker.tool || opts.defaultTool,
      model: worker.model,
      role: worker.role || `worker-${idx}`,
      label: worker.label || `fanout-${idx}`,
      schema: worker.schema,
      phase: worker.phase,
      timeout: worker.timeout || opts.workerTimeout || opts.timeout,
    };
  });

  const results = await parallel(tasks);

  if (opts.synthesize === false) {
    return results;
  }

  const { successes, failures } = partitionResults(results);
  if (successes.length === 0) {
    return { error: 'fanout(): all workers failed', stage: 'fanout', failures };
  }

  const MAX_RESULT_LEN = 2000;
  const resultSummary = successes
    .map(({ index, value }) => {
      const r = value;
      const i = index;
      const text = typeof r === 'string' ? r : JSON.stringify(r);
      const truncated = text.length > MAX_RESULT_LEN
        ? text.slice(0, MAX_RESULT_LEN) + `\n... [truncated ${text.length - MAX_RESULT_LEN} chars]`
        : text;
      return `[${workers[i].role || workers[i].label || `worker-${i}`}]:\n${truncated}`;
    })
    .join('\n\n---\n\n');

  const synthPrompt = opts.synthesizerPrompt ||
    `You are a synthesizer. Combine and reconcile these parallel results into a unified, coherent output.\n\nOriginal input: ${inputStr}\n\nParallel results:\n${resultSummary}\n\nProvide a synthesized result that captures the best insights from all workers. Resolve any conflicts by choosing the most well-supported conclusion.`;

  const synthesized = await agent(synthPrompt, {
    agentId: opts.synthesizerAgentId || opts.agentId || workers[0]?.agentId,
    tool: opts.synthesizerTool || opts.tool,
    model: opts.synthesizerModel || opts.model,
    role: opts.synthesizerRole || 'synthesizer',
    label: opts.label ? `${opts.label}-synthesize` : 'synthesize',
    schema: opts.synthesizerSchema,
    timeout: opts.synthesizerTimeout || opts.timeout,
  });

  if (isFailedResult(synthesized)) {
    return {
      error: synthesized.error,
      stage: 'fanout_synthesis',
      partialResults: successes.map(item => item.value),
      failures,
    };
  }
  return failures.length > 0
    ? { output: synthesized, partialFailures: failures }
    : synthesized;
}

/**
 * verify(output, opts) — Adversarial Verification pattern.
 *
 * Subjects an output to independent adversarial review. One or more verifier
 * agents challenge the output, and a judge decides accept/reject/revise.
 *
 * @param {string|Object} output - The output to verify
 * @param {Object} opts - Options: { criteria, verifiers, judgeTool, maxRounds, onReject }
 * @returns {{ accepted: boolean, output: *, feedback: string, rounds: number }}
 */
function normalizeVerificationReview(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  if (typeof value.approve !== 'boolean' || !Array.isArray(value.issues)) return null;

  const requiredIssueKeys = ['description', 'evidence', 'severity'];
  const issues = [];
  for (const issue of value.issues) {
    if (!issue || typeof issue !== 'object' || Array.isArray(issue)) return null;
    const keys = Object.keys(issue).sort();
    if (keys.length !== requiredIssueKeys.length ||
        keys.some((key, index) => key !== requiredIssueKeys[index])) {
      return null;
    }
    const severity = typeof issue.severity === 'string' ? issue.severity.trim().toLowerCase() : '';
    const description = typeof issue.description === 'string' ? issue.description.trim() : '';
    const evidence = typeof issue.evidence === 'string' ? issue.evidence.trim() : '';
    if (!['critical', 'major', 'minor'].includes(severity) || !description || !evidence) {
      return null;
    }
    issues.push({ severity, description, evidence });
  }
  if (!value.approve && issues.length === 0) return null;
  return { approve: value.approve, issues };
}

async function verify(output, opts = {}) {
  if (cancelled) throw new CancelledError();

  const maxRounds = Math.min(Math.max(Number(opts.maxRounds) || 2, 1), 5);
  const criteria = opts.criteria || 'correctness, completeness, security, quality';
  const verifiers = (opts.verifiers && opts.verifiers.length
    ? opts.verifiers
    : [{ agentId: opts.verifierAgentId || opts.agentId, tool: opts.tool, role: 'adversarial_verifier', focus: 'Find concrete correctness and safety issues' }]
  ).slice(0, 8);

  let currentOutput = output;
  let round = 0;
  let lastFeedback = '';
  const verifierFailures = [];
  const reviewLedger = [];
  const evidenceLedger = [];

  while (round < maxRounds) {
    round++;
    const outputStr = typeof currentOutput === 'string' ? currentOutput : JSON.stringify(currentOutput);

    const reviews = await parallel(
      verifiers.map((v, idx) => ({
        prompt: `You are an adversarial verifier. Your job is to FIND PROBLEMS.\n\nCriteria: ${criteria}\nFocus: ${v.focus || criteria}\n\nOutput to verify:\n${outputStr}\n\nRules:\n- Only report REAL issues with concrete evidence\n- Rate severity: critical / major / minor\n- Be thorough but fair\n\nRespond with JSON:\n{ "issues": [{ "severity": "critical|major|minor", "description": "", "evidence": "" }], "approve": true/false }`,
        agentId: v.agentId || opts.verifierAgentId || opts.agentId,
        tool: v.tool,
        role: v.role || `verifier-${idx}`,
        label: `verify-r${round}-${idx}`,
        schema: {
          issues: [{ severity: '', description: '', evidence: '' }],
          approve: false,
        },
        timeout: v.timeout || opts.verifierTimeout || opts.timeout,
      }))
    );

    const allIssues = [];
    let approvals = 0;
    let validReviews = 0;
    for (const [index, r] of reviews.entries()) {
      if (isFailedResult(r)) {
        verifierFailures.push({ round, verifier: index, error: r.error, stage: r.stage || 'verify' });
        continue;
      }
      const normalized = normalizeVerificationReview(r);
      if (!normalized) {
        verifierFailures.push({
          round,
          verifier: index,
          error: 'Verifier returned an invalid review shape or empty evidence',
          stage: 'verify_schema',
        });
        continue;
      }
      const verifier = verifiers[index] || {};
      const verifierName = verifier.role || verifier.tool || `verifier-${index}`;
      validReviews++;
      if (normalized.approve) approvals++;
      allIssues.push(...normalized.issues);
      reviewLedger.push({
        round,
        verifier: verifierName,
        approve: normalized.approve,
        issues: normalized.issues,
      });
      for (const issue of normalized.issues) {
        evidenceLedger.push({ round, verifier: verifierName, ...issue });
      }
    }

    // Verification is a barrier: no valid evidence can never mean approval.
    if (validReviews === 0) {
      lastFeedback = 'All verifiers failed to produce a valid review';
      break;
    }

    const criticals = allIssues.filter(i => i.severity === 'critical').length;
    const majors = allIssues.filter(i => i.severity === 'major').length;
    if (approvals === validReviews && criticals === 0 && majors === 0) {
      return {
        accepted: true,
        output: currentOutput,
        feedback: '',
        rounds: round,
        failures: verifierFailures,
        reviews: reviewLedger,
        evidence: evidenceLedger,
      };
    }

    lastFeedback = allIssues
      .filter(i => i.severity !== 'minor')
      .map(i => `[${i.severity}] ${i.description}\nEvidence: ${i.evidence}`)
      .join('\n');

    if (round >= maxRounds) break;

    if (typeof opts.onReject === 'function') {
      currentOutput = await opts.onReject(currentOutput, lastFeedback, round);
    } else {
      const revised = await agent(
        `Revise this output to address the following issues:\n\nCurrent output:\n${outputStr}\n\nIssues found:\n${lastFeedback}\n\nProvide a revised version that addresses ALL critical and major issues.`,
        {
          agentId: opts.reviserAgentId || opts.agentId || verifiers[0]?.agentId,
          tool: opts.reviseTool || opts.tool,
          role: 'reviser',
          label: `revise-r${round}`,
          timeout: opts.reviseTimeout || opts.timeout,
        }
      );
      if (isFailedResult(revised)) {
        verifierFailures.push({ round, error: revised.error, stage: 'revision' });
        lastFeedback = `${lastFeedback}\nRevision failed: ${revised.error}`.trim();
        break;
      }
      currentOutput = revised;
    }
  }

  return {
    accepted: false,
    output: currentOutput,
    feedback: lastFeedback,
    rounds: round,
    failures: verifierFailures,
    reviews: reviewLedger,
    evidence: evidenceLedger,
  };
}

/**
 * generate(count, generatorFn, filterFn, opts) — Generate-and-Filter pattern.
 *
 * Generates N candidates via parallel agent calls, then filters and ranks them.
 * Returns the top-K results after deduplication.
 *
 * @param {number} count - Number of candidates to generate
 * @param {Function|Object} generatorFn - Generator: (index) => agent-descriptor or prompt
 * @param {Function|Object} filterFn - Filter/ranker: receives all candidates, returns ranked subset
 * @param {Object} opts - Options: { topK, dedup, filterTool }
 * @returns {Array} Filtered and ranked results
 */
async function generate(count, generatorFn, filterFn, opts = {}) {
  if (cancelled) throw new CancelledError();
  if (typeof count !== 'number' || count < 1) {
    throw new TypeError('generate() expects a positive number as first argument');
  }
  if (count > 50) {
    throw new RangeError('generate() count must be <= 50');
  }
  const reservedCalls = typeof filterFn === 'function' ? 0 : 1;
  if (count + reservedCalls > maxTotalAgents - totalAgentCalls) {
    throw new RangeError('generate() exceeds remaining agent call capacity');
  }

  const tasks = [];
  for (let i = 0; i < count; i++) {
    if (typeof generatorFn === 'function') {
      const descriptor = generatorFn(i);
      if (typeof descriptor === 'string') {
        tasks.push({ prompt: descriptor, agentId: opts.generatorAgentId || opts.agentId, label: `gen-${i}`, timeout: opts.generatorTimeout || opts.timeout });
      } else if (typeof descriptor === 'object' && descriptor !== null) {
        tasks.push({ ...descriptor, agentId: descriptor.agentId || opts.generatorAgentId || opts.agentId, label: descriptor.label || `gen-${i}`, timeout: descriptor.timeout || opts.generatorTimeout || opts.timeout });
      } else {
        throw new TypeError(`generate() generatorFn must return a string or object, got ${typeof descriptor}`);
      }
    } else if (typeof generatorFn === 'object' && generatorFn !== null && generatorFn.prompt) {
      tasks.push({ ...generatorFn, agentId: generatorFn.agentId || opts.generatorAgentId || opts.agentId, label: `gen-${i}`, timeout: generatorFn.timeout || opts.generatorTimeout || opts.timeout });
    } else {
      throw new TypeError('generate() generatorFn must be a function or object with .prompt');
    }
  }

  const candidates = await parallel(tasks);
  const { successes, failures } = partitionResults(candidates);
  const successfulCandidates = successes.map(item => item.value);
  if (successfulCandidates.length === 0) {
    return { candidates: [], failures, error: 'generate(): all candidates failed' };
  }

  if (typeof filterFn === 'function') {
    return { candidates: await filterFn(successfulCandidates), failures };
  }

  const topK = Math.min(Math.max(Number(opts.topK) || 3, 1), successfulCandidates.length);
  const candidatesSummary = successfulCandidates
    .map((c, i) => `[Candidate ${i}]: ${typeof c === 'string' ? c.slice(0, 500) : JSON.stringify(c).slice(0, 500)}`)
    .join('\n\n');

  const filterPrompt = typeof filterFn === 'string' ? filterFn :
    `You are a quality filter. From the following ${count} candidates, select the top ${topK} best ones.\n\nCriteria: ${opts.criteria || 'quality, originality, correctness'}\n\nCandidates:\n${candidatesSummary}\n\nRespond with JSON: { "ranked": [0, 2, 1], "reasoning": "..." } where "ranked" is the candidate indices in order of quality.`;

  const filterResult = await agent(filterPrompt, {
    agentId: opts.filterAgentId || opts.agentId || tasks[0]?.agentId,
    tool: opts.filterTool || opts.tool,
    role: 'filter',
    label: 'filter-rank',
    schema: { ranked: [], reasoning: '' },
    timeout: opts.filterTimeout || opts.timeout,
  });

  if (filterResult && filterResult.ranked) {
    const seen = new Set();
    const ranked = filterResult.ranked
      .filter(idx => typeof idx === 'number' && idx >= 0 && idx < successfulCandidates.length && !seen.has(idx) && seen.add(idx))
      .slice(0, topK)
      .map(idx => successfulCandidates[idx]);
    return {
      candidates: ranked.length > 0 ? ranked : successfulCandidates.slice(0, topK),
      failures,
    };
  }

  if (isFailedResult(filterResult)) {
    failures.push({ index: -1, error: filterResult.error, stage: 'filter' });
  }
  return { candidates: successfulCandidates.slice(0, topK), failures };
}

/**
 * tournament(contestants, judgeFn, opts) — Tournament pattern.
 *
 * Runs multiple agents on the same task, then uses pairwise elimination
 * judging to determine the best result.
 *
 * @param {Array} contestants - Array of agent descriptors (each solves the same task differently)
 * @param {Function|Object} judgeFn - Judge: (a, b) => 'a' | 'b' (or agent descriptor for judging)
 * @param {Object} opts - Options: { judgeTool, task, bracket }
 * @returns {{ winner: *, bracket: Array, rounds: number }}
 */
async function tournament(contestants, judgeFn, opts = {}) {
  if (cancelled) throw new CancelledError();
  if (!Array.isArray(contestants) || contestants.length < 2) {
    throw new TypeError('tournament() requires at least 2 contestants');
  }

  const solutions = await parallel(
    contestants.map((c, idx) => {
      if (typeof c === 'function') return c;
      if (typeof c === 'object' && c.prompt) return { ...c, label: c.label || `contestant-${idx}`, timeout: c.timeout || opts.contestantTimeout || opts.timeout };
      throw new TypeError('tournament() contestants must be functions or agent descriptors');
    })
  );

  const partitioned = partitionResults(solutions);
  let remaining = partitioned.successes.map(({ index, value }) => ({
    solution: value,
    index,
    label: contestants[index]?.label || `contestant-${index}`,
  }));
  const bracket = [];
  const unresolved = [];
  let roundNum = 0;

  if (remaining.length === 0) {
    return {
      winner: null,
      bracket,
      rounds: 0,
      failures: partitioned.failures,
      error: 'tournament(): all contestants failed',
    };
  }

  while (remaining.length > 1) {
    if (cancelled) throw new CancelledError();
    roundNum++;
    const nextRound = [];
    const matchups = [];

    for (let i = 0; i < remaining.length; i += 2) {
      if (i + 1 >= remaining.length) {
        nextRound.push(remaining[i]);
        continue;
      }
      matchups.push([remaining[i], remaining[i + 1]]);
    }

    const judgeResults = await parallel(
      matchups.map(([a, b], mIdx) => {
        const aStr = typeof a.solution === 'string' ? a.solution.slice(0, 1500) : JSON.stringify(a.solution).slice(0, 1500);
        const bStr = typeof b.solution === 'string' ? b.solution.slice(0, 1500) : JSON.stringify(b.solution).slice(0, 1500);

        if (typeof judgeFn === 'function') {
          return () => judgeFn(a.solution, b.solution, a.label, b.label);
        }

        const task = opts.task || 'the given task';
        const criteria = opts.criteria || 'correctness, quality, completeness';
        return {
          prompt: `You are a judge in a tournament. Compare these two solutions for ${task}.\n\nCriteria: ${criteria}\n\n[Solution A - ${a.label}]:\n${aStr}\n\n[Solution B - ${b.label}]:\n${bStr}\n\nWhich is better? Respond with JSON: { "winner": "A" or "B", "reasoning": "brief explanation" }`,
          agentId: opts.judgeAgentId || opts.agentId || contestants[0]?.agentId,
          tool: opts.judgeTool || opts.tool || 'claude',
          role: 'tournament_judge',
          label: `judge-r${roundNum}-m${mIdx}`,
          schema: { winner: '', reasoning: '' },
          timeout: opts.judgeTimeout || opts.timeout,
        };
      })
    );

    for (let i = 0; i < matchups.length; i++) {
      const [a, b] = matchups[i];
      const result = judgeResults[i];
      let winnerIsA = null;
      if (typeof result === 'string') {
        const trimmed = result.trim();
        const firstLine = trimmed.split(/[.\n]/)[0].trim();
        const hasA = /\bA\b/.test(firstLine);
        const hasB = /\bB\b/.test(firstLine);
        if (hasA && !hasB) winnerIsA = true;
        else if (!hasA && hasB) winnerIsA = false;
        else winnerIsA = null;
      } else {
        const winner = result && typeof result.winner === 'string'
          ? result.winner.toUpperCase()
          : '';
        if (winner === 'A') winnerIsA = true;
        if (winner === 'B') winnerIsA = false;
      }
      if (winnerIsA === null) {
        unresolved.push({ round: roundNum, a: a.label, b: b.label, error: 'judge returned no valid winner' });
        // Continue deterministically only to drain the bracket; do not expose
        // this provisional choice as a verified winner below.
        winnerIsA = (roundNum + i) % 2 === 0;
      }
      const winner = winnerIsA ? a : b;
      const loser = winnerIsA ? b : a;
      bracket.push({ round: roundNum, winner: winner.label, loser: loser.label, reasoning: result?.reasoning });
      nextRound.push(winner);
    }

    remaining = nextRound;
  }

  return {
    winner: unresolved.length > 0 ? null : remaining[0].solution,
    winnerLabel: unresolved.length > 0 ? null : remaining[0].label,
    provisionalWinner: unresolved.length > 0 ? remaining[0].solution : null,
    bracket,
    rounds: roundNum,
    failures: partitioned.failures,
    unresolved,
    ...(unresolved.length > 0 && { error: 'tournament(): one or more judge decisions were invalid' }),
  };
}

/**
 * loop(taskFn, opts) — Loop-Until-Done pattern.
 *
 * Iteratively runs a task function until a stop condition is met.
 * Supports convergence detection (stop when no new findings), explicit
 * stop conditions, and maximum iteration limits.
 *
 * @param {Function} taskFn - (iteration, previousResult, allResults) => result
 * @param {Object} opts - Options: { maxIterations, stopWhen, convergenceCheck, onIteration }
 * @returns {{ results: Array, iterations: number, stoppedBy: string }}
 */
async function loop(taskFn, opts = {}) {
  if (cancelled) throw new CancelledError();
  if (typeof taskFn !== 'function') {
    throw new TypeError('loop() expects a function as first argument');
  }

  const MAX_LOOP_ITERATIONS = 50;
  const maxIterations = Math.min(Math.max(Number(opts.maxIterations) || 5, 1), MAX_LOOP_ITERATIONS);
  const results = [];
  let previousResult = null;
  let stoppedBy = 'max_iterations';

  for (let i = 0; i < maxIterations; i++) {
    if (cancelled) throw new CancelledError();

    const result = await taskFn(i, previousResult, results);
    results.push(result);

    if (isFailedResult(result)) {
      stoppedBy = 'error';
      break;
    }

    if (typeof opts.onIteration === 'function') {
      opts.onIteration(i, result);
    }

    if (typeof opts.stopWhen === 'function') {
      const shouldStop = await opts.stopWhen(result, i, results);
      if (shouldStop) {
        stoppedBy = 'stop_condition';
        break;
      }
    }

    if (typeof opts.convergenceCheck === 'function') {
      const converged = await opts.convergenceCheck(result, previousResult, results);
      if (converged) {
        stoppedBy = 'converged';
        break;
      }
    } else if (opts.convergence !== false && i > 0 && previousResult !== null) {
      const prevStr = typeof previousResult === 'string' ? previousResult : JSON.stringify(previousResult);
      const currStr = typeof result === 'string' ? result : JSON.stringify(result);
      if (prevStr === currStr) {
        stoppedBy = 'converged';
        break;
      }
    }

    previousResult = result;
  }

  return {
    results,
    iterations: results.length,
    stoppedBy,
    ...(stoppedBy === 'error' && { error: results[results.length - 1].error }),
  };
}

/**
 * sequence(steps) — Sequential execution helper.
 *
 * Runs steps strictly one after another, passing each result to the next.
 * Unlike pipeline() which runs items concurrently, sequence() is serial.
 *
 * @param {Array} steps - Array of functions: (previousResult) => result
 * @returns {*} Final step's result
 */
async function sequence(steps, opts = {}) {
  if (cancelled) throw new CancelledError();
  if (!Array.isArray(steps)) {
    throw new TypeError('sequence() expects an array of steps');
  }

  let result = undefined;
  for (let i = 0; i < steps.length; i++) {
    if (cancelled) throw new CancelledError();
    const step = steps[i];
    try {
      if (typeof step === 'function') {
        result = await step(result);
      } else if (typeof step === 'object' && step !== null && step.prompt) {
        result = await agent(step);
      } else {
        throw new TypeError(`sequence() step ${i} must be a function or agent descriptor`);
      }
    } catch (err) {
      if (err instanceof CancelledError) throw err;
      if (opts.throwOnFailure) throw err;
      return { error: sanitizePath(err.message || String(err)), failedAtStep: i };
    }
    if (isFailedResult(result) && !opts.continueOnFailure) {
      return { ...result, failedAtStep: i };
    }
  }
  return result;
}

/**
 * race(contestants, opts) — First-to-finish race.
 *
 * Runs multiple agents concurrently but returns as soon as the first
 * valid result arrives (like Promise.race but with validation).
 *
 * @param {Array} contestants - Array of agent descriptors or functions
 * @param {Object} opts - Options: { validate, timeout }
 * @returns {*} First valid result
 */
function abortRequest(request_id) {
  const entry = pendingRequests.get(request_id);
  if (!entry || entry.aborted) return;
  clearPendingTimer(entry);
  entry.aborted = true;
  entry.reject(new CancelledError());
  pendingRequests.delete(request_id);
  sendNotification('abort_request', { request_id });
}

async function race(contestants, opts = {}) {
  if (!Array.isArray(contestants) || contestants.length === 0) {
    throw new TypeError('race() requires a non-empty array');
  }

  const requestedTimeout = Number(opts.timeout);
  const normalizedContestants = contestants.map((contestant) => {
    if (contestant && typeof contestant === 'object' && !Array.isArray(contestant)) {
      return {
        ...contestant,
        timeout: contestant.timeout || (requestedTimeout > 0 ? requestedTimeout : undefined),
      };
    }
    return contestant;
  });
  if (!Number.isFinite(requestedTimeout) || requestedTimeout <= 0) {
    return raceCore(normalizedContestants, opts);
  }

  const tracker = { ids: new Set(), labels: new Map() };
  const timeoutMessage = `race() timed out after ${requestedTimeout}s`;
  const scope = { cancelled: false, timeoutMessage, failures: [] };
  const remainingMs = remainingWorkflowMs();
  const requestedMs = Math.max(1, Math.ceil(requestedTimeout * 1000));
  const boundedTimeoutMs = Number.isFinite(remainingMs)
    ? Math.max(1, Math.min(requestedMs, remainingMs))
    : requestedMs;
  const timeoutMs = Math.min(MAX_SAFE_TIMER_MS, boundedTimeoutMs);
  const timeoutSentinel = Symbol('race-timeout');
  let timer = null;

  try {
    const racePromise = withRequestCollector(
      tracker,
      () => Promise.resolve().then(() => raceCore(normalizedContestants, opts, scope)),
    );
    const deadlinePromise = new Promise((resolveDeadline) => {
      timer = setTimeout(() => resolveDeadline(timeoutSentinel), timeoutMs);
    });
    const outcome = await Promise.race([racePromise, deadlinePromise]);
    if (outcome !== timeoutSentinel) return outcome;

    scope.cancelled = true;
    const timeoutFailures = [];
    for (const requestId of tracker.ids) {
      const entry = pendingRequests.get(requestId);
      if (!entry || entry.aborted) continue;
      const label = tracker.labels.get(requestId) || `race-request-${requestId}`;
      clearPendingTimer(entry);
      entry.aborted = true;
      pendingRequests.delete(requestId);
      sendNotification('abort_request', { request_id: requestId });
      sendNotification('agent_aborted', {
        label,
        reason: 'race_timeout',
        request_id: requestId,
      });
      entry.reject(new JsonRpcError({ code: -32002, message: timeoutMessage }));
      timeoutFailures.push({
        request_id: requestId,
        label,
        error: timeoutMessage,
        stage: 'race_timeout',
      });
    }
    return {
      error: timeoutMessage,
      stage: 'race_timeout',
      timeout: true,
      failures: [...scope.failures, ...timeoutFailures],
    };
  } finally {
    if (timer !== null) clearTimeout(timer);
  }
}

async function raceCore(contestants, opts = {}, scope = null) {
  if (cancelled) throw new CancelledError();
  if (!Array.isArray(contestants) || contestants.length === 0) {
    throw new TypeError('race() requires a non-empty array');
  }

  const validate = opts.validate || ((r) => r != null && r !== '' && !r.error);

  const raceRequestIds = new Set();
  const requestLabels = new Map();
  const collector = { ids: raceRequestIds, labels: requestLabels };

  return withRequestCollector(collector, () => new Promise((resolve, reject) => {
    let settled = false;
    let completed = 0;
    const failures = scope ? scope.failures : [];

    // Track all agent_call request IDs created by this race, plus their
    // labels, so we can abort losers and report which agent was aborted.
    function abortLosers() {
      for (const rid of raceRequestIds) {
        const entry = pendingRequests.get(rid);
        if (entry && !entry.aborted) {
          const label = requestLabels.get(rid) || `race-contestant-${rid}`;
          abortRequest(rid);
          sendNotification('agent_aborted', {
            label,
            reason: 'race loser',
            request_id: rid,
          });
        }
      }
    }

    function finish(fn, arg) {
      if (settled) return;
      settled = true;
      fn(arg);
    }

    contestants.forEach((c, idx) => {
      const invoke = () => (typeof c === 'function' ? c() : agent(c));
      const p = scope ? raceContext.run(scope, invoke) : invoke();
      Promise.resolve(p).then(result => {
        if (settled) return;
        completed++;
        if (validate(result)) {
          abortLosers();
          finish(resolve, result);
        } else if (completed === contestants.length) {
          failures.push({ index: idx, error: isFailedResult(result) ? result.error : 'validation rejected result' });
          finish(resolve, { error: 'race(): no contestant produced a valid result', failures });
        } else {
          failures.push({ index: idx, error: isFailedResult(result) ? result.error : 'validation rejected result' });
        }
      }).catch(err => {
        if (settled) return;
        if (err instanceof CancelledError) {
          // A winner settles before loser aborts are observed. Reaching this
          // branch while unsettled therefore means host cancellation.
          finish(reject, err);
          return;
        }
        completed++;
        failures.push({ index: idx, error: sanitizePath(err.message || String(err)) });
        if (completed === contestants.length) {
          finish(resolve, { error: `race(): all ${contestants.length} contestants failed`, failures });
        }
      });
    });
  }));
}

// ---------------------------------------------------------------------------
// End of Dynamic Workflow Patterns
// ---------------------------------------------------------------------------

function phase(title) {
  sendNotification('phase', { title });
}

function log(msg) {
  const milestone = String(msg).replace(/\s+/g, ' ').trim();
  const bounded = milestone.length > 240 ? `${milestone.slice(0, 237)}...` : milestone;
  sendNotification('log', { message: bounded });
}

// ---------------------------------------------------------------------------
// Expose globals
// ---------------------------------------------------------------------------

function installGlobals() {
  globalThis.agent = agent;
  globalThis.parallel = parallel;
  globalThis.pipeline = pipeline;
  globalThis.phase = phase;
  globalThis.log = log;
  globalThis.classify = classify;
  globalThis.fanout = fanout;
  globalThis.verify = verify;
  globalThis.generate = generate;
  globalThis.tournament = tournament;
  globalThis.loop = loop;
  globalThis.sequence = sequence;
  globalThis.race = race;
}

// Wrap every function we hand into the vm sandbox so that attacker code
// cannot walk through ``fn.constructor.constructor(...)`` to reach the host
// Function constructor. Each wrapper is a plain closure with ``null``
// prototype; we additionally freeze the wrapper so its properties are
// immutable from within the sandbox.
function sandboxWrapHostFn(fn) {
  // eslint-disable-next-line func-names
  const wrapper = function (...args) { return fn(...args); };
  Object.setPrototypeOf(wrapper, null);
  Object.freeze(wrapper);
  return wrapper;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const scriptPath = process.argv[2];
  if (!scriptPath) {
    process.stderr.write('[runtime] Usage: node runtime.js <script-path>\n');
    flushAndExit(1);
    return;
  }

  // Setup readline for incoming NDJSON
  const rl = createInterface({ input: process.stdin, terminal: false });
  rl.on('line', handleMessage);

  // Signal ready
  sendNotification('ready', {});

  // Wait for init from Python host
  const initParams = await waitForInit();
  maxConcurrent = Number(initParams.max_concurrent) || 0;
  if (maxConcurrent < 0) maxConcurrent = 0;
  const hostAgentLimit = Number(initParams.max_total_agents);
  if (Number.isInteger(hostAgentLimit) && hostAgentLimit > 0) {
    maxTotalAgents = Math.min(DEFAULT_MAX_TOTAL_AGENTS, hostAgentLimit);
  }
  workflowStartedMs = Number(initParams.started_unix_ms) || Date.now();
  workflowDeadlineMs = Number(initParams.deadline_unix_ms) || 0;
  workflowTotalTimeoutMs = Number(initParams.total_timeout_s) > 0
    ? Number(initParams.total_timeout_s) * 1000
    : 0;
  // Per-agent timeout floor (seconds → ms); 0 means unlimited per-agent.
  workflowAgentCallTimeoutMs = Number(initParams.agent_call_timeout_s) > 0
    ? Number(initParams.agent_call_timeout_s) * 1000
    : 0;
  if (!workflowDeadlineMs && workflowTotalTimeoutMs > 0) {
    workflowDeadlineMs = workflowStartedMs + workflowTotalTimeoutMs;
  }
  const workflowArgs = initParams.args || {};

  // Read user script source
  const absolutePath = resolve(scriptPath);
  let source;
  try {
    source = readFileSync(absolutePath, 'utf-8');
  } catch (err) {
    sendNotification('error', {
      message: sanitizePath(`Failed to read script: ${err.message}`),
      stack: '',
    });
    flushAndExit(1);
    return;
  }

  // Create sandboxed context with only orchestration primitives.
  // Every host-side function is wrapped in a sandboxed closure before being
  // handed to the vm context so script code cannot reach
  // ``agent.constructor.constructor("return process")()`` or similar
  // prototype-chain escapes.
  const sandboxGlobals = {
    // Core primitives
    agent: sandboxWrapHostFn(agent),
    parallel: sandboxWrapHostFn(parallel),
    pipeline: sandboxWrapHostFn(pipeline),
    phase: sandboxWrapHostFn(phase),
    log: sandboxWrapHostFn(log),
    // Dynamic Workflow pattern primitives
    classify: sandboxWrapHostFn(classify),
    fanout: sandboxWrapHostFn(fanout),
    verify: sandboxWrapHostFn(verify),
    generate: sandboxWrapHostFn(generate),
    tournament: sandboxWrapHostFn(tournament),
    loop: sandboxWrapHostFn(loop),
    sequence: sandboxWrapHostFn(sequence),
    race: sandboxWrapHostFn(race),
    CancelledError: sandboxWrapHostFn(() => { throw new CancelledError(); }),
    workflowArgs,
    // Safe standard built-ins
    console: { log: sandboxWrapHostFn((...a) => debugLog(a.join(' '))),
                error: sandboxWrapHostFn((...a) => debugLog(a.join(' '))) },
    setTimeout: sandboxWrapHostFn((fn, ms) => setTimeout(fn, ms)),
    clearTimeout: sandboxWrapHostFn((id) => clearTimeout(id)),
    Promise,
    JSON,
    Array,
    Object,
    String,
    Number,
    Boolean,
    Map,
    Set,
    Date,
    Math,
    Error,
    TypeError,
    RangeError,
    RegExp,
    Symbol,
    parseInt,
    parseFloat,
    isNaN,
    isFinite,
    encodeURIComponent,
    decodeURIComponent,
    encodeURI,
    decodeURI,
    undefined,
    NaN,
    Infinity,
  };

  const context = vm.createContext(sandboxGlobals, {
    name: 'workflow-sandbox',
    codeGeneration: { strings: false, wasm: false },
  });

  // Harden the prototypes inside the sandbox so attacker code cannot patch
  // ``Object.prototype`` / ``Function.prototype`` with getters/setters that
  // leak back into the host. These calls only affect objects inside the
  // vm context; the host side is untouched.
  try {
    vm.runInContext('Object.freeze(Object.prototype);' +
                    'Object.freeze(Function.prototype);' +
                    'Object.freeze(Array.prototype);' +
                    'Object.freeze(String.prototype);' +
                    'Object.freeze(Number.prototype);' +
                    'Object.freeze(Boolean.prototype);' +
                    'Object.freeze(Promise.prototype);', context, {
      filename: 'sandbox-harden.js',
    });
  } catch (err) {
    sendNotification('error', {
      message: sanitizePath(`Sandbox hardening failed — aborting: ${err.message}`),
      stack: '',
    });
    flushAndExit(1);
    return;
  }

  // Load and execute using vm.SourceTextModule (sandboxed ESM)
  let mod;
  try {
    mod = new vm.SourceTextModule(source, {
      context,
      identifier: `file://${absolutePath}`,
    });

    // Link: deny all imports (sandbox enforcement)
    await mod.link((specifier) => {
      throw new Error(`Imports are not allowed in workflow scripts: "${specifier}"`);
    });

    await mod.evaluate();
  } catch (err) {
    sendNotification('error', {
      message: sanitizePath(`Failed to load script: ${err.message}`),
      stack: '',
    });
    flushAndExit(1);
    return;
  }

  // Extract exports from the module namespace
  const ns = mod.namespace;

  // Validate meta
  const meta = ns.meta;
  try {
    validateMeta(meta);
  } catch (err) {
    sendNotification('error', {
      message: sanitizePath(`Meta validation failed: ${err.message}`),
      stack: '',
    });
    flushAndExit(1);
    return;
  }

  // Execute workflow
  try {
    let result;
    if (typeof ns.default === 'function') {
      result = await ns.default(workflowArgs);
    } else if (typeof ns.run === 'function') {
      result = await ns.run(workflowArgs);
    } else {
      // Script has no explicit entry — meta is the payload
      result = { meta };
    }

    sendNotification('done', { result: result ?? null });
  } catch (err) {
    if (err instanceof CancelledError) {
      sendNotification('error', {
        message: sanitizePath(err.message),
        stack: '',
      });
    } else {
      sendNotification('error', {
        message: sanitizePath(err.message || String(err)),
        stack: '',
      });
    }
    flushAndExit(1);
    return;
  }

  // Clean shutdown — flush the `done` frame before exiting so the host never
  // sees a bare stdout EOF in place of the real result.
  rl.close();
  flushAndExit(0);
}

// ---------------------------------------------------------------------------
// Unhandled rejection safety net
// ---------------------------------------------------------------------------

process.on('unhandledRejection', (reason) => {
  const message = reason instanceof Error ? reason.message : String(reason);
  sendNotification('error', { message: sanitizePath(message), stack: '' });
  flushAndExit(1);
});

process.on('uncaughtException', (err) => {
  sendNotification('error', { message: sanitizePath(err.message), stack: '' });
  flushAndExit(1);
});

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

main();
