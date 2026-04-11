/**
 * Hub API Server — lightweight HTTP agent execution API for Alacrity Hub.
 *
 * Serves on port 4100 (configurable via HUB_API_PORT).
 * Accepts POST /api/agent/run with system prompt, tools, and Portkey config.
 * Runs an OpenAI-compatible tool-use agent loop through Portkey → Ollama.
 * Posts tool callbacks to the hub's AgentService for execution.
 *
 * This is separate from NanoClaw's main messaging bot process.
 * Start with: node dist/hub-api.js
 */

import { createServer, IncomingMessage, ServerResponse } from 'http';
import {
  existsSync,
  mkdirSync,
  writeFileSync,
  readFileSync,
  readdirSync,
  unlinkSync,
} from 'fs';
import { join, dirname } from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';
import { resolve, relative } from 'path';
import {
  buildImportGraph,
  computeBlastRadius,
} from '@alacrity/tools/import-graph';
import {
  buildDocGraph,
  queryDocDeps,
  computeDocImpact,
  queryDocContext,
} from '@alacrity/tools/doc-graph';
import {
  queryVaultGraph,
  getVaultStats,
  invalidateVaultCache,
  type VaultQueryParams,
} from '@alacrity/tools/vault-graph';
import {
  validateAssessmentBlock,
  extractAssessmentBlock,
  validatePreEditConfirmation,
  validateShipAudit,
  type ShipAuditState,
  type GitState,
} from '@alacrity/tools/validation';
import {
  parseLifecycle,
  inferPhase,
  type ArtifactState,
} from '@alacrity/tools/lifecycle-parser';
import { runGit } from './git-helpers.js';

const execAsync = promisify(exec);

// Load .env file if present (launchd doesn't source it)
// Resolve from compiled dist/ back to repo root
const envPath = join(import.meta.dirname ?? __dirname, '..', '.env');
if (existsSync(envPath)) {
  for (const line of readFileSync(envPath, 'utf-8').split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eqIdx = trimmed.indexOf('=');
    if (eqIdx < 0) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    const val = trimmed.slice(eqIdx + 1).trim();
    if (!process.env[key]) process.env[key] = val;
  }
}

const PORT = parseInt(process.env.HUB_API_PORT || '4100', 10);
const HOST = process.env.HUB_API_HOST || '127.0.0.1';

// Identity stamp for this NanoClaw process instance. Reported on /api/health
// so the hub can notice when NanoClaw has restarted (crash, reboot, launchd
// kickstart) and reclaim any stage_checkpoints rows that were marked
// in_progress by a prior, now-dead instance. See checkpoint-layer-v2 design.
const BOOT_EPOCH = Date.now();

const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://127.0.0.1:11434';
const PORTKEY_BASE_URL =
  process.env.PORTKEY_BASE_URL || 'http://127.0.0.1:8787';
const PORTKEY_API_KEY = process.env.PORTKEY_API_KEY || '';
const CHROMADB_HOST = process.env.CHROMADB_HOST || 'http://127.0.0.1:8000';
const EMBED_MODEL = 'nomic-embed-text';
const MAX_TOOL_ROUNDS = 10;
const MAX_CONCURRENT_AGENTS = 2;

// --- Concurrency Semaphore ---
// Limits concurrent runAsyncAgent calls to prevent D1 proxy fetch failures
class AgentSemaphore {
  private running = 0;
  private queue: Array<() => void> = [];

  async acquire(): Promise<void> {
    if (this.running < MAX_CONCURRENT_AGENTS) {
      this.running++;
      return;
    }
    return new Promise<void>((resolve) => {
      this.queue.push(() => {
        this.running++;
        resolve();
      });
    });
  }

  release(): void {
    this.running--;
    const next = this.queue.shift();
    if (next) next();
  }

  get active(): number {
    return this.running;
  }

  get queued(): number {
    return this.queue.length;
  }
}

const agentSemaphore = new AgentSemaphore();

// CF Access validation for cloud-first async endpoint
const CF_ACCESS_EXPECTED_ID = process.env.CF_ACCESS_EXPECTED_ID || '';
const CF_ACCESS_EXPECTED_SECRET = process.env.CF_ACCESS_EXPECTED_SECRET || '';
const AGENT_RESULTS_SECRET = process.env.AGENT_RESULTS_SECRET || '';
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || '';
const HUB_D1_PROXY_URL = process.env.HUB_D1_PROXY_URL || '';

// Path to the agents package on this machine
const AGENTS_BASE = join(
  process.env.HOME || '/root',
  'Vibe Sphere',
  'alacrity_hub',
  'packages',
  'agents',
);

// Alacrity Hub repo root — used by orientation tools (blast-radius, dep-graph, doc-graph)
const ALACRITY_HUB_ROOT = join(
  process.env.HOME || '/root',
  'Vibe Sphere',
  'alacrity_hub',
);

// Lazy D1 adapter for session routes (not dispatched by hub, so needs its own proxy ref)
const agentsPkgSrc = join(ALACRITY_HUB_ROOT, 'packages', 'agents', 'src');
let sessionD1: { prepare: (sql: string) => unknown } | null = null;
async function getSessionD1() {
  if (sessionD1) return sessionD1;
  if (!HUB_D1_PROXY_URL || !AGENT_RESULTS_SECRET) {
    throw new Error(
      'HUB_D1_PROXY_URL and AGENT_RESULTS_SECRET required for session routes',
    );
  }
  const { createD1ProxyAdapter } = await import(
    join(agentsPkgSrc, 'd1-proxy-adapter.ts')
  );
  sessionD1 = createD1ProxyAdapter(
    HUB_D1_PROXY_URL,
    AGENT_RESULTS_SECRET,
  ) as typeof sessionD1;
  return sessionD1!;
}

// Vault paths
const AGENT_VAULT_BASE = join(
  process.env.HOME || '/root',
  'Vaults',
  'AlacrityHub',
  'agent',
);
const QUARANTINE_DIR = join(AGENT_VAULT_BASE, 'quarantine');
const HUMAN_VAULT_PATH =
  process.env.HUMAN_VAULT_PATH ||
  join(process.env.HOME || '/root', 'Vaults', 'HumanVault');

function writeVaultFile(
  folder: string,
  filename: string,
  frontmatter: Record<string, string>,
  body: string,
): void {
  const dir = join(AGENT_VAULT_BASE, folder);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  const filePath = join(dir, filename);

  const fm = Object.entries(frontmatter)
    .map(([k, v]) => `${k}: ${v}`)
    .join('\n');
  const content = `---\n${fm}\n---\n\n${body}\n`;

  writeFileSync(filePath, content, 'utf-8');
  invalidateVaultCache();
  log(`vault-write: ${folder}/${filename}`);
}

/**
 * Virtual key → Ollama model mapping.
 * Mirrors config/portkey.yaml from the hub.
 * If the model is not a known virtual key, it's passed through as-is.
 */
const VIRTUAL_KEY_MODELS: Record<string, string> = {
  'local-reasoning': 'qwen2.5:14b-instruct-q4_K_M',
  'local-coding': 'qwen2.5-coder:7b-instruct-q4_K_M',
  'local-background': 'llama3.2:3b-instruct-q4_K_M',
};

function resolveModel(modelKeyOrName: string): string {
  return VIRTUAL_KEY_MODELS[modelKeyOrName] ?? modelKeyOrName;
}

// --- Types ---

interface PortkeyConfig {
  base_url: string;
  api_key: string;
  virtual_key: string;
}

interface ToolDefinition {
  name: string;
  description: string;
  parameters?: Record<string, unknown>;
}

/**
 * Checkpoint context passed through the agent loop when the hub dispatches a
 * single checkpoint (checkpoint-layer v2). Presence of this field is what
 * distinguishes a checkpoint dispatch from a plain whole-stage dispatch.
 *
 * When set, the intercepted `checkpoint_complete` handler uses it to:
 *   1. Commit the worktree via runGit
 *   2. POST the completion back to the hub with handoff metadata
 *   3. Signal the agent loop to exit (one dispatch = one checkpoint)
 */
interface CheckpointContext {
  missionId: string;
  stage: string;
  checkpointIndex: number;
  worktreePath: string;
  resultsUrl: string;
  resultsAuth: string;
}

interface AgentRunRequest {
  model: string;
  system_prompt: string;
  user_message: string;
  tools: ToolDefinition[];
  max_tokens: number;
  portkey: PortkeyConfig;
  container?: Record<string, unknown>;
  tool_callback_url: string;
  cloud?: { forceCloud: boolean; fallbackModel: string; apiKey: string };
  /** Present iff this dispatch is a single-checkpoint run (v2). */
  checkpointContext?: CheckpointContext;
}

interface AgentRunResponse {
  content: string;
  model: string;
  tokensUsed: number;
  duration: number;
  /**
   * Set to `true` by runAgentLoop when the loop exited because the agent
   * called `checkpoint_complete`. Signals to the caller (runAsyncAgent) that
   * the intercepted handler already POSTed results back to the hub, so the
   * normal post-loop results callback must be skipped to avoid duplicates.
   */
  checkpointCompleted?: boolean;
}

interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content?: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

interface ChatCompletionResponse {
  choices: Array<{
    message: {
      role: string;
      content: string | null;
      tool_calls?: ToolCall[];
    };
    finish_reason: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  model?: string;
}

interface AsyncAgentRequest {
  agent: string;
  missionId: string;
  pipelineStage: string;
  resultsUrl: string;
  resultsAuth: string;
  d1ProxyUrl: string;
  d1ProxyAuth: string;
  forceCloud?: boolean;
  fallbackModel?: string;
  /** User's preferred response language (e.g. "French"). When set and not "English", injected into system prompt. */
  language?: string;
  /** Paradigm-resolved model ID for this stage (e.g. 'qwen2.5:14b-instruct-q4_K_M'). Passed to toolDeps for artifact provenance. */
  resolvedModel?: string;
  /**
   * Checkpoint v2: present when the hub is dispatching a single checkpoint.
   * Together with `worktreePath`, these fields route the dispatch through
   * the intercepted-checkpoint-complete path instead of the legacy whole-stage
   * execution.
   */
  checkpointIndex?: number;
  worktreePath?: string;
}

// --- Agent configs: canonical source is packages/agents/canonical-configs.json ---
// NanoClaw reads the same JSON the hub uses, extending with NanoClaw-specific fields.

interface NanoClawAgentConfig {
  prompt: string;
  promptByStage?: Record<string, string>;
  modelKey: string;
  tools: string[];
  maxTokens: number;
  oversight: string;
  maxToolRounds: number;
}

// NanoClaw-specific extensions (fields and extra tools not in canonical configs)
const NANOCLAW_EXTENSIONS: Record<
  string,
  { maxToolRounds: number; extraTools?: string[] }
> = {
  pm: {
    maxToolRounds: 15,
    extraTools: ['vault-read', 'mission-write', 'd1-query', 'audit-query'],
  },
  architect: {
    maxToolRounds: 15,
    extraTools: [
      'vault-read',
      'mission-write',
      'd1-query',
      'audit-query',
      'git-log',
      'git-diff',
      'vault-list',
      'bookmark-read',
      'project-read',
    ],
  },
  developer: {
    maxToolRounds: 20,
    extraTools: [
      'vault-read',
      'test-run',
      'migration-check',
      'git-log',
      'git-diff',
      'd1-query',
      'mission-write',
      'artifact-write',
    ],
  },
  qa: {
    maxToolRounds: 15,
    extraTools: ['vault-read', 'file-list', 'd1-query', 'mission-write'],
  },
  devops: {
    maxToolRounds: 10,
    extraTools: [
      'mission-read',
      'd1-query',
      'mission-write',
      'git-log',
      'git-diff',
    ],
  },
  'tech-writer': {
    maxToolRounds: 15,
    extraTools: [
      'vault-read',
      'vault-write-agent',
      'file-list',
      'mission-write',
      'vault-list',
      'd1-query',
    ],
  },
};

// Load canonical configs from the shared JSON and merge with NanoClaw extensions
function loadCanonicalConfigs(): Record<string, NanoClawAgentConfig> {
  try {
    const raw = readFileSync(
      join(AGENTS_BASE, 'canonical-configs.json'),
      'utf-8',
    );
    const canonical = JSON.parse(raw) as Record<
      string,
      {
        prompt: string;
        modelKey: string;
        tools: string[];
        maxTokens: number;
        oversight: string;
      }
    >;

    // Only include builder agents that NanoClaw runs
    const builderAgents = [
      'pm',
      'architect',
      'developer',
      'qa',
      'devops',
      'tech-writer',
    ];
    const configs: Record<string, NanoClawAgentConfig> = {};

    for (const name of builderAgents) {
      const base = canonical[name];
      if (!base) {
        log(`Warning: agent "${name}" not found in canonical configs`);
        continue;
      }
      const ext = NANOCLAW_EXTENSIONS[name] || { maxToolRounds: 10 };
      configs[name] = {
        prompt: base.prompt,
        modelKey: base.modelKey,
        // Merge canonical tools with NanoClaw-specific tools (deduplicated)
        tools: [...new Set([...base.tools, ...(ext.extraTools || [])])],
        maxTokens: base.maxTokens,
        oversight: base.oversight,
        maxToolRounds: ext.maxToolRounds,
      };
    }

    log(`Loaded canonical configs for ${Object.keys(configs).length} agents`);
    return configs;
  } catch (err) {
    log(`Error loading canonical configs: ${err}. Falling back to empty.`);
    return {};
  }
}

const AGENT_CONFIGS = loadCanonicalConfigs();

const PROPOSAL_TYPES: Record<string, string> = {
  pm: 'builder-spec',
  architect: 'builder-architecture',
  developer: 'builder-implementation',
  qa: 'builder-qa-review',
  devops: 'builder-deploy',
  'tech-writer': 'builder-docs',
};

// --- Helpers ---

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk: Buffer) => {
      data += chunk.toString();
    });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

function jsonResponse(
  res: ServerResponse,
  status: number,
  body: unknown,
): void {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

function log(msg: string): void {
  const ts = new Date().toISOString();
  console.log(`[hub-api ${ts}] ${msg}`);
}

/**
 * Call OpenRouter's chat completions API (OpenAI-compatible).
 */
async function callOpenRouter(
  apiKey: string,
  model: string,
  messages: ChatMessage[],
  tools: ToolDefinition[],
  maxTokens: number,
): Promise<ChatCompletionResponse> {
  const url = 'https://openrouter.ai/api/v1/chat/completions';

  const body: Record<string, unknown> = {
    model,
    messages,
    max_tokens: maxTokens,
  };

  if (tools.length > 0) {
    body.tools = tools.map((t) => ({
      type: 'function',
      function: {
        name: t.name,
        description: t.description,
        parameters: t.parameters ?? { type: 'object', properties: {} },
      },
    }));
  }

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
      'HTTP-Referer': 'https://alacrityhub.ca',
      'X-Title': 'Alacrity Hub Pipeline',
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => 'unknown error');
    throw new Error(`OpenRouter call failed (${res.status}): ${text}`);
  }

  return (await res.json()) as ChatCompletionResponse;
}

/**
 * Call the LLM via Portkey (OpenAI-compatible chat completions).
 * If cloud config is provided:
 *   - forceCloud=true → skip Portkey, call OpenRouter directly
 *   - forceCloud=false → try Portkey first, fall back to OpenRouter on failure
 */
async function callLLM(
  portkey: PortkeyConfig,
  model: string,
  messages: ChatMessage[],
  tools: ToolDefinition[],
  maxTokens: number,
  cloud?: { forceCloud: boolean; fallbackModel: string; apiKey: string },
): Promise<ChatCompletionResponse> {
  // Force cloud path — skip Portkey entirely
  if (cloud?.forceCloud) {
    log(
      `callLLM: force_cloud=true, using OpenRouter model=${cloud.fallbackModel}`,
    );
    return callOpenRouter(
      cloud.apiKey,
      cloud.fallbackModel,
      messages,
      tools,
      maxTokens,
    );
  }

  // Try Portkey (local Ollama)
  const url = `${portkey.base_url}/v1/chat/completions`;
  const resolvedModel = resolveModel(model);

  const body: Record<string, unknown> = {
    model: resolvedModel,
    messages,
    max_tokens: maxTokens,
  };

  if (tools.length > 0) {
    body.tools = tools.map((t) => ({
      type: 'function',
      function: {
        name: t.name,
        description: t.description,
        parameters: t.parameters ?? { type: 'object', properties: {} },
      },
    }));
  }

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-portkey-provider': 'ollama',
        'x-portkey-custom-host': OLLAMA_HOST,
        ...(portkey.api_key
          ? { Authorization: `Bearer ${portkey.api_key}` }
          : {}),
        ...(portkey.virtual_key
          ? { 'x-portkey-virtual-key': portkey.virtual_key }
          : {}),
      },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const text = await res.text().catch(() => 'unknown error');
      throw new Error(`LLM call failed (${res.status}): ${text}`);
    }

    log(`callLLM: Portkey succeeded (model=${resolvedModel})`);
    return (await res.json()) as ChatCompletionResponse;
  } catch (portkeyErr) {
    // If cloud fallback is available, try OpenRouter
    if (cloud) {
      log(
        `callLLM: Portkey failed (${portkeyErr}), falling back to OpenRouter model=${cloud.fallbackModel}`,
      );
      return callOpenRouter(
        cloud.apiKey,
        cloud.fallbackModel,
        messages,
        tools,
        maxTokens,
      );
    }
    // No fallback — rethrow
    throw portkeyErr;
  }
}

/**
 * Execute a tool call by POSTing to the hub's callback URL.
 */
async function executeToolCallback(
  callbackUrl: string,
  toolName: string,
  params: Record<string, unknown>,
): Promise<{ result: string }> {
  const res = await fetch(callbackUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tool: toolName, params }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => 'callback error');
    return { result: `Tool error (${res.status}): ${text}` };
  }

  return (await res.json()) as { result: string };
}

// Intercepted tools — handled locally in hub-api, not sent to callback server
const INTERCEPTED_TOOLS = new Set(['checkpoint_complete', 'preflight_rerun']);

// --- Agent-loop exit flags (checkpoint v2, Task 16) ---
//
// When the intercepted `checkpoint_complete` handler fires, the agent must
// stop calling tools — one dispatch equals one checkpoint. Between tool-call
// rounds, runAgentLoop checks this map. If a flag is set for the current
// checkpoint key, the loop exits cleanly.
//
// Key shape: `${missionId}:${stage}:${checkpointIndex}`. Non-checkpoint
// dispatches never set or read this map.

const agentExitFlags = new Map<string, boolean>();

function checkpointExitKey(ctx: CheckpointContext): string {
  return `${ctx.missionId}:${ctx.stage}:${ctx.checkpointIndex}`;
}

function setCheckpointExit(ctx: CheckpointContext): void {
  agentExitFlags.set(checkpointExitKey(ctx), true);
}

function shouldCheckpointExit(ctx: CheckpointContext): boolean {
  return agentExitFlags.get(checkpointExitKey(ctx)) === true;
}

function clearCheckpointExit(ctx: CheckpointContext): void {
  agentExitFlags.delete(checkpointExitKey(ctx));
}

/**
 * Context passed to intercepted tool handlers. All fields are optional —
 * only checkpoint_complete needs the checkpoint context, only preflight_rerun
 * needs pipelineStage/missionId. A single handler call may use any subset.
 */
interface InterceptedToolContext {
  missionId?: string;
  pipelineStage?: string;
  checkpointContext?: CheckpointContext;
  /** Running token count as of the current tool call (for telemetry). */
  tokensUsedSoFar?: number;
  /** Model ID that produced the current tool call (for telemetry). */
  currentModel?: string;
}

/**
 * Handle intercepted tools locally without calling the callback server.
 *
 * `checkpoint_complete` (v2): the critical handoff point. Validates the
 * handoff payload, commits the worktree, POSTs completion back to the hub's
 * /api/agent-results with checkpoint metadata (the hub's observer writes the
 * D1 transition), and sets the agent-exit flag. The agent will be forced to
 * exit the tool loop on the next round.
 *
 * `preflight_rerun`: re-runs the preflight script for the current mission/
 * stage and returns its output to the agent.
 */
async function handleInterceptedTool(
  toolName: string,
  params: Record<string, unknown>,
  ctx: InterceptedToolContext = {},
): Promise<string> {
  if (toolName === 'checkpoint_complete') {
    return handleCheckpointComplete(params, ctx);
  }

  if (toolName === 'preflight_rerun') {
    const scriptPath = join(
      process.env.HUB_ROOT ??
        join(process.env.HOME || '/root', 'Vibe Sphere', 'alacrity_hub'),
      'scripts',
      'preflight.sh',
    );
    try {
      const stage = ctx.pipelineStage ?? 'unknown';
      const missionId = ctx.missionId ?? 'unknown';
      const { stdout } = await execAsync(
        `bash "${scriptPath}" "${stage}" "${missionId}"`,
        { timeout: 30000 },
      );
      return stdout.trim();
    } catch (err: any) {
      return JSON.stringify({ status: 'fail', error: err.message });
    }
  }

  return JSON.stringify({ error: `Unknown intercepted tool: ${toolName}` });
}

/**
 * Inner handler for `checkpoint_complete`. Split out so the control flow is
 * easier to read than a nested if-block. See checkpoint-layer-v2 spec §7 for
 * the full contract.
 *
 * Invariants enforced here:
 *   - Summary is truncated to 1000 chars before being stored anywhere.
 *   - Git commit happens BEFORE the D1 write (durability fence). If the
 *     commit fails, the completion is not reported and the agent sees the
 *     error — the checkpoint stays in_progress and the orphan scan will
 *     eventually reclaim it for a retry.
 *   - The hub observer branch in /api/agent-results writes the D1 transition
 *     using the bootEpoch carried in the metadata as the WHERE-clause fence.
 *     This handler does not write D1 directly — see the "cubicle-worker"
 *     design: NanoClaw is stateless and only reports its identity.
 */
async function handleCheckpointComplete(
  params: Record<string, unknown>,
  ctx: InterceptedToolContext,
): Promise<string> {
  const cp = ctx.checkpointContext;
  if (!cp) {
    // Agent called checkpoint_complete on a non-checkpoint dispatch. Defensive
    // fallback: acknowledge and move on. Upstream will reject any attempt to
    // use this as a checkpoint because no context was ever attached.
    log(
      'checkpoint_complete called without checkpointContext — no-op fallback',
    );
    return JSON.stringify({
      result: 'Checkpoint acknowledged (no context).',
      warning: 'no-checkpoint-context',
    });
  }

  const rawSummary = String(params.summary ?? '');
  const summary = rawSummary.slice(0, 1000);
  const handoff =
    typeof params.handoff === 'object' && params.handoff !== null
      ? (params.handoff as Record<string, unknown>)
      : {};

  // --- 1. Commit the worktree (durability fence) ---
  try {
    await runGit(cp.worktreePath, ['add', '-A']);
    const msgBody = summary.slice(0, 72) || `(no summary)`;
    await runGit(cp.worktreePath, [
      'commit',
      '-m',
      `checkpoint(${cp.stage}/${cp.checkpointIndex}): ${msgBody}`,
      '--allow-empty',
    ]);
    log(`checkpoint_commit: ${cp.missionId}/${cp.stage}/${cp.checkpointIndex}`);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    log(
      `checkpoint_commit_failed: ${cp.missionId}/${cp.stage}/${cp.checkpointIndex}: ${msg}`,
    );
    // Surface the error to the agent. The checkpoint row stays in_progress;
    // the orphan scan will reclaim it on the next epoch change.
    return JSON.stringify({
      error: `checkpoint commit failed: ${msg}`,
      retryable: true,
    });
  }

  // --- 2. POST completion back to the hub ---
  //
  // The hub's /api/agent-results observer branch reads metadata.checkpoint and
  // writes the D1 transition. We never touch D1 directly from NanoClaw.
  const resultPayload: Record<string, unknown> = {
    missionId: cp.missionId,
    pipelineStage: cp.stage,
    status: 'success',
    modelUsed: ctx.currentModel ?? null,
    metadata: {
      checkpoint: {
        checkpointIndex: cp.checkpointIndex,
        bootEpoch: BOOT_EPOCH,
        summary,
        handoff,
        tokensUsed: ctx.tokensUsedSoFar ?? 0,
        modelUsed: ctx.currentModel ?? null,
      },
    },
    auditEntries: [
      {
        agentName: 'nanoclaw',
        actionType: 'checkpoint_complete',
        actionDetail: `Checkpoint ${cp.stage}/${cp.checkpointIndex} complete: ${summary.slice(0, 200)}`,
        target: 'builder-pipeline',
        result: 'success',
        missionId: cp.missionId,
        metadata: {
          checkpointIndex: cp.checkpointIndex,
          bootEpoch: BOOT_EPOCH,
          tokensUsed: ctx.tokensUsedSoFar ?? 0,
        },
      },
    ],
  };

  try {
    await postResults(cp.resultsUrl, cp.resultsAuth, resultPayload);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    log(
      `checkpoint_complete_post_failed: ${cp.missionId}/${cp.stage}/${cp.checkpointIndex}: ${msg}`,
    );
    // postResults already has its own retry + fallback-file logic; if it still
    // threw, the fallback file has been written. We set the exit flag anyway
    // so the agent stops calling tools — the cubicle-worker pattern means the
    // next dispatch's orphan-scan reconciliation handles the "callback never
    // landed" case (the commit is durable on the worktree).
  }

  // --- 3. Set the exit flag so runAgentLoop terminates on the next round ---
  setCheckpointExit(cp);

  return JSON.stringify({
    result:
      'Checkpoint marked complete. Output a single final message and stop calling tools.',
    summary_stored: true,
    truncated: rawSummary.length > summary.length,
  });
}

/**
 * Run the agent loop: call LLM, execute tool calls, repeat until done.
 *
 * Checkpoint v2: when `req.checkpointContext` is set, the intercepted
 * `checkpoint_complete` tool can set an exit flag mid-round. Between rounds
 * this loop checks the flag and exits cleanly, returning with
 * `checkpointCompleted: true` so the caller knows not to double-post results.
 */
async function runAgentLoop(
  req: AgentRunRequest,
  maxToolRounds?: number,
): Promise<AgentRunResponse> {
  const maxRounds = maxToolRounds ?? MAX_TOOL_ROUNDS;
  const startTime = Date.now();
  let totalTokens = 0;
  let lastResolvedModel = resolveModel(req.model);

  // Clear any stale exit flag for this checkpoint on entry — defensive against
  // a prior aborted dispatch leaving its flag behind.
  if (req.checkpointContext) {
    clearCheckpointExit(req.checkpointContext);
  }

  const messages: ChatMessage[] = [
    { role: 'system', content: req.system_prompt },
    { role: 'user', content: req.user_message },
  ];

  for (let round = 0; round < maxRounds; round++) {
    // Checkpoint v2: honor the exit flag set by checkpoint_complete. This
    // check runs at the top of each round, so the agent sees one final round
    // after the flag is set (where it can emit a closing message) before the
    // loop actually terminates on the next iteration.
    if (req.checkpointContext && shouldCheckpointExit(req.checkpointContext)) {
      log(
        `Agent loop exiting after checkpoint_complete: ${req.checkpointContext.missionId}/${req.checkpointContext.stage}/${req.checkpointContext.checkpointIndex}`,
      );
      clearCheckpointExit(req.checkpointContext);
      const lastAssistant = messages
        .filter((m) => m.role === 'assistant' && m.content)
        .pop();
      return {
        content: lastAssistant?.content ?? 'Checkpoint complete.',
        model: lastResolvedModel,
        tokensUsed: totalTokens,
        duration: Date.now() - startTime,
        checkpointCompleted: true,
      };
    }

    const resolvedModel = resolveModel(req.model);
    lastResolvedModel = resolvedModel;
    log(
      `Round ${round + 1}: calling LLM (${messages.length} messages, model=${req.model} → ${resolvedModel})`,
    );

    const completion = await callLLM(
      req.portkey,
      req.model,
      messages,
      req.tools,
      req.max_tokens,
      req.cloud,
    );

    if (completion.usage) {
      totalTokens += completion.usage.total_tokens;
    }
    if (completion.model) {
      lastResolvedModel = completion.model;
    }

    const choice = completion.choices[0];
    if (!choice) {
      throw new Error('LLM returned no choices');
    }

    const assistantMsg: ChatMessage = {
      role: 'assistant',
      content: choice.message.content,
    };
    if (choice.message.tool_calls?.length) {
      assistantMsg.tool_calls = choice.message.tool_calls;
    }
    messages.push(assistantMsg);

    // If no tool calls, we're done
    if (!choice.message.tool_calls?.length) {
      log(`Agent complete after ${round + 1} rounds`);
      return {
        content: choice.message.content ?? '',
        model: completion.model ?? req.model,
        tokensUsed: totalTokens,
        duration: Date.now() - startTime,
      };
    }

    // Execute each tool call
    for (const toolCall of choice.message.tool_calls) {
      const toolName = toolCall.function.name;
      let params: Record<string, unknown>;
      try {
        params = JSON.parse(toolCall.function.arguments);
      } catch {
        params = { raw: toolCall.function.arguments };
      }

      log(`Tool call: ${toolName}(${JSON.stringify(params).slice(0, 200)})`);

      let result: { result: string };

      // Intercepted tools — handle locally, don't send to callback server
      if (INTERCEPTED_TOOLS.has(toolName)) {
        const interceptResult = await handleInterceptedTool(toolName, params, {
          missionId: req.checkpointContext?.missionId,
          pipelineStage: req.checkpointContext?.stage,
          checkpointContext: req.checkpointContext,
          tokensUsedSoFar: totalTokens,
          currentModel: lastResolvedModel,
        });
        result = { result: interceptResult };
      } else {
        result = await executeToolCallback(
          req.tool_callback_url,
          toolName,
          params,
        );
      }

      messages.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        content: result.result,
      });
    }
  }

  // Max rounds reached — return last assistant content
  const lastAssistant = messages
    .filter((m) => m.role === 'assistant' && m.content)
    .pop();
  return {
    content: lastAssistant?.content ?? 'Agent reached maximum tool rounds.',
    model: req.model,
    tokensUsed: totalTokens,
    duration: Date.now() - startTime,
  };
}

// --- Async Agent Execution ---

// Cached repo structure snippet injected into agent prompts
let _repoStructure: string | null = null;
function getRepoStructure(): string {
  if (_repoStructure) return _repoStructure;
  const repoRoot = join(
    process.env.HOME || '/root',
    'Vibe Sphere',
    'alacrity_hub',
  );
  try {
    const { readdirSync } = require('fs');
    const topDirs = readdirSync(repoRoot, { withFileTypes: true })
      .filter(
        (d: { isDirectory: () => boolean; name: string }) =>
          d.isDirectory() &&
          !d.name.startsWith('.') &&
          d.name !== 'node_modules',
      )
      .map((d: { name: string }) => d.name);

    const structure: string[] = [];
    for (const dir of topDirs) {
      const subPath = join(repoRoot, dir);
      try {
        const subs = readdirSync(subPath, { withFileTypes: true })
          .filter(
            (d: { isDirectory: () => boolean; name: string }) =>
              d.isDirectory() &&
              !d.name.startsWith('.') &&
              d.name !== 'node_modules',
          )
          .map((d: { name: string }) => d.name);
        if (subs.length > 0) {
          structure.push(`${dir}/: ${subs.join(', ')}`);
        } else {
          structure.push(`${dir}/`);
        }
      } catch {
        structure.push(`${dir}/`);
      }
    }
    _repoStructure = `\n\n## Repository Structure\n${structure.join('\n')}\n\nUse these paths with file-list and grep-search. Do NOT guess paths that are not listed here.\n`;
  } catch {
    _repoStructure = '';
  }
  return _repoStructure;
}

function loadAgentPrompt(promptPath: string): string {
  let preamble = '';
  try {
    preamble =
      readFileSync(join(AGENTS_BASE, 'prompts', '_preamble.md'), 'utf-8') +
      '\n\n';
  } catch {
    // Preamble missing — continue without it
  }
  const prompt = readFileSync(join(AGENTS_BASE, promptPath), 'utf-8');
  return preamble + prompt + getRepoStructure();
}

function loadToolDefs(toolNames: string[]): ToolDefinition[] {
  return toolNames.map((name) => {
    try {
      return JSON.parse(
        readFileSync(join(AGENTS_BASE, 'tools', `${name}.json`), 'utf-8'),
      ) as ToolDefinition;
    } catch (err) {
      log(`Warning: could not load tool definition for ${name}: ${err}`);
      return { name, description: `Tool ${name} (definition not found)` };
    }
  });
}

const FALLBACK_DIR = join(
  process.env.HOME || '/root',
  'Vibe Sphere',
  'agent-results-fallback',
);

async function postResults(
  url: string,
  authToken: string,
  payload: unknown,
): Promise<void> {
  const delays = [1000, 2000, 4000, 8000];
  for (let attempt = 0; attempt <= delays.length; attempt++) {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${authToken}`,
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(30_000),
      });
      if (res.ok) return;
      log(
        `Results POST to ${url} failed (${res.status}), attempt ${attempt + 1}`,
      );
    } catch (err) {
      log(`Results POST to ${url} error, attempt ${attempt + 1}: ${err}`);
    }
    if (attempt < delays.length) {
      await new Promise((r) => setTimeout(r, delays[attempt]));
    }
  }
  // All retries failed — write to fallback file
  mkdirSync(FALLBACK_DIR, { recursive: true });
  const p = payload as Record<string, unknown>;
  const filename = `${Date.now()}-${(p.missionId as string) || 'unknown'}.json`;
  // Store the resultsUrl and auth alongside the payload for replay
  const fallbackData = { _resultsUrl: url, _resultsAuth: authToken, ...p };
  writeFileSync(
    join(FALLBACK_DIR, filename),
    JSON.stringify(fallbackData, null, 2),
  );
  log(`All result POST retries failed. Written to fallback: ${filename}`);
}

async function runAsyncAgent(req: AsyncAgentRequest): Promise<void> {
  const {
    agent,
    missionId,
    pipelineStage,
    resultsUrl,
    resultsAuth,
    d1ProxyUrl,
    d1ProxyAuth,
    checkpointIndex,
    worktreePath,
  } = req;

  // Build the checkpoint context for v2 dispatches. If checkpointIndex is
  // present but worktreePath isn't, we log and fall through to non-checkpoint
  // execution — the hub is responsible for pairing the two, and running
  // checkpoint stages without a worktree to commit into is undefined.
  let checkpointContext: CheckpointContext | undefined;
  if (checkpointIndex !== undefined) {
    if (!worktreePath) {
      log(
        `checkpoint dispatch ${missionId}/${pipelineStage}/${checkpointIndex} missing worktreePath — falling back to non-checkpoint execution`,
      );
    } else {
      checkpointContext = {
        missionId,
        stage: pipelineStage,
        checkpointIndex,
        worktreePath,
        resultsUrl,
        resultsAuth,
      };
    }
  }

  const agentConfig = AGENT_CONFIGS[agent];
  if (!agentConfig) {
    await postResults(resultsUrl, resultsAuth, {
      missionId,
      pipelineStage,
      status: 'error',
      errorCode: 'unknown_agent',
      errorMessage: `Unknown agent: ${agent}`,
      auditEntries: [
        {
          agentName: agent,
          actionType: 'agent_error',
          actionDetail: `Unknown agent: ${agent}`,
          target: 'builder-pipeline',
          result: 'failed',
        },
      ],
    });
    return;
  }

  const startTime = Date.now();

  try {
    // Load prompt and tool definitions from filesystem
    // Select stage-specific prompt if available (e.g. briefing vs pm_spec)
    const promptPath =
      agentConfig.promptByStage?.[pipelineStage] ?? agentConfig.prompt;
    let systemPrompt = loadAgentPrompt(promptPath);
    if (req.language && req.language !== 'English') {
      systemPrompt += `\n\nAlways respond in ${req.language}. Do not switch languages unless the user explicitly asks.`;
    }
    const tools = loadToolDefs(agentConfig.tools);

    // Import tool handlers dynamically from the agents package
    const agentsPkgSrc = join(
      process.env.HOME || '/root',
      'Vibe Sphere',
      'alacrity_hub',
      'packages',
      'agents',
      'src',
    );

    let createToolHandlers: (
      deps: Record<string, unknown>,
    ) => Record<string, (params: Record<string, unknown>) => Promise<unknown>>;
    let createD1ProxyAdapter: (url: string, auth: string) => unknown;
    try {
      const thModule = await import(join(agentsPkgSrc, 'tool-handlers.ts'));
      createToolHandlers = thModule.createToolHandlers;
      const d1Module = await import(join(agentsPkgSrc, 'd1-proxy-adapter.ts'));
      createD1ProxyAdapter = d1Module.createD1ProxyAdapter;
    } catch (importErr) {
      await postResults(resultsUrl, resultsAuth, {
        missionId,
        pipelineStage,
        status: 'error',
        errorCode: 'import_error',
        errorMessage: `Failed to import tool handlers: ${String(importErr)}`,
        auditEntries: [
          {
            agentName: agent,
            actionType: 'agent_error',
            actionDetail: `Import error: ${String(importErr)}`,
            target: 'builder-pipeline',
            result: 'failed',
          },
        ],
      });
      return;
    }

    // Create D1 proxy adapter and tool handlers
    const d1 = createD1ProxyAdapter(d1ProxyUrl, d1ProxyAuth);
    const toolDeps: Record<string, unknown> = {
      d1,
      chromadbHost: CHROMADB_HOST,
      ollamaHost: OLLAMA_HOST,
      tavilyApiKey: process.env.TAVILY_API_KEY || '',
      vaultIndexerUrl: process.env.VAULT_INDEXER_URL || 'http://127.0.0.1:3001',
      fetch: globalThis.fetch,
      repoRoot: join(
        process.env.HOME || '/root',
        'Vibe Sphere',
        'alacrity_hub',
      ),
      resolvedModel: req.resolvedModel ?? null,
    };
    const toolHandlers = createToolHandlers(toolDeps);

    // Tool callback function for local execution
    const toolCallback = async (
      toolName: string,
      params: Record<string, unknown>,
    ): Promise<string> => {
      if (!agentConfig.tools.includes(toolName)) {
        return JSON.stringify({
          error: `Tool not allowed for agent ${agent}: ${toolName}`,
        });
      }
      const handler = toolHandlers[toolName];
      if (!handler) {
        return JSON.stringify({ error: `Unknown tool: ${toolName}` });
      }
      try {
        const result = await handler(params);
        return typeof result === 'string' ? result : JSON.stringify(result);
      } catch (err) {
        return JSON.stringify({ error: `Tool error: ${String(err)}` });
      }
    };

    // Start temporary callback server for tool execution
    // Use port 0 to let the OS assign an available ephemeral port (avoids collisions under concurrency)
    const callbackPort = 0;
    const callbackServer = createServer(async (cbReq, cbRes) => {
      if (cbReq.method !== 'POST' || cbReq.url !== '/tool-callback') {
        cbRes.writeHead(404);
        cbRes.end(JSON.stringify({ error: 'Not found' }));
        return;
      }
      let body = '';
      cbReq.on('data', (chunk: Buffer) => {
        body += chunk.toString();
      });
      cbReq.on('end', async () => {
        try {
          const { tool, params } = JSON.parse(body);
          const result = await toolCallback(tool, params);
          cbRes.writeHead(200, { 'Content-Type': 'application/json' });
          cbRes.end(JSON.stringify({ result }));
        } catch (err) {
          cbRes.writeHead(500, { 'Content-Type': 'application/json' });
          cbRes.end(
            JSON.stringify({ result: `Callback error: ${String(err)}` }),
          );
        }
      });
    });

    await new Promise<void>((resolve, reject) => {
      callbackServer.listen(callbackPort, '127.0.0.1', () => resolve());
      callbackServer.on('error', reject);
    });

    // Read the actual assigned port (needed when callbackPort is 0)
    const actualPort = (callbackServer.address() as { port: number }).port;

    // Construct cloud config for OpenRouter fallback
    const cloud =
      req.forceCloud && req.fallbackModel && OPENROUTER_API_KEY
        ? {
            forceCloud: true,
            fallbackModel: req.fallbackModel,
            apiKey: OPENROUTER_API_KEY,
          }
        : OPENROUTER_API_KEY && req.fallbackModel
          ? {
              forceCloud: false,
              fallbackModel: req.fallbackModel,
              apiKey: OPENROUTER_API_KEY,
            }
          : undefined;

    try {
      // Dispatch branch: checkpoint v2 uses the checkpoint context to thread
      // missionId/stage/checkpointIndex/worktreePath through the intercepted
      // checkpoint_complete handler. Non-checkpoint dispatches fall through
      // to a plain agent loop. (The legacy fetchCheckpoints/runCheckpointAware
      // path was removed in Task 17 — see git log.)
      let result: AgentRunResponse;
      if (checkpointContext) {
        log(
          `Checkpoint v2 dispatch: ${missionId}/${pipelineStage}/${checkpointContext.checkpointIndex} (worktree=${worktreePath})`,
        );
        result = await runAgentLoop(
          {
            model: agentConfig.modelKey,
            system_prompt: systemPrompt,
            user_message: JSON.stringify({
              missionId,
              stage: pipelineStage,
              checkpointIndex: checkpointContext.checkpointIndex,
              worktreePath,
            }),
            tools,
            max_tokens: agentConfig.maxTokens,
            portkey: {
              base_url: PORTKEY_BASE_URL,
              api_key: PORTKEY_API_KEY,
              virtual_key: agentConfig.modelKey,
            },
            tool_callback_url: `http://127.0.0.1:${actualPort}/tool-callback`,
            cloud,
            checkpointContext,
          },
          agentConfig.maxToolRounds,
        );
      } else {
        // Standard agent loop (no checkpoints)
        result = await runAgentLoop(
          {
            model: agentConfig.modelKey,
            system_prompt: systemPrompt,
            user_message: JSON.stringify({ missionId }),
            tools,
            max_tokens: agentConfig.maxTokens,
            portkey: {
              base_url: PORTKEY_BASE_URL,
              api_key: PORTKEY_API_KEY,
              virtual_key: agentConfig.modelKey,
            },
            tool_callback_url: `http://127.0.0.1:${actualPort}/tool-callback`,
            cloud,
          },
          agentConfig.maxToolRounds,
        );
      }

      const duration = Date.now() - startTime;

      // Determine which LLM path was actually used
      const llmPath = cloud?.forceCloud
        ? 'openrouter-forced'
        : cloud && result.model === cloud.fallbackModel
          ? 'openrouter-fallback'
          : 'portkey-local';

      // Build result payload
      const resultPayload: Record<string, unknown> = {
        missionId,
        pipelineStage,
        status: 'success',
        modelUsed: result.model,
        llmPath,
        auditEntries: [
          {
            agentName: agent,
            actionType: 'agent_complete',
            actionDetail: `Agent completed in ${duration}ms (${llmPath})`,
            target: 'builder-pipeline',
            result: 'success',
            metadata: {
              tokensUsed: result.tokensUsed,
              model: result.model,
              llmPath,
              duration,
            },
          },
        ],
      };

      // If HitL agent, construct proposal
      if (agentConfig.oversight === 'hitl') {
        let parsed: Record<string, unknown>;
        try {
          parsed = JSON.parse(result.content);
        } catch {
          parsed = { raw: result.content };
        }

        const agentBriefing = (parsed as Record<string, Record<string, string>>)
          .briefing;
        const briefing = agentBriefing
          ? {
              whats_changing:
                agentBriefing.whats_changing ||
                agentBriefing.whatsChanging ||
                'Agent proposal requires review',
              why_now:
                agentBriefing.why_now ||
                agentBriefing.whyNow ||
                'Triggered from builder-pipeline',
              long_term_benefit:
                agentBriefing.long_term_benefit ||
                agentBriefing.longTermBenefit ||
                'Maintains quality through human review',
              risk_if_approved:
                agentBriefing.risk_if_approved ||
                agentBriefing.riskIfApproved ||
                'Review the proposal details before approving',
            }
          : {
              whats_changing:
                (parsed as any).spec?.title ||
                (parsed as any).summary ||
                (parsed as any).raw?.slice(0, 200) ||
                'Agent proposal requires review',
              why_now: 'Triggered from builder-pipeline',
              long_term_benefit:
                (parsed as any).spec?.notes ||
                'Maintains quality through human review',
              risk_if_approved: 'Review the full proposal via View Diff',
            };

        resultPayload.proposal = {
          agentName: agent,
          proposalType: PROPOSAL_TYPES[agent] || 'note-refinement',
          payload: JSON.stringify(parsed),
          briefing: JSON.stringify(briefing),
        };
      }

      // Checkpoint v2: if runAgentLoop exited because checkpoint_complete
      // fired, the intercepted handler already POSTed results to the hub
      // with the checkpoint metadata. Skip the normal post-loop POST to
      // avoid a duplicate that would confuse the hub observer.
      if (result.checkpointCompleted) {
        log(
          `checkpoint_complete already posted results for ${missionId}/${pipelineStage}; skipping normal post-loop POST`,
        );
      } else {
        await postResults(resultsUrl, resultsAuth, resultPayload);
      }
    } finally {
      callbackServer.close();
    }
  } catch (err) {
    const duration = Date.now() - startTime;
    const isTimeout = err instanceof DOMException && err.name === 'AbortError';

    await postResults(resultsUrl, resultsAuth, {
      missionId,
      pipelineStage,
      status: 'error',
      errorCode: isTimeout ? 'agent_timeout' : 'unknown',
      errorMessage: isTimeout
        ? `Agent timed out after ${duration}ms`
        : `Agent failed: ${String(err)}`,
      auditEntries: [
        {
          agentName: agent,
          actionType: 'agent_error',
          actionDetail: String(err),
          target: 'builder-pipeline',
          result: 'failed',
        },
      ],
    });
  }
}

// --- HTTP Server ---

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? '/', `http://${HOST}:${PORT}`);
  const method = req.method?.toUpperCase() ?? 'GET';

  // CORS for local development
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'Content-Type, CF-Access-Client-Id, CF-Access-Client-Secret',
  );

  if (method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // GET /api/health
  if (url.pathname === '/api/health' && method === 'GET') {
    jsonResponse(res, 200, {
      status: 'ok',
      service: 'nanoclaw-hub-api',
      port: PORT,
      uptime: process.uptime(),
      boot_epoch: BOOT_EPOCH,
      concurrency: {
        active: agentSemaphore.active,
        queued: agentSemaphore.queued,
        max: MAX_CONCURRENT_AGENTS,
      },
    });
    return;
  }

  // POST /api/llm/chat — proxy LLM requests to local Portkey
  if (url.pathname === '/api/llm/chat' && method === 'POST') {
    try {
      const body = await readBody(req);
      const request = JSON.parse(body) as {
        messages: ChatMessage[];
        virtual_key?: string;
        session_mode?: string;
        pressure_level?: string;
        temperature?: number;
        max_tokens?: number;
      };

      if (!request.messages?.length) {
        jsonResponse(res, 400, { error: 'Missing required field: messages' });
        return;
      }

      const virtualKey = request.virtual_key || 'local-reasoning';
      const resolvedModel = resolveModel(virtualKey);

      log(
        `LLM proxy: virtual_key=${virtualKey}, model=${resolvedModel}, messages=${request.messages.length}`,
      );

      const portkeyRes = await fetch(
        `${PORTKEY_BASE_URL}/v1/chat/completions`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-portkey-provider': 'ollama',
            'x-portkey-custom-host': OLLAMA_HOST,
            ...(PORTKEY_API_KEY
              ? { Authorization: `Bearer ${PORTKEY_API_KEY}` }
              : {}),
            ...(virtualKey ? { 'x-portkey-virtual-key': virtualKey } : {}),
            ...(request.session_mode
              ? { 'x-portkey-session-mode': request.session_mode }
              : {}),
            ...(request.pressure_level
              ? { 'x-portkey-pressure-level': request.pressure_level }
              : {}),
          },
          body: JSON.stringify({
            model: resolvedModel,
            messages: request.messages,
            temperature: request.temperature ?? 0.7,
            max_tokens: request.max_tokens ?? 4096,
          }),
        },
      );

      if (!portkeyRes.ok) {
        const errorText = await portkeyRes.text().catch(() => 'unknown error');
        log(`LLM proxy error (${portkeyRes.status}): ${errorText}`);
        jsonResponse(res, portkeyRes.status, { error: errorText });
        return;
      }

      const data = await portkeyRes.json();
      jsonResponse(res, 200, data);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`LLM proxy error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // POST /api/rag/query — query ChromaDB via local Ollama embeddings
  if (url.pathname === '/api/rag/query' && method === 'POST') {
    try {
      const body = await readBody(req);
      const request = JSON.parse(body) as {
        query: string;
        scope?: 'human' | 'agent' | 'both';
        nResults?: number;
      };

      if (!request.query) {
        jsonResponse(res, 400, { error: 'Missing required field: query' });
        return;
      }

      const scope = request.scope || 'both';
      const nResults = request.nResults || 5;
      const collections: string[] = [];
      if (scope === 'human' || scope === 'both')
        collections.push('human-vault');
      if (scope === 'agent' || scope === 'both')
        collections.push('agent-vault');

      log(
        `RAG query: scope=${scope}, nResults=${nResults}, collections=${collections.join(',')}`,
      );

      // Generate embedding via Ollama
      const embedRes = await fetch(`${OLLAMA_HOST}/api/embed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: EMBED_MODEL, input: [request.query] }),
      });

      if (!embedRes.ok) {
        const errText = await embedRes.text().catch(() => 'embed error');
        log(`Ollama embed failed (${embedRes.status}): ${errText}`);
        jsonResponse(res, 502, { error: `Embedding failed: ${errText}` });
        return;
      }

      const embedData = (await embedRes.json()) as { embeddings: number[][] };
      const queryEmbedding = embedData.embeddings[0];

      // Query each collection in parallel
      const allResults: Array<{
        content: string;
        source: string;
        path: string;
        agent?: string;
        mission?: string;
        contentType?: string;
        score: number;
      }> = [];

      await Promise.all(
        collections.map(async (collectionName) => {
          try {
            const queryRes = await fetch(
              `${CHROMADB_HOST}/api/v1/collections/${collectionName}/query`,
              {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  query_embeddings: [queryEmbedding],
                  n_results: nResults,
                }),
              },
            );

            if (!queryRes.ok) {
              log(
                `ChromaDB query failed for ${collectionName}: ${queryRes.status}`,
              );
              return;
            }

            const results = (await queryRes.json()) as {
              ids: string[][];
              documents: (string | null)[][];
              metadatas: (Record<string, unknown> | null)[][];
              distances?: number[][];
            };

            const ids = results.ids[0] ?? [];
            const docs = results.documents[0] ?? [];
            const metas = results.metadatas[0] ?? [];
            const dists = results.distances?.[0] ?? [];

            for (let i = 0; i < ids.length; i++) {
              const doc = docs[i];
              if (!doc) continue;
              const meta = metas[i];
              const distance = dists[i] ?? 1;
              const score = Math.max(0, Math.min(1, 1 - distance));

              allResults.push({
                content: doc,
                source: collectionName,
                path:
                  (meta?.path as string) ||
                  (meta?.source as string) ||
                  'unknown',
                agent: meta?.agent as string | undefined,
                mission: meta?.mission as string | undefined,
                contentType: meta?.contentType as string | undefined,
                score,
              });
            }
          } catch (err) {
            log(`ChromaDB query error for ${collectionName}: ${err}`);
          }
        }),
      );

      // Sort by score descending
      allResults.sort((a, b) => b.score - a.score);

      jsonResponse(res, 200, {
        results: allResults,
        totalFound: allResults.length,
        scope,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`RAG query error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // POST /api/agent/run
  if (url.pathname === '/api/agent/run' && method === 'POST') {
    try {
      const body = await readBody(req);
      const request = JSON.parse(body) as AgentRunRequest;

      if (!request.system_prompt || !request.user_message) {
        jsonResponse(res, 400, {
          error: 'Missing required fields: system_prompt, user_message',
        });
        return;
      }

      log(
        `Agent run: model=${request.model}, tools=${request.tools?.length ?? 0}, callback=${request.tool_callback_url}`,
      );

      const result = await runAgentLoop(request);

      log(
        `Agent done: ${result.tokensUsed} tokens, ${result.duration}ms, content=${result.content.slice(0, 100)}`,
      );

      jsonResponse(res, 200, result);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`Agent error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // POST /api/agent/run-async — async agent execution (cloud-first)
  if (url.pathname === '/api/agent/run-async' && method === 'POST') {
    try {
      log(`Received /api/agent/run-async request`);

      // Auth: Cloudflare tunnel validates CF Access headers (strips them before
      // forwarding to origin). Hub-api validates the shared AGENT_RESULTS_SECRET
      // as a bearer token for defense-in-depth.
      if (AGENT_RESULTS_SECRET) {
        const auth = req.headers['authorization'];
        if (!auth || auth !== `Bearer ${AGENT_RESULTS_SECRET}`) {
          log('Agent async auth rejected: invalid bearer token');
          jsonResponse(res, 403, { error: 'Invalid authorization' });
          return;
        }
      }

      const body = JSON.parse(await readBody(req));
      const {
        agent,
        missionId,
        pipelineStage,
        resultsUrl,
        resultsAuth,
        d1ProxyUrl,
        d1ProxyAuth,
        forceCloud,
        fallbackModel,
        language,
        resolvedModel,
        checkpointIndex,
        worktreePath,
        metadata,
      } = body;

      if (
        !agent ||
        !missionId ||
        !pipelineStage ||
        !resultsUrl ||
        !resultsAuth
      ) {
        jsonResponse(res, 400, {
          error:
            'Missing required fields: agent, missionId, pipelineStage, resultsUrl, resultsAuth',
        });
        return;
      }

      // Checkpoint v2: checkpointIndex + worktreePath may come at the top
      // level of the body OR nested under metadata.{...}. Accept either
      // shape so the hub's dispatchCheckpoint is free to evolve without a
      // lock-step NanoClaw deploy.
      const resolvedCheckpointIndex: number | undefined =
        typeof checkpointIndex === 'number'
          ? checkpointIndex
          : typeof metadata?.checkpointIndex === 'number'
            ? metadata.checkpointIndex
            : undefined;
      const resolvedWorktreePath: string | undefined =
        worktreePath || metadata?.worktreePath || undefined;

      // Return 202 immediately, run agent in background
      const queuePos = agentSemaphore.queued;
      jsonResponse(res, 202, {
        status: 'accepted',
        agent,
        missionId,
        concurrency: { active: agentSemaphore.active, queued: queuePos },
      });

      // Spawn background execution with concurrency limit
      setImmediate(async () => {
        if (agentSemaphore.active >= MAX_CONCURRENT_AGENTS) {
          log(
            `Agent ${agent}/${missionId} queued (${agentSemaphore.active} active, ${agentSemaphore.queued + 1} will wait)`,
          );
        }
        await agentSemaphore.acquire();
        log(
          `Agent ${agent}/${missionId} acquired slot (${agentSemaphore.active} active, ${agentSemaphore.queued} waiting)`,
        );
        try {
          await runAsyncAgent({
            agent,
            missionId,
            pipelineStage,
            resultsUrl,
            resultsAuth,
            d1ProxyUrl: d1ProxyUrl || '',
            d1ProxyAuth: d1ProxyAuth || '',
            forceCloud: forceCloud || false,
            fallbackModel: fallbackModel || '',
            language: language || undefined,
            resolvedModel: resolvedModel || undefined,
            checkpointIndex: resolvedCheckpointIndex,
            worktreePath: resolvedWorktreePath,
          });
        } catch (err) {
          log(`Async agent error for ${agent}/${missionId}: ${err}`);
        } finally {
          log(
            `Agent ${agent}/${missionId} releasing slot (was ${agentSemaphore.active} active, ${agentSemaphore.queued} waiting)`,
          );
          agentSemaphore.release();
        }
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`Async agent request error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // POST /api/mission-intake — write mission to local markdown file
  if (url.pathname === '/api/mission-intake' && method === 'POST') {
    try {
      const body = await readBody(req);
      const intake = JSON.parse(body) as {
        id: string;
        title: string;
        work_type: string;
        app_area?: string;
        target_component?: string;
        description?: string;
        created_at: string;
      };

      if (!intake.id || !intake.title) {
        jsonResponse(res, 400, { error: 'Missing required fields: id, title' });
        return;
      }

      const intakeDir = join(
        process.env.HOME ?? '/root',
        'Vibe Sphere',
        'mission-intake',
      );
      mkdirSync(intakeDir, { recursive: true });

      const date =
        intake.created_at?.slice(0, 10) ??
        new Date().toISOString().slice(0, 10);
      const slug = intake.title
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-|-$/g, '')
        .slice(0, 50);
      const shortId = intake.id.slice(0, 8);
      const filename = `${date}-${slug}-${shortId}.md`;
      const filepath = join(intakeDir, filename);

      const workTypeLabel =
        intake.work_type === 'new-sub-app'
          ? 'New Sub-App'
          : intake.work_type === 'fix'
            ? 'Fix'
            : 'Enhancement';

      const md = [
        '---',
        `id: ${intake.id}`,
        `title: "${intake.title.replace(/"/g, '\\"')}"`,
        `work_type: ${intake.work_type}`,
        `app_area: ${intake.app_area || ''}`,
        `target_component: ${intake.target_component || ''}`,
        `status: pending`,
        `created_at: ${intake.created_at}`,
        `synced_to_d1: unknown`,
        '---',
        '',
        `# ${intake.title}`,
        '',
        `**Type:** ${workTypeLabel}`,
        intake.app_area ? `**Area:** ${intake.app_area}` : null,
        intake.target_component
          ? `**Component:** ${intake.target_component}`
          : null,
        '',
        '## Description',
        '',
        intake.description || '_No description provided._',
        '',
      ]
        .filter((line) => line !== null)
        .join('\n');

      writeFileSync(filepath, md, 'utf-8');
      log(`Mission intake written: ${filename}`);

      jsonResponse(res, 200, { status: 'ok', path: filepath, filename });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`Mission intake error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // POST /api/preflight — run preflight environment checks
  if (url.pathname === '/api/preflight' && method === 'POST') {
    try {
      const body = await readBody(req);
      const request = JSON.parse(body) as {
        stage: string;
        missionId: string;
        worktreePath?: string;
        d1ProxyUrl?: string;
        d1ProxyAuth?: string;
        forceCloud?: boolean;
      };

      if (!request.stage || !request.missionId) {
        jsonResponse(res, 400, {
          error: 'Missing required fields: stage, missionId',
        });
        return;
      }

      const scriptPath = join(
        process.env.HUB_ROOT ??
          join(process.env.HOME || '/root', 'Vibe Sphere', 'alacrity_hub'),
        'scripts',
        'preflight.sh',
      );

      const args = [
        request.stage,
        request.missionId,
        request.worktreePath ?? '',
        request.d1ProxyUrl ?? '',
        request.d1ProxyAuth ?? '',
        request.forceCloud ? 'true' : 'false',
      ];

      const cmd = `bash "${scriptPath}" ${args.map((a) => `"${a}"`).join(' ')}`;
      log(`Running preflight: ${cmd.slice(0, 200)}...`);

      try {
        const { stdout } = await execAsync(cmd, { timeout: 35000 });
        const result = JSON.parse(stdout.trim());
        jsonResponse(res, 200, result);
      } catch (execErr: any) {
        // Script execution failed — return structured error
        jsonResponse(res, 200, {
          status: 'fail',
          stage: request.stage,
          checks: [],
          error: execErr.message ?? 'preflight script failed',
          timestamp: new Date().toISOString(),
        });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`Preflight error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // =========================================================================
  // Orientation Tools — uniform access for all agent contexts
  // =========================================================================

  // POST /api/tools/blast-radius — compute transitive import impact
  if (url.pathname === '/api/tools/blast-radius' && method === 'POST') {
    try {
      const body = await readBody(req);
      const request = JSON.parse(body) as {
        path: string;
        maxDepth?: number;
      };

      if (!request.path) {
        jsonResponse(res, 400, { error: 'Missing required field: path' });
        return;
      }

      const graph = buildImportGraph(ALACRITY_HUB_ROOT);
      const absPath = resolve(ALACRITY_HUB_ROOT, request.path);

      if (!graph.has(absPath)) {
        jsonResponse(res, 404, {
          error: `File not found in import graph: ${request.path}`,
        });
        return;
      }

      const result = computeBlastRadius(
        graph,
        absPath,
        ALACRITY_HUB_ROOT,
        request.maxDepth ?? 5,
      );
      log(
        `blast-radius: ${request.path} → ${result.transitiveDependents} affected`,
      );
      jsonResponse(res, 200, { path: request.path, ...result });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`blast-radius error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // POST /api/tools/dep-graph — query direct dependencies
  if (url.pathname === '/api/tools/dep-graph' && method === 'POST') {
    try {
      const body = await readBody(req);
      const request = JSON.parse(body) as {
        path: string;
        direction?: 'imports' | 'importedBy' | 'both';
      };

      if (!request.path) {
        jsonResponse(res, 400, { error: 'Missing required field: path' });
        return;
      }

      const graph = buildImportGraph(ALACRITY_HUB_ROOT);
      const absPath = resolve(ALACRITY_HUB_ROOT, request.path);
      const node = graph.get(absPath);

      if (!node) {
        jsonResponse(res, 404, {
          error: `File not found in import graph: ${request.path}`,
        });
        return;
      }

      const direction = request.direction ?? 'both';
      const toRel = (p: string) => relative(ALACRITY_HUB_ROOT, p);
      const result: Record<string, unknown> = { path: toRel(absPath) };

      if (direction === 'imports' || direction === 'both') {
        result.imports = node.imports.map(toRel);
        result.importsCount = node.imports.length;
      }
      if (direction === 'importedBy' || direction === 'both') {
        result.importedBy = node.importedBy.map(toRel);
        result.importedByCount = node.importedBy.length;
      }

      log(`dep-graph: ${request.path} (${direction})`);
      jsonResponse(res, 200, result);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`dep-graph error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // POST /api/tools/doc-deps — query documentation references
  if (url.pathname === '/api/tools/doc-deps' && method === 'POST') {
    try {
      const body = await readBody(req);
      const request = JSON.parse(body) as {
        path: string;
        direction?: 'references' | 'referencedBy' | 'both';
      };

      if (!request.path) {
        jsonResponse(res, 400, { error: 'Missing required field: path' });
        return;
      }

      const graph = buildDocGraph(ALACRITY_HUB_ROOT);
      const absPath = resolve(ALACRITY_HUB_ROOT, request.path);

      if (!graph.has(absPath)) {
        jsonResponse(res, 404, {
          error: `File not found in doc graph: ${request.path}`,
        });
        return;
      }

      const result = queryDocDeps(
        graph,
        absPath,
        ALACRITY_HUB_ROOT,
        request.direction ?? 'both',
      );
      log(`doc-deps: ${request.path}`);
      jsonResponse(res, 200, result);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`doc-deps error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // POST /api/tools/doc-impact — compute documentation blast radius
  if (url.pathname === '/api/tools/doc-impact' && method === 'POST') {
    try {
      const body = await readBody(req);
      const request = JSON.parse(body) as {
        path: string;
        maxDepth?: number;
      };

      if (!request.path) {
        jsonResponse(res, 400, { error: 'Missing required field: path' });
        return;
      }

      const graph = buildDocGraph(ALACRITY_HUB_ROOT);
      const absPath = resolve(ALACRITY_HUB_ROOT, request.path);

      if (!graph.has(absPath)) {
        jsonResponse(res, 404, {
          error: `File not found in doc graph: ${request.path}`,
        });
        return;
      }

      const result = computeDocImpact(
        graph,
        absPath,
        ALACRITY_HUB_ROOT,
        request.maxDepth ?? 3,
      );
      log(
        `doc-impact: ${request.path} → ${result.affectedDocs.length} affected`,
      );
      jsonResponse(res, 200, { path: request.path, ...result });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`doc-impact error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // POST /api/tools/doc-context — find relevant docs by topic
  if (url.pathname === '/api/tools/doc-context' && method === 'POST') {
    try {
      const body = await readBody(req);
      const request = JSON.parse(body) as {
        topic: string;
        limit?: number;
      };

      if (!request.topic) {
        jsonResponse(res, 400, { error: 'Missing required field: topic' });
        return;
      }

      const graph = buildDocGraph(ALACRITY_HUB_ROOT);
      const result = queryDocContext(
        graph,
        ALACRITY_HUB_ROOT,
        request.topic,
        request.limit ?? 10,
      );
      log(`doc-context: "${request.topic}" → ${result.docs.length} docs`);
      jsonResponse(res, 200, result);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`doc-context error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // POST /api/tools/vault-query — search vault notes
  if (url.pathname === '/api/tools/vault-query' && method === 'POST') {
    try {
      const body = await readBody(req);
      const params = JSON.parse(body) as VaultQueryParams;

      const results = queryVaultGraph(params);
      log(
        `vault-query: ${params.query ?? '(no query)'} → ${results.length} results`,
      );
      jsonResponse(res, 200, { count: results.length, results });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`vault-query error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // GET /api/tools/vault-stats — vault index stats
  if (url.pathname === '/api/tools/vault-stats' && method === 'GET') {
    try {
      const stats = getVaultStats();
      jsonResponse(res, 200, stats);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`vault-stats error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // POST /api/tools/vault-refresh — invalidate vault cache
  if (url.pathname === '/api/tools/vault-refresh' && method === 'POST') {
    try {
      const body = await readBody(req);
      const request = JSON.parse(body) as { vault?: string };
      const target = request.vault ?? 'all';

      if (target === 'all') {
        invalidateVaultCache();
      } else {
        invalidateVaultCache(target);
      }

      log(`vault-refresh: ${target}`);
      jsonResponse(res, 200, { invalidated: target });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`vault-refresh error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // =========================================================================
  // Vault Promote / Quarantine — filesystem ops for vault promotion flow
  // =========================================================================

  // POST /api/vault/promote — quarantine or confirm a note for vault promotion
  if (url.pathname === '/api/vault/promote' && method === 'POST') {
    if (AGENT_RESULTS_SECRET) {
      const auth = req.headers['authorization'];
      if (!auth || auth !== `Bearer ${AGENT_RESULTS_SECRET}`) {
        jsonResponse(res, 403, { error: 'Invalid authorization' });
        return;
      }
    }
    try {
      const action = url.searchParams.get('action');
      const body = JSON.parse(await readBody(req));

      if (action === 'quarantine') {
        const { noteId, title, refinedContent, suggestedTags } = body as {
          noteId: string;
          title: string;
          refinedContent: string;
          suggestedTags?: string[];
        };
        if (!noteId || !title || !refinedContent) {
          jsonResponse(res, 400, {
            error: 'noteId, title, and refinedContent required',
          });
          return;
        }

        if (!existsSync(QUARANTINE_DIR))
          mkdirSync(QUARANTINE_DIR, { recursive: true });

        const safeTitle = title
          .replace(/[^a-zA-Z0-9_\- ]/g, '')
          .trim()
          .replace(/\s+/g, '-');
        const filename = `${safeTitle}-${noteId.slice(0, 8)}.md`;
        const filePath = join(QUARANTINE_DIR, filename);

        const tags = (suggestedTags ?? []).join(', ');
        const created = new Date().toISOString();
        const fm = `---\nagent: promoter\nmission: ${noteId}\ntags: [${tags}]\ncreated: ${created}\nstatus: quarantined\n---\n`;
        writeFileSync(filePath, fm + '\n' + refinedContent, 'utf-8');

        log(`vault-promote: quarantined ${filename}`);
        jsonResponse(res, 201, { quarantinePath: filePath, filename });
      } else if (action === 'confirm') {
        const { quarantinePath, targetFolder } = body as {
          noteId: string;
          quarantinePath: string;
          targetFolder: string;
        };
        if (!quarantinePath || !targetFolder) {
          jsonResponse(res, 400, {
            error: 'quarantinePath and targetFolder required',
          });
          return;
        }

        if (!existsSync(quarantinePath)) {
          jsonResponse(res, 404, {
            error: `Quarantine file not found: ${quarantinePath}`,
          });
          return;
        }

        const content = readFileSync(quarantinePath, 'utf-8');
        const filename = quarantinePath.split('/').pop() ?? 'note.md';

        const destDir = join(HUMAN_VAULT_PATH, targetFolder);
        if (!existsSync(destDir)) mkdirSync(destDir, { recursive: true });

        const destPath = join(destDir, filename);
        writeFileSync(destPath, content, 'utf-8');
        unlinkSync(quarantinePath);
        invalidateVaultCache();

        log(`vault-promote: confirmed → ${targetFolder}/${filename}`);
        jsonResponse(res, 200, { humanVaultPath: destPath });
      } else if (action === 'refine') {
        const { noteId, title, content } = body as {
          noteId: string;
          title: string;
          content: string;
        };
        if (!noteId || !title || !content) {
          jsonResponse(res, 400, {
            error: 'noteId, title, and content required',
          });
          return;
        }

        // Build simple LLM caller for promoter
        const callLLMSimple = async (
          messages: Array<{ role: string; content: string }>,
        ): Promise<string> => {
          const model = resolveModel('local-reasoning');
          const llmRes = await fetch(
            `${PORTKEY_BASE_URL}/v1/chat/completions`,
            {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'x-portkey-provider': 'ollama',
                'x-portkey-custom-host': OLLAMA_HOST,
              },
              body: JSON.stringify({ model, messages, max_tokens: 4096 }),
            },
          );
          if (!llmRes.ok) throw new Error(`LLM call failed: ${llmRes.status}`);
          const data = (await llmRes.json()) as any;
          return data.choices?.[0]?.message?.content ?? '';
        };

        const auditLog = (entry: string) => log(`vault-refine: ${entry}`);

        const { refineNote } = await import(join(agentsPkgSrc, 'promoter.ts'));
        const proposal = await refineNote(
          title,
          content,
          callLLMSimple,
          auditLog,
        );

        log(`vault-promote: refined note "${title}"`);
        jsonResponse(res, 200, { success: true, proposal });
      } else {
        jsonResponse(res, 400, { error: `Unknown action: ${action}` });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`vault-promote error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // GET /api/vault/quarantine — list quarantined notes
  if (url.pathname === '/api/vault/quarantine' && method === 'GET') {
    if (AGENT_RESULTS_SECRET) {
      const auth = req.headers['authorization'];
      if (!auth || auth !== `Bearer ${AGENT_RESULTS_SECRET}`) {
        jsonResponse(res, 403, { error: 'Invalid authorization' });
        return;
      }
    }
    try {
      if (!existsSync(QUARANTINE_DIR)) {
        jsonResponse(res, 200, []);
        return;
      }
      const files = readdirSync(QUARANTINE_DIR)
        .filter((f: string) => f.endsWith('.md'))
        .map((f: string) => ({ filename: f, path: join(QUARANTINE_DIR, f) }));
      log(`vault-quarantine: listed ${files.length} files`);
      jsonResponse(res, 200, files);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`vault-quarantine list error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // DELETE /api/vault/quarantine — reject a quarantined note
  if (url.pathname === '/api/vault/quarantine' && method === 'DELETE') {
    if (AGENT_RESULTS_SECRET) {
      const auth = req.headers['authorization'];
      if (!auth || auth !== `Bearer ${AGENT_RESULTS_SECRET}`) {
        jsonResponse(res, 403, { error: 'Invalid authorization' });
        return;
      }
    }
    try {
      const path = url.searchParams.get('path');
      if (!path) {
        jsonResponse(res, 400, { error: 'path query param required' });
        return;
      }
      if (!existsSync(path)) {
        jsonResponse(res, 404, { error: `Quarantine file not found: ${path}` });
        return;
      }
      unlinkSync(path);
      log(`vault-quarantine: rejected ${path.split('/').pop()}`);
      jsonResponse(res, 200, { deleted: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`vault-quarantine delete error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // =========================================================================
  // Session Registration — agents register work sessions, get bundled context
  // =========================================================================

  // POST /api/sessions/register — agent registers work session, gets bundled context
  if (url.pathname === '/api/sessions/register' && method === 'POST') {
    try {
      const d1 = (await getSessionD1()) as any;
      const body = await readBody(req);
      const request = JSON.parse(body) as {
        agent: string;
        area?: { files?: string[]; topic?: string };
      };

      if (!request.agent) {
        jsonResponse(res, 400, { error: 'Missing required field: agent' });
        return;
      }
      if (!request.area?.files?.length && !request.area?.topic) {
        jsonResponse(res, 400, {
          error: 'At least one of area.files or area.topic required',
        });
        return;
      }

      // Clean up stale sessions
      await d1
        .prepare(
          "UPDATE agent_sessions SET closed_at = datetime('now'), phase_at_close = 'stale' WHERE closed_at IS NULL AND started_at < datetime('now', '-4 hours')",
        )
        .run();

      // Generate session ID
      const now = new Date();
      const ts = now.toISOString().replace(/[-:T]/g, '').slice(0, 14);
      const sessionId = `s_${ts}_${request.agent}`;

      // Register session
      const areaFiles = request.area?.files
        ? JSON.stringify(request.area.files)
        : null;
      const areaTopic = request.area?.topic ?? null;
      await d1
        .prepare(
          'INSERT INTO agent_sessions (id, agent, area_files, area_topic) VALUES (?, ?, ?, ?)',
        )
        .bind(sessionId, request.agent, areaFiles, areaTopic)
        .run();

      // Assemble context bundle
      // 1. Handoffs (via vault-query)
      const handoffs = await queryVaultGraph({
        folder: 'handoffs',
        limit: 5,
      });

      // 2. Active sessions (exclude self)
      const activeSessions = await d1
        .prepare(
          'SELECT id, agent, area_topic, started_at FROM agent_sessions WHERE closed_at IS NULL',
        )
        .all();
      const otherSessions = activeSessions.results.filter(
        (s: any) => s.id !== sessionId,
      );

      // 3. D1 context (errors, missions, activity) — individual queries via d1.query fallback
      const errors = await d1
        .prepare(
          'SELECT id, source, message, severity, timestamp FROM error_logs WHERE resolved = FALSE ORDER BY timestamp DESC LIMIT 10',
        )
        .all();

      const missions = await d1
        .prepare(
          "SELECT id, title, status, completed_at FROM missions WHERE status = 'completed' ORDER BY completed_at DESC LIMIT 5",
        )
        .all();

      const activity = await d1
        .prepare(
          'SELECT route, model, timestamp, tokens_in, tokens_out FROM request_logs ORDER BY timestamp DESC LIMIT 10',
        )
        .all();

      log(
        `sessions/register: ${request.agent} registered session ${sessionId} (topic: ${areaTopic})`,
      );

      jsonResponse(res, 200, {
        sessionId,
        context: {
          handoffs: handoffs.map((h: any) => ({
            file: h.name || h.path,
            date: h.date || null,
            summary: h.title || h.name || h.path,
          })),
          activeSessions: otherSessions,
          unresolvedErrors: errors.results,
          relatedMissions: missions.results,
          recentActivity: activity.results,
        },
      });
      return;
    } catch (err: any) {
      log(`sessions/register error: ${err.message}`);
      jsonResponse(res, 500, { error: err.message });
      return;
    }
  }

  // POST /api/sessions/close — agent closes work session
  if (url.pathname === '/api/sessions/close' && method === 'POST') {
    try {
      const d1 = (await getSessionD1()) as any;
      const body = await readBody(req);
      const request = JSON.parse(body) as {
        sessionId: string;
        phaseAtClose?: string;
        feedback?: string;
        failureContext?: string;
      };

      if (!request.sessionId) {
        jsonResponse(res, 400, { error: 'Missing required field: sessionId' });
        return;
      }

      await d1
        .prepare(
          "UPDATE agent_sessions SET closed_at = datetime('now'), phase_at_close = ? WHERE id = ?",
        )
        .bind(request.phaseAtClose ?? null, request.sessionId)
        .run();

      // Vault writes (non-blocking — log errors but don't fail the close)
      try {
        const session = (await d1
          .prepare(
            'SELECT agent, area_topic, started_at FROM agent_sessions WHERE id = ?',
          )
          .bind(request.sessionId)
          .first()) as any;

        const agent = session?.agent ?? 'unknown';
        const topic = session?.area_topic ?? 'unknown';
        const startedAt = session?.started_at ?? '';
        const now = new Date().toISOString();
        const dateStr = now.slice(0, 10);

        // Feedback capture
        if (request.feedback) {
          writeVaultFile(
            'feedback',
            `${dateStr}-${agent}-feedback.md`,
            {
              agent,
              session_id: request.sessionId,
              created: now,
              status: 'active',
            },
            request.feedback,
          );
        }

        // Postmortem generation (agent bailed early — not ship, not stale)
        if (
          request.phaseAtClose &&
          request.phaseAtClose !== 'ship' &&
          request.phaseAtClose !== 'stale'
        ) {
          const duration = startedAt
            ? `${Math.round((Date.now() - new Date(startedAt + 'Z').getTime()) / 60000)} minutes`
            : 'unknown';
          const postmortemBody = [
            '## Session Postmortem',
            '',
            `**Agent:** ${agent}`,
            `**Topic:** ${topic}`,
            `**Phase stopped at:** ${request.phaseAtClose}`,
            `**Duration:** ${duration}`,
            request.failureContext
              ? `**Failure context:** ${request.failureContext}`
              : '',
            `**Session ID:** ${request.sessionId}`,
          ]
            .filter(Boolean)
            .join('\n');

          writeVaultFile(
            'postmortems',
            `${dateStr}-${agent}-postmortem.md`,
            {
              agent,
              session_id: request.sessionId,
              phase_at_close: request.phaseAtClose,
              created: now,
              status: 'active',
            },
            postmortemBody,
          );
        }
      } catch (vaultErr: any) {
        log(
          `sessions/close vault write error (non-fatal): ${vaultErr.message}`,
        );
      }

      log(
        `sessions/close: ${request.sessionId} (phase: ${request.phaseAtClose ?? 'unknown'})`,
      );
      jsonResponse(res, 200, { closed: true });
      return;
    } catch (err: any) {
      log(`sessions/close error: ${err.message}`);
      jsonResponse(res, 500, { error: err.message });
      return;
    }
  }

  // =========================================================================
  // Metrics — session performance + latest report for Ops Dashboard
  // =========================================================================

  // GET /api/metrics/agent-performance — session metrics + latest performance report
  if (url.pathname === '/api/metrics/agent-performance' && method === 'GET') {
    try {
      const d1 = (await getSessionD1()) as any;
      const agent = url.searchParams.get('agent') || undefined;
      const days = parseInt(url.searchParams.get('days') || '30', 10);

      // Session duration
      const durationResult = agent
        ? await d1
            .prepare(
              "SELECT agent, COUNT(*) as session_count, AVG((julianday(closed_at) - julianday(started_at)) * 86400) as avg_seconds FROM agent_sessions WHERE closed_at IS NOT NULL AND started_at >= datetime('now', '-' || ? || ' days') AND agent = ? GROUP BY agent",
            )
            .bind(days, agent)
            .all()
        : await d1
            .prepare(
              "SELECT agent, COUNT(*) as session_count, AVG((julianday(closed_at) - julianday(started_at)) * 86400) as avg_seconds FROM agent_sessions WHERE closed_at IS NOT NULL AND started_at >= datetime('now', '-' || ? || ' days') GROUP BY agent",
            )
            .bind(days)
            .all();

      // Completion rates
      const completionResult = agent
        ? await d1
            .prepare(
              "SELECT agent, COUNT(*) as total, SUM(CASE WHEN phase_at_close = 'ship' THEN 1 ELSE 0 END) as shipped, SUM(CASE WHEN phase_at_close = 'stale' THEN 1 ELSE 0 END) as stale, SUM(CASE WHEN phase_at_close NOT IN ('ship', 'stale') THEN 1 ELSE 0 END) as abandoned FROM agent_sessions WHERE closed_at IS NOT NULL AND started_at >= datetime('now', '-' || ? || ' days') AND agent = ? GROUP BY agent",
            )
            .bind(days, agent)
            .all()
        : await d1
            .prepare(
              "SELECT agent, COUNT(*) as total, SUM(CASE WHEN phase_at_close = 'ship' THEN 1 ELSE 0 END) as shipped, SUM(CASE WHEN phase_at_close = 'stale' THEN 1 ELSE 0 END) as stale, SUM(CASE WHEN phase_at_close NOT IN ('ship', 'stale') THEN 1 ELSE 0 END) as abandoned FROM agent_sessions WHERE closed_at IS NOT NULL AND started_at >= datetime('now', '-' || ? || ' days') GROUP BY agent",
            )
            .bind(days)
            .all();

      // Phase distribution
      const distResult = await d1
        .prepare(
          "SELECT phase_at_close, COUNT(*) as count FROM agent_sessions WHERE closed_at IS NOT NULL AND started_at >= datetime('now', '-' || ? || ' days') GROUP BY phase_at_close",
        )
        .bind(days)
        .all();

      // Latest performance report
      const reportResult = await d1
        .prepare(
          'SELECT * FROM performance_reports ORDER BY created_at DESC LIMIT 1',
        )
        .all();
      const latestReport = reportResult.results[0] ?? null;

      jsonResponse(res, 200, {
        sessions: {
          duration: durationResult.results,
          completionRates: completionResult.results,
          phaseDistribution: distResult.results,
        },
        latestReport: latestReport
          ? {
              id: latestReport.id,
              periodStart: latestReport.period_start,
              periodEnd: latestReport.period_end,
              missionCount: latestReport.mission_count,
              reportData: latestReport.report_data
                ? JSON.parse(String(latestReport.report_data))
                : null,
            }
          : null,
      });
      return;
    } catch (err: any) {
      log(`metrics/agent-performance error: ${err.message}`);
      jsonResponse(res, 500, { error: err.message });
      return;
    }
  }

  // =========================================================================
  // Phase Serving — hub returns current lifecycle phase based on artifact state
  // =========================================================================

  // POST /api/workflow/phase — return lifecycle phase instructions
  if (url.pathname === '/api/workflow/phase' && method === 'POST') {
    try {
      // Check for ?phase= query param (bypass inference, return specific phase)
      const requestedPhase = url.searchParams.get('phase');

      // Parse lifecycle.md on every request (no cache — file is small, edits take effect immediately)
      const lifecyclePath = resolve(
        import.meta.dirname ?? __dirname,
        '..',
        '..',
        'alacrity_hub',
        'packages',
        'agents',
        'lifecycle.md',
      );
      const phases = parseLifecycle(lifecyclePath);

      if (requestedPhase) {
        const found = phases.find((p) => p.phase === requestedPhase);
        if (!found) {
          jsonResponse(res, 400, {
            error: `Unknown phase: ${requestedPhase}. Valid: ${phases.map((p) => p.phase).join(', ')}`,
          });
          return;
        }
        log(`workflow/phase: returning requested phase "${requestedPhase}"`);
        jsonResponse(res, 200, found);
        return;
      }

      // Infer phase from artifact state
      const body = await readBody(req);
      const request = JSON.parse(body) as { artifacts?: ArtifactState };
      const artifacts = request.artifacts ?? {};
      const result = inferPhase(artifacts, phases);

      log(`workflow/phase: inferred phase "${result.phase}"`);
      jsonResponse(res, 200, result);
      return;
    } catch (err: any) {
      log(`workflow/phase error: ${err.message}`);
      jsonResponse(res, 500, { error: err.message });
      return;
    }
  }

  // =========================================================================
  // Gate Validation — uniform enforcement for all agent contexts
  // =========================================================================

  // POST /api/workflow/validate — validate gate deliverables
  if (url.pathname === '/api/workflow/validate' && method === 'POST') {
    try {
      const body = await readBody(req);
      const request = JSON.parse(body) as {
        gate: 'assess' | 'pre-edit' | 'ship';
        content?: string;
        targetFile?: string;
        auditState?: ShipAuditState;
      };

      if (!request.gate) {
        jsonResponse(res, 400, { error: 'Missing required field: gate' });
        return;
      }

      if (request.gate === 'assess') {
        if (!request.content) {
          jsonResponse(res, 400, {
            error: 'Missing required field: content (assessment block text)',
          });
          return;
        }

        // Try to extract assessment block from larger content
        const block =
          extractAssessmentBlock(request.content) ?? request.content;
        const result = validateAssessmentBlock(block);

        log(
          `validate/assess: ${result.valid ? 'PASSED' : 'FAILED'} (${result.missingSections.length} missing, ${result.placeholderCount} placeholders)`,
        );
        jsonResponse(res, 200, {
          gate: 'assess',
          ...result,
        });
        return;
      }

      if (request.gate === 'pre-edit') {
        if (!request.content) {
          jsonResponse(res, 400, {
            error:
              'Missing required field: content (pre-edit confirmation text)',
          });
          return;
        }

        const result = validatePreEditConfirmation(
          request.content,
          request.targetFile,
        );

        log(
          `validate/pre-edit: ${result.valid ? 'PASSED' : 'FAILED'}${result.filePath ? ` (${result.filePath})` : ''} risk=${result.risk ?? 'unknown'}`,
        );
        jsonResponse(res, 200, {
          gate: 'pre-edit',
          ...result,
        });
        return;
      }

      if (request.gate === 'ship') {
        if (!request.auditState) {
          jsonResponse(res, 400, {
            error:
              'Missing required field: auditState (git state, tier, doc-impact, handoff)',
          });
          return;
        }

        const result = validateShipAudit(request.auditState);

        log(
          `validate/ship: ${result.valid ? 'PASSED' : 'FAILED'} scope=${result.scopeCheckPassed} complete=${result.completenessCheckPassed} hygiene=${result.hygieneCheckPassed}`,
        );
        jsonResponse(res, 200, {
          gate: 'ship',
          ...result,
        });
        return;
      }

      jsonResponse(res, 400, {
        error: `Unknown gate: ${request.gate}. Valid gates: assess, pre-edit, ship`,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log(`validate error: ${message}`);
      jsonResponse(res, 500, { error: message });
    }
    return;
  }

  // 404
  jsonResponse(res, 404, { error: 'Not found' });
});

/**
 * Replay any fallback result files that failed to POST on previous runs.
 * Removes successfully replayed files; leaves failures for the next attempt.
 */
async function replayFallbackFiles(): Promise<void> {
  if (!existsSync(FALLBACK_DIR)) return;
  const files = readdirSync(FALLBACK_DIR).filter((f) => f.endsWith('.json'));
  if (files.length === 0) return;
  log(`Found ${files.length} fallback file(s) to replay`);
  for (const file of files) {
    const filePath = join(FALLBACK_DIR, file);
    try {
      const raw = JSON.parse(readFileSync(filePath, 'utf-8'));
      const { _resultsUrl, _resultsAuth, ...payload } = raw;
      if (!_resultsUrl || !_resultsAuth) {
        log(`Fallback ${file}: missing _resultsUrl or _resultsAuth, skipping`);
        continue;
      }
      const res = await fetch(_resultsUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${_resultsAuth}`,
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(30_000),
      });
      if (res.ok) {
        unlinkSync(filePath);
        log(`Fallback ${file}: replayed successfully, deleted`);
      } else {
        log(
          `Fallback ${file}: replay failed (${res.status}), will retry next startup`,
        );
      }
    } catch (err) {
      log(`Fallback ${file}: replay error: ${err}`);
    }
  }
}

server.listen(PORT, HOST, () => {
  log(`Hub API server listening on http://${HOST}:${PORT}`);
  log(`Health:       GET  /api/health`);
  log(`LLM:          POST /api/llm/chat`);
  log(`RAG:          POST /api/rag/query`);
  log(`Agent:        POST /api/agent/run`);
  log(`Async:        POST /api/agent/run-async`);
  log(`Intake:       POST /api/mission-intake`);
  log(`Blast-radius: POST /api/tools/blast-radius`);
  log(`Dep-graph:    POST /api/tools/dep-graph`);
  log(`Doc-deps:     POST /api/tools/doc-deps`);
  log(`Doc-impact:   POST /api/tools/doc-impact`);
  log(`Doc-context:  POST /api/tools/doc-context`);
  log(`Vault-query:  POST /api/tools/vault-query`);
  log(`Vault-stats:  GET  /api/tools/vault-stats`);
  log(`Vault-refresh:POST /api/tools/vault-refresh`);
  log(`Vault-promote:POST /api/vault/promote`);
  log(`Vault-qlist:  GET  /api/vault/quarantine`);
  log(`Vault-qdelete:DELETE /api/vault/quarantine`);
  log(`Register:     POST /api/sessions/register`);
  log(`Metrics:      GET  /api/metrics/agent-performance`);
  log(`Close:        POST /api/sessions/close`);
  log(`Phase:        POST /api/workflow/phase`);
  log(`Validate:     POST /api/workflow/validate`);

  // Replay any pending fallback files after a short delay
  setTimeout(
    () =>
      replayFallbackFiles().catch((e) => log(`Fallback replay error: ${e}`)),
    5000,
  );
});

process.on('SIGTERM', () => {
  log('Shutting down...');
  server.close();
  process.exit(0);
});
process.on('SIGINT', () => {
  log('Shutting down...');
  server.close();
  process.exit(0);
});
