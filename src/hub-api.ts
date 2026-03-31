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
import { existsSync, mkdirSync, writeFileSync, readFileSync } from 'fs';
import { join } from 'path';

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
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://127.0.0.1:11434';
const PORTKEY_BASE_URL =
  process.env.PORTKEY_BASE_URL || 'http://127.0.0.1:8787';
const PORTKEY_API_KEY = process.env.PORTKEY_API_KEY || '';
const CHROMADB_HOST = process.env.CHROMADB_HOST || 'http://127.0.0.1:8000';
const EMBED_MODEL = 'nomic-embed-text';
const MAX_TOOL_ROUNDS = 10;

// CF Access validation for cloud-first async endpoint
const CF_ACCESS_EXPECTED_ID = process.env.CF_ACCESS_EXPECTED_ID || '';
const CF_ACCESS_EXPECTED_SECRET = process.env.CF_ACCESS_EXPECTED_SECRET || '';
const AGENT_RESULTS_SECRET = process.env.AGENT_RESULTS_SECRET || '';

// Path to the agents package on this machine
const AGENTS_BASE = join(
  process.env.HOME || '/root',
  'Vibe Sphere',
  'alacrity_hub',
  'packages',
  'agents',
);

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

interface AgentRunRequest {
  model: string;
  system_prompt: string;
  user_message: string;
  tools: ToolDefinition[];
  max_tokens: number;
  portkey: PortkeyConfig;
  container?: Record<string, unknown>;
  tool_callback_url: string;
}

interface AgentRunResponse {
  content: string;
  model: string;
  tokensUsed: number;
  duration: number;
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
}

const AGENT_CONFIGS: Record<
  string,
  {
    prompt: string;
    modelKey: string;
    tools: string[];
    maxTokens: number;
    oversight: string;
    maxToolRounds: number;
  }
> = {
  pm: {
    prompt: 'prompts/pm.md',
    modelKey: 'local-reasoning',
    tools: [
      'mission-read',
      'vault-read',
      'spec-write',
      'mission-write',
      'd1-query',
      'audit-query',
      'bookmark-read',
      'project-read',
      'vault-list',
      'doc-read',
      'handoff-write',
      'notify-user',
      'schedule-create',
      'template-create',
      'project-write',
      'git-log',
      'kv-read',
      'pdf-generate',
      'docx-generate',
      'slides-generate',
      'spreadsheet-generate',
      'chart-render',
      'gantt-render',
      'html-report',
    ],
    oversight: 'hitl',
    maxTokens: 8192,
    maxToolRounds: 15,
  },
  architect: {
    prompt: 'prompts/architect.md',
    modelKey: 'local-reasoning',
    tools: [
      'mission-read',
      'vault-read',
      'arch-write',
      'mission-write',
      'd1-query',
      'audit-query',
      'file-read',
      'file-list',
      'grep-search',
      'doc-read',
      'agents-md-read',
      'git-log',
      'git-diff',
      'notify-user',
      'handoff-write',
      'vault-list',
      'bookmark-read',
      'project-read',
      'diagram-render',
    ],
    oversight: 'hitl',
    maxTokens: 8192,
    maxToolRounds: 15,
  },
  developer: {
    prompt: 'prompts/developer.md',
    modelKey: 'local-coding',
    tools: [
      'mission-read',
      'vault-read',
      'code-write',
      'build-check',
      'test-run',
      'lint-check',
      'type-check',
      'migration-write',
      'migration-check',
      'file-read',
      'file-list',
      'grep-search',
      'git-log',
      'git-diff',
      'git-branch-list',
      'd1-query',
      'doc-read',
      'agents-md-read',
      'notify-user',
      'handoff-write',
      'mission-write',
      'artifact-write',
    ],
    oversight: 'hitl',
    maxTokens: 8192,
    maxToolRounds: 20,
  },
  qa: {
    prompt: 'prompts/qa.md',
    modelKey: 'local-reasoning',
    tools: [
      'mission-read',
      'vault-read',
      'file-read',
      'file-list',
      'grep-search',
      'git-log',
      'git-diff',
      'd1-query',
      'api-test',
      'screenshot-capture',
      'bundle-size',
      'service-health',
      'create-defect',
      'notify-user',
      'mission-write',
      'handoff-write',
    ],
    oversight: 'hitl',
    maxTokens: 8192,
    maxToolRounds: 15,
  },
  devops: {
    prompt: 'prompts/devops.md',
    modelKey: 'local-background',
    tools: [
      'deploy-staging',
      'deploy-production',
      'rollback-deploy',
      'service-health',
      'logs-read',
      'system-info',
      'mission-read',
      'd1-query',
      'notify-user',
      'mission-write',
      'git-log',
      'git-diff',
    ],
    oversight: 'hitl',
    maxTokens: 4096,
    maxToolRounds: 10,
  },
  'tech-writer': {
    prompt: 'prompts/tech-writer.md',
    modelKey: 'local-reasoning',
    tools: [
      'mission-read',
      'vault-read',
      'vault-write-agent',
      'file-read',
      'file-list',
      'grep-search',
      'doc-read',
      'agents-md-read',
      'prompt-read',
      'artifact-write',
      'mission-write',
      'notify-user',
      'git-log',
      'handoff-write',
      'vault-list',
      'd1-query',
    ],
    oversight: 'hitl',
    maxTokens: 8192,
    maxToolRounds: 15,
  },
};

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
 * Call the LLM via Portkey (OpenAI-compatible chat completions).
 */
async function callLLM(
  portkey: PortkeyConfig,
  model: string,
  messages: ChatMessage[],
  tools: ToolDefinition[],
  maxTokens: number,
): Promise<ChatCompletionResponse> {
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

  return (await res.json()) as ChatCompletionResponse;
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

/**
 * Run the agent loop: call LLM, execute tool calls, repeat until done.
 */
async function runAgentLoop(
  req: AgentRunRequest,
  maxToolRounds?: number,
): Promise<AgentRunResponse> {
  const maxRounds = maxToolRounds ?? MAX_TOOL_ROUNDS;
  const startTime = Date.now();
  let totalTokens = 0;

  const messages: ChatMessage[] = [
    { role: 'system', content: req.system_prompt },
    { role: 'user', content: req.user_message },
  ];

  for (let round = 0; round < maxRounds; round++) {
    const resolvedModel = resolveModel(req.model);
    log(
      `Round ${round + 1}: calling LLM (${messages.length} messages, model=${req.model} → ${resolvedModel})`,
    );

    const completion = await callLLM(
      req.portkey,
      req.model,
      messages,
      req.tools,
      req.max_tokens,
    );

    if (completion.usage) {
      totalTokens += completion.usage.total_tokens;
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

      const result = await executeToolCallback(
        req.tool_callback_url,
        toolName,
        params,
      );

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

function loadAgentPrompt(promptPath: string): string {
  return readFileSync(join(AGENTS_BASE, promptPath), 'utf-8');
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

async function postResults(
  url: string,
  authToken: string,
  payload: unknown,
): Promise<void> {
  const delays = [1000, 2000, 4000];
  for (let attempt = 0; attempt <= delays.length; attempt++) {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${authToken}`,
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(10_000),
      });
      if (res.ok) return;
      log(`Results POST failed (${res.status}), attempt ${attempt + 1}`);
    } catch (err) {
      log(`Results POST error, attempt ${attempt + 1}: ${err}`);
    }
    if (attempt < delays.length) {
      await new Promise((r) => setTimeout(r, delays[attempt]));
    }
  }
  // All retries failed — write to fallback file
  const fallbackDir = join(
    process.env.HOME || '/root',
    'Vibe Sphere',
    'agent-results-fallback',
  );
  mkdirSync(fallbackDir, { recursive: true });
  const p = payload as Record<string, unknown>;
  const filename = `${Date.now()}-${(p.missionId as string) || 'unknown'}.json`;
  writeFileSync(join(fallbackDir, filename), JSON.stringify(payload, null, 2));
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
  } = req;

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
    const systemPrompt = loadAgentPrompt(agentConfig.prompt);
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
      const thModule = await import(join(agentsPkgSrc, 'tool-handlers.js'));
      createToolHandlers = thModule.createToolHandlers;
      const d1Module = await import(join(agentsPkgSrc, 'd1-proxy-adapter.js'));
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
    const callbackPort = 4101 + Math.floor(Math.random() * 100);
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

    try {
      // Run the agent loop
      const result = await runAgentLoop(
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
          tool_callback_url: `http://127.0.0.1:${callbackPort}/tool-callback`,
        },
        agentConfig.maxToolRounds,
      );

      const duration = Date.now() - startTime;

      // Build result payload
      const resultPayload: Record<string, unknown> = {
        missionId,
        pipelineStage,
        status: 'success',
        auditEntries: [
          {
            agentName: agent,
            actionType: 'agent_complete',
            actionDetail: `Agent completed in ${duration}ms`,
            target: 'builder-pipeline',
            result: 'success',
            metadata: {
              tokensUsed: result.tokensUsed,
              model: result.model,
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
              whats_changing: 'Agent proposal requires review',
              why_now: 'Triggered from builder-pipeline',
              long_term_benefit: 'Maintains quality through human review',
              risk_if_approved: 'Review the proposal details before approving',
            };

        resultPayload.proposal = {
          agentName: agent,
          proposalType: PROPOSAL_TYPES[agent] || 'note-refinement',
          payload: JSON.stringify(parsed),
          briefing: JSON.stringify(briefing),
        };
      }

      await postResults(resultsUrl, resultsAuth, resultPayload);
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
      // Validate CF Access headers if configured
      if (CF_ACCESS_EXPECTED_ID) {
        const clientId = req.headers['cf-access-client-id'];
        const clientSecret = req.headers['cf-access-client-secret'];
        if (
          clientId !== CF_ACCESS_EXPECTED_ID ||
          clientSecret !== CF_ACCESS_EXPECTED_SECRET
        ) {
          jsonResponse(res, 403, { error: 'Invalid CF Access credentials' });
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

      // Return 202 immediately, run agent in background
      jsonResponse(res, 202, { status: 'accepted', agent, missionId });

      // Spawn background execution
      setImmediate(() => {
        runAsyncAgent({
          agent,
          missionId,
          pipelineStage,
          resultsUrl,
          resultsAuth,
          d1ProxyUrl: d1ProxyUrl || '',
          d1ProxyAuth: d1ProxyAuth || '',
        }).catch((err) =>
          log(`Async agent error for ${agent}/${missionId}: ${err}`),
        );
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

  // 404
  jsonResponse(res, 404, { error: 'Not found' });
});

server.listen(PORT, HOST, () => {
  log(`Hub API server listening on http://${HOST}:${PORT}`);
  log(`Health: GET /api/health`);
  log(`LLM:    POST /api/llm/chat`);
  log(`RAG:    POST /api/rag/query`);
  log(`Agent:  POST /api/agent/run`);
  log(`Async:  POST /api/agent/run-async`);
  log(`Intake: POST /api/mission-intake`);
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
