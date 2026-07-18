/**
 * CodeAct primary execution path for NanoClaw.
 *
 * When a paradigm's `protocol` is `"codeact"`, the Worker passes `protocol`
 * through to NanoClaw's /api/agent/run-async. runAsyncAgent branches here
 * instead of into the existing tool_calls runAgentLoop.
 *
 * Architecture:
 *  - Spawn a per-stage Python kernel via `@alacrity/codeact` KernelManager.
 *  - Stand up a Unix-socket BridgeServer in `protocolRole: 'primary'` mode —
 *    tool calls actually dispatch (not journal-only as in shadow mode).
 *  - The dispatch callback calls the same hub callback URL that the
 *    tool_calls path uses, so artifacts/mutators behave identically.
 *  - Drive the LLM-write-Python loop via `@alacrity/codeact` runStage.
 *  - Map the result into a shape runAsyncAgent can splice into resultPayload.
 *
 * Spec: docs/superpowers/specs/2026-05-12-codeact-phase-2-primary-design.md
 * Plan: docs/superpowers/plans/2026-05-12-codeact-phase-2-track-a.md
 */

import { randomUUID } from 'node:crypto';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { runStage, type LLMClient } from '@alacrity/codeact/runner';
import { KernelManager } from '@alacrity/codeact/kernel';
import { BridgeServer } from '@alacrity/codeact/bridge';
import { DEFAULT_LIMITS } from '@alacrity/codeact/types';
import { loadMutatorTools } from '@alacrity/codeact';

/**
 * Enumerate all tool names from JSON schema files in the schema dir. The
 * BridgeServer rejects requests for tools not in this set, so primary mode
 * must include BOTH mutator and read tools (shadow mode would only need
 * mutators since reads are dispatched live anyway).
 */
function loadAllTools(schemaDir: string): Set<string> {
  const tools = new Set<string>();
  for (const f of readdirSync(schemaDir).filter((n) => n.endsWith('.json'))) {
    try {
      const parsed = JSON.parse(readFileSync(join(schemaDir, f), 'utf8')) as {
        name?: string;
      };
      if (parsed.name) tools.add(parsed.name);
    } catch {
      // Skip unreadable / malformed schema files — defensive against ad-hoc additions.
    }
  }
  return tools;
}

export interface CodeActRunRequest {
  systemPrompt: string;
  userMessage: string;
  /** Directory holding tool JSON schemas (and the `codeact_wrappers` Python module). */
  schemaDir: string;
  /** Hub callback URL — same one tool_calls uses. */
  toolCallbackUrl: string;
  /** LLM adapter that wraps NanoClaw's callLLM into CodeAct's `complete()` shape. */
  llm: LLMClient;
}

export interface CodeActRunResult {
  /** Final LLM text after the last round (or stop). */
  content: string;
  tokensIn: number;
  tokensOut: number;
  toolCallCount: number;
  /** Wall-clock duration in ms. */
  duration: number;
  /** True iff the loop completed without a kernel/LLM error. */
  ok: boolean;
  /** Brief description of why the run failed, if ok=false. */
  errorKind?: string;
}

/**
 * macOS Unix socket path limit is 104 chars (UNIX_PATH_MAX). Use a short
 * prefix to stay well under it across all tmpdir layouts.
 */
function makeSockPath(): string {
  return join(tmpdir(), `ca-${randomUUID().slice(0, 8)}.sock`);
}

/**
 * Run a stage in primary CodeAct mode. Tool calls dispatch through the
 * provided hub callback URL — same semantics as the tool_calls path.
 */
export async function runAgentLoopCodeAct(
  req: CodeActRunRequest,
): Promise<CodeActRunResult> {
  const start = Date.now();
  let kernel: KernelManager | null = null;
  let bridge: BridgeServer | null = null;

  try {
    const mutators = loadMutatorTools(req.schemaDir);
    const allTools = loadAllTools(req.schemaDir);
    const sockPath = makeSockPath();

    bridge = new BridgeServer({
      socketPath: sockPath,
      tools: allTools,
      mutatorTools: mutators,
      protocolRole: 'primary',
      dispatch: async (toolName, args) => {
        try {
          const res = await fetch(req.toolCallbackUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tool: toolName, params: args }),
          });
          if (!res.ok) {
            const text = await res.text().catch(() => 'callback error');
            return {
              ok: false,
              error: {
                type: 'HttpError',
                message: `Tool error (${res.status}): ${text}`,
              },
            };
          }
          const body = (await res.json()) as { result: unknown };
          return { ok: true, result: body.result };
        } catch (err) {
          return {
            ok: false,
            error: {
              type: 'FetchError',
              message: err instanceof Error ? err.message : String(err),
            },
          };
        }
      },
    });
    await bridge.start();

    kernel = new KernelManager({
      stagePath: tmpdir(),
      bridgeSocketPath: sockPath,
      wrappersDir: req.schemaDir,
    });
    await kernel.start();

    const outcome = await runStage({
      llm: req.llm,
      kernel,
      systemPrompt: req.systemPrompt,
      userPrompt: req.userMessage,
      limits: DEFAULT_LIMITS,
    });

    return {
      content: outcome.finalText,
      tokensIn: outcome.tokensIn,
      tokensOut: outcome.tokensOut,
      toolCallCount: outcome.runCount,
      duration: Date.now() - start,
      ok: true,
    };
  } catch (err) {
    return {
      content: '',
      tokensIn: 0,
      tokensOut: 0,
      toolCallCount: 0,
      duration: Date.now() - start,
      ok: false,
      errorKind: err instanceof Error ? err.constructor.name : 'Unknown',
    };
  } finally {
    try {
      if (kernel) await kernel.shutdown();
    } catch {
      kernel?.kill();
    }
    try {
      if (bridge) await bridge.close();
    } catch {
      /* ignore */
    }
  }
}
