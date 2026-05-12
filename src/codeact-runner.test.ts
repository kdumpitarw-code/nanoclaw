import { describe, it, expect } from 'vitest';
import { join } from 'node:path';
import { createServer, type Server } from 'node:http';
import { runAgentLoopCodeAct } from './codeact-runner.js';
import type { LLMClient } from '@alacrity/codeact/runner';

// Schema dir lives in the sibling alacrity_hub workspace.
const SCHEMA_DIR = join(import.meta.dirname, '../../alacrity_hub/packages/agents/tools');

function stubLLM(scriptedBlocks: string[]): LLMClient {
  let i = 0;
  return {
    async complete() {
      const block = scriptedBlocks[i++] ?? '';
      // After the scripted blocks, return empty text to signal stop (runStage
      // exits when parsePythonFences finds zero blocks).
      const wrapped = block ? '```python\n' + block + '\n```\n' : '';
      return {
        text: wrapped,
        tokensIn: block ? 100 : 0,
        tokensOut: block ? 50 : 0,
      };
    },
  };
}

describe('runAgentLoopCodeAct', () => {
  it('runs a Python block via a real kernel and returns populated telemetry', async () => {
    // Listen on a port that won't actually be hit (no tools called in this test)
    const result = await runAgentLoopCodeAct({
      systemPrompt: 'You are a coding agent.',
      userMessage: 'Print ok.',
      schemaDir: SCHEMA_DIR,
      toolCallbackUrl: 'http://127.0.0.1:0/unused',
      llm: stubLLM(['print("ok")']),
    });

    expect(result.ok).toBe(true);
    expect(result.tokensIn).toBe(100);
    expect(result.tokensOut).toBe(50);
    expect(result.toolCallCount).toBe(1);
    expect(result.duration).toBeGreaterThan(0);
  }, 30_000);

  it('dispatches mutator tool calls to the toolCallbackUrl', async () => {
    // Tiny HTTP server that captures the tool dispatch.
    let captured: { tool: string; params: unknown } | null = null;
    const httpServer = createServer((req, res) => {
      let body = '';
      req.on('data', (c) => (body += c.toString()));
      req.on('end', () => {
        captured = JSON.parse(body);
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ result: { wrote: true } }));
      });
    });
    await new Promise<void>((r) => httpServer.listen(0, '127.0.0.1', r));
    const port = (httpServer.address() as ReturnType<Server['address']> & { port: number }).port;

    try {
      // spec-write is a known mutator tool in the schema dir.
      const result = await runAgentLoopCodeAct({
        systemPrompt: 'You are a coding agent.',
        userMessage: 'Call spec-write.',
        schemaDir: SCHEMA_DIR,
        toolCallbackUrl: `http://127.0.0.1:${port}/cb`,
        llm: stubLLM([
          'result = spec_write(missionId="m1", title="t", summary="s", content="x")\nprint(result)',
        ]),
      });

      expect(result.ok).toBe(true);
      expect(captured).not.toBeNull();
      expect((captured as unknown as { tool: string }).tool).toBe('spec-write');
    } finally {
      httpServer.closeAllConnections();
      await new Promise<void>((r) => httpServer.close(() => r()));
    }
  }, 30_000);
});
