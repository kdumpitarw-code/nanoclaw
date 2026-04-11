/**
 * Git invocation helpers for NanoClaw.
 *
 * All git calls in NanoClaw flow through `runGit`. This enforces two invariants:
 *
 *   1. Command injection safety. Builder agents produce commit messages that
 *      contain arbitrary text (checkpoint summaries, handoff payloads). Passing
 *      those through a shell string is a command-injection vector. `execFile`
 *      with an args array passes each argument as a literal, so a summary
 *      containing backticks, semicolons, or `$(...)` is harmless.
 *
 *   2. Working-directory isolation. The spec requires that git operations target
 *      a specific worktree path regardless of the NanoClaw process's own cwd.
 *      Rather than relying on `process.chdir` (which is a global side effect),
 *      `runGit` always passes `-C <workingDir>` as the first argument to git.
 *      Git's `-C` flag changes git's effective working directory for that one
 *      invocation, without touching the process.
 *
 * See checkpoint-layer-v2 design spec §11 ("Git Integration") for the full
 * rationale.
 */

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

export interface RunGitResult {
  stdout: string;
  stderr: string;
}

/**
 * Run a git command inside the given working directory.
 *
 * The git binary is resolved from `PATH`. The `workingDir` is passed via git's
 * own `-C` flag rather than via `cwd`, so the call is safe to make from any
 * NanoClaw context without touching `process.chdir`.
 *
 * Throws on non-zero exit (inherited from `execFile`'s promisified form — the
 * rejection value carries `stdout`, `stderr`, `code`, and `signal`).
 */
export async function runGit(
  workingDir: string,
  args: string[],
): Promise<RunGitResult> {
  const { stdout, stderr } = await execFileAsync('git', [
    '-C',
    workingDir,
    ...args,
  ]);
  return { stdout, stderr };
}
