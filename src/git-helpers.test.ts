import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock child_process.execFile. vitest's vi.mock is hoisted, so we use a
// module-scoped mock fn and reference it from the factory.
const mockExecFile = vi.fn();
vi.mock('node:child_process', () => ({
  execFile: (
    file: string,
    args: string[],
    callback: (
      err: Error | null,
      result: { stdout: string; stderr: string } | null,
    ) => void,
  ) => mockExecFile(file, args, callback),
}));

import { runGit } from './git-helpers.js';

beforeEach(() => {
  vi.clearAllMocks();
});

describe('runGit', () => {
  it('invokes git with -C workingDir prepended to the args array', async () => {
    mockExecFile.mockImplementationOnce((_file, _args, cb) => {
      cb(null, { stdout: 'on branch main\n', stderr: '' });
    });

    const result = await runGit('/tmp/my-worktree', ['status', '--porcelain']);

    expect(mockExecFile).toHaveBeenCalledTimes(1);
    const [file, args] = mockExecFile.mock.calls[0];
    expect(file).toBe('git');
    expect(args).toEqual(['-C', '/tmp/my-worktree', 'status', '--porcelain']);
    expect(result).toEqual({ stdout: 'on branch main\n', stderr: '' });
  });

  it('passes agent-provided commit messages as literal args (no shell interpretation)', async () => {
    mockExecFile.mockImplementationOnce((_file, _args, cb) => {
      cb(null, { stdout: '', stderr: '' });
    });

    // A summary containing shell metacharacters that would be catastrophic
    // under shell-string invocation.
    const hostileSummary = 'fix: handle `rm -rf /`; $(whoami) & echo pwned';

    await runGit('/tmp/worktree', [
      'commit',
      '-m',
      `checkpoint(dev_build/2): ${hostileSummary}`,
      '--allow-empty',
    ]);

    const [, args] = mockExecFile.mock.calls[0];
    // The hostile summary must appear as a single intact argument — not split,
    // not interpolated, not interpreted.
    expect(args).toContain(`checkpoint(dev_build/2): ${hostileSummary}`);
    expect(args).toContain('--allow-empty');
  });

  it('rejects when git exits non-zero (e.g. dirty worktree, missing ref)', async () => {
    const gitErr = Object.assign(new Error('git exited with code 128'), {
      code: 128,
      stdout: '',
      stderr: 'fatal: not a git repository\n',
    });
    mockExecFile.mockImplementationOnce((_file, _args, cb) => {
      cb(gitErr, null);
    });

    await expect(runGit('/not/a/repo', ['status'])).rejects.toThrow(
      'git exited with code 128',
    );
  });
});
