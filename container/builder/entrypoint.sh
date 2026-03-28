#!/bin/bash
set -e

# Read JSON command from stdin into env var for Node.js access
export INPUT=$(cat)
ACTION=$(echo "$INPUT" | node -e "process.stdin.on('data',d=>{const j=JSON.parse(d);process.stdout.write(j.action||'')})")

cd /workspace/project

case "$ACTION" in
  code-write)
    # Use Node.js to write the file (avoids echo escape issues and preserves content exactly)
    node -e "
      const input = JSON.parse(process.env.INPUT);
      const p = input.params;
      const fs = require('fs');
      const path = require('path');
      const fullPath = path.join('/workspace/project', p.path);
      fs.mkdirSync(path.dirname(fullPath), { recursive: true });
      fs.writeFileSync(fullPath, p.content);
      const bytes = Buffer.byteLength(p.content);
      require('child_process').execSync('git add ' + JSON.stringify(p.path), { stdio: 'pipe' });
      process.stdout.write(JSON.stringify({ success: true, data: { bytesWritten: bytes, path: p.path } }));
    " 2>/dev/null
    ;;

  build-check)
    cd apps/hub
    BUILD_OUTPUT=$(pnpm install --frozen-lockfile --store-dir /workspace/pnpm-store 2>&1 && npx next build 2>&1) && BUILD_STATUS="pass" || BUILD_STATUS="fail"
    TRUNCATED=$(echo "$BUILD_OUTPUT" | tail -c 4096)
    ESCAPED=$(echo "$TRUNCATED" | node -e "let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(d)))")
    echo "{\"success\":true,\"data\":{\"status\":\"$BUILD_STATUS\",\"output\":$ESCAPED}}"
    ;;

  test-run)
    cd apps/hub
    HAS_TEST=$(node -e "const p=require('./package.json');process.stdout.write(p.scripts&&p.scripts.test?'yes':'no')")
    if [ "$HAS_TEST" = "no" ]; then
      echo "{\"success\":true,\"data\":{\"status\":\"no_tests\"}}"
      exit 0
    fi
    TEST_OUTPUT=$(pnpm test 2>&1) && TEST_STATUS="pass" || TEST_STATUS="fail"
    TRUNCATED=$(echo "$TEST_OUTPUT" | tail -c 4096)
    ESCAPED=$(echo "$TRUNCATED" | node -e "let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(d)))")
    echo "{\"success\":true,\"data\":{\"status\":\"$TEST_STATUS\",\"output\":$ESCAPED}}"
    ;;

  *)
    echo "{\"success\":false,\"error\":\"Unknown action: $ACTION\"}"
    exit 1
    ;;
esac
