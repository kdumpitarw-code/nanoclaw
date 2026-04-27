#!/bin/sh
# NanoClaw — git-bundle backup to AlacrityHub vault.
# Captures the entire repo (all refs + history) into a single timestamped
# bundle file under ~/Vaults/AlacrityHub/code/nanoclaw-backups/.
#
# Restore (from any vault host):
#   git clone <bundle-path> NanoClaw-restored
#
# Schedule (weekly): see ~/Library/LaunchAgents/com.alacrityhub.nanoclaw-backup.plist
# Manual run: bash scripts/backup-to-vault.sh

set -e

NANOCLAW_REPO="${NANOCLAW_REPO:-$HOME/Vibe Sphere/NanoClaw}"
VAULT_BACKUP_DIR="${VAULT_BACKUP_DIR:-$HOME/Vaults/AlacrityHub/code/nanoclaw-backups}"
KEEP_LAST="${KEEP_LAST:-8}"  # keep ~2 months at weekly cadence

LOG() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2; }

if [ ! -d "$NANOCLAW_REPO/.git" ]; then
  LOG "ERROR: NanoClaw repo not found at $NANOCLAW_REPO"
  exit 1
fi

mkdir -p "$VAULT_BACKUP_DIR"

TIMESTAMP=$(date +%Y-%m-%d)
BUNDLE_PATH="$VAULT_BACKUP_DIR/nanoclaw-$TIMESTAMP.bundle"

cd "$NANOCLAW_REPO"

# Verify repo state — bundle requires consistent history
if ! git rev-parse --verify HEAD > /dev/null 2>&1; then
  LOG "ERROR: NanoClaw HEAD is not resolvable"
  exit 1
fi

LOG "creating bundle: $BUNDLE_PATH"
git bundle create "$BUNDLE_PATH" --all
git bundle verify "$BUNDLE_PATH" > /dev/null

SIZE=$(du -h "$BUNDLE_PATH" | awk '{print $1}')
HEAD_SHA=$(git rev-parse --short HEAD)
LOG "bundle ok: ${SIZE}, HEAD=$HEAD_SHA"

# Rotate: keep newest $KEEP_LAST bundles
cd "$VAULT_BACKUP_DIR"
COUNT=$(ls -1 nanoclaw-*.bundle 2>/dev/null | wc -l | tr -d ' ')
if [ "$COUNT" -gt "$KEEP_LAST" ]; then
  TO_REMOVE=$((COUNT - KEEP_LAST))
  ls -1t nanoclaw-*.bundle | tail -n "$TO_REMOVE" | while read -r old; do
    LOG "rotating out: $old"
    rm -f "$old"
  done
fi

LOG "done; $(ls -1 nanoclaw-*.bundle | wc -l | tr -d ' ') bundle(s) retained"
exit 0
