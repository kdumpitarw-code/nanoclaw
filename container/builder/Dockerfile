# NanoClaw Builder Container
# Lightweight build environment for Developer/QA agents
# Receives JSON commands via stdin, returns JSON via stdout

FROM node:22-slim

# Install build dependencies (no Chromium, no browser tools)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pnpm globally
RUN corepack enable && corepack prepare pnpm@latest --activate

# Create workspace
WORKDIR /workspace

# Copy entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create pnpm store mount point
RUN mkdir -p /workspace/pnpm-store

# Configure git (needed for git add inside container)
RUN git config --global user.name "builder-agent" \
    && git config --global user.email "builder@alacrityhub.ca" \
    && git config --global --add safe.directory /workspace/project

ENTRYPOINT ["/app/entrypoint.sh"]
