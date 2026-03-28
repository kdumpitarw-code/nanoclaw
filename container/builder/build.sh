#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="nanoclaw-builder"
TAG="${1:-latest}"

echo "Building NanoClaw builder container image..."
container build -t "${IMAGE_NAME}:${TAG}" .

echo ""
echo "Build complete: ${IMAGE_NAME}:${TAG}"
echo ""
echo "Test with:"
echo "  echo '{\"action\":\"build-check\",\"params\":{}}' | container run -i --mount type=bind,source=\$(pwd),target=/workspace/project ${IMAGE_NAME}:${TAG}"
