#!/usr/bin/env bash
# Deploy ET API to Hugging Face Spaces
# Usage: HF_TOKEN=hf_xxx ./deploy-hf.sh

set -euo pipefail

HF_USER="${HF_USER:-devan-p}"
HF_SPACE="${HF_SPACE:-et-api}"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN environment variable}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Temp dir OUTSIDE the source tree so cp doesn't recurse into itself
DEPLOY_DIR="$(mktemp -d -t et-hf-deploy-XXXXXX)"

echo "==> Deploying ET API to https://huggingface.co/spaces/$HF_USER/$HF_SPACE"
echo "==> Using temp dir: $DEPLOY_DIR"

trap 'rm -rf "$DEPLOY_DIR"' EXIT

# Clone the HF Space (empty repo)
echo "==> Cloning HF Space repo..."
git clone "https://$HF_USER:$HF_TOKEN@huggingface.co/spaces/$HF_USER/$HF_SPACE" "$DEPLOY_DIR/space"
cd "$DEPLOY_DIR/space"

# Copy the engine and API code (using rsync to exclude junk)
echo "==> Copying engine and API source..."
rm -rf packages
mkdir -p packages/engine packages/web/api

rsync -a \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.ruff_cache' \
    --exclude='.mypy_cache' \
    --exclude='*.egg-info' \
    "$REPO_ROOT/packages/engine/" packages/engine/

rsync -a \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.ruff_cache' \
    --exclude='.mypy_cache' \
    --exclude='*.egg-info' \
    --exclude='.hf-deploy' \
    --exclude='deploy-hf.sh' \
    "$REPO_ROOT/packages/web/api/" packages/web/api/

# Move Dockerfile to root (HF requirement)
cp packages/web/api/Dockerfile ./Dockerfile

# Create the HF Space README with YAML frontmatter
cat > README.md << 'README_EOF'
---
title: ET API
emoji: 🔥
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Diagnose why your GPU is idle from a PyTorch Profiler trace
---

# ET API

FastAPI service wrapping the ET diagnostic engine.

POST `/analyze` with a PyTorch Profiler trace `.json` file to get a verdict.

GET `/health` for a liveness check.

Source: https://github.com/devan-p/ET
README_EOF

# Add and push
git add -A
git -c user.email="deploy@et.local" -c user.name="ET Deploy" commit -m "deploy: $(date -u +%Y-%m-%dT%H:%M:%SZ)" || echo "Nothing to commit"
git push

echo ""
echo "==> Deployed. Building..."
echo "==> Watch build: https://huggingface.co/spaces/$HF_USER/$HF_SPACE"
echo "==> Live URL:    https://$HF_USER-$HF_SPACE.hf.space"
