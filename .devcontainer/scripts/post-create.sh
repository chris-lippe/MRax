#!/usr/bin/env bash
set -euo pipefail
cd /workspace
mkdir -p data tmp
python -m pip install -e '.[all,dev,docs,notebooks]'
if [[ "${1:-}" == "--gpu" ]]; then
    python -m pip install 'jax[cuda12]>=0.4.35,<0.11'
fi
cd frontend
npm ci --no-audit --no-fund
npm run build
