#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/workspace/.cache/torch}
export PORT=${PORT:-8000}

mkdir -p "${HF_HOME}" "${TORCH_HOME}"

echo "=== STARTUP: listing /app ==="
ls -la /app || true
echo "=== STARTUP: listing /workspace (mounted volume) ==="
ls -la /workspace || true

echo "Python: $(python -V)"
echo "HF_HOME=${HF_HOME}"
echo "PORT=${PORT}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== nvidia-smi ==="
  nvidia-smi || true
fi

exec uvicorn hearme.main:app --host 0.0.0.0 --port "${PORT}" --workers 1
