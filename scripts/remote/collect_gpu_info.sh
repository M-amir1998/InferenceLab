#!/usr/bin/env bash
# Print GPU, driver, CUDA and container-runtime versions as JSON (for run metadata).
set -euo pipefail
gpus=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/"/\\"/g' | awk '{printf "%s\"%s\"", (NR>1?",":""), $0}')
cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)
docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unavailable")
printf '{"gpus":[%s],"cuda":"%s","docker":"%s"}\n' "$gpus" "$cuda" "$docker_version"
