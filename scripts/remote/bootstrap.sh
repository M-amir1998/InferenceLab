#!/usr/bin/env bash
# Prepare a rented GPU host for InferenceLab campaigns. Run it ON the GPU host, once:
#   bash bootstrap.sh
# Idempotent. It installs nothing system-wide except uv (user-local) and never stores
# credentials in this repository. It does NOT create, resize or terminate machines.
set -euo pipefail

say() { printf '[bootstrap] %s\n' "$*"; }
fail() { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found: install the NVIDIA driver first"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
command -v docker >/dev/null || fail "docker not found: install Docker Engine first"
docker info --format '{{json .Runtimes}}' | grep -q nvidia \
  || say "WARNING: nvidia container runtime not listed; install nvidia-container-toolkit"

if ! command -v uv >/dev/null; then
  say "installing uv (user-local) from https://astral.sh/uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

mkdir -p "$HOME/inferencelab"/{cfg,results,dist} "$HOME/.cache/huggingface"
mkdir -p "$HOME/.inferencelab"
chmod 700 "$HOME/.inferencelab"
if [ ! -f "$HOME/.inferencelab/env" ]; then
  umask 077
  cat > "$HOME/.inferencelab/env" <<'ENV'
# Engine secrets for docker --env-file. Fill in on this host only; never commit.
# HF_TOKEN=
ENV
  say "created $HOME/.inferencelab/env (empty template, mode 600)"
fi
df -h "$HOME" | tail -1
say "done. Next, from the laptop: inferencelab remote doctor configs/hosts/<host>.local.yaml"
