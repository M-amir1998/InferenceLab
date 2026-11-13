#!/usr/bin/env bash
# Secret scan used by Jenkins and before every push.
# 1) detect-secrets against the committed baseline (fails on any new finding)
# 2) gitleaks over the git history, when it is installed on the agent
set -euo pipefail
git ls-files -z | xargs -0 uv run --frozen detect-secrets-hook --baseline .secrets.baseline
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks git --no-banner --redact .
else
  echo "gitleaks not installed on this agent: history scan skipped (detect-secrets ran)"
fi
echo "secret scan clean"
