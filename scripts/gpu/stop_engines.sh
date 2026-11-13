#!/usr/bin/env bash
# Stop and remove every InferenceLab engine container on this host.
# Containers only: the machine itself is never shut down by InferenceLab tooling.
set -euo pipefail
ids=$(docker ps -aq --filter "name=^inferencelab-")
if [ -n "$ids" ]; then
  docker stop -t 30 $ids >/dev/null
  docker rm $ids >/dev/null
  echo "removed: $ids"
else
  echo "no inferencelab-* containers"
fi
