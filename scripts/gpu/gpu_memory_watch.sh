#!/usr/bin/env bash
# Sample GPU memory/utilisation once per second into a CSV until interrupted.
#   bash gpu_memory_watch.sh gpu_mem.csv
set -euo pipefail
out=${1:-gpu_mem.csv}
echo "timestamp,index,memory_used_mib,memory_total_mib,utilization_pct" > "$out"
exec nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits -lms 1000 >> "$out"
