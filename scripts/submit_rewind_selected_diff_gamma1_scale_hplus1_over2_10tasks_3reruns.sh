#!/usr/bin/env bash
# Submit 3 reruns for 10 selected ReWiND H128 online jobs.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
HORIZON="${HORIZON:-128}"
REPEAT_COUNT="${REPEAT_COUNT:-3}"

cd "$PROJECT_DIR"
mkdir -p logs

MANIFEST="${PROJECT_DIR}/logs/rewind_selected_diff_gamma1_scale_hplus1_over2_10tasks_3reruns_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "manifest=${MANIFEST}"
  echo "project_dir=${PROJECT_DIR}"
  echo "variant=diff_gamma1_scale_hplus1_over2_base_reward"
  echo "horizon=${HORIZON}"
  echo "repeat_count=${REPEAT_COUNT}"
  echo "tasks_total=30"
  echo "selected_combos=10"
  echo "selected_1=reach-wall-v2:seed0"
  echo "selected_2=reach-wall-v2:seed42"
  echo "selected_3=faucet-close-v2:seed0"
  echo "selected_4=faucet-close-v2:seed32"
  echo "selected_5=sweep-into-v2:seed0"
  echo "selected_6=sweep-into-v2:seed32"
  echo "selected_7=sweep-into-v2:seed42"
  echo "selected_8=door-lock-v2:seed0"
  echo "selected_9=door-lock-v2:seed32"
  echo "selected_10=door-lock-v2:seed42"
} | tee "$MANIFEST"

JOB_ID=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",HORIZON="$HORIZON",REPEAT_COUNT="$REPEAT_COUNT" \
  scripts/rewind_online_diff_gamma1_scale_hplus1_over2_selected_10tasks_3reruns.sbatch)

echo "job_id=${JOB_ID}" | tee -a "$MANIFEST"
