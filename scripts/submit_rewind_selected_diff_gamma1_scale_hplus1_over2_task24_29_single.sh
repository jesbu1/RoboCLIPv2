#!/usr/bin/env bash
# Submit TASK_INDEX=24..29 as six independent non-array jobs.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
HORIZON="${HORIZON:-128}"
REPEAT_COUNT="${REPEAT_COUNT:-3}"

cd "$PROJECT_DIR"

echo "project_dir=${PROJECT_DIR}"
echo "horizon=${HORIZON}"
echo "repeat_count=${REPEAT_COUNT}"
echo "submitting_task_indices=24 25 26 27 28 29"

for task_index in 24 25 26 27 28 29; do
  job_id=$(sbatch --parsable \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",HORIZON="$HORIZON",REPEAT_COUNT="$REPEAT_COUNT",TASK_INDEX="$task_index" \
    scripts/rewind_online_diff_gamma1_scale_hplus1_over2_selected_single_task.sbatch)
  echo "task_index=${task_index} job_id=${job_id}"
done
