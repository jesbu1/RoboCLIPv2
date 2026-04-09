#!/usr/bin/env bash
# Generic cleanup job that cancels a target Slurm job if it is still present.

#SBATCH --job-name=stop_job
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

TARGET_JOB_ID="${TARGET_JOB_ID:-}"

if [ -z "$TARGET_JOB_ID" ]; then
  echo "ERROR: TARGET_JOB_ID is empty"
  exit 1
fi

echo "Attempting to stop job ${TARGET_JOB_ID}"
if squeue -h -j "$TARGET_JOB_ID" | grep -q .; then
  scancel "$TARGET_JOB_ID" || true
  echo "Issued scancel for ${TARGET_JOB_ID}"
else
  echo "Target job ${TARGET_JOB_ID} is no longer in queue"
fi
