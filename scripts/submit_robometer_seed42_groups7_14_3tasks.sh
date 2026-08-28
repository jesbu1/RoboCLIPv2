#!/usr/bin/env bash
# Submit seed42 Robometer groups 7-14: 8 servers and 24 online tasks.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

GROUP_IDS=(7 8 9 10 11 12 13 14)
SERVER_JOB_IDS=()
ONLINE_JOB_IDS=()
STOP_JOB_IDS=()

MANIFEST="logs/robometer_seed42_groups7_14_$(date +%Y%m%d_%H%M%S).txt"

if [ "${SKIP_INPUT_PREFLIGHT:-0}" != "1" ]; then
  REQUIRED_INPUTS=(
    "logs/offline_robometer_baseline_bonus0_seed42/last_offline.zip"
    "logs/offline_robometer_diff_gamma099_bonus0_scaled_seed42/last_offline.zip"
    "logs/offline_robometer_diff_gamma0999_bonus0_scaled_seed42/last_offline.zip"
    "datasets/metaworld_labeled_robometer.h5"
    "datasets/metaworld_labeled_robometer_diff_gamma099.h5"
    "datasets/metaworld_labeled_robometer_diff_gamma0999.h5"
  )
  missing_inputs=()
  for input_path in "${REQUIRED_INPUTS[@]}"; do
    if [ ! -f "$input_path" ]; then
      missing_inputs+=("$input_path")
    fi
  done

  if [ "${#missing_inputs[@]}" -gt 0 ]; then
    echo "ERROR: Missing required seed42 inputs:"
    printf '  %s\n' "${missing_inputs[@]}"
    echo "Create the missing inputs first, or set SKIP_INPUT_PREFLIGHT=1 if you intentionally want to submit anyway."
    exit 1
  fi
fi

for group_id in "${GROUP_IDS[@]}"; do
  info_file="${PROJECT_DIR}/logs/robometer_server_seed42_group${group_id}_info.txt"
  rm -f "$info_file" "${info_file}.tmp"

  server_job_id=$(sbatch --parsable "scripts/robometer_server_seed42_group${group_id}.sbatch")
  SERVER_JOB_IDS+=("${server_job_id}")

  online_job_id=$(sbatch --parsable \
    --dependency="after:${server_job_id}" \
    --export=ALL,EXPECTED_SERVER_JOB_ID="${server_job_id}" \
    "scripts/robometer_online_seed42_group${group_id}_3tasks.sbatch")
  ONLINE_JOB_IDS+=("${online_job_id}")

  stop_job_id=$(sbatch --parsable \
    --job-name="stop_s42_g${group_id}" \
    --output="logs/stop_s42_g${group_id}_%j.out" \
    --error="logs/stop_s42_g${group_id}_%j.err" \
    --dependency="afterany:${online_job_id}_*" \
    --export=ALL,TARGET_JOB_ID="${server_job_id}" \
    scripts/stop_slurm_job_by_id.sbatch)
  STOP_JOB_IDS+=("${stop_job_id}")
done

{
  echo "manifest=${PROJECT_DIR}/${MANIFEST}"
  for idx in "${!GROUP_IDS[@]}"; do
    echo "server_group${GROUP_IDS[$idx]}=${SERVER_JOB_IDS[$idx]}"
  done
  for idx in "${!GROUP_IDS[@]}"; do
    echo "online_group${GROUP_IDS[$idx]}=${ONLINE_JOB_IDS[$idx]}"
  done
  for idx in "${!GROUP_IDS[@]}"; do
    echo "stop_group${GROUP_IDS[$idx]}=${STOP_JOB_IDS[$idx]}"
  done
} | tee "${PROJECT_DIR}/${MANIFEST}"
