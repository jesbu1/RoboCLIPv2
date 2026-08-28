#!/usr/bin/env bash
# Submit two seed42 Robometer rerun groups:
#   1) faucet-close-v2 baseline_base_reward, 3 repeats on one server
#   2) door-lock-v2 diff_gamma099_scaled_base_reward, 3 repeats on one server

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

GROUP_IDS=(23 24)
SERVER_LABELS=("seed42 faucet baseline 3reruns" "seed42 doorlock diff099 3reruns")
SERVER_JOB_NAMES=("robo_srv_s42_fb3" "robo_srv_s42_dl3")
ONLINE_JOB_NAMES=("robo_on_s42_fb3" "robo_on_s42_dl3")
INFO_FILES=(
  "${PROJECT_DIR}/logs/robometer_server_seed42_faucet_baseline_3reruns_info.txt"
  "${PROJECT_DIR}/logs/robometer_server_seed42_doorlock_diff099_3reruns_info.txt"
)
SERVER_PORTS=(8058 8059)
FALLBACK_STARTS=(18464 18470)
FALLBACK_ENDS=(18469 18475)
STARTUP_DELAYS=(0 120)
ENV_IDS=("faucet-close-v2" "door-lock-v2")
VARIANT_TAGS=("baseline_base_reward" "diff_gamma099_scaled_base_reward")

if [ "${SKIP_INPUT_PREFLIGHT:-0}" != "1" ]; then
  REQUIRED_INPUTS=(
    "logs/offline_robometer_baseline_bonus0_seed42/last_offline.zip"
    "logs/offline_robometer_diff_gamma099_bonus0_scaled_seed42/last_offline.zip"
    "datasets/metaworld_labeled_robometer.h5"
    "datasets/metaworld_labeled_robometer_diff_gamma099.h5"
  )
  missing_inputs=()
  for input_path in "${REQUIRED_INPUTS[@]}"; do
    if [ ! -f "$input_path" ]; then
      missing_inputs+=("$input_path")
    fi
  done

  if [ "${#missing_inputs[@]}" -gt 0 ]; then
    echo "ERROR: Missing required seed42 rerun inputs:"
    printf '  %s\n' "${missing_inputs[@]}"
    echo "Create the missing inputs first, or set SKIP_INPUT_PREFLIGHT=1 if you intentionally want to submit anyway."
    exit 1
  fi
fi

SERVER_JOB_IDS=()
ONLINE_JOB_IDS=()
STOP_JOB_IDS=()

for idx in "${!GROUP_IDS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  info_file="${INFO_FILES[$idx]}"
  rm -f "$info_file" "${info_file}.tmp"

  server_job_id=$(sbatch --parsable \
    --job-name="${SERVER_JOB_NAMES[$idx]}" \
    --output="logs/${SERVER_JOB_NAMES[$idx]}_%j.out" \
    --error="logs/${SERVER_JOB_NAMES[$idx]}_%j.err" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",GROUP_ID="${group_id}",SERVER_LABEL="${SERVER_LABELS[$idx]}",INFO_FILE="${info_file}",SERVER_PORT="${SERVER_PORTS[$idx]}",FALLBACK_START="${FALLBACK_STARTS[$idx]}",FALLBACK_END="${FALLBACK_ENDS[$idx]}",STARTUP_DELAY_SEC="${STARTUP_DELAYS[$idx]}",STARTUP_MAX_ATTEMPTS=6,STARTUP_BACKOFF_BASE_SEC=120 \
    scripts/robometer_server_mixed_group_generic.sbatch)
  SERVER_JOB_IDS+=("${server_job_id}")

  online_job_id=$(sbatch --parsable \
    --job-name="${ONLINE_JOB_NAMES[$idx]}" \
    --output="logs/${ONLINE_JOB_NAMES[$idx]}_%A_%a.out" \
    --error="logs/${ONLINE_JOB_NAMES[$idx]}_%A_%a.err" \
    --dependency="after:${server_job_id}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",ENV_ID="${ENV_IDS[$idx]}",VARIANT_TAG="${VARIANT_TAGS[$idx]}",INFO_FILE="${info_file}",EXPECTED_SERVER_JOB_ID="${server_job_id}" \
    scripts/robometer_online_seed42_same_task_3reps.sbatch)
  ONLINE_JOB_IDS+=("${online_job_id}")

  stop_job_id=$(sbatch --parsable \
    --job-name="stop_s42_rerun_g${group_id}" \
    --output="logs/stop_s42_rerun_g${group_id}_%j.out" \
    --error="logs/stop_s42_rerun_g${group_id}_%j.err" \
    --dependency="afterany:${online_job_id}_*" \
    --export=ALL,TARGET_JOB_ID="${server_job_id}" \
    scripts/stop_slurm_job_by_id.sbatch)
  STOP_JOB_IDS+=("${stop_job_id}")
done

MANIFEST="logs/robometer_seed42_two_task_3reruns_$(date +%Y%m%d_%H%M%S).txt"
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
