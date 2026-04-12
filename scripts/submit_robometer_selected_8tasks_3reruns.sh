#!/usr/bin/env bash
# Submit 8 Robometer online rerun groups, each repeated 3 times on its own server.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

GROUP_IDS=(25 26 27 28 29 30 31 32)
SERVER_JOB_NAMES=(
  "robo_srv_r25"
  "robo_srv_r26"
  "robo_srv_r27"
  "robo_srv_r28"
  "robo_srv_r29"
  "robo_srv_r30"
  "robo_srv_r31"
  "robo_srv_r32"
)
ONLINE_JOB_NAMES=(
  "robo_on_r25"
  "robo_on_r26"
  "robo_on_r27"
  "robo_on_r28"
  "robo_on_r29"
  "robo_on_r30"
  "robo_on_r31"
  "robo_on_r32"
)
SERVER_LABELS=(
  "rerun window-close seed32 baseline"
  "rerun window-close seed42 baseline"
  "rerun sweep-into seed42 baseline"
  "rerun door-lock seed42 baseline"
  "rerun sweep-into seed32 diff099"
  "rerun sweep-into seed42 diff099"
  "rerun sweep-into seed0 diff0999"
  "rerun sweep-into seed42 diff0999"
)
INFO_FILES=(
  "${PROJECT_DIR}/logs/robometer_server_rerun_group25_info.txt"
  "${PROJECT_DIR}/logs/robometer_server_rerun_group26_info.txt"
  "${PROJECT_DIR}/logs/robometer_server_rerun_group27_info.txt"
  "${PROJECT_DIR}/logs/robometer_server_rerun_group28_info.txt"
  "${PROJECT_DIR}/logs/robometer_server_rerun_group29_info.txt"
  "${PROJECT_DIR}/logs/robometer_server_rerun_group30_info.txt"
  "${PROJECT_DIR}/logs/robometer_server_rerun_group31_info.txt"
  "${PROJECT_DIR}/logs/robometer_server_rerun_group32_info.txt"
)
SERVER_PORTS=(8060 8061 8062 8063 8064 8065 8066 8067)
FALLBACK_STARTS=(18476 18482 18488 18494 18500 18506 18512 18518)
FALLBACK_ENDS=(18481 18487 18493 18499 18505 18511 18517 18523)
STARTUP_DELAYS=(0 120 240 360 480 600 720 840)
ENV_IDS=(
  "window-close-v2"
  "window-close-v2"
  "sweep-into-v2"
  "door-lock-v2"
  "sweep-into-v2"
  "sweep-into-v2"
  "sweep-into-v2"
  "sweep-into-v2"
)
SEEDS=(32 42 42 42 32 42 0 42)
VARIANT_TAGS=(
  "baseline_base_reward"
  "baseline_base_reward"
  "baseline_base_reward"
  "baseline_base_reward"
  "diff_gamma099_scaled_base_reward"
  "diff_gamma099_scaled_base_reward"
  "diff_gamma0999_scaled_base_reward"
  "diff_gamma0999_scaled_base_reward"
)

if [ "${SKIP_INPUT_PREFLIGHT:-0}" != "1" ]; then
  REQUIRED_INPUTS=(
    "logs/offline_robometer_baseline_bonus0_seed32/last_offline.zip"
    "logs/offline_robometer_baseline_bonus0_seed42/last_offline.zip"
    "logs/offline_robometer_diff_gamma099_bonus0_scaled_seed32/last_offline.zip"
    "logs/offline_robometer_diff_gamma099_bonus0_scaled_seed42/last_offline.zip"
    "logs/offline_robometer_diff_gamma0999_bonus0_scaled_seed0/last_offline.zip"
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
    echo "ERROR: Missing required rerun inputs:"
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
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",ENV_ID="${ENV_IDS[$idx]}",SEED="${SEEDS[$idx]}",VARIANT_TAG="${VARIANT_TAGS[$idx]}",INFO_FILE="${info_file}",EXPECTED_SERVER_JOB_ID="${server_job_id}" \
    scripts/robometer_online_same_task_3reps.sbatch)
  ONLINE_JOB_IDS+=("${online_job_id}")

  stop_job_id=$(sbatch --parsable \
    --job-name="stop_rerun_g${group_id}" \
    --output="logs/stop_rerun_g${group_id}_%j.out" \
    --error="logs/stop_rerun_g${group_id}_%j.err" \
    --dependency="afterany:${online_job_id}_*" \
    --export=ALL,TARGET_JOB_ID="${server_job_id}" \
    scripts/stop_slurm_job_by_id.sbatch)
  STOP_JOB_IDS+=("${stop_job_id}")
done

MANIFEST="logs/robometer_selected_8tasks_3reruns_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "manifest=${PROJECT_DIR}/${MANIFEST}"
  for idx in "${!GROUP_IDS[@]}"; do
    group_id="${GROUP_IDS[$idx]}"
    echo "task_group${group_id}=${ENV_IDS[$idx]} seed${SEEDS[$idx]} ${VARIANT_TAGS[$idx]}"
  done
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
