#!/usr/bin/env bash
# Submit 18 selected Robometer gamma1/scale100 online reruns:
#   9 env/seed combinations, each rerun twice.
# Uses 6 shared servers, each serving 3 online tasks.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROBOMETER_PROJECT_DIR="${ROBOMETER_PROJECT_DIR:-/project2/biyik_1165/haobaizh/rewind_robometer_valuemodel}"
MAX_FRAMES="${MAX_FRAMES:-4}"

cd "$PROJECT_DIR"
mkdir -p logs

if [ ! -f "${PROJECT_DIR}/test_scripts/test_iql.py" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"
  exit 1
fi
if [ ! -d "${PROJECT_DIR}/conda_envs/rewind_nochunk" ]; then
  echo "ERROR: rewind_nochunk conda env not found under ${PROJECT_DIR}/conda_envs/rewind_nochunk"
  exit 1
fi
if [ ! -d "${ROBOMETER_PROJECT_DIR}" ]; then
  echo "ERROR: ROBOMETER_PROJECT_DIR not found: ${ROBOMETER_PROJECT_DIR}"
  exit 1
fi
if [ ! -d "${ROBOMETER_PROJECT_DIR}/conda_envs/robometer" ]; then
  echo "ERROR: Robometer conda env not found: ${ROBOMETER_PROJECT_DIR}/conda_envs/robometer"
  exit 1
fi
if [ ! -f "${ROBOMETER_PROJECT_DIR}/robometer_server.py" ]; then
  echo "ERROR: robometer_server.py not found under ${ROBOMETER_PROJECT_DIR}"
  exit 1
fi

REQUIRED_INPUTS=(
  "logs/offline_robometer_diff_gamma1_scale100_10critics_seed0/last_offline.zip"
  "logs/offline_robometer_diff_gamma1_scale100_10critics_seed32/last_offline.zip"
  "logs/offline_robometer_diff_gamma1_scale100_10critics_seed42/last_offline.zip"
  "datasets/metaworld_labeled_robometer_diff_gamma1.h5"
)

missing_inputs=()
for input_path in "${REQUIRED_INPUTS[@]}"; do
  if [ ! -f "$input_path" ]; then
    missing_inputs+=("$input_path")
  fi
done

if [ "${#missing_inputs[@]}" -gt 0 ]; then
  echo "ERROR: Missing required gamma1/scale100 rerun inputs:"
  printf '  %s\n' "${missing_inputs[@]}"
  exit 1
fi

GROUP_IDS=(61 62 63 64 65 66)
SERVER_JOB_NAMES=(
  "robo_srv_d1s100_g61"
  "robo_srv_d1s100_g62"
  "robo_srv_d1s100_g63"
  "robo_srv_d1s100_g64"
  "robo_srv_d1s100_g65"
  "robo_srv_d1s100_g66"
)
ONLINE_JOB_NAMES=(
  "robo_on_d1s100_g61"
  "robo_on_d1s100_g62"
  "robo_on_d1s100_g63"
  "robo_on_d1s100_g64"
  "robo_on_d1s100_g65"
  "robo_on_d1s100_g66"
)
SERVER_LABELS=(
  "gamma1 scale100 reruns: sweep0-r1 window32-r1 faucet32-r1"
  "gamma1 scale100 reruns: reach0-r1 door0-r1 window42-r1"
  "gamma1 scale100 reruns: door42-r1 sweep32-r1 faucet42-r1"
  "gamma1 scale100 reruns: sweep0-r2 window32-r2 faucet32-r2"
  "gamma1 scale100 reruns: reach0-r2 door0-r2 window42-r2"
  "gamma1 scale100 reruns: door42-r2 sweep32-r2 faucet42-r2"
)
INFO_FILES=(
  "${PROJECT_DIR}/logs/robometer_server_d1s100_group61_info.txt"
  "${PROJECT_DIR}/logs/robometer_server_d1s100_group62_info.txt"
  "${PROJECT_DIR}/logs/robometer_server_d1s100_group63_info.txt"
  "${PROJECT_DIR}/logs/robometer_server_d1s100_group64_info.txt"
  "${PROJECT_DIR}/logs/robometer_server_d1s100_group65_info.txt"
  "${PROJECT_DIR}/logs/robometer_server_d1s100_group66_info.txt"
)
SERVER_PORTS=(8080 8081 8082 8083 8084 8085)
FALLBACK_STARTS=(18572 18578 18584 18590 18596 18602)
FALLBACK_ENDS=(18577 18583 18589 18595 18601 18607)
STARTUP_DELAYS=(0 120 240 360 480 600)
TASK_SPECS=(
  "sweep-into-v2:0:1|window-close-v2:32:1|faucet-close-v2:32:1"
  "reach-wall-v2:0:1|door-lock-v2:0:1|window-close-v2:42:1"
  "door-lock-v2:42:1|sweep-into-v2:32:1|faucet-close-v2:42:1"
  "sweep-into-v2:0:2|window-close-v2:32:2|faucet-close-v2:32:2"
  "reach-wall-v2:0:2|door-lock-v2:0:2|window-close-v2:42:2"
  "door-lock-v2:42:2|sweep-into-v2:32:2|faucet-close-v2:42:2"
)

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
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",ROBOMETER_PROJECT_DIR="${ROBOMETER_PROJECT_DIR}",GROUP_ID="${group_id}",SERVER_LABEL="${SERVER_LABELS[$idx]}",INFO_FILE="${info_file}",SERVER_PORT="${SERVER_PORTS[$idx]}",FALLBACK_START="${FALLBACK_STARTS[$idx]}",FALLBACK_END="${FALLBACK_ENDS[$idx]}",STARTUP_DELAY_SEC="${STARTUP_DELAYS[$idx]}",STARTUP_MAX_ATTEMPTS=6,STARTUP_BACKOFF_BASE_SEC=120 \
    scripts/robometer_server_mixed_group_generic.sbatch)
  SERVER_JOB_IDS+=("${server_job_id}")

  online_job_id=$(sbatch --parsable \
    --job-name="${ONLINE_JOB_NAMES[$idx]}" \
    --output="logs/${ONLINE_JOB_NAMES[$idx]}_%A_%a.out" \
    --error="logs/${ONLINE_JOB_NAMES[$idx]}_%A_%a.err" \
    --dependency="after:${server_job_id}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",GROUP_ID="${group_id}",INFO_FILE="${info_file}",EXPECTED_SERVER_JOB_ID="${server_job_id}",TASK_SPECS="${TASK_SPECS[$idx]}",MAX_FRAMES="${MAX_FRAMES}" \
    scripts/robometer_online_diff_gamma1_scale100_selected_3tasks.sbatch)
  ONLINE_JOB_IDS+=("${online_job_id}")

  stop_job_id=$(sbatch --parsable \
    --job-name="stop_robo_d1s100_g${group_id}" \
    --output="logs/stop_robo_d1s100_g${group_id}_%j.out" \
    --error="logs/stop_robo_d1s100_g${group_id}_%j.err" \
    --dependency="afterany:${online_job_id}_0:${online_job_id}_1:${online_job_id}_2" \
    --export=ALL,TARGET_JOB_ID="${server_job_id}" \
    scripts/stop_slurm_job_by_id.sbatch)
  STOP_JOB_IDS+=("${stop_job_id}")
done

MANIFEST="logs/robometer_selected_diff_gamma1_scale100_18tasks_2reruns_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "manifest=${PROJECT_DIR}/${MANIFEST}"
  echo "variant=diff_gamma1_scale100_base_reward"
  echo "max_frames=${MAX_FRAMES}"
  echo "tasks_total=18"
  echo "servers_total=6"
  echo "tasks_per_server=3"
  for idx in "${!GROUP_IDS[@]}"; do
    echo "task_group${GROUP_IDS[$idx]}=${TASK_SPECS[$idx]}"
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
