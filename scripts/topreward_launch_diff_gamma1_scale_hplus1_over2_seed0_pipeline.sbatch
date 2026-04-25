#!/usr/bin/env bash
# Wait for any TOPReward server to become healthy, then submit:
# diff gamma1 label -> offline scale=(HORIZON+1)/2 seed0 -> online 8 envs on 4 servers (2 envs per server).

#SBATCH --job-name=top_lau_d1_h
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
TOPREWARD_DIR="${TOPREWARD_DIR:-/scratch1/haobaizh/rewind_topreward}"
SERVER_JOB_IDS="${SERVER_JOB_IDS:?SERVER_JOB_IDS must be provided}"
SERVER_INFO_FILES="${SERVER_INFO_FILES:?SERVER_INFO_FILES must be provided}"
SUBMIT_MANIFEST="${SUBMIT_MANIFEST:-${PROJECT_DIR}/logs/topreward_diff_gamma1_scale_hplus1_over2_seed0_manifest.txt}"
HORIZON="${HORIZON:-128}"
MAX_FRAMES="${MAX_FRAMES:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$SUBMIT_MANIFEST")"

IFS='|' read -r -a SERVER_IDS <<< "$SERVER_JOB_IDS"
IFS='|' read -r -a INFO_FILES <<< "$SERVER_INFO_FILES"

if [ "${#SERVER_IDS[@]}" -ne 4 ] || [ "${#INFO_FILES[@]}" -ne 4 ]; then
  echo "ERROR: Expected 4 SERVER_JOB_IDS and 4 SERVER_INFO_FILES"
  exit 1
fi

SERVER_IDS_CSV="$(IFS=,; echo "${SERVER_IDS[*]}")"
ALLOWED_SERVER_IDS="|$(IFS='|'; echo "${SERVER_IDS[*]}")|"

read_server_url() {
  local info_file="$1"
  local info_line server_job_id server_url
  info_line=$(tr -d '\r\n' < "$info_file")
  if [ -z "$info_line" ]; then
    return 1
  fi
  if [[ "$info_line" == http* ]]; then
    printf '%s\n' "$info_line"
    return 0
  fi
  server_job_id="${info_line%% *}"
  server_url="${info_line#* }"
  if [[ "$ALLOWED_SERVER_IDS" != *"|${server_job_id}|"* ]]; then
    return 1
  fi
  if [ -z "$server_url" ] || [[ "$server_url" != http* ]]; then
    return 1
  fi
  printf '%s\n' "$server_url"
}

FIRST_READY=""
FIRST_INFO_FILE=""
FIRST_SERVER_JOB_ID=""
for _ in $(seq 1 34560); do
  for idx in "${!INFO_FILES[@]}"; do
    info_file="${INFO_FILES[$idx]}"
    if [ -s "$info_file" ] && candidate_url="$(read_server_url "$info_file")"; then
      if curl -fsS "${candidate_url}/health" >/dev/null 2>&1; then
        FIRST_READY="$candidate_url"
        FIRST_INFO_FILE="$info_file"
        FIRST_SERVER_JOB_ID="${SERVER_IDS[$idx]}"
        break 2
      fi
    fi
  done

  if ! squeue -h -j "$SERVER_IDS_CSV" | grep -q .; then
    echo "ERROR: All server jobs left queue before any healthy server became available"
    exit 1
  fi
  sleep 5
done

if [ -z "$FIRST_READY" ]; then
  echo "ERROR: No healthy TOPReward server found"
  exit 1
fi

GROUP_IDS=(91 92 93 94)
TASK_GROUPS=(
  "window-close-v2|reach-wall-v2"
  "faucet-close-v2|coffee-button-v2"
  "button-press-wall-v2|door-lock-v2"
  "handle-press-side-v2|sweep-into-v2"
)

DIFF_REWARD_SCALE=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", (h + 1) / 2 }')
REWARD_DIVISOR=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", 2 / (h + 1) }')
TAG="diff_gamma1_scale_hplus1_over2_H${HORIZON}"

{
  echo "launcher_first_ready_server=${FIRST_READY}"
  echo "launcher_first_ready_info_file=${FIRST_INFO_FILE}"
  echo "launcher_first_ready_server_job=${FIRST_SERVER_JOB_ID}"
  echo "variant=${TAG}_base_reward"
  echo "label_output=${TOPREWARD_DIR}/datasets/metaworld_topreward_labeled_diff_gamma1.h5"
  echo "offline_training_steps=100000"
  echo "online_seed=0"
  echo "success_bonus=0"
  echo "diff_gamma=1.0"
  echo "horizon=${HORIZON}"
  echo "diff_reward_scale=(HORIZON+1)/2=${DIFF_REWARD_SCALE}"
  echo "offline_reward_divisor=2/(HORIZON+1)=${REWARD_DIVISOR}"
  echo "max_frames=${MAX_FRAMES}"
  echo "online_base_reward=true"
  echo "base_reward_value=-1.0"
  echo "servers_total=4"
  echo "tasks_total=8"
  echo "tasks_per_server=2"
} | tee -a "$SUBMIT_MANIFEST"

LABEL_LOCK="${TOPREWARD_DIR}/logs/topreward_label_diff_gamma1_${FIRST_SERVER_JOB_ID}.lock"
LABEL_ID=$(sbatch --parsable \
  --dependency="after:${FIRST_SERVER_JOB_ID}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",INFO_FILE="$FIRST_INFO_FILE",EXPECTED_SERVER_JOB_ID="$FIRST_SERVER_JOB_ID",MAX_FRAMES="$MAX_FRAMES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES",LOCK_PATH="$LABEL_LOCK" \
  scripts/topreward_label_diff_gamma1.sbatch)
echo "label_diff_gamma1=${LABEL_ID}" | tee -a "$SUBMIT_MANIFEST"

OFFLINE_ID=$(sbatch --parsable \
  --dependency="afterok:${LABEL_ID}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",INFO_FILE="$FIRST_INFO_FILE",EXPECTED_SERVER_JOB_ID="$FIRST_SERVER_JOB_ID",HORIZON="$HORIZON",MAX_FRAMES="$MAX_FRAMES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES" \
  scripts/topreward_offline_diff_gamma1_scale_hplus1_over2_bonus0_seed0.sbatch)
echo "offline_diff_gamma1_scale_hplus1_over2_seed0=${OFFLINE_ID}" | tee -a "$SUBMIT_MANIFEST"

for idx in "${!GROUP_IDS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  server_job_id="${SERVER_IDS[$idx]}"
  info_file="${INFO_FILES[$idx]}"
  task_envs="${TASK_GROUPS[$idx]}"
  online_lock="${TOPREWARD_DIR}/logs/topreward_online_group${group_id}.lock"

  online_id=$(sbatch --parsable \
    --dependency="after:${server_job_id},afterok:${OFFLINE_ID}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",INFO_FILE="$info_file",EXPECTED_SERVER_JOB_ID="$server_job_id",GROUP_ID="$group_id",TASK_ENVS="$task_envs",HORIZON="$HORIZON",MAX_FRAMES="$MAX_FRAMES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES",LOCK_PATH="$online_lock" \
    scripts/topreward_online_diff_gamma1_scale_hplus1_over2_base_reward_seed0_group2tasks.sbatch)
  echo "online_group${group_id}=${online_id} tasks=${task_envs}" | tee -a "$SUBMIT_MANIFEST"
  echo "stop_group${group_id}=(disabled_no_auto_cancel)" | tee -a "$SUBMIT_MANIFEST"
done
