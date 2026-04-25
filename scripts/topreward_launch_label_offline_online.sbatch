#!/usr/bin/env bash
# Wait for any TOPReward server to become healthy, then submit:
# TOPReward labels -> no-action-chunk 3-seed offline -> no-action-chunk seed0 online.

#SBATCH --job-name=top_launch
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
SUBMIT_MANIFEST="${SUBMIT_MANIFEST:-${PROJECT_DIR}/logs/topreward_baseline_diff999_manifest.txt}"
SUCCESS_BONUS="${SUCCESS_BONUS:-0.0}"
DIFF_GAMMA="${DIFF_GAMMA:-0.999}"
DIFF_REWARD_SCALE="${DIFF_REWARD_SCALE:-1000.0}"
USE_REVERSE_PROGRESS_DIFF="${USE_REVERSE_PROGRESS_DIFF:-false}"
NUM_PREFIX_SAMPLES="${NUM_PREFIX_SAMPLES:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$SUBMIT_MANIFEST")"

IFS='|' read -r -a SERVER_IDS <<< "$SERVER_JOB_IDS"
IFS='|' read -r -a INFO_FILES <<< "$SERVER_INFO_FILES"

if [ "${#SERVER_IDS[@]}" -ne 8 ] || [ "${#INFO_FILES[@]}" -ne 8 ]; then
  echo "ERROR: Expected 8 SERVER_JOB_IDS and 8 SERVER_INFO_FILES"
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
  echo "ERROR: No healthy server found"
  exit 1
fi

{
  echo "launcher_first_ready_server=${FIRST_READY}"
  echo "launcher_first_ready_info_file=${FIRST_INFO_FILE}"
  echo "launcher_first_ready_server_job=${FIRST_SERVER_JOB_ID}"
  echo "training_entry=${PROJECT_DIR}/test_scripts/test_iql.py"
  echo "training_env=${PROJECT_DIR}/conda_envs/rewind_nochunk"
  echo "offline_training_steps=100000"
  echo "success_bonus=${SUCCESS_BONUS}"
  echo "diff_gamma=${DIFF_GAMMA}"
  echo "diff_reward_scale=${DIFF_REWARD_SCALE}"
  echo "use_reverse_progress_diff=${USE_REVERSE_PROGRESS_DIFF}"
  echo "diff_offline_reward_divisor=0.001"
  echo "num_prefix_samples=${NUM_PREFIX_SAMPLES}"
  echo "max_frames_per_query=${NUM_PREFIX_SAMPLES}"
  echo "request_timeout=${REQUEST_TIMEOUT}"
} | tee -a "$SUBMIT_MANIFEST"

LABEL_LOCK="${TOPREWARD_DIR}/logs/topreward_label_${FIRST_SERVER_JOB_ID}.lock"

BASELINE_LABEL_ID=$(sbatch --parsable \
  --dependency="after:${FIRST_SERVER_JOB_ID}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",INFO_FILE="$FIRST_INFO_FILE",EXPECTED_SERVER_JOB_ID="$FIRST_SERVER_JOB_ID",LABEL_VARIANT=baseline,LOCK_PATH="$LABEL_LOCK",NUM_PREFIX_SAMPLES="$NUM_PREFIX_SAMPLES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES" \
  scripts/topreward_label_variant.sbatch)
echo "label_baseline=${BASELINE_LABEL_ID}" | tee -a "$SUBMIT_MANIFEST"

DIFF999_LABEL_ID=$(sbatch --parsable \
  --dependency="after:${FIRST_SERVER_JOB_ID}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",INFO_FILE="$FIRST_INFO_FILE",EXPECTED_SERVER_JOB_ID="$FIRST_SERVER_JOB_ID",LABEL_VARIANT=diff_gamma0999,DIFF_GAMMA="$DIFF_GAMMA",DIFF_REWARD_SCALE="$DIFF_REWARD_SCALE",USE_REVERSE_PROGRESS_DIFF="$USE_REVERSE_PROGRESS_DIFF",LOCK_PATH="$LABEL_LOCK",NUM_PREFIX_SAMPLES="$NUM_PREFIX_SAMPLES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES" \
  scripts/topreward_label_variant.sbatch)
echo "label_diff_gamma0999=${DIFF999_LABEL_ID}" | tee -a "$SUBMIT_MANIFEST"

BASELINE_OFFLINE_ID=$(sbatch --parsable \
  --dependency="afterok:${BASELINE_LABEL_ID}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",VARIANT=baseline,SUCCESS_BONUS="$SUCCESS_BONUS",NUM_PREFIX_SAMPLES="$NUM_PREFIX_SAMPLES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES" \
  scripts/topreward_offline_3seeds.sbatch)
echo "offline_baseline_3seeds=${BASELINE_OFFLINE_ID}" | tee -a "$SUBMIT_MANIFEST"

DIFF999_OFFLINE_ID=$(sbatch --parsable \
  --dependency="afterok:${DIFF999_LABEL_ID}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",VARIANT=diff_gamma0999,SUCCESS_BONUS="$SUCCESS_BONUS",DIFF_GAMMA="$DIFF_GAMMA",DIFF_REWARD_SCALE="$DIFF_REWARD_SCALE",NUM_PREFIX_SAMPLES="$NUM_PREFIX_SAMPLES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES" \
  scripts/topreward_offline_3seeds.sbatch)
echo "offline_diff_gamma0999_3seeds=${DIFF999_OFFLINE_ID}" | tee -a "$SUBMIT_MANIFEST"

GROUP_IDS=(1 2 3 4 5 6 7 8)
ENV_IDS=(
  window-close-v2
  reach-wall-v2
  faucet-close-v2
  coffee-button-v2
  button-press-wall-v2
  door-lock-v2
  handle-press-side-v2
  sweep-into-v2
)

for idx in "${!GROUP_IDS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  env_id="${ENV_IDS[$idx]}"
  server_job_id="${SERVER_IDS[$idx]}"
  info_file="${INFO_FILES[$idx]}"
  online_lock="${TOPREWARD_DIR}/logs/topreward_online_group${group_id}.lock"

  baseline_online_id=$(sbatch --parsable \
    --dependency="after:${server_job_id},afterok:${BASELINE_OFFLINE_ID}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",INFO_FILE="$info_file",EXPECTED_SERVER_JOB_ID="$server_job_id",GROUP_ID="$group_id",ENV_ID="$env_id",VARIANT=baseline,SUCCESS_BONUS="$SUCCESS_BONUS",LOCK_PATH="$online_lock",NUM_PREFIX_SAMPLES="$NUM_PREFIX_SAMPLES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES" \
    scripts/topreward_online_seed0_env_variant.sbatch)
  echo "online_group${group_id}_baseline=${baseline_online_id}" | tee -a "$SUBMIT_MANIFEST"

  diff999_online_id=$(sbatch --parsable \
    --dependency="after:${server_job_id},afterok:${DIFF999_OFFLINE_ID}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",INFO_FILE="$info_file",EXPECTED_SERVER_JOB_ID="$server_job_id",GROUP_ID="$group_id",ENV_ID="$env_id",VARIANT=diff_gamma0999,SUCCESS_BONUS="$SUCCESS_BONUS",DIFF_GAMMA="$DIFF_GAMMA",DIFF_REWARD_SCALE="$DIFF_REWARD_SCALE",USE_REVERSE_PROGRESS_DIFF="$USE_REVERSE_PROGRESS_DIFF",LOCK_PATH="$online_lock",NUM_PREFIX_SAMPLES="$NUM_PREFIX_SAMPLES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES" \
    scripts/topreward_online_seed0_env_variant.sbatch)
  echo "online_group${group_id}_diff_gamma0999=${diff999_online_id}" | tee -a "$SUBMIT_MANIFEST"
  echo "stop_group${group_id}=(disabled_no_auto_cancel)" | tee -a "$SUBMIT_MANIFEST"
done
