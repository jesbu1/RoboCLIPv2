#!/usr/bin/env bash
# Wait for any of 8 raw TOPReward servers, then submit:
# baseline label/offline + diff label/offline + 16 seed0 online jobs.
# Servers 0-3 run baseline online jobs, servers 4-7 run diff online jobs.
# Each server runs one online first, then the second online after the first exits.

#SBATCH --job-name=top_lau_raw8
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
SERVER_GROUP_IDS="${SERVER_GROUP_IDS:?SERVER_GROUP_IDS must be provided}"
SUBMIT_MANIFEST="${SUBMIT_MANIFEST:-${PROJECT_DIR}/logs/topreward_raw_baseline_diff_seed0_manifest.txt}"
RUN_TAG="${RUN_TAG:?RUN_TAG must be set}"
HORIZON="${HORIZON:-128}"
MAX_FRAMES="${MAX_FRAMES:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"
TOPREWARD_REQUEST_FORMAT="${TOPREWARD_REQUEST_FORMAT:-raw}"
LABEL_LOCK="${LABEL_LOCK:-${TOPREWARD_DIR}/logs/topreward_raw_label_${RUN_TAG}.lock}"
ONLINE_LOG_ROOT="${ONLINE_LOG_ROOT:-/scratch1/haobaizh/rewind_nochunk_online_logs/topreward_raw}"
BASELINE_H5="${BASELINE_H5:?BASELINE_H5 must be set}"
DIFF_H5="${DIFF_H5:?DIFF_H5 must be set}"
BASELINE_OFFLINE_DIR="${BASELINE_OFFLINE_DIR:?BASELINE_OFFLINE_DIR must be set}"
DIFF_OFFLINE_DIR="${DIFF_OFFLINE_DIR:?DIFF_OFFLINE_DIR must be set}"

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$SUBMIT_MANIFEST")"

IFS='|' read -r -a SERVER_IDS <<< "$SERVER_JOB_IDS"
IFS='|' read -r -a INFO_FILES <<< "$SERVER_INFO_FILES"
IFS='|' read -r -a GROUP_IDS <<< "$SERVER_GROUP_IDS"

if [ "${#SERVER_IDS[@]}" -ne 8 ] || [ "${#INFO_FILES[@]}" -ne 8 ] || [ "${#GROUP_IDS[@]}" -ne 8 ]; then
  echo "ERROR: Expected 8 SERVER_JOB_IDS, 8 SERVER_INFO_FILES, and 8 SERVER_GROUP_IDS"
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
  echo "ERROR: No healthy TOPReward raw server found"
  exit 1
fi

FIRST_ENVS=(
  "window-close-v2"
  "faucet-close-v2"
  "button-press-wall-v2"
  "handle-press-side-v2"
)
SECOND_ENVS=(
  "reach-wall-v2"
  "coffee-button-v2"
  "door-lock-v2"
  "sweep-into-v2"
)

DIFF_REWARD_SCALE=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", (h + 1) / 2 }')
REWARD_DIVISOR=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", 2 / (h + 1) }')
BASELINE_CKPT_PATH="${BASELINE_OFFLINE_DIR}/last_offline"
DIFF_CKPT_PATH="${DIFF_OFFLINE_DIR}/last_offline"

{
  echo "launcher_first_ready_server=${FIRST_READY}"
  echo "launcher_first_ready_info_file=${FIRST_INFO_FILE}"
  echo "launcher_first_ready_server_job=${FIRST_SERVER_JOB_ID}"
  echo "run_tag=${RUN_TAG}"
  echo "baseline_h5=${BASELINE_H5}"
  echo "diff_h5=${DIFF_H5}"
  echo "baseline_offline_dir=${BASELINE_OFFLINE_DIR}"
  echo "diff_offline_dir=${DIFF_OFFLINE_DIR}"
  echo "baseline_ckpt_path=${BASELINE_CKPT_PATH}"
  echo "diff_ckpt_path=${DIFF_CKPT_PATH}"
  echo "offline_training_steps=100000"
  echo "online_seed=0"
  echo "success_bonus=0"
  echo "diff_gamma=1.0"
  echo "horizon=${HORIZON}"
  echo "diff_reward_scale=(HORIZON+1)/2=${DIFF_REWARD_SCALE}"
  echo "offline_reward_divisor=2/(HORIZON+1)=${REWARD_DIVISOR}"
  echo "max_frames=${MAX_FRAMES}"
  echo "topreward_request_format=${TOPREWARD_REQUEST_FORMAT}"
  echo "servers_total=8"
  echo "baseline_servers=${SERVER_IDS[0]},${SERVER_IDS[1]},${SERVER_IDS[2]},${SERVER_IDS[3]}"
  echo "diff_servers=${SERVER_IDS[4]},${SERVER_IDS[5]},${SERVER_IDS[6]},${SERVER_IDS[7]}"
  echo "first_wave=${FIRST_ENVS[*]}"
  echo "second_wave=${SECOND_ENVS[*]}"
} | tee -a "$SUBMIT_MANIFEST"

BASELINE_LABEL_ID=$(sbatch --parsable \
  --dependency="after:${FIRST_SERVER_JOB_ID}" \
  --job-name="top_lbl_base_raw" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",INFO_FILE="$FIRST_INFO_FILE",EXPECTED_SERVER_JOB_ID="$FIRST_SERVER_JOB_ID",VARIANT=baseline,OUTPUT_PATH="$BASELINE_H5",MAX_FRAMES="$MAX_FRAMES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES",TOPREWARD_REQUEST_FORMAT="$TOPREWARD_REQUEST_FORMAT",LOCK_PATH="$LABEL_LOCK" \
  scripts/topreward_label_raw_seed0_variant.sbatch)
echo "label_baseline_raw=${BASELINE_LABEL_ID} depends_on=after:${FIRST_SERVER_JOB_ID}" | tee -a "$SUBMIT_MANIFEST"

DIFF_LABEL_ID=$(sbatch --parsable \
  --dependency="after:${FIRST_SERVER_JOB_ID}" \
  --job-name="top_lbl_diff_raw" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",INFO_FILE="$FIRST_INFO_FILE",EXPECTED_SERVER_JOB_ID="$FIRST_SERVER_JOB_ID",VARIANT=diff_gamma1,OUTPUT_PATH="$DIFF_H5",MAX_FRAMES="$MAX_FRAMES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES",TOPREWARD_REQUEST_FORMAT="$TOPREWARD_REQUEST_FORMAT",LOCK_PATH="$LABEL_LOCK",DIFF_GAMMA=1.0 \
  scripts/topreward_label_raw_seed0_variant.sbatch)
echo "label_diff_raw=${DIFF_LABEL_ID} depends_on=after:${FIRST_SERVER_JOB_ID}" | tee -a "$SUBMIT_MANIFEST"

BASELINE_OFFLINE_ID=$(sbatch --parsable \
  --dependency="afterok:${BASELINE_LABEL_ID}" \
  --job-name="top_off_base_raw" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",INFO_FILE="$FIRST_INFO_FILE",EXPECTED_SERVER_JOB_ID="$FIRST_SERVER_JOB_ID",VARIANT=baseline,H5_PATH="$BASELINE_H5",LOG_DIR="$BASELINE_OFFLINE_DIR",LOG_GROUP="offline_topreward_raw_baseline_H${HORIZON}_10critics_${RUN_TAG}",HORIZON="$HORIZON",MAX_FRAMES="$MAX_FRAMES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES",TOPREWARD_REQUEST_FORMAT="$TOPREWARD_REQUEST_FORMAT",TOPREWARD_LOCK_PATH="$LABEL_LOCK" \
  scripts/topreward_offline_raw_seed0_variant.sbatch)
echo "offline_baseline_raw=${BASELINE_OFFLINE_ID} depends_on=afterok:${BASELINE_LABEL_ID}" | tee -a "$SUBMIT_MANIFEST"

DIFF_OFFLINE_ID=$(sbatch --parsable \
  --dependency="afterok:${DIFF_LABEL_ID}" \
  --job-name="top_off_diff_raw" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",INFO_FILE="$FIRST_INFO_FILE",EXPECTED_SERVER_JOB_ID="$FIRST_SERVER_JOB_ID",VARIANT=diff_gamma1,H5_PATH="$DIFF_H5",LOG_DIR="$DIFF_OFFLINE_DIR",LOG_GROUP="offline_topreward_raw_diff_gamma1_H${HORIZON}_10critics_${RUN_TAG}",HORIZON="$HORIZON",MAX_FRAMES="$MAX_FRAMES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES",TOPREWARD_REQUEST_FORMAT="$TOPREWARD_REQUEST_FORMAT",TOPREWARD_LOCK_PATH="$LABEL_LOCK",DIFF_GAMMA=1.0 \
  scripts/topreward_offline_raw_seed0_variant.sbatch)
echo "offline_diff_raw=${DIFF_OFFLINE_ID} depends_on=afterok:${DIFF_LABEL_ID}" | tee -a "$SUBMIT_MANIFEST"

submit_online_pair() {
  local variant="$1"
  local offline_id="$2"
  local h5_path="$3"
  local ckpt_path="$4"
  local server_idx="$5"
  local pair_idx="$6"
  local first_env="${FIRST_ENVS[$pair_idx]}"
  local second_env="${SECOND_ENVS[$pair_idx]}"
  local server_job_id="${SERVER_IDS[$server_idx]}"
  local info_file="${INFO_FILES[$server_idx]}"
  local group_id="${GROUP_IDS[$server_idx]}"
  local lock_path="${TOPREWARD_DIR}/logs/topreward_raw_online_group${group_id}_${RUN_TAG}.lock"
  local variant_short first_id second_id cancel_id

  case "$variant" in
    baseline) variant_short="base" ;;
    diff_gamma1) variant_short="diff" ;;
    *) echo "ERROR: bad variant ${variant}"; return 1 ;;
  esac

  if [ "$server_job_id" = "$FIRST_SERVER_JOB_ID" ]; then
    lock_path="$LABEL_LOCK"
  fi

  first_id=$(sbatch --parsable \
    --dependency="after:${server_job_id},afterok:${offline_id}" \
    --job-name="top_on_${variant_short}_${pair_idx}a" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",INFO_FILE="$info_file",EXPECTED_SERVER_JOB_ID="$server_job_id",GROUP_ID="$group_id",ENV_ID="$first_env",VARIANT="$variant",H5_PATH="$h5_path",CKPT_PATH="$ckpt_path",RUN_TAG="$RUN_TAG",HORIZON="$HORIZON",MAX_FRAMES="$MAX_FRAMES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES",TOPREWARD_REQUEST_FORMAT="$TOPREWARD_REQUEST_FORMAT",LOCK_PATH="$lock_path",LOG_ROOT="$ONLINE_LOG_ROOT",DIFF_GAMMA=1.0 \
    scripts/topreward_online_raw_seed0_env.sbatch)

  second_id=$(sbatch --parsable \
    --dependency="after:${server_job_id},afterok:${offline_id},afterany:${first_id}" \
    --job-name="top_on_${variant_short}_${pair_idx}b" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",INFO_FILE="$info_file",EXPECTED_SERVER_JOB_ID="$server_job_id",GROUP_ID="$group_id",ENV_ID="$second_env",VARIANT="$variant",H5_PATH="$h5_path",CKPT_PATH="$ckpt_path",RUN_TAG="$RUN_TAG",HORIZON="$HORIZON",MAX_FRAMES="$MAX_FRAMES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES",TOPREWARD_REQUEST_FORMAT="$TOPREWARD_REQUEST_FORMAT",LOCK_PATH="$lock_path",LOG_ROOT="$ONLINE_LOG_ROOT",DIFF_GAMMA=1.0 \
    scripts/topreward_online_raw_seed0_env.sbatch)

  cancel_id=$(sbatch --parsable \
    --job-name="top_stop_${group_id}" \
    --account=biyik_1165 \
    --partition=main \
    --time=00:05:00 \
    --cpus-per-task=1 \
    --mem=512M \
    --dependency="afterany:${second_id}" \
    --output="logs/top_stop_${group_id}_%j.out" \
    --error="logs/top_stop_${group_id}_%j.err" \
    --wrap="scancel ${server_job_id} || true")

  {
    echo "online_${variant}_group${group_id}_first=${first_id} env=${first_env} server=${server_job_id} lock=${lock_path} depends_on=after:${server_job_id},afterok:${offline_id}"
    echo "online_${variant}_group${group_id}_second=${second_id} env=${second_env} server=${server_job_id} lock=${lock_path} depends_on=after:${server_job_id},afterok:${offline_id},afterany:${first_id}"
    echo "cancel_server_group${group_id}=${cancel_id} server=${server_job_id} depends_on=afterany:${second_id}"
  } | tee -a "$SUBMIT_MANIFEST"
}

for pair_idx in 0 1 2 3; do
  submit_online_pair baseline "$BASELINE_OFFLINE_ID" "$BASELINE_H5" "$BASELINE_CKPT_PATH" "$pair_idx" "$pair_idx"
done

for pair_idx in 0 1 2 3; do
  server_idx=$((pair_idx + 4))
  submit_online_pair diff_gamma1 "$DIFF_OFFLINE_ID" "$DIFF_H5" "$DIFF_CKPT_PATH" "$server_idx" "$pair_idx"
done

echo "launcher_done=1" | tee -a "$SUBMIT_MANIFEST"
