#!/usr/bin/env bash
# Wait for any Robometer server to become healthy, then submit:
# optional label -> offline horizon-scale 3 seeds -> online 8 envs x 3 seeds.

#SBATCH --job-name=robo_launch_d1_h
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
SERVER_JOB_IDS="${SERVER_JOB_IDS:?SERVER_JOB_IDS must be provided}"
SERVER_INFO_FILES="${SERVER_INFO_FILES:?SERVER_INFO_FILES must be provided}"
SUBMIT_MANIFEST="${SUBMIT_MANIFEST:-${PROJECT_DIR}/logs/robometer_diff_gamma1_scale_hplus1_over2_manifest.txt}"
MAX_FRAMES="${MAX_FRAMES:-4}"
HORIZON="${HORIZON:-128}"
FORCE_LABEL="${FORCE_LABEL:-false}"

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
DIFF_REWARD_SCALE=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", (h + 1) / 2 }')
REWARD_DIVISOR=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", 2 / (h + 1) }')

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
  echo "ERROR: No healthy Robometer server found"
  exit 1
fi

{
  echo "launcher_first_ready_server=${FIRST_READY}"
  echo "launcher_first_ready_info_file=${FIRST_INFO_FILE}"
  echo "launcher_first_ready_server_job=${FIRST_SERVER_JOB_ID}"
  echo "variant=diff_gamma1_scale_hplus1_over2_base_reward"
  echo "label_output=datasets/metaworld_labeled_robometer_diff_gamma1.h5"
  echo "offline_training_steps=100000"
  echo "success_bonus=0"
  echo "diff_gamma=1.0"
  echo "horizon=${HORIZON}"
  echo "diff_reward_scale=(HORIZON+1)/2=${DIFF_REWARD_SCALE}"
  echo "offline_reward_divisor=2/(HORIZON+1)=${REWARD_DIVISOR}"
  echo "max_frames=${MAX_FRAMES}"
  echo "online_base_reward=true"
  echo "base_reward_value=-1.0"
} | tee -a "$SUBMIT_MANIFEST"

LABEL_H5="datasets/metaworld_labeled_robometer_diff_gamma1.h5"
LABEL_ID=""
if [ "$FORCE_LABEL" = "true" ] || [ ! -f "$LABEL_H5" ]; then
  LABEL_ID=$(sbatch --parsable \
    --dependency="after:${FIRST_SERVER_JOB_ID}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",INFO_FILE="$FIRST_INFO_FILE",EXPECTED_SERVER_JOB_ID="$FIRST_SERVER_JOB_ID",MAX_FRAMES="$MAX_FRAMES" \
    scripts/robometer_label_diff_gamma1.sbatch)
  echo "label_diff_gamma1=${LABEL_ID}" | tee -a "$SUBMIT_MANIFEST"
  OFFLINE_DEP="--dependency=after:${FIRST_SERVER_JOB_ID},afterok:${LABEL_ID}"
else
  echo "label_diff_gamma1=(skipped_existing_h5)" | tee -a "$SUBMIT_MANIFEST"
  OFFLINE_DEP="--dependency=after:${FIRST_SERVER_JOB_ID}"
fi

OFFLINE_ID=$(sbatch --parsable $OFFLINE_DEP \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",INFO_FILE="$FIRST_INFO_FILE",EXPECTED_SERVER_JOB_ID="$FIRST_SERVER_JOB_ID",MAX_FRAMES="$MAX_FRAMES",HORIZON="$HORIZON" \
  scripts/robometer_offline_diff_gamma1_scale_hplus1_over2_bonus0_3seeds.sbatch)
echo "offline_diff_gamma1_scale_hplus1_over2_3seeds=${OFFLINE_ID}" | tee -a "$SUBMIT_MANIFEST"

GROUP_IDS=(41 42 43 44 45 46 47 48)
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

  online_id=$(sbatch --parsable \
    --dependency="after:${server_job_id},afterok:${OFFLINE_ID}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",INFO_FILE="$info_file",EXPECTED_SERVER_JOB_ID="$server_job_id",GROUP_ID="$group_id",ENV_ID="$env_id",MAX_FRAMES="$MAX_FRAMES",HORIZON="$HORIZON" \
    scripts/robometer_online_diff_gamma1_scale_hplus1_over2_base_reward_env_3seeds.sbatch)
  echo "online_group${group_id}_${env_id}=${online_id}" | tee -a "$SUBMIT_MANIFEST"

  stop_id=$(sbatch --parsable \
    --job-name="stop_robo_d1_h_g${group_id}" \
    --output="logs/stop_robo_d1_h_g${group_id}_%j.out" \
    --error="logs/stop_robo_d1_h_g${group_id}_%j.err" \
    --dependency="afterany:${online_id}_*" \
    --export=ALL,TARGET_JOB_ID="$server_job_id" \
    scripts/stop_slurm_job_by_id.sbatch)
  echo "stop_group${group_id}=${stop_id}" | tee -a "$SUBMIT_MANIFEST"
done
