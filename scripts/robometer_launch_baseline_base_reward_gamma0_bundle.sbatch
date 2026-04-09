#!/usr/bin/env bash
# Wait for any mixed server to become healthy, then submit offline gamma=0
# pretraining plus the 16 dependent online gamma=0 jobs.

#SBATCH --job-name=robo_launch_bg0
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
SERVER_JOB_IDS="${SERVER_JOB_IDS:-}"
SERVER_INFO_FILES="${SERVER_INFO_FILES:-}"
RERUN_JOB_IDS="${RERUN_JOB_IDS:-}"
SUBMIT_MANIFEST="${SUBMIT_MANIFEST:-${PROJECT_DIR}/logs/robometer_baseline_gamma0_bundle_manifest.txt}"

if [ -z "$SERVER_JOB_IDS" ] || [ -z "$SERVER_INFO_FILES" ] || [ -z "$RERUN_JOB_IDS" ]; then
  echo "ERROR: SERVER_JOB_IDS, SERVER_INFO_FILES, and RERUN_JOB_IDS must be provided"
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$SUBMIT_MANIFEST")"

IFS=',' read -r -a SERVER_IDS <<< "$SERVER_JOB_IDS"
IFS=',' read -r -a INFO_FILES <<< "$SERVER_INFO_FILES"
IFS=',' read -r -a RERUN_IDS <<< "$RERUN_JOB_IDS"

if [ "${#SERVER_IDS[@]}" -ne 8 ] || [ "${#INFO_FILES[@]}" -ne 8 ] || [ "${#RERUN_IDS[@]}" -ne 8 ]; then
  echo "ERROR: Expected 8 SERVER_JOB_IDS, 8 SERVER_INFO_FILES, and 8 RERUN_JOB_IDS"
  exit 1
fi

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
  if [[ ",${SERVER_JOB_IDS}," != *",${server_job_id},"* ]]; then
    return 1
  fi
  if [ -z "$server_url" ] || [[ "$server_url" != http* ]]; then
    return 1
  fi
  printf '%s\n' "$server_url"
  return 0
}

FIRST_READY=""
for i in $(seq 1 34560); do
  for info_file in "${INFO_FILES[@]}"; do
    if [ -s "$info_file" ] && candidate_url="$(read_server_url "$info_file")"; then
      if curl -s "${candidate_url}/health" > /dev/null 2>&1; then
        FIRST_READY="$candidate_url"
        break 2
      fi
    fi
  done
  if ! squeue -h -j "$SERVER_JOB_IDS" | grep -q .; then
    echo "ERROR: All server jobs left the queue before any healthy server became available"
    exit 1
  fi
  sleep 5
done

echo "First healthy server: ${FIRST_READY}"
echo "launcher_first_ready_server=${FIRST_READY}" >> "$SUBMIT_MANIFEST"

OFFLINE_JOB_ID=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",SERVER_INFO_FILES="${SERVER_INFO_FILES}",SERVER_JOB_IDS="${SERVER_JOB_IDS}" \
  scripts/robometer_offline_baseline_bonus0_gamma0_2seeds_any_server.sbatch)

echo "offline_gamma0=${OFFLINE_JOB_ID}" | tee -a "$SUBMIT_MANIFEST"

GROUP_IDS=(15 16 17 18 19 20 21 22)
GAMMA0_ENVS=(
  "faucet-close-v2"
  "coffee-button-v2"
  "handle-press-v2"
  "window-close-v2"
  "button-press-v2"
  "reach-wall-v2"
  "sweep-into-v2"
  "door-lock-v2"
)

for idx in "${!GROUP_IDS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  server_id="${SERVER_IDS[$idx]}"
  info_file="${INFO_FILES[$idx]}"
  env_id="${GAMMA0_ENVS[$idx]}"
  rerun_id="${RERUN_IDS[$idx]}"

  job_id=$(sbatch --parsable \
    --job-name="robo_on_bg0_g${group_id}" \
    --output="logs/robo_on_bg0_g${group_id}_%A_%a.out" \
    --error="logs/robo_on_bg0_g${group_id}_%A_%a.err" \
    --dependency="after:${server_id},afterok:${OFFLINE_JOB_ID}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",ENV_ID="${env_id}",INFO_FILE="${info_file}",EXPECTED_SERVER_JOB_ID="${server_id}" \
    scripts/robometer_online_baseline_base_reward_gamma0_2seeds_env.sbatch)

  echo "gamma0_group${group_id}=${job_id}" | tee -a "$SUBMIT_MANIFEST"

  stop_job_id=$(sbatch --parsable \
    --job-name="stop_srv_g${group_id}" \
    --output="logs/stop_srv_g${group_id}_%j.out" \
    --error="logs/stop_srv_g${group_id}_%j.err" \
    --dependency="afterany:${rerun_id},afterany:${job_id}_*" \
    --export=ALL,TARGET_JOB_ID="${server_id}" \
    scripts/stop_slurm_job_by_id.sbatch)

  echo "stop_group${group_id}=${stop_job_id}" | tee -a "$SUBMIT_MANIFEST"
done
