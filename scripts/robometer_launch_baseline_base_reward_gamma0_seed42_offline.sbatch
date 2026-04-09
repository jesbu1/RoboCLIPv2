#!/usr/bin/env bash
# Wait for any recovery server to become healthy, then submit offline gamma=0
# pretraining for seed42 only.

#SBATCH --job-name=robo_launch_bg0_s42
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
SEED="${SEED:-42}"
SUBMIT_MANIFEST="${SUBMIT_MANIFEST:-${PROJECT_DIR}/logs/robometer_baseline_gamma0_recover_manifest.txt}"

if [ -z "$SERVER_JOB_IDS" ] || [ -z "$SERVER_INFO_FILES" ]; then
  echo "ERROR: SERVER_JOB_IDS and SERVER_INFO_FILES must be provided"
  exit 1
fi

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$SUBMIT_MANIFEST")"

IFS='|' read -r -a SERVER_IDS <<< "$SERVER_JOB_IDS"
IFS='|' read -r -a INFO_FILES <<< "$SERVER_INFO_FILES"

if [ "${#SERVER_IDS[@]}" -eq 0 ] || [ "${#SERVER_IDS[@]}" -ne "${#INFO_FILES[@]}" ]; then
  echo "ERROR: SERVER_JOB_IDS and SERVER_INFO_FILES must have matching non-zero lengths"
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
  if ! squeue -h -j "$SERVER_IDS_CSV" | grep -q .; then
    echo "ERROR: All recovery server jobs left the queue before any healthy server became available"
    exit 1
  fi
  sleep 5
done

echo "First healthy recovery server: ${FIRST_READY}"
echo "launcher_seed42_first_ready_server=${FIRST_READY}" >> "$SUBMIT_MANIFEST"

OFFLINE_JOB_ID=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",SERVER_URL="${FIRST_READY}",SEED="${SEED}" \
  scripts/robometer_offline_baseline_bonus0_gamma0_single_seed.sbatch)

echo "offline_gamma0_seed${SEED}=${OFFLINE_JOB_ID}" | tee -a "$SUBMIT_MANIFEST"
