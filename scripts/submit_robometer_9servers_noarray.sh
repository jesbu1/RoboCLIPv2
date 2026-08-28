#!/usr/bin/env bash
# Submit 9 independent Robometer server jobs. No Slurm arrays, no label/offline/online jobs.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROBOMETER_PROJECT_DIR="${ROBOMETER_PROJECT_DIR:-/project2/biyik_1165/haobaizh/rewind_robometer_valuemodel}"

cd "$PROJECT_DIR"
mkdir -p logs

if [ ! -f "${PROJECT_DIR}/test_scripts/test_iql.py" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"
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

GROUP_IDS=(40 41 42 43 44 45 46 47 48)
SERVER_LABELS=(
  extra
  window-close-v2
  reach-wall-v2
  faucet-close-v2
  coffee-button-v2
  button-press-wall-v2
  door-lock-v2
  handle-press-side-v2
  sweep-into-v2
)
SERVER_PORTS=(8080 8081 8082 8083 8084 8085 8086 8087 8088)
FALLBACK_STARTS=(18600 18606 18612 18618 18624 18630 18636 18642 18648)
FALLBACK_ENDS=(18605 18611 18617 18623 18629 18635 18641 18647 18653)
STARTUP_DELAYS=(0 120 240 360 480 600 720 840 960)

MANIFEST="${PROJECT_DIR}/logs/robometer_9servers_noarray_$(date +%Y%m%d_%H%M%S).txt"
SERVER_JOB_IDS=()
SERVER_INFO_FILES=()

{
  echo "manifest=${MANIFEST}"
  echo "project_dir=${PROJECT_DIR}"
  echo "robometer_project_dir=${ROBOMETER_PROJECT_DIR}"
  echo "servers_total=9"
  echo "array_usage=disabled"
  echo "label_jobs=none"
  echo "offline_jobs=none"
  echo "online_jobs=none"
  echo "auto_cancel_servers=disabled"
} | tee "$MANIFEST"

for idx in "${!GROUP_IDS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  server_label="${SERVER_LABELS[$idx]}"
  info_file="${PROJECT_DIR}/logs/robometer_server_9srv_group${group_id}_info.txt"
  rm -f "$info_file" "${info_file}.tmp"

  server_job_id=$(sbatch --parsable \
    --job-name="robo_srv_9_g${group_id}" \
    --output="logs/robo_srv_9_g${group_id}_%j.out" \
    --error="logs/robo_srv_9_g${group_id}_%j.err" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",ROBOMETER_PROJECT_DIR="${ROBOMETER_PROJECT_DIR}",GROUP_ID="${group_id}",SERVER_LABEL="9-server ${server_label}",INFO_FILE="${info_file}",SERVER_PORT="${SERVER_PORTS[$idx]}",FALLBACK_START="${FALLBACK_STARTS[$idx]}",FALLBACK_END="${FALLBACK_ENDS[$idx]}",STARTUP_DELAY_SEC="${STARTUP_DELAYS[$idx]}",STARTUP_MAX_ATTEMPTS=6,STARTUP_BACKOFF_BASE_SEC=120 \
    scripts/robometer_server_mixed_group_generic.sbatch)

  SERVER_JOB_IDS+=("$server_job_id")
  SERVER_INFO_FILES+=("$info_file")
  echo "server_group${group_id}_${server_label}=${server_job_id}" | tee -a "$MANIFEST"
  echo "server_info_group${group_id}=${info_file}" | tee -a "$MANIFEST"
done

SERVER_JOB_IDS_JOINED="$(IFS='|'; echo "${SERVER_JOB_IDS[*]}")"
SERVER_INFO_FILES_JOINED="$(IFS='|'; echo "${SERVER_INFO_FILES[*]}")"

{
  echo "server_jobs_pipe=${SERVER_JOB_IDS_JOINED}"
  echo "server_info_files_pipe=${SERVER_INFO_FILES_JOINED}"
} | tee -a "$MANIFEST"
