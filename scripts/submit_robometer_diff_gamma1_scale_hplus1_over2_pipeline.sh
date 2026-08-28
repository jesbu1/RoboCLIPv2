#!/usr/bin/env bash
# Submit Robometer diff-gamma=1.0 pipeline using scale=(HORIZON+1)/2:
# 8 servers -> first-ready label/offline -> 8 online groups, 3 seeds each.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROBOMETER_PROJECT_DIR="${ROBOMETER_PROJECT_DIR:-/project2/biyik_1165/haobaizh/rewind_robometer_valuemodel}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${ROBOMETER_PROJECT_DIR}/datasets}"
MAX_FRAMES="${MAX_FRAMES:-4}"
HORIZON="${HORIZON:-128}"
FORCE_LABEL="${FORCE_LABEL:-false}"

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
if [ ! -f "${RAW_DATA_DIR}/metaworld_generation.h5" ]; then
  echo "ERROR: metaworld_generation.h5 not found under ${RAW_DATA_DIR}"
  exit 1
fi
if [ ! -f "${RAW_DATA_DIR}/metaworld_embeddings_train.h5" ]; then
  echo "ERROR: metaworld_embeddings_train.h5 not found under ${RAW_DATA_DIR}"
  exit 1
fi

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
SERVER_PORTS=(8070 8071 8072 8073 8074 8075 8076 8077)
FALLBACK_STARTS=(18524 18530 18536 18542 18548 18554 18560 18566)
FALLBACK_ENDS=(18529 18535 18541 18547 18553 18559 18565 18571)
STARTUP_DELAYS=(0 120 240 360 480 600 720 840)

DIFF_REWARD_SCALE=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", (h + 1) / 2 }')
REWARD_DIVISOR=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", 2 / (h + 1) }')
MANIFEST="${PROJECT_DIR}/logs/robometer_diff_gamma1_scale_hplus1_over2_H${HORIZON}_pipeline_$(date +%Y%m%d_%H%M%S).txt"
SERVER_JOB_IDS=()
SERVER_INFO_FILES=()

{
  echo "manifest=${MANIFEST}"
  echo "project_dir=${PROJECT_DIR}"
  echo "robometer_project_dir=${ROBOMETER_PROJECT_DIR}"
  echo "raw_data_dir=${RAW_DATA_DIR}"
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
  echo "force_label=${FORCE_LABEL}"
  echo "tasks=${ENV_IDS[*]}"
  echo "seeds=42 32 0"
} | tee "$MANIFEST"

for idx in "${!GROUP_IDS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  env_id="${ENV_IDS[$idx]}"
  info_file="${PROJECT_DIR}/logs/robometer_server_d1_h_group${group_id}_info.txt"
  rm -f "$info_file" "${info_file}.tmp"

  server_job_id=$(sbatch --parsable \
    --job-name="robo_srv_d1_h_g${group_id}" \
    --output="logs/robo_srv_d1_h_g${group_id}_%j.out" \
    --error="logs/robo_srv_d1_h_g${group_id}_%j.err" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",ROBOMETER_PROJECT_DIR="${ROBOMETER_PROJECT_DIR}",GROUP_ID="${group_id}",SERVER_LABEL="diff gamma1 horizon-scale ${env_id}",INFO_FILE="${info_file}",SERVER_PORT="${SERVER_PORTS[$idx]}",FALLBACK_START="${FALLBACK_STARTS[$idx]}",FALLBACK_END="${FALLBACK_ENDS[$idx]}",STARTUP_DELAY_SEC="${STARTUP_DELAYS[$idx]}",STARTUP_MAX_ATTEMPTS=6,STARTUP_BACKOFF_BASE_SEC=120 \
    scripts/robometer_server_mixed_group_generic.sbatch)

  SERVER_JOB_IDS+=("$server_job_id")
  SERVER_INFO_FILES+=("$info_file")
  echo "server_group${group_id}_${env_id}=${server_job_id}" | tee -a "$MANIFEST"
done

SERVER_JOB_IDS_JOINED="$(IFS='|'; echo "${SERVER_JOB_IDS[*]}")"
SERVER_INFO_FILES_JOINED="$(IFS='|'; echo "${SERVER_INFO_FILES[*]}")"

launcher_id=$(sbatch --parsable \
  --job-name="robo_launch_d1_h" \
  --output="logs/robo_launch_d1_h_%j.out" \
  --error="logs/robo_launch_d1_h_%j.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",SERVER_JOB_IDS="$SERVER_JOB_IDS_JOINED",SERVER_INFO_FILES="$SERVER_INFO_FILES_JOINED",SUBMIT_MANIFEST="$MANIFEST",MAX_FRAMES="$MAX_FRAMES",HORIZON="$HORIZON",FORCE_LABEL="$FORCE_LABEL" \
  scripts/robometer_launch_diff_gamma1_scale_hplus1_over2_pipeline.sbatch)

echo "launcher=${launcher_id}" | tee -a "$MANIFEST"
