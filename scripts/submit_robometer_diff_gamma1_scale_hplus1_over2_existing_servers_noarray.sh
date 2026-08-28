#!/usr/bin/env bash
# Reuse existing Robometer H128 server jobs and submit the no-array launcher.
# SERVER_JOB_IDS must contain the 8 server job ids in group 41..48 order.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAX_FRAMES="${MAX_FRAMES:-4}"
HORIZON="${HORIZON:-128}"
FORCE_LABEL="${FORCE_LABEL:-false}"
SERVER_JOB_IDS="${SERVER_JOB_IDS:?SERVER_JOB_IDS must be provided in group 41..48 order}"

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

if [[ "$SERVER_JOB_IDS" == *"|"* ]]; then
  IFS='|' read -r -a SERVER_IDS <<< "$SERVER_JOB_IDS"
else
  read -r -a SERVER_IDS <<< "$SERVER_JOB_IDS"
fi

if [ "${#SERVER_IDS[@]}" -ne 8 ]; then
  echo "ERROR: Expected 8 server job ids, got ${#SERVER_IDS[@]}: ${SERVER_JOB_IDS}"
  exit 1
fi

SERVER_INFO_FILES=()
for group_id in "${GROUP_IDS[@]}"; do
  SERVER_INFO_FILES+=("${PROJECT_DIR}/logs/robometer_server_d1_h_group${group_id}_info.txt")
done

DIFF_REWARD_SCALE=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", (h + 1) / 2 }')
REWARD_DIVISOR=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", 2 / (h + 1) }')
MANIFEST="${PROJECT_DIR}/logs/robometer_diff_gamma1_scale_hplus1_over2_H${HORIZON}_existing_servers_noarray_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "manifest=${MANIFEST}"
  echo "project_dir=${PROJECT_DIR}"
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
  echo "server_jobs=${SERVER_IDS[*]}"
} | tee "$MANIFEST"

for idx in "${!GROUP_IDS[@]}"; do
  echo "server_group${GROUP_IDS[$idx]}_${ENV_IDS[$idx]}=${SERVER_IDS[$idx]}" | tee -a "$MANIFEST"
  echo "server_info_group${GROUP_IDS[$idx]}=${SERVER_INFO_FILES[$idx]}" | tee -a "$MANIFEST"
done

SERVER_JOB_IDS_JOINED="$(IFS='|'; echo "${SERVER_IDS[*]}")"
SERVER_INFO_FILES_JOINED="$(IFS='|'; echo "${SERVER_INFO_FILES[*]}")"

launcher_id=$(sbatch --parsable \
  --job-name="robo_launch_d1_h" \
  --output="logs/robo_launch_d1_h_%j.out" \
  --error="logs/robo_launch_d1_h_%j.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",SERVER_JOB_IDS="$SERVER_JOB_IDS_JOINED",SERVER_INFO_FILES="$SERVER_INFO_FILES_JOINED",SUBMIT_MANIFEST="$MANIFEST",MAX_FRAMES="$MAX_FRAMES",HORIZON="$HORIZON",FORCE_LABEL="$FORCE_LABEL" \
  scripts/robometer_launch_diff_gamma1_scale_hplus1_over2_pipeline.sbatch)

echo "launcher=${launcher_id}" | tee -a "$MANIFEST"
