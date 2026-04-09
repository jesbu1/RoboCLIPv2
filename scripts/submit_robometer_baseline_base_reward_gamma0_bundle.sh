#!/usr/bin/env bash
# Submit 8 mixed servers, 8 immediate seed32 reruns, and a launcher that
# submits offline gamma=0 pretraining plus 16 dependent online gamma=0 jobs.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

GROUP_IDS=(15 16 17 18 19 20 21 22)
SERVER_PORTS=(8040 8041 8042 8043 8044 8045 8046 8047)
FALLBACK_STARTS=(18368 18374 18380 18386 18392 18398 18404 18410)
FALLBACK_ENDS=(18373 18379 18385 18391 18397 18403 18409 18415)

RERUN_ENVS=(
  "faucet-close-v2"
  "faucet-close-v2"
  "faucet-close-v2"
  "faucet-close-v2"
  "window-close-v2"
  "window-close-v2"
  "window-close-v2"
  "window-close-v2"
)
RERUN_REPS=(1 2 3 4 1 2 3 4)

SERVER_JOB_IDS=()
INFO_FILES=()
RERUN_JOB_IDS=()

for idx in "${!GROUP_IDS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  info_file="${PROJECT_DIR}/logs/robometer_server_mixed_group${group_id}_info.txt"
  server_job_id=$(sbatch --parsable \
    --job-name="robo_srv_mix_g${group_id}" \
    --output="logs/robometer_server_mixed_group${group_id}_%j.out" \
    --error="logs/robometer_server_mixed_group${group_id}_%j.err" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",GROUP_ID="${group_id}",SERVER_LABEL="mixed_group_${group_id}",INFO_FILE="${info_file}",SERVER_PORT="${SERVER_PORTS[$idx]}",FALLBACK_START="${FALLBACK_STARTS[$idx]}",FALLBACK_END="${FALLBACK_ENDS[$idx]}" \
    scripts/robometer_server_mixed_group_generic.sbatch)
  SERVER_JOB_IDS+=("${server_job_id}")
  INFO_FILES+=("${info_file}")
done

for idx in "${!GROUP_IDS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  server_job_id="${SERVER_JOB_IDS[$idx]}"
  info_file="${INFO_FILES[$idx]}"
  env_id="${RERUN_ENVS[$idx]}"
  rep_idx="${RERUN_REPS[$idx]}"

  rerun_job_id=$(sbatch --parsable \
    --job-name="robo_on_bl32_r${rep_idx}_g${group_id}" \
    --output="logs/robo_on_bl32_r${rep_idx}_g${group_id}_%j.out" \
    --error="logs/robo_on_bl32_r${rep_idx}_g${group_id}_%j.err" \
    --dependency="after:${server_job_id}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",ENV_ID="${env_id}",REP_IDX="${rep_idx}",INFO_FILE="${info_file}",EXPECTED_SERVER_JOB_ID="${server_job_id}" \
    scripts/robometer_online_baseline_base_reward_seed32_rerun_env.sbatch)
  RERUN_JOB_IDS+=("${rerun_job_id}")
done

SERVER_JOB_IDS_CSV="$(IFS=,; echo "${SERVER_JOB_IDS[*]}")"
INFO_FILES_CSV="$(IFS=,; echo "${INFO_FILES[*]}")"
RERUN_JOB_IDS_CSV="$(IFS=,; echo "${RERUN_JOB_IDS[*]}")"
MANIFEST="logs/robometer_baseline_gamma0_bundle_$(date +%Y%m%d_%H%M%S).txt"

LAUNCHER_JOB_ID=$(sbatch --parsable \
  --job-name="robo_launch_bg0" \
  --output="logs/robo_launch_bg0_%j.out" \
  --error="logs/robo_launch_bg0_%j.err" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",SERVER_JOB_IDS="${SERVER_JOB_IDS_CSV}",SERVER_INFO_FILES="${INFO_FILES_CSV}",RERUN_JOB_IDS="${RERUN_JOB_IDS_CSV}",SUBMIT_MANIFEST="${PROJECT_DIR}/${MANIFEST}" \
  scripts/robometer_launch_baseline_base_reward_gamma0_bundle.sbatch)

{
  echo "manifest=${PROJECT_DIR}/${MANIFEST}"
  for idx in "${!GROUP_IDS[@]}"; do
    echo "server_group${GROUP_IDS[$idx]}=${SERVER_JOB_IDS[$idx]}"
  done
  for idx in "${!GROUP_IDS[@]}"; do
    echo "rerun_group${GROUP_IDS[$idx]}=${RERUN_JOB_IDS[$idx]}"
  done
  echo "launcher=${LAUNCHER_JOB_ID}"
} | tee "${PROJECT_DIR}/${MANIFEST}"
