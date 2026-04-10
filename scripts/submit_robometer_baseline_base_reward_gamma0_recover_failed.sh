#!/usr/bin/env bash
# Recover failed groups 16-22 only, reusing completed gamma0 offline ckpts for
# seeds 0/32, and additionally submit offline gamma0 for seed42.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

for seed in 0 32; do
  if [ ! -f "logs/offline_robometer_baseline_bonus0_gamma0_seed${seed}/last_offline.zip" ]; then
    echo "ERROR: Missing completed gamma0 offline checkpoint for seed${seed}"
    echo "Expected: logs/offline_robometer_baseline_bonus0_gamma0_seed${seed}/last_offline.zip"
    exit 1
  fi
done

GROUP_IDS=(16 17 18 19 20 21 22)
SERVER_PORTS=(8041 8042 8043 8044 8045 8046 8047)
FALLBACK_STARTS=(18374 18380 18386 18392 18398 18404 18410)
FALLBACK_ENDS=(18379 18385 18391 18397 18403 18409 18415)
STARTUP_DELAYS=(0 120 240 360 480 600 720)
SERVER_EXCLUDE="d23-[10,13-16],e21-[01-16],e22-[01-16],e23-01"

GAMMA0_ENVS=(
  "coffee-button-v2"
  "handle-press-v2"
  "window-close-v2"
  "button-press-v2"
  "reach-wall-v2"
  "sweep-into-v2"
  "door-lock-v2"
)

RERUN_ENVS=(
  "faucet-close-v2"
  "faucet-close-v2"
  "faucet-close-v2"
  "window-close-v2"
  "window-close-v2"
  "window-close-v2"
  "window-close-v2"
)
RERUN_REPS=(2 3 4 1 2 3 4)

SERVER_JOB_IDS=()
INFO_FILES=()
RERUN_JOB_IDS=()
GAMMA0_JOB_IDS=()

for idx in "${!GROUP_IDS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  info_file="${PROJECT_DIR}/logs/robometer_server_mixed_group${group_id}_info.txt"
  rm -f "$info_file" "${info_file}.tmp"

  server_job_id=$(sbatch --parsable \
    --job-name="robo_srv_mix_g${group_id}" \
    --exclude="${SERVER_EXCLUDE}" \
    --output="logs/robometer_server_mixed_group${group_id}_%j.out" \
    --error="logs/robometer_server_mixed_group${group_id}_%j.err" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",GROUP_ID="${group_id}",SERVER_LABEL="mixed_group_${group_id}",INFO_FILE="${info_file}",SERVER_PORT="${SERVER_PORTS[$idx]}",FALLBACK_START="${FALLBACK_STARTS[$idx]}",FALLBACK_END="${FALLBACK_ENDS[$idx]}",STARTUP_DELAY_SEC="${STARTUP_DELAYS[$idx]}",STARTUP_MAX_ATTEMPTS=6,STARTUP_BACKOFF_BASE_SEC=120 \
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

for idx in "${!GROUP_IDS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  server_job_id="${SERVER_JOB_IDS[$idx]}"
  info_file="${INFO_FILES[$idx]}"
  env_id="${GAMMA0_ENVS[$idx]}"

  gamma0_job_id=$(sbatch --parsable \
    --job-name="robo_on_bg0_g${group_id}" \
    --output="logs/robo_on_bg0_g${group_id}_%A_%a.out" \
    --error="logs/robo_on_bg0_g${group_id}_%A_%a.err" \
    --dependency="after:${server_job_id}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",ENV_ID="${env_id}",INFO_FILE="${info_file}",EXPECTED_SERVER_JOB_ID="${server_job_id}" \
    scripts/robometer_online_baseline_base_reward_gamma0_2seeds_env.sbatch)
  GAMMA0_JOB_IDS+=("${gamma0_job_id}")
done

STOP_JOB_IDS=()
for idx in "${!GROUP_IDS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  server_job_id="${SERVER_JOB_IDS[$idx]}"
  rerun_job_id="${RERUN_JOB_IDS[$idx]}"
  gamma0_job_id="${GAMMA0_JOB_IDS[$idx]}"

  stop_job_id=$(sbatch --parsable \
    --job-name="stop_srv_g${group_id}" \
    --output="logs/stop_srv_g${group_id}_%j.out" \
    --error="logs/stop_srv_g${group_id}_%j.err" \
    --dependency="afterany:${rerun_job_id},afterany:${gamma0_job_id}_*" \
    --export=ALL,TARGET_JOB_ID="${server_job_id}" \
    scripts/stop_slurm_job_by_id.sbatch)
  STOP_JOB_IDS+=("${stop_job_id}")
done

SERVER_JOB_IDS_LIST="$(IFS='|'; echo "${SERVER_JOB_IDS[*]}")"
INFO_FILES_LIST="$(IFS='|'; echo "${INFO_FILES[*]}")"
MANIFEST="logs/robometer_baseline_gamma0_recover_failed_$(date +%Y%m%d_%H%M%S).txt"

SEED42_LAUNCHER_JOB_ID=$(sbatch --parsable \
  --job-name="robo_launch_bg0_s42" \
  --output="logs/robo_launch_bg0_s42_%j.out" \
  --error="logs/robo_launch_bg0_s42_%j.err" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",SERVER_JOB_IDS="${SERVER_JOB_IDS_LIST}",SERVER_INFO_FILES="${INFO_FILES_LIST}",SEED=42,SUBMIT_MANIFEST="${PROJECT_DIR}/${MANIFEST}" \
  scripts/robometer_launch_baseline_base_reward_gamma0_seed42_offline.sbatch)

{
  echo "manifest=${PROJECT_DIR}/${MANIFEST}"
  for idx in "${!GROUP_IDS[@]}"; do
    echo "server_group${GROUP_IDS[$idx]}=${SERVER_JOB_IDS[$idx]}"
  done
  for idx in "${!GROUP_IDS[@]}"; do
    echo "rerun_group${GROUP_IDS[$idx]}=${RERUN_JOB_IDS[$idx]}"
  done
  for idx in "${!GROUP_IDS[@]}"; do
    echo "gamma0_group${GROUP_IDS[$idx]}=${GAMMA0_JOB_IDS[$idx]}"
  done
  for idx in "${!GROUP_IDS[@]}"; do
    echo "stop_group${GROUP_IDS[$idx]}=${STOP_JOB_IDS[$idx]}"
  done
  echo "launcher_seed42=${SEED42_LAUNCHER_JOB_ID}"
} | tee "${PROJECT_DIR}/${MANIFEST}"
