#!/usr/bin/env bash
# Submit 18 Robometer H128 online jobs on 9 existing servers.
# Each server runs exactly two online jobs sequentially: pass2 starts after pass1 succeeds.
# No Slurm arrays. WandB group/run names are produced by the single-task sbatch unchanged.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVER_JOB_IDS="${SERVER_JOB_IDS:?SERVER_JOB_IDS must contain 9 server ids in group40..48 order}"
MAX_FRAMES="${MAX_FRAMES:-4}"
HORIZON="${HORIZON:-128}"

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

if [[ "$SERVER_JOB_IDS" == *"|"* ]]; then
  IFS='|' read -r -a SERVER_IDS <<< "$SERVER_JOB_IDS"
else
  read -r -a SERVER_IDS <<< "$SERVER_JOB_IDS"
fi

if [ "${#SERVER_IDS[@]}" -ne 9 ]; then
  echo "ERROR: Expected 9 server ids, got ${#SERVER_IDS[@]}: ${SERVER_JOB_IDS}"
  exit 1
fi

GROUP_IDS=(40 41 42 43 44 45 46 47 48)
INFO_FILES=()
for group_id in "${GROUP_IDS[@]}"; do
  INFO_FILES+=("${PROJECT_DIR}/logs/robometer_server_9srv_group${group_id}_info.txt")
done

# Format: env_id:seed:rep_idx
# Pass 1 includes the seven replacement jobs from the canceled third rerun
# excluding window-close and handle-press, plus button-press seed0/seed32.
PASS1_SPECS=(
  "faucet-close-v2:0:3"
  "reach-wall-v2:32:3"
  "sweep-into-v2:42:3"
  "reach-wall-v2:0:3"
  "faucet-close-v2:42:3"
  "faucet-close-v2:32:3"
  "reach-wall-v2:42:3"
  "button-press-wall-v2:0:1"
  "button-press-wall-v2:32:1"
)

# Pass 2 adds two button-press seed0, two seed32, three seed42, and
# two handle-press-side seed0 jobs.
PASS2_SPECS=(
  "button-press-wall-v2:0:2"
  "button-press-wall-v2:0:3"
  "button-press-wall-v2:32:2"
  "button-press-wall-v2:32:3"
  "button-press-wall-v2:42:1"
  "button-press-wall-v2:42:2"
  "button-press-wall-v2:42:3"
  "handle-press-side-v2:0:1"
  "handle-press-side-v2:0:2"
)

DIFF_REWARD_SCALE=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", (h + 1) / 2 }')
MANIFEST="${PROJECT_DIR}/logs/robometer_buttonpress_handlepress_2pass_existing_9servers_H${HORIZON}_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "manifest=${MANIFEST}"
  echo "project_dir=${PROJECT_DIR}"
  echo "variant=diff_gamma1_scale_hplus1_over2_H${HORIZON}_base_reward"
  echo "diff_gamma=1.0"
  echo "horizon=${HORIZON}"
  echo "diff_reward_scale=(HORIZON+1)/2=${DIFF_REWARD_SCALE}"
  echo "servers_total=9"
  echo "tasks_total=18"
  echo "tasks_per_server=2"
  echo "array_usage=disabled"
  echo "pass2_dependency=afterok_pass1"
  echo "wandb_names=unchanged"
  echo "online_training.mix_buffers_ratio=0.0"
  echo "success_bonus=0"
  echo "base_reward_value=-1.0"
} | tee "$MANIFEST"

for idx in "${!GROUP_IDS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  server_job_id="${SERVER_IDS[$idx]}"
  info_file="${INFO_FILES[$idx]}"
  pass1_spec="${PASS1_SPECS[$idx]}"
  pass2_spec="${PASS2_SPECS[$idx]}"

  IFS=':' read -r pass1_env pass1_seed pass1_rep <<< "$pass1_spec"
  IFS=':' read -r pass2_env pass2_seed pass2_rep <<< "$pass2_spec"

  echo "server_group${group_id}=${server_job_id}" | tee -a "$MANIFEST"
  echo "pass1_group${group_id}=${pass1_env}:seed${pass1_seed}:rep${pass1_rep}" | tee -a "$MANIFEST"
  echo "pass2_group${group_id}=${pass2_env}:seed${pass2_seed}:rep${pass2_rep}" | tee -a "$MANIFEST"

  pass1_id=$(sbatch --parsable \
    --job-name="robo_on_h_mix_g${group_id}_p1" \
    --dependency="after:${server_job_id}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",INFO_FILE="$info_file",EXPECTED_SERVER_JOB_ID="$server_job_id",GROUP_ID="$group_id",ENV_ID="$pass1_env",SEED="$pass1_seed",REP_IDX="$pass1_rep",MAX_FRAMES="$MAX_FRAMES",HORIZON="$HORIZON" \
    scripts/robometer_online_diff_gamma1_scale_hplus1_over2_selected_single_task.sbatch)

  echo "online_pass1_group${group_id}_${pass1_env}_seed${pass1_seed}_rep${pass1_rep}=${pass1_id} depends_on=server_started:${server_job_id}" | tee -a "$MANIFEST"

  pass2_id=$(sbatch --parsable \
    --job-name="robo_on_h_mix_g${group_id}_p2" \
    --dependency="afterok:${pass1_id}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",INFO_FILE="$info_file",EXPECTED_SERVER_JOB_ID="$server_job_id",GROUP_ID="$group_id",ENV_ID="$pass2_env",SEED="$pass2_seed",REP_IDX="$pass2_rep",MAX_FRAMES="$MAX_FRAMES",HORIZON="$HORIZON" \
    scripts/robometer_online_diff_gamma1_scale_hplus1_over2_selected_single_task.sbatch)

  echo "online_pass2_group${group_id}_${pass2_env}_seed${pass2_seed}_rep${pass2_rep}=${pass2_id} depends_on=previous_online_success:${pass1_id}" | tee -a "$MANIFEST"
done
