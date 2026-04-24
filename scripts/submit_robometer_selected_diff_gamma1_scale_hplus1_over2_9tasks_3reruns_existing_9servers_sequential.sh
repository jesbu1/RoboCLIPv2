#!/usr/bin/env bash
# Submit 9 selected Robometer H128 online task-seed combos, 3 reruns each.
# Each existing server runs its 3 online reruns sequentially via afterok.
# The first 6 groups cover all selected task families, including sweep-into,
# and balance seeds as much as the selected combos allow.
# No Slurm arrays. WandB names are produced by the single-task sbatch unchanged.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVER_JOB_IDS="${SERVER_JOB_IDS:?SERVER_JOB_IDS must contain 9 server ids in group40..48 order}"
MAX_FRAMES="${MAX_FRAMES:-4}"
HORIZON="${HORIZON:-128}"
REPEAT_COUNT="${REPEAT_COUNT:-3}"

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

TASK_SPECS=(
  "faucet-close-v2:0"
  "reach-wall-v2:32"
  "handle-press-side-v2:32"
  "window-close-v2:42"
  "sweep-into-v2:42"
  "reach-wall-v2:0"
  "faucet-close-v2:42"
  "faucet-close-v2:32"
  "reach-wall-v2:42"
)

DIFF_REWARD_SCALE=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", (h + 1) / 2 }')
MANIFEST="${PROJECT_DIR}/logs/robometer_selected_diff_gamma1_scale_hplus1_over2_H${HORIZON}_9tasks_3reruns_existing_9servers_sequential_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "manifest=${MANIFEST}"
  echo "project_dir=${PROJECT_DIR}"
  echo "variant=diff_gamma1_scale_hplus1_over2_H${HORIZON}_base_reward"
  echo "diff_gamma=1.0"
  echo "horizon=${HORIZON}"
  echo "diff_reward_scale=(HORIZON+1)/2=${DIFF_REWARD_SCALE}"
  echo "repeat_count=${REPEAT_COUNT}"
  echo "selected_combos=9"
  echo "tasks_total=$((9 * REPEAT_COUNT))"
  echo "servers_total=9"
  echo "array_usage=disabled"
  echo "rerun_dependency=afterok_previous_rerun"
  echo "wandb_names=unchanged"
  echo "online_training.mix_buffers_ratio=0.0"
  echo "success_bonus=0"
  echo "base_reward_value=-1.0"
} | tee "$MANIFEST"

for idx in "${!TASK_SPECS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  server_job_id="${SERVER_IDS[$idx]}"
  info_file="${INFO_FILES[$idx]}"
  task_spec="${TASK_SPECS[$idx]}"
  IFS=':' read -r env_id seed <<< "$task_spec"
  previous_online_id=""

  echo "combo_$((idx + 1))=${env_id}:seed${seed}:server_group${group_id}:server_job${server_job_id}" | tee -a "$MANIFEST"

  for rep in $(seq 1 "$REPEAT_COUNT"); do
    if [ "$rep" -eq 1 ]; then
      dependency="after:${server_job_id}"
      dependency_note="server_started:${server_job_id}"
    else
      dependency="afterok:${previous_online_id}"
      dependency_note="previous_online_success:${previous_online_id}"
    fi

    online_id=$(sbatch --parsable \
      --job-name="robo_on_h_sel_g${group_id}_r${rep}" \
      --dependency="$dependency" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",INFO_FILE="$info_file",EXPECTED_SERVER_JOB_ID="$server_job_id",GROUP_ID="$group_id",ENV_ID="$env_id",SEED="$seed",REP_IDX="$rep",MAX_FRAMES="$MAX_FRAMES",HORIZON="$HORIZON" \
      scripts/robometer_online_diff_gamma1_scale_hplus1_over2_selected_single_task.sbatch)

    echo "online_${env_id}_seed${seed}_rep${rep}=${online_id} depends_on=${dependency_note}" | tee -a "$MANIFEST"
    previous_online_id="$online_id"
  done
done
