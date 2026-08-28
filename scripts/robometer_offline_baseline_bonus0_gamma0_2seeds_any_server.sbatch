#!/usr/bin/env bash
# Offline Robometer baseline pretraining with RL gamma=0 for seeds 0 and 32.

#SBATCH --job-name=robo_off_bl_g0
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
SERVER_INFO_FILES="${SERVER_INFO_FILES:-}"
SERVER_JOB_IDS="${SERVER_JOB_IDS:-}"
LOG_GROUP_NAME="${LOG_GROUP_NAME:-offline_robometer_baseline_bonus0_gamma0}"

if [ ! -f "${PROJECT_DIR}/test_scripts/test_iql.py" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"
  exit 1
fi
if [ ! -d "${PROJECT_DIR}/conda_envs/rewind_nochunk" ]; then
  echo "ERROR: rewind_nochunk conda env not found under ${PROJECT_DIR}/conda_envs/rewind_nochunk"
  exit 1
fi
if [ -z "$SERVER_INFO_FILES" ]; then
  echo "ERROR: SERVER_INFO_FILES is empty"
  exit 1
fi

module purge
module load gcc/13.3.0
module load mesa/24.2.3
module load mesa-glu/9.0.2
module load glew/2.2.0
module load glx/1.4

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "${PROJECT_DIR}/conda_envs/rewind_nochunk"

export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
GCC_LIBDIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

export WANDB_CONFIG_DIR="$HOME/.config/wandb"
export WANDB_CACHE_DIR="$HOME/.cache/wandb"
export WANDB_DIR="$HOME/wandb"
export WANDB_MODE=online

cd "$PROJECT_DIR"

IFS='|' read -r -a INFO_FILES <<< "$SERVER_INFO_FILES"
IFS='|' read -r -a SERVER_IDS <<< "$SERVER_JOB_IDS"
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
  if [ -n "$SERVER_JOB_IDS" ] && [[ "$ALLOWED_SERVER_IDS" != *"|${server_job_id}|"* ]]; then
    return 1
  fi
  if [ -z "$server_url" ] || [[ "$server_url" != http* ]]; then
    return 1
  fi
  printf '%s\n' "$server_url"
  return 0
}

SERVER_URL=""
for i in $(seq 1 360); do
  for info_file in "${INFO_FILES[@]}"; do
    if [ -s "$info_file" ] && candidate_url="$(read_server_url "$info_file")"; then
      if curl -s "${candidate_url}/health" > /dev/null 2>&1; then
        SERVER_URL="$candidate_url"
        break 2
      fi
    fi
  done
  if [ $i -eq 360 ]; then
    echo "ERROR: No healthy server found after 30 minutes"
    exit 1
  fi
  sleep 5
done

echo "Using server: ${SERVER_URL}"

SEEDS=(0 32)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

LOG_DIR="logs/offline_robometer_baseline_bonus0_gamma0_seed${SEED}"
mkdir -p "$LOG_DIR"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=robometer \
  reward_model.use_server=true \
  reward_model.server_url="$SERVER_URL" \
  reward_model.success_bonus=0 \
  general_training.gamma=0 \
  general_training.seed=$SEED \
  offline_training.offline_h5_path=datasets/metaworld_labeled_robometer.h5 \
  online_training.total_time_steps=0 \
  logging.wandb_group_name="$LOG_GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=.
