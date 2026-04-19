#!/usr/bin/env bash
# Robometer diff gamma=1.0 offline, success_bonus=0, scale=100, 3 seeds.

#SBATCH --job-name=robo_off_d1_s100
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
INFO_FILE="${INFO_FILE:?INFO_FILE must point to a ready Robometer server info file}"
EXPECTED_SERVER_JOB_ID="${EXPECTED_SERVER_JOB_ID:-}"
MAX_FRAMES="${MAX_FRAMES:-4}"

if [ ! -f "${PROJECT_DIR}/test_scripts/test_iql.py" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"
  exit 1
fi
if [ ! -d "${PROJECT_DIR}/conda_envs/rewind_nochunk" ]; then
  echo "ERROR: rewind_nochunk conda env not found under ${PROJECT_DIR}/conda_envs/rewind_nochunk"
  exit 1
fi
if [ ! -f "${PROJECT_DIR}/datasets/metaworld_labeled_robometer_diff_gamma1.h5" ]; then
  echo "ERROR: label H5 not found: ${PROJECT_DIR}/datasets/metaworld_labeled_robometer_diff_gamma1.h5"
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

read_server_url() {
  local info_line server_job_id server_url
  info_line=$(tr -d '\r\n' < "$INFO_FILE")
  if [ -z "$info_line" ]; then
    return 1
  fi
  if [[ "$info_line" == http* ]]; then
    printf '%s\n' "$info_line"
    return 0
  fi

  server_job_id="${info_line%% *}"
  server_url="${info_line#* }"
  if [ -n "$EXPECTED_SERVER_JOB_ID" ] && [ "$server_job_id" != "$EXPECTED_SERVER_JOB_ID" ]; then
    return 1
  fi
  if [ -z "$server_url" ] || [[ "$server_url" != http* ]]; then
    return 1
  fi

  printf '%s\n' "$server_url"
}

for i in $(seq 1 1440); do
  if [ -s "$INFO_FILE" ] && SERVER_URL="$(read_server_url)"; then
    break
  fi
  if [ "$i" -eq 1440 ]; then echo "ERROR: Server info not found after 120min"; exit 1; fi
  sleep 5
done
if [ -z "${SERVER_URL:-}" ]; then echo "ERROR: Server URL is empty"; exit 1; fi

for i in $(seq 1 1440); do
  if curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then echo "Server ready!"; break; fi
  if [ "$i" -eq 1440 ]; then echo "ERROR: Server not ready after 120min"; exit 1; fi
  sleep 5
done

SEEDS=(42 32 0)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
LOG_DIR="logs/offline_robometer_diff_gamma1_scale100_10critics_seed${SEED}"
GROUP_NAME="offline_robometer_diff_gamma1_scale100_10critics"
export WANDB_RUN_NAME="offline_robometer_diff_gamma1_scale100_10critics_seed${SEED}"

rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=robometer \
  reward_model.use_server=true \
  reward_model.server_url="$SERVER_URL" \
  reward_model.success_bonus=0 \
  reward_model.max_frames="$MAX_FRAMES" \
  reward_model.use_progress_diff=true \
  reward_model.diff_gamma=1.0 \
  reward_model.reward_divisor=0.01 \
  offline_training.offline_training_steps=100000 \
  online_training.total_time_steps=0 \
  general_training.seed=$SEED \
  general_training.n_critics=10 \
  general_training.n_critics_to_sample=2 \
  offline_training.offline_h5_path=datasets/metaworld_labeled_robometer_diff_gamma1.h5 \
  logging.eval_freq=0 \
  logging.video_freq=0 \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=.
