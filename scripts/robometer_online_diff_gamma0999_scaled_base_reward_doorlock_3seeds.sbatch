#!/usr/bin/env bash
# online diff gamma=0.999 Robometer on door-lock-v2, 3 seeds
# scale=1000, base reward enabled

#SBATCH --job-name=robo_on_g0999_sc_br_dl_3s
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
INFO_FILE="${PROJECT_DIR}/logs/robometer_server_doorlock_pipeline_info.txt"

module purge
module load gcc/13.3.0
module load mesa/24.2.3
module load mesa-glu/9.0.2
module load glew/2.2.0
module load glx/1.4

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "${PROJECT_DIR}/conda_envs/rewind_nochunk"

export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
GCC_LIBDIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}"

export WANDB_CONFIG_DIR="$HOME/.config/wandb"
export WANDB_CACHE_DIR="$HOME/.cache/wandb"
export WANDB_DIR="$HOME/wandb"
export WANDB_MODE=online

cd "$PROJECT_DIR"

SEEDS=(42 32 0)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
ENV_ID="door-lock-v2"

for i in $(seq 1 360); do
  if [ -f "$INFO_FILE" ]; then break; fi
  if [ $i -eq 360 ]; then echo "ERROR: Server info not found after 30min"; exit 1; fi
  sleep 5
done
SERVER_URL=$(cat "$INFO_FILE")

for i in $(seq 1 360); do
  if curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then echo "Server ready!"; break; fi
  if [ $i -eq 360 ]; then echo "ERROR: Server not ready after 30min"; exit 1; fi
  sleep 5
done

CKPT_DIR="logs/offline_robometer_diff_gamma0999_bonus0_scaled_seed${SEED}"
LOG_DIR="logs/${ENV_ID}_nochunk_robometer_diff_gamma0999_scaled_base_reward_seed${SEED}"
GROUP_NAME="${ENV_ID}_nochunk_robometer_diff_gamma0999_scaled_base_reward"
export WANDB_RUN_NAME="${ENV_ID}_nochunk_robometer_diff_gamma0999_scaled_base_reward_seed${SEED}"

mkdir -p "$LOG_DIR"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=robometer \
  reward_model.use_server=true \
  reward_model.server_url="$SERVER_URL" \
  reward_model.success_bonus=0 \
  reward_model.use_progress_diff=true \
  reward_model.diff_gamma=0.999 \
  reward_model.diff_reward_scale=1000 \
  reward_model.use_base_reward=true \
  reward_model.base_reward_value=-1.0 \
  general_training.seed=$SEED \
  environment.env_id=$ENV_ID \
  offline_training.offline_h5_path=datasets/metaworld_labeled_robometer_diff_gamma0999.h5 \
  offline_training.ckpt_path=$CKPT_DIR/last_offline \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=.
