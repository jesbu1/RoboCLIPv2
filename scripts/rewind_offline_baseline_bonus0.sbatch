#!/usr/bin/env bash
#SBATCH --job-name=rw_off_bl_b0
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2

set -e

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"

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
ENV_ID="coffee-button-v2"

LOG_DIR="logs/offline_rewind_baseline_bonus0_seed${SEED}"
GROUP_NAME="offline_rewind_baseline_bonus0"
export WANDB_RUN_NAME="offline_rewind_baseline_bonus0_seed${SEED}"

rm -rf "$LOG_DIR"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.success_bonus=0 \
  general_training.seed=$SEED \
  environment.env_id=$ENV_ID \
  offline_training.offline_h5_path=datasets/metaworld_labeled.h5 \
  online_training.total_time_steps=0 \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=.
