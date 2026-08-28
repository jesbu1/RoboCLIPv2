#!/usr/bin/env bash
#SBATCH --job-name=rw_on_d0999_sc_br_10c
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-23

# diff + gamma=0.999 + scale=1000 + base_reward=-1 + success_bonus=0 + 10 critics
# 8 tasks x 3 seeds = 24

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

ENVS=(
  window-close-v2
  reach-wall-v2
  faucet-close-v2
  coffee-button-v2
  button-press-wall-v2
  door-lock-v2
  handle-press-side-v2
  sweep-into-v2
)

SEEDS=(42 32 0)

ENV_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
ENV_ID=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}

CKPT_DIR="logs/offline_rewind_diff_gamma0999_bonus0_scaled_10critics_seed${SEED}"

mkdir -p /scratch1/haobaizh/rewind_nochunk_online_logs

LOG_DIR="/scratch1/haobaizh/rewind_nochunk_online_logs/${ENV_ID}_nochunk_rewind_diff_gamma0999_scaled_base_reward_10critics_seed${SEED}"
GROUP_NAME="${ENV_ID}_nochunk_rewind_diff_gamma0999_scaled_base_reward_10critics"
export WANDB_RUN_NAME="${ENV_ID}_nochunk_rewind_diff_gamma0999_scaled_base_reward_10critics_seed${SEED}"

rm -rf "$LOG_DIR"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.success_bonus=0 \
  reward_model.use_progress_diff=true \
  reward_model.diff_gamma=0.999 \
  reward_model.diff_reward_scale=1000 \
  reward_model.use_base_reward=true \
  reward_model.base_reward_value=-1.0 \
  general_training.seed=$SEED \
  environment.env_id=$ENV_ID \
  offline_training.offline_h5_path=datasets/metaworld_labeled_diff_gamma0999.h5 \
  offline_training.ckpt_path=$CKPT_DIR/last_offline \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=. \
  general_training.n_critics=10
