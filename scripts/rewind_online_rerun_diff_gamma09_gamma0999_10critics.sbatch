#!/usr/bin/env bash
#SBATCH --job-name=rw_on_rerun_g09_g0999
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-19

# Diff tasks with gamma=0.9 and gamma=0.999, scaled, 10 critics, 1 rep each
# Tasks: sweep-into(42,32,0), faucet-close(42,32,0), reach-wall(42), window-close(42,32,0)
# 10 task/seed combos x 2 gammas = 20

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

# 10 task/seed combos
COMBO_ENVS=(
  sweep-into-v2 sweep-into-v2 sweep-into-v2
  faucet-close-v2 faucet-close-v2 faucet-close-v2
  reach-wall-v2
  window-close-v2 window-close-v2 window-close-v2
)
COMBO_SEEDS=(
  42 32 0
  42 32 0
  42
  42 32 0
)

# 10 combos x 2 gammas = 20 tasks
# array 0-9: gamma=0.9, array 10-19: gamma=0.999
GAMMA_IDX=$(( SLURM_ARRAY_TASK_ID / 10 ))
COMBO_IDX=$(( SLURM_ARRAY_TASK_ID % 10 ))

ENV_ID=${COMBO_ENVS[$COMBO_IDX]}
SEED=${COMBO_SEEDS[$COMBO_IDX]}

if [ $GAMMA_IDX -eq 0 ]; then
  GAMMA=0.9
  GAMMA_TAG="gamma09"
  SCALE=10
  H5_PATH="datasets/metaworld_labeled_diff_gamma09.h5"
  CKPT_DIR="logs/offline_rewind_diff_gamma09_bonus0_scaled_10critics_seed${SEED}"
else
  GAMMA=0.999
  GAMMA_TAG="gamma0999"
  SCALE=1000
  H5_PATH="datasets/metaworld_labeled_diff_gamma0999.h5"
  CKPT_DIR="logs/offline_rewind_diff_gamma0999_bonus0_scaled_10critics_seed${SEED}"
fi

mkdir -p /scratch1/haobaizh/rewind_nochunk_online_logs

LOG_DIR="/scratch1/haobaizh/rewind_nochunk_online_logs/${ENV_ID}_nochunk_rewind_diff_${GAMMA_TAG}_scaled_base_reward_offscaled_10critics_seed${SEED}"
GROUP_NAME="${ENV_ID}_nochunk_rewind_diff_${GAMMA_TAG}_scaled_base_reward_offscaled_10critics"
export WANDB_RUN_NAME="${ENV_ID}_nochunk_rewind_diff_${GAMMA_TAG}_scaled_base_reward_offscaled_10critics_seed${SEED}"

rm -rf "$LOG_DIR"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.success_bonus=0 \
  reward_model.use_progress_diff=true \
  reward_model.diff_gamma=$GAMMA \
  reward_model.diff_reward_scale=$SCALE \
  reward_model.use_base_reward=true \
  reward_model.base_reward_value=-1.0 \
  general_training.seed=$SEED \
  environment.env_id=$ENV_ID \
  offline_training.offline_h5_path=$H5_PATH \
  offline_training.ckpt_path=$CKPT_DIR/last_offline \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=. \
  general_training.n_critics=10
