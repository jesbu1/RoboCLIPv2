#!/usr/bin/env bash
#SBATCH --job-name=rw_on_d1_h_br10c
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-23

# Online IQL: diff_gamma=1.0, base_reward=-1, success_bonus=0, 10 critics
# Scale heuristic follows (HORIZON + 1) / 2.
# 8 tasks x 3 seeds = 24 (array 0-23)

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
HORIZON="${HORIZON:-128}"

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
export LD_LIBRARY_PATH="${GCC_LIBDIR}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

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

DIFF_REWARD_SCALE=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", (h + 1) / 2 }')
TAG="diff_gamma1_scale_hplus1_over2_H${HORIZON}"
CKPT_DIR="logs/offline_rewind_${TAG}_10critics_seed${SEED}"

mkdir -p /scratch1/haobaizh/rewind_nochunk_online_logs

LOG_DIR="/scratch1/haobaizh/rewind_nochunk_online_logs/${ENV_ID}_nochunk_rewind_${TAG}_base_reward_10critics_seed${SEED}"
GROUP_NAME="${ENV_ID}_nochunk_rewind_${TAG}_base_reward_10critics"
export WANDB_RUN_NAME="${ENV_ID}_nochunk_rewind_${TAG}_base_reward_10critics_seed${SEED}"

rm -rf "$LOG_DIR"

echo "========== ReWiND online ${TAG} ${ENV_ID} seed ${SEED} =========="
echo "HORIZON=${HORIZON}"
echo "diff_reward_scale=(HORIZON+1)/2=${DIFF_REWARD_SCALE}"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.success_bonus=0 \
  reward_model.use_progress_diff=true \
  reward_model.diff_gamma=1.0 \
  reward_model.diff_reward_scale="$DIFF_REWARD_SCALE" \
  reward_model.use_base_reward=true \
  reward_model.base_reward_value=-1.0 \
  general_training.seed=$SEED \
  environment.env_id=$ENV_ID \
  offline_training.offline_training_steps=0 \
  offline_training.offline_h5_path=datasets/metaworld_labeled_diff_gamma1.h5 \
  offline_training.ckpt_path=$CKPT_DIR/last_offline \
  online_training.mix_buffers_ratio=0.0 \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=. \
  general_training.n_critics=10
