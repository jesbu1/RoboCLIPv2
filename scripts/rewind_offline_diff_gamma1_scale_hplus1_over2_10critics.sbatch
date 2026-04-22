#!/usr/bin/env bash
#SBATCH --job-name=rw_off_d1_h_10c
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2

# Offline IQL: diff_gamma=1.0, 10 critics
# Scale heuristic follows (HORIZON + 1) / 2.
# Since offline buffer loading divides by reward_divisor, we set:
#   reward_divisor = 2 / (HORIZON + 1)
# 3 seeds (array 0->seed42, 1->seed32, 2->seed0)

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
HORIZON="${HORIZON:-128}"

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

SEEDS=(42 32 0)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

DIFF_REWARD_SCALE=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", (h + 1) / 2 }')
REWARD_DIVISOR=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", 2 / (h + 1) }')
TAG="diff_gamma1_scale_hplus1_over2_H${HORIZON}"
LOG_DIR="logs/offline_rewind_${TAG}_10critics_seed${SEED}"
GROUP_NAME="offline_rewind_${TAG}_10critics"
export WANDB_RUN_NAME="offline_rewind_${TAG}_10critics_seed${SEED}"

rm -rf "$LOG_DIR"

echo "========== ReWiND offline ${TAG} seed ${SEED} =========="
echo "HORIZON=${HORIZON}"
echo "diff_reward_scale=(HORIZON+1)/2=${DIFF_REWARD_SCALE}"
echo "reward_divisor=2/(HORIZON+1)=${REWARD_DIVISOR}"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.success_bonus=0 \
  reward_model.use_progress_diff=true \
  reward_model.diff_gamma=1.0 \
  general_training.seed=$SEED \
  environment.env_id=coffee-button-v2 \
  offline_training.offline_h5_path=datasets/metaworld_labeled_diff_gamma1.h5 \
  online_training.total_time_steps=0 \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=. \
  general_training.n_critics=10 \
  reward_model.reward_divisor="$REWARD_DIVISOR"
