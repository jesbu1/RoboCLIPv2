#!/usr/bin/env bash
# Offline IQL for pure ReWiND reverse-diff variants.

#SBATCH --job-name=rw_off_rev_10c
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
VARIANT="${VARIANT:?VARIANT must be gamma099_scale100, gamma0999_scale1000, or gamma09999_scale10000}"

case "$VARIANT" in
  gamma099_scale100)
    DIFF_GAMMA="0.99"
    DIFF_REWARD_SCALE="100"
    REWARD_DIVISOR="0.01"
    H5_PATH="datasets/metaworld_labeled_reverse_diff_gamma099.h5"
    TAG="reverse_diff_gamma099_scale100"
    ;;
  gamma0999_scale1000)
    DIFF_GAMMA="0.999"
    DIFF_REWARD_SCALE="1000"
    REWARD_DIVISOR="0.001"
    H5_PATH="datasets/metaworld_labeled_reverse_diff_gamma0999.h5"
    TAG="reverse_diff_gamma0999_scale1000"
    ;;
  gamma09999_scale10000)
    DIFF_GAMMA="0.9999"
    DIFF_REWARD_SCALE="10000"
    REWARD_DIVISOR="0.0001"
    H5_PATH="datasets/metaworld_labeled_reverse_diff_gamma09999.h5"
    TAG="reverse_diff_gamma09999_scale10000"
    ;;
  *)
    echo "ERROR: Unknown VARIANT=${VARIANT}"
    exit 1
    ;;
esac

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

SEEDS=(0 32 42)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
ENV_ID="coffee-button-v2"

LOG_DIR="logs/offline_rewind_${TAG}_10critics_seed${SEED}"
GROUP_NAME="offline_rewind_${TAG}_10critics"
export WANDB_RUN_NAME="offline_rewind_${TAG}_10critics_seed${SEED}"

if [ ! -f "$H5_PATH" ]; then
  echo "ERROR: Missing labeled H5: ${H5_PATH}"
  exit 1
fi

rm -rf "$LOG_DIR"

echo "========== ReWiND reverse-diff offline =========="
echo "VARIANT=${VARIANT}"
echo "SEED=${SEED}"
echo "DIFF_GAMMA=${DIFF_GAMMA}"
echo "DIFF_REWARD_SCALE=${DIFF_REWARD_SCALE}"
echo "REWARD_DIVISOR=${REWARD_DIVISOR}"
echo "H5_PATH=${H5_PATH}"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.success_bonus=0 \
  reward_model.use_progress_diff=true \
  reward_model.use_reverse_progress_diff=true \
  reward_model.diff_gamma="$DIFF_GAMMA" \
  reward_model.diff_reward_scale="$DIFF_REWARD_SCALE" \
  general_training.seed="$SEED" \
  environment.env_id="$ENV_ID" \
  offline_training.offline_h5_path="$H5_PATH" \
  online_training.total_time_steps=0 \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=. \
  general_training.n_critics=10 \
  reward_model.reward_divisor="$REWARD_DIVISOR"
