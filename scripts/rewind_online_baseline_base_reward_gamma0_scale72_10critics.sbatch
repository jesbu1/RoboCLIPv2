#!/usr/bin/env bash
# Online ReWiND baseline on 8 tasks x 3 seeds with RL gamma=0 and learned reward scale=72.

#SBATCH --job-name=rw_on_bl_g0_s72_10c
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-23

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
REWARD_SCALE="${REWARD_SCALE:-72}"
REWARD_DIVISOR="${REWARD_DIVISOR:-0.0138888889}"
LOG_ROOT="${LOG_ROOT:-/scratch1/haobaizh/rewind_nochunk_online_logs}"

if [ ! -f "${PROJECT_DIR}/test_scripts/test_iql.py" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"
  exit 1
fi
if [ ! -d "${PROJECT_DIR}/conda_envs/rewind_nochunk" ]; then
  echo "ERROR: rewind_nochunk conda env not found under ${PROJECT_DIR}/conda_envs/rewind_nochunk"
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

SEEDS=(0 32 42)

ENV_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
ENV_ID=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}

CKPT_DIR="logs/offline_rewind_baseline_bonus0_gamma0_scale72_10critics_seed${SEED}"
OFFLINE_H5_PATH="datasets/metaworld_labeled.h5"
if [ ! -f "${CKPT_DIR}/last_offline.zip" ]; then
  echo "ERROR: Offline gamma0 scale72 checkpoint not found at ${CKPT_DIR}/last_offline.zip"
  exit 1
fi
if [ ! -f "$OFFLINE_H5_PATH" ]; then
  echo "ERROR: Offline dataset not found at ${OFFLINE_H5_PATH}"
  exit 1
fi

GROUP_NAME="${ENV_ID}_nochunk_rewind_baseline_base_reward_gamma0_scale72_10critics"
RUN_NAME="${ENV_ID}_nochunk_rewind_baseline_base_reward_gamma0_scale72_10critics_seed${SEED}"
LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
export WANDB_RUN_NAME="$RUN_NAME"

echo "Running online ReWiND baseline gamma0 scale${REWARD_SCALE}: ${ENV_ID} seed${SEED}"
mkdir -p "$LOG_ROOT"
rm -rf "$LOG_DIR"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.success_bonus=0 \
  reward_model.reward_divisor="${REWARD_DIVISOR}" \
  reward_model.use_base_reward=true \
  reward_model.base_reward_value=-1.0 \
  general_training.gamma=0 \
  general_training.seed=$SEED \
  environment.env_id=$ENV_ID \
  offline_training.offline_h5_path=$OFFLINE_H5_PATH \
  offline_training.ckpt_path=$CKPT_DIR/last_offline \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=. \
  general_training.n_critics=10
