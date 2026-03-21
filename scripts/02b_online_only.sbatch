#!/usr/bin/env bash
#SBATCH --job-name=online_only
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/project2/biyik_1165/haobaizh/rewind_no-action-chunk/logs/%x_%A_%a.out
#SBATCH --error=/project2/biyik_1165/haobaizh/rewind_no-action-chunk/logs/%x_%A_%a.err
#SBATCH --array=0-23

set -e

module purge
module load gcc/13.3.0
module load mesa/24.2.3
module load mesa-glu/9.0.2
module load glew/2.2.0
module load glx/1.4

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate /project2/biyik_1165/haobaizh/rewind_no-action-chunk/conda_envs/rewind_nochunk

export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
GCC_LIBDIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}"

export WANDB_CONFIG_DIR="$HOME/.config/wandb"
export WANDB_CACHE_DIR="$HOME/.cache/wandb"
export WANDB_DIR="$HOME/wandb"
export WANDB_MODE=online

cd /project2/biyik_1165/haobaizh/rewind_no-action-chunk

which python
python -c "import torch; print('torch=', torch.__version__)"

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

# 8 envs x 3 seeds = 24 jobs
ENV_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
ENV_ID=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Which reward type: set REWARD_TYPE and H5_PATH before submitting
REWARD_TYPE="${REWARD_TYPE:-baseline}"
H5_PATH="${H5_PATH:-datasets/metaworld_labeled.h5}"

# Set reward mode based on REWARD_TYPE
EXTRA_ARGS=""
if [[ "$REWARD_TYPE" == "diff" ]]; then
  EXTRA_ARGS="reward_model.use_progress_diff=true"
elif [[ "$REWARD_TYPE" == "diff_gamma099" ]]; then
  EXTRA_ARGS="reward_model.use_progress_diff=true reward_model.diff_gamma=0.99"
fi

# Checkpoint from offline-only run
CKPT_DIR="logs/offline_${REWARD_TYPE}_seed${SEED}"

GROUP_NAME="${ENV_ID}_nochunk_${REWARD_TYPE}"
export WANDB_RUN_NAME="${ENV_ID}_nochunk_${REWARD_TYPE}_seed${SEED}"
LOG_DIR="logs/${ENV_ID}_nochunk_${REWARD_TYPE}_seed${SEED}"

rm -rf "$LOG_DIR"

# Load offline checkpoint, only run online
srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  general_training.seed=$SEED \
  environment.env_id=$ENV_ID \
  offline_training.offline_h5_path=$H5_PATH \
  offline_training.ckpt_path=$CKPT_DIR/last_offline \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=. \
  $EXTRA_ARGS
