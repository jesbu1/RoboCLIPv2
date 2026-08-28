#!/usr/bin/env bash
#SBATCH --job-name=rw_on_rerun
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-63

# Rerun specific task/seed combos, 4 reps each, 10 critics
# Baseline: offline_rewind_baseline_bonus0_10critics + base_reward=-1
# Diff: offline_rewind_diff_gamma099_bonus0_scaled_10critics + scale=100 + base_reward=-1

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

# 16 combos: ENV METHOD BASE_SEED
# Baseline (6):
#   0: sweep-into baseline 42
#   1: sweep-into baseline 32
#   2: button-press-wall baseline 42
#   3: button-press-wall baseline 0
#   4: reach-wall baseline 42
#   5: reach-wall baseline 0
# Diff (10):
#   6: sweep-into diff 42
#   7: sweep-into diff 32
#   8: sweep-into diff 0
#   9: faucet-close diff 42
#  10: faucet-close diff 32
#  11: faucet-close diff 0
#  12: reach-wall diff 42
#  13: window-close diff 42
#  14: window-close diff 32
#  15: window-close diff 0

COMBO_ENVS=(
  sweep-into-v2 sweep-into-v2
  button-press-wall-v2 button-press-wall-v2
  reach-wall-v2 reach-wall-v2
  sweep-into-v2 sweep-into-v2 sweep-into-v2
  faucet-close-v2 faucet-close-v2 faucet-close-v2
  reach-wall-v2
  window-close-v2 window-close-v2 window-close-v2
)
COMBO_METHODS=(
  baseline baseline
  baseline baseline
  baseline baseline
  diff diff diff
  diff diff diff
  diff
  diff diff diff
)
COMBO_SEEDS=(
  42 32
  42 0
  42 0
  42 32 0
  42 32 0
  42
  42 32 0
)

# 16 combos x 4 reps = 64 tasks
COMBO_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
REP_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))

ENV_ID=${COMBO_ENVS[$COMBO_IDX]}
METHOD=${COMBO_METHODS[$COMBO_IDX]}
BASE_SEED=${COMBO_SEEDS[$COMBO_IDX]}

# Same seed for all 4 reps
SEED=$BASE_SEED

mkdir -p /scratch1/haobaizh/rewind_nochunk_online_logs

if [ "$METHOD" == "baseline" ]; then
  CKPT_DIR="logs/offline_rewind_baseline_bonus0_10critics_seed${BASE_SEED}"
  LOG_DIR="/scratch1/haobaizh/rewind_nochunk_online_logs/${ENV_ID}_nochunk_rewind_baseline_base_reward_10critics_seed${SEED}_rep${REP_IDX}"
  GROUP_NAME="${ENV_ID}_nochunk_rewind_baseline_base_reward_10critics"
  export WANDB_RUN_NAME="${ENV_ID}_nochunk_rewind_baseline_base_reward_10critics_seed${SEED}"

  rm -rf "$LOG_DIR"

  srun python -u test_scripts/test_iql.py \
    metaworld=off_on_15 \
    algorithm=wsrl_iql \
    reward=rewind \
    reward_model.success_bonus=0 \
    reward_model.use_base_reward=true \
    reward_model.base_reward_value=-1.0 \
    general_training.seed=$SEED \
    environment.env_id=$ENV_ID \
    offline_training.offline_h5_path=datasets/metaworld_labeled.h5 \
    offline_training.ckpt_path=$CKPT_DIR/last_offline \
    logging.wandb_group_name="$GROUP_NAME" \
    logging.log_dir="$LOG_DIR" \
    hydra.run.dir=. \
    general_training.n_critics=10

elif [ "$METHOD" == "diff" ]; then
  CKPT_DIR="logs/offline_rewind_diff_gamma099_bonus0_scaled_10critics_seed${BASE_SEED}"
  LOG_DIR="/scratch1/haobaizh/rewind_nochunk_online_logs/${ENV_ID}_nochunk_rewind_diff_gamma099_scaled_base_reward_offscaled_10critics_seed${SEED}_rep${REP_IDX}"
  GROUP_NAME="${ENV_ID}_nochunk_rewind_diff_gamma099_scaled_base_reward_offscaled_10critics"
  export WANDB_RUN_NAME="${ENV_ID}_nochunk_rewind_diff_gamma099_scaled_base_reward_offscaled_10critics_seed${SEED}"

  rm -rf "$LOG_DIR"

  srun python -u test_scripts/test_iql.py \
    metaworld=off_on_15 \
    algorithm=wsrl_iql \
    reward=rewind \
    reward_model.success_bonus=0 \
    reward_model.use_progress_diff=true \
    reward_model.diff_gamma=0.99 \
    reward_model.diff_reward_scale=100 \
    reward_model.use_base_reward=true \
    reward_model.base_reward_value=-1.0 \
    general_training.seed=$SEED \
    environment.env_id=$ENV_ID \
    offline_training.offline_h5_path=datasets/metaworld_labeled_diff_gamma099.h5 \
    offline_training.ckpt_path=$CKPT_DIR/last_offline \
    logging.wandb_group_name="$GROUP_NAME" \
    logging.log_dir="$LOG_DIR" \
    hydra.run.dir=. \
    general_training.n_critics=10
fi
