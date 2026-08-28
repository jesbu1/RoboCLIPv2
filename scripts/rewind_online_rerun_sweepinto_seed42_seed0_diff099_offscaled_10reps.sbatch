#!/usr/bin/env bash
# Rerun sweep-into-v2 ReWiND diff gamma=0.99, offline scaled:
# seed42 x 5 reps and seed0 x 5 reps.

#SBATCH --job-name=rw_on_sw_10rep
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-9

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"

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

ENV_ID="sweep-into-v2"
SEEDS=(42 0)
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
REP_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
SEED=${SEEDS[$SEED_IDX]}
CKPT_DIR="logs/offline_rewind_diff_gamma099_bonus0_scaled_10critics_seed${SEED}"
OFFLINE_H5_PATH="datasets/metaworld_labeled_diff_gamma099.h5"

if [ ! -f "${CKPT_DIR}/last_offline.zip" ]; then
  echo "ERROR: Offline checkpoint not found at ${CKPT_DIR}/last_offline.zip"
  exit 1
fi
if [ ! -f "$OFFLINE_H5_PATH" ]; then
  echo "ERROR: Offline dataset not found at ${OFFLINE_H5_PATH}"
  exit 1
fi

mkdir -p /scratch1/haobaizh/rewind_nochunk_online_logs

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
  offline_training.offline_h5_path=$OFFLINE_H5_PATH \
  offline_training.ckpt_path=$CKPT_DIR/last_offline \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=. \
  general_training.n_critics=10
