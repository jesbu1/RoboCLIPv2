#!/usr/bin/env bash
# Train TOPReward offline checkpoints for seeds 0, 32, 42.

#SBATCH --job-name=top_off_3s
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

TOPREWARD_DIR="${TOPREWARD_DIR:-/scratch1/haobaizh/rewind_topreward}"
TOPREWARD_CONDA="${TOPREWARD_CONDA:-${TOPREWARD_DIR}/conda_envs/rewind}"
PROJECT_CACHE="${PROJECT_CACHE:-/scratch1/haobaizh/.cache}"
VARIANT="${VARIANT:?VARIANT must be baseline or diff_gamma0999}"
SUCCESS_BONUS="${SUCCESS_BONUS:-64.0}"
DIFF_GAMMA="${DIFF_GAMMA:-0.999}"
DIFF_REWARD_SCALE="${DIFF_REWARD_SCALE:-1.0}"

SEEDS=(0 32 42)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

case "$VARIANT" in
  baseline)
    H5_PATH="../datasets/metaworld_topreward_labeled.h5"
    LOG_GROUP="offline_topreward_baseline_bonus64"
    LOG_DIR="logs/offline_topreward_baseline_bonus64_seed${SEED}"
    EXTRA_ARGS=(reward_model.use_progress_diff=false)
    ;;
  diff_gamma0999)
    H5_PATH="../datasets/metaworld_topreward_labeled_diff_gamma0999.h5"
    LOG_GROUP="offline_topreward_diff_gamma0999_bonus64"
    LOG_DIR="logs/offline_topreward_diff_gamma0999_bonus64_seed${SEED}"
    EXTRA_ARGS=(
      reward_model.use_progress_diff=true
      reward_model.diff_gamma="$DIFF_GAMMA"
      reward_model.diff_reward_scale="$DIFF_REWARD_SCALE"
    )
    ;;
  *)
    echo "ERROR: Unknown VARIANT=${VARIANT}"
    exit 1
    ;;
esac

if [ ! -d "$TOPREWARD_DIR" ]; then echo "ERROR: TOPREWARD_DIR not found: $TOPREWARD_DIR"; exit 1; fi
if [ ! -d "$TOPREWARD_CONDA" ]; then echo "ERROR: TopReward conda env not found: $TOPREWARD_CONDA"; exit 1; fi
if [ ! -f "${TOPREWARD_DIR}/metaworld_policy_training/${H5_PATH}" ]; then
  echo "ERROR: offline H5 not found: ${TOPREWARD_DIR}/metaworld_policy_training/${H5_PATH}"
  exit 1
fi

module purge
module load gcc/13.3.0
module load mesa/24.2.3
module load mesa-glu/9.0.2
module load glew/2.2.0
module load glx/1.4

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "$TOPREWARD_CONDA"

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
GCC_LIBDIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

export HF_HOME="${PROJECT_CACHE}/huggingface"
export TORCH_HOME="${PROJECT_CACHE}/torch"
export XDG_CACHE_HOME="${PROJECT_CACHE}"
export WANDB_CONFIG_DIR="$HOME/.config/wandb"
export WANDB_CACHE_DIR="$HOME/.cache/wandb"
export WANDB_DIR="$HOME/wandb"
export WANDB_MODE=online
export WANDB_RUN_NAME="${LOG_GROUP}_seed${SEED}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"

cd "${TOPREWARD_DIR}/metaworld_policy_training"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

echo "========== TOPReward offline ${VARIANT} seed ${SEED} =========="
which python
python --version
nvidia-smi

srun python -u train_policy.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=topreward \
  offline_training.offline_training_steps=15000 \
  online_training.total_time_steps=0 \
  general_training.seed="$SEED" \
  offline_training.offline_h5_path="$H5_PATH" \
  reward_model.success_bonus="$SUCCESS_BONUS" \
  logging.eval_freq=0 \
  logging.video_freq=0 \
  logging.wandb_group_name="$LOG_GROUP" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=. \
  "${EXTRA_ARGS[@]}"
