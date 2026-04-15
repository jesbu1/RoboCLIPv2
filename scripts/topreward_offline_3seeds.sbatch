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
PROJECT_DIR="${PROJECT_DIR:-/project2/biyik_1165/haobaizh/rewind_no-action-chunk}"
PROJECT_CACHE="${PROJECT_CACHE:-/home1/haobaizh/.cache}"
VARIANT="${VARIANT:?VARIANT must be baseline or diff_gamma0999}"
SUCCESS_BONUS="${SUCCESS_BONUS:-0.0}"
DIFF_GAMMA="${DIFF_GAMMA:-0.999}"
DIFF_REWARD_SCALE="${DIFF_REWARD_SCALE:-1000.0}"
NUM_PREFIX_SAMPLES="${NUM_PREFIX_SAMPLES:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"
TOPREWARD_LOCK_PATH="${TOPREWARD_LOCK_PATH:-${TOPREWARD_DIR}/logs/topreward_offline_${VARIANT}.lock}"

SEEDS=(0 32 42)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

case "$VARIANT" in
  baseline)
    H5_PATH="${TOPREWARD_DIR}/datasets/metaworld_topreward_labeled.h5"
    LOG_GROUP="offline_topreward_baseline_bonus0_10critics"
    LOG_DIR="logs/offline_topreward_baseline_bonus0_10critics_seed${SEED}"
    EXTRA_ARGS=(reward_model.use_progress_diff=false)
    ;;
  diff_gamma0999)
    H5_PATH="${TOPREWARD_DIR}/datasets/metaworld_topreward_labeled_diff_gamma0999.h5"
    LOG_GROUP="offline_topreward_diff_gamma0999_bonus0_scaled_10critics"
    LOG_DIR="logs/offline_topreward_diff_gamma0999_bonus0_scaled_10critics_seed${SEED}"
    EXTRA_ARGS=(
      reward_model.use_progress_diff=true
      reward_model.diff_gamma="$DIFF_GAMMA"
      reward_model.diff_reward_scale="$DIFF_REWARD_SCALE"
      reward_model.reward_divisor=0.001
    )
    ;;
  *)
    echo "ERROR: Unknown VARIANT=${VARIANT}"
    exit 1
    ;;
esac

if [ ! -d "$TOPREWARD_DIR" ]; then echo "ERROR: TOPREWARD_DIR not found: $TOPREWARD_DIR"; exit 1; fi
if [ ! -f "${PROJECT_DIR}/test_scripts/test_iql.py" ]; then echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"; exit 1; fi
if [ ! -d "${PROJECT_DIR}/conda_envs/rewind_nochunk" ]; then echo "ERROR: rewind_nochunk conda env not found: ${PROJECT_DIR}/conda_envs/rewind_nochunk"; exit 1; fi
if [ ! -f "${H5_PATH}" ]; then
  echo "ERROR: offline H5 not found: ${H5_PATH}"
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

export HF_HOME="${PROJECT_CACHE}/huggingface"
export TORCH_HOME="${PROJECT_CACHE}/torch"
export XDG_CACHE_HOME="${PROJECT_CACHE}"
export WANDB_CONFIG_DIR="$HOME/.config/wandb"
export WANDB_CACHE_DIR="$HOME/.cache/wandb"
export WANDB_DIR="$HOME/wandb"
export WANDB_MODE=online
export WANDB_RUN_NAME="${LOG_GROUP}_seed${SEED}"
export TOPREWARD_LOCK_PATH
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$(dirname "$TOPREWARD_LOCK_PATH")"

cd "$PROJECT_DIR"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

echo "========== TOPReward offline ${VARIANT} seed ${SEED} =========="
echo "HF_HOME=${HF_HOME}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "TOPREWARD_LOCK_PATH=${TOPREWARD_LOCK_PATH}"
echo "NUM_PREFIX_SAMPLES=${NUM_PREFIX_SAMPLES}"
echo "REQUEST_TIMEOUT=${REQUEST_TIMEOUT}"
which python
python --version
nvidia-smi

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=topreward \
  offline_training.offline_training_steps=100000 \
  online_training.total_time_steps=0 \
  general_training.seed="$SEED" \
  general_training.n_critics=10 \
  general_training.n_critics_to_sample=2 \
  offline_training.offline_h5_path="$H5_PATH" \
  reward_model.success_bonus="$SUCCESS_BONUS" \
  reward_model.max_frames="$NUM_PREFIX_SAMPLES" \
  reward_model.request_timeout="$REQUEST_TIMEOUT" \
  reward_model.request_retries="$REQUEST_RETRIES" \
  reward_model.lock_path="$TOPREWARD_LOCK_PATH" \
  logging.eval_freq=0 \
  logging.video_freq=0 \
  logging.wandb_group_name="$LOG_GROUP" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=. \
  "${EXTRA_ARGS[@]}"
