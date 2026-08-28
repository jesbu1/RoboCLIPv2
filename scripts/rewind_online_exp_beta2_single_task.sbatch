#!/usr/bin/env bash
# Run one exponential ReWiND online task.
# This file is intentionally non-array; submit one job per env/seed/variant.

#SBATCH --job-name=rw_on_exp_b2
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch1/haobaizh/rewind_nochunk_online_logs/slurm/%x_%j.out
#SBATCH --error=/scratch1/haobaizh/rewind_nochunk_online_logs/slurm/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/project2/biyik_1165/haobaizh/rewind_no-action-chunk}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch1/haobaizh/rewind_nochunk_online_logs}"
OFFLINE_BASE="${OFFLINE_BASE:-/scratch1/haobaizh/rewind_nochunk_offline_logs/rewind_exp_beta2}"
DATA_DIR="${DATA_DIR:-/scratch1/haobaizh/rewind_valuemodel/datasets}"
REWARD_CKPT="${REWARD_CKPT:-/scratch1/haobaizh/rewind_valuemodel/checkpoints/rewind_metaworld_exp_beta2_epoch_19.pth}"
CONDA_ENV="${CONDA_ENV:-${PROJECT_DIR}/conda_envs/rewind_nochunk}"
PROJECT_CACHE="${PROJECT_CACHE:-/home1/haobaizh/.cache}"
BETA="${BETA:-2.0}"
HORIZON="${HORIZON:-128}"
DIFF_GAMMA="${DIFF_GAMMA:-1.0}"
BASE_REWARD_VALUE="${BASE_REWARD_VALUE:--1.0}"
SUCCESS_BONUS="${SUCCESS_BONUS:-0}"
ONLINE_STEPS="${ONLINE_STEPS:-100000}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID}}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-0}"

ENV_ID="${ENV_ID:?ENV_ID must be set, e.g. window-close-v2}"
SEED="${SEED:?SEED must be set, e.g. 0, 32, or 42}"
VARIANT="${VARIANT:?VARIANT must be baseline or diff}"

BETA_TAG="${BETA//./}"
EXP_TAG="${EXP_TAG:-exp_beta${BETA_TAG}_H${HORIZON}}"

case "$VARIANT" in
  baseline)
    H5_PATH="${DATA_DIR}/metaworld_labeled_exp_beta2.h5"
    CKPT_PATH="${OFFLINE_BASE}/offline_rewind_${EXP_TAG}_baseline_10critics_seed${SEED}/last_offline"
    VARIANT_TAG="${EXP_TAG}_baseline"
    EXTRA_ARGS=(
      reward_model.use_progress_diff=false
      reward_model.reward_divisor=1.0
      reward_model.exponential_diff_reward_scale=1.0
      reward_model.diff_reward_scale=1.0
    )
    ;;
  diff)
    H5_PATH="${DATA_DIR}/metaworld_labeled_exp_beta2_diff.h5"
    CKPT_PATH="${OFFLINE_BASE}/offline_rewind_${EXP_TAG}_diff_10critics_seed${SEED}/last_offline"
    VARIANT_TAG="${EXP_TAG}_diff_gamma1_scaled"
    EXP_DIFF_REWARD_SCALE="$(awk -v h="$HORIZON" -v beta="$BETA" 'BEGIN {
      den = exp(beta) - 1.0
      sum = 0.0
      for (t = 1; t <= h; t++) {
        sum += (exp(beta * t / h) - 1.0) / den
      }
      printf "%.17g", sum
    }')"
    EXTRA_ARGS=(
      reward_model.use_progress_diff=true
      reward_model.diff_gamma="$DIFF_GAMMA"
      reward_model.exponential_diff_reward_scale="$EXP_DIFF_REWARD_SCALE"
      reward_model.diff_reward_scale=1.0
      reward_model.reward_divisor=1.0
    )
    ;;
  *)
    echo "ERROR: VARIANT must be baseline or diff, got ${VARIANT}"
    exit 1
    ;;
esac

GROUP_NAME="${ENV_ID}_nochunk_rewind_${VARIANT_TAG}_base_reward_10critics"
RUN_NAME="${ENV_ID}_rewind_${VARIANT_TAG}_base_reward_10critics_seed${SEED}_${RUN_TAG}"
LOG_DIR="${SCRATCH_BASE}/rewind_exp_beta2/${RUN_NAME}"

if [ ! -f "${PROJECT_DIR}/test_scripts/test_iql.py" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"
  exit 1
fi
if [ ! -d "$CONDA_ENV" ]; then
  echo "ERROR: conda env not found: ${CONDA_ENV}"
  exit 1
fi
if [ ! -f "$REWARD_CKPT" ]; then
  echo "ERROR: reward checkpoint not found: ${REWARD_CKPT}"
  exit 1
fi
if [ ! -f "$H5_PATH" ]; then
  echo "ERROR: offline H5 not found: ${H5_PATH}"
  exit 1
fi
if [ ! -f "${CKPT_PATH}.zip" ]; then
  echo "ERROR: offline checkpoint not found: ${CKPT_PATH}.zip"
  exit 1
fi
if [ -e "$LOG_DIR" ] && [ "$OVERWRITE_OUTPUT" != "1" ]; then
  echo "ERROR: refusing to overwrite existing online log dir: ${LOG_DIR}"
  echo "Set OVERWRITE_OUTPUT=1 if this rerun should replace it."
  exit 1
fi

mkdir -p "${SCRATCH_BASE}/rewind_exp_beta2" "${SCRATCH_BASE}/slurm"

module purge
module load gcc/13.3.0
module load mesa/24.2.3
module load mesa-glu/9.0.2
module load glew/2.2.0
module load glx/1.4

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
GCC_LIBDIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

export HF_HOME="${PROJECT_CACHE}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${PROJECT_CACHE}/torch"
export XDG_CACHE_HOME="${PROJECT_CACHE}"
export WANDB_CONFIG_DIR="$HOME/.config/wandb"
export WANDB_CACHE_DIR="${SCRATCH_BASE}/wandb_cache"
export WANDB_DIR="${SCRATCH_BASE}/wandb"
export WANDB_MODE=online
export WANDB_RUN_NAME="$RUN_NAME"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WANDB_CACHE_DIR" "$WANDB_DIR"

cd "$PROJECT_DIR"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

echo "========== Exponential ReWiND online ${VARIANT} =========="
echo "ENV_ID=${ENV_ID}"
echo "SEED=${SEED}"
echo "BETA=${BETA}"
echo "HORIZON=${HORIZON}"
echo "REWARD_CKPT=${REWARD_CKPT}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "H5_PATH=${H5_PATH}"
echo "HF_HOME=${HF_HOME}"
echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "BASE_REWARD_VALUE=${BASE_REWARD_VALUE}"
if [ "$VARIANT" = "diff" ]; then
  echo "exponential_diff_reward_scale=sum_t exp-progress=${EXP_DIFF_REWARD_SCALE}"
else
  echo "baseline_no_diff_scale=1.0"
fi
echo "GROUP_NAME=${GROUP_NAME}"
echo "RUN_NAME=${RUN_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "ONLINE_STEPS=${ONLINE_STEPS}"
which python
python --version
nvidia-smi || true

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.model_path="$REWARD_CKPT" \
  reward_model.success_bonus="$SUCCESS_BONUS" \
  reward_model.use_exponential_progress=true \
  reward_model.exponential_beta="$BETA" \
  reward_model.exponential_horizon="$HORIZON" \
  reward_model.use_base_reward=true \
  reward_model.base_reward_value="$BASE_REWARD_VALUE" \
  offline_training.ckpt_path="$CKPT_PATH" \
  offline_training.offline_h5_path="$H5_PATH" \
  offline_training.offline_training_steps=0 \
  online_training.total_time_steps="$ONLINE_STEPS" \
  online_training.mix_buffers_ratio=0.0 \
  general_training.seed="$SEED" \
  general_training.n_critics=10 \
  general_training.n_critics_to_sample=2 \
  environment.env_id="$ENV_ID" \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=. \
  "${EXTRA_ARGS[@]}"
