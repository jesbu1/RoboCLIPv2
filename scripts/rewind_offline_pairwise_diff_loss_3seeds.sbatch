#!/usr/bin/env bash
#SBATCH --job-name=rw_off_pair_d
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch1/haobaizh/rewind_nochunk_offline_logs/%x_%A_%a.out
#SBATCH --error=/scratch1/haobaizh/rewind_nochunk_offline_logs/%x_%A_%a.err
#SBATCH --array=0-2

# Offline IQL for the reward model trained with pairwise progress-diff loss.
# This follows the original linear-diff ReWiND downstream:
#   reward = P(s') - P(s)
#   scale  = (H + 1) / 2
# The offline buffer divides by reward_divisor, so we set:
#   reward_divisor = 1 / scale = 2 / (H + 1)

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/project2/biyik_1165/haobaizh/rewind_no-action-chunk}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch1/haobaizh/rewind_nochunk_offline_logs}"
DATA_DIR="${DATA_DIR:-/scratch1/haobaizh/rewind_valuemodel/datasets}"
REWARD_CKPT="${REWARD_CKPT:-/scratch1/haobaizh/rewind_valuemodel/checkpoints/rewind_metaworld_pairwise_diff_loss_epoch_19.pth}"
H5_PATH="${H5_PATH:-${DATA_DIR}/metaworld_labeled_pairwise_diff_loss_diff.h5}"
CONDA_ENV="${CONDA_ENV:-${PROJECT_DIR}/conda_envs/rewind_nochunk}"
HORIZON="${HORIZON:-128}"
SUCCESS_BONUS="${SUCCESS_BONUS:-0}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-0}"
PROJECT_CACHE="${PROJECT_CACHE:-/home1/haobaizh/.cache}"

SEEDS=(0 32 42)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

DIFF_REWARD_SCALE="$(awk -v h="$HORIZON" 'BEGIN { printf "%.17g", (h + 1.0) / 2.0 }')"
REWARD_DIVISOR="$(awk -v s="$DIFF_REWARD_SCALE" 'BEGIN { printf "%.17g", 1.0 / s }')"
TAG="pairwise_diff_loss_diff_gamma1_scaled_H${HORIZON}"
LOG_GROUP="offline_rewind_${TAG}_10critics"
LOG_DIR="${SCRATCH_BASE}/rewind_pairwise_diff_loss/${LOG_GROUP}_seed${SEED}"
STAGE_DIR="${TMPDIR:-/tmp/SLURM_${SLURM_JOB_ID}}/rewind_offline_pairwise_diff_seed${SEED}"
LOCAL_H5="${STAGE_DIR}/$(basename "$H5_PATH")"

if [ ! -f "${PROJECT_DIR}/test_scripts/test_iql.py" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}" >&2
  exit 1
fi
if [ ! -d "$CONDA_ENV" ]; then
  echo "ERROR: conda env not found: ${CONDA_ENV}" >&2
  exit 1
fi
if [ ! -f "$REWARD_CKPT" ]; then
  echo "ERROR: reward checkpoint not found: ${REWARD_CKPT}" >&2
  exit 1
fi
if [ ! -f "$H5_PATH" ]; then
  echo "ERROR: offline H5 not found: ${H5_PATH}" >&2
  exit 1
fi
if [ -e "$LOG_DIR" ] && [ "$OVERWRITE_OUTPUT" != "1" ]; then
  echo "ERROR: LOG_DIR already exists: ${LOG_DIR}" >&2
  echo "Set OVERWRITE_OUTPUT=1 if this rerun should replace it." >&2
  exit 1
fi

mkdir -p "$SCRATCH_BASE" "$(dirname "$LOG_DIR")" "$STAGE_DIR" "$PROJECT_CACHE"

echo "========== Stage pairwise-diff offline H5 =========="
echo "source=${H5_PATH}"
echo "target=${LOCAL_H5}"
date
cp "$H5_PATH" "$LOCAL_H5"
date

module purge
module load gcc/13.3.0
module load mesa/24.2.3
module load mesa-glu/9.0.2
module load glew/2.2.0
module load glx/1.4

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

python - "$H5_PATH" "$LOCAL_H5" <<'PY'
import os
import sys

import h5py
import numpy as np

src, dst = sys.argv[1:3]
src_size = os.path.getsize(src)
dst_size = os.path.getsize(dst)
print(f"VERIFY {os.path.basename(dst)}: src_size={src_size} dst_size={dst_size}", flush=True)
if src_size != dst_size:
    raise SystemExit("ERROR: staged H5 size mismatch")

required = ["state", "action", "rewards", "done", "policy_lang_embedding", "img_embedding", "env_id"]
with h5py.File(dst, "r") as h5:
    missing = [key for key in required if key not in h5]
    if missing:
        raise SystemExit(f"ERROR: missing H5 datasets: {missing}")
    n = int(h5["rewards"].shape[0])
    for key in required:
        if int(h5[key].shape[0]) != n:
            raise SystemExit(f"ERROR: dataset {key} has {h5[key].shape[0]} rows, expected {n}")
    rewards = h5["rewards"][()]
    print(
        "H5_VERIFY_OK "
        f"rows={n} reward_min={float(np.min(rewards)):.8g} "
        f"reward_max={float(np.max(rewards)):.8g} "
        f"reward_mean={float(np.mean(rewards)):.8g}",
        flush=True,
    )
PY

if [ "$OVERWRITE_OUTPUT" = "1" ]; then
  rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR" "${SCRATCH_BASE}/wandb" "${SCRATCH_BASE}/wandb_cache"

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
export WANDB_RUN_NAME="${LOG_GROUP}_seed${SEED}"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"

echo "========== Train pairwise-diff-loss ReWiND offline seed ${SEED} =========="
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "TRAIN_H5=${LOCAL_H5}"
echo "REWARD_CKPT=${REWARD_CKPT}"
echo "HORIZON=${HORIZON}"
echo "diff_reward_scale=(HORIZON+1)/2=${DIFF_REWARD_SCALE}"
echo "offline_reward_divisor=1/scale=${REWARD_DIVISOR}"
echo "LOG_DIR=${LOG_DIR}"
which python
python --version
nvidia-smi || true

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.model_path="$REWARD_CKPT" \
  reward_model.success_bonus="$SUCCESS_BONUS" \
  reward_model.use_progress_diff=true \
  reward_model.diff_gamma=1.0 \
  reward_model.diff_reward_scale=1.0 \
  reward_model.reward_divisor="$REWARD_DIVISOR" \
  offline_training.offline_training_steps=100000 \
  online_training.total_time_steps=0 \
  general_training.seed="$SEED" \
  general_training.n_critics=10 \
  general_training.n_critics_to_sample=2 \
  environment.env_id=coffee-button-v2 \
  offline_training.offline_h5_path="$LOCAL_H5" \
  logging.wandb_group_name="$LOG_GROUP" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=.
