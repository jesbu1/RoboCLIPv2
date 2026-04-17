#!/usr/bin/env bash
# Generate pure ReWiND reverse-diff labels:
#   r_t = P(s_{t+1}) - gamma * P(s_t)

#SBATCH --job-name=rw_lbl_rev
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
REWIND_BASE="${REWIND_BASE:-/project2/biyik_1165/haobaizh/rewind_valuemodel}"
VARIANT="${VARIANT:?VARIANT must be gamma099_scale100, gamma0999_scale1000, or gamma09999_scale10000}"

case "$VARIANT" in
  gamma099_scale100)
    DIFF_GAMMA="0.99"
    OUTPUT_PATH="datasets/metaworld_labeled_reverse_diff_gamma099.h5"
    ;;
  gamma0999_scale1000)
    DIFF_GAMMA="0.999"
    OUTPUT_PATH="datasets/metaworld_labeled_reverse_diff_gamma0999.h5"
    ;;
  gamma09999_scale10000)
    DIFF_GAMMA="0.9999"
    OUTPUT_PATH="datasets/metaworld_labeled_reverse_diff_gamma09999.h5"
    ;;
  *)
    echo "ERROR: Unknown VARIANT=${VARIANT}"
    exit 1
    ;;
esac

module purge
module load gcc/13.3.0

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "${PROJECT_DIR}/conda_envs/rewind_nochunk"

export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export MUJOCO_GL=egl

cd "$PROJECT_DIR"
mkdir -p datasets logs

echo "========== ReWiND reverse-diff label =========="
echo "VARIANT=${VARIANT}"
echo "DIFF_GAMMA=${DIFF_GAMMA}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "REWIND_BASE=${REWIND_BASE}"
which python
python --version

srun python -u scripts/generate_labeled_dataset.py \
  --h5_video_path "$REWIND_BASE/datasets/metaworld_generation.h5" \
  --h5_embedding_path "$REWIND_BASE/datasets/metaworld_embeddings_train.h5" \
  --reward_model_path "$REWIND_BASE/checkpoints/rewind_metaworld_epoch_19.pth" \
  --output_path "$OUTPUT_PATH" \
  --use_progress_diff \
  --use_reverse_progress_diff \
  --diff_gamma "$DIFF_GAMMA"
