#!/usr/bin/env bash
#SBATCH --job-name=rw_lbl_g1
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Generate diff-gamma=1.0 labels (shared by all scale conditions).
# Output: datasets/metaworld_labeled_diff_gamma1.h5

set -e

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"

module purge
module load gcc/13.3.0

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "${PROJECT_DIR}/conda_envs/rewind_nochunk"

export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export MUJOCO_GL=egl

cd "$PROJECT_DIR"
mkdir -p datasets logs

REWIND_BASE=/project2/biyik_1165/haobaizh/rewind_valuemodel

srun python -u scripts/generate_labeled_dataset.py \
  --h5_video_path   "$REWIND_BASE/datasets/metaworld_generation.h5" \
  --h5_embedding_path "$REWIND_BASE/datasets/metaworld_embeddings_train.h5" \
  --reward_model_path "$REWIND_BASE/checkpoints/rewind_metaworld_epoch_19.pth" \
  --output_path datasets/metaworld_labeled_diff_gamma1.h5 \
  --use_progress_diff \
  --diff_gamma 1.0
