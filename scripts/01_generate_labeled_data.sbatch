#!/usr/bin/env bash
#SBATCH --job-name=gen_labeled
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/project2/biyik_1165/haobaizh/rewind_no-action-chunk/logs/%x_%j.out
#SBATCH --error=/project2/biyik_1165/haobaizh/rewind_no-action-chunk/logs/%x_%j.err

set -e

# Load modules and activate conda env
module purge
module load gcc/13.3.0

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate /project2/biyik_1165/haobaizh/rewind_no-action-chunk/conda_envs/rewind_nochunk

export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export MUJOCO_GL=egl

cd /project2/biyik_1165/haobaizh/rewind_no-action-chunk

mkdir -p datasets
mkdir -p logs

which python
python -c "import torch; print('torch=', torch.__version__)"

REWIND_BASE=/project2/biyik_1165/haobaizh/rewind_valuemodel

# Generate baseline reward P(s)
srun python -u scripts/generate_labeled_dataset.py \
  --h5_video_path $REWIND_BASE/datasets/metaworld_generation.h5 \
  --h5_embedding_path $REWIND_BASE/datasets/metaworld_embeddings_train.h5 \
  --reward_model_path $REWIND_BASE/checkpoints/rewind_metaworld_epoch_19.pth \
  --output_path datasets/metaworld_labeled.h5

# Generate diff reward P(s')-P(s)  (gamma=1.0)
srun python -u scripts/generate_labeled_dataset.py \
  --h5_video_path $REWIND_BASE/datasets/metaworld_generation.h5 \
  --h5_embedding_path $REWIND_BASE/datasets/metaworld_embeddings_train.h5 \
  --reward_model_path $REWIND_BASE/checkpoints/rewind_metaworld_epoch_19.pth \
  --output_path datasets/metaworld_labeled_diff.h5 \
  --use_progress_diff

# Generate diff reward 0.99*P(s')-P(s)  (gamma=0.99, proper PBRS)
srun python -u scripts/generate_labeled_dataset.py \
  --h5_video_path $REWIND_BASE/datasets/metaworld_generation.h5 \
  --h5_embedding_path $REWIND_BASE/datasets/metaworld_embeddings_train.h5 \
  --reward_model_path $REWIND_BASE/checkpoints/rewind_metaworld_epoch_19.pth \
  --output_path datasets/metaworld_labeled_diff_gamma099.h5 \
  --use_progress_diff \
  --diff_gamma 0.99
