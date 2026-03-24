#!/usr/bin/env bash
#SBATCH --job-name=robo_off_bl
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/project2/biyik_1165/haobaizh/rewind_no-action-chunk/logs/%x_%j.out
#SBATCH --error=/project2/biyik_1165/haobaizh/rewind_no-action-chunk/logs/%x_%j.err

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

LOG_DIR="logs/offline_robometer_baseline_seed0"
mkdir -p "$LOG_DIR"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=robometer \
  reward_model.use_server=false \
  general_training.seed=0 \
  environment.env_id=coffee-button-v2 \
  offline_training.offline_h5_path=datasets/metaworld_labeled_robometer.h5 \
  online_training.total_time_steps=0 \
  logging.wandb_group_name="offline_robometer_baseline" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=.
