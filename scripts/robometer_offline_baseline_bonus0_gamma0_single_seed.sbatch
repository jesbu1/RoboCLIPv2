#!/usr/bin/env bash
# Offline Robometer baseline pretraining with RL gamma=0 for one seed.

#SBATCH --job-name=robo_off_bl_g0_1s
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
SERVER_URL="${SERVER_URL:-}"
SEED="${SEED:-42}"
LOG_GROUP_NAME="${LOG_GROUP_NAME:-offline_robometer_baseline_bonus0_gamma0}"

if [ ! -f "${PROJECT_DIR}/test_scripts/test_iql.py" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"
  exit 1
fi
if [ ! -d "${PROJECT_DIR}/conda_envs/rewind_nochunk" ]; then
  echo "ERROR: rewind_nochunk conda env not found under ${PROJECT_DIR}/conda_envs/rewind_nochunk"
  exit 1
fi
if [ -z "$SERVER_URL" ]; then
  echo "ERROR: SERVER_URL is empty"
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

for i in $(seq 1 360); do
  if curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo "Server ready!"
    break
  fi
  if [ $i -eq 360 ]; then
    echo "ERROR: Server not ready after 30 minutes"
    exit 1
  fi
  sleep 5
done

LOG_DIR="logs/offline_robometer_baseline_bonus0_gamma0_seed${SEED}"
mkdir -p "$LOG_DIR"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=robometer \
  reward_model.use_server=true \
  reward_model.server_url="$SERVER_URL" \
  reward_model.success_bonus=0 \
  general_training.gamma=0 \
  general_training.seed=$SEED \
  offline_training.offline_h5_path=datasets/metaworld_labeled_robometer.h5 \
  online_training.total_time_steps=0 \
  logging.wandb_group_name="$LOG_GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=.
