#!/usr/bin/env bash
# online baseline Robometer on sweep-into-v2, 3 seeds, base reward enabled

#SBATCH --job-name=robo_on_bl_br_si_3s
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2

set -e

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
INFO_FILE="${PROJECT_DIR}/logs/robometer_server_sweepinto_online_info.txt"

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

SEEDS=(42 32 0)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
ENV_ID="sweep-into-v2"

for i in $(seq 1 360); do
  if [ -s "$INFO_FILE" ]; then break; fi
  if [ $i -eq 360 ]; then echo "ERROR: Server info not found after 30min"; exit 1; fi
  sleep 5
done
SERVER_URL=$(tr -d '\r\n' < "$INFO_FILE")
if [ -z "$SERVER_URL" ]; then echo "ERROR: Server URL is empty"; exit 1; fi

for i in $(seq 1 360); do
  if curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then echo "Server ready!"; break; fi
  if [ $i -eq 360 ]; then echo "ERROR: Server not ready after 30min"; exit 1; fi
  sleep 5
done

CKPT_DIR="logs/offline_robometer_baseline_bonus0_seed${SEED}"
LOG_ROOT="/scratch1/haobaizh/rewind_nochunk_online_logs/robometer"
LOG_DIR="${LOG_ROOT}/${ENV_ID}_nochunk_robometer_baseline_base_reward_seed${SEED}"
GROUP_NAME="${ENV_ID}_nochunk_robometer_baseline_base_reward"
export WANDB_RUN_NAME="${ENV_ID}_nochunk_robometer_baseline_base_reward_seed${SEED}"

mkdir -p "$LOG_ROOT"
rm -rf "$LOG_DIR"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=robometer \
  reward_model.use_server=true \
  reward_model.server_url="$SERVER_URL" \
  reward_model.success_bonus=0 \
  reward_model.use_base_reward=true \
  reward_model.base_reward_value=-1.0 \
  general_training.seed=$SEED \
  environment.env_id=$ENV_ID \
  offline_training.offline_h5_path=datasets/metaworld_labeled_robometer.h5 \
  offline_training.ckpt_path=$CKPT_DIR/last_offline \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=.
