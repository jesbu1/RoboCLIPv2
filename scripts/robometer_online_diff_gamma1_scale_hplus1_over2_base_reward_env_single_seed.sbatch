#!/usr/bin/env bash
# Robometer online for one env and one seed, diff gamma=1.0,
# scale=(HORIZON+1)/2, base reward=-1. Non-array version.

#SBATCH --job-name=robo_on_d1_h_s
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
INFO_FILE="${INFO_FILE:?INFO_FILE must point to this Robometer server info file}"
EXPECTED_SERVER_JOB_ID="${EXPECTED_SERVER_JOB_ID:-}"
GROUP_ID="${GROUP_ID:-unknown}"
ENV_ID="${ENV_ID:?ENV_ID must be set}"
SEED="${SEED:?SEED must be set to one of 42,32,0}"
MAX_FRAMES="${MAX_FRAMES:-4}"
LOG_ROOT="${LOG_ROOT:-/scratch1/haobaizh/rewind_nochunk_online_logs/robometer}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-1440}"
SERVER_WAIT_TIMEOUT_MIN=$((SERVER_WAIT_ATTEMPTS * 5 / 60))
HORIZON="${HORIZON:-128}"

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

read_server_url() {
  local info_line server_job_id server_url
  info_line=$(tr -d '\r\n' < "$INFO_FILE")
  if [ -z "$info_line" ]; then
    return 1
  fi
  if [[ "$info_line" == http* ]]; then
    printf '%s\n' "$info_line"
    return 0
  fi

  server_job_id="${info_line%% *}"
  server_url="${info_line#* }"
  if [ -n "$EXPECTED_SERVER_JOB_ID" ] && [ "$server_job_id" != "$EXPECTED_SERVER_JOB_ID" ]; then
    return 1
  fi
  if [ -z "$server_url" ] || [[ "$server_url" != http* ]]; then
    return 1
  fi

  printf '%s\n' "$server_url"
}

for i in $(seq 1 "$SERVER_WAIT_ATTEMPTS"); do
  if [ -s "$INFO_FILE" ] && SERVER_URL="$(read_server_url)"; then
    break
  fi
  if [ "$i" -eq "$SERVER_WAIT_ATTEMPTS" ]; then
    echo "ERROR: Server info not found after ${SERVER_WAIT_TIMEOUT_MIN} minutes"
    exit 1
  fi
  sleep 5
done
if [ -z "${SERVER_URL:-}" ]; then echo "ERROR: Server URL is empty"; exit 1; fi

for i in $(seq 1 "$SERVER_WAIT_ATTEMPTS"); do
  if curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then echo "Server ready!"; break; fi
  if [ "$i" -eq "$SERVER_WAIT_ATTEMPTS" ]; then
    echo "ERROR: Server not ready after ${SERVER_WAIT_TIMEOUT_MIN} minutes"
    exit 1
  fi
  sleep 5
done

DIFF_REWARD_SCALE=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", (h + 1) / 2 }')
VARIANT_TAG="diff_gamma1_scale_hplus1_over2_H${HORIZON}_base_reward"
CKPT_DIR="logs/offline_robometer_diff_gamma1_scale_hplus1_over2_H${HORIZON}_10critics_seed${SEED}"
OFFLINE_H5_PATH="datasets/metaworld_labeled_robometer_diff_gamma1.h5"

if [ ! -f "${CKPT_DIR}/last_offline.zip" ]; then
  echo "ERROR: Offline checkpoint not found at ${CKPT_DIR}/last_offline.zip"
  exit 1
fi
if [ ! -f "$OFFLINE_H5_PATH" ]; then
  echo "ERROR: Offline dataset not found at ${OFFLINE_H5_PATH}"
  exit 1
fi

GROUP_NAME="${ENV_ID}_nochunk_robometer_${VARIANT_TAG}_10critics"
RUN_NAME="${ENV_ID}_nochunk_robometer_${VARIANT_TAG}_10critics_seed${SEED}"
LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
export WANDB_RUN_NAME="$RUN_NAME"

mkdir -p "$LOG_ROOT"
rm -rf "$LOG_DIR"

echo "========== Robometer online ${VARIANT_TAG} ${ENV_ID} seed ${SEED} =========="
echo "group_id=${GROUP_ID}"
echo "HORIZON=${HORIZON}"
echo "diff_reward_scale=(HORIZON+1)/2=${DIFF_REWARD_SCALE}"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=robometer \
  reward_model.use_server=true \
  reward_model.server_url="$SERVER_URL" \
  reward_model.success_bonus=0 \
  reward_model.max_frames="$MAX_FRAMES" \
  reward_model.use_progress_diff=true \
  reward_model.diff_gamma=1.0 \
  reward_model.diff_reward_scale="$DIFF_REWARD_SCALE" \
  reward_model.use_base_reward=true \
  reward_model.base_reward_value=-1.0 \
  offline_training.offline_training_steps=0 \
  online_training.total_time_steps=100000 \
  online_training.mix_buffers_ratio=0.0 \
  general_training.seed=$SEED \
  general_training.n_critics=10 \
  general_training.n_critics_to_sample=2 \
  environment.env_id=$ENV_ID \
  offline_training.offline_h5_path=$OFFLINE_H5_PATH \
  offline_training.ckpt_path=$CKPT_DIR/last_offline \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=.
