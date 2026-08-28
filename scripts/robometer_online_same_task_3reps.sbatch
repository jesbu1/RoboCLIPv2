#!/usr/bin/env bash
# Robometer online reruns for one fixed task/seed/variant, repeated 3 times.

#SBATCH --job-name=robo_on_3rep
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
INFO_FILE="${INFO_FILE:-}"
EXPECTED_SERVER_JOB_ID="${EXPECTED_SERVER_JOB_ID:-}"
ENV_ID="${ENV_ID:-}"
VARIANT_TAG="${VARIANT_TAG:-}"
SEED="${SEED:-}"
LOG_ROOT="${LOG_ROOT:-/scratch1/haobaizh/rewind_nochunk_online_logs/robometer}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-1440}"
SERVER_WAIT_TIMEOUT_MIN=$((SERVER_WAIT_ATTEMPTS * 5 / 60))
REP_IDX=$((SLURM_ARRAY_TASK_ID + 1))

if [ ! -f "${PROJECT_DIR}/test_scripts/test_iql.py" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"
  exit 1
fi
if [ ! -d "${PROJECT_DIR}/conda_envs/rewind_nochunk" ]; then
  echo "ERROR: rewind_nochunk conda env not found under ${PROJECT_DIR}/conda_envs/rewind_nochunk"
  exit 1
fi
if [ -z "$INFO_FILE" ] || [ -z "$ENV_ID" ] || [ -z "$VARIANT_TAG" ] || [ -z "$SEED" ]; then
  echo "ERROR: INFO_FILE, ENV_ID, VARIANT_TAG, and SEED must be provided"
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
  return 0
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
if [ -z "${SERVER_URL:-}" ]; then
  echo "ERROR: Server URL is empty"
  exit 1
fi

for i in $(seq 1 "$SERVER_WAIT_ATTEMPTS"); do
  if curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo "Server ready!"
    break
  fi
  if [ "$i" -eq "$SERVER_WAIT_ATTEMPTS" ]; then
    echo "ERROR: Server not ready after ${SERVER_WAIT_TIMEOUT_MIN} minutes"
    exit 1
  fi
  sleep 5
done

case "$VARIANT_TAG" in
  baseline_base_reward)
    CKPT_DIR="logs/offline_robometer_baseline_bonus0_seed${SEED}"
    OFFLINE_H5_PATH="datasets/metaworld_labeled_robometer.h5"
    EXTRA_ARGS=(
      reward_model.use_base_reward=true
      reward_model.base_reward_value=-1.0
    )
    ;;
  diff_gamma099_scaled_base_reward)
    CKPT_DIR="logs/offline_robometer_diff_gamma099_bonus0_scaled_seed${SEED}"
    OFFLINE_H5_PATH="datasets/metaworld_labeled_robometer_diff_gamma099.h5"
    EXTRA_ARGS=(
      reward_model.use_progress_diff=true
      reward_model.diff_gamma=0.99
      reward_model.diff_reward_scale=100
      reward_model.use_base_reward=true
      reward_model.base_reward_value=-1.0
    )
    ;;
  diff_gamma0999_scaled_base_reward)
    CKPT_DIR="logs/offline_robometer_diff_gamma0999_bonus0_scaled_seed${SEED}"
    OFFLINE_H5_PATH="datasets/metaworld_labeled_robometer_diff_gamma0999.h5"
    EXTRA_ARGS=(
      reward_model.use_progress_diff=true
      reward_model.diff_gamma=0.999
      reward_model.diff_reward_scale=1000
      reward_model.use_base_reward=true
      reward_model.base_reward_value=-1.0
    )
    ;;
  *)
    echo "ERROR: Unsupported VARIANT_TAG=${VARIANT_TAG}"
    exit 1
    ;;
esac

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
LOG_DIR="${LOG_ROOT}/${RUN_NAME}_rerun_${SLURM_ARRAY_JOB_ID:-manual}_rep${REP_IDX}"
export WANDB_RUN_NAME="$RUN_NAME"

mkdir -p "$LOG_ROOT"
rm -rf "$LOG_DIR"

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=robometer \
  reward_model.use_server=true \
  reward_model.server_url="$SERVER_URL" \
  reward_model.success_bonus=0 \
  "${EXTRA_ARGS[@]}" \
  general_training.gamma=0.99 \
  general_training.seed=$SEED \
  environment.env_id=$ENV_ID \
  offline_training.offline_h5_path=$OFFLINE_H5_PATH \
  offline_training.ckpt_path=$CKPT_DIR/last_offline \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=.
