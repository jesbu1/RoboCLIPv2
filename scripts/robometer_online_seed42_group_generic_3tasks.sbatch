#!/usr/bin/env bash
# Generic Seed42 Robometer online 3-task group runner.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
GROUP_ID="${GROUP_ID:-}"
INFO_FILE="${INFO_FILE:-${PROJECT_DIR}/logs/robometer_server_seed42_group${GROUP_ID}_info.txt}"
EXPECTED_SERVER_JOB_ID="${EXPECTED_SERVER_JOB_ID:-}"
SEED=42
LOG_ROOT="${LOG_ROOT:-/scratch1/haobaizh/rewind_nochunk_online_logs/robometer}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-1440}"
SERVER_WAIT_TIMEOUT_MIN=$((SERVER_WAIT_ATTEMPTS * 5 / 60))

if [ ! -f "${PROJECT_DIR}/test_scripts/test_iql.py" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"
  exit 1
fi
if [ ! -d "${PROJECT_DIR}/conda_envs/rewind_nochunk" ]; then
  echo "ERROR: rewind_nochunk conda env not found under ${PROJECT_DIR}/conda_envs/rewind_nochunk"
  exit 1
fi
if [ -z "$GROUP_ID" ]; then
  echo "ERROR: GROUP_ID must be provided"
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

case "${GROUP_ID}:${SLURM_ARRAY_TASK_ID}" in
  7:0)
    ENV_ID="faucet-close-v2"
    VARIANT_TAG="baseline_base_reward"
    CKPT_DIR="logs/offline_robometer_baseline_bonus0_seed${SEED}"
    OFFLINE_H5_PATH="datasets/metaworld_labeled_robometer.h5"
    EXTRA_ARGS=(
      reward_model.use_base_reward=true
      reward_model.base_reward_value=-1.0
    )
    ;;
  7:1)
    ENV_ID="door-lock-v2"
    VARIANT_TAG="diff_gamma099_scaled_base_reward"
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
  7:2)
    ENV_ID="button-press-v2"
    VARIANT_TAG="diff_gamma0999_scaled_base_reward"
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
  8:0)
    ENV_ID="faucet-close-v2"
    VARIANT_TAG="diff_gamma099_scaled_base_reward"
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
  8:1)
    ENV_ID="coffee-button-v2"
    VARIANT_TAG="baseline_base_reward"
    CKPT_DIR="logs/offline_robometer_baseline_bonus0_seed${SEED}"
    OFFLINE_H5_PATH="datasets/metaworld_labeled_robometer.h5"
    EXTRA_ARGS=(
      reward_model.use_base_reward=true
      reward_model.base_reward_value=-1.0
    )
    ;;
  8:2)
    ENV_ID="reach-wall-v2"
    VARIANT_TAG="diff_gamma0999_scaled_base_reward"
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
  9:0)
    ENV_ID="faucet-close-v2"
    VARIANT_TAG="diff_gamma0999_scaled_base_reward"
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
  9:1)
    ENV_ID="handle-press-v2"
    VARIANT_TAG="baseline_base_reward"
    CKPT_DIR="logs/offline_robometer_baseline_bonus0_seed${SEED}"
    OFFLINE_H5_PATH="datasets/metaworld_labeled_robometer.h5"
    EXTRA_ARGS=(
      reward_model.use_base_reward=true
      reward_model.base_reward_value=-1.0
    )
    ;;
  9:2)
    ENV_ID="button-press-v2"
    VARIANT_TAG="diff_gamma099_scaled_base_reward"
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
  10:0)
    ENV_ID="window-close-v2"
    VARIANT_TAG="baseline_base_reward"
    CKPT_DIR="logs/offline_robometer_baseline_bonus0_seed${SEED}"
    OFFLINE_H5_PATH="datasets/metaworld_labeled_robometer.h5"
    EXTRA_ARGS=(
      reward_model.use_base_reward=true
      reward_model.base_reward_value=-1.0
    )
    ;;
  10:1)
    ENV_ID="door-lock-v2"
    VARIANT_TAG="diff_gamma0999_scaled_base_reward"
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
  10:2)
    ENV_ID="reach-wall-v2"
    VARIANT_TAG="diff_gamma099_scaled_base_reward"
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
  11:0)
    ENV_ID="window-close-v2"
    VARIANT_TAG="diff_gamma099_scaled_base_reward"
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
  11:1)
    ENV_ID="handle-press-v2"
    VARIANT_TAG="diff_gamma0999_scaled_base_reward"
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
  11:2)
    ENV_ID="button-press-v2"
    VARIANT_TAG="baseline_base_reward"
    CKPT_DIR="logs/offline_robometer_baseline_bonus0_seed${SEED}"
    OFFLINE_H5_PATH="datasets/metaworld_labeled_robometer.h5"
    EXTRA_ARGS=(
      reward_model.use_base_reward=true
      reward_model.base_reward_value=-1.0
    )
    ;;
  12:0)
    ENV_ID="window-close-v2"
    VARIANT_TAG="diff_gamma0999_scaled_base_reward"
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
  12:1)
    ENV_ID="coffee-button-v2"
    VARIANT_TAG="diff_gamma099_scaled_base_reward"
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
  12:2)
    ENV_ID="reach-wall-v2"
    VARIANT_TAG="baseline_base_reward"
    CKPT_DIR="logs/offline_robometer_baseline_bonus0_seed${SEED}"
    OFFLINE_H5_PATH="datasets/metaworld_labeled_robometer.h5"
    EXTRA_ARGS=(
      reward_model.use_base_reward=true
      reward_model.base_reward_value=-1.0
    )
    ;;
  13:0)
    ENV_ID="sweep-into-v2"
    VARIANT_TAG="baseline_base_reward"
    CKPT_DIR="logs/offline_robometer_baseline_bonus0_seed${SEED}"
    OFFLINE_H5_PATH="datasets/metaworld_labeled_robometer.h5"
    EXTRA_ARGS=(
      reward_model.use_base_reward=true
      reward_model.base_reward_value=-1.0
    )
    ;;
  13:1)
    ENV_ID="sweep-into-v2"
    VARIANT_TAG="diff_gamma099_scaled_base_reward"
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
  13:2)
    ENV_ID="door-lock-v2"
    VARIANT_TAG="baseline_base_reward"
    CKPT_DIR="logs/offline_robometer_baseline_bonus0_seed${SEED}"
    OFFLINE_H5_PATH="datasets/metaworld_labeled_robometer.h5"
    EXTRA_ARGS=(
      reward_model.use_base_reward=true
      reward_model.base_reward_value=-1.0
    )
    ;;
  14:0)
    ENV_ID="sweep-into-v2"
    VARIANT_TAG="diff_gamma0999_scaled_base_reward"
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
  14:1)
    ENV_ID="coffee-button-v2"
    VARIANT_TAG="diff_gamma0999_scaled_base_reward"
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
  14:2)
    ENV_ID="handle-press-v2"
    VARIANT_TAG="diff_gamma099_scaled_base_reward"
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
  *)
    echo "ERROR: Unsupported GROUP_ID=${GROUP_ID}, task=${SLURM_ARRAY_TASK_ID}"
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
LOG_DIR="${LOG_ROOT}/${ENV_ID}_nochunk_robometer_${VARIANT_TAG}_10critics_seed${SEED}"
export WANDB_RUN_NAME="${ENV_ID}_nochunk_robometer_${VARIANT_TAG}_10critics_seed${SEED}"

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
