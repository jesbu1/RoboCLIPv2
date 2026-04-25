#!/usr/bin/env bash
# TOPReward offline, diff gamma=1.0, scale=(HORIZON+1)/2, success_bonus=0, seed0 only.

#SBATCH --job-name=top_off_d1_h
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

TOPREWARD_DIR="${TOPREWARD_DIR:-/scratch1/haobaizh/rewind_topreward}"
PROJECT_DIR="${PROJECT_DIR:-/project2/biyik_1165/haobaizh/rewind_no-action-chunk}"
PROJECT_CACHE="${PROJECT_CACHE:-/home1/haobaizh/.cache}"
INFO_FILE="${INFO_FILE:?INFO_FILE must point to a ready TOPReward server info file}"
EXPECTED_SERVER_JOB_ID="${EXPECTED_SERVER_JOB_ID:-}"
HORIZON="${HORIZON:-128}"
MAX_FRAMES="${MAX_FRAMES:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"
TOPREWARD_LOCK_PATH="${TOPREWARD_LOCK_PATH:-${TOPREWARD_DIR}/logs/topreward_offline_diff_gamma1.lock}"
SEED=0
H5_PATH="${TOPREWARD_DIR}/datasets/metaworld_topreward_labeled_diff_gamma1.h5"
DIFF_REWARD_SCALE=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", (h + 1) / 2 }')
REWARD_DIVISOR=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", 2 / (h + 1) }')
TAG="diff_gamma1_scale_hplus1_over2_H${HORIZON}"
LOG_GROUP="offline_topreward_${TAG}_10critics"
LOG_DIR="logs/offline_topreward_${TAG}_10critics_seed${SEED}"

if [ ! -d "$TOPREWARD_DIR" ]; then
  echo "ERROR: TOPREWARD_DIR not found: $TOPREWARD_DIR"
  exit 1
fi
if [ ! -f "${PROJECT_DIR}/test_scripts/test_iql.py" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"
  exit 1
fi
if [ ! -d "${PROJECT_DIR}/conda_envs/rewind_nochunk" ]; then
  echo "ERROR: rewind_nochunk conda env not found: ${PROJECT_DIR}/conda_envs/rewind_nochunk"
  exit 1
fi
if [ ! -f "$H5_PATH" ]; then
  echo "ERROR: offline H5 not found: $H5_PATH"
  exit 1
fi

read_server_url() {
  local info_line server_job_id server_url
  info_line=$(tr -d '\r\n' < "$INFO_FILE")
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

for i in $(seq 1 720); do
  if [ -s "$INFO_FILE" ] && TOPREWARD_API_URL="$(read_server_url)"; then
    break
  fi
  if [ "$i" -eq 720 ]; then
    echo "ERROR: Server info not ready after 60 minutes"
    exit 1
  fi
  sleep 5
done

for i in $(seq 1 720); do
  if curl -fsS "${TOPREWARD_API_URL}/health" >/dev/null 2>&1; then
    break
  fi
  if [ "$i" -eq 720 ]; then
    echo "ERROR: Server health not ready after 60 minutes"
    exit 1
  fi
  sleep 5
done

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

export HF_HOME="${PROJECT_CACHE}/huggingface"
export TORCH_HOME="${PROJECT_CACHE}/torch"
export XDG_CACHE_HOME="${PROJECT_CACHE}"
export WANDB_CONFIG_DIR="$HOME/.config/wandb"
export WANDB_CACHE_DIR="$HOME/.cache/wandb"
export WANDB_DIR="$HOME/wandb"
export WANDB_MODE=online
export WANDB_RUN_NAME="${LOG_GROUP}_seed${SEED}"
export TOPREWARD_LOCK_PATH
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$(dirname "$TOPREWARD_LOCK_PATH")"

cd "$PROJECT_DIR"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

echo "========== TOPReward offline ${TAG} seed ${SEED} =========="
echo "H5_PATH=${H5_PATH}"
echo "TOPREWARD_LOCK_PATH=${TOPREWARD_LOCK_PATH}"
echo "HORIZON=${HORIZON}"
echo "diff_reward_scale=(HORIZON+1)/2=${DIFF_REWARD_SCALE}"
echo "reward_divisor=2/(HORIZON+1)=${REWARD_DIVISOR}"
echo "MAX_FRAMES=${MAX_FRAMES}"
echo "REQUEST_TIMEOUT=${REQUEST_TIMEOUT}"
echo "TOPREWARD_API_URL=${TOPREWARD_API_URL}"
which python
python --version
nvidia-smi

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=topreward \
  offline_training.offline_training_steps=100000 \
  online_training.total_time_steps=0 \
  general_training.seed="$SEED" \
  general_training.n_critics=10 \
  general_training.n_critics_to_sample=2 \
  offline_training.offline_h5_path="$H5_PATH" \
  reward_model.success_bonus=0 \
  reward_model.server_url="$TOPREWARD_API_URL" \
  reward_model.max_frames="$MAX_FRAMES" \
  reward_model.request_timeout="$REQUEST_TIMEOUT" \
  reward_model.request_retries="$REQUEST_RETRIES" \
  reward_model.lock_path="$TOPREWARD_LOCK_PATH" \
  reward_model.use_progress_diff=true \
  reward_model.diff_gamma=1.0 \
  reward_model.reward_divisor="$REWARD_DIVISOR" \
  logging.eval_freq=0 \
  logging.video_freq=0 \
  logging.wandb_group_name="$LOG_GROUP" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=.
