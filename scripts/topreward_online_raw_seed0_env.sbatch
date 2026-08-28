#!/usr/bin/env bash
# Run one TOPReward raw baseline/diff-gamma1 online seed-configurable task.

#SBATCH --job-name=top_on_raw
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

TOPREWARD_DIR="${TOPREWARD_DIR:-/scratch1/haobaizh/rewind_topreward}"
PROJECT_DIR="${PROJECT_DIR:-/project2/biyik_1165/haobaizh/rewind_no-action-chunk}"
PROJECT_CACHE="${PROJECT_CACHE:-/home1/haobaizh/.cache}"
INFO_FILE="${INFO_FILE:?INFO_FILE must point to this task TOPReward server info file}"
EXPECTED_SERVER_JOB_ID="${EXPECTED_SERVER_JOB_ID:-}"
GROUP_ID="${GROUP_ID:-unknown}"
ENV_ID="${ENV_ID:?ENV_ID must be set}"
VARIANT="${VARIANT:?VARIANT must be baseline or diff_gamma1}"
H5_PATH="${H5_PATH:?H5_PATH must be set}"
CKPT_PATH="${CKPT_PATH:?CKPT_PATH must be set without .zip suffix}"
RUN_TAG="${RUN_TAG:?RUN_TAG must be set}"
HORIZON="${HORIZON:-128}"
MAX_FRAMES="${MAX_FRAMES:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"
TOPREWARD_REQUEST_FORMAT="${TOPREWARD_REQUEST_FORMAT:-raw}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-1440}"
LOCK_PATH="${LOCK_PATH:-${TOPREWARD_DIR}/logs/topreward_raw_online_group${GROUP_ID}.lock}"
LOG_ROOT="${LOG_ROOT:-/scratch1/haobaizh/rewind_nochunk_online_logs/topreward_raw}"
DIFF_GAMMA="${DIFF_GAMMA:-1.0}"
SEED="${SEED:-0}"

DIFF_REWARD_SCALE=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", (h + 1) / 2 }')

case "$VARIANT" in
  baseline)
    VARIANT_TAG="raw_baseline_H${HORIZON}"
    EXTRA_ARGS=(
      reward_model.use_progress_diff=false
      reward_model.use_base_reward=true
      reward_model.base_reward_value=-1.0
    )
    ;;
  diff_gamma1)
    VARIANT_TAG="raw_diff_gamma1_scale_hplus1_over2_H${HORIZON}"
    EXTRA_ARGS=(
      reward_model.use_progress_diff=true
      reward_model.diff_gamma="$DIFF_GAMMA"
      reward_model.diff_reward_scale="$DIFF_REWARD_SCALE"
      reward_model.use_base_reward=true
      reward_model.base_reward_value=-1.0
    )
    ;;
  *)
    echo "ERROR: Unknown VARIANT=${VARIANT}"
    exit 1
    ;;
esac

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

for i in $(seq 1 "$SERVER_WAIT_ATTEMPTS"); do
  if [ -s "$INFO_FILE" ] && TOPREWARD_API_URL="$(read_server_url)"; then
    break
  fi
  if [ "$i" -eq "$SERVER_WAIT_ATTEMPTS" ]; then echo "ERROR: Server info not ready"; exit 1; fi
  sleep 5
done

for i in $(seq 1 "$SERVER_WAIT_ATTEMPTS"); do
  if curl -fsS "${TOPREWARD_API_URL}/health" >/dev/null 2>&1; then
    break
  fi
  if [ "$i" -eq "$SERVER_WAIT_ATTEMPTS" ]; then echo "ERROR: Server health not ready"; exit 1; fi
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
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$(dirname "$LOCK_PATH")" "$LOG_ROOT"

cd "$PROJECT_DIR"
if [ ! -f "$H5_PATH" ]; then echo "ERROR: H5 not found: ${H5_PATH}"; exit 1; fi
if [ ! -f "${CKPT_PATH}.zip" ]; then echo "ERROR: checkpoint not found: ${PWD}/${CKPT_PATH}.zip"; exit 1; fi

GROUP_NAME="topreward_wsrl_iql_${ENV_ID}_${VARIANT_TAG}_base_reward_10critics"
RUN_NAME="${ENV_ID}_topreward_${VARIANT_TAG}_base_reward_10critics_seed${SEED}_${RUN_TAG}"
LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
export WANDB_RUN_NAME="$RUN_NAME"
export TOPREWARD_LOCK_PATH="$LOCK_PATH"

if [ -e "$LOG_DIR" ]; then
  echo "ERROR: refusing to overwrite existing online log dir: $LOG_DIR"
  exit 1
fi
mkdir -p "$LOG_DIR"

echo "========== TOPReward raw online ${ENV_ID} ${VARIANT} seed ${SEED} =========="
echo "TOPREWARD_API_URL=${TOPREWARD_API_URL}"
echo "TOPREWARD_LOCK_PATH=${TOPREWARD_LOCK_PATH}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "H5_PATH=${H5_PATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "HORIZON=${HORIZON}"
echo "DIFF_REWARD_SCALE=${DIFF_REWARD_SCALE}"
echo "MAX_FRAMES=${MAX_FRAMES}"
echo "REQUEST_TIMEOUT=${REQUEST_TIMEOUT}"
echo "TOPREWARD_REQUEST_FORMAT=${TOPREWARD_REQUEST_FORMAT}"
echo "offline_training.offline_training_steps=0"
echo "online_training.mix_buffers_ratio=0.0"
which python
python --version
nvidia-smi

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=topreward \
  offline_training.ckpt_path="$CKPT_PATH" \
  offline_training.offline_h5_path="$H5_PATH" \
  offline_training.offline_training_steps=0 \
  online_training.total_time_steps=100000 \
  online_training.mix_buffers_ratio=0.0 \
  general_training.seed="$SEED" \
  general_training.n_critics=10 \
  general_training.n_critics_to_sample=2 \
  environment.env_id="$ENV_ID" \
  reward_model.server_url="$TOPREWARD_API_URL" \
  reward_model.success_bonus=0 \
  reward_model.max_frames="$MAX_FRAMES" \
  reward_model.request_timeout="$REQUEST_TIMEOUT" \
  reward_model.request_retries="$REQUEST_RETRIES" \
  reward_model.request_format="$TOPREWARD_REQUEST_FORMAT" \
  reward_model.lock_path="$LOCK_PATH" \
  logging.wandb_group_name="$GROUP_NAME" \
  logging.log_dir="$LOG_DIR" \
  hydra.run.dir=. \
  "${EXTRA_ARGS[@]}"
