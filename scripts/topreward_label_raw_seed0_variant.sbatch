#!/usr/bin/env bash
# Label TOPReward baseline or diff-gamma1 H5 using a raw no-chat TOPReward server.

#SBATCH --job-name=top_lbl_raw
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
TOPREWARD_DIR="${TOPREWARD_DIR:-/scratch1/haobaizh/rewind_topreward}"
TOPREWARD_CONDA="${TOPREWARD_CONDA:-${TOPREWARD_DIR}/conda_envs/rewind}"
PROJECT_CACHE="${PROJECT_CACHE:-/scratch1/haobaizh/.cache}"
INFO_FILE="${INFO_FILE:?INFO_FILE must point to a ready TOPReward server info file}"
EXPECTED_SERVER_JOB_ID="${EXPECTED_SERVER_JOB_ID:-}"
VARIANT="${VARIANT:?VARIANT must be baseline or diff_gamma1}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH must be set and unique}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
MAX_FRAMES="${MAX_FRAMES:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"
TOPREWARD_REQUEST_FORMAT="${TOPREWARD_REQUEST_FORMAT:-raw}"
LOCK_PATH="${LOCK_PATH:-${TOPREWARD_DIR}/logs/topreward_raw_label.lock}"
DIFF_GAMMA="${DIFF_GAMMA:-1.0}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-false}"

case "$VARIANT" in
  baseline)
    MODE="baseline"
    REWARD_SCALE="${BASELINE_REWARD_SCALE:-1.0}"
    EXTRA_ARGS=()
    ;;
  diff_gamma1)
    MODE="diff"
    REWARD_SCALE="${DIFF_LABEL_REWARD_SCALE:-1.0}"
    EXTRA_ARGS=(--diff_gamma "$DIFF_GAMMA")
    ;;
  *)
    echo "ERROR: Unknown VARIANT=${VARIANT}"
    exit 1
    ;;
esac

if [ ! -d "$TOPREWARD_DIR" ]; then echo "ERROR: TOPREWARD_DIR not found: $TOPREWARD_DIR"; exit 1; fi
if [ ! -d "$TOPREWARD_CONDA" ]; then echo "ERROR: TOPReward conda env not found: $TOPREWARD_CONDA"; exit 1; fi
if [ ! -f "${PROJECT_DIR}/scripts/topreward_label_reward_gamma.py" ]; then echo "ERROR: label helper not found"; exit 1; fi
if [ -e "$OUTPUT_PATH" ] && [ "$OVERWRITE_OUTPUT" != "true" ]; then
  echo "ERROR: refusing to overwrite existing label H5: $OUTPUT_PATH"
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
  if [ "$i" -eq 720 ]; then echo "ERROR: Server info not ready after 60 minutes"; exit 1; fi
  sleep 5
done

for i in $(seq 1 720); do
  if curl -fsS "${TOPREWARD_API_URL}/health" >/dev/null 2>&1; then
    break
  fi
  if [ "$i" -eq 720 ]; then echo "ERROR: Server health not ready after 60 minutes"; exit 1; fi
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
conda activate "$TOPREWARD_CONDA"

export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
GCC_LIBDIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

export HF_HOME="${PROJECT_CACHE}/huggingface"
export TORCH_HOME="${PROJECT_CACHE}/torch"
export XDG_CACHE_HOME="${PROJECT_CACHE}"
export TRANSFORMERS_CACHE="${PROJECT_CACHE}/huggingface"
mkdir -p "${TOPREWARD_DIR}/logs" "$(dirname "$OUTPUT_PATH")" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"

echo "========== TOPReward raw label ${VARIANT} =========="
echo "TOPREWARD_API_URL=${TOPREWARD_API_URL}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "LOCK_PATH=${LOCK_PATH}"
echo "MODE=${MODE}"
echo "REWARD_SCALE=${REWARD_SCALE}"
echo "DIFF_GAMMA=${DIFF_GAMMA}"
echo "MAX_FRAMES=${MAX_FRAMES}"
echo "REQUEST_TIMEOUT=${REQUEST_TIMEOUT}"
echo "TOPREWARD_REQUEST_FORMAT=${TOPREWARD_REQUEST_FORMAT}"
which python
python --version
nvidia-smi

cd "$PROJECT_DIR"
srun python -u scripts/topreward_label_reward_gamma.py \
  --topreward_dir "$TOPREWARD_DIR" \
  --h5_video_path "${TOPREWARD_DIR}/datasets/metaworld_generation.h5" \
  --h5_embedding_path "${TOPREWARD_DIR}/datasets/metaworld_embeddings_train.h5" \
  --output_path "$OUTPUT_PATH" \
  --api_url "$TOPREWARD_API_URL" \
  --model_name "$MODEL_NAME" \
  --num_prefix_samples "$MAX_FRAMES" \
  --request_timeout "$REQUEST_TIMEOUT" \
  --request_retries "$REQUEST_RETRIES" \
  --request_format "$TOPREWARD_REQUEST_FORMAT" \
  --mode "$MODE" \
  --reward_scale "$REWARD_SCALE" \
  --lock_path "$LOCK_PATH" \
  "${EXTRA_ARGS[@]}"
