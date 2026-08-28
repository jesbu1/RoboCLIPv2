#!/usr/bin/env bash
# Generate Robometer diff labels with gamma=1.0 using one healthy server.

#SBATCH --job-name=robo_lbl_d1
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/project2/biyik_1165/haobaizh/rewind_robometer_valuemodel/datasets}"
INFO_FILE="${INFO_FILE:?INFO_FILE must point to a ready Robometer server info file}"
EXPECTED_SERVER_JOB_ID="${EXPECTED_SERVER_JOB_ID:-}"
MAX_FRAMES="${MAX_FRAMES:-4}"

if [ ! -f "${PROJECT_DIR}/scripts/generate_labeled_dataset_robometer.py" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"
  exit 1
fi
if [ ! -d "${PROJECT_DIR}/conda_envs/rewind_nochunk" ]; then
  echo "ERROR: rewind_nochunk conda env not found under ${PROJECT_DIR}/conda_envs/rewind_nochunk"
  exit 1
fi
if [ ! -f "${RAW_DATA_DIR}/metaworld_generation.h5" ]; then
  echo "ERROR: metaworld_generation.h5 not found under ${RAW_DATA_DIR}"
  exit 1
fi
if [ ! -f "${RAW_DATA_DIR}/metaworld_embeddings_train.h5" ]; then
  echo "ERROR: metaworld_embeddings_train.h5 not found under ${RAW_DATA_DIR}"
  exit 1
fi

module purge
module load gcc/13.3.0

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "${PROJECT_DIR}/conda_envs/rewind_nochunk"

export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export TORCH_HOME="${TORCH_HOME:-/home1/haobaizh/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/home1/haobaizh/.cache}"

cd "$PROJECT_DIR"
mkdir -p logs datasets

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

for i in $(seq 1 1440); do
  if [ -s "$INFO_FILE" ] && SERVER_URL="$(read_server_url)"; then
    break
  fi
  if [ "$i" -eq 1440 ]; then echo "ERROR: Server info not found after 120min"; exit 1; fi
  sleep 5
done
if [ -z "${SERVER_URL:-}" ]; then echo "ERROR: Server URL is empty"; exit 1; fi
echo "Using server: $SERVER_URL"

for i in $(seq 1 1440); do
  if curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo "Server ready!"
    break
  fi
  if [ "$i" -eq 1440 ]; then echo "ERROR: Server not ready after 120min"; exit 1; fi
  sleep 5
done

srun python -u scripts/generate_labeled_dataset_robometer.py \
  --h5_video_path "$RAW_DATA_DIR/metaworld_generation.h5" \
  --h5_embedding_path "$RAW_DATA_DIR/metaworld_embeddings_train.h5" \
  --server_url "$SERVER_URL" \
  --max_frames "$MAX_FRAMES" \
  --use_progress_diff \
  --diff_gamma 1.0 \
  --output_path datasets/metaworld_labeled_robometer_diff_gamma1.h5
