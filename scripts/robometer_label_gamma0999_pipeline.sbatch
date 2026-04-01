#!/usr/bin/env bash
# generate Robometer diff labels with gamma=0.999 using the shared server

#SBATCH --job-name=robo_lbl_g0999_pipe
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${PROJECT_DIR}/datasets}"
INFO_FILE="${PROJECT_DIR}/logs/robometer_server_doorlock_pipeline_info.txt"

if [ ! -f "${PROJECT_DIR}/scripts/generate_labeled_dataset_robometer.py" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"
  exit 1
fi
if [ ! -d "${PROJECT_DIR}/conda_envs/rewind_nochunk" ]; then
  echo "ERROR: rewind_nochunk conda env not found under ${PROJECT_DIR}/conda_envs/rewind_nochunk"
  exit 1
fi

module purge
module load gcc/13.3.0

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "${PROJECT_DIR}/conda_envs/rewind_nochunk"

export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"

cd "$PROJECT_DIR"

for i in $(seq 1 360); do
  if [ -s "$INFO_FILE" ]; then break; fi
  if [ $i -eq 360 ]; then echo "ERROR: Server info not found after 30min"; exit 1; fi
  sleep 5
done
SERVER_URL=$(tr -d '\r\n' < "$INFO_FILE")
if [ -z "$SERVER_URL" ]; then echo "ERROR: Server URL is empty"; exit 1; fi
echo "Using server: $SERVER_URL"

for i in $(seq 1 360); do
  if curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo "Server ready!"
    break
  fi
  if [ $i -eq 360 ]; then echo "ERROR: Server not ready after 30min"; exit 1; fi
  sleep 5
done

srun python -u scripts/generate_labeled_dataset_robometer.py \
  --h5_video_path "$RAW_DATA_DIR/metaworld_generation.h5" \
  --h5_embedding_path "$RAW_DATA_DIR/metaworld_embeddings_train.h5" \
  --server_url "$SERVER_URL" \
  --use_progress_diff \
  --diff_gamma 0.999 \
  --output_path datasets/metaworld_labeled_robometer_diff_gamma0999.h5
