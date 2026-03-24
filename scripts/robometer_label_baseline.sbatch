#!/usr/bin/env bash
#SBATCH --job-name=robo_lbl_bl
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=/project2/biyik_1165/haobaizh/rewind_no-action-chunk/logs/%x_%j.out
#SBATCH --error=/project2/biyik_1165/haobaizh/rewind_no-action-chunk/logs/%x_%j.err

set -e

module purge
module load gcc/13.3.0

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate /project2/biyik_1165/haobaizh/rewind_no-action-chunk/conda_envs/rewind_nochunk

export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"

cd /project2/biyik_1165/haobaizh/rewind_no-action-chunk

# Read server URL
INFO_FILE=logs/robometer_server_baseline_info.txt
if [ ! -f "$INFO_FILE" ]; then
  echo "ERROR: Server info not found. Start robometer_server_baseline first."
  exit 1
fi
SERVER_URL=$(cat "$INFO_FILE")
echo "Using server: $SERVER_URL"

# Wait for server to be ready
for i in $(seq 1 360); do
  if curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo "Server ready!"
    break
  fi
  if [ $i -eq 360 ]; then echo "ERROR: Server not ready after 30min"; exit 1; fi
  sleep 5
done

srun python -u scripts/generate_labeled_dataset_robometer.py \
  --server_url "$SERVER_URL" \
  --output_path datasets/metaworld_labeled_robometer.h5
