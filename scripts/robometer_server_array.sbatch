#!/usr/bin/env bash
#SBATCH --job-name=robo_srv
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=d23-[10,13-16],e21-[01-16],e22-[01-16],e23-01
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/robometer_server_%A_%a.out
#SBATCH --error=logs/robometer_server_%A_%a.err
#SBATCH --array=0-7

set -e

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
ROBOMETER_CONDA="${ROBOMETER_CONDA:-${PROJECT_DIR}/conda_envs/robometer}"

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "$ROBOMETER_CONDA"

SERVER_HOST=$(hostname)
SERVER_PORT=$((8000 + SLURM_ARRAY_TASK_ID))
INFO_FILE="${PROJECT_DIR}/logs/robometer_server_info_${SLURM_ARRAY_TASK_ID}.txt"
mkdir -p "$(dirname "$INFO_FILE")"
echo "http://${SERVER_HOST}:${SERVER_PORT}" > "$INFO_FILE"
echo "Server [${SLURM_ARRAY_TASK_ID}] at: http://${SERVER_HOST}:${SERVER_PORT}"

cd "$PROJECT_DIR"
python -u scripts/robometer_server.py --port $SERVER_PORT
