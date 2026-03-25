#!/usr/bin/env bash
#SBATCH --job-name=robo_srv_bl
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=d23-[10,13-16],e21-[01-16],e22-[01-16],e23-01
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/robometer_server_baseline_%j.out
#SBATCH --error=logs/robometer_server_baseline_%j.err

set -e

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
ROBOMETER_CONDA="${ROBOMETER_CONDA:-${PROJECT_DIR}/conda_envs/robometer}"

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "$ROBOMETER_CONDA"

SERVER_HOST=$(hostname)
SERVER_PORT=8000
INFO_FILE="${PROJECT_DIR}/logs/robometer_server_baseline_info.txt"
mkdir -p "$(dirname "$INFO_FILE")"
echo "http://${SERVER_HOST}:${SERVER_PORT}" > "$INFO_FILE"
echo "Server at: http://${SERVER_HOST}:${SERVER_PORT}"

cd "$PROJECT_DIR"
python -u scripts/robometer_server.py
