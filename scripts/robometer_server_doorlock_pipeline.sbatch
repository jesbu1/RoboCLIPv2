#!/usr/bin/env bash
# One shared Robometer server for:
#   - baseline offline
#   - diff gamma=0.99 offline
#   - diff gamma=0.999 labeling
#   - all 9 online jobs (baseline / 0.99 / 0.999 x 3 seeds)

#SBATCH --job-name=robo_srv_dlock
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=d23-[10,13-16],e21-[01-16],e22-[01-16],e23-01
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/robometer_server_doorlock_%j.out
#SBATCH --error=logs/robometer_server_doorlock_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
ROBOMETER_CONDA="${ROBOMETER_CONDA:-${PROJECT_DIR}/conda_envs/robometer}"
SERVER_PORT="${SERVER_PORT:-8000}"
INFO_FILE="${PROJECT_DIR}/logs/robometer_server_doorlock_pipeline_info.txt"

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "$ROBOMETER_CONDA"

SERVER_HOST=$(hostname)
mkdir -p "$(dirname "$INFO_FILE")"
echo "http://${SERVER_HOST}:${SERVER_PORT}" > "$INFO_FILE"
echo "Server at: http://${SERVER_HOST}:${SERVER_PORT}"

cd "$PROJECT_DIR"
python -u scripts/robometer_server.py --port "$SERVER_PORT"
