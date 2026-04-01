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

set -e

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
ROBOMETER_PROJECT_DIR="${ROBOMETER_PROJECT_DIR:-/project2/biyik_1165/haobaizh/rewind_robometer_valuemodel}"
ROBOMETER_CONDA="${ROBOMETER_CONDA:-${ROBOMETER_PROJECT_DIR}/conda_envs/robometer}"
SERVER_PORT="${SERVER_PORT:-8000}"
INFO_FILE="${PROJECT_DIR}/logs/robometer_server_doorlock_pipeline_info.txt"
TMP_INFO_FILE="${INFO_FILE}.tmp"

if [ ! -f "${PROJECT_DIR}/scripts/run_robometer_doorlock_pipeline.sh" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"
  exit 1
fi
if [ ! -d "$ROBOMETER_PROJECT_DIR" ]; then
  echo "ERROR: Robometer project dir not found: ${ROBOMETER_PROJECT_DIR}"
  exit 1
fi
if [ ! -d "$ROBOMETER_CONDA" ]; then
  echo "ERROR: Robometer conda env not found: ${ROBOMETER_CONDA}"
  exit 1
fi
if [ ! -f "${ROBOMETER_PROJECT_DIR}/robometer_server.py" ]; then
  echo "ERROR: robometer_server.py not found under ${ROBOMETER_PROJECT_DIR}"
  exit 1
fi

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "$ROBOMETER_CONDA"

echo "========== Robometer Server =========="
which python
python --version
nvidia-smi

SERVER_HOST=$(hostname)
mkdir -p "$(dirname "$INFO_FILE")"
echo "http://${SERVER_HOST}:${SERVER_PORT}" > "$TMP_INFO_FILE"
mv "$TMP_INFO_FILE" "$INFO_FILE"
echo "Server at: http://${SERVER_HOST}:${SERVER_PORT}"

cd "$ROBOMETER_PROJECT_DIR"
python -u robometer_server.py --port "$SERVER_PORT"
