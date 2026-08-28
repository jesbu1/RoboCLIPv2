#!/usr/bin/env bash
# Shared Robometer server for seed32 group 8 jobs.

#SBATCH --job-name=robo_srv_s32_g8
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=d23-[10,13-16],e21-[01-16],e22-[01-16],e23-01
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/robometer_server_seed32_group8_%j.out
#SBATCH --error=logs/robometer_server_seed32_group8_%j.err

set -e

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
ROBOMETER_PROJECT_DIR="${ROBOMETER_PROJECT_DIR:-/project2/biyik_1165/haobaizh/rewind_robometer_valuemodel}"
ROBOMETER_CONDA="${ROBOMETER_CONDA:-${ROBOMETER_PROJECT_DIR}/conda_envs/robometer}"
SERVER_PORT="${SERVER_PORT:-8033}"
FALLBACK_START="${FALLBACK_START:-18326}"
FALLBACK_END="${FALLBACK_END:-18331}"
INFO_FILE="${PROJECT_DIR}/logs/robometer_server_seed32_group8_info.txt"
TMP_INFO_FILE="${INFO_FILE}.tmp"
SERVER_PID=""

if [ ! -f "${PROJECT_DIR}/test_scripts/test_iql.py" ]; then
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

echo "========== Robometer Server (seed32 group 8) =========="
which python
python --version
nvidia-smi

cleanup() {
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

pick_port() {
  local candidate
  for candidate in "$SERVER_PORT" $(seq "$FALLBACK_START" "$FALLBACK_END"); do
    if python - "$candidate" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("0.0.0.0", port))
except OSError:
    sys.exit(1)
sock.close()
sys.exit(0)
PY
    then
      echo "$candidate"
      return 0
    fi
  done

  echo "ERROR: Could not find a free server port for seed32 group 8" >&2
  return 1
}

SERVER_HOST=$(hostname)
mkdir -p "$(dirname "$INFO_FILE")"
rm -f "$INFO_FILE" "$TMP_INFO_FILE"
SERVER_PORT="$(pick_port)"
HEALTH_URL="http://${SERVER_HOST}:${SERVER_PORT}/health"

echo "Server will use: http://${SERVER_HOST}:${SERVER_PORT}"

cd "$ROBOMETER_PROJECT_DIR"
python -u robometer_server.py --port "$SERVER_PORT" &
SERVER_PID=$!

for i in $(seq 1 360); do
  if curl -s "${HEALTH_URL}" > /dev/null 2>&1; then
    echo "${SLURM_JOB_ID} http://${SERVER_HOST}:${SERVER_PORT}" > "$TMP_INFO_FILE"
    mv "$TMP_INFO_FILE" "$INFO_FILE"
    echo "Server ready at: http://${SERVER_HOST}:${SERVER_PORT}"
    wait "${SERVER_PID}"
    exit $?
  fi

  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: Robometer server process exited before becoming healthy"
    wait "${SERVER_PID}"
    exit 1
  fi

  sleep 5
done

echo "ERROR: Robometer server did not become healthy within 30 minutes"
exit 1
