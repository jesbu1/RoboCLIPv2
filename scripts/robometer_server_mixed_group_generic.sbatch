#!/usr/bin/env bash
# Generic shared Robometer server for mixed-seed grouped jobs.

#SBATCH --job-name=robo_srv_mix
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=d23-[10,13-16],e21-[01-16],e22-[01-16],e23-01
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
ROBOMETER_PROJECT_DIR="${ROBOMETER_PROJECT_DIR:-/project2/biyik_1165/haobaizh/rewind_robometer_valuemodel}"
ROBOMETER_CONDA="${ROBOMETER_CONDA:-${ROBOMETER_PROJECT_DIR}/conda_envs/robometer}"
GROUP_ID="${GROUP_ID:-unknown}"
SERVER_LABEL="${SERVER_LABEL:-mixed_group_${GROUP_ID}}"
SERVER_PORT="${SERVER_PORT:-8040}"
FALLBACK_START="${FALLBACK_START:-18368}"
FALLBACK_END="${FALLBACK_END:-18373}"
INFO_FILE="${INFO_FILE:-${PROJECT_DIR}/logs/robometer_server_mixed_group${GROUP_ID}_info.txt}"
TMP_INFO_FILE="${INFO_FILE}.tmp"
STARTUP_DELAY_SEC="${STARTUP_DELAY_SEC:-0}"
STARTUP_MAX_ATTEMPTS="${STARTUP_MAX_ATTEMPTS:-6}"
STARTUP_BACKOFF_BASE_SEC="${STARTUP_BACKOFF_BASE_SEC:-120}"
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

echo "========== Robometer Server (${SERVER_LABEL}) =========="
which python
python --version
nvidia-smi
echo "Startup delay: ${STARTUP_DELAY_SEC}s"
echo "Startup max attempts: ${STARTUP_MAX_ATTEMPTS}"
echo "Startup backoff base: ${STARTUP_BACKOFF_BASE_SEC}s"

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

  echo "ERROR: Could not find a free server port for ${SERVER_LABEL}" >&2
  return 1
}

SERVER_HOST=$(hostname)
mkdir -p "$(dirname "$INFO_FILE")"
rm -f "$INFO_FILE" "$TMP_INFO_FILE"
cd "$ROBOMETER_PROJECT_DIR"

if [ "$STARTUP_DELAY_SEC" -gt 0 ]; then
  sleep "$STARTUP_DELAY_SEC"
fi

for attempt in $(seq 1 "$STARTUP_MAX_ATTEMPTS"); do
  SERVER_PORT="$(pick_port)"
  HEALTH_URL="http://${SERVER_HOST}:${SERVER_PORT}/health"

  echo "Launch attempt ${attempt}/${STARTUP_MAX_ATTEMPTS}"
  echo "Server will use: http://${SERVER_HOST}:${SERVER_PORT}"

  python -u robometer_server.py --port "$SERVER_PORT" &
  SERVER_PID=$!

  attempt_failed=0
  for i in $(seq 1 360); do
    if curl -s "${HEALTH_URL}" > /dev/null 2>&1; then
      echo "${SLURM_JOB_ID} http://${SERVER_HOST}:${SERVER_PORT}" > "$TMP_INFO_FILE"
      mv "$TMP_INFO_FILE" "$INFO_FILE"
      echo "Server ready at: http://${SERVER_HOST}:${SERVER_PORT}"
      wait "${SERVER_PID}"
      exit $?
    fi

    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      if wait "${SERVER_PID}"; then
        server_rc=0
      else
        server_rc=$?
      fi
      SERVER_PID=""
      echo "ERROR: Robometer server process exited before becoming healthy (attempt ${attempt}/${STARTUP_MAX_ATTEMPTS}, exit=${server_rc})"
      attempt_failed=1
      break
    fi

    sleep 5
  done

  if [ "$attempt_failed" -eq 0 ]; then
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill "${SERVER_PID}" 2>/dev/null || true
      wait "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
    echo "ERROR: Robometer server did not become healthy within 30 minutes (attempt ${attempt}/${STARTUP_MAX_ATTEMPTS})"
    attempt_failed=1
  fi

  if [ "$attempt" -lt "$STARTUP_MAX_ATTEMPTS" ]; then
    backoff_sec=$((STARTUP_BACKOFF_BASE_SEC * attempt + RANDOM % 60))
    echo "Retrying after ${backoff_sec}s"
    sleep "$backoff_sec"
  fi
done

echo "ERROR: Exhausted Robometer server startup attempts for ${SERVER_LABEL}"
exit 1
