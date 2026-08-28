#!/usr/bin/env bash
# Run the complete raw TOPReward check in one log:
#   1) existing import preflight
#   2) existing raw server preflight
#   3) flash_attention_2/import/backend check
#   4) real raw server health + optional /score smoke

#SBATCH --job-name=top_raw_fullchk
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
TOPREWARD_DIR="${TOPREWARD_DIR:-/scratch1/haobaizh/rewind_topreward}"
CONDA_ENV="${CONDA_ENV:-${TOPREWARD_DIR}/conda_envs/vllm}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
PROJECT_CACHE="${PROJECT_CACHE:-/scratch1/haobaizh/.cache}"
ALLOW_HF_DOWNLOAD="${ALLOW_HF_DOWNLOAD:-0}"
TOPREWARD_ATTN_IMPLEMENTATION="${TOPREWARD_ATTN_IMPLEMENTATION:-flash_attention_2}"
RUN_SCORE_SMOKE="${RUN_SCORE_SMOKE:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
SERVER_PORT="${SERVER_PORT:-8200}"
FALLBACK_START="${FALLBACK_START:-18800}"
FALLBACK_END="${FALLBACK_END:-18820}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-720}"
INFO_FILE="${INFO_FILE:-${PROJECT_DIR}/logs/topreward_raw_full_check_${SLURM_JOB_ID:-manual}_info.txt}"
SERVER_PID=""

section() {
  echo
  echo "========== $* =========="
}

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
  return 1
}

if [ ! -d "${PROJECT_DIR}" ]; then echo "ERROR: PROJECT_DIR not found: ${PROJECT_DIR}"; exit 1; fi
if [ ! -d "${TOPREWARD_DIR}" ]; then echo "ERROR: TOPREWARD_DIR not found: ${TOPREWARD_DIR}"; exit 1; fi
if [ ! -d "${CONDA_ENV}" ]; then echo "ERROR: CONDA_ENV not found: ${CONDA_ENV}"; exit 1; fi

cd "${PROJECT_DIR}"
mkdir -p logs "$(dirname "${INFO_FILE}")"

section "TOPReward raw full preflight + smoke"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "NODE=$(hostname)"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "TOPREWARD_DIR=${TOPREWARD_DIR}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "ALLOW_HF_DOWNLOAD=${ALLOW_HF_DOWNLOAD}"
echo "TOPREWARD_ATTN_IMPLEMENTATION=${TOPREWARD_ATTN_IMPLEMENTATION}"
echo "RUN_SCORE_SMOKE=${RUN_SCORE_SMOKE}"
echo "REQUEST_TIMEOUT=${REQUEST_TIMEOUT}"
echo "INFO_FILE=${INFO_FILE}"
nvidia-smi || true

section "1) Existing TOPReward import preflight output"
TOPREWARD_DIR="${TOPREWARD_DIR}" \
CONDA_ENV="${CONDA_ENV}" \
bash "${PROJECT_DIR}/scripts/topreward_preflight_import_test.sbatch"

section "2) Existing raw server preflight output"
PROJECT_DIR="${PROJECT_DIR}" \
TOPREWARD_DIR="${TOPREWARD_DIR}" \
CONDA_ENV="${CONDA_ENV}" \
MODEL_NAME="${MODEL_NAME}" \
PROJECT_CACHE="${PROJECT_CACHE}" \
ALLOW_HF_DOWNLOAD="${ALLOW_HF_DOWNLOAD}" \
TOPREWARD_ATTN_IMPLEMENTATION="${TOPREWARD_ATTN_IMPLEMENTATION}" \
bash "${PROJECT_DIR}/scripts/topreward_raw_server_preflight.sbatch"

section "3) flash_attention_2 package/backend check"
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export HF_HOME="${PROJECT_CACHE}/huggingface"
export TRITON_CACHE_DIR="${PROJECT_CACHE}/triton"
export TORCH_HOME="${PROJECT_CACHE}/torch"
export XDG_CACHE_HOME="${PROJECT_CACHE}"
export PIP_CACHE_DIR="${PROJECT_CACHE}/pip"
export MPLCONFIGDIR="${PROJECT_CACHE}/matplotlib"
export NUMBA_CACHE_DIR="${PROJECT_CACHE}/numba"
export CUDA_CACHE_PATH="${PROJECT_CACHE}/nv"
export TORCH_EXTENSIONS_DIR="${PROJECT_CACHE}/torch_extensions"
export XDG_CONFIG_HOME="${PROJECT_CACHE}/config"
export XDG_DATA_HOME="${PROJECT_CACHE}/local_share"
export TVM_HOME="${PROJECT_CACHE}/tvm"
export VLLM_USAGE_STATS=0
export TMPDIR="/tmp/topreward_raw_full_check_${SLURM_JOB_ID:-manual}"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_HOME}" "${TMPDIR}" \
  "${PIP_CACHE_DIR}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CONFIG_HOME}" \
  "${XDG_DATA_HOME}" "${TVM_HOME}"

which python
python --version

python -u <<'PY'
import importlib
import torch
import transformers

print(f"torch={torch.__version__}", flush=True)
print(f"torch_cuda={torch.version.cuda}", flush=True)
print(f"torch.cuda.is_available={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"gpu_name={torch.cuda.get_device_name(0)}", flush=True)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"gpu_capability=sm_{major}{minor}", flush=True)
print(f"transformers={transformers.__version__}", flush=True)

flash_attn = importlib.import_module("flash_attn")
print(f"flash_attn={getattr(flash_attn, '__version__', 'unknown')}", flush=True)
print("flash_attn import OK", flush=True)
PY

section "4) Real raw server health and score smoke"
SERVER_HOST="$(hostname)"
rm -f "${INFO_FILE}" "${INFO_FILE}.tmp"
SERVER_PORT="$(pick_port)"
HEALTH_URL="http://${SERVER_HOST}:${SERVER_PORT}/health"
SCORE_URL="http://${SERVER_HOST}:${SERVER_PORT}/score"
echo "Server will use: http://${SERVER_HOST}:${SERVER_PORT}"

python -u "${PROJECT_DIR}/scripts/topreward_raw_score_server.py" \
  --model "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${SERVER_PORT}" \
  --attn-implementation "${TOPREWARD_ATTN_IMPLEMENTATION}" &
SERVER_PID=$!

for i in $(seq 1 "${SERVER_WAIT_ATTEMPTS}"); do
  if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "${SLURM_JOB_ID:-manual} http://${SERVER_HOST}:${SERVER_PORT}" > "${INFO_FILE}.tmp"
    mv "${INFO_FILE}.tmp" "${INFO_FILE}"
    echo "Server ready at: http://${SERVER_HOST}:${SERVER_PORT}"
    curl -fsS "${HEALTH_URL}"
    echo
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}" || server_rc=$?
    echo "ERROR: raw TOPReward server exited before healthy, exit=${server_rc:-0}"
    exit 1
  fi
  if [ "$i" -eq "${SERVER_WAIT_ATTEMPTS}" ]; then
    echo "ERROR: raw TOPReward server did not become healthy"
    exit 1
  fi
  sleep 5
done

if [ "${RUN_SCORE_SMOKE}" = "1" ]; then
  python -u - "${SCORE_URL}" "${MODEL_NAME}" "${REQUEST_TIMEOUT}" <<'PY'
import base64
import io
import json
import sys

import requests
from PIL import Image

score_url = sys.argv[1]
model_name = sys.argv[2]
timeout = float(sys.argv[3])

frames_b64 = []
for shade in (32, 224):
    image = Image.new("RGB", (224, 224), (shade, shade, shade))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    frames_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

payload = {
    "model": model_name,
    "frames_b64": frames_b64,
    "instruction": "open the drawer",
}
resp = requests.post(score_url, json=payload, timeout=timeout)
print(f"score_status={resp.status_code}", flush=True)
resp.raise_for_status()
result = resp.json()
print(json.dumps(result, indent=2, sort_keys=True), flush=True)
print("score smoke OK", flush=True)
PY
else
  echo "RUN_SCORE_SMOKE=${RUN_SCORE_SMOKE}; skipping /score smoke"
fi

echo "TOPReward raw full check PASSED"
