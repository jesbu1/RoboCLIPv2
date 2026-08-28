#!/usr/bin/env bash
# Generic TOPReward vLLM server used by grouped online jobs.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
TOPREWARD_DIR="${TOPREWARD_DIR:-/scratch1/haobaizh/rewind_topreward}"
CONDA_ENV="${CONDA_ENV:-${TOPREWARD_DIR}/conda_envs/vllm}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
SERVER_BACKEND="${SERVER_BACKEND:-vllm}"
TOPREWARD_ATTN_IMPLEMENTATION="${TOPREWARD_ATTN_IMPLEMENTATION:-auto}"
TOPREWARD_PROMPT_LOGPROBS="${TOPREWARD_PROMPT_LOGPROBS:-20}"
TOPREWARD_SERVER_MAX_NUM_SEQS="${TOPREWARD_SERVER_MAX_NUM_SEQS:-8}"
TOPREWARD_SERVER_MAX_MODEL_LEN="${TOPREWARD_SERVER_MAX_MODEL_LEN:-32768}"
TOPREWARD_SERVER_GPU_MEMORY_UTILIZATION="${TOPREWARD_SERVER_GPU_MEMORY_UTILIZATION:-0.90}"
PROJECT_CACHE="${PROJECT_CACHE:-/scratch1/haobaizh/.cache}"
GROUP_ID="${GROUP_ID:-unknown}"
SERVER_LABEL="${SERVER_LABEL:-topreward_group_${GROUP_ID}}"
SERVER_PORT="${SERVER_PORT:-8200}"
FALLBACK_START="${FALLBACK_START:-18800}"
FALLBACK_END="${FALLBACK_END:-18820}"
INFO_FILE="${INFO_FILE:-${PROJECT_DIR}/logs/topreward_server_group${GROUP_ID}_info.txt}"
TMP_INFO_FILE="${INFO_FILE}.tmp"
SERVER_PID=""

if [ ! -d "${TOPREWARD_DIR}" ]; then
  echo "ERROR: TOPREWARD_DIR not found: ${TOPREWARD_DIR}"
  exit 1
fi
if [ ! -d "${CONDA_ENV}" ]; then
  echo "ERROR: vLLM conda env not found: ${CONDA_ENV}"
  exit 1
fi

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
export TMPDIR="/tmp/topreward_vllm_${SLURM_JOB_ID}"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_HOME}" "${TMPDIR}" \
  "${PIP_CACHE_DIR}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CONFIG_HOME}" \
  "${XDG_DATA_HOME}" "${TVM_HOME}" "$(dirname "${INFO_FILE}")"

echo "========== TOPReward Server (${SERVER_LABEL}) =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "NODE=$(hostname)"
echo "TOPREWARD_DIR=${TOPREWARD_DIR}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SERVER_BACKEND=${SERVER_BACKEND}"
echo "TOPREWARD_ATTN_IMPLEMENTATION=${TOPREWARD_ATTN_IMPLEMENTATION}"
echo "TOPREWARD_PROMPT_LOGPROBS=${TOPREWARD_PROMPT_LOGPROBS}"
echo "TOPREWARD_SERVER_MAX_NUM_SEQS=${TOPREWARD_SERVER_MAX_NUM_SEQS}"
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
  return 1
}

SERVER_HOST="$(hostname)"
rm -f "${INFO_FILE}" "${TMP_INFO_FILE}"
SERVER_PORT="$(pick_port)"
HEALTH_URL="http://${SERVER_HOST}:${SERVER_PORT}/health"
echo "Server will use: http://${SERVER_HOST}:${SERVER_PORT}"

case "$SERVER_BACKEND" in
  vllm)
    python -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_NAME}" \
      --host 0.0.0.0 \
      --port "${SERVER_PORT}" \
      --dtype auto \
      --max-model-len 32768 \
      --max-num-seqs 1 \
      --trust-remote-code \
      --gpu-memory-utilization 0.90 &
    ;;
  raw)
    python -u "${PROJECT_DIR}/scripts/topreward_raw_score_server.py" \
      --model "${MODEL_NAME}" \
      --host 0.0.0.0 \
      --port "${SERVER_PORT}" \
      --prompt-logprobs "${TOPREWARD_PROMPT_LOGPROBS}" \
      --max-num-seqs "${TOPREWARD_SERVER_MAX_NUM_SEQS}" \
      --max-model-len "${TOPREWARD_SERVER_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${TOPREWARD_SERVER_GPU_MEMORY_UTILIZATION}" &
    ;;
  *)
    echo "ERROR: unsupported SERVER_BACKEND=${SERVER_BACKEND}; expected vllm or raw"
    exit 1
    ;;
esac
SERVER_PID=$!

for i in $(seq 1 360); do
  if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "${SLURM_JOB_ID} http://${SERVER_HOST}:${SERVER_PORT}" > "${TMP_INFO_FILE}"
    mv "${TMP_INFO_FILE}" "${INFO_FILE}"
    echo "Server ready at: http://${SERVER_HOST}:${SERVER_PORT}"
    wait "${SERVER_PID}"
    exit $?
  fi

  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}" || server_rc=$?
    echo "ERROR: TOPReward vLLM server exited before healthy, exit=${server_rc:-0}"
    exit 1
  fi
  sleep 5
done

echo "ERROR: TOPReward vLLM server did not become healthy within 30 minutes"
exit 1
