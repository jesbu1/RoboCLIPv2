#!/usr/bin/env bash
#SBATCH --job-name=top_raw_pre
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
TOPREWARD_DIR="${TOPREWARD_DIR:-/scratch1/haobaizh/rewind_topreward}"
CONDA_ENV="${CONDA_ENV:-${TOPREWARD_DIR}/conda_envs/vllm}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
PROJECT_CACHE="${PROJECT_CACHE:-/scratch1/haobaizh/.cache}"
ALLOW_HF_DOWNLOAD="${ALLOW_HF_DOWNLOAD:-0}"
TOPREWARD_ATTN_IMPLEMENTATION="${TOPREWARD_ATTN_IMPLEMENTATION:-auto}"

echo "========== TOPReward raw server lightweight preflight =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "NODE=$(hostname)"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "TOPREWARD_DIR=${TOPREWARD_DIR}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "ALLOW_HF_DOWNLOAD=${ALLOW_HF_DOWNLOAD}"
echo "TOPREWARD_ATTN_IMPLEMENTATION=${TOPREWARD_ATTN_IMPLEMENTATION}"
echo "==============================================================="

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "ERROR: PROJECT_DIR not found: ${PROJECT_DIR}"
  exit 1
fi
if [ ! -d "${CONDA_ENV}" ]; then
  echo "ERROR: conda env not found: ${CONDA_ENV}"
  exit 1
fi

cd "${PROJECT_DIR}"

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
export TMPDIR="/tmp/topreward_raw_preflight_${SLURM_JOB_ID:-manual}"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_HOME}" "${TMPDIR}" \
  "${PIP_CACHE_DIR}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CONFIG_HOME}" \
  "${XDG_DATA_HOME}" "${TVM_HOME}" logs

which python
python --version

args=(
  --model "${MODEL_NAME}"
  --server-script "${PROJECT_DIR}/scripts/topreward_raw_score_server.py"
  --attn-implementation "${TOPREWARD_ATTN_IMPLEMENTATION}"
)
if [ "${ALLOW_HF_DOWNLOAD}" = "1" ]; then
  args+=(--allow-download)
fi

python -u "${PROJECT_DIR}/scripts/topreward_raw_server_preflight.py" "${args[@]}"
