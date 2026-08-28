#!/usr/bin/env bash
#SBATCH --job-name=rw_lbl_exp_b2_b
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch1/haobaizh/rewind_nochunk_label_logs/%x_%j.out
#SBATCH --error=/scratch1/haobaizh/rewind_nochunk_label_logs/%x_%j.err

# Generate the no-action-chunk exponential-baseline offline H5 using an
# exponential-progress ReWiND reward model. Inputs are staged to node-local
# TMPDIR; output stays on scratch1 so it survives job cleanup.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/project2/biyik_1165/haobaizh/rewind_no-action-chunk}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/scratch1/haobaizh/rewind_valuemodel/datasets}"
SCRATCH_DIR="${REWIND_SCRATCH_DIR:-/scratch1/haobaizh/rewind_valuemodel}"
CONDA_ENV="${CONDA_ENV:-${PROJECT_DIR}/conda_envs/rewind_nochunk}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${SCRATCH_DIR}/checkpoints/rewind_metaworld_exp_beta2_epoch_19.pth}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRATCH_DIR}/datasets/metaworld_labeled_exp_beta2.h5}"
BETA="${BETA:-2.0}"
EXP_HORIZON="${EXP_HORIZON:-128}"
EXP_DIFF_SCALE="${EXP_DIFF_SCALE:-44.46704380607797}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-false}"
STAGE_DATA="${STAGE_DATA:-1}"

mkdir -p \
  /scratch1/haobaizh/rewind_nochunk_label_logs \
  "${SCRATCH_DIR}/datasets" \
  "${SCRATCH_DIR}/.cache/torch"

if [ ! -f "${PROJECT_DIR}/scripts/generate_labeled_dataset.py" ]; then
  echo "ERROR: missing generate_labeled_dataset.py in ${PROJECT_DIR}" >&2
  exit 2
fi

if [ ! -f "${REWARD_MODEL_PATH}" ]; then
  echo "ERROR: missing reward model checkpoint: ${REWARD_MODEL_PATH}" >&2
  exit 2
fi

for file_name in metaworld_generation.h5 metaworld_embeddings_train.h5; do
  if [ ! -f "${RAW_DATA_DIR}/${file_name}" ]; then
    echo "ERROR: missing input H5: ${RAW_DATA_DIR}/${file_name}" >&2
    exit 2
  fi
done

if [ -e "${OUTPUT_PATH}" ] && [ "${OVERWRITE_OUTPUT}" != "true" ]; then
  echo "ERROR: refusing to overwrite existing output: ${OUTPUT_PATH}" >&2
  echo "Set OVERWRITE_OUTPUT=true if you really want to replace it." >&2
  exit 2
fi

TRAIN_DATA_DIR="${RAW_DATA_DIR}"
if [ "${STAGE_DATA}" = "1" ]; then
  TRAIN_DATA_DIR="${TMPDIR:-/tmp}/rewind_label_exp_beta2_baseline_inputs"
  mkdir -p "${TRAIN_DATA_DIR}"

  echo "========== Stage label inputs to local TMPDIR =========="
  echo "source=${RAW_DATA_DIR}"
  echo "target=${TRAIN_DATA_DIR}"
  date
  for file_name in metaworld_generation.h5 metaworld_embeddings_train.h5; do
    echo "copy ${file_name}"
    cp -p "${RAW_DATA_DIR}/${file_name}" "${TRAIN_DATA_DIR}/${file_name}"
  done
  date
  du -sh "${TRAIN_DATA_DIR}" || true
fi

module purge
module load gcc/13.3.0
module load mesa/24.2.3
module load mesa-glu/9.0.2
module load glew/2.2.0
module load glx/1.4

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export REWIND_SCRATCH_DIR="${SCRATCH_DIR}"
export TORCH_HOME="${SCRATCH_DIR}/.cache/torch"
export TOKENIZERS_PARALLELISM=false
export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
GCC_LIBDIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

if [ "${STAGE_DATA}" = "1" ]; then
  echo "========== Verify staged label inputs =========="
  export RAW_DATA_DIR TRAIN_DATA_DIR
  python - <<'PY'
import os
import h5py

src_dir = os.environ["RAW_DATA_DIR"]
dst_dir = os.environ["TRAIN_DATA_DIR"]
files = ["metaworld_generation.h5", "metaworld_embeddings_train.h5"]

for file_name in files:
    src_path = os.path.join(src_dir, file_name)
    dst_path = os.path.join(dst_dir, file_name)
    src_size = os.path.getsize(src_path)
    dst_size = os.path.getsize(dst_path)
    print(f"VERIFY {file_name}: src_size={src_size} dst_size={dst_size}", flush=True)
    if src_size != dst_size:
        raise RuntimeError(f"size mismatch for {file_name}")

    with h5py.File(src_path, "r") as src_h5, h5py.File(dst_path, "r") as dst_h5:
        src_keys = sorted(list(src_h5.keys()))
        dst_keys = sorted(list(dst_h5.keys()))
        if src_keys != dst_keys:
            raise RuntimeError(f"top-level key mismatch for {file_name}")
        print(f"VERIFY_OK {file_name}: keys={len(src_keys)}", flush=True)

print("STAGED_LABEL_INPUTS_VERIFY_OK", flush=True)
PY
fi

cd "${PROJECT_DIR}"

echo "========== Generate exponential beta=2 baseline labels =========="
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "RAW_DATA_DIR=${RAW_DATA_DIR}"
echo "TRAIN_DATA_DIR=${TRAIN_DATA_DIR}"
echo "REWARD_MODEL_PATH=${REWARD_MODEL_PATH}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "BETA=${BETA}"
echo "EXP_HORIZON=${EXP_HORIZON}"
echo "EXP_DIFF_SCALE=${EXP_DIFF_SCALE}"

srun python -u scripts/generate_labeled_dataset.py \
  --h5_video_path "${TRAIN_DATA_DIR}/metaworld_generation.h5" \
  --h5_embedding_path "${TRAIN_DATA_DIR}/metaworld_embeddings_train.h5" \
  --reward_model_path "${REWARD_MODEL_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --use_exponential_progress \
  --exponential_beta "${BETA}" \
  --exponential_horizon "${EXP_HORIZON}" \
  --exponential_diff_reward_scale "${EXP_DIFF_SCALE}"
