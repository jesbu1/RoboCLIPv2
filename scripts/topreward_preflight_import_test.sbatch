#!/usr/bin/env bash
#SBATCH --job-name=topreward_preflight
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

TOPREWARD_DIR="${TOPREWARD_DIR:-/scratch1/haobaizh/rewind_topreward}"
CONDA_ENV="${CONDA_ENV:-${TOPREWARD_DIR}/conda_envs/vllm}"

echo "========== TOPReward preflight import test =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "NODE=$(hostname)"
echo "TOPREWARD_DIR=${TOPREWARD_DIR}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "====================================================="

test -d "${TOPREWARD_DIR}"
test -d "${CONDA_ENV}"
test -f "${TOPREWARD_DIR}/data_preprocessing/topreward_label_reward.py"
test -f "${TOPREWARD_DIR}/metaworld_policy_training/reward_model/topreward_reward_model.py"

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

which python
python --version

python - <<'PY'
import importlib.util
import os
import sys

topreward_dir = os.environ.get("TOPREWARD_DIR", "/scratch1/haobaizh/rewind_topreward")
checks = [
    ("vllm", None),
    ("torch", None),
    ("transformers", None),
    ("requests", None),
]

for module_name, _ in checks:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise RuntimeError(f"Missing Python module: {module_name}")
    module = __import__(module_name)
    version = getattr(module, "__version__", "unknown")
    print(f"FOUND {module_name} version={version}")

paths = [
    "data_preprocessing/topreward_label_reward.py",
    "metaworld_policy_training/reward_model/topreward_reward_model.py",
    "metaworld_policy_training/configs/reward/topreward.yaml",
]

for rel in paths:
    path = os.path.join(topreward_dir, rel)
    if not os.path.exists(path):
        raise RuntimeError(f"Missing file: {path}")
    print(f"FOUND {path}")

print("TOPReward preflight import test PASSED")
PY
