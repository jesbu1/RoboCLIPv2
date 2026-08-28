#!/usr/bin/env bash
# Submit pure ReWiND diff-gamma=1.0 pipeline for scale=100, 1000, 10000.
#
# Stage 1: one label job (shared by all three scales)
# Stage 2: three offline arrays (3 seeds each) in parallel, all depend on label
# Stage 3: three online arrays (8 tasks x 3 seeds each) in parallel,
#          each depends on its corresponding offline array

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

if [ ! -f datasets/metaworld_labeled_diff_gamma1.h5 ]; then
  echo "INFO: metaworld_labeled_diff_gamma1.h5 not found, will generate via label job."
  LABEL_JOB_ID=$(sbatch --parsable \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
    scripts/rewind_label_diff_gamma1.sbatch)
  echo "label=${LABEL_JOB_ID}"
  LABEL_DEP="--dependency=afterok:${LABEL_JOB_ID}"
else
  echo "INFO: metaworld_labeled_diff_gamma1.h5 already exists, skipping label job."
  LABEL_JOB_ID="(skipped)"
  LABEL_DEP=""
fi

# ── Offline (3 scales in parallel) ──────────────────────────────────────────
OFF_100=$(sbatch --parsable $LABEL_DEP \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
  scripts/rewind_offline_diff_gamma1_scale100_10critics.sbatch)

OFF_1K=$(sbatch --parsable $LABEL_DEP \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
  scripts/rewind_offline_diff_gamma1_scale1000_10critics.sbatch)

OFF_10K=$(sbatch --parsable $LABEL_DEP \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
  scripts/rewind_offline_diff_gamma1_scale10000_10critics.sbatch)

echo "offline_scale100=${OFF_100}"
echo "offline_scale1000=${OFF_1K}"
echo "offline_scale10000=${OFF_10K}"

# ── Online (each depends on its own offline array) ───────────────────────────
ON_100=$(sbatch --parsable \
  --dependency="afterok:${OFF_100}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
  scripts/rewind_online_diff_gamma1_scale100_base_reward_10critics.sbatch)

ON_1K=$(sbatch --parsable \
  --dependency="afterok:${OFF_1K}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
  scripts/rewind_online_diff_gamma1_scale1000_base_reward_10critics.sbatch)

ON_10K=$(sbatch --parsable \
  --dependency="afterok:${OFF_10K}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
  scripts/rewind_online_diff_gamma1_scale10000_base_reward_10critics.sbatch)

echo "online_scale100=${ON_100}"
echo "online_scale1000=${ON_1K}"
echo "online_scale10000=${ON_10K}"

# ── Manifest ─────────────────────────────────────────────────────────────────
MANIFEST="logs/rewind_diff_gamma1_three_scales_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "manifest=${PROJECT_DIR}/${MANIFEST}"
  echo "label=${LABEL_JOB_ID}"
  echo "offline_scale100=${OFF_100}  (array 0-2, seeds=42,32,0)"
  echo "offline_scale1000=${OFF_1K}  (array 0-2, seeds=42,32,0)"
  echo "offline_scale10000=${OFF_10K}  (array 0-2, seeds=42,32,0)"
  echo "online_scale100=${ON_100}  (array 0-23, 8 tasks x 3 seeds)"
  echo "online_scale1000=${ON_1K}  (array 0-23, 8 tasks x 3 seeds)"
  echo "online_scale10000=${ON_10K}  (array 0-23, 8 tasks x 3 seeds)"
  echo "tasks=window-close-v2 reach-wall-v2 faucet-close-v2 coffee-button-v2 button-press-wall-v2 door-lock-v2 handle-press-side-v2 sweep-into-v2"
  echo "seeds=42 32 0"
  echo "total_online_jobs=72"
} | tee "${PROJECT_DIR}/${MANIFEST}"
