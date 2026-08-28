#!/usr/bin/env bash
# Submit pure ReWiND diff-gamma=1.0 pipeline using scale=(HORIZON+1)/2.
#
# Stage 1: one label job (shared)
# Stage 2: one offline array (3 seeds), depends on label
# Stage 3: one online array (8 tasks x 3 seeds), depends on offline

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HORIZON="${HORIZON:-128}"
cd "$PROJECT_DIR"
mkdir -p logs

DIFF_REWARD_SCALE=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", (h + 1) / 2 }')
REWARD_DIVISOR=$(awk -v h="$HORIZON" 'BEGIN { printf "%.10g", 2 / (h + 1) }')
TAG="diff_gamma1_scale_hplus1_over2_H${HORIZON}"

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

OFFLINE_JOB_ID=$(sbatch --parsable $LABEL_DEP \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",HORIZON="${HORIZON}" \
  scripts/rewind_offline_diff_gamma1_scale_hplus1_over2_10critics.sbatch)

ONLINE_JOB_ID=$(sbatch --parsable \
  --dependency="afterok:${OFFLINE_JOB_ID}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",HORIZON="${HORIZON}" \
  scripts/rewind_online_diff_gamma1_scale_hplus1_over2_base_reward_10critics.sbatch)

echo "offline=${OFFLINE_JOB_ID}"
echo "online=${ONLINE_JOB_ID}"

MANIFEST="logs/rewind_${TAG}_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "manifest=${PROJECT_DIR}/${MANIFEST}"
  echo "label=${LABEL_JOB_ID}"
  echo "horizon=${HORIZON}"
  echo "diff_reward_scale=(HORIZON+1)/2=${DIFF_REWARD_SCALE}"
  echo "reward_divisor=2/(HORIZON+1)=${REWARD_DIVISOR}"
  echo "offline=${OFFLINE_JOB_ID}  (array 0-2, seeds=42,32,0)"
  echo "online=${ONLINE_JOB_ID}  (array 0-23, 8 tasks x 3 seeds)"
  echo "tasks=window-close-v2 reach-wall-v2 faucet-close-v2 coffee-button-v2 button-press-wall-v2 door-lock-v2 handle-press-side-v2 sweep-into-v2"
  echo "seeds=42 32 0"
  echo "total_online_jobs=24"
} | tee "${PROJECT_DIR}/${MANIFEST}"
