#!/usr/bin/env bash
# Submit pure ReWiND baseline + RL gamma=0 + learned reward scale=72.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

REWARD_SCALE="${REWARD_SCALE:-72}"
REWARD_DIVISOR="${REWARD_DIVISOR:-0.0138888889}"

if [ ! -f datasets/metaworld_labeled.h5 ]; then
  echo "ERROR: Missing datasets/metaworld_labeled.h5"
  exit 1
fi

OFFLINE_JOB_ID=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",REWARD_SCALE="${REWARD_SCALE}",REWARD_DIVISOR="${REWARD_DIVISOR}" \
  scripts/rewind_offline_baseline_bonus0_gamma0_scale72_10critics.sbatch)

ONLINE_JOB_ID=$(sbatch --parsable \
  --dependency="afterok:${OFFLINE_JOB_ID}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",REWARD_SCALE="${REWARD_SCALE}",REWARD_DIVISOR="${REWARD_DIVISOR}" \
  scripts/rewind_online_baseline_base_reward_gamma0_scale72_10critics.sbatch)

MANIFEST="logs/rewind_baseline_gamma0_scale72_10critics_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "manifest=${PROJECT_DIR}/${MANIFEST}"
  echo "reward_scale=${REWARD_SCALE}"
  echo "reward_divisor=${REWARD_DIVISOR}"
  echo "offline=${OFFLINE_JOB_ID}"
  echo "online=${ONLINE_JOB_ID}"
  echo "online_array=0-23"
  echo "tasks=window-close-v2 reach-wall-v2 faucet-close-v2 coffee-button-v2 button-press-wall-v2 door-lock-v2 handle-press-side-v2 sweep-into-v2"
  echo "seeds=0 32 42"
} | tee "${PROJECT_DIR}/${MANIFEST}"
