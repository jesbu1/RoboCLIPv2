#!/usr/bin/env bash
# Submit 24 pairwise-diff-loss ReWiND online jobs without Slurm arrays:
#   8 envs x 3 seeds, diff only.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch1/haobaizh/rewind_nochunk_online_logs}"
OFFLINE_BASE="${OFFLINE_BASE:-/scratch1/haobaizh/rewind_nochunk_offline_logs/rewind_pairwise_diff_loss}"
DATA_DIR="${DATA_DIR:-/scratch1/haobaizh/rewind_valuemodel/datasets}"
REWARD_CKPT="${REWARD_CKPT:-/scratch1/haobaizh/rewind_valuemodel/checkpoints/rewind_metaworld_pairwise_diff_loss_epoch_19.pth}"
PROJECT_CACHE="${PROJECT_CACHE:-/home1/haobaizh/.cache}"
HORIZON="${HORIZON:-128}"
BASE_REWARD_VALUE="${BASE_REWARD_VALUE:--1.0}"
ONLINE_STEPS="${ONLINE_STEPS:-100000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_pairwise_diff_loss_online_24_noarray}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
DRY_RUN="${DRY_RUN:-0}"

cd "$PROJECT_DIR"

ENV_IDS=(
  window-close-v2
  reach-wall-v2
  faucet-close-v2
  coffee-button-v2
  button-press-wall-v2
  door-lock-v2
  handle-press-side-v2
  sweep-into-v2
)

SEEDS=(0 32 42)
TAG="pairwise_diff_loss_diff_gamma1_scaled_H${HORIZON}"
H5_PATH="${DATA_DIR}/metaworld_labeled_pairwise_diff_loss_diff.h5"
SLURM_LOG_DIR="${SCRATCH_BASE}/slurm"
MANIFEST_DIR="${SCRATCH_BASE}/manifests"
MANIFEST="${MANIFEST_DIR}/rewind_online_pairwise_diff_loss_24_noarray_${RUN_TAG}.txt"

mkdir -p "$SLURM_LOG_DIR" "$MANIFEST_DIR" "${SCRATCH_BASE}/rewind_pairwise_diff_loss"

if [ "$SKIP_PREFLIGHT" != "1" ]; then
  required_inputs=(
    "${REWARD_CKPT}"
    "${H5_PATH}"
  )
  for seed in "${SEEDS[@]}"; do
    required_inputs+=("${OFFLINE_BASE}/offline_rewind_${TAG}_10critics_seed${seed}/last_offline.zip")
  done

  missing_inputs=()
  for input_path in "${required_inputs[@]}"; do
    if [ ! -f "$input_path" ]; then
      missing_inputs+=("$input_path")
    fi
  done

  if [ "${#missing_inputs[@]}" -gt 0 ]; then
    echo "ERROR: Missing required pairwise-diff online inputs:"
    printf '  %s\n' "${missing_inputs[@]}"
    echo "Set SKIP_PREFLIGHT=1 only if you intentionally want to submit anyway."
    exit 1
  fi
fi

DIFF_REWARD_SCALE="$(awk -v h="$HORIZON" 'BEGIN { printf "%.17g", (h + 1.0) / 2.0 }')"

{
  echo "manifest=${MANIFEST}"
  echo "project_dir=${PROJECT_DIR}"
  echo "scratch_base=${SCRATCH_BASE}"
  echo "offline_base=${OFFLINE_BASE}"
  echo "data_dir=${DATA_DIR}"
  echo "h5_path=${H5_PATH}"
  echo "reward_ckpt=${REWARD_CKPT}"
  echo "project_cache=${PROJECT_CACHE}"
  echo "run_tag=${RUN_TAG}"
  echo "horizon=${HORIZON}"
  echo "tag=${TAG}"
  echo "diff_reward_scale_expression=(HORIZON+1)/2"
  echo "diff_reward_scale=${DIFF_REWARD_SCALE}"
  echo "base_reward=true"
  echo "base_reward_value=${BASE_REWARD_VALUE}"
  echo "online_steps=${ONLINE_STEPS}"
  echo "dry_run=${DRY_RUN}"
  echo "seeds=$(IFS='|'; echo "${SEEDS[*]}")"
  echo "envs=$(IFS='|'; echo "${ENV_IDS[*]}")"
  echo "jobs_total=24"
} | tee "$MANIFEST"

job_ids=()
job_count=0

for env_id in "${ENV_IDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    job_count=$((job_count + 1))
    env_short="${env_id//-/_}"
    job_name="rw_pd_${env_short}_${seed}"
    job_name="${job_name:0:128}"

    if [ "$DRY_RUN" = "1" ]; then
      job_id="DRY_RUN_${job_count}"
      echo "DRY_RUN sbatch --job-name=${job_name} ENV_ID=${env_id} SEED=${seed}" | tee -a "$MANIFEST"
    else
      job_id=$(sbatch --parsable \
        --job-name="$job_name" \
        --output="${SLURM_LOG_DIR}/${job_name}_%j.out" \
        --error="${SLURM_LOG_DIR}/${job_name}_%j.err" \
        --export=ALL,PROJECT_DIR="$PROJECT_DIR",SCRATCH_BASE="$SCRATCH_BASE",OFFLINE_BASE="$OFFLINE_BASE",DATA_DIR="$DATA_DIR",REWARD_CKPT="$REWARD_CKPT",PROJECT_CACHE="$PROJECT_CACHE",HORIZON="$HORIZON",BASE_REWARD_VALUE="$BASE_REWARD_VALUE",ONLINE_STEPS="$ONLINE_STEPS",RUN_TAG="$RUN_TAG",ENV_ID="$env_id",SEED="$seed" \
        scripts/rewind_online_pairwise_diff_loss_single_task.sbatch)
    fi
    job_ids+=("$job_id")
    echo "job_${job_count}=${job_id} env=${env_id} seed=${seed}" | tee -a "$MANIFEST"
  done
done

if [ "$job_count" -ne 24 ]; then
  echo "ERROR: expected 24 jobs, submitted ${job_count}" | tee -a "$MANIFEST"
  exit 1
fi

echo "submitted_jobs_total=${job_count}" | tee -a "$MANIFEST"
echo "job_ids=$(IFS=,; echo "${job_ids[*]}")" | tee -a "$MANIFEST"

if [ "$DRY_RUN" != "1" ]; then
  squeue -j "$(IFS=,; echo "${job_ids[*]}")" \
    -o "%.18i %.2t %.12M %.30R %.40j" | tee -a "$MANIFEST"
fi
