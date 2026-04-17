#!/usr/bin/env bash
# Submit pure ReWiND reverse-diff experiments:
#   1) gamma=0.99,   scale=100
#   2) gamma=0.999,  scale=1000
#   3) gamma=0.9999, scale=10000
#
# Each version has its own label H5, 3-seed offline array, and 8-task x 3-seed online array.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FORCE_LABEL="${FORCE_LABEL:-false}"

cd "$PROJECT_DIR"
mkdir -p logs

VARIANTS=(gamma099_scale100 gamma0999_scale1000 gamma09999_scale10000)
LABEL_OUTPUTS=(
  datasets/metaworld_labeled_reverse_diff_gamma099.h5
  datasets/metaworld_labeled_reverse_diff_gamma0999.h5
  datasets/metaworld_labeled_reverse_diff_gamma09999.h5
)

declare -a LABEL_IDS
declare -a OFFLINE_IDS
declare -a ONLINE_IDS

submit_label_if_needed() {
  local idx="$1"
  local variant="${VARIANTS[$idx]}"
  local output_path="${LABEL_OUTPUTS[$idx]}"
  local label_id

  if [ "$FORCE_LABEL" = "true" ] || [ ! -f "$output_path" ]; then
    label_id=$(sbatch --parsable \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",VARIANT="$variant" \
      scripts/rewind_label_reverse_diff_variant.sbatch)
    echo "$label_id"
  else
    echo "skipped"
  fi
}

MANIFEST="logs/rewind_reverse_diff_three_gammas_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "manifest=${PROJECT_DIR}/${MANIFEST}"
  echo "force_label=${FORCE_LABEL}"
  echo "formula=scale * (P(s_next) - gamma * P(s))"
  echo "versions=gamma099_scale100 gamma0999_scale1000 gamma09999_scale10000"
  echo "seeds=0 32 42"
  echo "tasks=window-close-v2 reach-wall-v2 faucet-close-v2 coffee-button-v2 button-press-wall-v2 door-lock-v2 handle-press-side-v2 sweep-into-v2"
} | tee "$MANIFEST"

for idx in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$idx]}"
  label_output="${LABEL_OUTPUTS[$idx]}"

  label_id="$(submit_label_if_needed "$idx")"
  LABEL_IDS[$idx]="$label_id"

  label_dep=()
  if [ "$label_id" != "skipped" ]; then
    label_dep=(--dependency="afterok:${label_id}")
  fi

  offline_id=$(sbatch --parsable \
    "${label_dep[@]}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",VARIANT="$variant" \
    scripts/rewind_offline_reverse_diff_variant_10critics.sbatch)
  OFFLINE_IDS[$idx]="$offline_id"

  online_id=$(sbatch --parsable \
    --dependency="afterok:${offline_id}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",VARIANT="$variant" \
    scripts/rewind_online_reverse_diff_variant_base_reward_10critics.sbatch)
  ONLINE_IDS[$idx]="$online_id"

  {
    echo "variant_${idx}=${variant}"
    echo "label_${variant}=${label_id}"
    echo "label_output_${variant}=${label_output}"
    echo "offline_${variant}=${offline_id}  (array 0-2, seeds=0,32,42)"
    echo "online_${variant}=${online_id}  (array 0-23, 8 tasks x 3 seeds)"
  } | tee -a "$MANIFEST"
done

echo "manifest=${PROJECT_DIR}/${MANIFEST}"
