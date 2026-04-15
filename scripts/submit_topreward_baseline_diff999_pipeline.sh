#!/usr/bin/env bash
# Submit TOPReward baseline + diff-gamma0999 pipeline:
# 8 TOPReward servers, first-ready labels, then no-action-chunk offline/online training.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TOPREWARD_DIR="${TOPREWARD_DIR:-/scratch1/haobaizh/rewind_topreward}"
SUCCESS_BONUS="${SUCCESS_BONUS:-0.0}"
DIFF_GAMMA="${DIFF_GAMMA:-0.999}"
DIFF_REWARD_SCALE="${DIFF_REWARD_SCALE:-1000.0}"
NUM_PREFIX_SAMPLES="${NUM_PREFIX_SAMPLES:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"

cd "$PROJECT_DIR"
mkdir -p logs

if [ ! -d "$TOPREWARD_DIR" ]; then
  echo "ERROR: TOPREWARD_DIR not found: $TOPREWARD_DIR"
  exit 1
fi
if [ ! -d "${TOPREWARD_DIR}/conda_envs/vllm" ]; then
  echo "ERROR: vLLM env not found: ${TOPREWARD_DIR}/conda_envs/vllm"
  exit 1
fi
if [ ! -d "${TOPREWARD_DIR}/conda_envs/rewind" ]; then
  echo "ERROR: TOPReward label env not found: ${TOPREWARD_DIR}/conda_envs/rewind"
  exit 1
fi
if [ ! -f "${PROJECT_DIR}/test_scripts/test_iql.py" ]; then
  echo "ERROR: PROJECT_DIR does not look like rewind_no-action-chunk: ${PROJECT_DIR}"
  exit 1
fi
if [ ! -d "${PROJECT_DIR}/conda_envs/rewind_nochunk" ]; then
  echo "ERROR: no-action conda env not found: ${PROJECT_DIR}/conda_envs/rewind_nochunk"
  exit 1
fi
if [ ! -f "${PROJECT_DIR}/configs/reward/topreward.yaml" ]; then
  echo "ERROR: TOPReward reward config missing from no-action repo: ${PROJECT_DIR}/configs/reward/topreward.yaml"
  exit 1
fi

REQUIRED_INPUTS=(
  "${TOPREWARD_DIR}/datasets/metaworld_generation.h5"
  "${TOPREWARD_DIR}/datasets/metaworld_embeddings_train.h5"
)
for input_path in "${REQUIRED_INPUTS[@]}"; do
  if [ ! -f "$input_path" ]; then
    echo "ERROR: Missing required input: $input_path"
    exit 1
  fi
done

GROUP_IDS=(1 2 3 4 5 6 7 8)
SERVER_JOB_IDS=()
SERVER_INFO_FILES=()

MANIFEST="${PROJECT_DIR}/logs/topreward_baseline_diff999_bonus0_base_reward_pipeline_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "manifest=${MANIFEST}"
  echo "topreward_dir=${TOPREWARD_DIR}"
  echo "training_entry=${PROJECT_DIR}/test_scripts/test_iql.py"
  echo "training_env=${PROJECT_DIR}/conda_envs/rewind_nochunk"
  echo "offline_training_steps=100000"
  echo "success_bonus=${SUCCESS_BONUS}"
  echo "diff_gamma=${DIFF_GAMMA}"
  echo "diff_reward_scale=${DIFF_REWARD_SCALE}"
  echo "diff_offline_reward_divisor=0.001"
  echo "num_prefix_samples=${NUM_PREFIX_SAMPLES}"
  echo "max_frames_per_query=${NUM_PREFIX_SAMPLES}"
  echo "request_timeout=${REQUEST_TIMEOUT}"
} | tee "$MANIFEST"

for group_id in "${GROUP_IDS[@]}"; do
  info_file="${PROJECT_DIR}/logs/topreward_server_group${group_id}_info.txt"
  rm -f "$info_file" "${info_file}.tmp"

  server_job_id=$(sbatch --parsable \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR" \
    "scripts/topreward_server_group${group_id}.sbatch")

  SERVER_JOB_IDS+=("$server_job_id")
  SERVER_INFO_FILES+=("$info_file")
  echo "server_group${group_id}=${server_job_id}" | tee -a "$MANIFEST"
done

SERVER_JOB_IDS_JOINED="$(IFS='|'; echo "${SERVER_JOB_IDS[*]}")"
SERVER_INFO_FILES_JOINED="$(IFS='|'; echo "${SERVER_INFO_FILES[*]}")"

launcher_job_id=$(sbatch --parsable \
  --job-name="top_launch" \
  --output="logs/top_launch_%j.out" \
  --error="logs/top_launch_%j.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",SERVER_JOB_IDS="$SERVER_JOB_IDS_JOINED",SERVER_INFO_FILES="$SERVER_INFO_FILES_JOINED",SUBMIT_MANIFEST="$MANIFEST",SUCCESS_BONUS="$SUCCESS_BONUS",DIFF_GAMMA="$DIFF_GAMMA",DIFF_REWARD_SCALE="$DIFF_REWARD_SCALE",NUM_PREFIX_SAMPLES="$NUM_PREFIX_SAMPLES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES" \
  scripts/topreward_launch_label_offline_online.sbatch)

echo "launcher=${launcher_job_id}" | tee -a "$MANIFEST"
