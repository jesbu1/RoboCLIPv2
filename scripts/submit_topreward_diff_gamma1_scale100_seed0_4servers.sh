#!/usr/bin/env bash
# Submit TOPReward diff gamma=1.0, scale=100, seed0 pipeline:
# 4 servers -> first-ready label -> seed0 offline -> 8 seed0 online tasks (2 per server).

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TOPREWARD_DIR="${TOPREWARD_DIR:-/scratch1/haobaizh/rewind_topreward}"
MAX_FRAMES="${MAX_FRAMES:-4}"
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
  echo "ERROR: TOPReward reward config missing from ${PROJECT_DIR}/configs/reward/topreward.yaml"
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

GROUP_IDS=(91 92 93 94)
TASK_GROUPS=(
  "window-close-v2|reach-wall-v2"
  "faucet-close-v2|coffee-button-v2"
  "button-press-wall-v2|door-lock-v2"
  "handle-press-side-v2|sweep-into-v2"
)
SERVER_JOB_IDS=()
SERVER_INFO_FILES=()

MANIFEST="${PROJECT_DIR}/logs/topreward_diff_gamma1_scale100_seed0_4servers_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "manifest=${MANIFEST}"
  echo "project_dir=${PROJECT_DIR}"
  echo "topreward_dir=${TOPREWARD_DIR}"
  echo "variant=diff_gamma1_scale100_base_reward"
  echo "label_output=${TOPREWARD_DIR}/datasets/metaworld_topreward_labeled_diff_gamma1.h5"
  echo "offline_training_steps=100000"
  echo "online_seed=0"
  echo "success_bonus=0"
  echo "diff_gamma=1.0"
  echo "diff_reward_scale=100"
  echo "offline_reward_divisor=0.01"
  echo "max_frames=${MAX_FRAMES}"
  echo "request_timeout=${REQUEST_TIMEOUT}"
  echo "request_retries=${REQUEST_RETRIES}"
  echo "servers_total=4"
  echo "tasks_total=8"
  echo "tasks_per_server=2"
} | tee "$MANIFEST"

for idx in "${!GROUP_IDS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  task_envs="${TASK_GROUPS[$idx]}"
  info_file="${PROJECT_DIR}/logs/topreward_server_group${group_id}_info.txt"
  rm -f "$info_file" "${info_file}.tmp"

  server_job_id=$(sbatch --parsable \
    --job-name="top_srv_g${group_id}" \
    --account=biyik_1165 \
    --partition=gpu \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=64G \
    --time=48:00:00 \
    --output="logs/topreward_server_group${group_id}_%j.out" \
    --error="logs/topreward_server_group${group_id}_%j.err" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",GROUP_ID="$group_id",SERVER_LABEL="topreward group ${group_id}",INFO_FILE="$info_file" \
    scripts/topreward_server_group_generic.sbatch)

  SERVER_JOB_IDS+=("$server_job_id")
  SERVER_INFO_FILES+=("$info_file")
  echo "server_group${group_id}=${server_job_id} tasks=${task_envs}" | tee -a "$MANIFEST"
done

SERVER_JOB_IDS_JOINED="$(IFS='|'; echo "${SERVER_JOB_IDS[*]}")"
SERVER_INFO_FILES_JOINED="$(IFS='|'; echo "${SERVER_INFO_FILES[*]}")"

launcher_job_id=$(sbatch --parsable \
  --job-name="top_lau_d1_s100" \
  --output="logs/top_lau_d1_s100_%j.out" \
  --error="logs/top_lau_d1_s100_%j.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",SERVER_JOB_IDS="$SERVER_JOB_IDS_JOINED",SERVER_INFO_FILES="$SERVER_INFO_FILES_JOINED",SUBMIT_MANIFEST="$MANIFEST",MAX_FRAMES="$MAX_FRAMES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES" \
  scripts/topreward_launch_diff_gamma1_scale100_seed0_pipeline.sbatch)

echo "launcher=${launcher_job_id}" | tee -a "$MANIFEST"
