#!/usr/bin/env bash
# Submit raw no-chat TOPReward baseline+diff gamma1 seed0 pipeline:
# 8 raw servers -> first-ready labels/offlines -> 16 online jobs.
# Baseline and diff each use 4 servers; each server runs two online jobs sequentially.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TOPREWARD_DIR="${TOPREWARD_DIR:-/scratch1/haobaizh/rewind_topreward}"
HORIZON="${HORIZON:-128}"
MAX_FRAMES="${MAX_FRAMES:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"
SERVER_BACKEND="${SERVER_BACKEND:-raw}"
TOPREWARD_REQUEST_FORMAT="${TOPREWARD_REQUEST_FORMAT:-raw}"
TOPREWARD_ATTN_IMPLEMENTATION="${TOPREWARD_ATTN_IMPLEMENTATION:-auto}"
SERVER_CONSTRAINT="${SERVER_CONSTRAINT:-a40|a100|l40s}"
SERVER_MEM="${SERVER_MEM:-64G}"
SERVER_TIME="${SERVER_TIME:-48:00:00}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_DIR"
mkdir -p logs

if [ "$SERVER_BACKEND" != "raw" ]; then
  echo "ERROR: this submitter is for raw no-chat TOPReward; SERVER_BACKEND must be raw"
  exit 1
fi
if [ "$TOPREWARD_REQUEST_FORMAT" != "raw" ]; then
  echo "ERROR: this submitter is for raw no-chat TOPReward; TOPREWARD_REQUEST_FORMAT must be raw"
  exit 1
fi
if [ ! -d "$TOPREWARD_DIR" ]; then
  echo "ERROR: TOPREWARD_DIR not found: $TOPREWARD_DIR"
  exit 1
fi
if [ ! -d "${TOPREWARD_DIR}/conda_envs/vllm" ]; then
  echo "ERROR: TOPReward server env not found: ${TOPREWARD_DIR}/conda_envs/vllm"
  exit 1
fi
if [ ! -d "${TOPREWARD_DIR}/conda_envs/rewind" ]; then
  echo "ERROR: TOPReward label env not found: ${TOPREWARD_DIR}/conda_envs/rewind"
  exit 1
fi
if [ ! -d "${PROJECT_DIR}/conda_envs/rewind_nochunk" ]; then
  echo "ERROR: no-action conda env not found: ${PROJECT_DIR}/conda_envs/rewind_nochunk"
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

GROUP_IDS=(501 502 503 504 601 602 603 604)
GROUP_LABELS=(b0 b1 b2 b3 d0 d1 d2 d3)
SERVER_JOB_IDS=()
SERVER_INFO_FILES=()

BASELINE_H5="${TOPREWARD_DIR}/datasets/metaworld_topreward_raw_baseline_seed0_${RUN_TAG}.h5"
DIFF_H5="${TOPREWARD_DIR}/datasets/metaworld_topreward_raw_diff_gamma1_H${HORIZON}_seed0_${RUN_TAG}.h5"
BASELINE_OFFLINE_DIR="logs/offline_topreward_raw_baseline_H${HORIZON}_10critics_seed0_${RUN_TAG}"
DIFF_OFFLINE_DIR="logs/offline_topreward_raw_diff_gamma1_scale_hplus1_over2_H${HORIZON}_10critics_seed0_${RUN_TAG}"
MANIFEST="${PROJECT_DIR}/logs/topreward_raw_baseline_diff_seed0_8servers_${RUN_TAG}.txt"

{
  echo "manifest=${MANIFEST}"
  echo "project_dir=${PROJECT_DIR}"
  echo "topreward_dir=${TOPREWARD_DIR}"
  echo "run_tag=${RUN_TAG}"
  echo "server_backend=${SERVER_BACKEND}"
  echo "topreward_request_format=${TOPREWARD_REQUEST_FORMAT}"
  echo "topreward_attn_implementation=${TOPREWARD_ATTN_IMPLEMENTATION}"
  echo "server_constraint=${SERVER_CONSTRAINT}"
  echo "server_mem=${SERVER_MEM}"
  echo "server_time=${SERVER_TIME}"
  echo "baseline_h5=${BASELINE_H5}"
  echo "diff_h5=${DIFF_H5}"
  echo "baseline_offline_dir=${BASELINE_OFFLINE_DIR}"
  echo "diff_offline_dir=${DIFF_OFFLINE_DIR}"
  echo "offline_training_steps=100000"
  echo "online_total_time_steps=100000"
  echo "online_seed=0"
  echo "success_bonus=0"
  echo "horizon=${HORIZON}"
  echo "max_frames=${MAX_FRAMES}"
  echo "request_timeout=${REQUEST_TIMEOUT}"
  echo "request_retries=${REQUEST_RETRIES}"
  echo "servers_total=8"
  echo "online_jobs_total=16"
  echo "baseline_servers=4"
  echo "diff_servers=4"
  echo "first_wave=window-close-v2|faucet-close-v2|button-press-wall-v2|handle-press-side-v2"
  echo "second_wave=reach-wall-v2|coffee-button-v2|door-lock-v2|sweep-into-v2"
} | tee "$MANIFEST"

for idx in "${!GROUP_IDS[@]}"; do
  group_id="${GROUP_IDS[$idx]}"
  label="${GROUP_LABELS[$idx]}"
  info_file="${PROJECT_DIR}/logs/topreward_raw_server_group${group_id}_${RUN_TAG}_info.txt"
  rm -f "$info_file" "${info_file}.tmp"

  server_job_id=$(sbatch --parsable \
    --job-name="top_raw_${label}" \
    --account=biyik_1165 \
    --partition=gpu \
    --gres=gpu:1 \
    --constraint="$SERVER_CONSTRAINT" \
    --cpus-per-task=4 \
    --mem="$SERVER_MEM" \
    --time="$SERVER_TIME" \
    --output="logs/topreward_raw_server_group${group_id}_${RUN_TAG}_%j.out" \
    --error="logs/topreward_raw_server_group${group_id}_${RUN_TAG}_%j.err" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",GROUP_ID="$group_id",SERVER_LABEL="topreward_raw_group_${group_id}",INFO_FILE="$info_file",SERVER_BACKEND="$SERVER_BACKEND",TOPREWARD_ATTN_IMPLEMENTATION="$TOPREWARD_ATTN_IMPLEMENTATION" \
    scripts/topreward_server_group_generic.sbatch)

  SERVER_JOB_IDS+=("$server_job_id")
  SERVER_INFO_FILES+=("$info_file")
  echo "server_group${group_id}=${server_job_id} label=${label} info_file=${info_file}" | tee -a "$MANIFEST"
done

SERVER_JOB_IDS_JOINED="$(IFS='|'; echo "${SERVER_JOB_IDS[*]}")"
SERVER_INFO_FILES_JOINED="$(IFS='|'; echo "${SERVER_INFO_FILES[*]}")"
SERVER_GROUP_IDS_JOINED="$(IFS='|'; echo "${GROUP_IDS[*]}")"

launcher_job_id=$(sbatch --parsable \
  --job-name="top_lau_raw8" \
  --output="logs/top_lau_raw8_%j.out" \
  --error="logs/top_lau_raw8_%j.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TOPREWARD_DIR="$TOPREWARD_DIR",SERVER_JOB_IDS="$SERVER_JOB_IDS_JOINED",SERVER_INFO_FILES="$SERVER_INFO_FILES_JOINED",SERVER_GROUP_IDS="$SERVER_GROUP_IDS_JOINED",SUBMIT_MANIFEST="$MANIFEST",RUN_TAG="$RUN_TAG",HORIZON="$HORIZON",MAX_FRAMES="$MAX_FRAMES",REQUEST_TIMEOUT="$REQUEST_TIMEOUT",REQUEST_RETRIES="$REQUEST_RETRIES",TOPREWARD_REQUEST_FORMAT="$TOPREWARD_REQUEST_FORMAT",BASELINE_H5="$BASELINE_H5",DIFF_H5="$DIFF_H5",BASELINE_OFFLINE_DIR="$BASELINE_OFFLINE_DIR",DIFF_OFFLINE_DIR="$DIFF_OFFLINE_DIR" \
  scripts/topreward_launch_raw_baseline_diff_seed0_8servers.sbatch)

echo "launcher=${launcher_job_id}" | tee -a "$MANIFEST"

squeue -j "$(IFS=,; echo "${SERVER_JOB_IDS[*]}"),${launcher_job_id}" \
  -o "%.18i %.2t %.12M %.30R %.40j %.120E" | tee -a "$MANIFEST"
