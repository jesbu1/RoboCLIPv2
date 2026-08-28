#!/usr/bin/env bash
# Score real orange-cup videos with a running ROBOMETER server.

#SBATCH --job-name=score_orange_robo
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
ROOT="${ROOT:-/scratch1/haobaizh/real_robot/put_orange_cup_into_the_box}"
TASK="${TASK:-Put the orange cup in the box}"
MAX_FRAMES="${MAX_FRAMES:-8}"
REQUEST_MAX_EDGE="${REQUEST_MAX_EDGE:-0}"
TIMEOUT="${TIMEOUT:-120}"
RETRIES="${RETRIES:-3}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
INFO_FILE_WAIT_SEC="${INFO_FILE_WAIT_SEC:-86400}"
SLEEP_BETWEEN_REQUESTS="${SLEEP_BETWEEN_REQUESTS:-0}"
ONLY_SPLIT="${ONLY_SPLIT:-all}"
START="${START:-0}"
LIMIT="${LIMIT:-0}"
FORCE="${FORCE:-0}"
INCLUDE_SUCCESS="${INCLUDE_SUCCESS:-0}"
SUCCESS_OUTPUT_DIR="${SUCCESS_OUTPUT_DIR:-success_score_2x2}"
UNSUCCESS_OUTPUT_DIR="${UNSUCCESS_OUTPUT_DIR:-unsuccess_score_2x2}"
SERVER_URL="${SERVER_URL:-}"
INFO_FILE="${INFO_FILE:-}"

ROBOMETER_PROJECT_DIR="${ROBOMETER_PROJECT_DIR:-/project2/biyik_1165/haobaizh/rewind_robometer_valuemodel}"
ROBOMETER_CONDA="${ROBOMETER_CONDA:-${ROBOMETER_PROJECT_DIR}/conda_envs/robometer}"

if [ ! -d "$ROBOMETER_CONDA" ]; then
  echo "ERROR: Robometer conda env not found: ${ROBOMETER_CONDA}"
  exit 1
fi
if [ -z "$SERVER_URL" ] && [ -z "$INFO_FILE" ]; then
  echo "ERROR: set either SERVER_URL or INFO_FILE"
  exit 1
fi

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "$ROBOMETER_CONDA"

cd "$PROJECT_DIR"
mkdir -p logs

echo "========== Real orange-cup ROBOMETER scoring =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "NODE=$(hostname)"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "ROOT=$ROOT"
echo "TASK=$TASK"
echo "MAX_FRAMES=$MAX_FRAMES"
echo "REQUEST_MAX_EDGE=$REQUEST_MAX_EDGE"
echo "ONLY_SPLIT=$ONLY_SPLIT"
echo "START=$START"
echo "LIMIT=$LIMIT"
echo "FORCE=$FORCE"
echo "INCLUDE_SUCCESS=$INCLUDE_SUCCESS"
echo "SUCCESS_OUTPUT_DIR=$SUCCESS_OUTPUT_DIR"
echo "UNSUCCESS_OUTPUT_DIR=$UNSUCCESS_OUTPUT_DIR"
echo "SERVER_URL=$SERVER_URL"
echo "INFO_FILE=$INFO_FILE"
which python
python --version

args=(
  --root "$ROOT"
  --task "$TASK"
  --max-frames "$MAX_FRAMES"
  --request-max-edge "$REQUEST_MAX_EDGE"
  --timeout "$TIMEOUT"
  --retries "$RETRIES"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --info-file-wait-sec "$INFO_FILE_WAIT_SEC"
  --sleep-between-requests "$SLEEP_BETWEEN_REQUESTS"
  --only-split "$ONLY_SPLIT"
  --start "$START"
)

if [ "$LIMIT" != "0" ]; then
  args+=(--limit "$LIMIT")
fi
if [ "$FORCE" = "1" ]; then
  args+=(--force)
fi
if [ "$INCLUDE_SUCCESS" = "1" ]; then
  args+=(
    --include-success
    --success-output-dir "$SUCCESS_OUTPUT_DIR"
    --unsuccess-output-dir "$UNSUCCESS_OUTPUT_DIR"
  )
fi
if [ -n "$SERVER_URL" ]; then
  args+=(--server-url "$SERVER_URL")
else
  args+=(--info-file "$INFO_FILE")
fi

python -u scripts/score_real_robot_orange_cup_robometer.py "${args[@]}"
