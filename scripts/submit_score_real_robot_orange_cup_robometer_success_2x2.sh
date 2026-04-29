#!/usr/bin/env bash
# Submit one progress+success ROBOMETER server and one 2x2 scorer job.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
ROOT="${ROOT:-/scratch1/haobaizh/real_robot/put_orange_cup_into_the_box}"
TASK="${TASK:-Put the orange cup in the box}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
GROUP_ID="${GROUP_ID:-orange_ps_${RUN_TAG}}"
SERVER_LABEL="${SERVER_LABEL:-robo_orange_progress_success}"
SERVER_PORT="${SERVER_PORT:-8040}"
SERVER_MEM="${SERVER_MEM:-64G}"
SERVER_TIME="${SERVER_TIME:-48:00:00}"
SERVER_INFO_FILE="${SERVER_INFO_FILE:-${PROJECT_DIR}/logs/robometer_orange_cup_progress_success_${RUN_TAG}_info.txt}"
MAX_FRAMES="${MAX_FRAMES:-8}"
REQUEST_MAX_EDGE="${REQUEST_MAX_EDGE:-0}"
SCORE_PARTITION="${SCORE_PARTITION:-main}"
SCORE_GRES="${SCORE_GRES:-}"
SCORE_MEM="${SCORE_MEM:-32G}"
SCORE_TIME="${SCORE_TIME:-48:00:00}"
INFO_FILE_WAIT_SEC="${INFO_FILE_WAIT_SEC:-86400}"
ONLY_SPLIT="${ONLY_SPLIT:-all}"
FORCE="${FORCE:-0}"
SUCCESS_OUTPUT_DIR="${SUCCESS_OUTPUT_DIR:-success_score_2x2}"
UNSUCCESS_OUTPUT_DIR="${UNSUCCESS_OUTPUT_DIR:-unsuccess_score_2x2}"

cd "$PROJECT_DIR"
mkdir -p logs

server_id=$(sbatch --parsable \
  --job-name="robo_orange_ps" \
  --mem="$SERVER_MEM" \
  --time="$SERVER_TIME" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",GROUP_ID="$GROUP_ID",SERVER_LABEL="$SERVER_LABEL",SERVER_PORT="$SERVER_PORT",INFO_FILE="$SERVER_INFO_FILE" \
  scripts/robometer_server_progress_success_generic.sbatch)

score_args=(
  --job-name="score_orange_2x2"
  --partition="$SCORE_PARTITION"
  --mem="$SCORE_MEM"
  --time="$SCORE_TIME"
  --dependency="after:${server_id}"
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",ROOT="$ROOT",TASK="$TASK",MAX_FRAMES="$MAX_FRAMES",REQUEST_MAX_EDGE="$REQUEST_MAX_EDGE",INFO_FILE="$SERVER_INFO_FILE",INFO_FILE_WAIT_SEC="$INFO_FILE_WAIT_SEC",ONLY_SPLIT="$ONLY_SPLIT",FORCE="$FORCE",INCLUDE_SUCCESS=1,SUCCESS_OUTPUT_DIR="$SUCCESS_OUTPUT_DIR",UNSUCCESS_OUTPUT_DIR="$UNSUCCESS_OUTPUT_DIR"
)

if [ -n "$SCORE_GRES" ]; then
  score_args+=(--gres="$SCORE_GRES")
fi

score_id=$(sbatch --parsable "${score_args[@]}" scripts/score_real_robot_orange_cup_robometer.sbatch)

manifest="${PROJECT_DIR}/logs/robometer_orange_cup_progress_success_2x2_${RUN_TAG}.txt"
{
  echo "manifest=$manifest"
  echo "project_dir=$PROJECT_DIR"
  echo "root=$ROOT"
  echo "task=$TASK"
  echo "run_tag=$RUN_TAG"
  echo "server_id=$server_id"
  echo "score_id=$score_id"
  echo "server_info_file=$SERVER_INFO_FILE"
  echo "max_frames=$MAX_FRAMES"
  echo "request_max_edge=$REQUEST_MAX_EDGE"
  echo "only_split=$ONLY_SPLIT"
  echo "force=$FORCE"
  echo "score_partition=$SCORE_PARTITION"
  echo "score_gres=$SCORE_GRES"
  echo "success_output_dir=$SUCCESS_OUTPUT_DIR"
  echo "unsuccess_output_dir=$UNSUCCESS_OUTPUT_DIR"
} | tee "$manifest"

squeue -j "$server_id","$score_id" \
  -o "%.18i %.9P %.35j %.8u %.2t %.12M %.6D %.30R %.120E" | tee -a "$manifest"
