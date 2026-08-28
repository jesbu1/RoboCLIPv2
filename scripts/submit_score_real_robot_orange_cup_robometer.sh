#!/usr/bin/env bash
# Submit one ROBOMETER server and one scorer job for the real orange-cup videos.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
ROOT="${ROOT:-/scratch1/haobaizh/real_robot/put_orange_cup_into_the_box}"
TASK="${TASK:-Put the orange cup in the box}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
GROUP_ID="${GROUP_ID:-orange_${RUN_TAG}}"
SERVER_LABEL="${SERVER_LABEL:-robo_orange_cup}"
SERVER_PORT="${SERVER_PORT:-8040}"
SERVER_MEM="${SERVER_MEM:-64G}"
SERVER_TIME="${SERVER_TIME:-48:00:00}"
SERVER_INFO_FILE="${SERVER_INFO_FILE:-${PROJECT_DIR}/logs/robometer_orange_cup_${RUN_TAG}_info.txt}"
MAX_FRAMES="${MAX_FRAMES:-8}"
REQUEST_MAX_EDGE="${REQUEST_MAX_EDGE:-0}"
SCORE_MEM="${SCORE_MEM:-32G}"
SCORE_TIME="${SCORE_TIME:-48:00:00}"
INFO_FILE_WAIT_SEC="${INFO_FILE_WAIT_SEC:-86400}"
ONLY_SPLIT="${ONLY_SPLIT:-all}"
FORCE="${FORCE:-0}"

cd "$PROJECT_DIR"
mkdir -p logs

server_id=$(sbatch --parsable \
  --job-name="robo_orange_srv" \
  --mem="$SERVER_MEM" \
  --time="$SERVER_TIME" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",GROUP_ID="$GROUP_ID",SERVER_LABEL="$SERVER_LABEL",SERVER_PORT="$SERVER_PORT",INFO_FILE="$SERVER_INFO_FILE" \
  scripts/robometer_server_mixed_group_generic.sbatch)

score_id=$(sbatch --parsable \
  --job-name="score_orange_robo" \
  --mem="$SCORE_MEM" \
  --time="$SCORE_TIME" \
  --dependency="after:${server_id}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",ROOT="$ROOT",TASK="$TASK",MAX_FRAMES="$MAX_FRAMES",REQUEST_MAX_EDGE="$REQUEST_MAX_EDGE",INFO_FILE="$SERVER_INFO_FILE",INFO_FILE_WAIT_SEC="$INFO_FILE_WAIT_SEC",ONLY_SPLIT="$ONLY_SPLIT",FORCE="$FORCE" \
  scripts/score_real_robot_orange_cup_robometer.sbatch)

manifest="${PROJECT_DIR}/logs/robometer_orange_cup_score_${RUN_TAG}.txt"
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
} | tee "$manifest"

squeue -j "$server_id","$score_id" \
  -o "%.18i %.9P %.35j %.8u %.2t %.12M %.6D %.30R %.120E" | tee -a "$manifest"
