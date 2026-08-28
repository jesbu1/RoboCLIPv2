#!/usr/bin/env bash
# =============================================================
# Robometer Online Training — Launch Script
# =============================================================
# Reuses the 2 existing labeling servers (ports 8000, 8001) and
# starts 6 additional servers (ports 8002-8007) for online training.
#
# Prerequisites:
#   - 2 servers already running (from robometer_server_baseline/gamma099)
#   - Offline checkpoints exist in logs/
#   - Labeled datasets exist in datasets/
#
# Usage:
#   bash scripts/run_robometer_online.sh
# =============================================================

set -euo pipefail

ROBOMETER_CONDA="${ROBOMETER_CONDA:-}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

ROBOMETER_CONDA="${ROBOMETER_CONDA:-${PROJECT_DIR}/conda_envs/robometer}"

export PROJECT_DIR ROBOMETER_CONDA
echo "Project dir:      $PROJECT_DIR"
echo "Robometer conda:  $ROBOMETER_CONDA"
echo ""

mkdir -p logs

# =================================================================
# Reuse existing servers for task 0 and 1
# =================================================================
echo "=== Reusing existing servers for task 0 and 1 ==="

BL_INFO=logs/robometer_server_baseline_info.txt
G099_INFO=logs/robometer_server_gamma099_info.txt

if [ ! -f "$BL_INFO" ]; then
  echo "ERROR: $BL_INFO not found. Start robometer_server_baseline first."
  exit 1
fi
if [ ! -f "$G099_INFO" ]; then
  echo "ERROR: $G099_INFO not found. Start robometer_server_gamma099 first."
  exit 1
fi

# Map existing servers to array indices
cp "$BL_INFO" logs/robometer_server_info_0.txt
cp "$G099_INFO" logs/robometer_server_info_1.txt
echo "  Task 0 → $(cat logs/robometer_server_info_0.txt)"
echo "  Task 1 → $(cat logs/robometer_server_info_1.txt)"

# =================================================================
# Start 6 new servers for tasks 2-7
# =================================================================
echo ""
echo "=== Starting 6 new servers (tasks 2-7) ==="
SERVER_JOB=$(sbatch --parsable --array=2-7 \
  --output="$PROJECT_DIR/logs/robometer_server_%A_%a.out" \
  --error="$PROJECT_DIR/logs/robometer_server_%A_%a.err" \
  scripts/robometer_server_array.sbatch)
echo "Server array job: $SERVER_JOB (array 2-7)"

# =================================================================
# Submit online training (all 8 tasks)
# =================================================================
echo ""
echo "=== Submitting online baseline (tasks 0-7) ==="
BL_JOB=$(sbatch --parsable --dependency=after:$SERVER_JOB \
  --output="$PROJECT_DIR/logs/%x_%A_%a.out" \
  --error="$PROJECT_DIR/logs/%x_%A_%a.err" \
  scripts/robometer_online_baseline.sbatch)
echo "Online baseline job: $BL_JOB"

echo "=== Submitting online diff_gamma099 (tasks 0-7) ==="
G099_JOB=$(sbatch --parsable --dependency=after:$SERVER_JOB \
  --output="$PROJECT_DIR/logs/%x_%A_%a.out" \
  --error="$PROJECT_DIR/logs/%x_%A_%a.err" \
  scripts/robometer_online_gamma099.sbatch)
echo "Online gamma099 job: $G099_JOB"

echo ""
echo "=== All jobs submitted ==="
echo "  Existing servers: task 0 (port 8000), task 1 (port 8001)"
echo "  New servers:      $SERVER_JOB (tasks 2-7, ports 8002-8007)"
echo "  Baseline:         $BL_JOB (array 0-7)"
echo "  Gamma099:         $G099_JOB (array 0-7)"
echo ""
echo "Monitor with: squeue -u \$USER"
