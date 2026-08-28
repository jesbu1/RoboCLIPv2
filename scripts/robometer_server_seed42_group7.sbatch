#!/usr/bin/env bash
# Shared Robometer server for seed42 group 7 jobs.

#SBATCH --job-name=robo_srv_s42_g7
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=d23-[10,13-16],e21-[01-16],e22-[01-16],e23-01
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/robometer_server_seed42_group7_%j.out
#SBATCH --error=logs/robometer_server_seed42_group7_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
export PROJECT_DIR
export GROUP_ID=7
export SERVER_LABEL="seed42 group 7"
export INFO_FILE="${PROJECT_DIR}/logs/robometer_server_seed42_group7_info.txt"
export SERVER_PORT="${SERVER_PORT:-8050}"
export FALLBACK_START="${FALLBACK_START:-18416}"
export FALLBACK_END="${FALLBACK_END:-18421}"
export STARTUP_DELAY_SEC="${STARTUP_DELAY_SEC:-0}"
export STARTUP_MAX_ATTEMPTS="${STARTUP_MAX_ATTEMPTS:-6}"
export STARTUP_BACKOFF_BASE_SEC="${STARTUP_BACKOFF_BASE_SEC:-120}"

bash "${PROJECT_DIR}/scripts/robometer_server_mixed_group_generic.sbatch"
