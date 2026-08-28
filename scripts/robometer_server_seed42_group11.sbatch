#!/usr/bin/env bash
# Shared Robometer server for seed42 group 11 jobs.

#SBATCH --job-name=robo_srv_s42_g11
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=d23-[10,13-16],e21-[01-16],e22-[01-16],e23-01
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/robometer_server_seed42_group11_%j.out
#SBATCH --error=logs/robometer_server_seed42_group11_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
export PROJECT_DIR
export GROUP_ID=11
export SERVER_LABEL="seed42 group 11"
export INFO_FILE="${PROJECT_DIR}/logs/robometer_server_seed42_group11_info.txt"
export SERVER_PORT="${SERVER_PORT:-8054}"
export FALLBACK_START="${FALLBACK_START:-18440}"
export FALLBACK_END="${FALLBACK_END:-18445}"
export STARTUP_DELAY_SEC="${STARTUP_DELAY_SEC:-480}"
export STARTUP_MAX_ATTEMPTS="${STARTUP_MAX_ATTEMPTS:-6}"
export STARTUP_BACKOFF_BASE_SEC="${STARTUP_BACKOFF_BASE_SEC:-120}"

bash "${PROJECT_DIR}/scripts/robometer_server_mixed_group_generic.sbatch"
