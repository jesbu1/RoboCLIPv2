#!/usr/bin/env bash
#SBATCH --job-name=top_srv_g8
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/topreward_server_group8_%j.out
#SBATCH --error=logs/topreward_server_group8_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
export PROJECT_DIR GROUP_ID=8 SERVER_LABEL="topreward group 8"
export INFO_FILE="${PROJECT_DIR}/logs/topreward_server_group8_info.txt"
export SERVER_PORT="${SERVER_PORT:-8218}" FALLBACK_START="${FALLBACK_START:-18881}" FALLBACK_END="${FALLBACK_END:-18890}"
bash "${PROJECT_DIR}/scripts/topreward_server_group_generic.sbatch"
