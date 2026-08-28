#!/usr/bin/env bash
#SBATCH --job-name=top_srv_g5
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/topreward_server_group5_%j.out
#SBATCH --error=logs/topreward_server_group5_%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
export PROJECT_DIR GROUP_ID=5 SERVER_LABEL="topreward group 5"
export INFO_FILE="${PROJECT_DIR}/logs/topreward_server_group5_info.txt"
export SERVER_PORT="${SERVER_PORT:-8215}" FALLBACK_START="${FALLBACK_START:-18851}" FALLBACK_END="${FALLBACK_END:-18860}"
bash "${PROJECT_DIR}/scripts/topreward_server_group_generic.sbatch"
